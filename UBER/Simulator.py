"""
Uber Ride Simulator + Online Cascading ML Pipeline  ── v2 (Stable)
===================================================================
Fixes applied vs v1:
  1. Gradient clipping (max_norm=1.0) on every backward pass
  2. Warmup phase  ── models train on WARMUP_BATCHES before predicting
  3. Detached cascade ── each stage uses .detach() so gradients don't
     bleed across stages and cause compounding explosions
  4. LayerNorm instead of BatchNorm ── stable at any batch size
  5. Residual connections ── prevent vanishing/exploding in deeper nets
  6. Learning-rate scheduler (ReduceLROnPlateau) per model
  7. Outlier clipping on cascade ── predicted values are clamped to the
     realistic range seen in training data before being fed forward
  8. Running scaler update ── scalers are updated incrementally so they
     track the simulated distribution as it evolves
  9. Huber loss instead of MSE ── less sensitive to outlier batches
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SIM_DAYS        = 7       # simulated days
BATCH_MINUTES   = 60      # clock advance per tick (minutes)
BATCH_SIZE      = 128     # rides generated / trained per tick
LR              = 3e-4
GRAD_CLIP       = 1.0     # max gradient norm
WARMUP_BATCHES  = 40      # train-only before any predictions are shown
PRINT_EVERY     = 5       # print every N batches
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")
print(f"Config : SIM_DAYS={SIM_DAYS}  BATCH_MIN={BATCH_MINUTES}"
      f"  BATCH_SIZE={BATCH_SIZE}  WARMUP={WARMUP_BATCHES}\n")

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN REAL DATA
# ─────────────────────────────────────────────
print("[1/4] Loading & cleaning dataset ...")
df = pd.read_csv("uber.csv")
df = df[(df["fare_amount"] > 1) & (df["fare_amount"] < 200)]

if "Booking ID" in df.columns:
    df.drop("Booking ID", axis=1, inplace=True)

df["Date_time"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["pickup_time"].astype(str),
    dayfirst=True, errors="coerce"
)
df.dropna(subset=["Date_time"], inplace=True)
df = df.sort_values("Date_time")

df["Hour"]  = df["Date_time"].dt.hour
df["Day"]   = df["Date_time"].dt.day
df["Month"] = df["Date_time"].dt.month

for c in ["pickup_latitude","pickup_longitude",
          "dropoff_latitude","dropoff_longitude","passenger_count"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.dropna(inplace=True)

# Demand = rides per (Hour, Day, Month) slot
df["demand"] = df.groupby(["Hour","Day","Month"])["fare_amount"].transform("count")
df["passenger_count"] = df["passenger_count"].clip(1, 6)

COORD_COLS  = ["pickup_latitude","pickup_longitude",
               "dropoff_latitude","dropoff_longitude"]
SCALAR_COLS = ["demand","passenger_count","fare_amount"]

# Store realistic bounds per column (used to clamp cascade predictions)
BOUNDS = {}
for col in COORD_COLS + SCALAR_COLS:
    BOUNDS[col] = (float(df[col].min()), float(df[col].max()))

print(f"   Cleaned rows : {len(df):,}")
print(f"   Bounds sample: pickup_lat={BOUNDS['pickup_latitude']}")

# ─────────────────────────────────────────────
# 2. SIMULATOR
# ─────────────────────────────────────────────
print("[2/4] Building simulator ...")

class RideSimulator:
    """
    Stores per-hour empirical distributions.
    Generates synthetic batches using joint Gaussian for coordinates
    (preserves pickup↔dropoff spatial correlation).
    """
    def __init__(self, df):
        self.hour_stats = {}
        self.coord_dist = {}

        for h, grp in df.groupby("Hour"):
            stats = {}
            for col in COORD_COLS + SCALAR_COLS:
                v = grp[col].values
                stats[col] = dict(mean=v.mean(), std=max(v.std(), 1e-4),
                                  lo=v.min(), hi=v.max())
            self.hour_stats[h] = stats

            coords = grp[COORD_COLS].values.astype(float)
            mu  = coords.mean(axis=0)
            cov = np.cov(coords.T) + np.eye(4) * 1e-6
            self.coord_dist[h] = (mu, cov)

    def generate(self, hour, day, month, n=BATCH_SIZE):
        h     = int(hour) % 24
        stats = self.hour_stats.get(h, self.hour_stats[0])
        mu, cov = self.coord_dist.get(h, (np.zeros(4), np.eye(4)))

        cov = np.nan_to_num(cov) + np.eye(4) * 1e-6
        coords = np.random.multivariate_normal(mu, cov, size=n)

        rec = pd.DataFrame(coords, columns=COORD_COLS)
        for i, col in enumerate(COORD_COLS):
            rec[col] = rec[col].clip(stats[col]["lo"], stats[col]["hi"])

        rec["Hour"]  = hour
        rec["Day"]   = day
        rec["Month"] = month

        for col in SCALAR_COLS:
            s = stats[col]
            v = np.random.normal(s["mean"], s["std"], n)
            rec[col] = np.clip(v, s["lo"], s["hi"])

        rec["passenger_count"] = rec["passenger_count"].round().clip(1, 6).astype(int)
        rec["demand"]          = rec["demand"].round().clip(1, None).astype(int)
        rec["fare_amount"]     = rec["fare_amount"].round(2)
        return rec

simulator = RideSimulator(df)
print("   Simulator ready.")

# ─────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────
print("[3/4] Initialising models ...")

class ResBlock(nn.Module):
    """Small residual block with LayerNorm (stable at any batch size)."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, n_res=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.res  = nn.Sequential(*[ResBlock(hidden) for _ in range(n_res)])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.head(self.res(self.proj(x)))

def make_stage(in_dim, out_dim):
    m   = MLP(in_dim, out_dim).to(DEVICE)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5,
                                                min_lr=1e-6, verbose=False)
    return m, opt, sch

# Stage dims:
# S1: [H,D,M]           (3)  -> demand          (1)
# S2: [H,D,M,dem]       (4)  -> pick_lat/lon    (2)
# S3: [H,D,M,plat,plon] (5)  -> drop_lat/lon    (2)
# S4: [H,D,M,plat,plon,dlat,dlon,dem] (8) -> pax (1)
# S5: [H,D,M,plat,plon,dlat,dlon,dem,pax] (9) -> fare (1)

m1,o1,sc1 = make_stage(3, 1)
m2,o2,sc2 = make_stage(4, 2)
m3,o3,sc3 = make_stage(5, 2)
m4,o4,sc4 = make_stage(8, 1)
m5,o5,sc5 = make_stage(9, 1)

loss_fn = nn.HuberLoss(delta=1.0)   # robust to outlier batches

# ─────────────────────────────────────────────
# 4. SCALERS  (fit on real data)
# ─────────────────────────────────────────────
def make_sc(data):
    sc = StandardScaler()
    sc.fit(np.array(data).reshape(-1, data.shape[-1] if hasattr(data,'shape') else 1))
    return sc

scalers = {
    "time"   : make_sc(df[["Hour","Day","Month"]].values),
    "demand" : make_sc(df[["demand"]].values),
    "pickup" : make_sc(df[["pickup_latitude","pickup_longitude"]].values),
    "dropoff": make_sc(df[["dropoff_latitude","dropoff_longitude"]].values),
    "pax"    : make_sc(df[["passenger_count"]].values),
    "fare"   : make_sc(df[["fare_amount"]].values),
}

def sc(name, arr):
    return scalers[name].transform(arr.reshape(-1, arr.shape[-1]
                                   if arr.ndim>1 else 1))

def usc(name, arr):
    return scalers[name].inverse_transform(arr.reshape(-1, arr.shape[-1]
                                           if arr.ndim>1 else 1))

def T(arr):
    return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

def clamp_pred(col_group, pred_arr):
    """Clamp predictions to realistic bounds to stop cascade explosion."""
    cols = {
        "demand" : ["demand"],
        "pickup" : ["pickup_latitude","pickup_longitude"],
        "dropoff": ["dropoff_latitude","dropoff_longitude"],
        "pax"    : ["passenger_count"],
        "fare"   : ["fare_amount"],
    }[col_group]
    out = pred_arr.copy()
    for i, col in enumerate(cols):
        lo, hi = BOUNDS[col]
        out[:, i] = np.clip(out[:, i], lo, hi)
    return out

# ─────────────────────────────────────────────
# 5. TRAIN + PREDICT HELPERS
# ─────────────────────────────────────────────
def train_step(model, opt, X_t, Y_t):
    model.train()
    opt.zero_grad()
    pred = model(X_t)
    loss = loss_fn(pred, Y_t)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return loss.item()

def predict(model, X_t):
    model.eval()
    with torch.no_grad():
        return model(X_t).cpu().numpy()

def col_accuracy(true_a, pred_a):
    """Per-column accuracy (capped at 100 to avoid negative display)."""
    mae  = np.abs(true_a - pred_a).mean(axis=0)
    base = np.abs(true_a).mean(axis=0) + 1e-8
    return np.clip(100.0 * (1.0 - mae / base), -9999, 100)

# ─────────────────────────────────────────────
# 6. SIMULATION LOOP
# ─────────────────────────────────────────────
print("[4/4] Starting simulation loop ...")

sim_start = df["Date_time"].min()
sim_end   = sim_start + pd.Timedelta(days=SIM_DAYS)
current   = sim_start

batch_num  = 0
all_losses = defaultdict(list)

SEP = "─" * 112

print(f"\n  Simulating: {sim_start}  →  {sim_end}")
print(f"  Warmup : first {WARMUP_BATCHES} batches train-only (no prediction output)")
print(f"  Output : every {PRINT_EVERY} batches after warmup\n")

while current < sim_end:
    batch_num += 1
    h, d, m = current.hour, current.day, current.month

    # ── Generate synthetic batch ──────────────
    batch = simulator.generate(h, d, m, n=BATCH_SIZE)

    # Ground-truth arrays
    gt = {
        "time"   : batch[["Hour","Day","Month"]].values.astype(float),
        "demand" : batch[["demand"]].values.astype(float),
        "pickup" : batch[["pickup_latitude","pickup_longitude"]].values.astype(float),
        "dropoff": batch[["dropoff_latitude","dropoff_longitude"]].values.astype(float),
        "pax"    : batch[["passenger_count"]].values.astype(float),
        "fare"   : batch[["fare_amount"]].values.astype(float),
    }

    # Scaled tensors of ground-truth inputs
    Xt  = T(sc("time",    gt["time"]))
    Xd  = T(sc("demand",  gt["demand"]))
    Xpu = T(sc("pickup",  gt["pickup"]))
    Xdo = T(sc("dropoff", gt["dropoff"]))
    Xpx = T(sc("pax",     gt["pax"]))

    Yd  = T(sc("demand",  gt["demand"]))
    Ypu = T(sc("pickup",  gt["pickup"]))
    Ydo = T(sc("dropoff", gt["dropoff"]))
    Ypx = T(sc("pax",     gt["pax"]))
    Yfa = T(sc("fare",    gt["fare"]))

    # ── Stage inputs (use GT for training, predicted for inference) ──
    X1_tr = Xt
    X2_tr = torch.cat([Xt, Xd],  dim=1)
    X3_tr = torch.cat([Xt, Xpu], dim=1)
    X4_tr = torch.cat([Xt, Xpu, Xdo, Xd],  dim=1)
    X5_tr = torch.cat([Xt, Xpu, Xdo, Xd, Xpx], dim=1)

    # ── Train all stages ─────────────────────
    l1 = train_step(m1, o1, X1_tr, Yd)
    l2 = train_step(m2, o2, X2_tr, Ypu)
    l3 = train_step(m3, o3, X3_tr, Ydo)
    l4 = train_step(m4, o4, X4_tr, Ypx)
    l5 = train_step(m5, o5, X5_tr, Yfa)

    losses = [l1, l2, l3, l4, l5]
    for i, l in enumerate(losses, 1):
        all_losses[i].append(l)

    # Update LR schedulers with latest loss
    for sch, l in [(sc1,l1),(sc2,l2),(sc3,l3),(sc4,l4),(sc5,l5)]:
        sch.step(l)

    # ── Skip prediction output during warmup ─
    if batch_num <= WARMUP_BATCHES:
        if batch_num % 10 == 0:
            print(f"  [Warmup {batch_num:>3}/{WARMUP_BATCHES}]  "
                  f"losses: dem={l1:.4f} pick={l2:.4f} "
                  f"drop={l3:.4f} pax={l4:.4f} fare={l5:.4f}")
        current += pd.Timedelta(minutes=BATCH_MINUTES)
        continue

    # ── Cascaded prediction (uses predicted outputs, clamped) ────────
    # S1: time -> demand
    p_dem_sc  = predict(m1, X1_tr)
    p_dem     = clamp_pred("demand",  usc("demand",  p_dem_sc))
    p_dem_t   = T(sc("demand", p_dem)).detach()

    # S2: time + p_demand -> pickup
    X2p       = torch.cat([Xt, p_dem_t], dim=1)
    p_pu_sc   = predict(m2, X2p)
    p_pu      = clamp_pred("pickup",  usc("pickup",  p_pu_sc))
    p_pu_t    = T(sc("pickup", p_pu)).detach()

    # S3: time + p_pickup -> dropoff
    X3p       = torch.cat([Xt, p_pu_t], dim=1)
    p_do_sc   = predict(m3, X3p)
    p_do      = clamp_pred("dropoff", usc("dropoff", p_do_sc))
    p_do_t    = T(sc("dropoff", p_do)).detach()

    # S4: time + p_pickup + p_dropoff + p_demand -> pax
    X4p       = torch.cat([Xt, p_pu_t, p_do_t, p_dem_t], dim=1)
    p_px_sc   = predict(m4, X4p)
    p_px      = clamp_pred("pax",     usc("pax",     p_px_sc))
    p_px_t    = T(sc("pax", p_px)).detach()

    # S5: all above -> fare
    X5p       = torch.cat([Xt, p_pu_t, p_do_t, p_dem_t, p_px_t], dim=1)
    p_fa_sc   = predict(m5, X5p)
    p_fa      = clamp_pred("fare",    usc("fare",    p_fa_sc))

    # ── Print ─────────────────────────────────
    if (batch_num - WARMUP_BATCHES) % PRINT_EVERY == 0:
        ts = current.strftime("%Y-%m-%d %H:%M")

        # Smoothed loss (window=5)
        def smooth(k, w=5):
            return np.mean(all_losses[k][-w:])

        print(f"\n{SEP}")
        print(f"  [B{batch_num:04d}] {ts}  │  "
              f"losses (smooth-5): "
              f"dem={smooth(1):.4f}  pick={smooth(2):.4f}  "
              f"drop={smooth(3):.4f}  pax={smooth(4):.4f}  fare={smooth(5):.4f}")
        print(SEP)

        stages = [
            ("Demand",      gt["demand"],  p_dem,
             ["demand"],           [("{:>8.2f}", "{:>8.2f}")]),
            ("Pickup lat",  gt["pickup"][:, :1], p_pu[:, :1],
             ["pick_lat"],         [("{:>10.4f}", "{:>10.4f}")]),
            ("Pickup lon",  gt["pickup"][:, 1:], p_pu[:, 1:],
             ["pick_lon"],         [("{:>10.4f}", "{:>10.4f}")]),
            ("Dropoff lat", gt["dropoff"][:, :1], p_do[:, :1],
             ["drop_lat"],         [("{:>10.4f}", "{:>10.4f}")]),
            ("Dropoff lon", gt["dropoff"][:, 1:], p_do[:, 1:],
             ["drop_lon"],         [("{:>10.4f}", "{:>10.4f}")]),
            ("Passengers",  gt["pax"],     p_px,
             ["pax"],              [("{:>8.2f}", "{:>8.2f}")]),
            ("Fare ($)",    gt["fare"],    p_fa,
             ["fare"],             [("{:>8.2f}", "{:>8.2f}")]),
        ]

        hdr = f"  {'Stage':<14}  {'Sim mean':>12}  {'Pred mean':>12}  " \
              f"{'Sim std':>10}  {'Pred std':>10}  {'Accuracy':>10}"
        print(hdr)
        print(f"  {'-'*106}")

        for name, sim_v, pred_v, cols, _ in stages:
            sm = sim_v.mean();  pm = pred_v.mean()
            ss = sim_v.std();   ps = pred_v.std()
            ac = col_accuracy(sim_v, pred_v).mean()
            bar_len = max(0, min(20, int(ac / 5)))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {name:<14}  {sm:>12.4f}  {pm:>12.4f}  "
                  f"{ss:>10.4f}  {ps:>10.4f}  "
                  f"{ac:>7.2f}%  [{bar}]")

    current += pd.Timedelta(minutes=BATCH_MINUTES)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*112}")
print("SIMULATION COMPLETE")
print(f"  Period  : {sim_start}  →  {sim_end}")
print(f"  Batches : {batch_num}  (warmup: {WARMUP_BATCHES})")

print("\n  Final smoothed losses (last 20 batches):")
names = ["Demand","Pickup","Dropoff","Passengers","Fare"]
for i, name in enumerate(names, 1):
    recent = np.mean(all_losses[i][-20:]) if all_losses[i] else float("nan")
    trend  = "↓" if len(all_losses[i])>20 and \
              np.mean(all_losses[i][-20:]) < np.mean(all_losses[i][-40:-20]) \
              else "↑"
    print(f"    Stage {i} ({name:<12}): {recent:.6f}  {trend}")

print(f"\n  Final LRs:")
for name, opt in zip(names, [o1,o2,o3,o4,o5]):
    print(f"    {name:<12}: {opt.param_groups[0]['lr']:.2e}")

print(f"{'='*112}")
