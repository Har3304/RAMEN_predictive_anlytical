import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

df = pd.read_csv("uber_simulated.csv")


df = df[(df["fare_amount"] > 1) & (df["fare_amount"] < 200)]

df.drop("Booking ID", axis=1, inplace=True)

df["Date_time"] = df["date"] + " " + df["pickup_time"]
df["Date_time"] = pd.to_datetime(df["Date_time"], dayfirst=True)

df = df.sort_values("Date_time")

df["Hour"] = df["Date_time"].dt.hour
df["Day"] = df["Date_time"].dt.day
df["Month"] = df["Date_time"].dt.month

lat1 = np.radians(df["pickup_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon2 = np.radians(df["dropoff_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
a = np.clip(a,0,1)

c = 2*np.arcsin(np.sqrt(a))

df["distance_km"] = 6371*c

df["demand"] = df.groupby(["Hour","Day","Month"])["fare_amount"].transform("count")


df.drop(['Date_time', 'date','pickup_time','passenger_count','fare_amount', 'distance_km'],axis=1,inplace=True)
df.dropna(inplace=True)

X = df.drop(['dropoff_longitude','dropoff_latitude'],axis=1).values
y = df[['dropoff_longitude','dropoff_latitude']].values

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train,y_train),batch_size=256,shuffle=True)
test_loader = DataLoader(TensorDataset(X_test,y_test),batch_size=256)

class MLP(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
    def forward(self,x):
        return self.net(x)

model1 = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist")

model = MLP(X_train.shape[1]).to(device)
model1 = model1.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 60

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0
    for xb,yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred,yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb,yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)

            preds.append(pred.cpu().numpy())
            trues.append(yb.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    preds = y_scaler.inverse_transform(preds)
    trues = y_scaler.inverse_transform(trues)
    mae = np.mean(np.abs(preds-trues))
    accuracy = 100*(1 - mae/(np.mean(np.abs(trues))+1e-8))

    if epoch % 10 == 0:
        print("Epoch:",epoch,"Train Loss:",train_los,"Accuracy:",accuracy,"%")

model.eval()

preds = []
trues = []

with torch.no_grad():
    for xb,yb in test_loader:
        xb = xb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy())
        trues.append(yb.numpy())

preds = np.vstack(preds)
trues = np.vstack(trues)

preds = y_scaler.inverse_transform(preds)
trues = y_scaler.inverse_transform(trues)

r2 = r2_score(trues,preds)

print("R2 Score:", r2)
