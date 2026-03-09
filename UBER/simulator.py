import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

n_samples = 200000

city_center_lat = 40.7128
city_center_lon = -74.0060

def random_location():
    lat = city_center_lat + np.random.normal(0,0.03)
    lon = city_center_lon + np.random.normal(0,0.03)
    return lat,lon

def haversine(lat1,lon1,lat2,lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return 6371*c

start_date = datetime(2023,1,1)

rows = []

for i in range(n_samples):

    day_offset = np.random.randint(0,365)
    hour = np.random.randint(0,24)
    minute = np.random.randint(0,60)

    ride_time = start_date + timedelta(days=int(day_offset),hours=int(hour),minutes=int(minute))

    pickup_lat,pickup_lon = random_location()

    drop_lat = pickup_lat + np.random.normal(0,0.02)
    drop_lon = pickup_lon + np.random.normal(0,0.02)

    distance = haversine(pickup_lat,pickup_lon,drop_lat,drop_lon)

    passenger_count = np.random.choice([1,2,3,4],p=[0.65,0.2,0.1,0.05])

    base_fare = 2.5
    per_km = 1.8

    surge = 1

    if hour in [7,8,9,17,18,19]:
        surge = np.random.uniform(1.2,1.8)

    if hour in [0,1,2,3]:
        surge = np.random.uniform(1.1,1.4)

    fare = (base_fare + distance*per_km)*surge
    fare += np.random.normal(0,1)

    fare = max(3, fare)

    rows.append([
        i+1,
        ride_time.strftime("%d-%m-%Y"),
        ride_time.strftime("%H:%M:%S"),
        drop_lon,
        drop_lat,
        pickup_lon,
        pickup_lat,
        round(fare,2),
        passenger_count
    ])

df = pd.DataFrame(rows,columns=[
"Booking ID",
"date",
"pickup_time",
"dropoff_longitude",
"dropoff_latitude",
"pickup_longitude",
"pickup_latitude",
"fare_amount",
"passenger_count"
])

df.to_csv("uber_simulated.csv",index=False)

print("Dataset generated:",df.shape)
print(df.head())
