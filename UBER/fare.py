from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv('uber.csv')
print(df.head())
print(df.columns)
print(df.select_dtypes(include='object').columns)

df.drop('Booking ID', axis=1, inplace=True)
df['Date_time'] = df['date'] + " " + df['pickup_time']
df['Date_time'] = pd.to_datetime(df['Date_time'], dayfirst=True)
df['Day'] = df['Date_time'].dt.day
df['Month'] = df['Date_time'].dt.month
df['Year'] = df['Date_time'].dt.year
df['Hour']=df['Date_time'].dt.hour
df['Minute']=df['Date_time'].dt.minute
df['Second']=df['Date_time'].dt.second

lat1, lon1 = np.radians(df['pickup_latitude']), np.radians(df['pickup_longitude'])
lat2, lon2 = np.radians(df['dropoff_latitude']), np.radians(df['dropoff_longitude'])

dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

c = 2 * np.arcsin(np.sqrt(a))
df['distance_km'] = 6371 * c

df.dropna(inplace=True)


X = df.drop(['date', 'Date_time', 'fare_amount', 'pickup_time'], axis=1)
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X.values)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_test, y_test)
print(model.score(X_test, y_test))
