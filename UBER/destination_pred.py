from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('uber.csv')

df.head()

df.drop(['date', 'pickup_time', 'passenger_count', 'fare_amount'], axis=1, inplace=True)
df.dropna(inplace=True)

print(df.columns)

X = df.drop(['dropoff_longitude', 'dropoff_latitude'], axis=1)
y = df[['dropoff_longitude', 'dropoff_latitude']]


scaler = StandardScaler()

rfr = RandomForestRegressor(n_estimators=100, random_state=42)

X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfr.fit(X_train, y_train)

print(rfr.score(X_test, y_test))
