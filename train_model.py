import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Data Generation
np.random.seed(42)
n_samples = 1000
daily_screen_time = np.random.uniform(1, 12, n_samples)
social_media_usage = daily_screen_time * np.random.uniform(0.3, 0.8, n_samples)
productivity_usage = daily_screen_time * np.random.uniform(0.05, 0.2, n_samples)
sleep_time = 8 - (daily_screen_time * 0.2) + np.random.normal(0, 0.5, n_samples)
sleep_time = np.clip(sleep_time, 3, 10)

raw_score = (daily_screen_time * 60) + (social_media_usage * 40) - (sleep_time * 30) - (productivity_usage * 20)
addiction_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min()) * 1000

df = pd.DataFrame({
    'daily_screen_time': daily_screen_time,
    'social_media_usage': social_media_usage,
    'productivity_usage': productivity_usage,
    'sleep_time': sleep_time,
    'addiction_score': addiction_score
})

# Train Regression Model
X = df[['daily_screen_time', 'social_media_usage', 'productivity_usage', 'sleep_time']]
y = df['addiction_score']
reg_model = LinearRegression()
reg_model.fit(X, y)

# Train Clustering Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['daily_screen_time', 'addiction_score']])
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Map Risk levels
cluster_means = df.groupby('cluster')['addiction_score'].mean().sort_values()
risk_mapping = {cluster_means.index[0]: 'Low', cluster_means.index[1]: 'Medium', cluster_means.index[2]: 'High'}

# Save models
joblib.dump(reg_model, 'models/reg_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(kmeans, 'models/kmeans.pkl')
joblib.dump(risk_mapping, 'models/risk_mapping.pkl')

print("Models trained and saved to 'models/' folder.")
