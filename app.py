from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Global placeholders
reg_model = None
scaler = None
kmeans = None
risk_mapping = None

def load_models():
    global reg_model, scaler, kmeans, risk_mapping
    if not os.path.exists('models/reg_model.pkl'):
        print("Training models...")
        import subprocess
        subprocess.run(["python", "train_model.py"])
    
    reg_model = joblib.load('models/reg_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    kmeans = joblib.load('models/kmeans.pkl')
    risk_mapping = joblib.load('models/risk_mapping.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            if reg_model is None:
                load_models()
            # Get data from user
            daily_screen_time = float(request.form['screen_time'])
            social_media_usage = float(request.form['social_media'])
            productivity_usage = float(request.form['productivity'])
            sleep_time = float(request.form['sleep'])
            
            # Predict Score
            user_input = [[daily_screen_time, social_media_usage, productivity_usage, sleep_time]]
            score = reg_model.predict(user_input)[0]
            score = np.clip(score, 0, 1000)
            
            # Predict Cluster / Risk Level
            user_scaled = scaler.transform([[daily_screen_time, score]])
            cluster = kmeans.predict(user_scaled)[0]
            risk = risk_mapping[cluster]
            
            # Suggestions
            if risk == "Low":
                advice = "Great job! Your digital life is balanced. Keep it up!"
            elif risk == "Medium":
                advice = f"Warning: High social media usage ({social_media_usage} hrs). Try to reduce it by {social_media_usage * 0.2:.1f} hrs."
            else:
                advice = f"Urgent Action Needed: Limit overall screen time. Focus on offline activities and try to sleep at least 7.5 hrs."
            
            # Visual report
            img_data = generate_visual_report(score, daily_screen_time, risk)
            
            result = {
                'score': round(score),
                'risk': risk,
                'advice': advice,
                'image': img_data
            }
        except Exception as e:
            result = {'error': str(e)}
            
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Initial model run for safety
    if not os.path.exists('models'):
        print("Training models for first time...")
        import os as _os
        _os.system('python train_model.py')
    
    # Use environment port for production
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
