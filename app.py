from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
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

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usage_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class UsageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    screen_time = db.Column(db.Float, nullable=False)
    social_media = db.Column(db.Float, nullable=False)
    productivity = db.Column(db.Float, nullable=False)
    sleep = db.Column(db.Float, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    risk = db.Column(db.String(20), nullable=False)

# Create Database tables
with app.app_context():
    db.create_all()

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

def generate_visual_report(score, screen_time, risk):
    plt.figure(figsize=(6, 4))
    
    # Simple Gauge Chart (mock)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    risk_color = colors[0] if risk == 'Low' else colors[1] if risk == 'Medium' else colors[2]
    
    plt.barh(['Addiction Level'], [score], color=risk_color)
    plt.xlim(0, 1000)
    plt.title(f"Score: {score:.0f}/1000 ({risk} Risk)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

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
            
            # Save to Database
            new_record = UsageRecord(
                screen_time=daily_screen_time,
                social_media=social_media_usage,
                productivity=productivity_usage,
                sleep=sleep_time,
                score=round(score),
                risk=risk
            )
            db.session.add(new_record)
            db.session.commit()
            
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

@app.route('/dashboard')
def dashboard():
    records = UsageRecord.query.order_by(UsageRecord.timestamp.desc()).all()
    # Prepare data for Chart.js
    dates = [r.timestamp.strftime("%Y-%m-%d %H:%M") for r in records[::-1]]
    scores = [r.score for r in records[::-1]]
    return render_template('dashboard.html', records=records, dates=dates, scores=scores)

@app.route('/clear_history')
def clear_history():
    UsageRecord.query.delete()
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    # Initial model run for safety
    if not os.path.exists('models'):
        print("Training models for first time...")
        import os as _os
        _os.system('python train_model.py')
    
    # Use environment port for production
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
