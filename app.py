from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
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
app.config['SECRET_KEY'] = 'dev-secret-key-change-this-in-prod'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///digital_detox.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# --- Database Models ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    records = db.relationship('UsageRecord', backref='user', lazy=True)
    
    # Gamification stats
    zen_streak = db.Column(db.Integer, default=0)
    last_record_date = db.Column(db.Date)
    badges = db.Column(db.String(255), default="") # Stores comma-separated badge names

class UsageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Core Data
    screen_time = db.Column(db.Float, nullable=False)
    sleep = db.Column(db.Float, nullable=False)
    productivity = db.Column(db.Float, nullable=False)
    
    # App Breakdown (Social Media)
    instagram = db.Column(db.Float, default=0)
    tiktok = db.Column(db.Float, default=0)
    youtube = db.Column(db.Float, default=0)
    linkedin = db.Column(db.Float, default=0)
    whatsapp = db.Column(db.Float, default=0)
    
    # Results
    score = db.Column(db.Integer, nullable=False)
    risk = db.Column(db.String(20), nullable=False)

# Create Database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- AI Logic Helpers ---

def ensure_models():
    if not os.path.exists('models/reg_model.pkl'):
        print("Models not found. Training now...")
        import subprocess
        subprocess.run(["python", "train_model.py"])

ensure_models()

reg_model = joblib.load('models/reg_model.pkl')
scaler = joblib.load('models/scaler.pkl')
kmeans = joblib.load('models/kmeans.pkl')
risk_mapping = joblib.load('models/risk_mapping.pkl')

def generate_visual_report(score, risk):
    plt.figure(figsize=(6, 4))
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    risk_color = colors[0] if risk == 'Low' else colors[1] if risk == 'Medium' else colors[2]
    plt.barh(['Addiction Level'], [score], color=risk_color)
    plt.xlim(0, 1000)
    plt.title(f"Score: {score:.0f}/1000 ({risk} Risk)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# --- Authentication Routes ---

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
            return redirect(url_for('signup'))
            
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password!')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Main App Routes ---

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    result = None
    if request.method == 'POST':
        try:
            # Get data
            st = float(request.form['screen_time'])
            sleep = float(request.form['sleep'])
            prod = float(request.form['productivity'])
            
            # App Breakdown
            insta = float(request.form.get('instagram', 0))
            tiktok = float(request.form.get('tiktok', 0))
            yt = float(request.form.get('youtube', 0))
            link = float(request.form.get('linkedin', 0))
            wa = float(request.form.get('whatsapp', 0))
            
            social_sum = insta + tiktok + yt + link + wa
            
            # Predict Score
            user_input = [[st, social_sum, prod, sleep]]
            score = reg_model.predict(user_input)[0]
            score = int(np.clip(score, 0, 1000))
            
            user_scaled = scaler.transform([[st, score]])
            risk = risk_mapping[kmeans.predict(user_scaled)[0]]
            
            # Update Gamification (Streaks & Badges)
            today = datetime.utcnow().date()
            if current_user.last_record_date == today - timedelta(days=1):
                if risk == 'Low': current_user.zen_streak += 1
            elif current_user.last_record_date != today:
                current_user.zen_streak = 1 if risk == 'Low' else 0
            
            current_user.last_record_date = today
            
            # Check for Badges
            earned_badges = current_user.badges.split(',') if current_user.badges else []
            if sleep >= 8 and "Sleep Warrior" not in earned_badges: earned_badges.append("Sleep Warrior")
            if prod > social_sum and "Productivity King" not in earned_badges: earned_badges.append("Productivity King")
            if current_user.zen_streak >= 3 and "Digital Monk" not in earned_badges: earned_badges.append("Digital Monk")
            current_user.badges = ",".join(earned_badges)

            # Save Record
            record = UsageRecord(
                user_id=current_user.id, screen_time=st, sleep=sleep, productivity=prod,
                instagram=insta, tiktok=tiktok, youtube=yt, linkedin=link, whatsapp=wa,
                score=score, risk=risk
            )
            db.session.add(record)
            db.session.commit()
            
            result = {'score': score, 'risk': risk, 'image': generate_visual_report(score, risk)}
        except Exception as e:
            result = {'error': str(e)}
            
    return render_template('index.html', result=result)

@app.route('/dashboard')
@login_required
def dashboard():
    records = UsageRecord.query.filter_by(user_id=current_user.id).order_by(UsageRecord.timestamp.desc()).all()
    dates = [r.timestamp.strftime("%Y-%m-%d %H:%M") for r in records[::-1]]
    scores = [r.score for r in records[::-1]]
    
    # Calculate latest App Breakdown for Pie Chart
    app_data = {}
    if records:
        latest = records[0]
        app_data = {
            'Instagram': latest.instagram, 'TikTok': latest.tiktok,
            'YouTube': latest.youtube, 'LinkedIn': latest.linkedin, 'WhatsApp': latest.whatsapp
        }
        
    return render_template('dashboard.html', records=records, dates=dates, scores=scores, app_data=app_data)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
