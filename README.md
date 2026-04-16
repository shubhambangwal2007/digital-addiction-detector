# Digital Addiction Detector

An AI-powered tool to help users recognize and manage their mobile usage patterns using Data Analysis, Regression, and Clustering.

## 🚀 Features
- **Usage Tracking**: Analyzes screen time, social media habits, and sleep patterns.
- **Addiction Scoring**: Predicts a score from 0 to 1000 using Linear Regression.
- **Risk Assessment**: Categorizes users into Low, Medium, or High risk using K-Means Clustering.
- **Actionable Plans**: Suggests specific reductions (e.g., "Reduce social media by 1.5 hrs") to improve digital well-being.

## 📁 Project Structure
- `digital_addiction_detector.ipynb`: The primary Jupyter Notebook with all logic and visualizations.
- `requirements.txt`: Python dependencies.

## 🛠️ Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook using Jupyter Lab or VS Code.

## 📊 Logic
- **Regression**: Trained on synthetic data to weight social media usage and lack of sleep as high indicators of addiction.
- **Clustering**: Automatically identifies boundaries between "Healthy", "Warning", and "Addicted" users based on their scores and screen time.
