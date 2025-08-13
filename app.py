from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)

# Initialize models and scaler
models = {}
scaler = MinMaxScaler()

def initialize_models():
    """Initialize pre-trained models with sample data"""
    np.random.seed(42)
    X_sample = np.random.rand(1000, 23)  # 23 features after feature selection
    y_sample = np.random.rand(1000) * 100000  # Sales values
    
    # Initialize models
    models['linear_regression'] = LinearRegression()
    models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['knn'] = KNeighborsRegressor(n_neighbors=5)
    models['xgboost'] = XGBRegressor(random_state=42)
    
    # Fit models with sample data
    for model_name, model in models.items():
        model.fit(X_sample, y_sample)
    
    # Fit scaler
    scaler.fit(X_sample)

def preprocess_input(store, dept, date, is_holiday, temperature=70, fuel_price=3.5, cpi=200, unemployment=7):
    """Preprocess input data to match model expectations"""
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    week = date_obj.isocalendar()[1]
    
    features = [
        store,                    # Store
        dept,                     # Department
        year - 2010,             # Year normalized
        month,                   # Month
        week,                    # Week
        int(is_holiday),         # IsHoliday
        temperature,             # Temperature
        fuel_price,              # Fuel_Price
        cpi,                     # CPI
        unemployment,            # Unemployment
        0,                       # Total_MarkDown
        50000,                   # max sales
        0,                       # min sales
        25000,                   # mean sales
        20000,                   # median sales
        15000,                   # std sales
        0.5 if store <= 20 else 0,  # Store type A
        0.3 if 20 < store <= 35 else 0,  # Store type B
        0.2 if store > 35 else 0,  # Store type C
        42000,                   # Size (normalized)
        1 if dept in [1, 92, 95] else 0,  # Popular departments
        1 if month in [11, 12] else 0,    # Holiday season
        1 if week in [47, 48, 49, 50, 51, 52] else 0  # Holiday weeks
    ]
    
    return np.array(features).reshape(1, -1)

def get_prediction_confidence(predictions):
    """Calculate prediction confidence based on model agreement"""
    std_dev = np.std(predictions)
    mean_pred = np.mean(predictions)
    
    cv = std_dev / mean_pred if mean_pred > 0 else 0
    confidence = max(0, min(100, 100 - (cv * 100)))
    
    return confidence

def get_influencing_factors(store, dept, date, is_holiday):
    """Generate influencing factors based on input"""
    factors = []
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    month = date_obj.month
    
    if month in [11, 12]:
        factors.append("Holiday Season (November-December)")
    elif month in [6, 7, 8]:
        factors.append("Summer Season")
    elif month in [1, 2]:
        factors.append("Post-Holiday Period")

    if store <= 20:
        factors.append("High-Traffic Store Location")
    elif store > 35:
        factors.append("Smaller Store Format")
    
    high_demand_depts = [1, 92, 95, 72, 74]
    if dept in high_demand_depts:
        factors.append("High-Demand Department")
    
    if is_holiday:
        factors.append("Holiday Week Impact")
    
    factors.extend([
        "Historical Sales Trends",
        "Seasonal Patterns",
        "Economic Indicators"
    ])
    
    return factors[:5]

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
        return file.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        store = int(data['store'])
        dept = int(data['dept'])
        date = data['date']
        is_holiday = data['isHoliday']
        
        features = preprocess_input(store, dept, date, is_holiday)
        features_scaled = scaler.transform(features)
        
        predictions = {}
        pred_values = []
        
        for model_name, model in models.items():
            pred = model.predict(features_scaled)[0]
            pred = max(0, pred)
            predictions[model_name] = pred
            pred_values.append(pred)
        
        ensemble_prediction = np.mean(pred_values)
        confidence = get_prediction_confidence(pred_values)
        factors = get_influencing_factors(store, dept, date, is_holiday)
        
        formatted_prediction = f"${ensemble_prediction:,.2f}"
        model_results = {}
        for model_name, pred in predictions.items():
            model_results[model_name.replace('_', ' ').title()] = f"${pred:,.2f}"
        
        return jsonify({
            'success': True,
            'prediction': formatted_prediction,
            'confidence': round(confidence, 1),
            'factors': factors,
            'model_results': model_results,
            'ensemble_prediction': ensemble_prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Generate sample analytics data"""
    try:
        analytics_data = {
            'total_predictions': np.random.randint(5000, 10000),
            'avg_accuracy': 87.5,
            'top_stores': [
                {'id': 2, 'sales': 2847392, 'accuracy': 92.1},
                {'id': 4, 'sales': 2634821, 'accuracy': 89.7},
                {'id': 14, 'sales': 2456783, 'accuracy': 91.3},
                {'id': 13, 'sales': 2398472, 'accuracy': 88.9},
                {'id': 10, 'sales': 2287654, 'accuracy': 90.2}
            ],
            'top_departments': [
                {'id': 92, 'name': 'Grocery', 'sales': 8947392},
                {'id': 95, 'name': 'Clothing', 'sales': 6234821},
                {'id': 1, 'name': 'General Merchandise', 'sales': 5856783},
                {'id': 72, 'name': 'Electronics', 'sales': 4398472},
                {'id': 74, 'name': 'Appliances', 'sales': 3287654}
            ],
            'monthly_trends': [
                {'month': 'Jan', 'sales': 45283947},
                {'month': 'Feb', 'sales': 42847263},
                {'month': 'Mar', 'sales': 48392847},
                {'month': 'Apr', 'sales': 46284729},
                {'month': 'May', 'sales': 49274839},
                {'month': 'Jun', 'sales': 52847362},
                {'month': 'Jul', 'sales': 54928374},
                {'month': 'Aug', 'sales': 53746829},
                {'month': 'Sep', 'sales': 48293847},
                {'month': 'Oct', 'sales': 51847293},
                {'month': 'Nov', 'sales': 62847392},
                {'month': 'Dec', 'sales': 78293847}
            ]
        }
        
        return jsonify(analytics_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)