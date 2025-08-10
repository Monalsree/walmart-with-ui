from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import random
import math

app = Flask(__name__)
CORS(app)

# Simulated machine learning model for Walmart sales forecasting
class WalmartSalesPredictor:
    def __init__(self):
        # Initialize with some realistic parameters
        self.base_sales = {
            1: 15000,   # Electronics
            2: 12000,   # Grocery
            3: 8000,    # Clothing
            4: 6000,    # Home & Garden
            5: 9000,    # Sports
            6: 7000,    # Automotive
            7: 11000,   # Pharmacy
            8: 5000,    # Jewelry
        }
        
        # Store multipliers (different store sizes/locations)
        self.store_multipliers = {}
        for i in range(1, 51):  # Support 50 stores
            self.store_multipliers[i] = random.uniform(0.7, 1.8)
    
    def predict_sales(self, store_id, dept_id, date_str, is_holiday=False):
        """
        Predict sales for given parameters
        """
        try:
            # Parse date
            forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Get base sales for department
            base_sales = self.base_sales.get(dept_id, 8000)
            
            # Apply store multiplier
            store_multiplier = self.store_multipliers.get(store_id, 1.0)
            
            # Seasonal adjustments
            month = forecast_date.month
            seasonal_multiplier = self._get_seasonal_multiplier(month)

            # Day of week effect
            day_of_week = forecast_date.weekday()
            day_multiplier = self._get_day_multiplier(day_of_week)
            
            # Holiday effect
            holiday_multiplier = 1.4 if is_holiday else 1.0
            
            # Random variation (±15%)
            random_factor = random.uniform(0.85, 1.15)
            
            # Calculate final prediction
            prediction = (base_sales * 
                         store_multiplier * 
                         seasonal_multiplier * 
                         day_multiplier * 
                         holiday_multiplier * 
                         random_factor)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(store_id, dept_id, is_holiday)
            
            return {
                'prediction': max(0, prediction),
                'confidence': confidence,
                'factors': {
                    'base_sales': base_sales,
                    'store_multiplier': store_multiplier,
                    'seasonal_multiplier': seasonal_multiplier,
                    'day_multiplier': day_multiplier,
                    'holiday_multiplier': holiday_multiplier,
                    'random_factor': random_factor
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error in prediction: {str(e)}")
    
    def _get_seasonal_multiplier(self, month):
        """Get seasonal adjustment factor"""
        seasonal_factors = {
            1: 0.9,   # January (post-holiday low)
            2: 0.85,  # February
            3: 1.0,   # March
            4: 1.05,  # April
            5: 1.1,   # May
            6: 1.15,  # June
            7: 1.2,   # July
            8: 1.15,  # August (back-to-school)
            9: 1.1,   # September
            10: 1.2,  # October
            11: 1.4,  # November (Black Friday)
            12: 1.5   # December (Holiday season)
        }
        return seasonal_factors.get(month, 1.0)
    
    def _get_day_multiplier(self, day_of_week):
        """Get day of week adjustment factor"""
        # Monday=0, Sunday=6
        day_factors = {
            0: 0.9,   # Monday
            1: 0.95,  # Tuesday
            2: 1.0,   # Wednesday
            3: 1.05,  # Thursday
            4: 1.2,   # Friday
            5: 1.3,   # Saturday
            6: 1.1    # Sunday
        }
        return day_factors.get(day_of_week, 1.0)
    
    def _calculate_confidence(self, store_id, dept_id, is_holiday):
        """Calculate prediction confidence score"""
        base_confidence = 0.85
        
        # Higher confidence for common departments
        if dept_id in [1, 2, 3]:  # Electronics, Grocery, Clothing
            base_confidence += 0.05
        
        # Higher confidence for established stores
        if store_id <= 20:
            base_confidence += 0.03
        
        # Slightly lower confidence for holidays (more variability)
        if is_holiday:
            base_confidence -= 0.02
        
        # Add small random variation
        confidence = base_confidence + random.uniform(-0.05, 0.05)
        
        return max(0.7, min(0.99, confidence))

# Initialize the predictor
predictor = WalmartSalesPredictor()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return open('index.html').read()

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """API endpoint for sales forecasting"""
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ['store', 'dept', 'date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        store_id = int(data['store'])
        dept_id = int(data['dept'])
        date_str = data['date']
        is_holiday = data.get('isHoliday', False)
        
        # Validate ranges
        if store_id < 1 or store_id > 50:
            return jsonify({'error': 'Store ID must be between 1 and 50'}), 400
        
        if dept_id < 1 or dept_id > 20:
            return jsonify({'error': 'Department ID must be between 1 and 20'}), 400
        
        # Get prediction
        result = predictor.predict_sales(store_id, dept_id, date_str, is_holiday)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """API endpoint for analytics data"""
    try:
        # Generate sample analytics data
        analytics_data = {
            'total_revenue': 2400000000,  # $2.4B
            'active_stores': 4743,
            'daily_customers': 234000,
            'top_departments': [
                {'name': 'Electronics', 'revenue': 485000000},
                {'name': 'Grocery', 'revenue': 423000000},
                {'name': 'Clothing', 'revenue': 312000000},
                {'name': 'Home & Garden', 'revenue': 298000000}
            ],
            'seasonal_trends': [
                {'season': 'Holiday Season', 'increase': 45},
                {'season': 'Back-to-School', 'increase': 32},
                {'season': 'Summer', 'increase': 18}
            ],
            'growth_metrics': {
                'revenue_growth': 12.5,
                'store_growth': 2.1,
                'customer_growth': 8.3
            }
        }
        
        return jsonify(analytics_data)
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Walmart Sales Forecasting API...")
    print(f"Server running on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Access the application at: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)