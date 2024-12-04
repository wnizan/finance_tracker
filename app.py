from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.signal import find_peaks

app = Flask(__name__)

def get_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return hist

def identify_patterns(data):
    patterns = []
    close_prices = data['Close'].values
    
    # Cup and Handle pattern
    def find_cup_and_handle(prices, window=60):
        patterns = []
        for i in range(window, len(prices)-20):
            segment = prices[i-window:i+20]
            min_idx = np.argmin(segment) + (i-window)
            
            if min_idx > i-window+10 and min_idx < i+10:
                left_height = segment[0]
                right_height = segment[-1]
                bottom = np.min(segment)
                
                if abs(left_height - right_height) < (left_height * 0.02) and \
                   (left_height - bottom) > (left_height * 0.15):
                    cup_height = left_height - bottom
                    target_price = right_height + cup_height
                    
                    is_current = (len(prices) - (i + 20)) <= 5
                    
                    patterns.append({
                        'type': 'Cup and Handle',
                        'start_idx': i-window,
                        'end_idx': i+20,
                        'bottom_idx': min_idx,
                        'prediction': 'Bullish',
                        'target_price': target_price,
                        'is_current': is_current
                    })
        return patterns

    # Head and Shoulders pattern
    def find_head_and_shoulders(prices, window=60):
        patterns = []
        peaks, _ = find_peaks(prices, distance=20)
        
        for i in range(1, len(peaks)-1):
            if i+1 >= len(peaks):
                break
                
            left_shoulder = prices[peaks[i-1]]
            head = prices[peaks[i]]
            right_shoulder = prices[peaks[i+1]]
            
            if head > left_shoulder and head > right_shoulder and \
               abs(left_shoulder - right_shoulder) < (left_shoulder * 0.05):
                neckline = min(prices[peaks[i-1]:peaks[i+1]])
                pattern_height = head - neckline
                target_price = neckline - pattern_height
                
                is_current = (len(prices) - peaks[i+1]) <= 5
                
                patterns.append({
                    'type': 'Head and Shoulders',
                    'start_idx': peaks[i-1],
                    'head_idx': peaks[i],
                    'end_idx': peaks[i+1],
                    'prediction': 'Bearish',
                    'target_price': target_price,
                    'is_current': is_current
                })
        return patterns

    # Double Top pattern
    def find_double_top(prices, window=40):
        patterns = []
        peaks, _ = find_peaks(prices, distance=15)
        
        for i in range(len(peaks)-1):
            if peaks[i+1] - peaks[i] < window:
                peak1 = prices[peaks[i]]
                peak2 = prices[peaks[i+1]]
                
                if abs(peak1 - peak2) < (peak1 * 0.02):
                    support_level = min(prices[peaks[i]:peaks[i+1]])
                    pattern_height = peak1 - support_level
                    target_price = support_level - pattern_height
                    
                    is_current = (len(prices) - peaks[i+1]) <= 5
                    
                    patterns.append({
                        'type': 'Double Top',
                        'start_idx': peaks[i],
                        'end_idx': peaks[i+1],
                        'prediction': 'Bearish',
                        'target_price': target_price,
                        'is_current': is_current
                    })
        return patterns

    patterns.extend(find_cup_and_handle(close_prices))
    patterns.extend(find_head_and_shoulders(close_prices))
    patterns.extend(find_double_top(close_prices))
    
    return patterns

def prepare_prediction_data(data):
    return data[['Close']].values

def generate_prediction(data, days):
    last_30_days = data[-30:]
    base_price = data[-1]
    
    np.random.seed(42)  
    daily_returns = np.random.normal(0, 0.015, 30)  
    
    cumulative_returns = np.exp(np.cumsum(daily_returns)) 
    
    all_predictions = base_price * cumulative_returns
    
    dates = pd.date_range(start=datetime.now(), periods=days)
    predictions = all_predictions[:days].tolist()
    
    return dates, predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    symbol = request.args.get('symbol', '^GSPC')  
    period = request.args.get('period', '1y')
    
    try:
        data = get_stock_data(symbol, period)
        patterns = identify_patterns(data)
        
        dates_3d, pred_3d = generate_prediction(data['Close'].values, 3)
        dates_1w, pred_1w = generate_prediction(data['Close'].values, 7)
        dates_1m, pred_1m = generate_prediction(data['Close'].values, 30)
        
        pattern_annotations = []
        pattern_shapes = []
        current_patterns = []
        
        for idx, pattern in enumerate(patterns):
            pattern_type = pattern['type']
            
            annotation = {
                'showarrow': True,
                'arrowhead': 1,
                'text': f'{pattern_type}<br>Prediction: {pattern["prediction"]}'
            }
            
            if pattern_type == 'Cup and Handle':
                start_date = data.index[pattern['start_idx']]
                end_date = data.index[pattern['end_idx']]
                bottom_date = data.index[pattern['bottom_idx']]
                
                annotation.update({
                    'x': bottom_date.strftime('%Y-%m-%d'),
                    'y': data['Close'][pattern['bottom_idx']]
                })
                
                if pattern['is_current']:
                    current_patterns.append({
                        'type': pattern_type,
                        'prediction': pattern['prediction'],
                        'target_price': pattern['target_price'],
                        'current_price': data['Close'].iloc[-1],
                        'potential_gain': ((pattern['target_price'] / data['Close'].iloc[-1]) - 1) * 100
                    })
                
            elif pattern_type == 'Head and Shoulders':
                start_date = data.index[pattern['start_idx']]
                head_date = data.index[pattern['head_idx']]
                end_date = data.index[pattern['end_idx']]
                
                annotation.update({
                    'x': head_date.strftime('%Y-%m-%d'),
                    'y': data['Close'][pattern['head_idx']]
                })
                
                if pattern['is_current']:
                    current_patterns.append({
                        'type': pattern_type,
                        'prediction': pattern['prediction'],
                        'target_price': pattern['target_price'],
                        'current_price': data['Close'].iloc[-1],
                        'potential_loss': ((pattern['target_price'] / data['Close'].iloc[-1]) - 1) * 100
                    })
                
            elif pattern_type == 'Double Top':
                start_date = data.index[pattern['start_idx']]
                end_date = data.index[pattern['end_idx']]
                
                annotation.update({
                    'x': start_date.strftime('%Y-%m-%d'),
                    'y': data['Close'][pattern['start_idx']]
                })
                
                if pattern['is_current']:
                    current_patterns.append({
                        'type': pattern_type,
                        'prediction': pattern['prediction'],
                        'target_price': pattern['target_price'],
                        'current_price': data['Close'].iloc[-1],
                        'potential_loss': ((pattern['target_price'] / data['Close'].iloc[-1]) - 1) * 100
                    })
            
            pattern_annotations.append(annotation)
        
        response = {
            'historical': {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'prices': data['Close'].tolist()
            },
            'predictions': {
                '3d': {
                    'dates': dates_3d.strftime('%Y-%m-%d').tolist(),
                    'prices': pred_3d
                },
                '1w': {
                    'dates': dates_1w.strftime('%Y-%m-%d').tolist(),
                    'prices': pred_1w
                },
                '1m': {
                    'dates': dates_1m.strftime('%Y-%m-%d').tolist(),
                    'prices': pred_1m
                }
            },
            'patterns': {
                'annotations': pattern_annotations,
                'shapes': pattern_shapes,
                'current_patterns': current_patterns
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
