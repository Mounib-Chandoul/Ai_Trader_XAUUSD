from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import json
from bot import GoldBot
from ml_model import fetch_real_gold_data, MLModel
from indicators import calculate_atr
import threading

app = Flask(__name__)
CORS(app)

# Global state to store backtest results
results_cache = {
    "status": "idle",
    "data": None
}

def run_backtest_thread():
    global results_cache
    results_cache["status"] = "running"
    try:
        # 1. Fetch Real Data (15m granularity for last 60 days)
        data = fetch_real_gold_data(period="60d", interval="15m")
        
        # 2. Split into Train/Backtest (70/30)
        split_idx = int(len(data) * 0.7)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # 3. Train ML Model
        model = MLModel(horizon=16) 
        model.train(train_data)
        
        # 4. Run Simulation
        bot = GoldBot(ml_model=model)
        window_size = 200
        
        trade_history = []
        equity_curve = [10000]
        
        for i in range(window_size, len(test_data)):
            window = test_data.iloc[i-window_size:i+1]
            
            # Capture balance before to check if any portion of trade closed
            balance_before = bot.risk_manager.balance
            
            bot.run_iteration(window)
            
            # If balance changed, a partial or full close occurred
            if bot.risk_manager.balance != balance_before:
                last_pnl = bot.trades[-1]
                trade_history.append({
                    "time": window.index[-1].strftime("%Y-%m-%d %H:%M"),
                    "pnl": round(last_pnl, 2),
                    "balance": round(bot.risk_manager.balance, 2)
                })
                equity_curve.append(round(bot.risk_manager.balance, 2))

        # Final stats
        total_pnl = sum(bot.trades)
        win_rate = (len([t for t in bot.trades if t > 0]) / len(bot.trades) * 100) if bot.trades else 0
        
        results_cache["data"] = {
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": len(bot.trades),
            "trades": trade_history,
            "equity_curve": equity_curve,
            "prices": test_data['close'].tolist(),
            "times": [t.strftime("%Y-%m-%d %H:%M") for t in test_data.index]
        }
        results_cache["status"] = "finished"
    except Exception as e:
        print(f"Error in backtest: {e}")
        results_cache["status"] = "error"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_backtest', methods=['POST'])
def start_backtest():
    if results_cache["status"] == "running":
        return jsonify({"status": "already_running"})
    
    thread = threading.Thread(target=run_backtest_thread)
    thread.start()
    return jsonify({"status": "started"})

@app.route('/api/results', methods=['GET'])
def get_results():
    return jsonify(results_cache)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
