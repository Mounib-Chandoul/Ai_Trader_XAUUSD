# AI-Powered Gold (XAU/USD) Trading Bot

![Dashboard Preview](dashboard.png)

This repository contains a professional-grade automated trading bot architecture designed for Gold (XAU/USD). The system integrates **Machine Learning** for signal generation with a rigorous **Risk Management** engine, simulating real-world execution costs like spreads and commissions.

## 🚀 Key Features
* **ML-Driven Strategy:** Uses a trained Machine Learning model to predict price movements with a 4-hour forecast horizon.
* **Smart Session Management:** Targets high-volatility windows (London and New York sessions) to maximize liquidity and execution quality.
* **Professional Trade Management:**
    * **Dynamic Partial Scaling:** Scales out 50% at TP1 to lock in gains.
    * **Risk Mitigation:** Automatically moves Stop-Loss to Break-Even after TP1 is hit.
    * **ATR-Based Stops:** Volatility-adjusted stop-loss calculation using the Average True Range.
* **Realistic Backtesting:** Incorporates dynamic spread simulation and commission costs based on market hours to ensure "backtest-to-live" performance parity.
* **Circuit Breaker:** Built-in safety mechanism to protect account equity during high-loss sequences.

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **Data Handling:** `pandas`, `numpy`, `yfinance`
* **Modeling:** Custom `MLModel` wrapper (supports Scikit-learn/XGBoost backends)
* **Web UI:** FastAPI with Jinja2 Templates
* **Architecture:** Object-Oriented Design (MVC-inspired)

## 📂 Project Structure
```text
/GoldBot
├── /templates          # HTML/Frontend files for the Dashboard
│   └── index.html      
├── main.py             # Execution engine & backtest loop
├── indicators.py       # Technical indicator library (ATR, etc.)
├── risk.py             # Position sizing & risk management logic
├── signals.py          # Signal generation interface
└── ml_model.py         # Data fetching & ML training pipeline
