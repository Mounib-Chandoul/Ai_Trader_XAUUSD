import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from indicators import (
    calculate_ema, calculate_atr, calculate_rsi, 
    calculate_macd, calculate_hurst, calculate_adx,
    calculate_kalman_filter
)

def fetch_real_gold_data(period="60d", interval="15m"):
    """Downloads historical gold futures data from Yahoo Finance."""
    print(f"Downloading {period} of {interval} data for GC=F...")
    df = yf.download("GC=F", period=period, interval=interval)
    
    # Handle MultiIndex columns (flatten them if they exist)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    # Standardize columns to lowercase for consistency with existing code
    df.columns = [str(col).lower() for col in df.columns]
    return df

def engineer_features(df):
    """Adds technical indicators and price returns as features for the ML model."""
    df = df.copy()
    initial_len = len(df)
    
    # 1. Basic Indicators (Adjusted for 15m frequency)
    df['ema21'] = calculate_ema(df, 21)
    df['ema50'] = calculate_ema(df, 50)
    df['ema_ratio'] = df['ema21'] / df['ema50']
    
    df['atr_rel'] = calculate_atr(df) / df['close']
    df['rsi'] = calculate_rsi(df)
    
    macd_line, signal_line, hist = calculate_macd(df)
    df['macd_hist_rel'] = hist / df['close']
    
    df['adx'] = calculate_adx(df)
    df['hurst'] = calculate_hurst(df['close'], window=100)
    
    # 2. Kalman Filter Residuals
    kalman = calculate_kalman_filter(df['close'])
    df['kalman_diff'] = (df['close'] - kalman) / df['close']
    
    # 3. Price Returns (Momentum - adjusted for 15m: 4 bars = 1h, 16 bars = 4h)
    for window in [4, 16, 48, 96]: 
        df[f'return_{window}b'] = df['close'].pct_change(window)
        
    # 4. Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=48).std()
    
    # Drop rows with NaN values created by indicators
    df = df.dropna()
    return df

def create_target(df, horizon=16):
    """Creates a binary target: 1 if price is higher in 'horizon' bars (4h), 0 otherwise."""
    df = df.copy()
    df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    return df.iloc[:-horizon]

class MLModel:
    def __init__(self, horizon=16):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        self.horizon = horizon
        self.features = []

    def train(self, df):
        """Trains the Random Forest model on the provided dataframe."""
        df_feats = engineer_features(df)
        df_final = create_target(df_feats, horizon=self.horizon)
        
        # Define features (exclude price data and target, and noisy absolute values)
        exclude = ['open', 'high', 'low', 'close', 'adj close', 'volume', 'target', 'ema21', 'ema50']
        self.features = [col for col in df_final.columns if col not in exclude]
        
        X = df_final[self.features]
        y = df_final['target']
        
        # Sequential split (no shuffling for time series)
        split = int(len(X) * 0.8) # More training data
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Out-of-sample Accuracy: {acc:.2%}")
        
        # Feature Importance
        importances = pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)
        print("\nTop 5 Features:")
        print(importances.head(5))
        
        return acc

    def predict(self, window_df):
        """Returns the probability of price increase and the signal."""
        df_feats = engineer_features(window_df)
        if len(df_feats) == 0:
            return 0.5, 'NEUTRAL'
            
        X = df_feats[self.features].tail(1)
        prob = self.model.predict_proba(X)[0][1] # Probability of class 1 (Up)
        
        # Strict thresholds for high-probability session trading
        if prob > 0.55:
            return prob, 'LONG'
        elif prob < 0.45:
            return prob, 'SHORT'
        return prob, 'NEUTRAL'

if __name__ == "__main__":
    # Test script
    data = fetch_real_gold_data()
    model = MLModel()
    model.train(data)
