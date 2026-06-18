import numpy as np
import pandas as pd
from indicators import (
    calculate_ema, calculate_atr, calculate_rsi, 
    calculate_macd, calculate_hurst, calculate_adx,
    calculate_kalman_filter, detect_divergence
)

class SignalGenerator:
    def __init__(self, ml_model=None):
        self.ml_model = ml_model

    def get_signal(self, df):
        """
        Generates signals using the ML model if available.
        Otherwise falls back to neutral.
        """
        if self.ml_model:
            prob, signal = self.ml_model.predict(df)
            return signal
        
        return 'NEUTRAL'
