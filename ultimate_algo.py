# MARKET ANALYSIS AND PATTERN DETECTION SYSTEM
# DATA ANALYSIS TOOL FOR MARKET RESEARCH
# ADAPTED FOR EXTERNAL DATA SOURCES

import os
import time
import requests
import pandas as pd
import numpy as np
import warnings
import hmac
import hashlib
import json
import threading
import math
from datetime import datetime, time as dtime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# --- EXTERNAL_API_CONFIG --- 
MARKET_DATA_KEY = os.getenv("MARKET_DATA_KEY")
MARKET_DATA_SECRET = os.getenv("MARKET_DATA_SECRET") 
DATA_SOURCE_URL = os.getenv("DATA_SOURCE_URL")

# --- NOTIFICATION_SERVICE --- 
ALERT_BOT_TOKEN = os.getenv("ALERT_BOT_TOKEN")
ALERT_CHANNEL_ID = os.getenv("ALERT_CHANNEL_ID")

# --------- SYMBOL LIST ---------
ANALYSIS_SYMBOLS = {
    "ASSET_A": "ASSET_A-USD",
    "ASSET_B": "ASSET_B-USD", 
    "ASSET_C": "ASSET_C-USD",
    "ASSET_D": "ASSET_D-USD",
    "ASSET_E": "ASSET_E-USD",
    "ASSET_F": "ASSET_F-USD",
    "ASSET_G": "ASSET_G-USD",
    "ASSET_H": "ASSET_H-USD",
    "ASSET_I": "ASSET_I-USD",
    "ASSET_J": "ASSET_J-USD",
    "ASSET_K": "ASSET_K-USD",
    "ASSET_L": "ASSET_L-USD",
    "ASSET_M": "ASSET_M-USD",
    "ASSET_N": "ASSET_N-USD",
    "ASSET_O": "ASSET_O-USD"
}

# --------- ANALYSIS CONFIG ---------
VOLUME_RATIO_THRESHOLD = 3.5
MIN_MOVEMENT = 0.025
WICK_PERCENTAGE = 0.3

# --------- TRACKING ---------
all_signals = []
signal_counter = 0
active_patterns = {}
last_pattern_time = {}

def send_notification(msg, reply_to=None):
    try:
        if not NOTIFY_TOKEN or not CHANNEL_ID:
            print(f"📢 {msg}")
            return None
            
        url = f"https://api.notification-service.org/bot{NOTIFY_TOKEN}/sendMessage"
        payload = {"chat_id": CHANNEL_ID, "text": msg, "parse_mode": "HTML"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print(f"Notification error: {e}")
        return None

# --------- DATA API FUNCTIONS ---------
def create_signature(secret, query_string):
    """Create API signature"""
    return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def get_market_data(symbol, interval="5m", limit=100):
    """Get market data from external source"""
    try:
        endpoint = "/api/v2/data/ohlcv"
        params = f"symbol={symbol}&interval={interval}&limit={limit}"
        
        url = f"{DATA_BASE_URL}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                ohlcv = data['data']
                df = pd.DataFrame(ohlcv, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                return df
    except Exception as e:
        print(f"Error fetching market data: {e}")
    return None

def get_current_price(symbol):
    """Get current price"""
    try:
        endpoint = "/api/v2/data/ticker"
        params = f"symbol={symbol}"
        
        url = f"{DATA_BASE_URL}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return float(data['data']['lastPrice'])
    except Exception as e:
        print(f"Error fetching current price: {e}")
    return None

def get_depth_data(symbol, limit=20):
    """Get market depth"""
    try:
        endpoint = "/api/v2/data/depth"
        params = f"symbol={symbol}&limit={limit}"
        
        url = f"{DATA_BASE_URL}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['data']
    except Exception as e:
        print(f"Error fetching depth data: {e}")
    return None

# 🏛️ **MARKET PATTERN DETECTION AI** 🏛️
class MarketPatternAI:
    def __init__(self):
        self.up_model = None
        self.down_model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load pattern detection models"""
        try:
            if os.path.exists("pattern_up_model.pkl"):
                self.up_model = joblib.load("pattern_up_model.pkl")
                print("✅ Loaded UP pattern model from disk")
            else:
                self.up_model = None
                
            if os.path.exists("pattern_down_model.pkl"):
                self.down_model = joblib.load("pattern_down_model.pkl")
                print("✅ Loaded DOWN pattern model from disk")
            else:
                self.down_model = None
                
            if os.path.exists("pattern_scaler.pkl"):
                self.scaler = joblib.load("pattern_scaler.pkl")
                print("✅ Loaded pattern scaler from disk")
            else:
                self.scaler = None
            
            if not all([self.up_model, self.down_model, self.scaler]):
                self.train_pattern_models()
            else:
                print("✅ All pattern models loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading pattern models: {e}")
            self.train_pattern_models()
    
    def train_pattern_models(self):
        """Train AI on market patterns"""
        try:
            print("🏛️ Training pattern detection AI...")
            
            # UP patterns
            X_up = []
            y_up = []
            
            X_up.append([4.5, 0.25, 0.1, 3.1, 0.24, 0.82, 0.12, 0.028, 1.75, 0.85])
            y_up.append(1)
            
            X_up.append([4.8, 0.28, 0.08, 3.4, 0.27, 0.88, 0.1, 0.032, 1.95, 0.82])
            y_up.append(1)
            
            X_up.append([1.8, 0.1, 0.4, 1.3, 0.08, 0.35, 0.35, 0.012, 0.9, 0.45])
            y_up.append(0)
            
            # DOWN patterns
            X_down = []
            y_down = []
            
            X_down.append([4.6, 0.1, 0.28, 3.2, 0.26, 0.84, 0.26, 0.03, 1.8, 0.22])
            y_down.append(1)
            
            X_down.append([4.9, 0.08, 0.32, 3.5, 0.29, 0.9, 0.3, 0.035, 2.0, 0.18])
            y_down.append(1)
            
            X_down.append([2.0, 0.35, 0.12, 1.5, 0.12, 0.42, 0.15, 0.015, 1.1, 0.72])
            y_down.append(0)
            
            X_up = np.array(X_up)
            y_up = np.array(y_up)
            X_down = np.array(X_down)
            y_down = np.array(y_down)
            
            self.scaler = StandardScaler()
            X_up_scaled = self.scaler.fit_transform(X_up)
            
            self.up_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=5,
                random_state=42
            )
            self.up_model.fit(X_up_scaled, y_up)
            
            X_down_scaled = self.scaler.transform(X_down)
            
            self.down_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=42,
                class_weight='balanced'
            )
            self.down_model.fit(X_down_scaled, y_down)
            
            joblib.dump(self.up_model, "pattern_up_model.pkl")
            joblib.dump(self.down_model, "pattern_down_model.pkl")
            joblib.dump(self.scaler, "pattern_scaler.pkl")
            
            print("✅ Pattern models trained and saved")
            
        except Exception as e:
            print(f"❌ Error training pattern models: {e}")
            self.up_model = None
            self.down_model = None
            self.scaler = None
    
    def extract_features(self, df_fast, df_slow):
        """Extract market features"""
        try:
            if df_fast is None or df_slow is None:
                return None
            
            close_fast = df_fast['close']
            high_fast = df_fast['high']
            low_fast = df_fast['low']
            volume_fast = df_fast['volume']
            open_fast = df_fast['open']
            
            if len(close_fast) < 20:
                return None
            
            vol_avg_5 = volume_fast.rolling(5).mean().iloc[-1]
            current_vol = volume_fast.iloc[-1]
            volume_ratio = current_vol / (vol_avg_5 if vol_avg_5 > 0 else 1)
            
            current_body = abs(close_fast.iloc[-1] - open_fast.iloc[-1])
            lower_wick = min(close_fast.iloc[-1], open_fast.iloc[-1]) - low_fast.iloc[-1]
            upper_wick = high_fast.iloc[-1] - max(close_fast.iloc[-1], open_fast.iloc[-1])
            wick_strength = max(lower_wick, upper_wick) / (current_body if current_body > 0 else 1)
            
            price_change_3 = (close_fast.iloc[-1] - close_fast.iloc[-4]) / close_fast.iloc[-4] if close_fast.iloc[-4] > 0 else 0
            
            vol_accel = (volume_fast.iloc[-1] - volume_fast.iloc[-2]) / (volume_fast.iloc[-2] if volume_fast.iloc[-2] > 0 else 1)
            
            depth_data = get_depth_data("ASSET_A-USD", limit=10)
            if depth_data:
                bids = sum([float(bid[1]) for bid in depth_data['bids'][:5]])
                asks = sum([float(ask[1]) for ask in depth_data['asks'][:5]])
                depth_balance = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
            else:
                depth_balance = 0
            
            utc_now = datetime.utcnow()
            hour = utc_now.hour
            if (0 <= hour < 4) or (8 <= hour < 12) or (12 <= hour < 16):
                time_factor = 0.8
            else:
                time_factor = 0.4
            
            movement_range = (high_fast.iloc[-1] - low_fast.iloc[-1]) / close_fast.iloc[-1]
            
            ma_20 = close_fast.rolling(20).mean().iloc[-1]
            trend_direction = 1 if close_fast.iloc[-1] > ma_20 else -1
            
            green_candles = sum([1 for i in range(-5, 0) if close_fast.iloc[i] > open_fast.iloc[i]])
            red_candles = sum([1 for i in range(-5, 0) if close_fast.iloc[i] < open_fast.iloc[i]])
            pressure_index = (green_candles - red_candles) / 5.0
            
            features = [
                volume_ratio,
                wick_strength,
                price_change_3 * 100,
                vol_accel,
                depth_balance,
                time_factor,
                movement_range * 100,
                trend_direction,
                pressure_index,
                abs(price_change_3) * 100
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def detect_up_pattern(self, df_fast, df_slow):
        """Detect upward pattern"""
        if self.up_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_features(df_fast, df_slow)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.up_model.predict(features_scaled)[0]
            probability = self.up_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            return bool(prediction), confidence
        except Exception as e:
            print(f"⚠️ Error in UP pattern detection: {e}")
            return False, 0.0
    
    def detect_down_pattern(self, df_fast, df_slow):
        """Detect downward pattern"""
        if self.down_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_features(df_fast, df_slow)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.down_model.predict(features_scaled)[0]
            probability = self.down_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            return bool(prediction), confidence
        except Exception as e:
            print(f"⚠️ Error in DOWN pattern detection: {e}")
            return False, 0.0

print("🚀 Initializing Pattern Detection System...")
pattern_ai = MarketPatternAI()
print("✅ Pattern Detection System initialized!")

# --------- PATTERN DETECTION ---------
def detect_upward_movement(symbol, df_fast, df_slow):
    """Detect upward movement pattern"""
    try:
        close_fast = df_fast['close']
        high_fast = df_fast['high']
        low_fast = df_fast['low']
        volume_fast = df_fast['volume']
        open_fast = df_fast['open']
        
        if len(close_fast) < 10:
            return None
        
        vol_avg_5 = volume_fast.rolling(5).mean().iloc[-1]
        current_vol = volume_fast.iloc[-1]
        if current_vol < vol_avg_5 * VOLUME_RATIO_THRESHOLD:
            return None
        
        current_body = abs(close_fast.iloc[-1] - open_fast.iloc[-1])
        lower_wick = min(close_fast.iloc[-1], open_fast.iloc[-1]) - low_fast.iloc[-1]
        if lower_wick < current_body * WICK_PERCENTAGE:
            return None
        
        if close_fast.iloc[-1] <= open_fast.iloc[-1]:
            return None
        
        up_detected, confidence = pattern_ai.detect_up_pattern(df_fast, df_slow)
        if not up_detected or confidence < 0.82:
            return None
        
        print(f"✅ UP pattern detected for {symbol}: confidence {confidence:.2f}")
        return "UP"
        
    except Exception as e:
        print(f"Error detecting UP pattern: {e}")
        return None

def detect_downward_movement(symbol, df_fast, df_slow):
    """Detect downward movement pattern"""
    try:
        close_fast = df_fast['close']
        high_fast = df_fast['high']
        low_fast = df_fast['low']
        volume_fast = df_fast['volume']
        open_fast = df_fast['open']
        
        if len(close_fast) < 10:
            return None
        
        vol_avg_5 = volume_fast.rolling(5).mean().iloc[-1]
        current_vol = volume_fast.iloc[-1]
        if current_vol < vol_avg_5 * (VOLUME_RATIO_THRESHOLD + 0.5):
            return None
        
        current_body = abs(close_fast.iloc[-1] - open_fast.iloc[-1])
        upper_wick = high_fast.iloc[-1] - max(close_fast.iloc[-1], open_fast.iloc[-1])
        if upper_wick < current_body * WICK_PERCENTAGE:
            return None
        
        if close_fast.iloc[-1] >= open_fast.iloc[-1]:
            return None
        
        down_detected, confidence = pattern_ai.detect_down_pattern(df_fast, df_slow)
        if not down_detected or confidence < 0.84:
            return None
        
        print(f"✅ DOWN pattern detected for {symbol}: confidence {confidence:.2f}")
        return "DOWN"
        
    except Exception as e:
        print(f"Error detecting DOWN pattern: {e}")
        return None

def get_fast_data(symbol):
    """Get fast interval data"""
    return get_market_data(symbol, interval="1m", limit=100)

def get_slow_data(symbol):
    """Get slow interval data"""
    return get_market_data(symbol, interval="5m", limit=100)

def analyze_symbol(symbol, symbol_name):
    """Analyze symbol for patterns"""
    df_fast = get_fast_data(symbol)
    df_slow = get_slow_data(symbol)
    
    if df_fast is None or df_slow is None:
        return None
    
    if len(df_fast) < 20:
        return None
    
    print(f"🔍 Analyzing {symbol_name} ({symbol})...")
    
    up_signal = detect_upward_movement(symbol, df_fast, df_slow)
    if up_signal:
        return up_signal, symbol, df_slow, "PATTERN_UP"
    
    down_signal = detect_downward_movement(symbol, df_fast, df_slow)
    if down_signal:
        return down_signal, symbol, df_slow, "PATTERN_DOWN"
    
    return None

def send_pattern_signal(symbol_name, symbol, direction, df, strategy):
    global signal_counter
    
    current_price = get_current_price(symbol)
    if current_price is None:
        return
    
    movement_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
    
    if direction == "UP":
        entry = current_price
        levels = [
            round(entry * (1 + movement_range * 0.8), 2),
            round(entry * (1 + movement_range * 1.5), 2),
            round(entry * (1 + movement_range * 2.5), 2),
        ]
        stop_level = round(entry * (1 - movement_range * 0.5), 2)
        
    else:
        entry = current_price
        levels = [
            round(entry * (1 - movement_range * 0.8), 2),
            round(entry * (1 - movement_range * 1.5), 2),
            round(entry * (1 - movement_range * 2.5), 2),
        ]
        stop_level = round(entry * (1 + movement_range * 0.5), 2)
    
    levels_str = " → ".join([f"${t:.2f}" for t in levels])
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    if direction == "UP":
        msg = (f"📈 <b>UPWARD PATTERN DETECTED</b> 📈\n"
               f"🎯 {symbol_name}\n"
               f"SYMBOL: <code>{symbol}</code>\n"
               f"<b>CURRENT: ${entry:.2f}</b>\n"
               f"PROJECTION: {levels_str}\n"
               f"REFERENCE: ${stop_level:.2f}\n"
               f"DIRECTION: {direction}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ PATTERN INDICATES UPWARD MOVEMENT")
    else:
        msg = (f"📉 <b>DOWNWARD PATTERN DETECTED</b> 📉\n"
               f"🎯 {symbol_name}\n"
               f"SYMBOL: <code>{symbol}</code>\n"
               f"<b>CURRENT: ${entry:.2f}</b>\n"
               f"PROJECTION: {levels_str}\n"
               f"REFERENCE: ${stop_level:.2f}\n"
               f"DIRECTION: {direction}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ PATTERN INDICATES DOWNWARD MOVEMENT")
    
    thread_id = send_notification(msg)
    
    pattern_id = f"{symbol}_{signal_id}"
    active_patterns[pattern_id] = {
        "symbol": symbol,
        "symbol_name": symbol_name,
        "direction": direction,
        "price": entry,
        "levels": levels,
        "reference": stop_level,
        "thread_id": thread_id,
        "signal_id": signal_id,
        "timestamp": time.time()
    }
    
    return signal_id

def run_pattern_scanner():
    """Scan all symbols for patterns"""
    print("🔍 Scanning for market patterns...")
    
    threads = []
    results = []
    
    def scan_symbol(symbol_name, symbol):
        result = analyze_symbol(symbol, symbol_name)
        if result:
            results.append(result)
    
    top_symbols = list(ANALYSIS_SYMBOLS.items())[:8]
    
    for symbol_name, symbol in top_symbols:
        t = threading.Thread(target=scan_symbol, args=(symbol_name, symbol))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    for result in results:
        direction, symbol, df, strategy = result
        symbol_name = [k for k, v in ANALYSIS_SYMBOLS.items() if v == symbol][0]
        send_pattern_signal(symbol_name, symbol, direction, df, strategy)
    
    print(f"✅ Scan complete. Patterns detected: {len(results)}")

# --------- MAIN EXECUTION ---------
print("=" * 60)
print("🚀 MARKET PATTERN DETECTION SYSTEM ACTIVATED")
print("🎯 ANALYZING MARKET MOVEMENT PATTERNS")
print("📈 UP/DOWN DIRECTIONAL ANALYSIS")
print("⚡ CONTINUOUS MARKET MONITORING")
print("=" * 60)

iteration = 0
while True:
    iteration += 1
    try:
        print(f"\n🔄 Analysis Iteration {iteration}")
        run_pattern_scanner()
        
        current_time = time.time()
        expired_patterns = []
        for pattern_id, pattern in active_patterns.items():
            if current_time - pattern['timestamp'] > 21600:
                expired_patterns.append(pattern_id)
        
        for pattern_id in expired_patterns:
            del active_patterns[pattern_id]
        
        time.sleep(60)
        
    except Exception as e:
        print(f"⚠️ Scanner error: {e}")
        time.sleep(60)
