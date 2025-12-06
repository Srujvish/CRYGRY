# INSTITUTIONAL BEHAVIOR DETECTION SYSTEM
# AI-BASED INSTITUTIONAL FLOW ANALYSIS

import os
import time
import requests
import pandas as pd
import numpy as np
import warnings
import hmac
import hashlib
import threading
import math
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# --- SYSTEM CONFIGURATION --- 
SYSTEM_KEY = os.getenv("SYSTEM_KEY")
SYSTEM_SECRET = os.getenv("SYSTEM_SECRET")
API_BASE = "https://open-api.bingx.com"

# --- ALERT SERVICE --- 
ALERT_TOKEN = os.getenv("ALERT_TOKEN")
ALERT_TARGET = os.getenv("ALERT_TARGET")

# --- REAL CRYPTO SYMBOLS --- 
DIGITAL_ASSETS = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT", 
    "BNB": "BNB-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT",
    "ADA": "ADA-USDT",
    "AVAX": "AVAX-USDT",
    "DOGE": "DOGE-USDT"
}

# --- INSTITUTIONAL BEHAVIOR CONFIG ---
LARGE_VOLUME_RATIO = 3.8    # Big size ratio
MIN_MOVE_SIZE = 0.018       # Minimum significant move
STOP_DISTANCE = 0.01        # Stop hunt distance
ABSORPTION_RATIO = 0.25     # Wick absorption ratio
MAX_SIGNALS_PER_HOUR = 2    # Institutions trade less frequently

# --- BEHAVIOR TYPES ---
BEHAVIOR_TYPES = {
    "large_accumulation": "LARGE ACCUMULATION",
    "large_distribution": "LARGE DISTRIBUTION", 
    "stop_reversal": "STOP REVERSAL",
    "liquidity_take": "LIQUIDITY TAKE"
}

# --- TRACKING ---
detected_signals = []
signal_tracker = 0
active_behaviors = {}
behavior_history = {}
last_signal_time = {}

def send_alert(message, reply_to=None):
    """Send alert notification"""
    try:
        if not ALERT_TOKEN or not ALERT_TARGET:
            print(f"📢 {message}")
            return None
            
        url = f"https://api.telegram.org/bot{ALERT_TOKEN}/sendMessage"
        payload = {"chat_id": ALERT_TARGET, "text": message, "parse_mode": "HTML"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print(f"Alert error: {e}")
        return None

# --- MARKET DATA FUNCTIONS ---
def get_market_data(symbol, interval="5m", limit=100):
    """Get price and volume data from BingX"""
    try:
        endpoint = "/openApi/swap/v3/quote/klines"
        params = f"symbol={symbol}&interval={interval}&limit={limit}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                klines = data['data']
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                return df
    except Exception as e:
        print(f"Error fetching {symbol} data: {e}")
    return None

def get_current_price(symbol):
    """Get current price from BingX"""
    try:
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = f"symbol={symbol}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                return float(data['data']['lastPrice'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
    return None

def get_order_book(symbol, limit=20):
    """Get order book depth"""
    try:
        endpoint = "/openApi/swap/v2/quote/depth"
        params = f"symbol={symbol}&limit={limit}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                return data['data']
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
    return None

# 🏛️ **INSTITUTIONAL BEHAVIOR AI** 🏛️
class LargeFlowAI:
    def __init__(self):
        self.accumulation_model = None
        self.distribution_model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load AI models trained on institutional behavior"""
        try:
            if os.path.exists("flow_accumulation_model.pkl"):
                self.accumulation_model = joblib.load("flow_accumulation_model.pkl")
                print("✅ Loaded accumulation model")
            else:
                self.accumulation_model = None
                
            if os.path.exists("flow_distribution_model.pkl"):
                self.distribution_model = joblib.load("flow_distribution_model.pkl")
                print("✅ Loaded distribution model")
            else:
                self.distribution_model = None
                
            if os.path.exists("flow_scaler.pkl"):
                self.scaler = joblib.load("flow_scaler.pkl")
                print("✅ Loaded scaler")
            else:
                self.scaler = None
            
            if not all([self.accumulation_model, self.distribution_model, self.scaler]):
                self.train_models()
            else:
                print("✅ All models loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.train_models()
    
    def train_models(self):
        """Train AI on institutional behavior patterns"""
        try:
            print("🏛️ Training flow detection models...")
            
            # ACCUMULATION PATTERNS (UP moves)
            X_acc = []
            y_acc = []
            
            # Positive examples - accumulation
            X_acc.append([4.2, 0.22, 0.85, 2.8, 0.18, 0.72, 0.65, 0.025, 1.4, 0.35])
            y_acc.append(1)
            
            X_acc.append([3.8, 0.18, 0.82, 2.5, 0.15, 0.68, 0.62, 0.022, 1.3, 0.4])
            y_acc.append(1)
            
            # Negative examples
            X_acc.append([1.2, 0.05, 0.3, 0.8, 0.03, 0.2, 0.4, 0.005, 0.5, 0.8])
            y_acc.append(0)
            
            # DISTRIBUTION PATTERNS (DOWN moves)
            X_dist = []
            y_dist = []
            
            # Positive examples - distribution
            X_dist.append([4.0, 0.2, 0.15, 2.7, 0.22, 0.7, 0.3, 0.023, 1.35, 0.25])
            y_dist.append(1)
            
            X_dist.append([4.3, 0.23, 0.12, 3.1, 0.24, 0.78, 0.25, 0.027, 1.55, 0.28])
            y_dist.append(1)
            
            # Negative examples
            X_dist.append([2.0, 0.08, 0.5, 1.5, 0.1, 0.4, 0.5, 0.01, 0.8, 0.85])
            y_dist.append(0)
            
            X_acc = np.array(X_acc)
            y_acc = np.array(y_acc)
            X_dist = np.array(X_dist)
            y_dist = np.array(y_dist)
            
            # Train accumulation model
            self.scaler = StandardScaler()
            X_acc_scaled = self.scaler.fit_transform(X_acc)
            
            self.accumulation_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            self.accumulation_model.fit(X_acc_scaled, y_acc)
            
            # Train distribution model
            X_dist_scaled = self.scaler.transform(X_dist)
            
            self.distribution_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
            self.distribution_model.fit(X_dist_scaled, y_dist)
            
            # Save models
            joblib.dump(self.accumulation_model, "flow_accumulation_model.pkl")
            joblib.dump(self.distribution_model, "flow_distribution_model.pkl")
            joblib.dump(self.scaler, "flow_scaler.pkl")
            
            print("✅ Models trained and saved")
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            self.accumulation_model = None
            self.distribution_model = None
            self.scaler = None
    
    def extract_flow_features(self, df):
        """Extract features revealing large flow behavior"""
        try:
            if df is None or len(df) < 10:
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            open_price = df['open']
            
            # 1. VOLUME SIGNATURE
            vol_avg_10 = volume.rolling(10).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            volume_signature = current_vol / (vol_avg_10 if vol_avg_10 > 0 else 1)
            
            # 2. ABSORPTION RATIO
            current_body = abs(close.iloc[-1] - open_price.iloc[-1])
            lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
            absorption_ratio = lower_wick / (current_body if current_body > 0 else 1)
            
            # 3. DISTRIBUTION RATIO
            distribution_ratio = upper_wick / (current_body if current_body > 0 else 1)
            
            # 4. PRICE STRENGTH
            recent_low = low.iloc[-8:-2].min()
            recent_high = high.iloc[-8:-2].max()
            current_price = close.iloc[-1]
            
            if current_price > recent_high * 0.9:
                price_strength = (current_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            else:
                price_strength = 0.3
            
            # 5. VOLUME CONFIRMATION
            vol_prev_3 = volume.iloc[-4:-1].mean()
            vol_confirmation = current_vol / (vol_prev_3 if vol_prev_3 > 0 else 1)
            
            # 6. MOMENTUM QUALITY
            price_change_3 = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] if close.iloc[-4] > 0 else 0
            momentum_quality = price_change_3 / (volume_signature if volume_signature > 0 else 1)
            
            # 7. ORDER BOOK IMBALANCE
            order_book = get_order_book("BTC-USDT", limit=10)
            if order_book:
                bids = sum([float(bid[1]) for bid in order_book['bids'][:5]])
                asks = sum([float(ask[1]) for ask in order_book['asks'][:5]])
                book_imbalance = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
            else:
                book_imbalance = 0
            
            # 8. TREND ALIGNMENT
            ma_20 = close.rolling(20).mean().iloc[-1]
            trend_alignment = 1 if (current_price > ma_20 and close.iloc[-1] > close.iloc[-2]) else 0.5
            
            # 9. PRESSURE INDEX
            buying_pressure = sum([1 for i in range(-3, 0) if close.iloc[i] > open_price.iloc[i]])
            selling_pressure = sum([1 for i in range(-3, 0) if close.iloc[i] < open_price.iloc[i]])
            pressure_index = buying_pressure / (buying_pressure + selling_pressure + 1)
            
            # 10. TIME EFFICIENCY
            utc_now = datetime.utcnow()
            hour = utc_now.hour
            if (0 <= hour < 4) or (8 <= hour < 12) or (12 <= hour < 16):
                time_efficiency = 0.8
            else:
                time_efficiency = 0.4
            
            features = [
                volume_signature,
                absorption_ratio,
                distribution_ratio,
                price_strength,
                vol_confirmation,
                momentum_quality,
                book_imbalance,
                trend_alignment,
                pressure_index,
                time_efficiency
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting flow features: {e}")
            return None
    
    def detect_accumulation(self, df):
        """Detect accumulation behavior"""
        if self.accumulation_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_flow_features(df)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.accumulation_model.predict(features_scaled)[0]
            probability = self.accumulation_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            return bool(prediction), confidence
        except Exception as e:
            print(f"⚠️ Error in accumulation detection: {e}")
            return False, 0.0
    
    def detect_distribution(self, df):
        """Detect distribution behavior"""
        if self.distribution_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_flow_features(df)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.distribution_model.predict(features_scaled)[0]
            probability = self.distribution_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            return bool(prediction), confidence
        except Exception as e:
            print(f"⚠️ Error in distribution detection: {e}")
            return False, 0.0

print("🚀 Initializing Large Flow AI...")
flow_ai = LargeFlowAI()
print("✅ Flow AI initialized!")

# --- BEHAVIOR DETECTION FUNCTIONS ---
def detect_liquidity_zones(df, lookback=20):
    """Detect liquidity zones"""
    try:
        high_series = df['high'].dropna()
        low_series = df['low'].dropna()
        
        if len(high_series) > lookback:
            high_zone = float(high_series.rolling(lookback).max().iloc[-2])
        else:
            high_zone = float(high_series.max()) if len(high_series) > 0 else None
            
        if len(low_series) > lookback:
            low_zone = float(low_series.rolling(lookback).min().iloc[-2])
        else:
            low_zone = float(low_series.min()) if len(low_series) > 0 else None
        
        return high_zone, low_zone
    except Exception as e:
        print(f"Error detecting liquidity zones: {e}")
        return None, None

def detect_large_accumulation(df):
    """Detect large accumulation behavior (like CE)"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 10:
            return None
        
        # 1. VOLUME SIGNATURE
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_10 * LARGE_VOLUME_RATIO:
            return None
        
        # 2. ABSORPTION CANDLE
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if lower_wick < current_body * ABSORPTION_RATIO:
            return None
        
        # 3. PRICE HOLDING
        support_level = low.iloc[-8:-2].min()
        if close.iloc[-1] < support_level * 0.992:
            return None
        
        # 4. AI CONFIRMATION
        accumulation_detected, confidence = flow_ai.detect_accumulation(df)
        if not accumulation_detected or confidence < 0.82:
            return None
        
        # 5. FOLLOW-THROUGH
        if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
            return None
        
        return "UP"
        
    except Exception as e:
        print(f"Error in accumulation detection: {e}")
        return None

def detect_large_distribution(df):
    """Detect large distribution behavior (like PE)"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 10:
            return None
        
        # Avoid late moves
        hour = datetime.utcnow().hour
        if hour >= 22:  # Late UTC
            return None
        
        # 1. VOLUME SIGNATURE
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_10 * (LARGE_VOLUME_RATIO + 0.5):
            return None
        
        # 2. DISTRIBUTION CANDLE
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if upper_wick < current_body * ABSORPTION_RATIO:
            return None
        
        # 3. PRICE REJECTION
        resistance_level = high.iloc[-8:-2].max()
        if close.iloc[-1] > resistance_level * 1.008:
            return None
        
        # 4. AI CONFIRMATION
        distribution_detected, confidence = flow_ai.detect_distribution(df)
        if not distribution_detected or confidence < 0.85:
            return None
        
        # 5. FOLLOW-THROUGH
        if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
            return None
        
        return "DOWN"
        
    except Exception as e:
        print(f"Error in distribution detection: {e}")
        return None

def detect_stop_reversal(df):
    """Detect stop hunt reversals"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 15:
            return None
        
        # Find recent range
        recent_high = high.iloc[-12:-2].max()
        recent_low = low.iloc[-12:-2].min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        # Volume check
        vol_avg = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # BULL STOP HUNT (then UP)
        if (current_low < recent_low * (1 - STOP_DISTANCE) and
            current_close > recent_low * 1.008 and
            current_vol > vol_avg * 3.5 and
            current_close > prev_close and
            (current_high - current_close) < (current_close - current_low) * 0.4):
            
            return "UP"
        
        # BEAR STOP HUNT (then DOWN)
        if (current_high > recent_high * (1 + STOP_DISTANCE) and
            current_close < recent_high * 0.992 and
            current_vol > vol_avg * 4.0 and
            current_close < prev_close and
            (current_close - current_low) < (current_high - current_close) * 0.4):
            
            return "DOWN"
        
    except Exception as e:
        print(f"Error in stop reversal detection: {e}")
        return None
    return None

def detect_liquidity_take(df):
    """Detect liquidity grabs"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 20:
            return None
        
        # Find liquidity zones
        high_zone, low_zone = detect_liquidity_zones(df, lookback=18)
        current_price = close.iloc[-1]
        
        if high_zone is None or low_zone is None:
            return None
        
        # Volume signature
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # LIQUIDITY TAKE AT HIGHS (then DOWN)
        if high_zone and abs(current_price - high_zone) <= high_zone * 0.006:
            if (current_vol > vol_avg_15 * 4.0 and
                high.iloc[-1] > high_zone * 1.008 and
                close.iloc[-1] < high_zone * 0.997 and
                volume.iloc[-1] > volume.iloc[-2] * 2.0):
                
                return "DOWN"
        
        # LIQUIDITY TAKE AT LOWS (then UP)
        if low_zone and abs(current_price - low_zone) <= low_zone * 0.006:
            if (current_vol > vol_avg_15 * 4.0 and
                low.iloc[-1] < low_zone * 0.992 and
                close.iloc[-1] > low_zone * 1.003 and
                volume.iloc[-1] > volume.iloc[-2] * 2.0):
                
                return "UP"
        
    except Exception as e:
        print(f"Error in liquidity take detection: {e}")
        return None
    return None

def can_send_signal(asset):
    """Check if signal can be sent"""
    current_time = time.time()
    
    if asset in last_signal_time:
        time_since_last = current_time - last_signal_time[asset]
        if time_since_last < 2700:  # 45 minutes
            print(f"⏳ {asset} in cooldown: {int(2700 - time_since_last)}s remaining")
            return False
    
    return True

def update_signal_time(asset):
    """Update last signal time"""
    last_signal_time[asset] = time.time()

def analyze_asset_flow(asset_name, asset_symbol):
    """Analyze asset for institutional flow"""
    df_5min = get_market_data(asset_symbol, interval="5m", limit=100)
    df_1min = get_market_data(asset_symbol, interval="1m", limit=100)
    
    if df_5min is None or df_1min is None:
        return None
    
    if len(df_5min) < 20:
        return None
    
    print(f"🔍 Analyzing {asset_name} ({asset_symbol})...")
    
    # Check all behaviors
    behaviors = [
        ("large_accumulation", detect_large_accumulation(df_5min)),
        ("stop_reversal", detect_stop_reversal(df_5min)),
        ("liquidity_take", detect_liquidity_take(df_5min)),
        ("large_distribution", detect_large_distribution(df_5min))
    ]
    
    for behavior_type, direction in behaviors:
        if direction:
            print(f"✅ {asset_name}: {BEHAVIOR_TYPES[behavior_type]} detected - {direction}")
            return direction, asset_symbol, df_5min, behavior_type
    
    return None

def send_flow_alert(asset_name, asset_symbol, direction, df, behavior_type):
    """Send institutional flow alert"""
    global signal_tracker
    
    if not can_send_signal(asset_name):
        return
    
    current_price = get_current_price(asset_symbol)
    if current_price is None:
        return
    
    # Calculate movement range
    movement_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
    
    if direction == "UP":
        entry = current_price
        levels = [
            round(entry * (1 + movement_range * 0.8), 4),
            round(entry * (1 + movement_range * 1.5), 4),
            round(entry * (1 + movement_range * 2.2), 4),
            round(entry * (1 + movement_range * 3.0), 4)
        ]
        stop_level = round(entry * (1 - movement_range * 0.35), 4)
        
    else:  # DOWN
        entry = current_price
        levels = [
            round(entry * (1 - movement_range * 0.8), 4),
            round(entry * (1 - movement_range * 1.5), 4),
            round(entry * (1 - movement_range * 2.2), 4),
            round(entry * (1 - movement_range * 3.0), 4)
        ]
        stop_level = round(entry * (1 + movement_range * 0.35), 4)
    
    levels_str = " → ".join([f"${t:.4f}" for t in levels])
    
    signal_id = f"FLOW{signal_tracker:04d}"
    signal_tracker += 1
    
    behavior_name = BEHAVIOR_TYPES[behavior_type]
    
    if behavior_type == "large_accumulation":
        message = (f"🏛️ <b>LARGE ACCUMULATION DETECTED</b> 🏛️\n"
                  f"🎯 {asset_name}\n"
                  f"🔢 {asset_symbol}\n"
                  f"<b>ENTRY: ${entry:.4f}</b>\n"
                  f"LEVELS: {levels_str}\n"
                  f"REFERENCE: ${stop_level:.4f}\n"
                  f"DIRECTION: {direction}\n"
                  f"SIGNAL: {signal_id}\n"
                  f"⚠️ LARGE FLOW ACCUMULATING BEFORE UP MOVE")
    
    elif behavior_type == "large_distribution":
        message = (f"🏛️ <b>LARGE DISTRIBUTION DETECTED</b> 🏛️\n"
                  f"🎯 {asset_name}\n"
                  f"🔢 {asset_symbol}\n"
                  f"<b>ENTRY: ${entry:.4f}</b>\n"
                  f"LEVELS: {levels_str}\n"
                  f"REFERENCE: ${stop_level:.4f}\n"
                  f"DIRECTION: {direction}\n"
                  f"SIGNAL: {signal_id}\n"
                  f"⚠️ LARGE FLOW DISTRIBUTING BEFORE DOWN MOVE")
    
    elif behavior_type == "stop_reversal":
        message = (f"🏛️ <b>STOP HUNT REVERSAL</b> 🏛️\n"
                  f"🎯 {asset_name}\n"
                  f"🔢 {asset_symbol}\n"
                  f"<b>ENTRY: ${entry:.4f}</b>\n"
                  f"LEVELS: {levels_str}\n"
                  f"REFERENCE: ${stop_level:.4f}\n"
                  f"DIRECTION: {direction}\n"
                  f"SIGNAL: {signal_id}\n"
                  f"⚠️ STOP HUNT COMPLETE - REVERSAL IMMINENT")
    
    else:  # liquidity_take
        message = (f"🏛️ <b>LIQUIDITY GRAB DETECTED</b> 🏛️\n"
                  f"🎯 {asset_name}\n"
                  f"🔢 {asset_symbol}\n"
                  f"<b>ENTRY: ${entry:.4f}</b>\n"
                  f"LEVELS: {levels_str}\n"
                  f"REFERENCE: ${stop_level:.4f}\n"
                  f"DIRECTION: {direction}\n"
                  f"SIGNAL: {signal_id}\n"
                  f"⚠️ LIQUIDITY TAKEN - BIG MOVE FOLLOWING")
    
    thread_id = send_alert(message)
    update_signal_time(asset_name)
    
    # Store for tracking
    flow_id = f"{asset_symbol}_{signal_id}"
    active_behaviors[flow_id] = {
        "asset": asset_name,
        "symbol": asset_symbol,
        "direction": direction,
        "entry": entry,
        "levels": levels,
        "reference": stop_level,
        "thread_id": thread_id,
        "signal_id": signal_id,
        "timestamp": time.time()
    }
    
    return signal_id

def run_flow_scanner():
    """Scan all assets for institutional flow"""
    print("🔍 Scanning for large flow behavior...")
    
    threads = []
    results = []
    
    def scan_asset(asset_name, asset_symbol):
        result = analyze_asset_flow(asset_name, asset_symbol)
        if result:
            results.append(result)
    
    # Scan top 4 assets
    top_assets = list(DIGITAL_ASSETS.items())[:4]
    
    for asset_name, asset_symbol in top_assets:
        t = threading.Thread(target=scan_asset, args=(asset_name, asset_symbol))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    # Send alerts
    for result in results:
        direction, symbol, df, behavior_type = result
        asset_name = [k for k, v in DIGITAL_ASSETS.items() if v == symbol][0]
        send_flow_alert(asset_name, symbol, direction, df, behavior_type)
    
    print(f"✅ Scan complete. Behaviors detected: {len(results)}")

# --------- MAIN EXECUTION ---------
print("=" * 60)
print("🏛️ INSTITUTIONAL FLOW DETECTION SYSTEM")
print("🎯 DETECTING LARGE ACCUMULATION/DISTRIBUTION")
print("🚨 STOP HUNT REVERSAL DETECTION")
print("🌊 LIQUIDITY GRAB IDENTIFICATION")
print("=" * 60)

# Send startup message
send_alert("🏛️ <b>INSTITUTIONAL FLOW SYSTEM ACTIVATED</b>\n"
           "🎯 Detecting large accumulation/distribution\n"
           "🚨 Stop hunt reversal detection\n"
           "🌊 Liquidity grab identification\n"
           f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC")

iteration = 0
while True:
    iteration += 1
    try:
        print(f"\n🔄 Flow Analysis Iteration {iteration}")
        run_flow_scanner()
        
        # Clean old behaviors (older than 6 hours)
        current_time = time.time()
        expired_behaviors = []
        for behavior_id, behavior in active_behaviors.items():
            if current_time - behavior['timestamp'] > 21600:  # 6 hours
                expired_behaviors.append(behavior_id)
        
        for behavior_id in expired_behaviors:
            del active_behaviors[behavior_id]
        
        # Wait before next scan
        time.sleep(60)
        
    except Exception as e:
        print(f"⚠️ Scanner error: {e}")
        time.sleep(60)
