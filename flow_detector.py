# =================== INSTITUTIONAL FLOW & CRYPTO TRADING BOT ===================
# AI-BASED INSTITUTIONAL FLOW DETECTION WITH CONTINUOUS MONITORING

import os
import time
import requests
import pandas as pd
import numpy as np
import warnings
import threading
import math
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =================== CREDENTIALS ===================
SYSTEM_KEY = os.getenv("SYSTEM_KEY")
SYSTEM_SECRET = os.getenv("SYSTEM_SECRET")
API_BASE = "https://open-api.bingx.com"

ALERT_TOKEN = os.getenv("ALERT_TOKEN")
ALERT_TARGET = os.getenv("ALERT_TARGET")

# =================== SYMBOLS & ASSETS ======================
TRADING_SYMBOLS = ["BTC-USDT", "ETH-USDT"]
DIGITAL_ASSETS = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT", 
    "BNB": "BNB-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT"
}

# =================== INSTITUTIONAL FLOW SETTINGS ============
INSTITUTIONAL_VOLUME_RATIO = 3.8
MIN_MOVE_FOR_ENTRY = 0.018
STOP_HUNT_DISTANCE = 0.01
ABSORPTION_WICK_RATIO = 0.25

# =================== TECHNICAL TRADING SETTINGS ============
EMA_PERIOD = 20
RSI_PERIOD = 14
CANDLE_LIMIT = 100

SL_BUFFER = 0.0020
TARGETS = [0.0025, 0.004, 0.007]

ABS_MAX_ENTRY_USD = {
    "BTC-USDT": 80.0,
    "ETH-USDT": 8.0
}

TRADING_MODES = {
    "SCALP": {
        "interval": "1m",
        "recent_hl_window": 3,
        "rsi_long_min": 48, "rsi_long_max": 72,
        "rsi_short_min": 28, "rsi_short_max": 52,
        "entry_buffer_long": 0.0004,
        "entry_buffer_short": 0.0004,
        "max_entry_drift": 0.0010,
        "immediate_mode": True,
        "immediate_tol": 0.0010,
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 20
    },
    "SWING": {
        "interval": "5m",
        "recent_hl_window": 10,
        "rsi_long_min": 45, "rsi_long_max": 75,
        "rsi_short_min": 25, "rsi_short_max": 55,
        "entry_buffer_long": 0.0015,
        "entry_buffer_short": 0.0015,
        "max_entry_drift": 0.0040,
        "immediate_mode": True,
        "immediate_tol": 0.0015,
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 30
    }
}

# =================== INSTITUTIONAL BEHAVIOR TYPES ===========
BEHAVIOR_TYPES = {
    "large_accumulation": "🏛️ LARGE ACCUMULATION",
    "large_distribution": "🏛️ LARGE DISTRIBUTION", 
    "stop_reversal": "🎯 STOP HUNT REVERSAL",
    "liquidity_take": "🌊 LIQUIDITY GRAB"
}

# =================== GLOBAL TRACKING ========================
signal_counter = 0
active_signals = {}
last_signal_time = {}
active_monitoring_threads = {}
completed_signals = []

# Signal cooldown periods (in seconds)
SIGNAL_COOLDOWN = 1800  # 30 minutes for same symbol

# =================== UTILITIES ==============================
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

def send_telegram(msg: str):
    """Alias for send_alert"""
    return send_alert(msg)

# =================== MARKET DATA FUNCTIONS ==================
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
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
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

# =================== INSTITUTIONAL FLOW AI ==================
class InstitutionalFlowAI:
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
                print("✅ All institutional AI models loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.train_models()
    
    def train_models(self):
        """Train AI on institutional behavior patterns"""
        try:
            print("🏛️ Training Institutional Flow AI models...")
            
            # INSTITUTIONAL ACCUMULATION PATTERNS (UP moves)
            X_acc = []
            y_acc = []
            
            X_acc.append([4.2, 0.22, 0.85, 2.8, 0.18, 0.72, 0.65, 0.025, 1.4, 0.35])
            y_acc.append(1)
            
            X_acc.append([3.8, 0.18, 0.82, 2.5, 0.15, 0.68, 0.62, 0.022, 1.3, 0.4])
            y_acc.append(1)
            
            X_acc.append([1.2, 0.05, 0.3, 0.8, 0.03, 0.2, 0.4, 0.005, 0.5, 0.8])
            y_acc.append(0)
            
            # INSTITUTIONAL DISTRIBUTION PATTERNS (DOWN moves)
            X_dist = []
            y_dist = []
            
            X_dist.append([4.0, 0.2, 0.15, 2.7, 0.22, 0.7, 0.3, 0.023, 1.35, 0.25])
            y_dist.append(1)
            
            X_dist.append([4.3, 0.23, 0.12, 3.1, 0.24, 0.78, 0.25, 0.027, 1.55, 0.28])
            y_dist.append(1)
            
            X_dist.append([2.0, 0.08, 0.5, 1.5, 0.1, 0.4, 0.5, 0.01, 0.8, 0.85])
            y_dist.append(0)
            
            X_acc = np.array(X_acc)
            y_acc = np.array(y_acc)
            X_dist = np.array(X_dist)
            y_dist = np.array(y_dist)
            
            self.scaler = StandardScaler()
            X_acc_scaled = self.scaler.fit_transform(X_acc)
            
            self.accumulation_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            self.accumulation_model.fit(X_acc_scaled, y_acc)
            
            X_dist_scaled = self.scaler.transform(X_dist)
            
            self.distribution_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
            self.distribution_model.fit(X_dist_scaled, y_dist)
            
            joblib.dump(self.accumulation_model, "flow_accumulation_model.pkl")
            joblib.dump(self.distribution_model, "flow_distribution_model.pkl")
            joblib.dump(self.scaler, "flow_scaler.pkl")
            
            print("✅ Institutional AI models trained and saved")
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            self.accumulation_model = None
            self.distribution_model = None
            self.scaler = None
    
    def extract_flow_features(self, df):
        """Extract features revealing institutional flow behavior"""
        try:
            if df is None or len(df) < 10:
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            open_price = df['open']
            
            # 1. VOLUME SIGNATURE (Institutional size)
            vol_avg_10 = volume.rolling(10).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            volume_signature = current_vol / (vol_avg_10 if vol_avg_10 > 0 else 1)
            
            # 2. ABSORPTION RATIO (Institutions absorbing)
            current_body = abs(close.iloc[-1] - open_price.iloc[-1])
            lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
            absorption_ratio = lower_wick / (current_body if current_body > 0 else 1)
            
            # 3. DISTRIBUTION RATIO (Institutions distributing)
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

# Initialize Flow AI
print("🚀 Initializing Institutional Flow AI...")
flow_ai = InstitutionalFlowAI()
print("✅ Flow AI initialized!")

# =================== TECHNICAL INDICATORS ===================
def add_technical_indicators(df):
    """Add technical indicators"""
    if df.empty:
        return df
    
    # EMA
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df

# =================== INSTITUTIONAL BEHAVIOR DETECTION =======
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
    """Detect large accumulation behavior"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 10:
            return None
        
        # 1. INSTITUTIONAL VOLUME SIGNATURE
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_10 * INSTITUTIONAL_VOLUME_RATIO:
            return None
        
        # 2. ABSORPTION CANDLE
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if lower_wick < current_body * ABSORPTION_WICK_RATIO:
            return None
        
        # 3. PRICE HOLDING AT SUPPORT
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
        
        return "LONG"
        
    except Exception as e:
        print(f"Error in accumulation detection: {e}")
        return None

def detect_large_distribution(df):
    """Detect large distribution behavior"""
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
        if hour >= 22:
            return None
        
        # 1. INSTITUTIONAL VOLUME SIGNATURE
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_10 * (INSTITUTIONAL_VOLUME_RATIO + 0.5):
            return None
        
        # 2. DISTRIBUTION CANDLE
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if upper_wick < current_body * ABSORPTION_WICK_RATIO:
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
        
        return "SHORT"
        
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
        
        recent_high = high.iloc[-12:-2].max()
        recent_low = low.iloc[-12:-2].min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        vol_avg = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # BULL STOP HUNT (then LONG)
        if (current_low < recent_low * (1 - STOP_HUNT_DISTANCE) and
            current_close > recent_low * 1.008 and
            current_vol > vol_avg * 3.5 and
            current_close > prev_close and
            (current_high - current_close) < (current_close - current_low) * 0.4):
            
            return "LONG"
        
        # BEAR STOP HUNT (then SHORT)
        if (current_high > recent_high * (1 + STOP_HUNT_DISTANCE) and
            current_close < recent_high * 0.992 and
            current_vol > vol_avg * 4.0 and
            current_close < prev_close and
            (current_close - current_low) < (current_high - current_close) * 0.4):
            
            return "SHORT"
        
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
        
        high_zone, low_zone = detect_liquidity_zones(df, lookback=18)
        current_price = close.iloc[-1]
        
        if high_zone is None or low_zone is None:
            return None
        
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # LIQUIDITY TAKE AT HIGHS (then SHORT)
        if high_zone and abs(current_price - high_zone) <= high_zone * 0.006:
            if (current_vol > vol_avg_15 * 4.0 and
                high.iloc[-1] > high_zone * 1.008 and
                close.iloc[-1] < high_zone * 0.997 and
                volume.iloc[-1] > volume.iloc[-2] * 2.0):
                
                return "SHORT"
        
        # LIQUIDITY TAKE AT LOWS (then LONG)
        if low_zone and abs(current_price - low_zone) <= low_zone * 0.006:
            if (current_vol > vol_avg_15 * 4.0 and
                low.iloc[-1] < low_zone * 0.992 and
                close.iloc[-1] > low_zone * 1.003 and
                volume.iloc[-1] > volume.iloc[-2] * 2.0):
                
                return "LONG"
        
    except Exception as e:
        print(f"Error in liquidity take detection: {e}")
        return None
    return None

# =================== TECHNICAL TRADING FUNCTIONS ============
def check_technical_conditions(df, mode_cfg, side):
    """Check trading conditions for technical strategy"""
    try:
        price = df["close"].iloc[-1]
        ema = df["ema"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        
        if side == "LONG":
            return (price > ema) and (mode_cfg["rsi_long_min"] <= rsi <= mode_cfg["rsi_long_max"])
        else:
            return (price < ema) and (mode_cfg["rsi_short_min"] <= rsi <= mode_cfg["rsi_short_max"])
    except:
        return False

def compute_technical_entry(df, side, mode_cfg, symbol):
    """Compute entry price with filters"""
    if df.empty:
        return None
    
    price = df["close"].iloc[-1]
    w = mode_cfg["recent_hl_window"]
    recent_high = df["high"].iloc[-w:].max()
    recent_low  = df["low"].iloc[-w:].min()

    if side == "LONG":
        raw = recent_high * (1 + mode_cfg["entry_buffer_long"])
        predicted = max(raw, price)
        drift = abs(predicted - price) / price
        if drift > mode_cfg["max_entry_drift"]:
            predicted = price * (1 + mode_cfg["max_entry_drift"])
    else:
        raw = recent_low * (1 - mode_cfg["entry_buffer_short"])
        predicted = min(raw, price)
        drift = abs(predicted - price) / price
        if drift > mode_cfg["max_entry_drift"]:
            predicted = price * (1 - mode_cfg["max_entry_drift"])

    if mode_cfg["immediate_mode"] and abs(predicted - price) / price <= mode_cfg["immediate_tol"]:
        predicted = price

    if mode_cfg["volume_filter"]:
        lb = mode_cfg["volume_lookback"]
        vol_avg = df["volume"].iloc[-lb:].mean()
        last_vol = df["volume"].iloc[-1]
        if last_vol < 0.8 * vol_avg:
            return None

    abs_cap = ABS_MAX_ENTRY_USD.get(symbol, None)
    if abs_cap is not None and abs(predicted - price) > abs_cap:
        if predicted > price:
            predicted = price + abs_cap
        else:
            predicted = price - abs_cap

    return predicted

def calculate_trade_levels(entry_price, side, move_percentage=0.02):
    """Calculate stop loss and target levels"""
    if side == "LONG":
        sl = entry_price * (1 - SL_BUFFER)
        # Dynamic targets based on volatility
        tps = [
            entry_price * (1 + move_percentage * 0.8),
            entry_price * (1 + move_percentage * 1.5),
            entry_price * (1 + move_percentage * 2.2),
            entry_price * (1 + move_percentage * 3.0)
        ]
    else:
        sl = entry_price * (1 + SL_BUFFER)
        tps = [
            entry_price * (1 - move_percentage * 0.8),
            entry_price * (1 - move_percentage * 1.5),
            entry_price * (1 - move_percentage * 2.2),
            entry_price * (1 - move_percentage * 3.0)
        ]
    
    # Ensure targets are in correct order
    if side == "LONG":
        tps.sort()
    else:
        tps.sort(reverse=True)
    
    return sl, tps

# =================== SIGNAL MANAGEMENT ======================
def can_send_signal(symbol, signal_type="flow"):
    """Check if signal can be sent"""
    current_time = time.time()
    
    if signal_type == "flow":
        signal_dict = last_signal_time
        cooldown = 1800  # 30 minutes for flow signals
    else:
        signal_dict = last_signal_time
        cooldown = 900  # 15 minutes for technical signals
    
    if symbol in signal_dict:
        time_since_last = current_time - signal_dict[symbol]
        if time_since_last < cooldown:
            print(f"⏳ {symbol} in cooldown: {int(cooldown - time_since_last)}s remaining")
            return False
    
    return True

def update_signal_time(symbol):
    """Update last signal time"""
    last_signal_time[symbol] = time.time()

# =================== MONITORING SYSTEM ======================
def monitor_trade_live(symbol, side, entry, sl, targets, strategy_name, signal_id):
    """Continuous monitoring thread for a trade"""
    
    def monitoring_thread():
        print(f"🔍 Starting monitoring for {symbol}")
        
        max_price = entry
        min_price = entry
        entry_triggered = False
        targets_hit = [False] * len(targets)
        last_update_time = time.time()
        
        while True:
            current_time = time.time()
            
            # Check if monitoring should stop (30 minutes of no entry)
            if not entry_triggered and (current_time - last_update_time) > 1800:
                print(f"⏰ Monitoring stopped for {symbol} - no entry triggered")
                if signal_id in active_monitoring_threads:
                    del active_monitoring_threads[signal_id]
                break
            
            # Get current price
            price = get_current_price(symbol)
            if price is None:
                time.sleep(10)
                continue
            
            last_update_time = current_time
            
            # Update min/max
            if price > max_price:
                max_price = price
            if price < min_price:
                min_price = price
            
            # Check entry
            if not entry_triggered:
                if (side == "LONG" and price >= entry) or (side == "SHORT" and price <= entry):
                    entry_triggered = True
                    send_telegram(f"✅ ENTRY TRIGGERED: {symbol} {side} @ ${price:.2f}")
            
            if entry_triggered:
                # Check targets
                for i, target in enumerate(targets):
                    if not targets_hit[i]:
                        if (side == "LONG" and price >= target) or (side == "SHORT" and price <= target):
                            targets_hit[i] = True
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit @ ${target:.2f}")
                
                # Check stop loss
                if (side == "LONG" and price <= sl) or (side == "SHORT" and price >= sl):
                    send_telegram(f"🛑 STOP LOSS HIT: {symbol} @ ${price:.2f}")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
                
                # Check if all targets hit
                if all(targets_hit):
                    send_telegram(f"🏆 {symbol}: ALL TARGETS HIT!")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
            
            # Price milestone updates
            if entry_triggered:
                milestone_distance = max_price * 0.01  # 1% milestone
                if abs(price - max_price) < milestone_distance and price > entry * 1.02:
                    send_telegram(f"📈 {symbol} progressing well: ${price:.2f}")
            
            time.sleep(10)
    
    # Start monitoring thread
    thread = threading.Thread(target=monitoring_thread, daemon=True)
    thread.start()
    active_monitoring_threads[signal_id] = thread

# =================== ALERT FUNCTIONS ========================
def send_flow_alert(symbol, side, entry, sl, targets, behavior_type):
    """Send institutional flow alert"""
    global signal_counter
    
    signal_id = f"FLOW{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    if behavior_type == "large_accumulation":
        message = (f"🏛️ <b>INSTITUTIONAL ACCUMULATION DETECTED</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n\n"
                  f"⚠️ Large institutions accumulating before UP move")
    
    elif behavior_type == "large_distribution":
        message = (f"🏛️ <b>INSTITUTIONAL DISTRIBUTION DETECTED</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n\n"
                  f"⚠️ Large institutions distributing before DOWN move")
    
    elif behavior_type == "stop_reversal":
        message = (f"🎯 <b>STOP HUNT REVERSAL DETECTED</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n\n"
                  f"⚠️ Stop hunt complete - reversal imminent")
    
    else:  # liquidity_take
        message = (f"🌊 <b>LIQUIDITY GRAB DETECTED</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n\n"
                  f"⚠️ Institutions grabbed liquidity - big move following")
    
    send_telegram(message)
    update_signal_time(symbol)
    
    # Store signal data
    signal_data = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "behavior_type": behavior_type,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_data
    
    # Start monitoring thread
    monitor_trade_live(symbol, side, entry, sl, targets, 
                      BEHAVIOR_TYPES[behavior_type], signal_id)
    
    return signal_id

def send_technical_alert(symbol, side, entry, sl, targets, strategy):
    """Send technical trading alert"""
    global signal_counter
    
    signal_id = f"TECH{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    message = (f"📊 <b>{strategy} SIGNAL</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Entry Above:</b> ${entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Strategy:</b> {strategy}")
    
    send_telegram(message)
    update_signal_time(symbol)
    
    # Store signal data
    signal_data = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "strategy": strategy,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_data
    
    # Start monitoring thread
    monitor_trade_live(symbol, side, entry, sl, targets, strategy, signal_id)
    
    return signal_id

# =================== ANALYSIS FUNCTIONS =====================
def analyze_institutional_flow(symbol):
    """Analyze for institutional flow behavior"""
    df_5min = get_market_data(symbol, "5m", 100)
    
    if df_5min is None or len(df_5min) < 20:
        return None
    
    print(f"🔍 Analyzing {symbol} for institutional flow...")
    
    # Check all institutional behaviors
    behaviors = [
        ("large_accumulation", detect_large_accumulation(df_5min)),
        ("stop_reversal", detect_stop_reversal(df_5min)),
        ("liquidity_take", detect_liquidity_take(df_5min)),
        ("large_distribution", detect_large_distribution(df_5min))
    ]
    
    for behavior_type, direction in behaviors:
        if direction:
            print(f"✅ {symbol}: {BEHAVIOR_TYPES[behavior_type]} - {direction}")
            
            # Get current price for entry
            current_price = get_current_price(symbol)
            if current_price is None:
                return None
            
            # Calculate levels based on volatility
            movement_range = (df_5min['high'].iloc[-1] - df_5min['low'].iloc[-1]) / df_5min['close'].iloc[-1]
            move_percentage = max(movement_range, MIN_MOVE_FOR_ENTRY)
            
            entry = current_price
            sl, targets = calculate_trade_levels(entry, direction, move_percentage)
            
            return {
                "symbol": symbol,
                "side": direction,
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "behavior_type": behavior_type
            }
    
    return None

def analyze_technical_strategy(symbol, mode_name):
    """Analyze for technical trading signals"""
    cfg = TRADING_MODES[mode_name]
    df = get_market_data(symbol, cfg["interval"], CANDLE_LIMIT)
    
    if df is None or df.empty:
        return None
    
    df = add_technical_indicators(df)
    
    # Check both LONG and SHORT
    for side in ["LONG", "SHORT"]:
        if check_technical_conditions(df, cfg, side):
            entry = compute_technical_entry(df, side, cfg, symbol)
            if entry is None:
                continue
            
            # Calculate levels
            move_percentage = 0.02  # Default 2% move
            sl, targets = calculate_trade_levels(entry, side, move_percentage)
            
            return {
                "symbol": symbol,
                "side": side,
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy": mode_name
            }
    
    return None

# =================== SCANNER FUNCTIONS ======================
def run_institutional_scanner():
    """Scan for institutional flow signals"""
    print("🔍 Scanning for institutional flow...")
    
    signals_found = 0
    threads = []
    results = []
    
    def scan_symbol(symbol):
        result = analyze_institutional_flow(symbol)
        if result:
            results.append(result)
    
    # Scan top symbols
    for symbol in list(DIGITAL_ASSETS.values())[:3]:  # BTC, ETH, BNB
        t = threading.Thread(target=scan_symbol, args=(symbol,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    # Send alerts for found signals
    for result in results:
        if can_send_signal(result["symbol"], "flow"):
            send_flow_alert(
                result["symbol"],
                result["side"],
                result["entry"],
                result["sl"],
                result["targets"],
                result["behavior_type"]
            )
            signals_found += 1
    
    print(f"✅ Institutional scan complete. Signals: {signals_found}")
    return signals_found

def run_technical_scanner():
    """Scan for technical trading signals"""
    print("📊 Scanning for technical signals...")
    
    signals_found = 0
    
    for symbol in TRADING_SYMBOLS:
        for strategy in ["SCALP", "SWING"]:
            result = analyze_technical_strategy(symbol, strategy)
            if result and can_send_signal(symbol, "technical"):
                send_technical_alert(
                    result["symbol"],
                    result["side"],
                    result["entry"],
                    result["sl"],
                    result["targets"],
                    result["strategy"]
                )
                signals_found += 1
    
    print(f"✅ Technical scan complete. Signals: {signals_found}")
    return signals_found

# =================== STATUS MONITORING ======================
def check_active_signals():
    """Check status of active signals"""
    current_time = time.time()
    
    # Clean up old completed signals (older than 1 hour)
    completed_signal_ids = []
    for signal_id, signal_data in active_signals.items():
        if signal_data.get("status") == "COMPLETED":
            if current_time - signal_data["timestamp"] > 3600:
                completed_signal_ids.append(signal_id)
    
    for signal_id in completed_signal_ids:
        del active_signals[signal_id]
    
    # Send status update if there are active signals
    if active_signals:
        active_count = len(active_signals)
        print(f"📊 Active signals: {active_count}")
        
        # Send summary every 30 minutes
        if int(time.time()) % 1800 < 60:  # Every 30 minutes
            status_msg = f"📊 <b>ACTIVE SIGNALS STATUS</b>\n\n"
            for signal_id, signal_data in list(active_signals.items())[:5]:  # Show first 5
                status_msg += f"• {signal_data['symbol']} {signal_data['side']} - {signal_id}\n"
            
            if len(active_signals) > 5:
                status_msg += f"\n... and {len(active_signals) - 5} more signals"
            
            send_telegram(status_msg)
    
    return len(active_signals)

# =================== MAIN EXECUTION =========================
def main():
    """Main execution loop"""
    print("=" * 60)
    print("🏛️ INSTITUTIONAL FLOW & TRADING BOT")
    print("📊 Continuous Monitoring System")
    print("🎯 Signals tracked until completion")
    print("=" * 60)
    
    send_telegram("🤖 <b>CRYPTO TRADING BOT ACTIVATED</b>\n"
                  "🏛️ Institutional Flow Detection\n"
                  "📊 Technical Strategies\n"
                  "🔍 Continuous Signal Monitoring\n"
                  f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    iteration = 0
    flow_scan_counter = 0
    
    while True:
        iteration += 1
        try:
            print(f"\n🔄 Iteration {iteration} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            
            # Check active signals
            active_count = check_active_signals()
            print(f"📈 Active signals: {active_count}")
            
            # Run institutional scanner every 2 iterations
            if flow_scan_counter % 2 == 0:
                flow_signals = run_institutional_scanner()
                print(f"🏛️ Institutional signals found: {flow_signals}")
            
            # Run technical scanner
            tech_signals = run_technical_scanner()
            print(f"📊 Technical signals found: {tech_signals}")
            
            flow_scan_counter += 1
            
            # Wait for next scan (1 minute)
            print("⏳ Waiting 60 seconds for next scan...")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down bot...")
            
            # Send final status
            if active_signals:
                send_telegram(f"🛑 <b>BOT SHUTTING DOWN</b>\n"
                             f"Active signals: {len(active_signals)}\n"
                             "Monitor remaining signals manually")
            else:
                send_telegram("🛑 <b>BOT SHUTTING DOWN</b>\n"
                             "No active signals")
            
            break
            
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            send_telegram(f"⚠️ <b>BOT ERROR</b>\n"
                         f"Error: {str(e)[:100]}")
            time.sleep(60)

if __name__ == "__main__":
    # Test credentials first
    print("🔧 Testing bot configuration...")
    print(f"API Key configured: {'Yes' if SYSTEM_KEY and SYSTEM_KEY.strip() and SYSTEM_KEY != ' ' else 'No'}")
    print(f"API Secret configured: {'Yes' if SYSTEM_SECRET and SYSTEM_SECRET.strip() and SYSTEM_SECRET != ' ' else 'No'}")
    print(f"Telegram Token configured: {'Yes' if ALERT_TOKEN and ALERT_TOKEN.strip() and ALERT_TOKEN != ' ' else 'No'}")
    print(f"Telegram Chat ID configured: {'Yes' if ALERT_TARGET and ALERT_TARGET.strip() and ALERT_TARGET != ' ' else 'No'}")
    
    # Start main loop
    main()
