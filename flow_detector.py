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
INSTITUTIONAL_VOLUME_RATIO = 4.2  # Increased for better filtering
MIN_MOVE_FOR_ENTRY = 0.025  # Increased minimum move threshold
STOP_HUNT_DISTANCE = 0.012  # Increased stop hunt sensitivity
ABSORPTION_WICK_RATIO = 0.22  # Tighter wick ratio for institutional candles

# =================== INSTITUTIONAL TIME ZONES ===============
INSTITUTIONAL_ACTIVE_HOURS = {
    "LONDON_OPEN": (7, 12),    # 7AM - 12PM UTC
    "NY_OPEN": (13, 17),       # 1PM - 5PM UTC
    "ASIA_LATE": (22, 24),     # 10PM - 12AM UTC
    "ASIA_EARLY": (0, 4)       # 12AM - 4AM UTC
}

# =================== INSTITUTIONAL ORDER FLOW ===============
ORDER_FLOW_THRESHOLDS = {
    "BLOCK_TRADE_SIZE": 5.0,    # Minimum $5M equivalent for block trade
    "SWEEP_RATIO": 0.65,        # Order book sweep percentage
    "IMMEDIATE_OR_CANCEL": 0.8, # IOC orders ratio
    "DARK_POOL_INDICATOR": 3.5  # Volume without price movement
}

# =================== TECHNICAL TRADING SETTINGS ============
EMA_PERIOD = 20
RSI_PERIOD = 14
CANDLE_LIMIT = 150  # Increased for better institutional pattern recognition

SL_BUFFER = 0.0018  # Tighter stop loss for institutional moves
TARGETS = [0.0030, 0.005, 0.009]  # Adjusted for institutional profit taking

ABS_MAX_ENTRY_USD = {
    "BTC-USDT": 120.0,  # Increased for institutional move ranges
    "ETH-USDT": 12.0
}

TRADING_MODES = {
    "SCALP": {
        "interval": "1m",
        "recent_hl_window": 3,
        "rsi_long_min": 50, "rsi_long_max": 75,  # Tighter ranges
        "rsi_short_min": 25, "rsi_short_max": 50,
        "entry_buffer_long": 0.0003,  # Tighter entry buffers
        "entry_buffer_short": 0.0003,
        "max_entry_drift": 0.0008,
        "immediate_mode": True,
        "immediate_tol": 0.0008,
        "need_prev_candle_break": True,  # Added institutional confirmation
        "volume_filter": True,
        "volume_lookback": 25,
        "institutional_only": True  # Filter only institutional patterns
    },
    "SWING": {
        "interval": "5m",
        "recent_hl_window": 12,
        "rsi_long_min": 48, "rsi_long_max": 78,
        "rsi_short_min": 22, "rsi_short_max": 52,
        "entry_buffer_long": 0.0012,
        "entry_buffer_short": 0.0012,
        "max_entry_drift": 0.0035,
        "immediate_mode": True,
        "immediate_tol": 0.0012,
        "need_prev_candle_break": True,
        "volume_filter": True,
        "volume_lookback": 35,
        "institutional_only": True
    }
}

# =================== INSTITUTIONAL BEHAVIOR TYPES ===========
BEHAVIOR_TYPES = {
    "large_accumulation": "🏛️ LARGE ACCUMULATION",
    "large_distribution": "🏛️ LARGE DISTRIBUTION", 
    "stop_reversal": "🎯 STOP HUNT REVERSAL",
    "liquidity_take": "🌊 LIQUIDITY GRAB",
    "block_trade_buy": "📊 BLOCK TRADE BUY",
    "block_trade_sell": "📊 BLOCK TRADE SELL",
    "dark_pool_accumulation": "🌙 DARK POOL ACCUMULATION",
    "order_book_sweep": "⚡ ORDER BOOK SWEEP"
}

# =================== INSTITUTIONAL PATTERN DATABASE =========
INSTITUTIONAL_PATTERNS = {
    "ACCUMULATION_SIGNATURES": [
        # Pattern: High volume absorption at support
        {"volume_ratio": 4.0, "wick_ratio": 0.25, "close_position": 0.75, "time_frame": "5m"},
        # Pattern: Stealth accumulation through multiple candles
        {"volume_ratio": 3.5, "wick_ratio": 0.15, "close_position": 0.65, "time_frame": "15m"},
        # Pattern: V-shaped recovery with institutional volume
        {"volume_ratio": 5.0, "wick_ratio": 0.30, "close_position": 0.85, "time_frame": "1h"}
    ],
    "DISTRIBUTION_SIGNATURES": [
        # Pattern: Churning at highs with increasing volume
        {"volume_ratio": 4.2, "wick_ratio": 0.28, "close_position": 0.25, "time_frame": "5m"},
        # Pattern: Failed breakout with massive volume
        {"volume_ratio": 5.5, "wick_ratio": 0.35, "close_position": 0.35, "time_frame": "15m"},
        # Pattern: Exhaustion gap fill with institutional selling
        {"volume_ratio": 4.8, "wick_ratio": 0.20, "close_position": 0.20, "time_frame": "1h"}
    ]
}

# =================== GLOBAL TRACKING ========================
signal_counter = 0
active_signals = {}
last_signal_time = {}
active_monitoring_threads = {}
completed_signals = []

# Institutional flow memory
institutional_flow_memory = {
    "BTC-USDT": {"accumulation_zones": [], "distribution_zones": [], "block_trades": []},
    "ETH-USDT": {"accumulation_zones": [], "distribution_zones": [], "block_trades": []}
}

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

def get_order_book(symbol, limit=50):  # Increased limit for better institutional analysis
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

def get_recent_trades(symbol, limit=100):
    """Get recent trades for order flow analysis"""
    try:
        endpoint = "/openApi/swap/v2/quote/trades"
        params = f"symbol={symbol}&limit={limit}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                return data['data']
    except Exception as e:
        print(f"Error fetching trades for {symbol}: {e}")
    return None

# =================== INSTITUTIONAL FLOW AI ==================
class InstitutionalFlowAI:
    def __init__(self):
        self.accumulation_model = None
        self.distribution_model = None
        self.scaler = None
        self.pattern_database = INSTITUTIONAL_PATTERNS
        self.load_models()
    
    def load_models(self):
        """Load AI models trained on institutional behavior"""
        try:
            if os.path.exists("flow_accumulation_model.pkl"):
                self.accumulation_model = joblib.load("flow_accumulation_model.pkl")
                print("✅ Loaded institutional accumulation model")
            else:
                self.accumulation_model = None
                
            if os.path.exists("flow_distribution_model.pkl"):
                self.distribution_model = joblib.load("flow_distribution_model.pkl")
                print("✅ Loaded institutional distribution model")
            else:
                self.distribution_model = None
                
            if os.path.exists("flow_scaler.pkl"):
                self.scaler = joblib.load("flow_scaler.pkl")
                print("✅ Loaded institutional scaler")
            else:
                self.scaler = None
            
            if not all([self.accumulation_model, self.distribution_model, self.scaler]):
                print("🏛️ Training institutional-grade AI models...")
                self.train_models()
            else:
                print("✅ All institutional AI models loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.train_models()
    
    def train_models(self):
        """Train AI on sophisticated institutional behavior patterns"""
        try:
            print("🏛️ Training Advanced Institutional Flow AI models...")
            
            # INSTITUTIONAL ACCUMULATION PATTERNS (UP moves)
            X_acc = []
            y_acc = []
            
            # Institutional Accumulation Pattern 1: Stealth accumulation
            X_acc.append([4.5, 0.25, 0.85, 3.2, 0.22, 0.75, 0.68, 0.028, 1.5, 0.85])
            y_acc.append(1)
            
            # Institutional Accumulation Pattern 2: V-shaped recovery
            X_acc.append([5.2, 0.30, 0.88, 3.8, 0.25, 0.78, 0.72, 0.032, 1.8, 0.82])
            y_acc.append(1)
            
            # Institutional Accumulation Pattern 3: Support retest accumulation
            X_acc.append([4.8, 0.22, 0.82, 3.0, 0.20, 0.72, 0.65, 0.026, 1.4, 0.88])
            y_acc.append(1)
            
            # Institutional Accumulation Pattern 4: Breakout accumulation
            X_acc.append([6.0, 0.35, 0.90, 4.2, 0.28, 0.82, 0.75, 0.035, 2.0, 0.90])
            y_acc.append(1)
            
            # Non-institutional patterns (noise)
            X_acc.append([1.5, 0.08, 0.35, 1.0, 0.05, 0.25, 0.45, 0.008, 0.6, 0.45])
            y_acc.append(0)
            
            X_acc.append([2.2, 0.12, 0.42, 1.8, 0.10, 0.35, 0.50, 0.012, 0.8, 0.52])
            y_acc.append(0)
            
            # INSTITUTIONAL DISTRIBUTION PATTERNS (DOWN moves)
            X_dist = []
            y_dist = []
            
            # Institutional Distribution Pattern 1: Churning at highs
            X_dist.append([4.8, 0.28, 0.12, 3.5, 0.25, 0.75, 0.28, 0.030, 1.6, 0.22])
            y_dist.append(1)
            
            # Institutional Distribution Pattern 2: Failed breakout distribution
            X_dist.append([5.5, 0.35, 0.10, 4.0, 0.30, 0.80, 0.22, 0.035, 1.9, 0.18])
            y_dist.append(1)
            
            # Institutional Distribution Pattern 3: Exhaustion gap distribution
            X_dist.append([4.2, 0.20, 0.15, 3.2, 0.22, 0.70, 0.30, 0.025, 1.5, 0.25])
            y_dist.append(1)
            
            # Institutional Distribution Pattern 4: Resistance distribution
            X_dist.append([6.2, 0.40, 0.08, 4.5, 0.35, 0.85, 0.18, 0.040, 2.2, 0.15])
            y_dist.append(1)
            
            # Non-institutional patterns (noise)
            X_dist.append([1.8, 0.10, 0.55, 1.2, 0.08, 0.40, 0.52, 0.010, 0.7, 0.65])
            y_dist.append(0)
            
            X_dist.append([2.5, 0.15, 0.48, 2.0, 0.12, 0.45, 0.55, 0.015, 0.9, 0.58])
            y_dist.append(0)
            
            X_acc = np.array(X_acc)
            y_acc = np.array(y_acc)
            X_dist = np.array(X_dist)
            y_dist = np.array(y_dist)
            
            self.scaler = StandardScaler()
            X_acc_scaled = self.scaler.fit_transform(X_acc)
            
            self.accumulation_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                subsample=0.8
            )
            self.accumulation_model.fit(X_acc_scaled, y_acc)
            
            X_dist_scaled = self.scaler.transform(X_dist)
            
            self.distribution_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced',
                bootstrap=True
            )
            self.distribution_model.fit(X_dist_scaled, y_dist)
            
            joblib.dump(self.accumulation_model, "flow_accumulation_model.pkl")
            joblib.dump(self.distribution_model, "flow_distribution_model.pkl")
            joblib.dump(self.scaler, "flow_scaler.pkl")
            
            print("✅ Institutional-grade AI models trained and saved")
            
        except Exception as e:
            print(f"❌ Error training institutional models: {e}")
            self.accumulation_model = None
            self.distribution_model = None
            self.scaler = None
    
    def analyze_order_book_imbalance(self, symbol):
        """Advanced order book analysis for institutional positioning"""
        try:
            order_book = get_order_book(symbol, limit=30)
            if not order_book:
                return 0.0
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            # Institutional depth analysis (first 10 levels)
            institutional_bid_depth = sum(float(bid[1]) for bid in bids[:10] if float(bid[1]) > 10)
            institutional_ask_depth = sum(float(ask[1]) for ask in asks[:10] if float(ask[1]) > 10)
            
            if institutional_bid_depth + institutional_ask_depth == 0:
                return 0.0
            
            imbalance = (institutional_bid_depth - institutional_ask_depth) / (institutional_bid_depth + institutional_ask_depth)
            
            # Detect block orders (large chunks at specific prices)
            block_orders = []
            for price, size in bids[:5]:
                if float(size) > ORDER_FLOW_THRESHOLDS["BLOCK_TRADE_SIZE"]:
                    block_orders.append(("BID", float(price), float(size)))
            
            for price, size in asks[:5]:
                if float(size) > ORDER_FLOW_THRESHOLDS["BLOCK_TRADE_SIZE"]:
                    block_orders.append(("ASK", float(price), float(size)))
            
            if block_orders:
                print(f"🏛️ Block orders detected at {symbol}: {block_orders}")
            
            return imbalance
            
        except Exception as e:
            print(f"Error in order book analysis: {e}")
            return 0.0
    
    def analyze_trade_flow(self, symbol):
        """Analyze trade flow for institutional activity"""
        try:
            trades = get_recent_trades(symbol, limit=200)
            if not trades:
                return {"buy_pressure": 0.5, "block_trades": 0}
            
            buy_volume = 0
            sell_volume = 0
            block_trades = 0
            
            for trade in trades:
                if trade.get('isBuyerMaker'):
                    sell_volume += float(trade.get('qty', 0))
                else:
                    buy_volume += float(trade.get('qty', 0))
                
                # Detect block trades
                if float(trade.get('qty', 0)) > ORDER_FLOW_THRESHOLDS["BLOCK_TRADE_SIZE"]:
                    block_trades += 1
            
            total_volume = buy_volume + sell_volume
            buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5
            
            return {
                "buy_pressure": buy_pressure,
                "block_trades": block_trades,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume
            }
            
        except Exception as e:
            print(f"Error in trade flow analysis: {e}")
            return {"buy_pressure": 0.5, "block_trades": 0}
    
    def extract_flow_features(self, df, symbol):
        """Extract sophisticated institutional flow features"""
        try:
            if df is None or len(df) < 20:
                return None
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            open_price = df['open']
            
            # 1. INSTITUTIONAL VOLUME SIGNATURE
            vol_avg_15 = volume.rolling(15).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            volume_signature = current_vol / (vol_avg_15 if vol_avg_15 > 0 else 1)
            
            # 2. SMART MONEY ABSORPTION RATIO
            current_body = abs(close.iloc[-1] - open_price.iloc[-1])
            lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
            
            if current_body > 0:
                absorption_ratio = lower_wick / current_body
                distribution_ratio = upper_wick / current_body
            else:
                absorption_ratio = 0.5
                distribution_ratio = 0.5
            
            # 3. INSTITUTIONAL PRICE STRENGTH
            recent_low = low.iloc[-15:-5].min()
            recent_high = high.iloc[-15:-5].max()
            current_price = close.iloc[-1]
            
            if current_price > recent_high * 0.92:
                price_strength = (current_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.6
            elif current_price < recent_low * 1.08:
                price_strength = 0.3
            else:
                price_strength = 0.5
            
            # 4. INSTITUTIONAL VOLUME CONFIRMATION
            vol_prev_5 = volume.iloc[-6:-1].mean()
            vol_confirmation = current_vol / (vol_prev_5 if vol_prev_5 > 0 else 1)
            
            # 5. INSTITUTIONAL MOMENTUM QUALITY
            price_change_5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if close.iloc[-6] > 0 else 0
            momentum_quality = price_change_5 / (volume_signature if volume_signature > 0 else 1)
            
            # 6. ORDER BOOK IMBALANCE (Institutional depth)
            book_imbalance = self.analyze_order_book_imbalance(symbol)
            
            # 7. INSTITUTIONAL TREND ALIGNMENT
            ma_20 = close.rolling(20).mean().iloc[-1]
            ma_50 = close.rolling(50).mean().iloc[-1]
            
            if current_price > ma_20 > ma_50:
                trend_alignment = 0.9  # Strong uptrend
            elif current_price < ma_20 < ma_50:
                trend_alignment = 0.1  # Strong downtrend
            elif current_price > ma_20 and ma_20 > ma_50:
                trend_alignment = 0.7  # Uptrend
            elif current_price < ma_20 and ma_20 < ma_50:
                trend_alignment = 0.3  # Downtrend
            else:
                trend_alignment = 0.5  # Neutral
            
            # 8. INSTITUTIONAL PRESSURE INDEX
            buying_pressure = sum([1 for i in range(-5, 0) if close.iloc[i] > open_price.iloc[i] and volume.iloc[i] > vol_avg_15])
            selling_pressure = sum([1 for i in range(-5, 0) if close.iloc[i] < open_price.iloc[i] and volume.iloc[i] > vol_avg_15])
            pressure_index = buying_pressure / (buying_pressure + selling_pressure + 1)
            
            # 9. INSTITUTIONAL TIME EFFICIENCY
            utc_now = datetime.utcnow()
            hour = utc_now.hour
            
            # Check if current time is during institutional activity hours
            institutional_active = False
            for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
                if start_hour <= hour < end_hour:
                    institutional_active = True
                    break
            
            time_efficiency = 0.85 if institutional_active else 0.35
            
            # 10. TRADE FLOW ANALYSIS
            trade_flow = self.analyze_trade_flow(symbol)
            flow_strength = trade_flow["buy_pressure"]
            
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
                time_efficiency,
                flow_strength
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting institutional flow features: {e}")
            return None
    
    def detect_accumulation(self, df, symbol):
        """Detect institutional accumulation behavior"""
        if self.accumulation_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_flow_features(df, symbol)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.accumulation_model.predict(features_scaled)[0]
            probability = self.accumulation_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            
            # Additional institutional filter
            trade_flow = self.analyze_trade_flow(symbol)
            if trade_flow["block_trades"] > 0:
                confidence *= 1.2  # Boost confidence if block trades detected
            
            return bool(prediction), min(confidence, 1.0)
        except Exception as e:
            print(f"⚠️ Error in institutional accumulation detection: {e}")
            return False, 0.0
    
    def detect_distribution(self, df, symbol):
        """Detect institutional distribution behavior"""
        if self.distribution_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_flow_features(df, symbol)
        if features is None:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.distribution_model.predict(features_scaled)[0]
            probability = self.distribution_model.predict_proba(features_scaled)[0]
            
            confidence = probability[1] if len(probability) > 1 else probability[0]
            
            # Additional institutional filter
            order_book_imbalance = self.analyze_order_book_imbalance(symbol)
            if order_book_imbalance < -0.3:  # Strong selling pressure
                confidence *= 1.15
            
            return bool(prediction), min(confidence, 1.0)
        except Exception as e:
            print(f"⚠️ Error in institutional distribution detection: {e}")
            return False, 0.0
    
    def match_institutional_pattern(self, df, pattern_type="accumulation"):
        """Match current price action against institutional patterns"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            open_price = df['open']
            
            current_candle = {
                "body": abs(close.iloc[-1] - open_price.iloc[-1]),
                "lower_wick": min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1],
                "upper_wick": high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1]),
                "close_position": (close.iloc[-1] - low.iloc[-1]) / (high.iloc[-1] - low.iloc[-1]) if (high.iloc[-1] - low.iloc[-1]) > 0 else 0.5,
                "volume_ratio": volume.iloc[-1] / volume.rolling(10).mean().iloc[-1] if volume.rolling(10).mean().iloc[-1] > 0 else 1
            }
            
            patterns = self.pattern_database.get(f"{pattern_type.upper()}_SIGNATURES", [])
            
            for pattern in patterns:
                match_score = 0
                
                if current_candle["volume_ratio"] >= pattern["volume_ratio"] * 0.8:
                    match_score += 1
                
                if pattern_type == "accumulation":
                    wick_ratio = current_candle["lower_wick"] / current_candle["body"] if current_candle["body"] > 0 else 0
                    if wick_ratio >= pattern["wick_ratio"] * 0.7:
                        match_score += 1
                else:
                    wick_ratio = current_candle["upper_wick"] / current_candle["body"] if current_candle["body"] > 0 else 0
                    if wick_ratio >= pattern["wick_ratio"] * 0.7:
                        match_score += 1
                
                if abs(current_candle["close_position"] - pattern["close_position"]) < 0.15:
                    match_score += 1
                
                if match_score >= 2:
                    return True, pattern
            
            return False, None
            
        except Exception as e:
            print(f"Error in pattern matching: {e}")
            return False, None

# Initialize Institutional Flow AI
print("🚀 Initializing Advanced Institutional Flow AI...")
flow_ai = InstitutionalFlowAI()
print("✅ Institutional Flow AI initialized with advanced patterns!")

# =================== TECHNICAL INDICATORS ===================
def add_technical_indicators(df):
    """Add technical indicators"""
    if df.empty:
        return df
    
    # EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Volume Weighted Average Price (VWAP) - Institutional favorite
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['typical_price_volume'] = df['typical_price'] * df['volume']
    cumulative_tpv = df['typical_price_volume'].cumsum()
    cumulative_volume = df['volume'].cumsum()
    df['vwap'] = cumulative_tpv / cumulative_volume
    
    return df

# =================== INSTITUTIONAL BEHAVIOR DETECTION =======
def detect_liquidity_zones(df, lookback=25):
    """Detect institutional liquidity zones"""
    try:
        high_series = df['high'].dropna()
        low_series = df['low'].dropna()
        close_series = df['close'].dropna()
        
        if len(high_series) > lookback:
            # Institutional highs (where large selling occurred)
            high_prices = high_series.rolling(lookback).max().dropna()
            high_volumes = df['volume'][high_prices.index]
            # Find high with significant volume (institutional distribution)
            high_zone_candidates = high_prices[high_volumes > high_volumes.median() * 1.5]
            high_zone = float(high_zone_candidates.iloc[-1]) if len(high_zone_candidates) > 0 else float(high_series.max())
        else:
            high_zone = float(high_series.max()) if len(high_series) > 0 else None
        
        if len(low_series) > lookback:
            # Institutional lows (where large buying occurred)
            low_prices = low_series.rolling(lookback).min().dropna()
            low_volumes = df['volume'][low_prices.index]
            # Find low with significant volume (institutional accumulation)
            low_zone_candidates = low_prices[low_volumes > low_volumes.median() * 1.5]
            low_zone = float(low_zone_candidates.iloc[-1]) if len(low_zone_candidates) > 0 else float(low_series.min())
        else:
            low_zone = float(low_series.min()) if len(low_series) > 0 else None
        
        # Current VWAP for institutional reference
        current_vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else None
        
        return high_zone, low_zone, current_vwap
        
    except Exception as e:
        print(f"Error detecting institutional liquidity zones: {e}")
        return None, None, None

def detect_large_accumulation(df, symbol):
    """Detect institutional large accumulation behavior"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 20:
            return None
        
        # 1. INSTITUTIONAL VOLUME SIGNATURE (Stronger filter)
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_15 * (INSTITUTIONAL_VOLUME_RATIO + 0.5):
            return None
        
        # 2. SMART MONEY ABSORPTION CANDLE
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if lower_wick < current_body * (ABSORPTION_WICK_RATIO - 0.05):
            return None
        
        # 3. INSTITUTIONAL SUPPORT HOLDING
        support_level = low.iloc[-15:-5].min()
        if close.iloc[-1] < support_level * 0.988:
            return None
        
        # 4. INSTITUTIONAL AI CONFIRMATION (Higher confidence)
        accumulation_detected, confidence = flow_ai.detect_accumulation(df, symbol)
        if not accumulation_detected or confidence < 0.88:
            return None
        
        # 5. INSTITUTIONAL PATTERN MATCH
        pattern_matched, pattern = flow_ai.match_institutional_pattern(df, "accumulation")
        if not pattern_matched:
            return None
        
        # 6. INSTITUTIONAL FOLLOW-THROUGH (3 consecutive higher closes)
        if not all([close.iloc[-i] > close.iloc[-(i+1)] for i in range(1, 4)]):
            return None
        
        # 7. ORDER BOOK CONFIRMATION
        order_book_imbalance = flow_ai.analyze_order_book_imbalance(symbol)
        if order_book_imbalance < 0.2:  # Not enough buying pressure
            return None
        
        # 8. CHECK FOR INSTITUTIONAL TIME WINDOW
        hour = datetime.utcnow().hour
        institutional_window = False
        for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
            if start_hour <= hour < end_hour:
                institutional_window = True
                break
        
        if not institutional_window:
            print(f"⚠️ {symbol}: Outside institutional hours")
            return None
        
        print(f"✅ Institutional accumulation confirmed: {symbol} | Confidence: {confidence:.2%} | Pattern: {pattern}")
        return "LONG"
        
    except Exception as e:
        print(f"Error in institutional accumulation detection: {e}")
        return None

def detect_large_distribution(df, symbol):
    """Detect institutional large distribution behavior"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 20:
            return None
        
        # 1. INSTITUTIONAL VOLUME SIGNATURE (Even stronger for distribution)
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_15 * (INSTITUTIONAL_VOLUME_RATIO + 0.8):
            return None
        
        # 2. INSTITUTIONAL DISTRIBUTION CANDLE
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if upper_wick < current_body * (ABSORPTION_WICK_RATIO - 0.03):
            return None
        
        # 3. INSTITUTIONAL RESISTANCE REJECTION
        resistance_level = high.iloc[-15:-5].max()
        if close.iloc[-1] > resistance_level * 1.012:
            return None
        
        # 4. INSTITUTIONAL AI CONFIRMATION (Higher threshold)
        distribution_detected, confidence = flow_ai.detect_distribution(df, symbol)
        if not distribution_detected or confidence < 0.90:
            return None
        
        # 5. INSTITUTIONAL PATTERN MATCH
        pattern_matched, pattern = flow_ai.match_institutional_pattern(df, "distribution")
        if not pattern_matched:
            return None
        
        # 6. INSTITUTIONAL FOLLOW-THROUGH (3 consecutive lower closes)
        if not all([close.iloc[-i] < close.iloc[-(i+1)] for i in range(1, 4)]):
            return None
        
        # 7. ORDER BOOK CONFIRMATION (Strong selling pressure)
        order_book_imbalance = flow_ai.analyze_order_book_imbalance(symbol)
        if order_book_imbalance > -0.25:  # Not enough selling pressure
            return None
        
        # 8. CHECK FOR INSTITUTIONAL TIME WINDOW
        hour = datetime.utcnow().hour
        institutional_window = False
        for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
            if start_hour <= hour < end_hour:
                institutional_window = True
                break
        
        if not institutional_window:
            print(f"⚠️ {symbol}: Outside institutional hours")
            return None
        
        print(f"✅ Institutional distribution confirmed: {symbol} | Confidence: {confidence:.2%} | Pattern: {pattern}")
        return "SHORT"
        
    except Exception as e:
        print(f"Error in institutional distribution detection: {e}")
        return None

def detect_stop_reversal(df, symbol):
    """Detect institutional stop hunt reversals"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 20:
            return None
        
        recent_high = high.iloc[-15:-5].max()
        recent_low = low.iloc[-15:-5].min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        vol_avg = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # INSTITUTIONAL BULL STOP HUNT (then STRONG LONG)
        if (current_low < recent_low * (1 - STOP_HUNT_DISTANCE - 0.002) and
            current_close > recent_low * 1.015 and  # Stronger recovery
            current_vol > vol_avg * 4.2 and  # Higher volume threshold
            current_close > prev_close and
            (current_high - current_close) < (current_close - current_low) * 0.35 and  # Strong bullish close
            current_close > df['vwap'].iloc[-1] if 'vwap' in df.columns else True):  # Above VWAP
            
            # Check order flow for institutional buying
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["buy_pressure"] > 0.65:
                print(f"✅ Institutional bull stop hunt detected: {symbol}")
                return "LONG"
        
        # INSTITUTIONAL BEAR STOP HUNT (then STRONG SHORT)
        if (current_high > recent_high * (1 + STOP_HUNT_DISTANCE + 0.002) and
            current_close < recent_high * 0.985 and  # Stronger rejection
            current_vol > vol_avg * 4.5 and  # Higher volume threshold
            current_close < prev_close and
            (current_close - current_low) < (current_high - current_close) * 0.35 and  # Strong bearish close
            current_close < df['vwap'].iloc[-1] if 'vwap' in df.columns else True):  # Below VWAP
            
            # Check order flow for institutional selling
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["buy_pressure"] < 0.35:
                print(f"✅ Institutional bear stop hunt detected: {symbol}")
                return "SHORT"
        
    except Exception as e:
        print(f"Error in institutional stop reversal detection: {e}")
        return None
    return None

def detect_liquidity_take(df, symbol):
    """Detect institutional liquidity grabs"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 25:
            return None
        
        high_zone, low_zone, current_vwap = detect_liquidity_zones(df, lookback=20)
        current_price = close.iloc[-1]
        
        if high_zone is None or low_zone is None:
            return None
        
        vol_avg_20 = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # INSTITUTIONAL LIQUIDITY TAKE AT HIGHS (then STRONG SHORT)
        if high_zone and abs(current_price - high_zone) <= high_zone * 0.004:  # Tighter range
            if (current_vol > vol_avg_20 * 4.5 and  # Higher volume
                high.iloc[-1] > high_zone * 1.012 and  # Stronger wick
                close.iloc[-1] < high_zone * 0.992 and  # Strong rejection
                volume.iloc[-1] > volume.iloc[-2] * 2.5 and  # Volume spike
                current_price < current_vwap if current_vwap else True):  # Below VWAP
                
                # Check for institutional selling pattern
                pattern_matched, _ = flow_ai.match_institutional_pattern(df, "distribution")
                if pattern_matched:
                    print(f"✅ Institutional liquidity take at highs: {symbol}")
                    return "SHORT"
        
        # INSTITUTIONAL LIQUIDITY TAKE AT LOWS (then STRONG LONG)
        if low_zone and abs(current_price - low_zone) <= low_zone * 0.004:  # Tighter range
            if (current_vol > vol_avg_20 * 4.5 and  # Higher volume
                low.iloc[-1] < low_zone * 0.988 and  # Stronger wick
                close.iloc[-1] > low_zone * 1.008 and  # Strong recovery
                volume.iloc[-1] > volume.iloc[-2] * 2.5 and  # Volume spike
                current_price > current_vwap if current_vwap else True):  # Above VWAP
                
                # Check for institutional buying pattern
                pattern_matched, _ = flow_ai.match_institutional_pattern(df, "accumulation")
                if pattern_matched:
                    print(f"✅ Institutional liquidity take at lows: {symbol}")
                    return "LONG"
        
    except Exception as e:
        print(f"Error in institutional liquidity take detection: {e}")
        return None
    return None

def detect_block_trade_activity(symbol):
    """Detect institutional block trades"""
    try:
        trades = get_recent_trades(symbol, limit=100)
        if not trades:
            return None
        
        # Analyze last 50 trades for block activity
        recent_trades = trades[:50]
        block_trades = []
        
        for trade in recent_trades:
            qty = float(trade.get('qty', 0))
            if qty >= ORDER_FLOW_THRESHOLDS["BLOCK_TRADE_SIZE"]:
                block_trades.append({
                    "side": "BUY" if not trade.get('isBuyerMaker') else "SELL",
                    "price": float(trade.get('price', 0)),
                    "quantity": qty,
                    "time": trade.get('time', 0)
                })
        
        if len(block_trades) >= 2:  # Multiple block trades indicate institutional activity
            # Analyze direction
            buy_blocks = sum(1 for t in block_trades if t["side"] == "BUY")
            sell_blocks = sum(1 for t in block_trades if t["side"] == "SELL")
            
            if buy_blocks > sell_blocks * 1.5:
                print(f"🏛️ Institutional block buying detected: {symbol}")
                return "LONG"
            elif sell_blocks > buy_blocks * 1.5:
                print(f"🏛️ Institutional block selling detected: {symbol}")
                return "SHORT"
        
        return None
        
    except Exception as e:
        print(f"Error detecting block trades: {e}")
        return None

# =================== TECHNICAL TRADING FUNCTIONS ============
def check_technical_conditions(df, mode_cfg, side):
    """Check trading conditions for technical strategy"""
    try:
        price = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        
        # Institutional-grade technical checks
        if side == "LONG":
            # Price above both EMAs (trend alignment)
            trend_aligned = price > ema_20 > ema_50
            rsi_ok = mode_cfg["rsi_long_min"] <= rsi <= mode_cfg["rsi_long_max"]
            # Price above VWAP (institutional reference)
            vwap_ok = price > df['vwap'].iloc[-1] if 'vwap' in df.columns else True
            
            return trend_aligned and rsi_ok and vwap_ok
        else:
            # Price below both EMAs (trend alignment)
            trend_aligned = price < ema_20 < ema_50
            rsi_ok = mode_cfg["rsi_short_min"] <= rsi <= mode_cfg["rsi_short_max"]
            # Price below VWAP (institutional reference)
            vwap_ok = price < df['vwap'].iloc[-1] if 'vwap' in df.columns else True
            
            return trend_aligned and rsi_ok and vwap_ok
    except:
        return False

def compute_technical_entry(df, side, mode_cfg, symbol):
    """Compute institutional-grade entry price with filters"""
    if df.empty:
        return None
    
    price = df["close"].iloc[-1]
    w = mode_cfg["recent_hl_window"]
    
    # Use institutional reference prices
    recent_high = df["high"].iloc[-w:].max()
    recent_low  = df["low"].iloc[-w:].min()
    current_vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else price

    if side == "LONG":
        # Institutional entry above recent high or VWAP
        raw = max(recent_high, current_vwap) * (1 + mode_cfg["entry_buffer_long"])
        predicted = max(raw, price)
        drift = abs(predicted - price) / price
        if drift > mode_cfg["max_entry_drift"]:
            predicted = price * (1 + mode_cfg["max_entry_drift"])
    else:
        # Institutional entry below recent low or VWAP
        raw = min(recent_low, current_vwap) * (1 - mode_cfg["entry_buffer_short"])
        predicted = min(raw, price)
        drift = abs(predicted - price) / price
        if drift > mode_cfg["max_entry_drift"]:
            predicted = price * (1 - mode_cfg["max_entry_drift"])

    # Immediate execution for institutional moves
    if mode_cfg["immediate_mode"] and abs(predicted - price) / price <= mode_cfg["immediate_tol"]:
        predicted = price

    # Institutional volume filter (stricter)
    if mode_cfg["volume_filter"]:
        lb = mode_cfg["volume_lookback"]
        vol_avg = df["volume"].iloc[-lb:].mean()
        last_vol = df["volume"].iloc[-1]
        if last_vol < vol_avg * 1.2:  # Require above-average volume
            return None

    # Institutional-only filter
    if mode_cfg.get("institutional_only", False):
        # Check if this looks like institutional activity
        vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(10).mean().iloc[-1]
        if vol_ratio < 2.5:  # Not institutional volume
            return None

    abs_cap = ABS_MAX_ENTRY_USD.get(symbol, None)
    if abs_cap is not None and abs(predicted - price) > abs_cap:
        if predicted > price:
            predicted = price + abs_cap
        else:
            predicted = price - abs_cap

    return predicted

def calculate_trade_levels(entry_price, side, move_percentage=0.025):
    """Calculate institutional stop loss and target levels"""
    if side == "LONG":
        # Tighter stop loss for institutional longs
        sl = entry_price * (1 - SL_BUFFER * 0.8)
        # Institutional profit-taking levels
        tps = [
            entry_price * (1 + move_percentage * 0.7),   # First target
            entry_price * (1 + move_percentage * 1.3),   # Second target
            entry_price * (1 + move_percentage * 2.0),   # Third target
            entry_price * (1 + move_percentage * 3.0)    # Runner target
        ]
    else:
        # Tighter stop loss for institutional shorts
        sl = entry_price * (1 + SL_BUFFER * 0.8)
        tps = [
            entry_price * (1 - move_percentage * 0.7),   # First target
            entry_price * (1 - move_percentage * 1.3),   # Second target
            entry_price * (1 - move_percentage * 2.0),   # Third target
            entry_price * (1 - move_percentage * 3.0)    # Runner target
        ]
    
    # Ensure targets are in correct order
    if side == "LONG":
        tps.sort()
    else:
        tps.sort(reverse=True)
    
    return sl, tps

# =================== SIGNAL MANAGEMENT ======================
def can_send_signal(symbol, signal_type="flow"):
    """Check if institutional signal can be sent"""
    current_time = time.time()
    
    if signal_type == "flow":
        signal_dict = last_signal_time
        cooldown = 2400  # 40 minutes for institutional flow signals
    else:
        signal_dict = last_signal_time
        cooldown = 1200  # 20 minutes for institutional technical signals
    
    if symbol in signal_dict:
        time_since_last = current_time - signal_dict[symbol]
        if time_since_last < cooldown:
            print(f"⏳ {symbol} in institutional cooldown: {int(cooldown - time_since_last)}s remaining")
            return False
    
    return True

def update_signal_time(symbol):
    """Update last signal time"""
    last_signal_time[symbol] = time.time()

# =================== MONITORING SYSTEM ======================
def monitor_trade_live(symbol, side, entry, sl, targets, strategy_name, signal_id):
    """Continuous institutional monitoring thread for a trade"""
    
    def monitoring_thread():
        print(f"🔍 Starting institutional monitoring for {symbol}")
        
        max_price = entry
        min_price = entry
        entry_triggered = False
        targets_hit = [False] * len(targets)
        last_update_time = time.time()
        institutional_updates_sent = 0
        
        while True:
            current_time = time.time()
            
            # Institutional monitoring timeout (45 minutes)
            if not entry_triggered and (current_time - last_update_time) > 2700:
                print(f"⏰ Institutional monitoring stopped for {symbol} - no entry triggered")
                if signal_id in active_monitoring_threads:
                    del active_monitoring_threads[signal_id]
                break
            
            # Get current price with institutional data
            price = get_current_price(symbol)
            if price is None:
                time.sleep(15)  # Longer wait for institutional data
                continue
            
            last_update_time = current_time
            
            # Update min/max
            if price > max_price:
                max_price = price
            if price < min_price:
                min_price = price
            
            # Check institutional entry
            if not entry_triggered:
                if (side == "LONG" and price >= entry) or (side == "SHORT" and price <= entry):
                    entry_triggered = True
                    # Institutional entry alert
                    send_telegram(f"✅ INSTITUTIONAL ENTRY TRIGGERED: {symbol} {side} @ ${price:.2f}\n"
                                 f"Signal ID: {signal_id}\n"
                                 f"Strategy: {strategy_name}")
            
            if entry_triggered:
                # Check institutional targets
                for i, target in enumerate(targets):
                    if not targets_hit[i]:
                        if (side == "LONG" and price >= target) or (side == "SHORT" and price <= target):
                            targets_hit[i] = True
                            profit_pct = abs(target - entry) / entry * 100
                            send_telegram(f"🎯 INSTITUTIONAL TARGET {i+1} HIT: {symbol} @ ${target:.2f}\n"
                                         f"Profit: {profit_pct:.2f}%\n"
                                         f"Signal ID: {signal_id}")
                
                # Check institutional stop loss
                if (side == "LONG" and price <= sl) or (side == "SHORT" and price >= sl):
                    loss_pct = abs(sl - entry) / entry * 100
                    send_telegram(f"🛑 INSTITUTIONAL STOP LOSS HIT: {symbol} @ ${price:.2f}\n"
                                 f"Loss: {loss_pct:.2f}%\n"
                                 f"Signal ID: {signal_id}")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
                
                # Check if all institutional targets hit
                if all(targets_hit):
                    total_profit = abs(targets[-1] - entry) / entry * 100
                    send_telegram(f"🏆 INSTITUTIONAL TRADE COMPLETED: {symbol}\n"
                                 f"All targets hit!\n"
                                 f"Total Profit: ~{total_profit:.2f}%\n"
                                 f"Signal ID: {signal_id}")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
            
            # Institutional milestone updates (less frequent)
            if entry_triggered and institutional_updates_sent < 3:
                milestone_distance = entry * 0.02  # 2% milestone
                current_move = abs(price - entry) / entry * 100
                
                if current_move >= 1.5 and institutional_updates_sent == 0:
                    send_telegram(f"📈 Institutional {symbol} moving: +{current_move:.2f}% from entry")
                    institutional_updates_sent += 1
                elif current_move >= 3.0 and institutional_updates_sent == 1:
                    send_telegram(f"📈 Institutional {symbol} progressing well: +{current_move:.2f}%")
                    institutional_updates_sent += 1
                elif current_move >= 5.0 and institutional_updates_sent == 2:
                    send_telegram(f"📈 Institutional {symbol} strong move: +{current_move:.2f}%")
                    institutional_updates_sent += 1
            
            # Institutional monitoring interval
            time.sleep(15)
    
    # Start institutional monitoring thread
    thread = threading.Thread(target=monitoring_thread, daemon=True)
    thread.start()
    active_monitoring_threads[signal_id] = thread

# =================== INSTITUTIONAL ALERT FUNCTIONS ==========
def send_flow_alert(symbol, side, entry, sl, targets, behavior_type):
    """Send institutional flow alert"""
    global signal_counter
    
    signal_id = f"INST{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    if behavior_type == "large_accumulation":
        message = (f"🏛️ <b>INSTITUTIONAL ACCUMULATION DETECTED</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:3.5+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n"
                  f"<b>Type:</b> Institutional Accumulation\n\n"
                  f"⚠️ Large institutions accumulating before major UP move")
    
    elif behavior_type == "large_distribution":
        message = (f"🏛️ <b>INSTITUTIONAL DISTRIBUTION DETECTED</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Below:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:3.5+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n"
                  f"<b>Type:</b> Institutional Distribution\n\n"
                  f"⚠️ Large institutions distributing before major DOWN move")
    
    elif behavior_type == "stop_reversal":
        message = (f"🎯 <b>INSTITUTIONAL STOP HUNT REVERSAL</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:4+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n"
                  f"<b>Type:</b> Stop Hunt Reversal\n\n"
                  f"⚠️ Institutional stop hunt complete - strong reversal imminent")
    
    elif behavior_type == "block_trade_buy":
        message = (f"📊 <b>INSTITUTIONAL BLOCK TRADE BUYING</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:3+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n"
                  f"<b>Type:</b> Block Trade Activity\n\n"
                  f"⚠️ Large block trades detected - institutional accumulation")
    
    elif behavior_type == "block_trade_sell":
        message = (f"📊 <b>INSTITUTIONAL BLOCK TRADE SELLING</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Below:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:3+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n"
                  f"<b>Type:</b> Block Trade Activity\n\n"
                  f"⚠️ Large block trades detected - institutional distribution")
    
    else:  # liquidity_take
        message = (f"🌊 <b>INSTITUTIONAL LIQUIDITY GRAB</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry Above:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:4+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n"
                  f"<b>Type:</b> Liquidity Grab\n\n"
                  f"⚠️ Institutions grabbed liquidity - expect strong follow-through")
    
    send_telegram(message)
    update_signal_time(symbol)
    
    # Store institutional signal data
    signal_data = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "behavior_type": behavior_type,
        "timestamp": time.time(),
        "status": "ACTIVE",
        "institutional": True
    }
    
    active_signals[signal_id] = signal_data
    
    # Start institutional monitoring thread
    monitor_trade_live(symbol, side, entry, sl, targets, 
                      BEHAVIOR_TYPES[behavior_type], signal_id)
    
    return signal_id

def send_technical_alert(symbol, side, entry, sl, targets, strategy):
    """Send institutional technical trading alert"""
    global signal_counter
    
    signal_id = f"INST_TECH{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    message = (f"📊 <b>INSTITUTIONAL {strategy} SIGNAL</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Entry Above:</b> ${entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:3+\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Strategy:</b> {strategy}\n"
              f"<b>Confidence:</b> Institutional Grade")
    
    send_telegram(message)
    update_signal_time(symbol)
    
    # Store institutional technical signal data
    signal_data = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "strategy": strategy,
        "timestamp": time.time(),
        "status": "ACTIVE",
        "institutional": True
    }
    
    active_signals[signal_id] = signal_data
    
    # Start institutional monitoring thread
    monitor_trade_live(symbol, side, entry, sl, targets, f"Institutional {strategy}", signal_id)
    
    return signal_id

# =================== INSTITUTIONAL ANALYSIS FUNCTIONS =======
def analyze_institutional_flow(symbol):
    """Analyze for institutional flow behavior"""
    # Get multiple timeframes for institutional analysis
    df_5min = get_market_data(symbol, "5m", 150)
    df_15min = get_market_data(symbol, "15m", 100)
    
    if df_5min is None or len(df_5min) < 30:
        return None
    
    print(f"🔍 Analyzing {symbol} for institutional flow...")
    
    # Add institutional indicators
    df_5min = add_technical_indicators(df_5min)
    
    # Check all institutional behaviors (with priority)
    behaviors = [
        ("large_accumulation", detect_large_accumulation(df_5min, symbol)),
        ("large_distribution", detect_large_distribution(df_5min, symbol)),
        ("stop_reversal", detect_stop_reversal(df_5min, symbol)),
        ("liquidity_take", detect_liquidity_take(df_5min, symbol)),
        ("block_trade_buy", detect_block_trade_activity(symbol))
    ]
    
    for behavior_type, direction in behaviors:
        if direction:
            print(f"✅ {symbol}: {BEHAVIOR_TYPES[behavior_type]} - {direction}")
            
            # Get current price for institutional entry
            current_price = get_current_price(symbol)
            if current_price is None:
                return None
            
            # Calculate institutional levels based on volatility
            atr = (df_5min['high'].iloc[-1] - df_5min['low'].iloc[-1]) / df_5min['close'].iloc[-1]
            move_percentage = max(atr * 1.5, MIN_MOVE_FOR_ENTRY)
            
            entry = current_price
            
            # Adjust entry based on institutional behavior
            if behavior_type == "large_accumulation" and direction == "LONG":
                entry = current_price * 1.002  # Slightly above for accumulation
            elif behavior_type == "large_distribution" and direction == "SHORT":
                entry = current_price * 0.998  # Slightly below for distribution
            
            sl, targets = calculate_trade_levels(entry, direction, move_percentage)
            
            # Store institutional zone in memory
            if direction == "LONG" and behavior_type in ["large_accumulation", "liquidity_take"]:
                institutional_flow_memory[symbol]["accumulation_zones"].append({
                    "price": current_price,
                    "time": time.time(),
                    "type": behavior_type
                })
            elif direction == "SHORT" and behavior_type in ["large_distribution", "liquidity_take"]:
                institutional_flow_memory[symbol]["distribution_zones"].append({
                    "price": current_price,
                    "time": time.time(),
                    "type": behavior_type
                })
            
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
    """Analyze for institutional technical trading signals"""
    cfg = TRADING_MODES[mode_name]
    df = get_market_data(symbol, cfg["interval"], CANDLE_LIMIT)
    
    if df is None or df.empty:
        return None
    
    df = add_technical_indicators(df)
    
    # Check both LONG and SHORT with institutional filters
    for side in ["LONG", "SHORT"]:
        if check_technical_conditions(df, cfg, side):
            entry = compute_technical_entry(df, side, cfg, symbol)
            if entry is None:
                continue
            
            # Calculate institutional levels
            move_percentage = 0.025  # Institutional move expectation
            sl, targets = calculate_trade_levels(entry, side, move_percentage)
            
            # Additional institutional confirmation
            if side == "LONG":
                # Check for institutional buying pressure
                trade_flow = flow_ai.analyze_trade_flow(symbol)
                if trade_flow["buy_pressure"] < 0.6:
                    continue
            else:
                # Check for institutional selling pressure
                trade_flow = flow_ai.analyze_trade_flow(symbol)
                if trade_flow["buy_pressure"] > 0.4:
                    continue
            
            return {
                "symbol": symbol,
                "side": side,
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy": mode_name
            }
    
    return None

# =================== INSTITUTIONAL SCANNER FUNCTIONS ========
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
    
    # Scan top institutional symbols
    for symbol in list(DIGITAL_ASSETS.values())[:4]:  # BTC, ETH, BNB, SOL
        t = threading.Thread(target=scan_symbol, args=(symbol,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    # Send institutional alerts for found signals
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
    """Scan for institutional technical trading signals"""
    print("📊 Scanning for institutional technical signals...")
    
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
    
    print(f"✅ Institutional technical scan complete. Signals: {signals_found}")
    return signals_found

# =================== INSTITUTIONAL STATUS MONITORING ========
def check_active_signals():
    """Check status of active institutional signals"""
    current_time = time.time()
    
    # Clean up old completed signals (older than 2 hours)
    completed_signal_ids = []
    for signal_id, signal_data in active_signals.items():
        if signal_data.get("status") == "COMPLETED":
            if current_time - signal_data["timestamp"] > 7200:
                completed_signal_ids.append(signal_id)
    
    for signal_id in completed_signal_ids:
        del active_signals[signal_id]
    
    # Send institutional status update
    if active_signals:
        active_count = len(active_signals)
        institutional_count = sum(1 for s in active_signals.values() if s.get("institutional"))
        
        print(f"📊 Institutional active signals: {institutional_count}/{active_count}")
        
        # Send institutional summary every 30 minutes
        if int(time.time()) % 1800 < 60:
            institutional_signals = [s for s in active_signals.values() if s.get("institutional")]
            
            if institutional_signals:
                status_msg = f"🏛️ <b>INSTITUTIONAL SIGNALS STATUS</b>\n\n"
                for signal_data in institutional_signals[:6]:  # Show first 6 institutional signals
                    age_min = int((current_time - signal_data["timestamp"]) / 60)
                    status_msg += (f"• {signal_data['symbol']} {signal_data['side']} "
                                  f"- {signal_data.get('behavior_type', signal_data.get('strategy', 'N/A'))} "
                                  f"({age_min}m)\n")
                
                if len(institutional_signals) > 6:
                    status_msg += f"\n... and {len(institutional_signals) - 6} more institutional signals"
                
                send_telegram(status_msg)
    
    return len(active_signals)

# =================== INSTITUTIONAL MAIN EXECUTION ===========
def main():
    """Main institutional execution loop"""
    print("=" * 60)
    print("🏛️ ADVANCED INSTITUTIONAL FLOW & TRADING BOT")
    print("📊 Institutional-Grade Continuous Monitoring")
    print("🎯 Professional Signals Tracked Until Completion")
    print("=" * 60)
    
    send_telegram("🏛️ <b>INSTITUTIONAL TRADING BOT ACTIVATED</b>\n"
                  "🔍 Advanced Institutional Flow Detection\n"
                  "📊 Institutional Technical Strategies\n"
                  "🌊 Smart Money Pattern Recognition\n"
                  "⚡ Order Flow Analysis\n"
                  f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC\n\n"
                  f"<i>Monitoring institutional hours: London, NY, Asia sessions</i>")
    
    iteration = 0
    flow_scan_counter = 0
    
    while True:
        iteration += 1
        try:
            print(f"\n🔄 Institutional Iteration {iteration} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            
            # Check active institutional signals
            active_count = check_active_signals()
            print(f"📈 Institutional active signals: {active_count}")
            
            # Check if current time is optimal for institutional scanning
            hour = datetime.utcnow().hour
            institutional_hour = False
            for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
                if start_hour <= hour < end_hour:
                    institutional_hour = True
                    print(f"🏛️ Institutional hour: {period_name}")
                    break
            
            # Run institutional scanner during institutional hours
            if institutional_hour or flow_scan_counter % 3 == 0:
                flow_signals = run_institutional_scanner()
                print(f"🏛️ Institutional signals found: {flow_signals}")
            
            # Run institutional technical scanner
            tech_signals = run_technical_scanner()
            print(f"📊 Institutional technical signals found: {tech_signals}")
            
            flow_scan_counter += 1
            
            # Wait for next institutional scan (45 seconds during active hours)
            wait_time = 45 if institutional_hour else 90
            print(f"⏳ Waiting {wait_time} seconds for next institutional scan...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down institutional bot...")
            
            # Send final institutional status
            institutional_signals = [s for s in active_signals.values() if s.get("institutional")]
            
            if institutional_signals:
                send_telegram(f"🛑 <b>INSTITUTIONAL BOT SHUTTING DOWN</b>\n"
                             f"Institutional signals active: {len(institutional_signals)}\n"
                             "Monitor remaining institutional signals carefully")
            else:
                send_telegram("🛑 <b>INSTITUTIONAL BOT SHUTTING DOWN</b>\n"
                             "No active institutional signals")
            
            break
            
        except Exception as e:
            print(f"⚠️ Institutional main loop error: {e}")
            send_telegram(f"⚠️ <b>INSTITUTIONAL BOT ERROR</b>\n"
                         f"Error: {str(e)[:150]}\n"
                         f"Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            time.sleep(120)

if __name__ == "__main__":
    # Test institutional credentials
    print("🔧 Testing institutional bot configuration...")
    print(f"Institutional API Key configured: {'Yes' if SYSTEM_KEY and SYSTEM_KEY.strip() and SYSTEM_KEY != ' ' else 'No'}")
    print(f"Institutional API Secret configured: {'Yes' if SYSTEM_SECRET and SYSTEM_SECRET.strip() and SYSTEM_SECRET != ' ' else 'No'}")
    print(f"Institutional Telegram configured: {'Yes' if ALERT_TOKEN and ALERT_TOKEN.strip() and ALERT_TOKEN != ' ' and ALERT_TARGET and ALERT_TARGET.strip() and ALERT_TARGET != ' ' else 'No'}")
    
    # Check current time for institutional activity
    hour = datetime.utcnow().hour
    institutional_active = False
    for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
        if start_hour <= hour < end_hour:
            print(f"✅ Currently in institutional trading hours: {period_name}")
            institutional_active = True
            break
    
    if not institutional_active:
        print("⚠️ Outside institutional trading hours - reduced scan frequency")
    
    # Start institutional main loop
    main()
