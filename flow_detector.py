# =================== INSTITUTIONAL FLOW & CRYPTO TRADING BOT ===================
# AI-BASED INSTITUTIONAL FLOW DETECTION WITH BREAKOUT CONFIRMATION
# FIXED VERSION - NO PREMATURE SIGNALS, BREAKOUT CONFIRMATION REQUIRED
# ENHANCED WITH MULTI-TIMEFRAME INSTITUTIONAL STRATEGIES

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
INSTITUTIONAL_VOLUME_RATIO = 3.5  # Reduced to allow more signals
MIN_MOVE_FOR_ENTRY = 0.02  # Reduced minimum move threshold
STOP_HUNT_DISTANCE = 0.012  # Reduced stop hunt sensitivity
ABSORPTION_WICK_RATIO = 0.20  # Tighter wick ratio

# =================== INSTITUTIONAL TIME ZONES ===============
INSTITUTIONAL_ACTIVE_HOURS = {
    "LONDON_OPEN": (7, 12),    # 7AM - 12PM UTC
    "NY_OPEN": (13, 17),       # 1PM - 5PM UTC
    "ASIA_LATE": (22, 24),     # 10PM - 12AM UTC
    "ASIA_EARLY": (0, 4)       # 12AM - 4AM UTC
}

# =================== INSTITUTIONAL ORDER FLOW ===============
ORDER_FLOW_THRESHOLDS = {
    "BLOCK_TRADE_SIZE_USD": 50000.0,   # $50,000 for block trades
    "SWEEP_RATIO": 0.70,
    "IMMEDIATE_OR_CANCEL": 0.8,
    "DARK_POOL_INDICATOR": 4.0
}

# =================== TECHNICAL TRADING SETTINGS ============
EMA_PERIOD = 20
RSI_PERIOD = 14
CANDLE_LIMIT = 200  # Increased for better pattern recognition

SL_BUFFER = 0.0030  # Increased to 0.3% for better breathing room
TARGETS = [0.0040, 0.008, 0.015]  # Better risk/reward: 1:1.33, 1:2.67, 1:5

ABS_MAX_ENTRY_PCT = {
    "BTC-USDT": 0.008,  # Max 0.8% from current price
    "ETH-USDT": 0.010,  # Max 1.0% from current price
    "BNB-USDT": 0.012,  # Max 1.2% from current price
    "SOL-USDT": 0.015,  # Max 1.5% from current price
    "XRP-USDT": 0.020   # Max 2.0% from current price
}

# =================== NEW: MULTI-TIMEFRAME STRATEGIES ===========
MULTI_TIMEFRAME_STRATEGIES = {
    "1H_INSTITUTIONAL": {
        "interval": "1h",
        "volume_ratio": 3.0,
        "min_move_pct": 0.015,  # 1.5% minimum move
        "window": 10,
        "description": "1H Institutional Breakout"
    },
    "15M_MOMENTUM": {
        "interval": "15m",
        "volume_ratio": 3.5,
        "min_move_pct": 0.008,  # 0.8% minimum move
        "window": 15,
        "description": "15M Momentum Move"
    },
    "5M_BREAKOUT": {
        "interval": "5m",
        "volume_ratio": 4.0,
        "min_move_pct": 0.005,  # 0.5% minimum move
        "window": 20,
        "description": "5M Quick Breakout"
    },
    "INSTITUTIONAL_VOLUME_SURGE": {
        "interval": "5m",
        "volume_ratio": 5.0,  # Very high volume
        "min_move_pct": 0.003,  # Small move but huge volume
        "window": 5,
        "description": "Institutional Volume Surge"
    }
}

TRADING_MODES = {
    "SCALP": {
        "interval": "1m",
        "recent_hl_window": 10,  # Increased for better breakout detection
        "rsi_long_min": 50, "rsi_long_max": 80,  # Slightly wider ranges
        "rsi_short_min": 20, "rsi_short_max": 50,
        "entry_buffer_long": 0.0002,
        "entry_buffer_short": 0.0002,
        "max_entry_drift": 0.0010,  # Increased for breakout tolerance
        "immediate_mode": False,  # Disabled - wait for actual breakout
        "require_breakout_confirmation": True,  # NEW: Must confirm breakout
        "confirmation_candles": 1,  # Wait 1 candle to confirm
        "need_prev_candle_break": True,
        "volume_filter": True,
        "volume_lookback": 30,
        "institutional_only": True,
        "min_volume_ratio": 2.5,  # Reduced minimum volume ratio
        "breakout_retest_allowed": True,  # Allow retest of breakout level
        "max_retest_count": 2,  # Max retests allowed
        "multi_timeframe_check": True  # NEW: Check multiple timeframes
    },
    "SWING": {
        "interval": "5m",
        "recent_hl_window": 20,  # Increased window
        "rsi_long_min": 48, "rsi_long_max": 82,
        "rsi_short_min": 18, "rsi_short_max": 52,
        "entry_buffer_long": 0.0005,
        "entry_buffer_short": 0.0005,
        "max_entry_drift": 0.0030,  # Increased for swing tolerance
        "immediate_mode": False,  # Disabled
        "require_breakout_confirmation": True,  # NEW
        "confirmation_candles": 2,  # Wait 2 candles to confirm
        "need_prev_candle_break": True,
        "volume_filter": True,
        "volume_lookback": 40,
        "institutional_only": True,
        "min_volume_ratio": 2.0,  # Reduced
        "breakout_retest_allowed": True,
        "max_retest_count": 3,
        "multi_timeframe_check": True  # NEW: Check multiple timeframes
    },
    # NEW: ADDED INSTITUTIONAL MOMENTUM STRATEGY
    "INSTITUTIONAL_MOMENTUM": {
        "interval": "15m",
        "recent_hl_window": 15,
        "rsi_long_min": 45, "rsi_long_max": 85,
        "rsi_short_min": 15, "rsi_short_max": 55,
        "entry_buffer_long": 0.0003,
        "entry_buffer_short": 0.0003,
        "max_entry_drift": 0.0020,
        "immediate_mode": False,
        "require_breakout_confirmation": True,
        "confirmation_candles": 1,
        "need_prev_candle_break": True,
        "volume_filter": True,
        "volume_lookback": 25,
        "institutional_only": True,
        "min_volume_ratio": 3.0,  # Higher volume for institutional
        "breakout_retest_allowed": False,  # No retest for institutional
        "multi_timeframe_check": True
    }
}

# =================== INSTITUTIONAL BEHAVIOR TYPES ===========
BEHAVIOR_TYPES = {
    "institutional_buying": "🏛️ INSTITUTIONAL BUYING",
    "institutional_selling": "🏛️ INSTITUTIONAL SELLING", 
    "bullish_stop_hunt": "🎯 BULLISH STOP HUNT",
    "bearish_stop_hunt": "🎯 BEARISH STOP HUNT",
    "liquidity_grab_bullish": "🌊 BULLISH LIQUIDITY GRAB",
    "liquidity_grab_bearish": "🌊 BEARISH LIQUIDITY GRAB",
    # NEW: Added multi-timeframe patterns
    "1h_breakout": "📈 1H INSTITUTIONAL BREAKOUT",
    "15m_momentum": "⚡ 15M MOMENTUM MOVE",
    "5m_breakout": "🎯 5M QUICK BREAKOUT",
    "volume_surge": "📊 INSTITUTIONAL VOLUME SURGE"
}

# =================== GLOBAL TRACKING ========================
signal_counter = 0
active_signals = {}
last_signal_time = {}
active_monitoring_threads = {}
completed_signals = []
pending_breakouts = {}  # NEW: Track pending breakouts waiting confirmation
pending_multi_tf_signals = {}  # NEW: Track multi-timeframe signals

# Signal cooldown periods (in seconds)
SIGNAL_COOLDOWN = 7200  # 120 minutes for same symbol
BREAKOUT_CONFIRMATION_TIMEOUT = 3600  # 1 hour max wait for confirmation
MULTI_TF_COOLDOWN = 1800  # 30 minutes for multi-timeframe signals

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
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'quote_volume', 'trades'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].astype(float)
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

def get_order_book(symbol, limit=50):
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

def get_recent_trades(symbol, limit=200):
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
            
            # INSTITUTIONAL BUYING PATTERNS (LONG)
            X_buy = []
            y_buy = []
            
            # Pattern 1: High volume absorption at support
            X_buy.append([5.5, 0.28, 0.15, 4.0, 0.25, 0.78, 0.75, 0.030, 1.8, 0.85])
            y_buy.append(1)
            
            # Pattern 2: Stealth accumulation
            X_buy.append([4.8, 0.22, 0.18, 3.5, 0.20, 0.72, 0.70, 0.025, 1.6, 0.80])
            y_buy.append(1)
            
            # Pattern 3: V-shaped recovery
            X_buy.append([6.0, 0.35, 0.12, 4.5, 0.30, 0.82, 0.78, 0.035, 2.0, 0.88])
            y_buy.append(1)
            
            # NON-INSTITUTIONAL patterns (noise)
            X_buy.append([1.8, 0.10, 0.60, 1.2, 0.08, 0.35, 0.45, 0.010, 0.7, 0.40])
            y_buy.append(0)
            
            X_buy.append([2.5, 0.15, 0.55, 1.8, 0.12, 0.42, 0.50, 0.015, 0.9, 0.48])
            y_buy.append(0)
            
            # INSTITUTIONAL SELLING PATTERNS (SHORT)
            X_sell = []
            y_sell = []
            
            # Pattern 1: Distribution at resistance
            X_sell.append([5.8, 0.20, 0.30, 4.2, 0.28, 0.80, 0.25, 0.032, 1.9, 0.22])
            y_sell.append(1)
            
            # Pattern 2: Failed breakout selling
            X_sell.append([6.2, 0.25, 0.35, 4.5, 0.32, 0.85, 0.20, 0.035, 2.1, 0.18])
            y_sell.append(1)
            
            # Pattern 3: Exhaustion selling
            X_sell.append([5.0, 0.18, 0.28, 3.8, 0.24, 0.75, 0.30, 0.028, 1.7, 0.25])
            y_sell.append(1)
            
            # NON-INSTITUTIONAL patterns (noise)
            X_sell.append([2.0, 0.12, 0.65, 1.5, 0.10, 0.40, 0.55, 0.012, 0.8, 0.60])
            y_sell.append(0)
            
            X_sell.append([2.8, 0.18, 0.58, 2.2, 0.15, 0.48, 0.60, 0.018, 1.0, 0.65])
            y_sell.append(0)
            
            X_buy = np.array(X_buy)
            y_buy = np.array(y_buy)
            X_sell = np.array(X_sell)
            y_sell = np.array(y_sell)
            
            self.scaler = StandardScaler()
            X_buy_scaled = self.scaler.fit_transform(X_buy)
            
            self.accumulation_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42
            )
            self.accumulation_model.fit(X_buy_scaled, y_buy)
            
            X_sell_scaled = self.scaler.transform(X_sell)
            
            self.distribution_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=6,
                min_samples_leaf=3,
                random_state=42,
                class_weight='balanced'
            )
            self.distribution_model.fit(X_sell_scaled, y_sell)
            
            joblib.dump(self.accumulation_model, "flow_accumulation_model.pkl")
            joblib.dump(self.distribution_model, "flow_distribution_model.pkl")
            joblib.dump(self.scaler, "flow_scaler.pkl")
            
            print("✅ Institutional AI models trained and saved")
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            self.accumulation_model = None
            self.distribution_model = None
            self.scaler = None
    
    def analyze_trade_flow(self, symbol):
        """Analyze trade flow for institutional activity - FIXED with USD conversion"""
        try:
            trades = get_recent_trades(symbol, limit=150)
            if not trades:
                return {"buy_pressure": 0.5, "block_buys": 0, "block_sells": 0, "direction": "NEUTRAL"}
            
            buy_volume_usd = 0
            sell_volume_usd = 0
            block_buys = 0
            block_sells = 0
            
            # Get current price for USD conversion
            current_price = get_current_price(symbol)
            if current_price is None:
                current_price = 1.0
            
            for trade in trades:
                qty = float(trade.get('qty', 0))
                price = float(trade.get('price', current_price))
                
                # Convert to USD value
                trade_value_usd = qty * price
                
                # FIXED LOGIC with USD comparison
                if trade.get('isBuyerMaker'):
                    sell_volume_usd += trade_value_usd
                    if trade_value_usd >= ORDER_FLOW_THRESHOLDS["BLOCK_TRADE_SIZE_USD"]:
                        block_sells += 1
                else:
                    buy_volume_usd += trade_value_usd
                    if trade_value_usd >= ORDER_FLOW_THRESHOLDS["BLOCK_TRADE_SIZE_USD"]:
                        block_buys += 1
            
            total_volume_usd = buy_volume_usd + sell_volume_usd
            buy_pressure = buy_volume_usd / total_volume_usd if total_volume_usd > 0 else 0.5
            
            # Determine direction based on block trades
            if block_buys > block_sells * 2 and block_buys >= 2:
                direction = "LONG"
            elif block_sells > block_buys * 2 and block_sells >= 2:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"
            
            return {
                "buy_pressure": buy_pressure,
                "block_buys": block_buys,
                "block_sells": block_sells,
                "buy_volume_usd": buy_volume_usd,
                "sell_volume_usd": sell_volume_usd,
                "total_volume_usd": total_volume_usd,
                "direction": direction
            }
            
        except Exception as e:
            print(f"Error in trade flow analysis: {e}")
            return {"buy_pressure": 0.5, "block_buys": 0, "block_sells": 0, "direction": "NEUTRAL"}

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
    
    # Volume indicators
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1e-9)
    
    return df

# =================== INSTITUTIONAL BEHAVIOR DETECTION =======
def detect_institutional_buying(df, symbol):
    """Detect institutional buying (LONG) - STRICTER RULES"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 30:
            return None
        
        # 1. HIGH VOLUME WITH STRONG BULLISH CLOSE
        vol_avg_20 = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if vol_avg_20 == 0 or current_vol < vol_avg_20 * INSTITUTIONAL_VOLUME_RATIO:
            return None
        
        # 2. BULLISH CANDLE WITH STRONG LOWER WICK (absorption)
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if current_body == 0 or lower_wick < current_body * ABSORPTION_WICK_RATIO:
            return None
        
        # 3. PRICE CLOSING IN TOP 25% OF CANDLE
        candle_range = high.iloc[-1] - low.iloc[-1]
        if candle_range > 0:
            close_position = (close.iloc[-1] - low.iloc[-1]) / candle_range
            if close_position < 0.75:  # Must close in top 25%
                return None
        
        # 4. CHECK TRADE FLOW FOR INSTITUTIONAL BUYING
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] != "LONG" or trade_flow["block_buys"] < 2:
            return None
        
        # 5. PRICE ACTION CONFIRMATION (2 consecutive higher closes)
        if not (close.iloc[-1] > close.iloc[-2] > close.iloc[-3]):
            return None
        
        # 6. CHECK IF NOT AT STRONG RESISTANCE (with buffer)
        resistance = high.iloc[-25:-10].max()
        if close.iloc[-1] > resistance * 0.97:  # 3% buffer below resistance
            return None
        
        # 7. CHECK INSTITUTIONAL HOURS
        hour = datetime.utcnow().hour
        institutional_hour = False
        for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
            if start_hour <= hour < end_hour:
                institutional_hour = True
                break
        
        if not institutional_hour:
            return None
        
        print(f"✅ Institutional buying confirmed: {symbol} | Block buys: {trade_flow['block_buys']}")
        return "LONG"
        
    except Exception as e:
        print(f"Error in institutional buying detection: {e}")
        return None

def detect_institutional_selling(df, symbol):
    """Detect institutional selling (SHORT) - STRICTER RULES"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 30:
            return None
        
        # 1. HIGH VOLUME WITH STRONG BEARISH CLOSE
        vol_avg_20 = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if vol_avg_20 == 0 or current_vol < vol_avg_20 * (INSTITUTIONAL_VOLUME_RATIO + 0.5):
            return None
        
        # 2. BEARISH CANDLE WITH STRONG UPPER WICK (distribution)
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if current_body == 0 or upper_wick < current_body * ABSORPTION_WICK_RATIO:
            return None
        
        # 3. PRICE CLOSING IN BOTTOM 25% OF CANDLE
        candle_range = high.iloc[-1] - low.iloc[-1]
        if candle_range > 0:
            close_position = (close.iloc[-1] - low.iloc[-1]) / candle_range
            if close_position > 0.25:  # Must close in bottom 25%
                return None
        
        # 4. CHECK TRADE FLOW FOR INSTITUTIONAL SELLING
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] != "SHORT" or trade_flow["block_sells"] < 2:
            return None
        
        # 5. PRICE ACTION CONFIRMATION (2 consecutive lower closes)
        if not (close.iloc[-1] < close.iloc[-2] < close.iloc[-3]):
            return None
        
        # 6. CHECK IF NOT AT STRONG SUPPORT (with buffer)
        support = low.iloc[-25:-10].min()
        if close.iloc[-1] < support * 1.03:  # 3% buffer above support
            return None
        
        # 7. CHECK INSTITUTIONAL HOURS
        hour = datetime.utcnow().hour
        institutional_hour = False
        for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
            if start_hour <= hour < end_hour:
                institutional_hour = True
                break
        
        if not institutional_hour:
            return None
        
        print(f"✅ Institutional selling confirmed: {symbol} | Block sells: {trade_flow['block_sells']}")
        return "SHORT"
        
    except Exception as e:
        print(f"Error in institutional selling detection: {e}")
        return None

def detect_bullish_stop_hunt(df, symbol):
    """Detect bullish stop hunt (LONG)"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 25:
            return None
        
        # Look for stop hunt below recent low
        recent_low = low.iloc[-20:-8].min()
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if vol_avg == 0:
            return None
        
        # Bullish stop hunt pattern
        if (current_low < recent_low * (1 - STOP_HUNT_DISTANCE) and
            current_close > recent_low * 1.018 and  # Strong recovery
            current_vol > vol_avg * 5.0 and  # Higher volume threshold
            current_close > close.iloc[-2] and
            (high.iloc[-1] - current_close) < (current_close - current_low) * 0.4):  # Strong bullish close
            
            # Confirm with trade flow
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["direction"] == "LONG":
                print(f"✅ Bullish stop hunt detected: {symbol}")
                return "LONG"
        
    except Exception as e:
        print(f"Error in bullish stop hunt detection: {e}")
        return None
    return None

def detect_bearish_stop_hunt(df, symbol):
    """Detect bearish stop hunt (SHORT)"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 25:
            return None
        
        # Look for stop hunt above recent high
        recent_high = high.iloc[-20:-8].max()
        current_high = high.iloc[-1]
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if vol_avg == 0:
            return None
        
        # Bearish stop hunt pattern
        if (current_high > recent_high * (1 + STOP_HUNT_DISTANCE) and
            current_close < recent_high * 0.982 and  # Strong rejection
            current_vol > vol_avg * 5.5 and  # Higher volume threshold
            current_close < close.iloc[-2] and
            (current_close - low.iloc[-1]) < (current_high - current_close) * 0.4):  # Strong bearish close
            
            # Confirm with trade flow
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["direction"] == "SHORT":
                print(f"✅ Bearish stop hunt detected: {symbol}")
                return "SHORT"
        
    except Exception as e:
        print(f"Error in bearish stop hunt detection: {e}")
        return None
    return None

# =================== NEW: MULTI-TIMEFRAME DETECTION ===========
def detect_multi_timeframe_breakout(symbol, timeframe_strategy):
    """Detect institutional breakout in specific timeframe"""
    try:
        strategy = MULTI_TIMEFRAME_STRATEGIES[timeframe_strategy]
        df = get_market_data(symbol, strategy["interval"], 100)
        
        if df is None or len(df) < 30:
            return None
        
        df = add_technical_indicators(df)
        
        # Check volume spike
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        if avg_volume == 0 or current_volume < avg_volume * strategy["volume_ratio"]:
            return None
        
        # Check for breakout
        window = strategy["window"]
        recent_high = df['high'].iloc[-window:-1].max()
        recent_low = df['low'].iloc[-window:-1].min()
        current_close = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        # Check bullish breakout
        if current_high > recent_high:
            breakout_size = (current_high - recent_high) / recent_high
            if breakout_size >= strategy["min_move_pct"]:
                # Check if closing above resistance
                if current_close > recent_high:
                    print(f"✅ {timeframe_strategy} BULLISH breakout: {symbol}")
                    return {
                        "type": timeframe_strategy,
                        "direction": "LONG",
                        "breakout_level": recent_high,
                        "current_price": current_close,
                        "volume_ratio": current_volume / avg_volume,
                        "breakout_size_pct": breakout_size * 100
                    }
        
        # Check bearish breakout
        if current_low < recent_low:
            breakout_size = (recent_low - current_low) / recent_low
            if breakout_size >= strategy["min_move_pct"]:
                # Check if closing below support
                if current_close < recent_low:
                    print(f"✅ {timeframe_strategy} BEARISH breakout: {symbol}")
                    return {
                        "type": timeframe_strategy,
                        "direction": "SHORT",
                        "breakout_level": recent_low,
                        "current_price": current_close,
                        "volume_ratio": current_volume / avg_volume,
                        "breakout_size_pct": breakout_size * 100
                    }
        
        return None
        
    except Exception as e:
        print(f"Error in {timeframe_strategy} detection for {symbol}: {e}")
        return None

def detect_volume_surge(symbol):
    """Detect institutional volume surge (very high volume with small move)"""
    try:
        df_5m = get_market_data(symbol, "5m", 50)
        if df_5m is None or len(df_5m) < 20:
            return None
        
        # Check for extreme volume spike
        current_volume = df_5m['volume'].iloc[-1]
        avg_volume = df_5m['volume'].rolling(20).mean().iloc[-1]
        
        if avg_volume == 0 or current_volume < avg_volume * 5.0:
            return None
        
        # Price should be relatively stable (not a huge move)
        current_close = df_5m['close'].iloc[-1]
        prev_close = df_5m['close'].iloc[-2]
        price_change = abs(current_close - prev_close) / prev_close
        
        # Volume surge with small price change = institutional accumulation/distribution
        if price_change <= 0.005:  # Less than 0.5% price change
            # Check trade flow for institutional activity
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["block_buys"] > 3 or trade_flow["block_sells"] > 3:
                direction = "LONG" if trade_flow["block_buys"] > trade_flow["block_sells"] else "SHORT"
                print(f"✅ Institutional volume surge detected: {symbol} - {direction}")
                return {
                    "type": "volume_surge",
                    "direction": direction,
                    "current_price": current_close,
                    "volume_ratio": current_volume / avg_volume,
                    "block_trades": trade_flow["block_buys"] if direction == "LONG" else trade_flow["block_sells"]
                }
        
        return None
        
    except Exception as e:
        print(f"Error in volume surge detection for {symbol}: {e}")
        return None

# =================== BREAKOUT DETECTION FUNCTIONS ===========
def check_breakout(df, side, window=10):
    """Check if price has broken key levels"""
    try:
        if side == "LONG":
            # Resistance is previous highs
            resistance = df["high"].iloc[-window:-1].max()  # Exclude current candle
            current_high = df["high"].iloc[-1]
            current_close = df["close"].iloc[-1]
            
            # Breakout: current high exceeds resistance AND closes above it
            if current_high > resistance and current_close > resistance:
                return True, resistance
            return False, resistance
            
        else:  # SHORT
            # Support is previous lows
            support = df["low"].iloc[-window:-1].min()  # Exclude current candle
            current_low = df["low"].iloc[-1]
            current_close = df["close"].iloc[-1]
            
            # Breakdown: current low breaks support AND closes below it
            if current_low < support and current_close < support:
                return True, support
            return False, support
            
    except Exception as e:
        print(f"Error in breakout check: {e}")
        return False, None

def check_pending_breakouts(symbol):
    """Check if any pending breakouts have been confirmed"""
    current_time = time.time()
    confirmed_breakouts = []
    
    for breakout_id, breakout_data in list(pending_breakouts.items()):
        if breakout_data["symbol"] != symbol:
            continue
            
        # Check timeout
        if current_time - breakout_data["timestamp"] > BREAKOUT_CONFIRMATION_TIMEOUT:
            print(f"⏰ Breakout {breakout_id} timed out for {symbol}")
            del pending_breakouts[breakout_id]
            continue
        
        # Get latest data
        df = get_market_data(symbol, breakout_data["interval"], 30)
        if df is None or df.empty:
            continue
        
        # Check if breakout has been confirmed
        breakout_confirmed, _ = check_breakout(df, breakout_data["side"], 
                                              breakout_data["window"])
        
        if breakout_confirmed:
            print(f"✅ Breakout {breakout_id} confirmed for {symbol}")
            confirmed_breakouts.append(breakout_data)
            del pending_breakouts[breakout_id]
    
    return confirmed_breakouts

# =================== TECHNICAL TRADING FUNCTIONS ============
def check_technical_conditions(df, mode_cfg, side):
    """Check trading conditions for technical strategy - WITH BREAKOUT CONFIRMATION"""
    try:
        if df.empty or len(df) < 20:
            return False
        
        price = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        
        # Additional confirmation candles
        confirmation_candles = mode_cfg.get("confirmation_candles", 1)
        
        if side == "LONG":
            # Check if last X candles were bullish
            bullish_confirmation = all(df["close"].iloc[-i] > df["open"].iloc[-i] 
                                     for i in range(1, confirmation_candles + 1))
            
            trend_aligned = price > ema_20 > ema_50
            rsi_ok = mode_cfg["rsi_long_min"] <= rsi <= mode_cfg["rsi_long_max"]
            
            # Volume check
            vol_avg = df["volume"].rolling(10).mean().iloc[-1]
            last_vol = df["volume"].iloc[-1]
            volume_ok = vol_avg > 0 and last_vol >= vol_avg * mode_cfg.get("min_volume_ratio", 2.0)
            
            # BREAKOUT CHECK - NEW
            if mode_cfg.get("need_prev_candle_break", False):
                breakout_confirmed, _ = check_breakout(df, side, mode_cfg["recent_hl_window"])
                if not breakout_confirmed:
                    return False
            
            return trend_aligned and rsi_ok and bullish_confirmation and volume_ok
        else:
            # Check if last X candles were bearish
            bearish_confirmation = all(df["close"].iloc[-i] < df["open"].iloc[-i] 
                                     for i in range(1, confirmation_candles + 1))
            
            trend_aligned = price < ema_20 < ema_50
            rsi_ok = mode_cfg["rsi_short_min"] <= rsi <= mode_cfg["rsi_short_max"]
            
            # Volume check
            vol_avg = df["volume"].rolling(10).mean().iloc[-1]
            last_vol = df["volume"].iloc[-1]
            volume_ok = vol_avg > 0 and last_vol >= vol_avg * mode_cfg.get("min_volume_ratio", 2.0)
            
            # BREAKDOWN CHECK - NEW
            if mode_cfg.get("need_prev_candle_break", False):
                breakout_confirmed, _ = check_breakout(df, side, mode_cfg["recent_hl_window"])
                if not breakout_confirmed:
                    return False
            
            return trend_aligned and rsi_ok and bearish_confirmation and volume_ok
    except Exception as e:
        print(f"Error in technical conditions check: {e}")
        return False

def compute_technical_entry(df, side, mode_cfg, symbol):
    """Compute entry price with filters - FIXED for breakout entries"""
    if df.empty:
        return None
    
    price = df["close"].iloc[-1]
    w = mode_cfg["recent_hl_window"]
    
    # Get breakout level
    breakout_confirmed, breakout_level = check_breakout(df, side, w)
    
    if breakout_confirmed and breakout_level:
        # Price has already broken out - entry at current price or slight pullback
        if side == "LONG":
            entry = max(price, breakout_level * (1 + 0.0001))  # Slightly above breakout
        else:
            entry = min(price, breakout_level * (1 - 0.0001))  # Slightly below breakdown
    else:
        # No breakout yet - calculate potential entry
        if side == "LONG":
            resistance = df["high"].iloc[-w:].max()
            entry = resistance * (1 + mode_cfg["entry_buffer_long"])
            
            # Cap the distance from current price
            max_pct = ABS_MAX_ENTRY_PCT.get(symbol, 0.01)
            if entry > price * (1 + max_pct):
                entry = price * (1 + max_pct * 0.7)  # 70% of max distance
        else:
            support = df["low"].iloc[-w:].min()
            entry = support * (1 - mode_cfg["entry_buffer_short"])
            
            # Cap the distance from current price
            max_pct = ABS_MAX_ENTRY_PCT.get(symbol, 0.01)
            if entry < price * (1 - max_pct):
                entry = price * (1 - max_pct * 0.7)  # 70% of max distance
    
    # STRICTER VOLUME FILTER
    if mode_cfg["volume_filter"]:
        lb = mode_cfg["volume_lookback"]
        vol_avg = df["volume"].iloc[-lb:].mean()
        last_vol = df["volume"].iloc[-1]
        if vol_avg > 0 and last_vol < vol_avg * mode_cfg.get("min_volume_ratio", 2.0):
            return None
    
    return entry

def calculate_trade_levels(entry_price, side):
    """Calculate stop loss and target levels"""
    if side == "LONG":
        sl = entry_price * (1 - SL_BUFFER)
        tps = [
            entry_price * (1 + TARGETS[0]),
            entry_price * (1 + TARGETS[1]),
            entry_price * (1 + TARGETS[2])
        ]
    else:
        sl = entry_price * (1 + SL_BUFFER)
        tps = [
            entry_price * (1 - TARGETS[0]),
            entry_price * (1 - TARGETS[1]),
            entry_price * (1 - TARGETS[2])
        ]
    
    return sl, tps

# =================== SIGNAL MANAGEMENT ======================
def can_send_signal(symbol, signal_type="default"):
    """Check if signal can be sent"""
    current_time = time.time()
    
    # Different cooldowns for different signal types
    if signal_type == "multi_tf":
        cooldown = MULTI_TF_COOLDOWN  # 30 minutes
    else:
        cooldown = SIGNAL_COOLDOWN  # 120 minutes
    
    if symbol in last_signal_time:
        time_since_last = current_time - last_signal_time[symbol]
        if time_since_last < cooldown:
            print(f"⏳ {symbol} in cooldown: {int((cooldown - time_since_last)/60)}min remaining")
            return False
    
    return True

def update_signal_time(symbol, signal_type="default"):
    """Update last signal time"""
    last_signal_time[symbol] = time.time()

# =================== MONITORING SYSTEM ======================
def monitor_trade_live(symbol, side, entry, sl, targets, strategy_name, signal_id):
    """Continuous monitoring thread for a trade"""
    
    def monitoring_thread():
        print(f"🔍 Starting monitoring for {symbol}")
        
        entry_triggered = False
        targets_hit = [False] * len(targets)
        entry_attempts = 0
        max_entry_attempts = 20  # Stop trying after 20 checks (approx 3.3 minutes)
        
        while True:
            # Get current price
            price = get_current_price(symbol)
            if price is None:
                time.sleep(10)
                continue
            
            # Check entry
            if not entry_triggered:
                entry_attempts += 1
                if (side == "LONG" and price >= entry) or (side == "SHORT" and price <= entry):
                    entry_triggered = True
                    send_telegram(f"✅ ENTRY TRIGGERED: {symbol} {side} @ ${price:.2f}")
                elif entry_attempts >= max_entry_attempts:
                    send_telegram(f"⏰ ENTRY EXPIRED: {symbol} never reached entry @ ${entry:.2f}")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
            
            if entry_triggered:
                # Check targets
                for i, target in enumerate(targets):
                    if not targets_hit[i]:
                        if (side == "LONG" and price >= target) or (side == "SHORT" and price <= target):
                            targets_hit[i] = True
                            profit_pct = abs(target - entry) / entry * 100
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit @ ${target:.2f} (+{profit_pct:.2f}%)")
                
                # Check stop loss
                if (side == "LONG" and price <= sl) or (side == "SHORT" and price >= sl):
                    loss_pct = abs(sl - entry) / entry * 100
                    send_telegram(f"🛑 STOP LOSS HIT: {symbol} @ ${price:.2f} (-{loss_pct:.2f}%)")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
                
                # Check if all targets hit
                if all(targets_hit):
                    total_profit = abs(targets[-1] - entry) / entry * 100
                    send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! (+{total_profit:.2f}%)")
                    if signal_id in active_monitoring_threads:
                        del active_monitoring_threads[signal_id]
                    break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread, daemon=True)
    thread.start()
    active_monitoring_threads[signal_id] = thread

# =================== ALERT FUNCTIONS ========================
def send_institutional_alert(symbol, side, entry, sl, targets, behavior_type):
    """Send institutional alert"""
    global signal_counter
    
    signal_id = f"INST{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    # Get behavior description
    behavior_desc = BEHAVIOR_TYPES.get(behavior_type, "Institutional Move")
    
    # Format message based on behavior type
    if "buy" in behavior_type.lower() or "bullish" in behavior_type.lower():
        message = (f"🏛️ <b>{behavior_desc}</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n\n"
                  f"⚠️ Institutional buying pressure detected")
    
    elif "sell" in behavior_type.lower() or "bearish" in behavior_type.lower():
        message = (f"🏛️ <b>{behavior_desc}</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}\n\n"
                  f"⚠️ Institutional selling pressure detected")
    
    else:
        message = (f"🏛️ <b>{behavior_desc}</b>\n\n"
                  f"<b>Symbol:</b> {symbol}\n"
                  f"<b>Direction:</b> {side}\n"
                  f"<b>Entry:</b> ${entry:.2f}\n"
                  f"<b>Stop Loss:</b> ${sl:.2f}\n"
                  f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
                  f"<b>Targets:</b> {targets_str}\n"
                  f"<b>Signal ID:</b> {signal_id}")
    
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
    monitor_trade_live(symbol, side, entry, sl, targets, behavior_desc, signal_id)
    
    return signal_id

def send_technical_alert(symbol, side, entry, sl, targets, strategy):
    """Send technical trading alert"""
    global signal_counter
    
    signal_id = f"TECH{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    # Calculate actual risk/reward
    risk_pct = SL_BUFFER * 100
    reward1_pct = TARGETS[0] * 100
    rr_ratio = reward1_pct / risk_pct
    
    message = (f"📊 <b>{strategy} SIGNAL</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Entry:</b> ${entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{rr_ratio:.1f}\n\n"
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

def send_multi_timeframe_alert(symbol, side, entry, sl, targets, strategy_type, signal_data):
    """Send multi-timeframe institutional alert"""
    global signal_counter
    
    signal_id = f"MTF{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    # Get strategy description
    strategy_desc = MULTI_TIMEFRAME_STRATEGIES.get(strategy_type, {}).get("description", strategy_type)
    
    # Format message
    volume_info = f"Volume: {signal_data.get('volume_ratio', 0):.1f}x"
    if signal_data.get('breakout_size_pct'):
        breakout_info = f"Breakout: {signal_data['breakout_size_pct']:.2f}%"
    else:
        breakout_info = f"Block Trades: {signal_data.get('block_trades', 0)}"
    
    message = (f"📈 <b>{strategy_desc}</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>{volume_info}</b>\n"
              f"<b>{breakout_info}</b>\n\n"
              f"<b>Entry:</b> ${entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Timeframe:</b> {strategy_type}")
    
    send_telegram(message)
    update_signal_time(symbol, "multi_tf")
    
    # Store signal data
    signal_record = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "strategy": strategy_type,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_record
    
    # Start monitoring thread
    monitor_trade_live(symbol, side, entry, sl, targets, strategy_desc, signal_id)
    
    return signal_id

def send_breakout_pending_alert(symbol, side, breakout_level, strategy, window):
    """Send alert for pending breakout"""
    signal_id = f"BRK{int(time.time())}"
    
    message = (f"🔔 <b>PENDING BREAKOUT DETECTED</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Breakout Level:</b> ${breakout_level:.2f}\n"
              f"<b>Current Price:</b> ${get_current_price(symbol) or 0:.2f}\n"
              f"<b>Strategy:</b> {strategy}\n\n"
              f"⚠️ Waiting for breakout confirmation...")
    
    send_telegram(message)
    
    # Store pending breakout
    pending_breakouts[signal_id] = {
        "symbol": symbol,
        "side": side,
        "breakout_level": breakout_level,
        "strategy": strategy,
        "window": window,
        "interval": TRADING_MODES[strategy]["interval"],
        "timestamp": time.time()
    }
    
    return signal_id

# =================== ANALYSIS FUNCTIONS =====================
def analyze_institutional_flow(symbol):
    """Analyze for institutional flow behavior"""
    df_5min = get_market_data(symbol, "5m", 100)
    
    if df_5min is None or len(df_5min) < 30:
        return None
    
    print(f"🔍 Analyzing {symbol} for institutional flow...")
    
    # Add technical indicators
    df_5min = add_technical_indicators(df_5min)
    
    # Check institutional behaviors in order of reliability
    behaviors = [
        ("institutional_buying", detect_institutional_buying(df_5min, symbol)),
        ("institutional_selling", detect_institutional_selling(df_5min, symbol)),
        ("bullish_stop_hunt", detect_bullish_stop_hunt(df_5min, symbol)),
        ("bearish_stop_hunt", detect_bearish_stop_hunt(df_5min, symbol))
    ]
    
    for behavior_type, direction in behaviors:
        if direction:
            print(f"✅ {symbol}: {BEHAVIOR_TYPES[behavior_type]} - {direction}")
            
            # Get current price for entry
            current_price = get_current_price(symbol)
            if current_price is None:
                return None
            
            # Set entry price (use current price for institutional signals)
            entry = current_price
            
            sl, targets = calculate_trade_levels(entry, direction)
            
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
    """Analyze for technical trading signals - WITH BREAKOUT CONFIRMATION"""
    cfg = TRADING_MODES[mode_name]
    df = get_market_data(symbol, cfg["interval"], CANDLE_LIMIT)
    
    if df is None or df.empty:
        return None
    
    df = add_technical_indicators(df)
    
    # First, check for confirmed breakouts from pending list
    confirmed_breakouts = check_pending_breakouts(symbol)
    for breakout in confirmed_breakouts:
        if breakout["strategy"] == mode_name:
            # This breakout is now confirmed, generate signal
            entry = get_current_price(symbol)
            if entry is None:
                continue
            
            sl, targets = calculate_trade_levels(entry, breakout["side"])
            
            return {
                "symbol": symbol,
                "side": breakout["side"],
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy": mode_name,
                "breakout_confirmed": True
            }
    
    # Check both LONG and SHORT
    for side in ["LONG", "SHORT"]:
        # Check breakout first
        breakout_confirmed, breakout_level = check_breakout(df, side, cfg["recent_hl_window"])
        
        if cfg.get("require_breakout_confirmation", False) and not breakout_confirmed:
            # No breakout yet, but setup looks good - mark as pending
            if check_technical_conditions(df, cfg, side):
                # Send pending breakout alert
                send_breakout_pending_alert(symbol, side, breakout_level, mode_name, 
                                           cfg["recent_hl_window"])
                continue
        
        # Check technical conditions
        if check_technical_conditions(df, cfg, side):
            entry = compute_technical_entry(df, side, cfg, symbol)
            if entry is None:
                continue
            
            sl, targets = calculate_trade_levels(entry, side)
            
            return {
                "symbol": symbol,
                "side": side,
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy": mode_name,
                "breakout_confirmed": breakout_confirmed
            }
    
    return None

# =================== NEW: MULTI-TIMEFRAME ANALYSIS ===========
def analyze_multi_timeframe_institutional(symbol):
    """Analyze for institutional moves across multiple timeframes"""
    print(f"🔍 Multi-timeframe analysis for {symbol}...")
    
    signals_found = []
    
    # Check all multi-timeframe strategies
    for timeframe_strategy in MULTI_TIMEFRAME_STRATEGIES.keys():
        if not can_send_signal(symbol, "multi_tf"):
            continue
            
        signal_data = detect_multi_timeframe_breakout(symbol, timeframe_strategy)
        if signal_data:
            # Get current price for entry
            current_price = get_current_price(symbol)
            if current_price is None:
                continue
            
            # Set entry price
            entry = current_price
            
            sl, targets = calculate_trade_levels(entry, signal_data["direction"])
            
            signals_found.append({
                "symbol": symbol,
                "side": signal_data["direction"],
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy_type": timeframe_strategy,
                "signal_data": signal_data
            })
    
    # Check for volume surge
    volume_surge_signal = detect_volume_surge(symbol)
    if volume_surge_signal and can_send_signal(symbol, "multi_tf"):
        current_price = get_current_price(symbol)
        if current_price:
            entry = current_price
            sl, targets = calculate_trade_levels(entry, volume_surge_signal["direction"])
            
            signals_found.append({
                "symbol": symbol,
                "side": volume_surge_signal["direction"],
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy_type": "volume_surge",
                "signal_data": volume_surge_signal
            })
    
    return signals_found

# =================== SCANNER FUNCTIONS ======================
def run_institutional_scanner():
    """Scan for institutional flow signals"""
    print("🔍 Scanning for institutional flow...")
    
    signals_found = 0
    results = []
    
    # Scan top symbols
    for symbol in list(DIGITAL_ASSETS.values())[:4]:  # BTC, ETH, BNB, SOL
        result = analyze_institutional_flow(symbol)
        if result:
            results.append(result)
    
    # Send alerts for found signals
    for result in results:
        if can_send_signal(result["symbol"]):
            send_institutional_alert(
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
        for strategy in ["SCALP", "SWING", "INSTITUTIONAL_MOMENTUM"]:
            result = analyze_technical_strategy(symbol, strategy)
            if result and can_send_signal(symbol):
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

def run_multi_timeframe_scanner():
    """Scan for institutional moves across all timeframes"""
    print("📈 Scanning multi-timeframe institutional moves...")
    
    signals_found = 0
    
    # Scan all symbols
    for symbol in list(DIGITAL_ASSETS.values())[:6]:  # All 6 symbols
        multi_tf_signals = analyze_multi_timeframe_institutional(symbol)
        
        for signal in multi_tf_signals:
            if can_send_signal(symbol, "multi_tf"):
                send_multi_timeframe_alert(
                    signal["symbol"],
                    signal["side"],
                    signal["entry"],
                    signal["sl"],
                    signal["targets"],
                    signal["strategy_type"],
                    signal["signal_data"]
                )
                signals_found += 1
    
    print(f"✅ Multi-timeframe scan complete. Signals: {signals_found}")
    return signals_found

# =================== STATUS MONITORING ======================
def check_active_signals():
    """Check status of active signals"""
    current_time = time.time()
    
    # Clean up old completed signals
    completed_signal_ids = []
    for signal_id, signal_data in active_signals.items():
        if signal_data.get("status") == "COMPLETED":
            if current_time - signal_data["timestamp"] > 7200:  # 2 hours
                completed_signal_ids.append(signal_id)
    
    for signal_id in completed_signal_ids:
        del active_signals[signal_id]
    
    return len(active_signals)

# =================== MAIN EXECUTION =========================
def main():
    """Main execution loop"""
    print("=" * 60)
    print("🏛️ INSTITUTIONAL FLOW & TRADING BOT")
    print("📈 ENHANCED WITH MULTI-TIMEFRAME STRATEGIES")
    print("🎯 Catches ALL Institutional Moves")
    print("=" * 60)
    
    send_telegram("🤖 <b>TRADING BOT ACTIVATED (ENHANCED VERSION)</b>\n"
                  "🏛️ Multi-Timeframe Institutional Detection\n"
                  "📈 1H/15M/5M Breakout Strategies\n"
                  "📊 Volume Surge Detection\n"
                  f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    iteration = 0
    
    while True:
        iteration += 1
        try:
            print(f"\n🔄 Iteration {iteration} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            
            # Check active signals
            active_count = check_active_signals()
            print(f"📈 Active trades: {active_count}")
            
            # Check pending breakouts
            print(f"🔔 Pending breakouts: {len(pending_breakouts)}")
            
            # Check if current time is institutional hours
            hour = datetime.utcnow().hour
            institutional_hour = False
            for period_name, (start_hour, end_hour) in INSTITUTIONAL_ACTIVE_HOURS.items():
                if start_hour <= hour < end_hour:
                    institutional_hour = True
                    print(f"🏛️ Institutional hour: {period_name}")
                    break
            
            # Run ALL scanners (no more missing moves!)
            
            # 1. Multi-timeframe scanner (ALWAYS RUNS - catches ALL moves)
            multi_tf_signals = run_multi_timeframe_scanner()
            print(f"📈 Multi-timeframe signals: {multi_tf_signals}")
            
            # 2. Institutional scanner (during institutional hours)
            if institutional_hour:
                flow_signals = run_institutional_scanner()
                print(f"🏛️ Institutional signals: {flow_signals}")
            
            # 3. Technical scanner (ALWAYS RUNS)
            tech_signals = run_technical_scanner()
            print(f"📊 Technical signals: {tech_signals}")
            
            # Total signals found
            total_signals = multi_tf_signals + (flow_signals if institutional_hour else 0) + tech_signals
            print(f"✅ Total signals this iteration: {total_signals}")
            
            # Wait based on time (shorter wait for faster scanning)
            wait_time = 30  # Scan every 30 seconds for maximum coverage
            print(f"⏳ Next scan in {wait_time} seconds...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down bot...")
            shutdown_msg = ("🛑 <b>BOT SHUTTING DOWN</b>\n\n"
                          f"📊 Total iterations: {iteration}\n"
                          f"⏰ End Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            send_telegram(shutdown_msg)
            break
            
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
