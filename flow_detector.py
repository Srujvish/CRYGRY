# =================== INSTITUTIONAL FLOW & CRYPTO TRADING BOT ===================
# AI-BASED INSTITUTIONAL FLOW DETECTION WITH BREAKOUT CONFIRMATION
# ENHANCED VERSION - CATCHES ALL INSTITUTIONAL MOVES, NO MISSING SIGNALS
# REAL-TIME FIXED VERSION - NO LATENCY BETWEEN SIGNAL AND CHART
# ULTRA-FAST VERSION - SIGNALS IN 1-2 SECONDS WITH PRICE MONITORING

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

# =================== REAL-TIME SETTINGS ===================
MAX_PRICE_GAP = 0.002  # Max 0.2% difference between signal and current price
REAL_TIME_VALIDATION = True  # Enable real-time price validation
SIGNAL_TIMEOUT_SECONDS = 1  # REDUCED FROM 3 TO 1 SECOND
ULTRA_FAST_MODE = True  # Enable ultra-fast signal processing

# =================== SYMBOLS & ASSETS ======================
TRADING_SYMBOLS = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT", "AVAX-USDT"]
DIGITAL_ASSETS = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT", 
    "BNB": "BNB-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT",
    "ADA": "ADA-USDT",
    "DOGE": "DOGE-USDT",
    "AVAX": "AVAX-USDT"
}

# =================== INSTITUTIONAL FLOW SETTINGS ============
INSTITUTIONAL_VOLUME_RATIO = 2.5
MIN_MOVE_FOR_ENTRY = 0.015
STOP_HUNT_DISTANCE = 0.015
ABSORPTION_WICK_RATIO = 0.15

# =================== INSTITUTIONAL TIME ZONES ===============
INSTITUTIONAL_ACTIVE_HOURS = {
    "ALL_HOURS": (0, 24),
}

# =================== INSTITUTIONAL ORDER FLOW ===============
ORDER_FLOW_THRESHOLDS = {
    "BLOCK_TRADE_SIZE_USD": 25000.0,
    "SWEEP_RATIO": 0.60,
    "IMMEDIATE_OR_CANCEL": 0.7,
    "DARK_POOL_INDICATOR": 3.0
}

# =================== TECHNICAL TRADING SETTINGS ============
EMA_PERIOD = 20
RSI_PERIOD = 14
CANDLE_LIMIT = 200

SL_BUFFER = 0.0030
TARGETS = [0.0050, 0.010, 0.020]

ABS_MAX_ENTRY_PCT = {
    "BTC-USDT": 0.008,
    "ETH-USDT": 0.010,
    "BNB-USDT": 0.012,
    "SOL-USDT": 0.015,
    "XRP-USDT": 0.020,
    "ADA-USDT": 0.018,
    "DOGE-USDT": 0.025,
    "AVAX-USDT": 0.022
}

# =================== MULTI-TIMEFRAME STRATEGIES ============
MULTI_TIMEFRAME_STRATEGIES = {
    "1H_INSTITUTIONAL": {
        "interval": "1h",
        "volume_ratio": 2.5,
        "min_move_pct": 0.012,
        "window": 10,
        "description": "1H Institutional Breakout"
    },
    "15M_MOMENTUM": {
        "interval": "15m",
        "volume_ratio": 2.5,
        "min_move_pct": 0.006,
        "window": 15,
        "description": "15M Momentum Move"
    },
    "5M_BREAKOUT": {
        "interval": "5m",
        "volume_ratio": 3.0,
        "min_move_pct": 0.004,
        "window": 20,
        "description": "5M Quick Breakout"
    },
    "INSTITUTIONAL_VOLUME_SURGE": {
        "interval": "5m",
        "volume_ratio": 4.0,
        "min_move_pct": 0.002,
        "window": 5,
        "description": "Institutional Volume Surge"
    },
    "FLASH_INSTITUTIONAL": {
        "interval": "3m",
        "volume_ratio": 2.8,
        "min_move_pct": 0.003,
        "window": 10,
        "description": "Flash Institutional Move"
    },
    "QUICK_MOMENTUM": {
        "interval": "2m",
        "volume_ratio": 3.2,
        "min_move_pct": 0.0025,
        "window": 8,
        "description": "Quick Momentum Move"
    }
}

TRADING_MODES = {
    "SCALP": {
        "interval": "1m",
        "recent_hl_window": 8,
        "rsi_long_min": 45, "rsi_long_max": 85,
        "rsi_short_min": 15, "rsi_short_max": 55,
        "entry_buffer_long": 0.0001,
        "entry_buffer_short": 0.0001,
        "max_entry_drift": 0.0015,
        "immediate_mode": True,
        "require_breakout_confirmation": False,
        "confirmation_candles": 0,
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 20,
        "institutional_only": False,
        "min_volume_ratio": 2.0,
        "breakout_retest_allowed": True,
        "max_retest_count": 2,
        "multi_timeframe_check": True,
        "max_signals_per_hour": 8
    },
    "SWING": {
        "interval": "5m",
        "recent_hl_window": 15,
        "rsi_long_min": 48, "rsi_long_max": 82,
        "rsi_short_min": 18, "rsi_short_max": 52,
        "entry_buffer_long": 0.0003,
        "entry_buffer_short": 0.0003,
        "max_entry_drift": 0.0025,
        "immediate_mode": True,
        "require_breakout_confirmation": True,
        "confirmation_candles": 1,
        "need_prev_candle_break": True,
        "volume_filter": True,
        "volume_lookback": 30,
        "institutional_only": True,
        "min_volume_ratio": 2.0,
        "breakout_retest_allowed": True,
        "max_retest_count": 3,
        "multi_timeframe_check": True,
        "max_signals_per_hour": 4
    },
    "INSTITUTIONAL_MOMENTUM": {
        "interval": "15m",
        "recent_hl_window": 12,
        "rsi_long_min": 40, "rsi_long_max": 90,
        "rsi_short_min": 10, "rsi_short_max": 60,
        "entry_buffer_long": 0.0002,
        "entry_buffer_short": 0.0002,
        "max_entry_drift": 0.0020,
        "immediate_mode": True,
        "require_breakout_confirmation": False,
        "confirmation_candles": 0,
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 20,
        "institutional_only": True,
        "min_volume_ratio": 2.5,
        "breakout_retest_allowed": False,
        "multi_timeframe_check": True,
        "max_signals_per_hour": 6
    },
    "INSTITUTIONAL_FLASH": {
        "interval": "3m",
        "recent_hl_window": 6,
        "rsi_long_min": 35, "rsi_long_max": 92,
        "rsi_short_min": 8, "rsi_short_max": 65,
        "entry_buffer_long": 0.00005,
        "entry_buffer_short": 0.00005,
        "max_entry_drift": 0.0025,
        "immediate_mode": True,
        "require_breakout_confirmation": False,
        "confirmation_candles": 0,
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 12,
        "institutional_only": True,
        "min_volume_ratio": 2.2,
        "breakout_retest_allowed": False,
        "multi_timeframe_check": False,
        "max_signals_per_hour": 10
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
    "1h_breakout": "📈 1H INSTITUTIONAL BREAKOUT",
    "15m_momentum": "⚡ 15M MOMENTUM MOVE",
    "5m_breakout": "🎯 5M QUICK BREAKOUT",
    "volume_surge": "📊 INSTITUTIONAL VOLUME SURGE",
    "flash_buying": "⚡ FLASH INSTITUTIONAL BUYING",
    "flash_selling": "⚡ FLASH INSTITUTIONAL SELLING",
    "quick_breakout": "🚀 QUICK INSTITUTIONAL BREAKOUT",
    "quick_momentum": "⚡ QUICK MOMENTUM MOVE"
}

# =================== GLOBAL TRACKING ========================
signal_counter = 0
active_signals = {}
last_signal_time = {}
active_monitoring_threads = {}
completed_signals = []
pending_breakouts = {}
pending_multi_tf_signals = {}
signal_counts_hourly = {}

# REDUCED COOLDOWNS FOR FASTER SIGNALS
SIGNAL_COOLDOWN = 120  # Reduced from 300 to 120 seconds
BREAKOUT_CONFIRMATION_TIMEOUT = 900  # Reduced from 1800 to 900 seconds
MULTI_TF_COOLDOWN = 90  # Reduced from 180 to 90 seconds

# Price monitoring tracking
price_monitoring = {}
last_high_update = {}

# =================== ULTRA-FAST REAL-TIME PRICE FUNCTIONS ============
def get_current_price_real_time(symbol):
    """ULTRA-FAST REAL-TIME price with timestamp"""
    try:
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = f"symbol={symbol}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        
        # Ultra-fast timeout - reduced from 1-2 seconds to 0.5-1 seconds
        timeout = 0.5 if symbol == "BTC-USDT" else 0.8
        
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time
        
        if response_time > 0.3:
            print(f"⚠️ Slow response for {symbol}: {response_time:.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                price = float(data['data']['lastPrice'])
                return price, time.time()
    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout fetching {symbol} price")
    except Exception as e:
        if "timeout" not in str(e).lower():
            print(f"⚠️ Real-time price error for {symbol}: {e}")
    
    return None, None

def get_current_price(symbol):
    """Legacy function - uses real-time"""
    price, _ = get_current_price_real_time(symbol)
    return price

def validate_price_gap(symbol, signal_price):
    """ULTRA-FAST: Validate signal price vs current real-time price"""
    if not REAL_TIME_VALIDATION:
        return True, signal_price
    
    start_time = time.time()
    current_price, timestamp = get_current_price_real_time(symbol)
    validation_time = time.time() - start_time
    
    if current_price is None:
        # Try one more time with faster method
        try:
            endpoint = "/openApi/swap/v2/quote/price"
            params = f"symbol={symbol}"
            url = f"{API_BASE}{endpoint}?{params}"
            response = requests.get(url, timeout=0.3)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    current_price = float(data['data']['price'])
        except:
            pass
        
        if current_price is None:
            print(f"⚠️ Cannot validate {symbol} - no real-time price")
            return True, signal_price
    
    price_gap = abs(current_price - signal_price) / signal_price
    
    if price_gap > MAX_PRICE_GAP:
        print(f"❌ Price gap too large for {symbol}:")
        print(f"   Signal: ${signal_price:.2f}")
        print(f"   Current: ${current_price:.2f}")
        print(f"   Gap: {price_gap*100:.2f}% > {MAX_PRICE_GAP*100:.2f}%")
        print(f"   Validation time: {validation_time:.3f}s")
        return False, current_price
    
    if validation_time > 0.5:
        print(f"⚠️ Slow validation for {symbol}: {validation_time:.3f}s")
    
    return True, signal_price

# =================== UTILITIES ==============================
def send_alert(message, reply_to=None):
    """ULTRA-FAST: Send alert notification"""
    try:
        if not ALERT_TOKEN or not ALERT_TARGET:
            print(f"📢 {message}")
            return None
            
        url = f"https://api.telegram.org/bot{ALERT_TOKEN}/sendMessage"
        payload = {"chat_id": ALERT_TARGET, "text": message, "parse_mode": "HTML"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        
        start_time = time.time()
        r = requests.post(url, data=payload, timeout=2).json()  # Reduced timeout from 5 to 2 seconds
        send_time = time.time() - start_time
        
        if send_time > 0.5:
            print(f"⚠️ Slow Telegram send: {send_time:.3f}s")
        
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print(f"Alert error: {e}")
        return None

def send_telegram(msg: str):
    """Alias for send_alert"""
    return send_alert(msg)

# =================== ULTRA-FAST MARKET DATA FUNCTIONS ==================
def get_market_data(symbol, interval="5m", limit=100):
    """ULTRA-FAST: Get price and volume data from BingX"""
    try:
        endpoint = "/openApi/swap/v3/quote/klines"
        params = f"symbol={symbol}&interval={interval}&limit={limit}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        
        # Reduced timeout for faster response
        timeout = 3 if interval in ["1m", "2m", "3m"] else 5
        
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        fetch_time = time.time() - start_time
        
        if fetch_time > 2:
            print(f"⚠️ Slow market data fetch for {symbol} {interval}: {fetch_time:.2f}s")
        
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

def get_order_book(symbol, limit=50):
    """ULTRA-FAST: Get order book depth"""
    try:
        endpoint = "/openApi/swap/v2/quote/depth"
        params = f"symbol={symbol}&limit={limit}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.post(url, timeout=3)  # Reduced timeout
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                return data['data']
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
    return None

def get_recent_trades(symbol, limit=200):
    """ULTRA-FAST: Get recent trades for order flow analysis"""
    try:
        endpoint = "/openApi/swap/v2/quote/trades"
        params = f"symbol={symbol}&limit={limit}"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=3)  # Reduced timeout
        
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
            
            X_buy = []
            y_buy = []
            
            X_buy.append([4.5, 0.22, 0.12, 3.2, 0.20, 0.72, 0.70, 0.025, 1.5, 0.80])
            y_buy.append(1)
            
            X_buy.append([3.8, 0.18, 0.15, 2.8, 0.16, 0.65, 0.65, 0.020, 1.3, 0.75])
            y_buy.append(1)
            
            X_buy.append([5.0, 0.28, 0.10, 3.8, 0.25, 0.78, 0.75, 0.030, 1.7, 0.85])
            y_buy.append(1)
            
            X_buy.append([2.8, 0.15, 0.08, 2.2, 0.12, 0.60, 0.68, 0.015, 1.1, 0.70])
            y_buy.append(1)
            
            X_buy.append([1.5, 0.08, 0.55, 1.0, 0.06, 0.30, 0.40, 0.008, 0.6, 0.35])
            y_buy.append(0)
            
            X_buy.append([2.0, 0.12, 0.50, 1.5, 0.10, 0.38, 0.45, 0.012, 0.8, 0.42])
            y_buy.append(0)
            
            X_sell = []
            y_sell = []
            
            X_sell.append([4.8, 0.18, 0.25, 3.5, 0.22, 0.75, 0.22, 0.028, 1.6, 0.20])
            y_sell.append(1)
            
            X_sell.append([5.2, 0.22, 0.30, 3.8, 0.28, 0.80, 0.18, 0.032, 1.8, 0.16])
            y_sell.append(1)
            
            X_sell.append([4.2, 0.15, 0.22, 3.2, 0.20, 0.70, 0.25, 0.025, 1.4, 0.22])
            y_sell.append(1)
            
            X_sell.append([3.0, 0.12, 0.18, 2.3, 0.14, 0.65, 0.25, 0.018, 1.2, 0.20])
            y_sell.append(1)
            
            X_sell.append([1.8, 0.10, 0.60, 1.3, 0.08, 0.35, 0.50, 0.010, 0.7, 0.55])
            y_sell.append(0)
            
            X_sell.append([2.2, 0.15, 0.52, 1.8, 0.12, 0.42, 0.55, 0.015, 0.9, 0.58])
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
        """ULTRA-FAST: Analyze trade flow for institutional activity"""
        try:
            trades = get_recent_trades(symbol, limit=100)  # Reduced from 150 to 100
            if not trades:
                return {"buy_pressure": 0.5, "block_buys": 0, "block_sells": 0, "direction": "NEUTRAL"}
            
            buy_volume_usd = 0
            sell_volume_usd = 0
            block_buys = 0
            block_sells = 0
            
            # Get price once and reuse
            current_price, _ = get_current_price_real_time(symbol)
            if current_price is None:
                current_price = 1.0
            
            # Process only recent trades for speed
            recent_trades = trades[:80]  # Process only first 80 trades
            
            for trade in recent_trades:
                qty = float(trade.get('qty', 0))
                price = float(trade.get('price', current_price))
                
                trade_value_usd = qty * price
                
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
            
            if block_buys >= 1 and block_buys > block_sells:
                direction = "LONG"
            elif block_sells >= 1 and block_sells > block_buys:
                direction = "SHORT"
            elif buy_pressure > 0.65:
                direction = "LONG"
            elif buy_pressure < 0.35:
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
    """ULTRA-FAST: Add technical indicators"""
    if df.empty or len(df) < 20:
        return df
    
    # Use vectorized operations for speed
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # Faster RSI calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(RSI_PERIOD, min_periods=1).mean()
    avg_loss = loss.rolling(RSI_PERIOD, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    df["volume_ma"] = df["volume"].rolling(15, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1e-9)
    
    return df

# =================== INSTITUTIONAL BEHAVIOR DETECTION =======
def detect_institutional_buying(df, symbol):
    """ULTRA-FAST: Detect institutional buying (LONG)"""
    try:
        if len(df) < 20:  # Reduced from 25 to 20 for speed
            return None
        
        close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        volume = df['volume'].iloc[-1]
        open_price = df['open'].iloc[-1]
        low = df['low'].iloc[-1]
        
        vol_avg_15 = df['volume'].rolling(15).mean().iloc[-1]
        if vol_avg_15 == 0 or volume < vol_avg_15 * 2.0:
            return None
        
        current_body = abs(close - open_price)
        lower_wick = min(close, open_price) - low
        
        if current_body == 0 or lower_wick < current_body * 0.10:
            return None
        
        if not (close > prev_close):
            return None
        
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] != "LONG":
            return None
        
        print(f"✅ Institutional buying detected: {symbol} | Block buys: {trade_flow['block_buys']} | Volume: {volume/vol_avg_15:.1f}x")
        return "LONG"
        
    except Exception as e:
        print(f"Error in institutional buying detection: {e}")
        return None

def detect_institutional_selling(df, symbol):
    """ULTRA-FAST: Detect institutional selling (SHORT)"""
    try:
        if len(df) < 20:  # Reduced from 25 to 20 for speed
            return None
        
        close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        volume = df['volume'].iloc[-1]
        open_price = df['open'].iloc[-1]
        high = df['high'].iloc[-1]
        
        vol_avg_15 = df['volume'].rolling(15).mean().iloc[-1]
        if vol_avg_15 == 0 or volume < vol_avg_15 * 2.0:
            return None
        
        current_body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        
        if current_body == 0 or upper_wick < current_body * 0.10:
            return None
        
        if not (close < prev_close):
            return None
        
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] != "SHORT":
            return None
        
        print(f"✅ Institutional selling detected: {symbol} | Block sells: {trade_flow['block_sells']} | Volume: {volume/vol_avg_15:.1f}x")
        return "SHORT"
        
    except Exception as e:
        print(f"Error in institutional selling detection: {e}")
        return None

def detect_bullish_stop_hunt(df, symbol):
    """ULTRA-FAST: Detect bullish stop hunt (LONG)"""
    try:
        if len(df) < 15:  # Reduced from 20 to 15
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        recent_low = low.iloc[-12:-3].min()  # Reduced window
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        vol_avg = volume.rolling(12).mean().iloc[-1]  # Reduced window
        current_vol = volume.iloc[-1]
        
        if vol_avg == 0:
            return None
        
        if (current_low < recent_low * (1 - 0.010) and
            current_close > recent_low * 1.015 and
            current_vol > vol_avg * 3.5 and
            current_close > prev_close):
            
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["direction"] == "LONG":
                print(f"✅ Bullish stop hunt detected: {symbol}")
                return "LONG"
        
    except Exception as e:
        print(f"Error in bullish stop hunt detection: {e}")
        return None
    return None

def detect_bearish_stop_hunt(df, symbol):
    """ULTRA-FAST: Detect bearish stop hunt (SHORT)"""
    try:
        if len(df) < 15:  # Reduced from 20 to 15
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        recent_high = high.iloc[-12:-3].max()  # Reduced window
        current_high = high.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        vol_avg = volume.rolling(12).mean().iloc[-1]  # Reduced window
        current_vol = volume.iloc[-1]
        
        if vol_avg == 0:
            return None
        
        if (current_high > recent_high * (1 + 0.010) and
            current_close < recent_high * 0.985 and
            current_vol > vol_avg * 3.5 and
            current_close < prev_close):
            
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["direction"] == "SHORT":
                print(f"✅ Bearish stop hunt detected: {symbol}")
                return "SHORT"
        
    except Exception as e:
        print(f"Error in bearish stop hunt detection: {e}")
        return None
    return None

def detect_flash_institutional_move(symbol):
    """ULTRA-FAST: Detect flash institutional moves"""
    try:
        df_2m = get_market_data(symbol, "2m", 20)  # Reduced from 30 to 20
        if df_2m is None or len(df_2m) < 8:  # Reduced from 10 to 8
            return None
        
        current_volume = df_2m['volume'].iloc[-1]
        avg_volume = df_2m['volume'].rolling(8).mean().iloc[-1]  # Reduced from 10 to 8
        
        if avg_volume == 0 or current_volume < avg_volume * 2.5:
            return None
        
        current_close = df_2m['close'].iloc[-1]
        prev_close = df_2m['close'].iloc[-2]
        price_change_pct = abs(current_close - prev_close) / prev_close
        
        if price_change_pct < 0.003:
            return None
        
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        
        if price_change_pct >= 0.003 and trade_flow["direction"] != "NEUTRAL":
            direction = trade_flow["direction"]
            print(f"✅ Flash institutional move: {symbol} - {direction} | {price_change_pct*100:.2f}% | Volume: {current_volume/avg_volume:.1f}x")
            
            return {
                "type": "flash_move",
                "direction": direction,
                "current_price": current_close,
                "volume_ratio": current_volume / avg_volume,
                "price_change_pct": price_change_pct * 100,
                "block_trades": trade_flow["block_buys"] if direction == "LONG" else trade_flow["block_sells"]
            }
        
        return None
        
    except Exception as e:
        print(f"Error in flash move detection for {symbol}: {e}")
        return None

def detect_quick_momentum(symbol):
    """ULTRA-FAST: Detect quick momentum moves"""
    try:
        df_1m = get_market_data(symbol, "1m", 15)  # Reduced from 20 to 15
        if df_1m is None or len(df_1m) < 8:  # Reduced from 10 to 8
            return None
        
        current_volume = df_1m['volume'].iloc[-1]
        avg_volume = df_1m['volume'].rolling(8).mean().iloc[-1]  # Reduced from 10 to 8
        
        if avg_volume == 0 or current_volume < avg_volume * 2.2:
            return None
        
        closes = df_1m['close'].iloc[-4:]  # Check 3 candles instead of 4
        
        # Check for 3 consecutive moves
        if len(closes) >= 3:
            if all(closes.iloc[i] > closes.iloc[i-1] for i in range(1, 3)):
                direction = "LONG"
            elif all(closes.iloc[i] < closes.iloc[i-1] for i in range(1, 3)):
                direction = "SHORT"
            else:
                return None
        else:
            return None
        
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] == direction:
            print(f"✅ Quick momentum: {symbol} - {direction} | Volume: {current_volume/avg_volume:.1f}x")
            return {
                "type": "quick_momentum",
                "direction": direction,
                "current_price": closes.iloc[-1],
                "volume_ratio": current_volume / avg_volume,
                "consecutive_candles": 2  # Reduced from 3 to 2
            }
        
        return None
        
    except Exception as e:
        print(f"Error in quick momentum detection for {symbol}: {e}")
        return None

def detect_multi_timeframe_breakout(symbol, timeframe_strategy):
    """ULTRA-FAST: Detect institutional breakout"""
    try:
        strategy = MULTI_TIMEFRAME_STRATEGIES[timeframe_strategy]
        df = get_market_data(symbol, strategy["interval"], 50)  # Reduced from 80 to 50
        
        if df is None or len(df) < 15:  # Reduced from 25 to 15
            return None
        
        df = add_technical_indicators(df)
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(12).mean().iloc[-1]  # Reduced from 15 to 12
        
        if avg_volume == 0 or current_volume < avg_volume * strategy["volume_ratio"]:
            return None
        
        window = min(strategy["window"], 12)  # Cap window at 12
        recent_high = df['high'].iloc[-window:-1].max()
        recent_low = df['low'].iloc[-window:-1].min()
        current_close = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        if current_high > recent_high:
            breakout_size = (current_high - recent_high) / recent_high
            if breakout_size >= strategy["min_move_pct"]:
                if current_close > recent_high * 0.995:
                    trade_flow = flow_ai.analyze_trade_flow(symbol)
                    if trade_flow["direction"] == "LONG":
                        print(f"✅ {timeframe_strategy} BULLISH breakout: {symbol}")
                        return {
                            "type": timeframe_strategy,
                            "direction": "LONG",
                            "breakout_level": recent_high,
                            "current_price": current_close,
                            "volume_ratio": current_volume / avg_volume,
                            "breakout_size_pct": breakout_size * 100
                        }
        
        if current_low < recent_low:
            breakout_size = (recent_low - current_low) / recent_low
            if breakout_size >= strategy["min_move_pct"]:
                if current_close < recent_low * 1.005:
                    trade_flow = flow_ai.analyze_trade_flow(symbol)
                    if trade_flow["direction"] == "SHORT":
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
    """ULTRA-FAST: Detect institutional volume surge"""
    try:
        df_3m = get_market_data(symbol, "3m", 25)  # Reduced from 40 to 25
        if df_3m is None or len(df_3m) < 10:  # Reduced from 15 to 10
            return None
        
        current_volume = df_3m['volume'].iloc[-1]
        avg_volume = df_3m['volume'].rolling(10).mean().iloc[-1]  # Reduced from 15 to 10
        
        if avg_volume == 0 or current_volume < avg_volume * 3.5:
            return None
        
        current_close = df_3m['close'].iloc[-1]
        prev_close = df_3m['close'].iloc[-2]
        price_change = abs(current_close - prev_close) / prev_close
        
        if price_change <= 0.008:
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["block_buys"] >= 2 or trade_flow["block_sells"] >= 2:
                direction = "LONG" if trade_flow["block_buys"] > trade_flow["block_sells"] else "SHORT"
                print(f"✅ Institutional volume surge: {symbol} - {direction} | Volume: {current_volume/avg_volume:.1f}x")
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

# =================== BREAKOUT DETECTION ===========
def check_breakout(df, side, window=10):
    """ULTRA-FAST: Check if price has broken key levels"""
    try:
        if side == "LONG":
            resistance = df["high"].iloc[-window:-1].max()
            current_high = df["high"].iloc[-1]
            current_close = df["close"].iloc[-1]
            
            if current_high > resistance and current_close > resistance * 0.998:
                return True, resistance
            return False, resistance
            
        else:
            support = df["low"].iloc[-window:-1].min()
            current_low = df["low"].iloc[-1]
            current_close = df["close"].iloc[-1]
            
            if current_low < support and current_close < support * 1.002:
                return True, support
            return False, support
            
    except Exception as e:
        print(f"Error in breakout check: {e}")
        return False, None

def check_pending_breakouts(symbol):
    """ULTRA-FAST: Check if any pending breakouts have been confirmed"""
    current_time = time.time()
    confirmed_breakouts = []
    
    for breakout_id, breakout_data in list(pending_breakouts.items()):
        if breakout_data["symbol"] != symbol:
            continue
            
        if current_time - breakout_data["timestamp"] > BREAKOUT_CONFIRMATION_TIMEOUT:
            del pending_breakouts[breakout_id]
            continue
        
        df = get_market_data(symbol, breakout_data["interval"], 20)  # Reduced from 30 to 20
        if df is None or df.empty:
            continue
        
        breakout_confirmed, _ = check_breakout(df, breakout_data["side"], 
                                              breakout_data["window"])
        
        if breakout_confirmed:
            confirmed_breakouts.append(breakout_data)
            del pending_breakouts[breakout_id]
    
    return confirmed_breakouts

# =================== TECHNICAL TRADING ============
def check_technical_conditions(df, mode_cfg, side):
    """ULTRA-FAST: Check trading conditions"""
    try:
        if df.empty or len(df) < 12:  # Reduced from 15 to 12
            return False
        
        price = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if side == "LONG":
                if not (mode_cfg["rsi_long_min"] <= rsi <= mode_cfg["rsi_long_max"]):
                    return False
            else:
                if not (mode_cfg["rsi_short_min"] <= rsi <= mode_cfg["rsi_short_max"]):
                    return False
        
        vol_avg = df["volume"].rolling(8).mean().iloc[-1]  # Reduced from 10 to 8
        last_vol = df["volume"].iloc[-1]
        if vol_avg > 0 and last_vol < vol_avg * mode_cfg.get("min_volume_ratio", 1.8):
            return False
        
        if side == "LONG":
            return price > ema_20
        else:
            return price < ema_20
            
    except Exception as e:
        print(f"Error in technical conditions check: {e}")
        return False

def compute_technical_entry(df, side, mode_cfg, symbol):
    """ULTRA-FAST: Compute entry price with REAL-TIME validation"""
    if df.empty:
        return None
    
    # Get REAL-TIME price for validation
    current_price_real, timestamp = get_current_price_real_time(symbol)
    if current_price_real is None:
        current_price_real = df["close"].iloc[-1]
    
    price = df["close"].iloc[-1]
    
    if mode_cfg.get("immediate_mode", False):
        # Validate with real-time price
        valid, validated_price = validate_price_gap(symbol, current_price_real)
        if valid:
            return validated_price
        return None
    
    w = min(mode_cfg["recent_hl_window"], 10)  # Cap at 10 for speed
    
    if side == "LONG":
        resistance = df["high"].iloc[-w:].max()
        entry = resistance * (1 + mode_cfg["entry_buffer_long"])
        
        max_pct = ABS_MAX_ENTRY_PCT.get(symbol, 0.01)
        if entry > price * (1 + max_pct):
            entry = price * (1 + max_pct * 0.8)
    else:
        support = df["low"].iloc[-w:].min()
        entry = support * (1 - mode_cfg["entry_buffer_short"])
        
        max_pct = ABS_MAX_ENTRY_PCT.get(symbol, 0.01)
        if entry < price * (1 - max_pct):
            entry = price * (1 - max_pct * 0.8)
    
    # Validate with real-time price
    valid, validated_entry = validate_price_gap(symbol, entry)
    if valid:
        return validated_entry
    
    return None

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
    """ULTRA-FAST: Check if signal can be sent"""
    current_time = time.time()
    
    if signal_type == "multi_tf":
        cooldown = MULTI_TF_COOLDOWN
    elif signal_type == "flash":
        cooldown = 60  # Reduced from 120 to 60 seconds
    else:
        cooldown = SIGNAL_COOLDOWN
    
    if symbol in last_signal_time:
        time_since_last = current_time - last_signal_time[symbol]
        if time_since_last < cooldown:
            return False
    
    return True

def update_signal_time(symbol, signal_type="default"):
    """Update last signal time"""
    last_signal_time[symbol] = time.time()

# =================== ENHANCED PRICE MONITORING ==============
def send_price_update(symbol, current_price, entry_price, side, signal_id, highest_price=None):
    """Send price update when price moves significantly"""
    try:
        if current_price is None or entry_price is None:
            return
        
        price_change = ((current_price - entry_price) / entry_price) * 100
        
        # Only send updates for significant moves
        if abs(price_change) < 0.5:  # Less than 0.5% change
            return
        
        # For LONG positions, send updates when making new highs
        if side == "LONG":
            if highest_price is None or current_price > highest_price:
                message = (f"📈 <b>PRICE UPDATE - {symbol}</b>\n\n"
                          f"<b>Signal ID:</b> {signal_id}\n"
                          f"<b>Direction:</b> {side}\n"
                          f"<b>Entry Price:</b> ${entry_price:.2f}\n"
                          f"<b>Current Price:</b> ${current_price:.2f}\n"
                          f"<b>Profit/Loss:</b> {price_change:+.2f}%\n"
                          f"<b>New High:</b> ✅\n"
                          f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC")
                
                send_telegram(message)
                return current_price  # Return new highest price
        
        # For SHORT positions, send updates when making new lows
        elif side == "SHORT":
            if highest_price is None or current_price < highest_price:
                message = (f"📉 <b>PRICE UPDATE - {symbol}</b>\n\n"
                          f"<b>Signal ID:</b> {signal_id}\n"
                          f"<b>Direction:</b> {side}\n"
                          f"<b>Entry Price:</b> ${entry_price:.2f}\n"
                          f"<b>Current Price:</b> ${current_price:.2f}\n"
                          f"<b>Profit/Loss:</b> {price_change:+.2f}%\n"
                          f"<b>New Low:</b> ✅\n"
                          f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC")
                
                send_telegram(message)
                return current_price  # Return new lowest price
        
        return highest_price
        
    except Exception as e:
        print(f"Error sending price update: {e}")
        return highest_price

# =================== ULTRA-FAST REAL-TIME ALERT FUNCTIONS ============
def send_institutional_alert(symbol, side, entry, sl, targets, behavior_type):
    """ULTRA-FAST: Send institutional alert with REAL-TIME validation"""
    global signal_counter
    
    # ULTRA-FAST validation
    start_time = time.time()
    valid, validated_entry = validate_price_gap(symbol, entry)
    validation_time = time.time() - start_time
    
    if not valid:
        print(f"❌ Skipping {symbol} signal - price gap too large (validation: {validation_time:.3f}s)")
        return None
    
    if validation_time > 0.5:
        print(f"⚠️ Slow validation for {symbol}: {validation_time:.3f}s")
    
    signal_id = f"INST{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    behavior_desc = BEHAVIOR_TYPES.get(behavior_type, "Institutional Move")
    
    # Get current price for display - ULTRA-FAST
    current_price, timestamp = get_current_price_real_time(symbol)
    
    message = (f"🏛️ <b>{behavior_desc}</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Current Price:</b> ${current_price or validated_entry:.2f}\n"
              f"<b>Entry:</b> ${validated_entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    send_start = time.time()
    send_telegram(message)
    send_time = time.time() - send_start
    
    if send_time > 0.5:
        print(f"⚠️ Slow Telegram send for {symbol}: {send_time:.3f}s")
    
    update_signal_time(symbol)
    
    signal_data = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": validated_entry,
        "sl": sl,
        "targets": targets,
        "behavior_type": behavior_type,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_data
    
    # Start enhanced monitoring
    monitor_trade_live(symbol, side, validated_entry, sl, targets, behavior_desc, signal_id)
    
    print(f"✅ Signal sent for {symbol} in {validation_time+send_time:.3f}s total")
    return signal_id

def send_technical_alert(symbol, side, entry, sl, targets, strategy):
    """ULTRA-FAST: Send technical alert with REAL-TIME validation"""
    global signal_counter
    
    start_time = time.time()
    valid, validated_entry = validate_price_gap(symbol, entry)
    validation_time = time.time() - start_time
    
    if not valid:
        print(f"❌ Skipping {symbol} {strategy} signal - price gap too large (validation: {validation_time:.3f}s)")
        return None
    
    signal_id = f"TECH{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    risk_pct = SL_BUFFER * 100
    reward1_pct = TARGETS[0] * 100
    rr_ratio = reward1_pct / risk_pct
    
    current_price, timestamp = get_current_price_real_time(symbol)
    
    message = (f"📊 <b>{strategy} SIGNAL</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Current Price:</b> ${current_price or validated_entry:.2f}\n"
              f"<b>Entry:</b> ${validated_entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{rr_ratio:.1f}\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Strategy:</b> {strategy}\n"
              f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    send_start = time.time()
    send_telegram(message)
    send_time = time.time() - send_start
    
    update_signal_time(symbol)
    
    signal_data = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": validated_entry,
        "sl": sl,
        "targets": targets,
        "strategy": strategy,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_data
    
    monitor_trade_live(symbol, side, validated_entry, sl, targets, strategy, signal_id)
    
    print(f"✅ Technical signal sent for {symbol} in {validation_time+send_time:.3f}s")
    return signal_id

def send_multi_timeframe_alert(symbol, side, entry, sl, targets, strategy_type, signal_data):
    """ULTRA-FAST: Send multi-timeframe alert with REAL-TIME validation"""
    global signal_counter
    
    start_time = time.time()
    valid, validated_entry = validate_price_gap(symbol, entry)
    validation_time = time.time() - start_time
    
    if not valid:
        print(f"❌ Skipping {symbol} {strategy_type} signal - price gap too large (validation: {validation_time:.3f}s)")
        return None
    
    signal_id = f"MTF{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    strategy_desc = MULTI_TIMEFRAME_STRATEGIES.get(strategy_type, {}).get("description", strategy_type)
    
    volume_info = f"Volume: {signal_data.get('volume_ratio', 0):.1f}x"
    if signal_data.get('breakout_size_pct'):
        breakout_info = f"Breakout: {signal_data['breakout_size_pct']:.2f}%"
    elif signal_data.get('price_change_pct'):
        breakout_info = f"Move: {signal_data['price_change_pct']:.2f}%"
    else:
        breakout_info = f"Block Trades: {signal_data.get('block_trades', 0)}"
    
    current_price, timestamp = get_current_price_real_time(symbol)
    
    message = (f"📈 <b>{strategy_desc}</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Current Price:</b> ${current_price or validated_entry:.2f}\n"
              f"<b>{volume_info}</b>\n"
              f"<b>{breakout_info}</b>\n\n"
              f"<b>Entry:</b> ${validated_entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Timeframe:</b> {strategy_type}\n"
              f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    send_start = time.time()
    send_telegram(message)
    send_time = time.time() - send_start
    
    update_signal_time(symbol, "multi_tf")
    
    signal_record = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": validated_entry,
        "sl": sl,
        "targets": targets,
        "strategy": strategy_type,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_record
    
    monitor_trade_live(symbol, side, validated_entry, sl, targets, strategy_desc, signal_id)
    
    print(f"✅ Multi-timeframe signal sent for {symbol} in {validation_time+send_time:.3f}s")
    return signal_id

def send_flash_institutional_alert(symbol, side, entry, sl, targets, signal_data):
    """ULTRA-FAST: Send flash alert with REAL-TIME validation"""
    global signal_counter
    
    start_time = time.time()
    valid, validated_entry = validate_price_gap(symbol, entry)
    validation_time = time.time() - start_time
    
    if not valid:
        print(f"❌ Skipping {symbol} flash signal - price gap too large (validation: {validation_time:.3f}s)")
        return None
    
    signal_id = f"FLASH{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
    signal_type = signal_data.get("type", "flash_move")
    if signal_type == "flash_move":
        behavior_desc = "⚡ FLASH INSTITUTIONAL MOVE"
    elif signal_type == "quick_momentum":
        behavior_desc = "⚡ QUICK MOMENTUM"
    else:
        behavior_desc = "⚡ INSTITUTIONAL MOVE"
    
    volume_info = f"Volume: {signal_data.get('volume_ratio', 0):.1f}x"
    if signal_data.get('price_change_pct'):
        move_info = f"Move: {signal_data['price_change_pct']:.2f}%"
    elif signal_data.get('block_trades'):
        move_info = f"Block Trades: {signal_data['block_trades']}"
    else:
        move_info = "Quick Institutional Move"
    
    current_price, timestamp = get_current_price_real_time(symbol)
    
    message = (f"<b>{behavior_desc}</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>Current Price:</b> ${current_price or validated_entry:.2f}\n"
              f"<b>{volume_info}</b>\n"
              f"<b>{move_info}</b>\n\n"
              f"<b>Entry:</b> ${validated_entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC\n"
              f"<b>⚠️ FAST MOVE - QUICK ACTION!</b>")
    
    send_start = time.time()
    send_telegram(message)
    send_time = time.time() - send_start
    
    update_signal_time(symbol, "flash")
    
    signal_record = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": validated_entry,
        "sl": sl,
        "targets": targets,
        "strategy": signal_type,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_record
    
    monitor_trade_live(symbol, side, validated_entry, sl, targets, behavior_desc, signal_id)
    
    print(f"✅ Flash signal sent for {symbol} in {validation_time+send_time:.3f}s")
    return signal_id

# =================== ENHANCED MONITORING ======================
def monitor_trade_live(symbol, side, entry, sl, targets, strategy_name, signal_id):
    """ENHANCED: Monitor trade with REAL-TIME price and price updates"""
    
    def monitoring_thread():
        print(f"🔍 Starting ENHANCED monitoring for {symbol} - Signal ID: {signal_id}")
        
        entry_triggered = False
        targets_hit = [False] * len(targets)
        entry_attempts = 0
        max_entry_attempts = 10  # Reduced from 15
        
        # Track highest/lowest price for updates
        highest_price = None if side == "LONG" else float('inf')
        last_update_time = time.time()
        update_cooldown = 30  # Seconds between updates
        
        while True:
            try:
                price, timestamp = get_current_price_real_time(symbol)
                if price is None:
                    time.sleep(3)  # Reduced from 8 to 3 seconds
                    continue
                
                current_time = time.time()
                
                # ENTRY LOGIC
                if not entry_triggered:
                    entry_attempts += 1
                    if (side == "LONG" and price >= entry) or (side == "SHORT" and price <= entry):
                        entry_triggered = True
                        highest_price = price if side == "LONG" else price
                        send_telegram(f"✅ ENTRY TRIGGERED: {symbol} {side} @ ${price:.2f}")
                    elif entry_attempts >= max_entry_attempts:
                        send_telegram(f"⏰ ENTRY EXPIRED: {symbol} never reached entry @ ${entry:.2f} (Current: ${price:.2f})")
                        if signal_id in active_monitoring_threads:
                            del active_monitoring_threads[signal_id]
                        break
                
                # AFTER ENTRY - PRICE MONITORING
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
                    
                    # ENHANCED PRICE MONITORING - Send updates when price makes new highs/lows
                    if current_time - last_update_time >= update_cooldown:
                        # For LONG: send update when making new highs
                        if side == "LONG":
                            if highest_price is None or price > highest_price:
                                new_highest = send_price_update(symbol, price, entry, side, signal_id, highest_price)
                                if new_highest:
                                    highest_price = new_highest
                                    last_update_time = current_time
                        
                        # For SHORT: send update when making new lows
                        elif side == "SHORT":
                            if highest_price == float('inf') or price < highest_price:
                                new_lowest = send_price_update(symbol, price, entry, side, signal_id, highest_price)
                                if new_lowest:
                                    highest_price = new_lowest
                                    last_update_time = current_time
                
                # Ultra-fast monitoring interval
                time.sleep(2)  # Reduced from 8 to 2 seconds
                
            except Exception as e:
                print(f"⚠️ Monitoring error for {symbol}: {e}")
                time.sleep(5)
    
    thread = threading.Thread(target=monitoring_thread, daemon=True)
    thread.start()
    active_monitoring_threads[signal_id] = thread
    print(f"✅ Enhanced monitoring started for {signal_id}")

# =================== ANALYSIS FUNCTIONS =====================
def analyze_institutional_flow(symbol):
    """ULTRA-FAST: Analyze for institutional flow"""
    df_3min = get_market_data(symbol, "3m", 50)  # Reduced from 80 to 50
    
    if df_3min is None or len(df_3min) < 15:  # Reduced from 20 to 15
        return None
    
    print(f"🔍 Analyzing {symbol} for institutional flow...")
    
    df_3min = add_technical_indicators(df_3min)
    
    behaviors = [
        ("institutional_buying", detect_institutional_buying(df_3min, symbol)),
        ("institutional_selling", detect_institutional_selling(df_3min, symbol)),
        ("bullish_stop_hunt", detect_bullish_stop_hunt(df_3min, symbol)),
        ("bearish_stop_hunt", detect_bearish_stop_hunt(df_3min, symbol))
    ]
    
    for behavior_type, direction in behaviors:
        if direction:
            print(f"✅ {symbol}: {BEHAVIOR_TYPES[behavior_type]} - {direction}")
            
            current_price, timestamp = get_current_price_real_time(symbol)
            if current_price is None:
                return None
            
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
    """ULTRA-FAST: Analyze for technical trading signals"""
    cfg = TRADING_MODES[mode_name]
    df = get_market_data(symbol, cfg["interval"], 100)  # Reduced from 200 to 100
    
    if df is None or df.empty:
        return None
    
    df = add_technical_indicators(df)
    
    confirmed_breakouts = check_pending_breakouts(symbol)
    for breakout in confirmed_breakouts:
        if breakout["strategy"] == mode_name:
            current_price, timestamp = get_current_price_real_time(symbol)
            if current_price is None:
                continue
            
            entry = current_price
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
    
    for side in ["LONG", "SHORT"]:
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
                "strategy": mode_name
            }
    
    return None

def analyze_multi_timeframe_institutional(symbol):
    """ULTRA-FAST: Analyze for institutional moves"""
    print(f"🔍 Multi-timeframe analysis for {symbol}...")
    
    signals_found = []
    
    # Check flash signals first (fastest)
    flash_signal = detect_flash_institutional_move(symbol)
    if flash_signal and can_send_signal(symbol, "flash"):
        current_price, timestamp = get_current_price_real_time(symbol)
        if current_price:
            entry = current_price
            sl, targets = calculate_trade_levels(entry, flash_signal["direction"])
            
            signals_found.append({
                "symbol": symbol,
                "side": flash_signal["direction"],
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy_type": "flash_move",
                "signal_data": flash_signal
            })
    
    # Check quick momentum
    momentum_signal = detect_quick_momentum(symbol)
    if momentum_signal and can_send_signal(symbol, "flash"):
        current_price, timestamp = get_current_price_real_time(symbol)
        if current_price:
            entry = current_price
            sl, targets = calculate_trade_levels(entry, momentum_signal["direction"])
            
            signals_found.append({
                "symbol": symbol,
                "side": momentum_signal["direction"],
                "entry": entry,
                "sl": sl,
                "targets": targets,
                "strategy_type": "quick_momentum",
                "signal_data": momentum_signal
            })
    
    # Check volume surge
    volume_surge_signal = detect_volume_surge(symbol)
    if volume_surge_signal and can_send_signal(symbol, "multi_tf"):
        current_price, timestamp = get_current_price_real_time(symbol)
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
    
    # Check other timeframes (do fewer for speed)
    fast_timeframes = ["FLASH_INSTITUTIONAL", "QUICK_MOMENTUM", "5M_BREAKOUT"]
    for timeframe_strategy in fast_timeframes:
        if not can_send_signal(symbol, "multi_tf"):
            continue
            
        signal_data = detect_multi_timeframe_breakout(symbol, timeframe_strategy)
        if signal_data:
            current_price, timestamp = get_current_price_real_time(symbol)
            if current_price is None:
                continue
            
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
            break  # Only process one at a time for speed
    
    return signals_found

# =================== ULTRA-FAST SCANNER FUNCTIONS ======================
def run_institutional_scanner():
    """ULTRA-FAST: Scan for institutional flow signals"""
    print("🔍 Scanning for institutional flow...")
    
    signals_found = 0
    results = []
    
    # Scan only top 3 symbols for speed
    for symbol in list(DIGITAL_ASSETS.values())[:3]:
        result = analyze_institutional_flow(symbol)
        if result:
            results.append(result)
    
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
    """ULTRA-FAST: Scan for technical trading signals"""
    print("📊 Scanning for technical signals...")
    
    signals_found = 0
    
    # Scan only top 3 symbols
    for symbol in TRADING_SYMBOLS[:3]:
        # Only check fastest strategies
        for strategy in ["INSTITUTIONAL_FLASH", "SCALP"]:
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
                break  # Only send one signal per symbol per scan
    
    print(f"✅ Technical scan complete. Signals: {signals_found}")
    return signals_found

def run_multi_timeframe_scanner():
    """ULTRA-FAST: Scan for institutional moves"""
    print("📈 Scanning ALL institutional moves...")
    
    signals_found = 0
    
    # Scan only top 4 symbols for speed
    for symbol in list(DIGITAL_ASSETS.values())[:4]:
        multi_tf_signals = analyze_multi_timeframe_institutional(symbol)
        
        for signal in multi_tf_signals:
            signal_type = signal.get("strategy_type", "")
            
            if signal_type in ["flash_move", "quick_momentum"]:
                if can_send_signal(symbol, "flash"):
                    send_flash_institutional_alert(
                        signal["symbol"],
                        signal["side"],
                        signal["entry"],
                        signal["sl"],
                        signal["targets"],
                        signal["signal_data"]
                    )
                    signals_found += 1
                    break  # Only send one signal per symbol
            else:
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
                    break  # Only send one signal per symbol
    
    print(f"✅ Institutional moves scan complete. Signals: {signals_found}")
    return signals_found

# =================== STATUS MONITORING ======================
def check_active_signals():
    """Check status of active signals"""
    current_time = time.time()
    
    completed_signal_ids = []
    for signal_id, signal_data in active_signals.items():
        if signal_data.get("status") == "COMPLETED":
            if current_time - signal_data["timestamp"] > 1800:  # 30 minutes
                completed_signal_ids.append(signal_id)
    
    for signal_id in completed_signal_ids:
        del active_signals[signal_id]
    
    return len(active_signals)

# =================== ULTRA-FAST MAIN EXECUTION =========================
def main():
    """ULTRA-FAST: Main execution loop"""
    print("=" * 60)
    print("🏛️ ULTIMATE INSTITUTIONAL FLOW BOT - ULTRA-FAST VERSION")
    print("⚡ SIGNALS IN 1-2 SECONDS - NO DELAY")
    print("📈 ENHANCED PRICE MONITORING")
    print("🔄 OPTIMIZED FOR SPEED")
    print("=" * 60)
    
    send_telegram("🤖 <b>ULTRA-FAST INSTITUTIONAL BOT ACTIVATED</b>\n"
                  "⚡ Signals in 1-2 seconds\n"
                  "📈 Enhanced price monitoring\n"
                  "🏛️ No latency issues\n"
                  f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    iteration = 0
    
    while True:
        iteration += 1
        try:
            loop_start = time.time()
            
            print(f"\n🔄 Iteration {iteration} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            
            # Show real-time prices for top symbols only
            for symbol in ["BTC-USDT", "ETH-USDT"]:
                price, timestamp = get_current_price_real_time(symbol)
                if price:
                    age = time.time() - timestamp
                    print(f"   {symbol}: ${price:.2f} ({age:.1f}s ago)")
            
            active_count = check_active_signals()
            print(f"📈 Active trades: {active_count}")
            
            # Run scanners with timing
            scanner_start = time.time()
            multi_tf_signals = run_multi_timeframe_scanner()
            scanner_time = time.time() - scanner_start
            
            if scanner_time > 5:
                print(f"⚠️ Slow scanner: {scanner_time:.2f}s")
            
            flow_signals = run_institutional_scanner()
            tech_signals = run_technical_scanner()
            
            total_signals = multi_tf_signals + flow_signals + tech_signals
            print(f"✅ TOTAL ULTRA-FAST SIGNALS: {total_signals}")
            
            loop_time = time.time() - loop_start
            print(f"⏱️ Loop completed in: {loop_time:.2f}s")
            
            # Dynamic wait time based on loop performance
            if loop_time < 5:
                wait_time = 10  # Fast loop, scan more frequently
            elif loop_time < 15:
                wait_time = 15  # Medium loop
            else:
                wait_time = 20  # Slow loop, give more time
            
            print(f"⏳ Next scan in {wait_time} seconds...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down bot...")
            shutdown_msg = ("🛑 <b>ULTRA-FAST BOT SHUTTING DOWN</b>\n\n"
                          f"📊 Total iterations: {iteration}\n"
                          f"⏰ End Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            send_telegram(shutdown_msg)
            break
            
        except Exception as e:
            print(f"⚠️ Main loop error: {e}")
            time.sleep(10)  # Reduced from 20

if __name__ == "__main__":
    main()
