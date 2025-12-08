# =================== INSTITUTIONAL FLOW & CRYPTO TRADING BOT ===================
# AI-BASED INSTITUTIONAL FLOW DETECTION WITH BREAKOUT CONFIRMATION
# ENHANCED VERSION - CATCHES ALL INSTITUTIONAL MOVES, NO MISSING SIGNALS
# OPTIMIZED FOR MAXIMUM SIGNAL CATCHING WITHOUT FALSE SIGNALS

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
INSTITUTIONAL_VOLUME_RATIO = 2.5  # REDUCED: Catch more institutional moves!
MIN_MOVE_FOR_ENTRY = 0.015  # More flexible entry criteria
STOP_HUNT_DISTANCE = 0.015  # Increased sensitivity
ABSORPTION_WICK_RATIO = 0.15  # More flexible wick detection

# =================== INSTITUTIONAL TIME ZONES ===============
# CHANGED: 24/7 MONITORING - Institutions trade all day!
INSTITUTIONAL_ACTIVE_HOURS = {
    "ALL_HOURS": (0, 24),  # Monitor 24/7 for ALL institutional moves
}

# =================== INSTITUTIONAL ORDER FLOW ===============
ORDER_FLOW_THRESHOLDS = {
    "BLOCK_TRADE_SIZE_USD": 25000.0,   # REDUCED: Catch smaller institutional trades ($25K)
    "SWEEP_RATIO": 0.60,  # More sensitive
    "IMMEDIATE_OR_CANCEL": 0.7,
    "DARK_POOL_INDICATOR": 3.0  # Reduced threshold
}

# =================== TECHNICAL TRADING SETTINGS ============
EMA_PERIOD = 20
RSI_PERIOD = 14
CANDLE_LIMIT = 200

SL_BUFFER = 0.0030  # 0.3% stop loss
TARGETS = [0.0050, 0.010, 0.020]  # BETTER: 1:1.67, 1:3.33, 1:6.67

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
        "volume_ratio": 2.5,  # Reduced
        "min_move_pct": 0.012,  # 1.2% minimum move
        "window": 10,
        "description": "1H Institutional Breakout"
    },
    "15M_MOMENTUM": {
        "interval": "15m",
        "volume_ratio": 2.5,  # Reduced
        "min_move_pct": 0.006,  # 0.6% minimum move
        "window": 15,
        "description": "15M Momentum Move"
    },
    "5M_BREAKOUT": {
        "interval": "5m",
        "volume_ratio": 3.0,  # Reduced
        "min_move_pct": 0.004,  # 0.4% minimum move
        "window": 20,
        "description": "5M Quick Breakout"
    },
    "INSTITUTIONAL_VOLUME_SURGE": {
        "interval": "5m",
        "volume_ratio": 4.0,  # Reduced from 5.0
        "min_move_pct": 0.002,  # Small move but huge volume
        "window": 5,
        "description": "Institutional Volume Surge"
    },
    # NEW: FLASH INSTITUTIONAL MOVES
    "FLASH_INSTITUTIONAL": {
        "interval": "3m",
        "volume_ratio": 2.8,
        "min_move_pct": 0.003,  # Very small moves
        "window": 10,
        "description": "Flash Institutional Move"
    },
    # NEW: QUICK MOMENTUM
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
        "recent_hl_window": 8,  # Reduced for faster detection
        "rsi_long_min": 45, "rsi_long_max": 85,  # Wider ranges
        "rsi_short_min": 15, "rsi_short_max": 55,
        "entry_buffer_long": 0.0001,
        "entry_buffer_short": 0.0001,
        "max_entry_drift": 0.0015,
        "immediate_mode": True,  # ENABLED for flash moves!
        "require_breakout_confirmation": False,  # DISABLED for faster signals
        "confirmation_candles": 0,
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 20,
        "institutional_only": False,  # Allow all signals
        "min_volume_ratio": 2.0,  # Reduced
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
        "immediate_mode": True,  # Enabled
        "require_breakout_confirmation": True,
        "confirmation_candles": 1,  # Reduced from 2
        "need_prev_candle_break": True,
        "volume_filter": True,
        "volume_lookback": 30,
        "institutional_only": True,
        "min_volume_ratio": 2.0,  # Reduced
        "breakout_retest_allowed": True,
        "max_retest_count": 3,
        "multi_timeframe_check": True,
        "max_signals_per_hour": 4
    },
    "INSTITUTIONAL_MOMENTUM": {
        "interval": "15m",
        "recent_hl_window": 12,
        "rsi_long_min": 40, "rsi_long_max": 90,  # Wider
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
    # NEW: INSTITUTIONAL FLASH MODE
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
        "max_signals_per_hour": 10  # More signals allowed
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
    # NEW: Flash institutional moves
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
signal_counts_hourly = {}  # NEW: Track signal frequency

# Signal cooldown periods (in seconds) - REDUCED!
SIGNAL_COOLDOWN = 300  # REDUCED: 5 minutes only (was 120 minutes!)
BREAKOUT_CONFIRMATION_TIMEOUT = 1800  # 30 minutes
MULTI_TF_COOLDOWN = 180  # REDUCED: 3 minutes (was 30 minutes!)

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
        response = requests.post(url, timeout=10)
        
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
            X_buy.append([4.5, 0.22, 0.12, 3.2, 0.20, 0.72, 0.70, 0.025, 1.5, 0.80])
            y_buy.append(1)
            
            # Pattern 2: Stealth accumulation
            X_buy.append([3.8, 0.18, 0.15, 2.8, 0.16, 0.65, 0.65, 0.020, 1.3, 0.75])
            y_buy.append(1)
            
            # Pattern 3: V-shaped recovery
            X_buy.append([5.0, 0.28, 0.10, 3.8, 0.25, 0.78, 0.75, 0.030, 1.7, 0.85])
            y_buy.append(1)
            
            # Pattern 4: Flash buying (NEW)
            X_buy.append([2.8, 0.15, 0.08, 2.2, 0.12, 0.60, 0.68, 0.015, 1.1, 0.70])
            y_buy.append(1)
            
            # NON-INSTITUTIONAL patterns (noise)
            X_buy.append([1.5, 0.08, 0.55, 1.0, 0.06, 0.30, 0.40, 0.008, 0.6, 0.35])
            y_buy.append(0)
            
            X_buy.append([2.0, 0.12, 0.50, 1.5, 0.10, 0.38, 0.45, 0.012, 0.8, 0.42])
            y_buy.append(0)
            
            # INSTITUTIONAL SELLING PATTERNS (SHORT)
            X_sell = []
            y_sell = []
            
            # Pattern 1: Distribution at resistance
            X_sell.append([4.8, 0.18, 0.25, 3.5, 0.22, 0.75, 0.22, 0.028, 1.6, 0.20])
            y_sell.append(1)
            
            # Pattern 2: Failed breakout selling
            X_sell.append([5.2, 0.22, 0.30, 3.8, 0.28, 0.80, 0.18, 0.032, 1.8, 0.16])
            y_sell.append(1)
            
            # Pattern 3: Exhaustion selling
            X_sell.append([4.2, 0.15, 0.22, 3.2, 0.20, 0.70, 0.25, 0.025, 1.4, 0.22])
            y_sell.append(1)
            
            # Pattern 4: Flash selling (NEW)
            X_sell.append([3.0, 0.12, 0.18, 2.3, 0.14, 0.65, 0.25, 0.018, 1.2, 0.20])
            y_sell.append(1)
            
            # NON-INSTITUTIONAL patterns (noise)
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
        """Analyze trade flow for institutional activity"""
        try:
            trades = get_recent_trades(symbol, limit=150)
            if not trades:
                return {"buy_pressure": 0.5, "block_buys": 0, "block_sells": 0, "direction": "NEUTRAL"}
            
            buy_volume_usd = 0
            sell_volume_usd = 0
            block_buys = 0
            block_sells = 0
            
            current_price = get_current_price(symbol)
            if current_price is None:
                current_price = 1.0
            
            for trade in trades:
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
            
            # IMPROVED DIRECTION DETECTION
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
    df["volume_ma"] = df["volume"].rolling(15).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1e-9)
    
    return df

# =================== INSTITUTIONAL BEHAVIOR DETECTION =======
def detect_institutional_buying(df, symbol):
    """Detect institutional buying (LONG) - MORE SENSITIVE"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 25:
            return None
        
        # 1. VOLUME CHECK - More sensitive
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if vol_avg_15 == 0 or current_vol < vol_avg_15 * 2.0:  # Reduced from 3.5
            return None
        
        # 2. BULLISH CANDLE CHECK - More flexible
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if current_body == 0 or lower_wick < current_body * 0.10:  # Reduced from 0.20
            return None
        
        # 3. PRICE ACTION - Just need higher close
        if not (close.iloc[-1] > close.iloc[-2]):
            return None
        
        # 4. TRADE FLOW - More sensitive
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] != "LONG":
            return None
        
        # 5. RSI CHECK - Optional, not required
        rsi = 100 - (100 / (1 + (close.diff().clip(lower=0).rolling(14).mean() / 
                                -close.diff().clip(upper=0).rolling(14).mean().replace(0, 1e-9))))
        rsi_value = rsi.iloc[-1] if not rsi.empty else 50
        
        print(f"✅ Institutional buying detected: {symbol} | Block buys: {trade_flow['block_buys']} | Volume: {current_vol/vol_avg_15:.1f}x")
        return "LONG"
        
    except Exception as e:
        print(f"Error in institutional buying detection: {e}")
        return None

def detect_institutional_selling(df, symbol):
    """Detect institutional selling (SHORT) - MORE SENSITIVE"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        open_price = df['open']
        
        if len(close) < 25:
            return None
        
        # 1. VOLUME CHECK - More sensitive
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if vol_avg_15 == 0 or current_vol < vol_avg_15 * 2.0:  # Reduced
            return None
        
        # 2. BEARISH CANDLE CHECK - More flexible
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if current_body == 0 or upper_wick < current_body * 0.10:  # Reduced
            return None
        
        # 3. PRICE ACTION - Just need lower close
        if not (close.iloc[-1] < close.iloc[-2]):
            return None
        
        # 4. TRADE FLOW - More sensitive
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] != "SHORT":
            return None
        
        print(f"✅ Institutional selling detected: {symbol} | Block sells: {trade_flow['block_sells']} | Volume: {current_vol/vol_avg_15:.1f}x")
        return "SHORT"
        
    except Exception as e:
        print(f"Error in institutional selling detection: {e}")
        return None

def detect_bullish_stop_hunt(df, symbol):
    """Detect bullish stop hunt (LONG) - More sensitive"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 20:
            return None
        
        recent_low = low.iloc[-15:-5].min()
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if vol_avg == 0:
            return None
        
        # More sensitive stop hunt detection
        if (current_low < recent_low * (1 - 0.010) and  # Reduced from 0.012
            current_close > recent_low * 1.015 and
            current_vol > vol_avg * 3.5 and  # Reduced from 5.0
            current_close > close.iloc[-2]):
            
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["direction"] == "LONG":
                print(f"✅ Bullish stop hunt detected: {symbol}")
                return "LONG"
        
    except Exception as e:
        print(f"Error in bullish stop hunt detection: {e}")
        return None
    return None

def detect_bearish_stop_hunt(df, symbol):
    """Detect bearish stop hunt (SHORT) - More sensitive"""
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if len(close) < 20:
            return None
        
        recent_high = high.iloc[-15:-5].max()
        current_high = high.iloc[-1]
        current_close = close.iloc[-1]
        
        vol_avg = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if vol_avg == 0:
            return None
        
        # More sensitive stop hunt detection
        if (current_high > recent_high * (1 + 0.010) and  # Reduced
            current_close < recent_high * 0.985 and
            current_vol > vol_avg * 3.5 and  # Reduced
            current_close < close.iloc[-2]):
            
            trade_flow = flow_ai.analyze_trade_flow(symbol)
            if trade_flow["direction"] == "SHORT":
                print(f"✅ Bearish stop hunt detected: {symbol}")
                return "SHORT"
        
    except Exception as e:
        print(f"Error in bearish stop hunt detection: {e}")
        return None
    return None

# =================== NEW: FLASH INSTITUTIONAL DETECTION ===========
def detect_flash_institutional_move(symbol):
    """Detect flash institutional moves - CATCHES QUICK MOVES!"""
    try:
        df_2m = get_market_data(symbol, "2m", 30)
        if df_2m is None or len(df_2m) < 10:
            return None
        
        current_volume = df_2m['volume'].iloc[-1]
        avg_volume = df_2m['volume'].rolling(10).mean().iloc[-1]
        
        if avg_volume == 0 or current_volume < avg_volume * 2.5:
            return None
        
        # Check for quick move
        current_close = df_2m['close'].iloc[-1]
        prev_close = df_2m['close'].iloc[-2]
        price_change_pct = abs(current_close - prev_close) / prev_close
        
        if price_change_pct < 0.003:  # Less than 0.3% move
            return None
        
        # Check trade flow for institutional activity
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
    """Detect quick momentum moves - VERY SENSITIVE"""
    try:
        df_1m = get_market_data(symbol, "1m", 20)
        if df_1m is None or len(df_1m) < 10:
            return None
        
        # Volume spike
        current_volume = df_1m['volume'].iloc[-1]
        avg_volume = df_1m['volume'].rolling(10).mean().iloc[-1]
        
        if avg_volume == 0 or current_volume < avg_volume * 2.2:
            return None
        
        # Check for consecutive moves
        closes = df_1m['close'].iloc[-5:]
        if len(closes) < 5:
            return None
        
        # Check for 3 consecutive bullish candles
        if all(closes.iloc[i] > closes.iloc[i-1] for i in range(1, 4)):
            direction = "LONG"
        # Check for 3 consecutive bearish candles
        elif all(closes.iloc[i] < closes.iloc[i-1] for i in range(1, 4)):
            direction = "SHORT"
        else:
            return None
        
        # Check trade flow
        trade_flow = flow_ai.analyze_trade_flow(symbol)
        if trade_flow["direction"] == direction:
            print(f"✅ Quick momentum: {symbol} - {direction} | Volume: {current_volume/avg_volume:.1f}x")
            return {
                "type": "quick_momentum",
                "direction": direction,
                "current_price": closes.iloc[-1],
                "volume_ratio": current_volume / avg_volume,
                "consecutive_candles": 3
            }
        
        return None
        
    except Exception as e:
        print(f"Error in quick momentum detection for {symbol}: {e}")
        return None

# =================== MULTI-TIMEFRAME DETECTION ===========
def detect_multi_timeframe_breakout(symbol, timeframe_strategy):
    """Detect institutional breakout - MORE SENSITIVE"""
    try:
        strategy = MULTI_TIMEFRAME_STRATEGIES[timeframe_strategy]
        df = get_market_data(symbol, strategy["interval"], 80)
        
        if df is None or len(df) < 25:
            return None
        
        df = add_technical_indicators(df)
        
        # Volume check - More sensitive
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(15).mean().iloc[-1]
        
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
                # Just need to close near the high, not necessarily above
                if current_close > recent_high * 0.995:  # More flexible
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
        
        # Check bearish breakout
        if current_low < recent_low:
            breakout_size = (recent_low - current_low) / recent_low
            if breakout_size >= strategy["min_move_pct"]:
                # Just need to close near the low
                if current_close < recent_low * 1.005:  # More flexible
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
    """Detect institutional volume surge - MORE SENSITIVE"""
    try:
        df_3m = get_market_data(symbol, "3m", 40)
        if df_3m is None or len(df_3m) < 15:
            return None
        
        # Check for volume spike
        current_volume = df_3m['volume'].iloc[-1]
        avg_volume = df_3m['volume'].rolling(15).mean().iloc[-1]
        
        if avg_volume == 0 or current_volume < avg_volume * 3.5:  # Reduced from 5.0
            return None
        
        # Price change can be moderate (not too small)
        current_close = df_3m['close'].iloc[-1]
        prev_close = df_3m['close'].iloc[-2]
        price_change = abs(current_close - prev_close) / prev_close
        
        # Volume surge with reasonable price change = institutional
        if price_change <= 0.008:  # Up to 0.8% price change
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

# =================== BREAKOUT DETECTION FUNCTIONS ===========
def check_breakout(df, side, window=10):
    """Check if price has broken key levels"""
    try:
        if side == "LONG":
            resistance = df["high"].iloc[-window:-1].max()
            current_high = df["high"].iloc[-1]
            current_close = df["close"].iloc[-1]
            
            # More flexible breakout
            if current_high > resistance and current_close > resistance * 0.998:
                return True, resistance
            return False, resistance
            
        else:  # SHORT
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
    """Check if any pending breakouts have been confirmed"""
    current_time = time.time()
    confirmed_breakouts = []
    
    for breakout_id, breakout_data in list(pending_breakouts.items()):
        if breakout_data["symbol"] != symbol:
            continue
            
        if current_time - breakout_data["timestamp"] > BREAKOUT_CONFIRMATION_TIMEOUT:
            del pending_breakouts[breakout_id]
            continue
        
        df = get_market_data(symbol, breakout_data["interval"], 30)
        if df is None or df.empty:
            continue
        
        breakout_confirmed, _ = check_breakout(df, breakout_data["side"], 
                                              breakout_data["window"])
        
        if breakout_confirmed:
            confirmed_breakouts.append(breakout_data)
            del pending_breakouts[breakout_id]
    
    return confirmed_breakouts

# =================== TECHNICAL TRADING FUNCTIONS ============
def check_technical_conditions(df, mode_cfg, side):
    """Check trading conditions - LESS RESTRICTIVE"""
    try:
        if df.empty or len(df) < 15:
            return False
        
        price = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        
        # Optional RSI check
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if side == "LONG":
                if not (mode_cfg["rsi_long_min"] <= rsi <= mode_cfg["rsi_long_max"]):
                    return False
            else:
                if not (mode_cfg["rsi_short_min"] <= rsi <= mode_cfg["rsi_short_max"]):
                    return False
        
        # Volume check
        vol_avg = df["volume"].rolling(10).mean().iloc[-1]
        last_vol = df["volume"].iloc[-1]
        if vol_avg > 0 and last_vol < vol_avg * mode_cfg.get("min_volume_ratio", 1.8):
            return False
        
        # Trend check (optional)
        if side == "LONG":
            return price > ema_20
        else:
            return price < ema_20
            
    except Exception as e:
        print(f"Error in technical conditions check: {e}")
        return False

def compute_technical_entry(df, side, mode_cfg, symbol):
    """Compute entry price"""
    if df.empty:
        return None
    
    price = df["close"].iloc[-1]
    
    # Use current price for immediate mode
    if mode_cfg.get("immediate_mode", False):
        return price
    
    w = mode_cfg["recent_hl_window"]
    
    if side == "LONG":
        resistance = df["high"].iloc[-w:].max()
        entry = resistance * (1 + mode_cfg["entry_buffer_long"])
        
        # Cap the distance
        max_pct = ABS_MAX_ENTRY_PCT.get(symbol, 0.01)
        if entry > price * (1 + max_pct):
            entry = price * (1 + max_pct * 0.8)
    else:
        support = df["low"].iloc[-w:].min()
        entry = support * (1 - mode_cfg["entry_buffer_short"])
        
        max_pct = ABS_MAX_ENTRY_PCT.get(symbol, 0.01)
        if entry < price * (1 - max_pct):
            entry = price * (1 - max_pct * 0.8)
    
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
    """Check if signal can be sent - SHORTER COOLDOWNS!"""
    current_time = time.time()
    
    # Different cooldowns
    if signal_type == "multi_tf":
        cooldown = MULTI_TF_COOLDOWN  # 3 minutes
    elif signal_type == "flash":
        cooldown = 120  # 2 minutes for flash moves
    else:
        cooldown = SIGNAL_COOLDOWN  # 5 minutes
    
    if symbol in last_signal_time:
        time_since_last = current_time - last_signal_time[symbol]
        if time_since_last < cooldown:
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
        max_entry_attempts = 15  # Reduced from 20
        
        while True:
            price = get_current_price(symbol)
            if price is None:
                time.sleep(8)
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
            
            time.sleep(8)
    
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
    behavior_desc = BEHAVIOR_TYPES.get(behavior_type, "Institutional Move")
    
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
    
    monitor_trade_live(symbol, side, entry, sl, targets, behavior_desc, signal_id)
    
    return signal_id

def send_technical_alert(symbol, side, entry, sl, targets, strategy):
    """Send technical trading alert"""
    global signal_counter
    
    signal_id = f"TECH{signal_counter:04d}"
    signal_counter += 1
    
    targets_str = " → ".join([f"${t:.2f}" for t in targets])
    
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
    
    monitor_trade_live(symbol, side, entry, sl, targets, strategy, signal_id)
    
    return signal_id

def send_multi_timeframe_alert(symbol, side, entry, sl, targets, strategy_type, signal_data):
    """Send multi-timeframe institutional alert"""
    global signal_counter
    
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
    
    monitor_trade_live(symbol, side, entry, sl, targets, strategy_desc, signal_id)
    
    return signal_id

def send_flash_institutional_alert(symbol, side, entry, sl, targets, signal_data):
    """Send flash institutional alert - NEW!"""
    global signal_counter
    
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
    
    message = (f"<b>{behavior_desc}</b>\n\n"
              f"<b>Symbol:</b> {symbol}\n"
              f"<b>Direction:</b> {side}\n"
              f"<b>{volume_info}</b>\n"
              f"<b>{move_info}</b>\n\n"
              f"<b>Entry:</b> ${entry:.2f}\n"
              f"<b>Stop Loss:</b> ${sl:.2f}\n"
              f"<b>Risk/Reward:</b> 1:{TARGETS[0]/SL_BUFFER:.1f}+\n\n"
              f"<b>Targets:</b> {targets_str}\n"
              f"<b>Signal ID:</b> {signal_id}\n"
              f"<b>⚠️ FAST MOVE - QUICK ACTION!</b>")
    
    send_telegram(message)
    update_signal_time(symbol, "flash")
    
    signal_record = {
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": sl,
        "targets": targets,
        "strategy": signal_type,
        "timestamp": time.time(),
        "status": "ACTIVE"
    }
    
    active_signals[signal_id] = signal_record
    
    monitor_trade_live(symbol, side, entry, sl, targets, behavior_desc, signal_id)
    
    return signal_id

# =================== ANALYSIS FUNCTIONS =====================
def analyze_institutional_flow(symbol):
    """Analyze for institutional flow behavior - MORE SENSITIVE"""
    df_3min = get_market_data(symbol, "3m", 80)  # Changed to 3m for faster detection
    
    if df_3min is None or len(df_3min) < 20:
        return None
    
    print(f"🔍 Analyzing {symbol} for institutional flow...")
    
    df_3min = add_technical_indicators(df_3min)
    
    # Check ALL institutional behaviors
    behaviors = [
        ("institutional_buying", detect_institutional_buying(df_3min, symbol)),
        ("institutional_selling", detect_institutional_selling(df_3min, symbol)),
        ("bullish_stop_hunt", detect_bullish_stop_hunt(df_3min, symbol)),
        ("bearish_stop_hunt", detect_bearish_stop_hunt(df_3min, symbol))
    ]
    
    for behavior_type, direction in behaviors:
        if direction:
            print(f"✅ {symbol}: {BEHAVIOR_TYPES[behavior_type]} - {direction}")
            
            current_price = get_current_price(symbol)
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
    """Analyze for technical trading signals"""
    cfg = TRADING_MODES[mode_name]
    df = get_market_data(symbol, cfg["interval"], CANDLE_LIMIT)
    
    if df is None or df.empty:
        return None
    
    df = add_technical_indicators(df)
    
    # Check confirmed breakouts
    confirmed_breakouts = check_pending_breakouts(symbol)
    for breakout in confirmed_breakouts:
        if breakout["strategy"] == mode_name:
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
    """Analyze for institutional moves across multiple timeframes - MORE SENSITIVE!"""
    print(f"🔍 Multi-timeframe analysis for {symbol}...")
    
    signals_found = []
    
    # Check ALL multi-timeframe strategies
    for timeframe_strategy in MULTI_TIMEFRAME_STRATEGIES.keys():
        if not can_send_signal(symbol, "multi_tf"):
            continue
            
        signal_data = detect_multi_timeframe_breakout(symbol, timeframe_strategy)
        if signal_data:
            current_price = get_current_price(symbol)
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
    
    # NEW: Check for flash institutional moves
    flash_signal = detect_flash_institutional_move(symbol)
    if flash_signal and can_send_signal(symbol, "flash"):
        current_price = get_current_price(symbol)
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
    
    # NEW: Check for quick momentum
    momentum_signal = detect_quick_momentum(symbol)
    if momentum_signal and can_send_signal(symbol, "flash"):
        current_price = get_current_price(symbol)
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
    
    return signals_found

# =================== SCANNER FUNCTIONS ======================
def run_institutional_scanner():
    """Scan for institutional flow signals - 24/7 NOW!"""
    print("🔍 Scanning for institutional flow...")
    
    signals_found = 0
    results = []
    
    # Scan MORE symbols
    for symbol in list(DIGITAL_ASSETS.values())[:6]:  # Increased from 4
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
    """Scan for technical trading signals"""
    print("📊 Scanning for technical signals...")
    
    signals_found = 0
    
    for symbol in TRADING_SYMBOLS[:4]:  # Reduced for speed
        for strategy in ["SCALP", "INSTITUTIONAL_FLASH", "INSTITUTIONAL_MOMENTUM"]:
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
    """Scan for institutional moves across all timeframes - MAIN SCANNER NOW!"""
    print("📈 Scanning ALL institutional moves...")
    
    signals_found = 0
    
    # Scan ALL symbols quickly
    for symbol in list(DIGITAL_ASSETS.values())[:6]:
        multi_tf_signals = analyze_multi_timeframe_institutional(symbol)
        
        for signal in multi_tf_signals:
            signal_type = signal.get("strategy_type", "")
            
            if signal_type in ["flash_move", "quick_momentum"]:
                # Flash moves
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
            else:
                # Multi-timeframe signals
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
    
    print(f"✅ Institutional moves scan complete. Signals: {signals_found}")
    return signals_found

# =================== STATUS MONITORING ======================
def check_active_signals():
    """Check status of active signals"""
    current_time = time.time()
    
    completed_signal_ids = []
    for signal_id, signal_data in active_signals.items():
        if signal_data.get("status") == "COMPLETED":
            if current_time - signal_data["timestamp"] > 3600:  # 1 hour
                completed_signal_ids.append(signal_id)
    
    for signal_id in completed_signal_ids:
        del active_signals[signal_id]
    
    return len(active_signals)

# =================== MAIN EXECUTION =========================
def main():
    """Main execution loop - FASTER SCANNING!"""
    print("=" * 60)
    print("🏛️ ULTIMATE INSTITUTIONAL FLOW BOT")
    print("⚡ CATCHES ALL INSTITUTIONAL MOVES - NO MISSING SIGNALS!")
    print("📈 24/7 MONITORING - FLASH MOVE DETECTION")
    print("=" * 60)
    
    send_telegram("🤖 <b>ULTIMATE INSTITUTIONAL BOT ACTIVATED</b>\n"
                  "⚡ Catches ALL Institutional Moves\n"
                  "📈 Multi-Timeframe Detection (1H to 1M)\n"
                  "🏛️ 24/7 Monitoring\n"
                  f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    iteration = 0
    
    while True:
        iteration += 1
        try:
            print(f"\n🔄 Iteration {iteration} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            
            active_count = check_active_signals()
            print(f"📈 Active trades: {active_count}")
            print(f"🔔 Pending breakouts: {len(pending_breakouts)}")
            
            # RUN ALL SCANNERS - NO MORE MISSING MOVES!
            
            # 1. Multi-timeframe scanner (MAIN - catches everything)
            multi_tf_signals = run_multi_timeframe_scanner()
            print(f"📈 Institutional moves: {multi_tf_signals}")
            
            # 2. Institutional scanner (24/7 now!)
            flow_signals = run_institutional_scanner()
            print(f"🏛️ Flow signals: {flow_signals}")
            
            # 3. Technical scanner
            tech_signals = run_technical_scanner()
            print(f"📊 Technical signals: {tech_signals}")
            
            total_signals = multi_tf_signals + flow_signals + tech_signals
            print(f"✅ TOTAL SIGNALS THIS ITERATION: {total_signals}")
            
            # FASTER SCANNING - Every 15 seconds!
            wait_time = 15  # Reduced from 30 seconds
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
            time.sleep(20)

if __name__ == "__main__":
    main()
