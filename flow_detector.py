import os
import time
import requests
import pandas as pd
from typing import Optional

# =================== CREDENTIALS ===================
API_KEY = os.getenv("SYSTEM_KEY", " ")
API_SECRET = os.getenv("SYSTEM_SECRET", " ")
PLATFORM_API_URL = "https://open-api.bingx.com"

BOT_TOKEN = os.getenv("ALERT_TOKEN", "8107235827:AAGIEJ5qnGxl6smXrbTsRX6fxWm_x6wlh0I")
CHAT_ID = os.getenv("ALERT_TARGET", "6093932842")

# =================== SETTINGS ======================
SYMBOLS = ["BTC-USDT", "ETH-USDT"]

EMA_PERIOD = 20
RSI_PERIOD = 14
CANDLE_LIMIT = 100

SL_BUFFER = 0.0020                 # 0.20%
TARGETS = [0.0025, 0.004, 0.007]   # TP levels

ABS_MAX_ENTRY_USD = {
    "BTC-USDT": 80.0,
    "ETH-USDT": 8.0
}

MODES = {
    "SCALP": {
        "interval": "1m",
        "recent_hl_window": 3,
        "rsi_long_min": 48, "rsi_long_max": 72,
        "rsi_short_min": 28, "rsi_short_max": 52,
        "entry_buffer_long": 0.0004,     # 0.04%
        "entry_buffer_short": 0.0004,
        "max_entry_drift": 0.0010,       # 0.10%
        "immediate_mode": True,
        "immediate_tol": 0.0010,         # 0.10%
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 20
    },
    "SWING": {
        "interval": "5m",
        "recent_hl_window": 10,
        "rsi_long_min": 45, "rsi_long_max": 75,
        "rsi_short_min": 25, "rsi_short_max": 55,
        "entry_buffer_long": 0.0015,     # 0.15%
        "entry_buffer_short": 0.0015,
        "max_entry_drift": 0.0040,       # 0.40%
        "immediate_mode": True,
        "immediate_tol": 0.0015,         # 0.15%
        "need_prev_candle_break": False,
        "volume_filter": True,
        "volume_lookback": 30
    }
}

MILESTONE_STEP_USD = {"BTC-USDT": 5, "ETH-USDT": 2}
POLL_SECS = 5
SCAN_SECS = 300

# =================== PLATFORM API ==================
def platform_get_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """Get market data from platform"""
    endpoint = "/openApi/swap/v3/quote/klines"
    url = f"{PLATFORM_API_URL}{endpoint}"
    
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"
    }
    
    params = {
        "symbol": symbol,
        "interval": interval_map.get(interval, interval),
        "limit": limit
    }
    
    headers = {"X-BX-APIKEY": API_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data.get("code") == 0 and "data" in data:
            candles = data["data"]
            df = pd.DataFrame(candles, columns=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except Exception as e:
        print(f"Error fetching data: {e}")
    
    return pd.DataFrame()

def platform_get_price(symbol: str) -> Optional[float]:
    """Get current price"""
    endpoint = "/openApi/swap/v3/quote/ticker/price"
    url = f"{PLATFORM_API_URL}{endpoint}"
    
    params = {"symbol": symbol}
    headers = {"X-BX-APIKEY": API_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data.get("code") == 0 and "data" in data:
            if isinstance(data["data"], list) and len(data["data"]) > 0:
                price_str = data["data"][0].get("price")
            elif isinstance(data["data"], dict):
                price_str = data["data"].get("price")
            else:
                price_str = str(data["data"])
                
            return float(price_str) if price_str else None
    except Exception as e:
        print(f"Error getting price: {e}")
    
    return None

# =================== UTILITIES =====================
def send_telegram(msg: str):
    """Send message to Telegram"""
    if not BOT_TOKEN or not CHAT_ID:
        return
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass

def get_candles(symbol, interval, limit=CANDLE_LIMIT):
    """Get candles wrapper"""
    return platform_get_candles(symbol, interval, limit)

def add_indicators(df):
    """Add technical indicators"""
    if df.empty:
        return df
    
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def check_conditions(price, ema, rsi, mode_cfg, side):
    """Check trading conditions"""
    if side == "LONG":
        return (price > ema) and (mode_cfg["rsi_long_min"] <= rsi <= mode_cfg["rsi_long_max"])
    else:
        return (price < ema) and (mode_cfg["rsi_short_min"] <= rsi <= mode_cfg["rsi_short_max"])

def calculate_levels(entry_price, side):
    """Calculate stop loss and target levels"""
    if side == "LONG":
        sl = entry_price * (1 - SL_BUFFER)
        tps = [entry_price * (1 + x) for x in TARGETS]
    else:
        sl = entry_price * (1 + SL_BUFFER)
        tps = [entry_price * (1 - x) for x in TARGETS]
    return sl, tps

def compute_entry_price(df, side, mode_cfg, symbol):
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

def wait_for_entry(symbol, side, entry):
    """Wait for entry condition"""
    send_telegram(f"🕒 Waiting for entry: {symbol} {side} @ {entry:.2f}")
    while True:
        price = platform_get_price(symbol)
        if price is None:
            time.sleep(POLL_SECS)
            continue
            
        if (side == "LONG" and price >= entry) or (side == "SHORT" and price <= entry):
            send_telegram(f"✅ Entry filled: {symbol} {side} @ {price:.2f}")
            return price
        time.sleep(POLL_SECS)

def monitor_position(symbol, side, entry_price, sl, targets):
    """Monitor open position"""
    step = MILESTONE_STEP_USD.get(symbol, 5)
    last_reported = entry_price
    while True:
        price = platform_get_price(symbol)
        if price is None:
            time.sleep(POLL_SECS)
            continue

        if side == "LONG" and price - last_reported >= step:
            send_telegram(f"📈 {symbol} Price Up → {price:.2f}")
            last_reported = price
        elif side == "SHORT" and last_reported - price >= step:
            send_telegram(f"📉 {symbol} Price Down → {price:.2f}")
            last_reported = price

        if side == "LONG" and price >= targets[0]:
            send_telegram(f"🎯 Target hit: {symbol} LONG | Price: {price:.2f}")
            targets.pop(0); last_reported = price
            if not targets: break
        elif side == "SHORT" and price <= targets[0]:
            send_telegram(f"🎯 Target hit: {symbol} SHORT | Price: {price:.2f}")
            targets.pop(0); last_reported = price
            if not targets: break

        if side == "LONG" and price <= sl:
            send_telegram(f"🛑 SL hit: {symbol} LONG | Price: {price:.2f}")
            break
        if side == "SHORT" and price >= sl:
            send_telegram(f"🛑 SL hit: {symbol} SHORT | Price: {price:.2f}")
            break

        time.sleep(POLL_SECS)

def send_alert(symbol, side, entry, sl, targets, mode_label):
    """Send trading alert"""
    msg = (
        f"<b>📡 SIGNAL DETECTED</b>\n\n"
        f"<b>Asset:</b> {symbol}\n"
        f"<b>Strategy:</b> {mode_label}\n"
        f"<b>Direction:</b> {side}\n"
        f"<b>Entry Price:</b> ${entry:.2f}\n"
        f"<b>Stop Loss:</b> ${sl:.2f}\n\n"
        f"<b>Target Levels:</b>\n"
    )
    for t in targets:
        msg += f"🎯 ${t:.2f}\n"
    msg += f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    send_telegram(msg)

def analyze_symbol(symbol, mode_name):
    """Analyze single symbol with given mode"""
    cfg = MODES[mode_name]
    df = get_candles(symbol, cfg["interval"])
    
    if df.empty:
        return
    
    df = add_indicators(df)

    price = df["close"].iloc[-1]
    ema = df["ema"].iloc[-1]
    rsi = df["rsi"].iloc[-1]

    side = None
    if check_conditions(price, ema, rsi, cfg, "LONG"):
        side = "LONG"
    elif check_conditions(price, ema, rsi, cfg, "SHORT"):
        side = "SHORT"
    else:
        return

    entry = compute_entry_price(df, side, cfg, symbol)
    if entry is None:
        return

    sl, tps = calculate_levels(entry, side)
    send_alert(symbol, side, entry, sl, tps, mode_name)
    wait_for_entry(symbol, side, entry)
    monitor_position(symbol, side, entry, sl, tps)

# =================== MAIN ==========================
def main():
    """Main execution loop"""
    print("Starting market analysis system...")
    print(f"Monitoring symbols: {SYMBOLS}")
    
    while True:
        try:
            for sym in SYMBOLS:
                analyze_symbol(sym, "SCALP")
                analyze_symbol(sym, "SWING")
                
            time.sleep(SCAN_SECS)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            send_telegram(f"❌ System error: {str(e)[:100]}")
            time.sleep(SCAN_SECS)

if __name__ == "__main__":
    main()
