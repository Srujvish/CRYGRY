# REAL BINGX PATTERN DETECTOR
import os
import time
import requests
import pandas as pd
from datetime import datetime

print("=" * 50)
print("🚀 REAL BINGX PATTERN DETECTOR")
print("=" * 50)

# --- TELEGRAM SETUP --- 
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

print(f"🔧 BOT_TOKEN: {'✅ Set' if BOT_TOKEN else '❌ Missing'}")
print(f"🔧 CHAT_ID: {'✅ Set' if CHAT_ID else '❌ Missing'}")

def send_telegram(msg):
    """Send Telegram notification"""
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print(f"❌ Telegram not configured")
            return False
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Telegram sent")
            return True
        else:
            print(f"❌ Telegram error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram failed: {e}")
        return False

# --- REAL BINGX API FUNCTIONS ---
BINGX_BASE_URL = "https://open-api.bingx.com"

def get_real_price(symbol):
    """Get REAL price from BingX"""
    try:
        endpoint = "/openApi/swap/v2/quote/ticker"
        url = f"{BINGX_BASE_URL}{endpoint}?symbol={symbol}"
        
        print(f"📡 Fetching {symbol}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                price = float(data['data']['lastPrice'])
                print(f"✅ {symbol}: ${price:.2f}")
                return price
            else:
                print(f"❌ API error for {symbol}: {data.get('msg')}")
        else:
            print(f"❌ HTTP error for {symbol}: {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
    
    return None

def get_klines(symbol, interval="5m", limit=10):
    """Get candle data for pattern detection"""
    try:
        endpoint = "/openApi/swap/v3/quote/klines"
        url = f"{BINGX_BASE_URL}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                klines = data['data']
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                return df
    except Exception as e:
        print(f"❌ Klines error for {symbol}: {e}")
    
    return None

def detect_real_pattern(symbol, df):
    """Detect REAL patterns from candle data"""
    if df is None or len(df) < 5:
        return None, 0.0
    
    try:
        # Get latest candles
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate percentage change
        current_price = latest['close']
        prev_price = prev['close']
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Calculate volume change
        current_volume = latest['volume']
        avg_volume = df['volume'].tail(5).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Pattern conditions
        if abs(change_pct) > 1.0 and volume_ratio > 2.0:
            direction = "UP 📈" if change_pct > 0 else "DOWN 📉"
            return direction, abs(change_pct), current_price
        
    except Exception as e:
        print(f"❌ Pattern detection error: {e}")
    
    return None, 0.0, 0.0

# --- REAL SYMBOLS ---
REAL_SYMBOLS = [
    "BTC-USDT",
    "ETH-USDT", 
    "BNB-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "DOT-USDT"
]

# --- SEND START MESSAGE ---
print("\n🧪 Testing Telegram...")
start_msg = f"🚀 REAL Pattern Detection Started\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
if send_telegram(start_msg):
    print("✅ Telegram connected!")

# --- TEST BINGX API ---
print("\n🧪 Testing BingX API...")
test_price = get_real_price("BTC-USDT")
if test_price:
    print(f"✅ BingX API working! BTC: ${test_price:.2f}")
else:
    print("❌ BingX API not working")
    # Exit if API not working
    exit(1)

# --- MAIN LOOP WITH REAL DATA ---
print(f"\n📊 Starting REAL monitoring at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
print(f"🔍 Checking {len(REAL_SYMBOLS)} symbols every 60 seconds...")

# Store price history
price_history = {symbol: [] for symbol in REAL_SYMBOLS}

iteration = 0
while True:
    iteration += 1
    current_time = datetime.utcnow().strftime('%H:%M:%S')
    
    print(f"\n🔄 Run {iteration} - {current_time} UTC")
    print("-" * 40)
    
    signals_detected = 0
    
    for symbol in REAL_SYMBOLS:
        try:
            # Get real price
            current_price = get_real_price(symbol)
            if current_price is None:
                continue
            
            # Add to history
            price_history[symbol].append(current_price)
            if len(price_history[symbol]) > 20:
                price_history[symbol] = price_history[symbol][-20:]
            
            # Get klines for pattern detection
            df = get_klines(symbol, interval="5m", limit=10)
            
            # Detect pattern
            direction, change_pct, pattern_price = detect_real_pattern(symbol, df)
            
            if direction:
                signals_detected += 1
                
                # Create Telegram message
                coin_name = symbol.replace("-USDT", "")
                msg = (f"<b>{direction} PATTERN</b>\n"
                      f"🪙 {coin_name}\n"
                      f"💰 ${pattern_price:.2f}\n"
                      f"📊 Change: {change_pct:.2f}%\n"
                      f"⏰ {current_time} UTC\n"
                      f"🏷️ Signal #{iteration:04d}")
                
                print(f"🚨 {coin_name}: {direction} {change_pct:.2f}%")
                send_telegram(msg)
                
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue
    
    # Send summary if signals found
    if signals_detected > 0:
        print(f"✅ Found {signals_detected} pattern(s)")
    
    # Heartbeat every 10 iterations
    if iteration % 10 == 0:
        heartbeat = f"💓 Run #{iteration} - {signals_detected} signals at {current_time} UTC"
        send_telegram(heartbeat)
    
    print(f"⏳ Next check in 60 seconds...")
    time.sleep(60)
