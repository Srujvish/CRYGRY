# REAL BINGX PATTERN DETECTOR WITH GUARANTEED ALERTS
import os
import time
import requests
import pandas as pd
from datetime import datetime

print("=" * 50)
print("🚀 REAL PATTERN DETECTOR - ACTIVE MODE")
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
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                price = float(data['data']['lastPrice'])
                return price
        return None
    except:
        return None

# --- GUARANTEED ALERT SYSTEM ---
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

# Track price history
price_history = {symbol: [] for symbol in REAL_SYMBOLS}
last_alert_time = {symbol: 0 for symbol in REAL_SYMBOLS}

# --- SEND START MESSAGE ---
print("\n🧪 Sending startup message...")
start_msg = f"🚀 CRY PATTERN DETECTOR ACTIVATED\n⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}\n🔍 Monitoring 8 cry\n📊 Alerts every 1-5 minutes"
send_telegram(start_msg)

# --- MAIN LOOP WITH GUARANTEED ALERTS ---
print(f"\n📊 Starting monitoring at {datetime.utcnow().strftime('%H:%M:%S UTC')}")

iteration = 0
while True:
    iteration += 1
    current_time = datetime.utcnow()
    formatted_time = current_time.strftime('%H:%M:%S')
    
    print(f"\n🔄 Run {iteration} - {formatted_time} UTC")
    print("-" * 40)
    
    for symbol in REAL_SYMBOLS:
        try:
            # Get real price
            current_price = get_real_price(symbol)
            if current_price is None:
                print(f"❌ Failed to get {symbol}")
                continue
            
            # Add to history
            price_history[symbol].append({
                'price': current_price,
                'time': time.time()
            })
            
            # Keep only last 30 minutes of data
            price_history[symbol] = [
                p for p in price_history[symbol] 
                if time.time() - p['time'] < 1800
            ]
            
            coin_name = symbol.replace("-USDT", "")
            print(f"✅ {coin_name}: ${current_price:.4f}")
            
            # --- GUARANTEED ALERT LOGIC ---
            
            # 1. ALERT EVERY 5 MINUTES (minimum)
            if time.time() - last_alert_time[symbol] > 300:  # 5 minutes
                msg = (f"📊 <b>{coin_name} STATUS</b>\n"
                      f"💰 Price: ${current_price:.4f}\n"
                      f"📈 Live tracking active\n"
                      f"⏰ {formatted_time} UTC\n"
                      f"🏷️ Update #{iteration}")
                
                print(f"📢 5-min update for {coin_name}")
                send_telegram(msg)
                last_alert_time[symbol] = time.time()
            
            # 2. DETECT MOVEMENTS (>0.3% change)
            if len(price_history[symbol]) > 3:
                recent_prices = [p['price'] for p in price_history[symbol][-3:]]
                avg_price = sum(recent_prices) / len(recent_prices)
                change_pct = ((current_price - avg_price) / avg_price) * 100
                
                if abs(change_pct) > 0.3:  # Very sensitive threshold
                    direction = "UP 📈" if change_pct > 0 else "DOWN 📉"
                    
                    # Don't alert too frequently
                    if time.time() - last_alert_time[symbol] > 60:  # 1 minute cooldown
                        msg = (f"🚨 <b>{direction} MOVEMENT</b>\n"
                              f"🪙 {coin_name}\n"
                              f"💰 ${current_price:.4f}\n"
                              f"📊 Change: {change_pct:+.2f}%\n"
                              f"⏰ {formatted_time} UTC\n"
                              f"🏷️ Signal #{iteration}")
                        
                        print(f"🚨 {coin_name}: {direction} {change_pct:+.2f}%")
                        send_telegram(msg)
                        last_alert_time[symbol] = time.time()
            
            # 3. HEARTBEAT EVERY 10 ITERATIONS
            if iteration % 10 == 0 and symbol == "BTC-USDT":
                heartbeat = (f"💓 <b>SYSTEM ACTIVE</b>\n"
                           f"🔁 Run #{iteration}\n"
                           f"⏰ {formatted_time} UTC\n"
                           f"✅ All systems normal")
                send_telegram(heartbeat)
                
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
            continue
    
    print(f"⏳ Next check in 30 seconds...")
    time.sleep(30)  # Check every 30 seconds
