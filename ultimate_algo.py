# SIMPLE PATTERN DETECTOR
import os
import time
import requests
import random
from datetime import datetime

print("=" * 50)
print("🚀 PATTERN DETECTION STARTED")
print("=" * 50)

# --- TELEGRAM SETUP --- 
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

print(f"🔧 BOT_TOKEN set: {'YES' if BOT_TOKEN else 'NO'}")
print(f"🔧 CHAT_ID set: {'YES' if CHAT_ID else 'NO'}")

def send_telegram(msg):
    """Send Telegram notification"""
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print(f"❌ Telegram not configured: {msg[:50]}...")
            return False
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Telegram sent: {msg[:30]}...")
            return True
        else:
            print(f"❌ Telegram error {response.status_code}: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram failed: {e}")
        return False

# --- SEND TEST MESSAGE ---
print("\n🧪 Testing Telegram connection...")
test_msg = "✅ GitHub Actions connected!\nPattern detection started at " + datetime.utcnow().strftime("%H:%M:%S UTC")
if send_telegram(test_msg):
    print("✅ Telegram test successful!")
else:
    print("❌ Telegram test failed")

# --- SIMULATED DATA ---
SYMBOLS = ["BTC", "ETH", "BNB", "SOL"]
prices = {
    "BTC": 45000 + random.uniform(-1000, 1000),
    "ETH": 2500 + random.uniform(-100, 100),
    "BNB": 300 + random.uniform(-10, 10),
    "SOL": 100 + random.uniform(-5, 5)
}

# --- MAIN LOOP ---
print(f"\n📊 Starting monitoring at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
print("🔍 Checking patterns every 30 seconds...")

iteration = 0
while True:
    iteration += 1
    current_time = datetime.utcnow().strftime('%H:%M:%S')
    
    print(f"\n🔄 Run {iteration} - {current_time} UTC")
    print("-" * 40)
    
    # Simulate price changes
    for symbol in SYMBOLS:
        change = random.uniform(-2, 2)  # -2% to +2%
        old_price = prices[symbol]
        prices[symbol] = old_price * (1 + change/100)
        
        print(f"{symbol}: ${old_price:.2f} → ${prices[symbol]:.2f} ({change:+.2f}%)")
        
        # Detect pattern (simple threshold)
        if abs(change) > 1.5:  # Big move detected
            direction = "UP 📈" if change > 0 else "DOWN 📉"
            
            # Create Telegram message
            msg = (f"<b>{direction} DETECTED</b>\n"
                   f"🪙 {symbol}\n"
                   f"💰 ${prices[symbol]:.2f}\n"
                   f"📊 Change: {change:+.2f}%\n"
                   f"⏰ {current_time} UTC\n"
                   f"🏷️ Signal #{iteration:04d}")
            
            print(f"🚨 Pattern: {symbol} {direction} ({change:+.2f}%)")
            send_telegram(msg)
    
    # Also send heartbeat every 5 iterations
    if iteration % 5 == 0:
        heartbeat = f"💓 Heartbeat: Run #{iteration} at {current_time} UTC"
        send_telegram(heartbeat)
    
    print(f"⏳ Next check in 30 seconds...")
    time.sleep(30)
