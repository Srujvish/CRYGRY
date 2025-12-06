# MARKET ANALYSIS AND PATTERN DETECTION SYSTEM
import os
import time
import requests
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# DEBUG: Check environment
print("🔧 DEBUG: Script starting")
print(f"🔧 Python version: {pd.__version__}")

# --- API CONFIGURATION --- 
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET") 
BASE_URL = "https://open-api.bingx.com"

# --- NOTIFICATION SERVICE --- 
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

print(f"🔧 DEBUG: API_KEY exists: {bool(API_KEY)}")
print(f"🔧 DEBUG: BOT_TOKEN exists: {bool(BOT_TOKEN)}")

# --------- SIMPLIFIED SYMBOL LIST ---------
SYMBOLS = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT", 
    "BNB": "BNB-USDT"
}

def send_notification(msg):
    """Send Telegram notification"""
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print(f"📢 {msg}")
            return
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID, 
            "text": msg, 
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Telegram notification sent")
        else:
            print(f"❌ Telegram error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")

def get_market_data(symbol, interval="5m", limit=10):
    """Get market data from BingX"""
    try:
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = f"symbol={symbol}"
        
        url = f"{BASE_URL}{endpoint}?{params}"
        print(f"🔍 Fetching {symbol}...")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                price = float(data['data']['lastPrice'])
                print(f"✅ {symbol} price: ${price:.2f}")
                return price
            else:
                print(f"❌ API error: {data.get('msg')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
    return None

def test_api_connection():
    """Test if BingX API is working"""
    print("🧪 Testing API connection...")
    
    test_url = "https://open-api.bingx.com/openApi/swap/v2/quote/contracts"
    try:
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                print("✅ API connection successful")
                return True
            else:
                print(f"❌ API error: {data.get('msg')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    return False

class SimplePatternAI:
    def __init__(self):
        print("🤖 Simple Pattern AI initialized")
        
    def check_pattern(self, prices):
        """Simple pattern detection"""
        if len(prices) < 2:
            return None, 0.0
        
        # Simple moving average check
        current = prices[-1]
        previous = prices[-2] if len(prices) > 1 else prices[0]
        
        change = ((current - previous) / previous) * 100
        
        if change > 1.0:  # Up 1%
            return "UP", abs(change)
        elif change < -1.0:  # Down 1%
            return "DOWN", abs(change)
        
        return None, 0.0

# --------- MAIN EXECUTION ---------
print("=" * 50)
print("🚀 PATTERN DETECTION SYSTEM")
print("=" * 50)

# Test API first
if test_api_connection():
    # Initialize AI
    pattern_ai = SimplePatternAI()
    
    # Track prices
    price_history = {symbol: [] for symbol in SYMBOLS.values()}
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n🔄 Iteration {iteration} - {datetime.utcnow().strftime('%H:%M:%S')} UTC")
        
        for asset_name, symbol in SYMBOLS.items():
            price = get_market_data(symbol)
            
            if price:
                # Add to history
                price_history[symbol].append(price)
                if len(price_history[symbol]) > 10:
                    price_history[symbol] = price_history[symbol][-10:]
                
                # Check pattern
                direction, confidence = pattern_ai.check_pattern(price_history[symbol])
                
                if direction and confidence > 1.5:
                    print(f"✅ Pattern detected: {asset_name} {direction} ({confidence:.1f}%)")
                    
                    # Send Telegram notification
                    message = (f"📊 <b>{direction} PATTERN DETECTED</b>\n"
                              f"🎯 {asset_name}\n"
                              f"💰 Price: ${price:.2f}\n"
                              f"📈 Change: {confidence:.1f}%\n"
                              f"🕐 Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
                    
                    send_notification(message)
        
        print(f"⏳ Waiting 60 seconds...")
        time.sleep(60)
        
else:
    print("❌ Cannot connect to API. Exiting...")
