import os
import time
import requests
import sys

# --- CONFIGURATION ---
SYSTEM_KEY = os.getenv("SYSTEM_KEY")  # Your BingX API key
SYSTEM_SECRET = os.getenv("SYSTEM_SECRET")  # Your BingX secret
API_BASE = "https://open-api.bingx.com"

def test_api_connection():
    """Test basic API connection"""
    try:
        # Simple endpoint to check connectivity
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = "symbol=BTC-USDT"
        
        url = f"{API_BASE}{endpoint}?{params}"
        print(f"🔗 Testing connection to: {url}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status Code: {data.get('code')}")
            print(f"✅ API Message: {data.get('msg')}")
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def fetch_btc_price():
    """Fetch current BTC price from BingX"""
    try:
        endpoint = "/openApi/swap/v2/quote/ticker"
        params = "symbol=BTC-USDT"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                ticker_data = data['data']
                price = float(ticker_data['lastPrice'])
                change_percent = float(ticker_data['priceChangePercent'])
                volume = float(ticker_data['volume'])
                
                return {
                    'price': price,
                    'change_percent': change_percent,
                    'volume': volume,
                    'high': float(ticker_data['highPrice']),
                    'low': float(ticker_data['lowPrice']),
                    'timestamp': int(time.time())
                }
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching price: {e}")
    
    return None

def fetch_order_book():
    """Fetch BTC order book for depth analysis"""
    try:
        endpoint = "/openApi/swap/v2/quote/depth"
        params = "symbol=BTC-USDT&limit=5"
        
        url = f"{API_BASE}{endpoint}?{params}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                book_data = data['data']
                
                # Calculate bid/ask totals
                bids_total = sum(float(bid[1]) for bid in book_data['bids'][:3])
                asks_total = sum(float(ask[1]) for ask in book_data['asks'][:3])
                
                return {
                    'bids_total': bids_total,
                    'asks_total': asks_total,
                    'imbalance': (bids_total - asks_total) / (bids_total + asks_total) if (bids_total + asks_total) > 0 else 0
                }
                
    except Exception as e:
        print(f"❌ Error fetching order book: {e}")
    
    return None

def main():
    """Main loop to fetch BTC price repeatedly"""
    print("=" * 50)
    print("🔍 BINGX BTC PRICE MONITOR")
    print("=" * 50)
    
    # Test connection first
    if not test_api_connection():
        print("\n❌ Failed to connect to BingX API. Please check:")
        print("   1. SYSTEM_KEY environment variable")
        print("   2. SYSTEM_SECRET environment variable")
        print("   3. Internet connection")
        print("   4. API permissions")
        sys.exit(1)
    
    print("\n✅ API Connection Successful!")
    print("📡 Starting BTC price monitoring...\n")
    
    # Monitoring loop
    iteration = 0
    while True:
        iteration += 1
        try:
            timestamp = time.strftime('%H:%M:%S')
            print(f"\n🔄 Iteration #{iteration} | {timestamp}")
            print("-" * 40)
            
            # Fetch price
            price_data = fetch_btc_price()
            
            if price_data:
                change_icon = "📈" if price_data['change_percent'] >= 0 else "📉"
                
                print(f"💰 BTC Price: ${price_data['price']:,.2f}")
                print(f"{change_icon} 24h Change: {price_data['change_percent']:.2f}%")
                print(f"📊 24h Volume: ${price_data['volume']:,.0f}")
                print(f"⬆️  24h High: ${price_data['high']:,.2f}")
                print(f"⬇️  24h Low: ${price_data['low']:,.2f}")
                
                # Fetch order book
                book_data = fetch_order_book()
                if book_data:
                    imbalance_icon = "⚖️"
                    if book_data['imbalance'] > 0.1:
                        imbalance_icon = "🟢"
                    elif book_data['imbalance'] < -0.1:
                        imbalance_icon = "🔴"
                    
                    print(f"\n📊 Order Book Depth:")
                    print(f"   Bid Total: {book_data['bids_total']:.2f} BTC")
                    print(f"   Ask Total: {book_data['asks_total']:.2f} BTC")
                    print(f"   {imbalance_icon} Imbalance: {book_data['imbalance']:+.3f}")
            
            else:
                print("❌ Failed to fetch price data")
            
            # Wait before next fetch
            print(f"\n⏳ Next update in 10 seconds...")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error in monitoring loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    # Check environment variables
    if not SYSTEM_KEY or not SYSTEM_SECRET:
        print("❌ Missing environment variables!")
        print("Please set:")
        print("  export SYSTEM_KEY='your_bingx_api_key'")
        print("  export SYSTEM_SECRET='your_bingx_secret'")
        sys.exit(1)
    
    main()
