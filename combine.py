import pandas as pd
import numpy as np

# Initial portfolio settings
initial_cash = 10_000_000
portfolio = {
    "cash": initial_cash,
    "stocks": {},  # Dictionary to store stock holdings
}

# Track performance metrics
total_trades = 0
winning_trades = 0
portfolio_values = [initial_cash]  # Track portfolio value over time for drawdown calculation
peak_value = initial_cash  # Initialize the peak value for drawdown calculation

# Function to execute a trade
def execute_trade(stock_code, price, volume, trade_type):
    global portfolio, total_trades, winning_trades, portfolio_values, peak_value
    
    # Calculate trade value
    trade_value = price * volume
    
    # Store initial cash value before trade
    cash_before = portfolio["cash"]
    
    if trade_type == "Buy":
        # Check if enough cash is available
        if portfolio["cash"] >= trade_value:
            portfolio["cash"] -= trade_value
            portfolio["stocks"][stock_code] = portfolio["stocks"].get(stock_code, 0) + volume
            print(f"Bought {volume} of {stock_code} at {price} THB")
        else:
            print(f"Not enough cash to buy {volume} of {stock_code} at {price} THB")
    elif trade_type == "Sell":
        # Check if enough stocks are available
        if portfolio["stocks"].get(stock_code, 0) >= volume:
            portfolio["cash"] += trade_value
            portfolio["stocks"][stock_code] -= volume
            print(f"Sold {volume} of {stock_code} at {price} THB")
        else:
            print(f"Not enough stocks to sell {volume} of {stock_code} at {price} THB")
    
    # Calculate portfolio value after the trade
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    portfolio_values.append(total_value)
    
    # Track maximum drawdown
    peak_value = max(peak_value, total_value)
    
    # Calculate win rate for individual trades (based on cash difference)
    if trade_type == "Sell" and total_value > cash_before:
        winning_trades += 1
    total_trades += 1

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate portfolio value
def calculate_portfolio_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return portfolio["cash"], stock_value, total_value

# Load the CSV file
file_path = "Daily_Ticks.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

last_prices = {}
short_window = 5  # Short-term moving average 5 ticks
long_window = 20  # Long-term moving average 20 ticks

# Calculate moving averages
data["ShortMA"] = data["LastPrice"].rolling(window=short_window).mean()
data["LongMA"] = data["LastPrice"].rolling(window=long_window).mean()

# Precompute RSI values for the strategy
data["RSI"] = calculate_rsi(data["LastPrice"])

# Strategy: Buy when short-term MA crosses above long-term MA and RSI < 30
#           Sell when short-term MA crosses below long-term MA and RSI > 70

for i in range(long_window, len(data)):
    stock_code = data.loc[i, "ShareCode"]
    last_price = data.loc[i, "LastPrice"]
    short_ma = data.loc[i, "ShortMA"]
    long_ma = data.loc[i, "LongMA"]
    rsi = data.loc[i, "RSI"]

    # Skip "OPEN1_E" or other non-trade rows
    if data.loc[i, "Flag"] not in ["Buy", "Sell"]:
        continue

    # Buy signal
    if short_ma > long_ma and data.loc[i-1, "ShortMA"] <= data.loc[i-1, "LongMA"] and rsi < 30:
        execute_trade(stock_code, last_price, 100, "Buy")
    # Sell signal
    elif short_ma < long_ma and data.loc[i-1, "ShortMA"] >= data.loc[i-1, "LongMA"] and rsi > 70:
        execute_trade(stock_code, last_price, 100, "Sell")

# Display the final portfolio at the end of the day
cash_balance, stock_value, total_value = calculate_portfolio_value(last_prices)
print("\nEnd of Day Portfolio Summary:")
print(f"Cash Balance: {cash_balance:.2f} THB")
print(f"Stock Holdings Value: {stock_value:.2f} THB")
print(f"Total Portfolio Value: {total_value:.2f} THB")

# Calculate % Return
start_value = initial_cash
end_value = total_value
return_percentage = (end_value - start_value) / start_value * 100
print(f"\n% Return: {return_percentage:.2f}%")

# Calculate % Maximum Drawdown
drawdown = (peak_value - min(portfolio_values)) / peak_value * 100
print(f"% Maximum Drawdown: {drawdown:.2f}%")

# Calculate % Win Rate
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
print(f"% Win Rate: {win_rate:.2f}%")
