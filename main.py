"""
The most return percentage right now.
"""

import numpy as np
import pandas as pd

file_path = 'Daily_Ticks.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head(), data.info()

# Step 1: Preprocess the data
data['TradeDateTime'] = pd.to_datetime(data['TradeDateTime'])  # Convert to datetime
data.sort_values('TradeDateTime', inplace=True)  # Sort by datetime

# Step 2: Define technical indicators
def calculate_moving_average(df, window=5):
    """Calculate moving average for the last price."""
    df[f'MA_{window}'] = df['LastPrice'].rolling(window=window).mean()
    return df

def calculate_volume_signal(df, threshold=1000):
    """Generate buy/sell signals based on volume."""
    df['VolumeSignal'] = np.where(df['Volume'] > threshold, 'Buy', 'Hold')
    return df

# Apply technical indicators
data = calculate_moving_average(data, window=5)
data = calculate_volume_signal(data, threshold=1000)

data.to_csv('processed_data.csv', index=False)

# Initial portfolio settings
initial_cash = 10_000_000
portfolio = {
    "cash": initial_cash,
    "stocks": {},  # Dictionary to store stock holdings
}

# Function to calculate portfolio value
def calculate_portfolio_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return portfolio["cash"], stock_value, total_value

# Function to execute a trade
def execute_trade(stock_code, price, volume, trade_type):
    global portfolio
    
    # Calculate trade value
    trade_value = price * volume
    
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

last_prices = {}

for _, row in data.iterrows():
    stock_code = row["ShareCode"]
    price = row["LastPrice"]
    volume = row["Volume"]
    flag = row["Flag"]
    
    # Skip "OPEN1_E" or other non-trade rows
    if flag not in ["Buy", "Sell"]:
        continue

    if row['VolumeSignal'] == 'Buy' and row['Flag'] == 'Sell':
        cash = portfolio["cash"]
        shares_to_buy = cash // price
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            if row["Volume"] < shares_to_buy:
                shares_to_buy = row["Volume"]
        execute_trade(stock_code, price, shares_to_buy, "Buy")
    
    elif row['VolumeSignal'] == 'Hold' and row['Flag'] == 'Buy':
        if stock_code in last_prices:
            shares_to_sell = portfolio["stocks"].get(stock_code, 0)
            if shares_to_sell > 0:
                execute_trade(stock_code, price, shares_to_sell, "Sell")
    # Update last prices
    last_prices[stock_code] = price

# Display the final portfolio at the end of the day
cash_balance, stock_value, total_value = calculate_portfolio_value(last_prices)
print("\nEnd of Day Portfolio Summary:")
print(f"Cash Balance: {cash_balance:.2f} THB")
print(f"Stock Holdings Value: {stock_value:.2f} THB")
print(f"Total Portfolio Value: {total_value:.2f} THB")

start_value = initial_cash
end_value = total_value
return_percentage = (end_value - start_value) / start_value * 100
print(f"\n% Return: {return_percentage:.2f}%")