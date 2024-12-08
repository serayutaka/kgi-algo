import pandas as pd

# Initial portfolio settings
initial_cash = 10_000_000
portfolio = {
    "cash": initial_cash,
    "stocks": {},  # Dictionary to store stock holdings
}

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

# Function to calculate portfolio value
def calculate_portfolio_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return portfolio["cash"], stock_value, total_value

# Load the CSV file
file_path = "Daily_Ticks.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Sample trading logic: Buy when the price drops from the previous tick
last_prices = {}

for _, row in data.iterrows():
    stock_code = row["ShareCode"]
    price = row["LastPrice"]
    volume = row["Volume"]
    flag = row["Flag"]
    
    # Skip "OPEN1_E" or other non-trade rows
    if flag not in ["Buy", "Sell"]:
        continue

    # Simple strategy: buy 100 units when the price drops
    if stock_code in last_prices and price < last_prices[stock_code]:
        execute_trade(stock_code, price, 100, "Buy")
    # Otherwise, if price increases, sell 100 units
    elif stock_code in last_prices and price > last_prices[stock_code]:
        execute_trade(stock_code, price, 100, "Sell")
    
    # Update the last known price
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
