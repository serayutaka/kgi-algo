import numpy as np
import pandas as pd
import os
from datetime import datetime

# Define paths for reading and writing files
file_path = 'Daily_Ticks.csv'

# Define the base directory dynamically based on the OS
home_dir = os.path.expanduser('~')  # Expands to the user's home directory
previous_file_path = os.path.join(home_dir, 'Desktop', 'Previous', 'Result.csv')

output_dir = os.path.join(home_dir, 'Desktop', 'competition_api', 'Day1')
os.makedirs(os.path.join(output_dir, 'portfolio'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'statement'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'summary'), exist_ok=True)

print(f"Previous file path: {previous_file_path}")
print(f"Output directories created under: {output_dir}")

# Read the input data
data = pd.read_csv(file_path)
# previous_data = pd.read_csv(previous_file_path)

# Preprocess the data
data['TradeDateTime'] = pd.to_datetime(data['TradeDateTime'])  # Convert to datetime
data.sort_values('TradeDateTime', inplace=True)  # Sort by datetime

# Define technical indicators
def calculate_moving_average(df, window=5):
    df[f'MA_{window}'] = df['LastPrice'].rolling(window=window).mean()
    return df

def calculate_volume_signal(df, threshold=1000):
    df['VolumeSignal'] = np.where(df['Volume'] > threshold, 'Buy', 'Hold')
    return df

# Apply technical indicators
data = calculate_moving_average(data, window=5)
data = calculate_volume_signal(data, threshold=1000)

data.to_csv('processed_data.csv', index=False)

# Portfolio settings
initial_cash = 10_000_000
portfolio = {
    "cash": initial_cash,
    "stocks": {},  # Dictionary to store stock holdings
}
nav = [10_000_000]

# Function to calculate portfolio value
def calculate_portfolio_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return portfolio["cash"], stock_value, total_value

def calculate_total_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return total_value

statements = []
number_of_wins = 0
queue_for_calculate_pl = []
# Function to execute a trade
def execute_trade(stock_code, price, volume, trade_type, date, time):
    global portfolio
    global number_of_wins
    
    # Calculate trade value
    trade_value = price * volume
    
    if trade_type == "Buy":
        if portfolio["cash"] >= trade_value:
            portfolio["cash"] -= trade_value
            portfolio["stocks"][stock_code] = portfolio["stocks"].get(stock_code, 0) + volume
            print(f"{date} {time} Bought {volume} of {stock_code} at {price} THB, amount cost: {trade_value} THB")
            statements.append({
                "Table Name": "Statement",
                "File Name": "017_วิศวะเซินเจิ้น.py",
                "Stock Name": stock_code,
                "Date": date,
                "Time": time,
                "Side": "Buy",
                "Volume": volume,
                "Price": price,
                "Amount cost": trade_value,
                "End_line_available": portfolio["cash"]
            })
            if stock_code not in queue_for_calculate_pl:
                queue_for_calculate_pl.append({"Stock Name": stock_code, "Price": trade_value})
        else:
            print(f"Not enough cash to buy {volume} of {stock_code} at {price} THB")
    elif trade_type == "Sell":
        if portfolio["stocks"].get(stock_code, 0) >= volume:
            portfolio["cash"] += trade_value
            portfolio["stocks"][stock_code] -= volume
            print(f"{date} {time} Sold {volume} of {stock_code} at {price} THB, amount cost: {trade_value} THB")
            statements.append({
                "Table Name": "Statement",
                "File Name": "017_วิศวะเซินเจิ้น.py",
                "Stock Name": stock_code,
                "Date": date,
                "Time": time,
                "Side": "Sell",
                "Volume": volume,
                "Price": price,
                "Amount cost": trade_value,
                "End_line_available": portfolio["cash"]
            })
            for obj in queue_for_calculate_pl:
                if obj["Stock Name"] == stock_code:
                    sell_price = trade_value
                    buy_price = obj["Price"]
                    profit_or_loss = (sell_price - buy_price) * volume
                    if profit_or_loss > 0:
                        number_of_wins += 1
                    queue_for_calculate_pl.remove(obj)
        else:
            print(f"Not enough stocks to sell {volume} of {stock_code} at {price} THB")

last_prices = {}
last_prices_buy = {}
last_prices_sell = {}
match_trades = 0

for _, row in data.iterrows():
    stock_code = row["ShareCode"]
    price = row["LastPrice"]
    volume = row["Volume"]
    flag = row["Flag"]
    
    if flag not in ["Buy", "Sell"]:
        continue

    if row['VolumeSignal'] == 'Buy' and row['Flag'] == 'Sell':
        cash = portfolio["cash"]
        shares_to_buy = cash // price
        while shares_to_buy % 100 != 0:
            shares_to_buy -= 1
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            if row["Volume"] < shares_to_buy:
                shares_to_buy = row["Volume"]
            execute_trade(stock_code, price, shares_to_buy, "Buy", row['TradeDateTime'].date(), row['TradeDateTime'].time())
            match_trades += 1
            last_prices_buy[stock_code] = price
            nav.append(calculate_total_value(last_prices))
    
    elif row['VolumeSignal'] == 'Hold' and row['Flag'] == 'Buy':
        if stock_code in last_prices:
            shares_to_sell = portfolio["stocks"].get(stock_code, 0)
            while shares_to_sell % 100 != 0:
                shares_to_sell -= 1
            if shares_to_sell > 0:
                execute_trade(stock_code, price, shares_to_sell, "Sell", row['TradeDateTime'].date(), row['TradeDateTime'].time())
                match_trades += 1
                last_prices_sell[stock_code] = price
                nav.append(calculate_total_value(last_prices))

                
    
    # Update last prices
    last_prices[stock_code] = price

# Final portfolio summary
cash_balance, stock_value, total_value = calculate_portfolio_value(last_prices)
print("\nEnd of Day Portfolio Summary:")
print(f"Cash Balance: {cash_balance:.2f} THB")
print(f"Stock Holdings Value: {stock_value:.2f} THB")
print(f"Total Portfolio Value: {total_value:.2f} THB")

start_value = initial_cash
end_value = total_value
return_percentage = (end_value - start_value) / start_value * 100
print(f"\n% Return: {return_percentage:.2f}%")

# Export Statements table to CSV
statements_df = pd.DataFrame(statements)
statements_df.to_csv(f'{output_dir}/statement/วิศวะเซินเจิ้น_Statement_day1_2024.csv', index=False)

# Prepare the Portfolio Table for export
portfolio_data = []

# Filter rows with 'OPEN1_E' flag for each Share Code
open1e_rows = data[data['Flag'] == 'OPEN1_E'].groupby('ShareCode').last()

# Filter rows with 'ATC' flag for each Share Code
atc_rows = data[data['Flag'] == 'ATC'].groupby('ShareCode').last()

# Reset index to make Share Code a column (optional)
open1e_rows_reset = open1e_rows.reset_index()
atc_rows_reset = atc_rows.reset_index()

buy_summary = statements_df[statements_df['Side'] == 'Buy'].groupby('Stock Name').agg({
    'Price': 'sum',
    'Volume': 'sum'
}).reset_index()
sell_summary = statements_df[statements_df['Side'] == 'Sell'].groupby('Stock Name').agg({
    'Price': 'sum',
    'Volume': 'sum'
}).reset_index()

for stock_code, volume in portfolio["stocks"].items():
    if volume == 0:
        continue
    price = last_prices.get(stock_code, 0)
    amount_cost = buy_summary["Price"].loc[buy_summary["Stock Name"] == stock_code].values[0]
    market_value = atc_rows_reset["Value"].loc[atc_rows_reset["ShareCode"] == stock_code].values[0]
    market_price = atc_rows_reset["LastPrice"].loc[atc_rows_reset["ShareCode"] == stock_code].values[0]
    unrealized_pl = (market_price - last_prices_buy.get(stock_code, 0)) * volume
    unrealized_pl_pct = (market_price - last_prices_buy.get(stock_code, 0)) / last_prices_buy.get(stock_code, 0) * 100
    realized_pl = (sell_summary["Price"].loc[sell_summary["Stock Name"] == "TLI"].values[0] - amount_cost) * sell_summary["Volume"].loc[sell_summary["Stock Name"] == "TLI"].values[0]
    
    portfolio_data.append({
        'Table Name': 'Portfolio',
        'File Name': '017_วิศวะเซินเจิ้น.py',
        'Stock Name': stock_code,
        'Start Vol': 0,
        'Actual Vol': "{:.4f}".format(volume),
        'Avg Cost': "{:.4f}".format(amount_cost / volume if volume > 0 else 0),
        'Market Price': "{:.4f}".format(market_price),
        'Amount Cost': "{:.4f}".format(amount_cost),
        'Market Value': "{:.4f}".format(market_value),
        'Unrealized P/L': "{:.4f}".format(unrealized_pl),
        '%Unrealized P/L': "{:.4f}".format(unrealized_pl_pct),
        'Realized P/L': "{:.4f}".format(realized_pl)
    })

portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df.to_csv(f'{output_dir}/portfolio/วิศวะเซินเจิ้น_Portfolio_day1_2024.csv', index=False)

annualized_return = (total_value / initial_cash) ** (1 / 365) - 1
relative_drawdown = (total_value - max(nav)) / max(nav) * 100 if max(nav) > 0 else 0
maximum_drawdown = (min(nav) - max(nav)) / max(nav) * 100 if max(nav) > 0 else 0
calmar_ratio = annualized_return / maximum_drawdown * 100 if maximum_drawdown > 0 else 0

# Prepare the Summary Table for export
summary_data = [{
    'Table Name': 'Summary',
    'File Name': '017_วิศวะเซินเจิ้น.py',
    'End Line Available': portfolio["cash"],
    'Start Line Available': portfolio["cash"],
    'Number of Wins': number_of_wins,
    'Number of Matched Trades': match_trades,
    'Number of Transactions': len(statements),
    'Sum of Unrealized P/L': "{:.4f}".format(sum([float(row['Unrealized P/L']) for _, row in portfolio_df.iterrows()])),
    'Sum of %Unrealized P/L': "{:.4f}".format(sum([float(row['%Unrealized P/L']) for _, row in portfolio_df.iterrows()])),
    'Sum of Realized P/L': "{:.4f}".format(sum([float(row['Realized P/L']) for _, row in portfolio_df.iterrows()])),
    'Maximum Value': "{:.4f}".format(max(nav)),
    'Minimum Value': "{:.4f}".format(min(nav)),
    'Win Rate': "{:.4f}".format(number_of_wins / len(statements) * 100),
    'Calmar Ratio': "{:.4f}".format(calmar_ratio),
    'Relative Drawdown': "{:.4f}".format(relative_drawdown),
    'Maximum Drawdown': "{:.4f}".format(maximum_drawdown),  
    '%Return': "{:.4f}".format(return_percentage)
}]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{output_dir}/summary/วิศวะเซินเจิ้น_Summary_day1_2024.csv', index=False)

print("CSV files exported successfully.")