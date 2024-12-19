import numpy as np
import pandas as pd
import os
import time

start = time.time()

# Define the base directory dynamically based on the OS
home_dir = os.path.expanduser('~')  # Expands to the user's home directory

# Define paths for reading and writing files
file_path = os.path.join(home_dir, 'Desktop', 'Daily_Ticks.csv')

output_dir = os.path.join(home_dir, 'Desktop', 'competition_api', 'Result')
os.makedirs(os.path.join(output_dir, 'portfolio'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'statement'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'summary'), exist_ok=True)

def load_previous(file_type, teamName):
    output_dir = os.path.expanduser("~/Desktop/competition_api")
    folder_path = os.path.join(output_dir, "Previous", file_type)
    file_path = os.path.join(folder_path, f"{teamName}_{file_type}.csv")

    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            print(f"Loaded '{file_type}' data for team {teamName}.")
            return data
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None
    
print(f"Output directories created under: {output_dir}")

data = pd.read_csv(file_path)

# Preprocess the data
data['TradeDateTime'] = pd.to_datetime(data['TradeDateTime'])  # Convert to datetime
data.sort_values('TradeDateTime', inplace=True)  # Sort by datetime

def calculate_indicators(df):
    # Group once to improve performance
    grouped = df.groupby('ShareCode')
    
    # Moving Average - use more efficient rolling
    df['MA_3'] = grouped['LastPrice'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # MACD - simplified calculation
    df['MACD'] = grouped['LastPrice'].transform(lambda x: 
        x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()
    )
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Vectorized Volume Signal
    condition_1 = df['LastPrice'] < df['MA_3']
    condition_2 = df['MACD'] > df['MACD_Signal']
    
    passed_conditions = (
        condition_1.astype(int) + 
        condition_2.astype(int)
    )
    
    df['VolumeSignal'] = np.where(
        df['MACD'].isna(),
        np.where(df['Volume'] > 19000, 'Buy', 'Hold'),
        np.where(passed_conditions == 2, 'Buy', 'Hold')
    )

    df['FinalSignal'] = np.where(
        (df['VolumeSignal'] == 'Buy') & (df['Flag'] == 'Sell'), 'Buy', 
        np.where((df['VolumeSignal'] == 'Hold') & (df['Flag'] == 'Buy'), 'Sell', 'nan')
    )
    
    return data

# Apply technical indicators
data = calculate_indicators(data)

previous_summary = load_previous("summary", "017")
previous_portfolio = load_previous("portfolio", "017")
previous_statement = load_previous("statement", "017")

# Portfolio settings
initial_cash = previous_summary['End Line Available'].iloc[-1] if isinstance(previous_summary, pd.DataFrame) else 10_000_000
portfolio = {
    "cash": initial_cash,
    "stocks": previous_portfolio.set_index('Stock Name')['Actual Vol'].to_dict() if isinstance(previous_portfolio, pd.DataFrame) else {}
}
print(f"Initial Cash: {initial_cash}")
print(f"Initial Portfolio: {portfolio}")

def calculate_previous_portfolio_value():
    stock_value = sum(previous_portfolio["Actual Vol"] * previous_portfolio["Market Price"])
    total_value = portfolio["cash"] + stock_value
    return total_value

previous_portfolio_total_value = previous_summary["NAV"] if isinstance(previous_summary, pd.DataFrame) else initial_cash
nav_lst = previous_statement["NAV"].to_list() if isinstance(previous_statement, pd.DataFrame) else [previous_portfolio_total_value]
last_prices = {}
# Filter rows with 'ATC' flag for each Share Code
atc_rows = data[data['Flag'] == 'ATC'].groupby('ShareCode').last()
atc_rows_reset = atc_rows.reset_index()

# Function to calculate portfolio value
def calculate_portfolio_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return portfolio["cash"], stock_value, total_value

def calculate_total_value():
    stock_value = 0
    for code, volume in portfolio["stocks"].items():
        price = atc_rows_reset["LastPrice"].loc[atc_rows_reset["ShareCode"] == code].values[0]
        stock_value += volume * price
    total_value = portfolio["cash"] + stock_value
    return total_value

def calculate_stock_value():
    stock_value = 0
    for code, volume in portfolio["stocks"].items():
        price = atc_rows_reset["LastPrice"].loc[atc_rows_reset["ShareCode"] == code].values[0]
        stock_value += volume * price
    return stock_value

statements = []
number_of_wins = 0
queue_for_calculate_pl_dict = dict()

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
            temp_portfolio_total_value = calculate_total_value()
            nav_lst.append(temp_portfolio_total_value)
            statements.append({
                "Table Name": "Statement",
                "File Name": "017_statement.csv",
                "Stock Name": stock_code,
                "Date": date,
                "Time": time,
                "Side": "Buy",
                "Volume": volume,
                "Price": price,
                "Amount cost": trade_value,
                "End_line_available": portfolio["cash"],
                "Portfolio Value": calculate_stock_value(),
                "NAV": temp_portfolio_total_value
            })
            if len(queue_for_calculate_pl_dict) == 0 or stock_code not in queue_for_calculate_pl_dict :
                queue_for_calculate_pl_dict[stock_code] = {"Stock Name": stock_code, "Price": price} 
            else:
                queue_for_calculate_pl_dict[stock_code]["Price"] = price     
        else:
            print(f"Not enough cash to buy {volume} of {stock_code} at {price} THB")
    elif trade_type == "Sell":
        if portfolio["stocks"].get(stock_code, 0) >= volume:
            portfolio["cash"] += trade_value
            portfolio["stocks"][stock_code] -= volume
            print(f"{date} {time} Sold {volume} of {stock_code} at {price} THB, amount cost: {trade_value} THB")
            temp_portfolio_total_value = calculate_total_value()
            nav_lst.append(temp_portfolio_total_value)
            statements.append({
                "Table Name": "Statement",
                "File Name": "017_statement.csv",
                "Stock Name": stock_code,
                "Date": date,
                "Time": time,
                "Side": "Sell",
                "Volume": volume,
                "Price": price,
                "Amount cost": trade_value,
                "End_line_available": portfolio["cash"],
                "Portfolio Value": calculate_stock_value(),
                "NAV": temp_portfolio_total_value
            })
            if stock_code in queue_for_calculate_pl_dict :
                sell_price = price
                buy_price = queue_for_calculate_pl_dict[stock_code]["Price"]
                if sell_price - buy_price > 0:
                    number_of_wins += 1
        else:
            print(f"Not enough stocks to sell {volume} of {stock_code} at {price} THB")

last_prices_buy = {}
last_prices_sell = {}
bought_row = []
match_trades = 0

def optimize_trading_strategy(data):
    # Create a copy of the DataFrame to track processed rows
    processed_data = data.copy()
    processed_data['Processed'] = False
    
    match_trades = 0
    last_prices_buy = {}
    last_prices_sell = {}

    for index in range(len(processed_data)):
        row = processed_data.iloc[index]
        last_prices[row["ShareCode"]] = row["LastPrice"]
        
        # Skip already processed rows or invalid rows
        if row['Processed'] or row['FinalSignal'] == "nan":
            continue

        stock_code = row["ShareCode"]
        price = row["LastPrice"]
        
        # Buying strategy
        if row['FinalSignal'] == 'Buy':
            cash = portfolio["cash"]
            affordable_share = (cash // price) // 100 * 100
            if affordable_share >= 100:
                # Find subsequent unprocessed rows with matching criteria
                subsequent_data = processed_data.iloc[index:].loc[
                    (processed_data.iloc[index:]['ShareCode'] == stock_code) & 
                    (processed_data.iloc[index:]['LastPrice'] == price) &
                    (processed_data.iloc[index:]['FinalSignal'] == 'Buy') & 
                    (processed_data.iloc[index:]['Processed'] == False)
                ]
                
                # Cumulative volume calculation for unprocessed rows
                shares_to_buy = subsequent_data['Volume'].cumsum()
                valid_shares = shares_to_buy[shares_to_buy <= affordable_share]
                if not valid_shares.empty:
                    total_shares = valid_shares.iloc[-1]
                    
                    # Mark processed rows
                    processed_rows_index = subsequent_data.index[:len(valid_shares)]
                    processed_data.loc[processed_rows_index, 'Processed'] = True
                    processed_data.loc[index, 'Processed'] = True
                    
                    execute_trade(stock_code, price, total_shares, "Buy", 
                                  processed_data.loc[processed_rows_index[-1]]['TradeDateTime'].date(), 
                                  processed_data.loc[processed_rows_index[-1]]['TradeDateTime'].time())
                    match_trades += 1
                    last_prices_buy[stock_code] = price

        # Selling strategy
        elif row['FinalSignal'] == 'Sell':
            if stock_code in portfolio["stocks"].keys():
                shares = portfolio["stocks"].get(stock_code, 0)
                shares = shares - (shares % 100)
                if shares >= 100:
                    subsequent_data = processed_data.iloc[index:].loc[
                        (processed_data.iloc[index:]['ShareCode'] == stock_code) & 
                        (processed_data.iloc[index:]['LastPrice'] == price) &
                        (processed_data.iloc[index:]['FinalSignal'] == 'Sell') &
                        (processed_data.iloc[index:]['Processed'] == False)
                    ]
                    
                    shares_to_sell = subsequent_data['Volume'].cumsum()
                    valid_shares = shares_to_sell[shares_to_sell <= shares]
                    if not valid_shares.empty:
                        total_shares = valid_shares.iloc[-1]
                        
                        # Mark processed rows
                        processed_rows_index = subsequent_data.index[:len(valid_shares)]
                        processed_data.loc[processed_rows_index, 'Processed'] = True
                        processed_data.loc[index, 'Processed'] = True
                        
                        execute_trade(stock_code, price, total_shares, "Sell", 
                                      processed_data.loc[processed_rows_index[-1]]['TradeDateTime'].date(), 
                                      processed_data.loc[processed_rows_index[-1]]['TradeDateTime'].time())
                        match_trades += 1
                        last_prices_sell[stock_code] = price

    return match_trades, last_prices_buy, last_prices_sell

match_trades, last_prices_buy, last_prices_sell = optimize_trading_strategy(data)

# Final portfolio summary
cash_balance, stock_value, total_value = calculate_portfolio_value(last_prices)
nav_lst.append(total_value)
print("\nEnd of Day Portfolio Summary:")
print(f"Cash Balance: {cash_balance:.2f} THB")
print(f"Stock Holdings Value: {stock_value:.2f} THB")
print(f"Total Portfolio Value: {total_value:.2f} THB")

start_value = 10_000_000
end_value = total_value
return_percentage = (end_value - start_value) / start_value * 100
print(f"\n% Return: {return_percentage:.2f}%")

# Export Statements table to CSV
statements_df = pd.DataFrame(statements)
statements_df.to_csv(f'{output_dir}/statement/017_statement.csv', index=False)

statements_df_copy = statements_df.copy()
statements_df_copy.sort_values("Time", inplace=True)

# Prepare the Portfolio Table for export
portfolio_data = []

# Filter rows with 'OPEN1_E' flag for each Share Code
open1e_rows = data[data['Flag'] == 'OPEN1_E'].groupby('ShareCode').last()

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

def get_previous_actual_volume(stock_code, previous_portfolio):
    if not isinstance(previous_portfolio, pd.DataFrame):
        return 0
    stock_row = previous_portfolio[previous_portfolio["Stock Name"] == stock_code]
    if not stock_row.empty:
        return stock_row["Actual Vol"].values[0]
    return 0

sum_market_value = 0
for stock_code, volume in portfolio["stocks"].items():
    if volume == 0:
        continue
    price = last_prices.get(stock_code, 0)
    start_vol = get_previous_actual_volume(stock_code, previous_portfolio)
    actual_vol = volume
    avg_cost = data[(data['ShareCode'] == stock_code) & (data['Flag'].isin(['Buy', 'Sell']))]['LastPrice'].mean()
    market_price = atc_rows_reset["LastPrice"].loc[atc_rows_reset["ShareCode"] == stock_code].values[0]
    amount_cost = avg_cost * volume
    market_value = actual_vol * market_price
    sum_market_value += market_value
    unrealized_pl = market_value - amount_cost
    unrealized_pl_pct = (unrealized_pl / amount_cost) * 100 if amount_cost > 0 else 0

    sell_tot_amount = sell_summary["Price"].loc[sell_summary["Stock Name"] == stock_code].values[0]
    buy_tot_amount = buy_summary["Price"].loc[buy_summary["Stock Name"] == stock_code].values[0]
    realized_pl = sell_tot_amount - buy_tot_amount
    
    portfolio_data.append({
        'Table Name': 'Portfolio',
        'File Name': '017_portfolio.csv',
        'Stock Name': stock_code,
        'Start Vol': start_vol,
        'Actual Vol': "{:.4f}".format(actual_vol),
        'Avg Cost': "{:.4f}".format(avg_cost),
        'Market Price': "{:.4f}".format(market_price),
        'Amount Cost': "{:.4f}".format(amount_cost),
        'Market Value': "{:.4f}".format(market_value),
        'Unrealized P/L': "{:.4f}".format(unrealized_pl),
        '%Unrealized P/L': "{:.4f}".format(unrealized_pl_pct),
        'Realized P/L': "{:.4f}".format(realized_pl)
    })

portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df.to_csv(f'{output_dir}/portfolio/017_portfolio.csv', index=False)

maximum_value = max(nav_lst)
start_index = statements_df_copy[statements_df_copy['NAV'] == maximum_value].index[0]
test = statements_df_copy.iloc[start_index:]['NAV'].tolist()
minimum_value = min(test)

maximum_drawdown = (minimum_value - maximum_value) / maximum_value if max(nav_lst) > 0 else 0
relative_drawdown = (maximum_drawdown / 10_000_000) * 100
calmar_ratio = ((sum_market_value + portfolio["cash"] - initial_cash) / initial_cash * 100) / ((sum_market_value + portfolio["cash"] - 10_000_000) / 10_000_000) if maximum_drawdown != 0 else 0
total_wins = previous_summary['Number of Wins'].iloc[-1] + number_of_wins if isinstance(previous_summary, pd.DataFrame) else number_of_wins
total_matches = previous_summary['Number of Matched Trades'].iloc[-1] + match_trades if isinstance(previous_summary, pd.DataFrame) else match_trades
total_transactions = previous_summary['Number of Transactions'].iloc[-1] + len(statements) if isinstance(previous_summary, pd.DataFrame) else len(statements)
sum_unrealized_pl = sum([float(row['Unrealized P/L']) for _, row in portfolio_df.iterrows()])
sum_unrealized_pl_pct = sum([float(row['%Unrealized P/L']) for _, row in portfolio_df.iterrows()])
sum_realized_pl = sum([float(row['Realized P/L']) for _, row in portfolio_df.iterrows()])
win_rate = total_wins / total_transactions * 100
trading_day = int(previous_summary['trading_day'].iloc[0])+1 if isinstance(previous_summary, pd.DataFrame) else 1

# Prepare the Summary Table for export
summary_data = [{
    'Table Name': 'Summary',
    'File Name': '017_summary.csv',
    'trading_day': trading_day,
    'NAV': total_value,
    'End Line Available': "{:.4f}".format(portfolio["cash"]),
    'Start Line Available': "{:.4f}".format(initial_cash),
    'Number of Wins': total_wins,
    'Number of Matched Trades': total_matches,
    'Number of Transactions': total_transactions,
    'Sum of Unrealized P/L': "{:.4f}".format(sum_unrealized_pl),
    'Sum of %Unrealized P/L': "{:.4f}".format(sum_unrealized_pl_pct),
    'Sum of Realized P/L': "{:.4f}".format(sum_realized_pl),
    'Maximum Value': "{:.4f}".format(maximum_value),
    'Minimum Value': "{:.4f}".format(minimum_value),
    'Win Rate': "{:.4f}".format(win_rate),
    'Calmar Ratio': "{:.4f}".format(calmar_ratio),
    'Relative Drawdown': "{:.4f}".format(relative_drawdown),
    'Maximum Drawdown': "{:.4f}".format(maximum_drawdown),  
    '%Return': "{:.4f}".format(return_percentage)
}]

summary_df = pd.DataFrame(summary_data)
previous_summary = pd.concat([previous_summary, summary_df], ignore_index=True) if isinstance(previous_summary, pd.DataFrame) else summary_df
previous_summary.to_csv(f'{output_dir}/summary/017_summary.csv', index=False)

end = time.time()
print(f"time to run program: {end-start}")

print("CSV files exported successfully.")
