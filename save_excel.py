import pandas as pd
import yfinance as yf

period = "1y"
interval = "1d"
# Define the ticker symbols
ndxt_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA']

df_all = pd.DataFrame(columns = ['Stock', 'Open', 'High', 'Low', 'Close', 'SMA100', 'SMA20'])
for ticker in ndxt_symbols:
    # Download the data from Yahoo Finance for the ticker and specified period
    data = yf.download(ticker, period=period, interval=interval)

    # Remove this month's data
    data = data[data.index.month != pd.Timestamp.now().month]
    # Calculate the 100-day and 20-day SMA for the stock using pandas
    # For long term
    sma100 = data["Close"].rolling(100).mean()
    # For short term
    sma20 = data["Close"].rolling(20).mean()

    # Calculate the 100-day EMA for the stock using pandas
    #ema = data["Close"].ewm(span=100, adjust=False).mean()

    # Create a pandas DataFrame with the data
    df = pd.DataFrame({'Stock': ticker,'Open': data['Open'],'High': data['High'],'Low': data['Low'],'Close': data['Close'],
                       'SMA100': sma100, 'SMA20': sma20})
    df_all=df_all._append(df)

# Save the DataFrame to an csv file
df_all.to_csv('output.csv', index=True)