import pandas as pd
from torch.utils.data import Dataset, DataLoader



#  there are simple methods to calculate the technical indicators
#  for all other columns we have the data in the csv file



def calculate_macd(df):
    # techinical indicator 
    short_term = 12
    long_term = 26
    signal_period = 9

    df['EMA12'] = df['Close/Last'].ewm(span=short_term,adjust=False).mean()
    df['EMA26'] = df['Close/Last'].ewm(span=long_term,adjust=False).mean()

    df['MACD Line'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD Line'].ewm(span=signal_period,adjust=False).mean()
    df['MACD'] = df['MACD Line'] - df['Signal Line']
    df.drop(['EMA12','EMA26','MACD Line','Signal Line'],inplace=True,axis=1)
    return df

def calucate_atr(data):
    # techinical indicator 
    atr_period = 14
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = (data['High'] - data['Close/Last'].shift(1)).abs()
    data['L-PC'] = (data['Low'] - data['Close/Last'].shift(1)).abs()

    data['TR'] = data[['H-L','H-PC','L-PC']].max(axis=1)

    data['ATR'] = data['TR'].rolling(atr_period).mean()
    data.drop(['H-L','H-PC','L-PC','TR'],inplace=True,axis=1)
    return data

def calculate_rsi(data):
    # techinical indicator 
    rsi_period = 14
    data['Price Change'] = data['Close/Last'].diff()
    data['Gain'] = data['Price Change'].apply(lambda x: x if x > 0 else 0)
    data['Loss'] = data['Price Change'].apply(lambda x: -x if x < 0 else 0)
    data['Average Gain'] = data['Gain'].rolling(window = rsi_period).mean()
    data['Average Loss'] = data['Loss'].rolling(window = rsi_period).mean()
    data['Relative Strength'] = data['Average Gain']/data['Average Loss']
    data['RSI'] = 100 - (100/(1+data['Relative Strength']))

    data.drop(['Price Change','Gain','Loss','Average Gain','Average Loss','Relative Strength'],inplace=True,axis=1)

    return data


def get_data(path):
    df = pd.read_csv(path)
    df.Date = df['Date'].apply(lambda x: pd.to_datetime(x,format="%m/%d/%Y"))
    return df

if __name__ == "__main__":
    df = get_data("../data/HistoricalData_1698863282356.csv")
    df = calculate_macd(df)    # Moving Average Convergence Divergence
    df = calucate_atr(df)      # market volatility
    df = calculate_rsi(df)
    df.to_csv("../data/final_dataset.csv",index=False)
