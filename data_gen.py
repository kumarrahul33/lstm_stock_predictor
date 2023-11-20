import pandas as pd
from torch.utils.data import Dataset, DataLoader
from os.path import join as path_join

class Constants:
    SHORT_TERM = 12
    LONG_TERM = 26
    SIGNAL_PERIOD = 9
    RSI_PERIOD = 14
    ATR_PERIOD = 14

class DataSources:
    DATA_DIR = "data"
    HISTORICAL_OC = path_join(DATA_DIR, "HistoricalData.csv")
    EFFR = path_join(DATA_DIR, "EFFR.csv")
    VIX = path_join(DATA_DIR, "VIX.csv")
    DX = path_join(DATA_DIR, "DX-Y.NYB.csv")
    UNRATE = path_join(DATA_DIR, "UNRATE.csv")
    UMCSENT = path_join(DATA_DIR, "UMCSENT.csv")
    
def calculate_macd(df):
    # technical indicator 
    df['EMA12'] = df['Close'].ewm(span=Constants.SHORT_TERM,adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=Constants.LONG_TERM,adjust=False).mean()

    df['MACD Line'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD Line'].ewm(span=Constants.SIGNAL_PERIOD,adjust=False).mean()
    df['MACD'] = df['MACD Line'] - df['Signal Line']
    df.drop(['EMA12','EMA26','MACD Line','Signal Line'],inplace=True,axis=1)
    return df

def calculate_atr(data):
    # technical indicator 
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = (data['High'] - data['Close'].shift(1)).abs()
    data['L-PC'] = (data['Low'] - data['Close'].shift(1)).abs()

    data['TR'] = data[['H-L','H-PC','L-PC']].max(axis=1)

    data['ATR'] = data['TR'].rolling(Constants.ATR_PERIOD).mean()
    data.drop(['H-L','H-PC','L-PC','TR'],inplace=True,axis=1)
    return data

def calculate_rsi(data):
    # technical indicator 
    data['Price Change'] = data['Close'].diff()
    data['Gain'] = data['Price Change'].apply(lambda x: x if x > 0 else 0)
    data['Loss'] = data['Price Change'].apply(lambda x: -x if x < 0 else 0)
    data['Average Gain'] = data['Gain'].rolling(window = Constants.RSI_PERIOD).mean()
    data['Average Loss'] = data['Loss'].rolling(window = Constants.RSI_PERIOD).mean()
    data['Relative Strength'] = data['Average Gain']/data['Average Loss']
    data['RSI'] = 100 - (100/(1+data['Relative Strength']))

    data.drop(['Price Change','Gain','Loss','Average Gain','Average Loss','Relative Strength'],inplace=True,axis=1)

    return data


def get_data(path):
    df = pd.read_csv(path)
    df.Date = df['Date'].apply(lambda x: pd.to_datetime(x,format="%m/%d/%y"))
    return df

def calculate_whole(data_df, path, date_format):
    df_add = pd.read_csv(path)
    # print(df_add)
    df_add.Date = df_add['Date'].apply(lambda x: pd.to_datetime(x,format=date_format))
    merged_df = pd.merge_ordered(data_df, df_add, on='Date', how='left')
    return merged_df


def calculate_column(data_df, path, date_format, feature):
    df_add = pd.read_csv(path)
    # print(df_add)
    df_add.Date = df_add['Date'].apply(lambda x: pd.to_datetime(x,format=date_format))
    column = "Adj Close"
    merged_df = pd.merge_ordered(data_df, df_add[['Date', column]], on='Date', how='left')
    merged_df.rename(columns={column: feature}, inplace=True)
    return merged_df

def calculate_monthly(data_df, path, date_format):
    df_add = pd.read_csv(path)
    # print(df_add)
    # df_add.Date = df_add['DATE'].apply(lambda x: pd.to_datetime(x,format=date_format))
    df_add['Date'] = pd.to_datetime(df_add['DATE']).dt.to_period('M').dt.to_timestamp()
    df_add['YearMonth'] = df_add['Date'].dt.to_period('M')
    data_df['YearMonth'] = data_df['Date'].dt.to_period('M')
    # print(data_df)
    merged_df = pd.merge_ordered(data_df, df_add, left_on='YearMonth', right_on='YearMonth', how='left')
    merged_df.drop(['YearMonth', 'DATE', 'Date_y'], axis=1, inplace=True)
    merged_df.rename(columns={'Date_x': 'Date'}, inplace=True)
    return merged_df


def refine_data(data_df):
    # iterate through the columns EEFR, VIX, USDX, UNRATE, UMCSENT
    for i in range(1, len(data_df)):
        for column in data_df.columns:
            if column == 'Date':
                continue
            else:
                try:
                    temp = float(data_df[column][i])
                except:
                    # print("Error: ", data_df[column][i])
                    # if(i == 0): data_df[column][i] = 0
                    # else: data_df[column][i] = data_df[column][i-1]
                    # use iloc operation for the above lines
                    if(i == 0): data_df.iloc[i, data_df.columns.get_loc(column)] = 0
                    else:   data_df.iloc[i, data_df.columns.get_loc(column)] = data_df.iloc[i-1, data_df.columns.get_loc(column)]
    return data_df

            
def prepareFinalDataset(final_name="final_dataset.csv"):
    df = get_data(DataSources.HISTORICAL_OC)

    # Technical Indicators
    df = calculate_macd(df)             # Moving Average Convergence Divergence
    df = calculate_atr(df)              # Market Volatility
    df = calculate_rsi(df)              # Relative Strength Index

    df = calculate_whole(df, DataSources.EFFR, date_format = "%d-%m-%Y")
    df = calculate_column(df, DataSources.VIX, date_format = "%d-%m-%Y", feature = "VIX" )
    df = calculate_column(df, DataSources.DX, date_format = "%Y-%m-%d", feature = "USDX" )
    df = calculate_monthly(df, DataSources.UNRATE, date_format = "%Y-%m-%d")
    df = calculate_monthly(df, DataSources.UMCSENT, date_format = "%Y-%m-%d")
    df['Date'] = df['Date'].apply(lambda x: x.strftime('%d-%m-%Y'))
    df = refine_data(df)
    df.to_csv(path_join(DataSources.DATA_DIR, final_name),index=False)

if __name__ == "__main__":
    prepareFinalDataset()