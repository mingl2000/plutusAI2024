import yfinance as yf
import numpy as np
from os.path import exists
import datetime
import pandas as pd
import sys
import os
def get1mYahooData(ticker):
    if not os.path.exists("data"):
       os.mkdir("data")

    data=[]
    

    tick = yf.Ticker(ticker)
    df = tick.history(period='7d', interval='1m')
    #df=df.loc[(df.index.hour*60+data.index.minute)<=690 | (df.index.hour>12)]
    data.append(df)


    #minstart = datetime.date.today()-datetime.timedelta(29)
    #minstart = datetime.datetime(minstart.year, minstart.month, minstart.day)
    minstart =np.min(df.index)-datetime.timedelta(21)
    end=np.min(df.index)
    start=end-datetime.timedelta(7)
    
    while start>minstart:
        df = tick.history(start=start, end=end,interval='1m')
        data.insert(0, df)
        start=start- datetime.timedelta(7)
        end=end- datetime.timedelta(7)

    start=minstart
    df = tick.history(start=start, end=end,interval='1m')
    data.insert(0, df)    
    df=pd.concat(data)
    df.to_csv((ticker +'_1m.csv'), index=True)
    return df

def getYahooData_v1(ticker, interval='1m'):
    if not os.path.exists("data"):
       os.mkdir("data")

    if ticker.lower()=='spx':
        ticker='^GSPC'

    if interval=='1m':
        return get1mYahooData(ticker)
    
    if  interval.endswith('m'):
        period='60d'
    elif  interval.endswith('h'):
        period='730d'
    else:
        period='max'
    dataFileName="data/"+ticker +'_max_'+ interval +".csv"
    if interval.endswith(('d','D')) and exists(dataFileName):
      print('read yahoo data from cache: ',ticker)
      df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    else:
      df = yf.download(tickers=ticker, period=period, interval=interval)
    
    #df=df.loc[(df.index.hour*60+df.index.minute)<=690 | (df.index.hour>12)]
    return df
def GetYahooData_v2(symbol, bars=500, interval='1d'):
  if not os.path.exists("data"):
    os.mkdir("data")

  #start=datetime.date.today()-datetime.timedelta(days=days)
  #end=datetime.date.today()
  if symbol=='SPX':
    symbol='^GSPC'
  #df=.gepdrt_data_yahoo(symbols=symbol,  start=start, end=end,interval=interval)
  
  #if interval.endswith('m') or interval.endswith('h'):
  #  period='max'
  
  if interval.endswith('1m'):
    period='7d'
  elif  interval.endswith('m'):
    period='60d'
  elif  interval.endswith('h'):
    period='730d'
  else:
    period='max'
  
  #elif interval.endswith('d'):
    #period=str(days)+'d'
  #  period='max'
  #elif  interval.endswith('w'):
  #  period=str(days)+'wk'
  
  dataFileName="data/"+symbol +'_max_'+ interval +".csv"
  #dataFileName1="data/"+symbol+'_' +'max'+'_'+ interval +".csv"
  #if interval.endswith(('d','D')) and datetime.datetime.now().hour>=13 and exists(dataFileName):
  if interval.endswith(('d','D')) and exists(dataFileName):
    print('read yahoo data from cache: ',symbol)
    df=pd.read_csv(dataFileName, header=0, index_col=0, encoding='utf-8', parse_dates=True)
    #df.index=df["Date"]
  else:
    print('read yahoo data from web')
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.to_csv(dataFileName, index=True, date_format='%Y-%m-%d %H:%M:%S')
  #dataFileName="data/"+symbol+".csv"
  
  #df = pd.read_csv(dataFileName,index_col=0,parse_dates=True)
  #df.shape
  df.dropna(inplace = True)
  df =df [-bars:]
  df.head(3)
  df.tail(3)
  df["id"]=np.arange(len(df))
  #df["date1"]=df.index.astype(str)
  #df["datefmt"]=df.index.strftime('%m/%d/%Y')
  
  return df
if __name__ == '__main__':
  if len(sys.argv) >=2:
    ticker=sys.argv[1]
  else:
    ticker='QQQ'

  print('ticker=',ticker)

  if len(sys.argv) >=3:
    historylen=int(sys.argv[2])
  else:
    historylen=500
  if len(sys.argv) >=4:
    interval=sys.argv[3]
  else:
    interval='1d'
  from datetime import *
  start=datetime.now()
  df=GetYahooData_v2(ticker,historylen,interval)
  end=datetime.now()
  print('ticker=',ticker, 'len=', len(df), 'speed=',end-start)