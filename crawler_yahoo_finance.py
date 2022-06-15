#!/usr/bin/python
import re
import sys
import os
import time
import datetime
import urllib3
import pandas as pd
from subprocess import Popen, PIPE

import random
import json
from math import log
# output file name: input/stockPrices_raw.json
# json structure: crawl daily price data from yahoo finance
#          ticker
#         /  |   \
#     open close adjust ...
#       /    |     \
#    dates dates  dates ...
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_stock_Prices():
    fin=pd.read_csv('./input/tickerList.csv',low_memory=False)
    fin=fin['symbol']
    #fin = open('./input/finished.reuters')
    output = './input/stockPrices_raw.json'
    priceSet = {}
    for num, line in enumerate(fin):
        ticker = line.strip()
        print(num, ticker)
        priceSet[ticker] = repeatDownload(ticker)

    with open(output, 'w') as outfile:
        json.dump(priceSet, outfile, indent=4)


def repeatDownload(ticker):
    repeat_times = 3 # repeat download for N times
    priceStr = ""
    for _ in range(repeat_times):
        try:
            time.sleep(random.uniform(1, 2))
            priceStr = PRICE(ticker)
            if len(priceStr) > 0: # skip loop if data is not empty
                print('********* '+ticker+' dowloaded')
                break
        except:
            if _ == 0: print(ticker, "Http error!")
    return priceStr


def PRICE(ticker):
    period1=int(time.mktime(datetime.datetime(2011, 5, 1,23,59).timetuple()))
    period2=int(time.mktime(datetime.datetime(2999, 12, 1,23,59).timetuple()))
    inter="1d"
   # ' https://finance.yahoo.com/quote/'AAPL/history?period1=1601424000&period2=1632960000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
    # Construct url
    # url1 = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker
    url1 = 'https://query1.finance.yahoo.com/v7/finance/download/' + ticker
    url2="?period1=" + str(period1) +"&period2=" +str(period2) +"&interval=" +inter+ "&events=history&includeAdjustedClose=true"
    # url=    "https://query1.finance.yahoo.com/v7/finance/download/TXG?period1=1304290740&amp;period2=32501087940&amp;interval=1d&amp;events=history&amp;includeAdjustedClose=true" 

    df=pd.read_csv(url1+url2)
    # get historical price
    ticker_price = {}
    df_variables=index = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    index = ['open', 'high', 'low', 'close', 'volume', 'adjClose']
    for i in range(0,len(index)):
        typeName=index[i]
        dfName=df_variables[i]
        for num in range(0,len(df)):
            date = df.Date[num]
            # check if the date type matched with the standard type
            if not re.search(r'^[12]\d{3}-[01]\d-[0123]\d$', date): continue
            date=date.replace('-','')
            # open, high, low, close, volume, adjClose : 1,2,3,4,5,6
            try:
                ticker_price[typeName][date] = round(float(df[dfName][num]),2)
            except:
                ticker_price[typeName] = {}
    return ticker_price


if __name__ == "__main__":
    get_stock_Prices()
