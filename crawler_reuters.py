#!/usr/bin/python
import re
import urllib3
import csv
import os
import sys
import time
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dateutil.relativedelta import relativedelta

import numpy as np
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def exact_date(idelta):
    strtofind=[' days',' months',' years']
    delta=''
    for istr in strtofind:
        indx=idelta.find(istr)
        if indx!=-1:
            delta=-1*int(idelta[0:indx])
            break
        elif idelta.find(istr[0:-1])!=-1:
            delta=-1
            break
        
        
    if delta!='':
        base = datetime.datetime.today()
        attributes={istr.replace(' ',''):delta}
        xx=base + relativedelta(**attributes)
        return xx.strftime("%Y%m%d")
        
    else:
        return 'Not found'

class news_Reuters:
    def __init__(self):
        fin = open('./input/tickerList.csv')

        filterList = set()
        try: # this is used when we restart a task
            fList = open('./input/finished.reuters')
            for l in fList:
                filterList.add(l.strip())
        except: pass

        dateList = self.dateGenerator(3000) # look back on the past X days
        for line in fin: # iterate all possible tickers
            line = line.strip().split(',')
            ticker=line[1]
            name=line[2]
            exchange=line[3]
            MarketCap=line[4]
            if ticker=='symbol': continue
            #ticker, name, exchange, MarketCap = line
            if ticker in filterList: continue
            print("%s - %s - %s - %s" % (ticker, name, exchange, MarketCap))
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument("no-sandbox")
            chrome_options.add_argument("headless")
            chrome_options.add_argument("start-maximized")
            chrome_options.add_argument("window-size=1900,1080")
            #chrome_option.add_argument("--proxy-server=https://internet.ford.com:83/" )
            CHROMEDRIVER_PATH='./chromedriver.exe'
            driver = webdriver.Chrome(CHROMEDRIVER_PATH,options=chrome_options)
            self.contents(ticker, name, line, dateList, exchange,driver)
            driver.quit() 

    def contents(self, ticker, name, line, dateList, exchange,driver):
        # https://uk.reuters.com/info/disclaimer
        suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
        url = "https://www.reuters.com/companies/" + ticker + suffix[exchange]
        proxy = urllib3.ProxyManager('https://internet.ford.com:83/', maxsize=10)
        #http = urllib3.PoolManager()
        has_Content = 0
        repeat_times = 4
        # check the website to see if that ticker has many news
        # if true, iterate url with date, otherwise stop
        for _ in range(repeat_times): # repeat in case of http failure
            try:
                time.sleep(np.random.poisson(3))
                response = proxy.request('GET', url)
                soup = BeautifulSoup(response.data, "lxml")
                has_Content = len(soup.find_all("div", {'class': ['NewsList-story-3G86I','MarketStoryItem-container-3rpwz NewsFeed-story-1bEiu NewsFeed-first-2kYZX' ,'MarketStoryItem-container-3rpwz NewsFeed-story-1bEiu']}))                
                break
            except:
                continue


        ticker_failed = open('./input/news_failed_tickers.csv', 'a+')
        fout = open('./input/news_reuters.csv', 'a+',encoding = 'utf-8')

        if has_Content > 0:
            #missing_days = 0
            timestamp=dateList
            #for timestamp in dateList:
                
            self.DownloadKeyDevelopments(ticker, line, url+'/key-developments', timestamp,driver,fout)
            self.DownloadNews(ticker, line, url+'/news', timestamp,driver,fout)
 
        else:
            print("%s has no news" % (ticker))
            today = datetime.datetime.today().strftime("%Y%m%d")
            ticker_failed.write(ticker + ',' + today + ',' + 'LOWEST\n')
        ticker_failed.close()
        fout.close()
        
    def DownloadNews(self, ticker, line, url, timefinish,driver,fout):

        driver.set_page_load_timeout(40)
        repeat_times = 3
        # if true, iterate url with date, otherwise stop
        for _ in range(repeat_times): # repeat in case of http failure
            try:
                time.sleep(np.random.poisson(2)/3)
                driver.get(url)
                
                break
            except:
                print(ticker, 'News page Not accessible')
                continue
        
        lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        match=False
        istart=0

        while(match==False):
            lastCount = lenOfPage
            time.sleep(np.random.poisson(3)/12)
            lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            soup = BeautifulSoup(driver.page_source, 'lxml')
            News_reuter = soup.find_all("div", {'class': ['MarketStoryItem-container-3rpwz NewsFeed-story-1bEiu NewsFeed-first-2kYZX','MarketStoryItem-container-3rpwz NewsFeed-story-1bEiu']})
            #istart=len(News_reuter)    
            if lastCount==lenOfPage:
                match=True 
            ##### crawling key developements news
        for inews in range(istart,len(News_reuter)):
            news=News_reuter[inews]
            # if true, iterate url with date, otherwise stop
            for _ in range(repeat_times): # repeat in case of http failure
                try:
                    time.sleep(np.random.poisson(3)/3)
                    driver.get(news.a['href'])
                    
                    break
                except:
                    print(ticker, 'News', inews, 'URL do not work')
                    continue

            soup = BeautifulSoup(driver.page_source, 'lxml')
            news_dates = soup.find_all("time")
            timestamp=''
            # getting timestamp from article related to the news
            
            for idate in news_dates:
                try:
                    timestamp=datetime.datetime.strptime(idate.get_text(), '%B %d, %Y').strftime('%Y%m%d')
                    break# stop if we found the date
                except:# date not found
                    pass

                try:
                    span=idate.find_all('span')
                    for ispan in span:
                        try:
                            timestamp=datetime.datetime.strptime(ispan.get_text(), '%B %d, %Y').strftime('%Y%m%d')
                            break# stop if we found the date
                        except:
                            print(ticker, 'News', inews, 'Not found')
                    break
                
                except:
                    print(ticker, 'News', inews, 'Time has no span class')
                    
            try:
                datetime.datetime.strptime(timestamp, '%Y%m%d')
            except:
                print(ticker,'!!!!!!!', 'News',inews, '!!!!!!! skipped Wrong timestamp')
                continue        
            if timestamp=='':
                print(ticker,'!!!!!!!', 'News',inews, '!!!!!!! skipped no timestamp')
                continue
            if int(timefinish)>int(timestamp):# stop fetching if date news is older than the time to finish
                match=True
                break
            else:# writing the news
                   title = news.a.get_text().replace(",", " ").replace("\n", " ")
                   body =  news.p.get_text().replace(",", " ").replace("\n", " ")
                   if inews == 0 : news_type = 'topStory'
                   else: news_type = 'normal'
                   print(ticker, timestamp, title, news_type)
                   fout.write(','.join([ticker, line[2], timestamp, title, body, news_type, 'News'])+'\n')
                


  
        print('*********** ',ticker,'News','Dowloaded','*************')
    def DownloadKeyDevelopments(self, ticker, line, url, timefinish,driver,fout):

        driver.set_page_load_timeout(40)
        repeat_times = 3
        # if true, iterate url with date, otherwise stop
        for _ in range(repeat_times): # repeat in case of http failure
            try:
                time.sleep(np.random.poisson(3)/10)
                driver.get(url)
                
                break
            except:
                print(ticker, 'News page Not accessible')
                continue
        lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        match=False
        istart=0
        #fout = open('./input/news_reuters.csv', 'a+',encoding = 'utf-8')
        while(match==False):
            lastCount = lenOfPage
            time.sleep(np.random.poisson(1)/4)
            lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            soup = BeautifulSoup(driver.page_source, 'lxml')
            keydev = soup.find_all("div", {'class': ['MarketStoryItem-container-3rpwz NewsFeed-story-1bEiu NewsFeed-first-2kYZX','MarketStoryItem-container-3rpwz NewsFeed-story-1bEiu']})
            #istart=len(keydev)
            if lastCount==lenOfPage:
                match=True 
        ##### crawling key developements news
        for ikey in range(istart,len(keydev)):
            ikeydev=keydev[ikey]
            try:
                keydate=''
                text_body=''
                body=ikeydev.p.get_text()
                indx=body.find(' (Reuters) -')
                if indx!=-1:
                    keydate=body[0:indx]
                    text_body=body[indx+13:]
                    idelta=ikeydev.time.get_text()    
                    timestamp=exact_date(idelta)
                    keydate=keydate.replace('Sept','Sep')
                    try:
                        md=datetime.datetime.strptime(keydate, '%b %d').strftime('%m%d')
                    except:
                        md=datetime.datetime.strptime(keydate, '%B %d').strftime('%m%d') 
                            
                    timestamp=timestamp[0:4]+md
                    
                else:
                    error_message = " (Reuter) - Not found"
                    raise ValueError(error_message)
              
            except Exception as e:
                print(str(e))
                continue 
            try:
                datetime.datetime.strptime(timestamp, '%Y%m%d')
            except:
                print(ticker,'!!!!!!!', 'key dev:',ikey, '!!!!!!! skipped Wrong timestamp')
                continue        
            if timestamp=='':
                print(ticker,'!!!!!!!', 'key dev:',ikey, '!!!!!!! skipped no timestamp')
                continue            
            if int(timefinish)>int(timestamp):# stop fetching if date news is older than the time to finish
                match=True
                break
            else:# writing the news
                   title = ikeydev.h4.get_text().replace(",", " ").replace("\n", " ")
                   body = text_body.replace(",", " ").replace("\n", " ")
                   if ikey == 0 : news_type = 'topStory'
                   else: news_type = 'normal'
                   print(ticker, timestamp, title, news_type)
                   fout.write(','.join([ticker, line[2], timestamp, title, body, news_type, 'Key Developments'])+'\n')
                    
                

           
        print('*********** ',ticker,'Key Developpements','Dowloaded','*************')



    def dateGenerator(self, numdays): # generate N days until now
        base = datetime.datetime.today()
        date_list = base - datetime.timedelta(days=numdays)
        date_list = date_list.strftime("%Y%m%d")
        return date_list

def main():
    news_Reuters()

if __name__ == "__main__":
    main()
