#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import yfinance as yf
import datetime,time,requests,io
import numpy as np
import datetime
import io
import requests
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


def Equity_curuve_values(symbols):
    stocks =pd.DataFrame()
    
    for i in symbols:
        
        stock = []
        stock = yf.download(i+".ns",start,end)        
        stock['Name'] = i
        stocks = stocks.append(stock)
    
    tmp = stocks.select_dtypes(include=[np.number])
    stocks.loc[:, tmp.columns] = np.round(tmp,2)
    st = stocks.reset_index()
    
    stg = st.groupby(['Date','Name']).first().reset_index()[0:len(symbols)]
    stg['Number_of_Shares_bought'] = np.floor((investment/len(symbols))/stg.Open)
    Equally_Invested_Amount = (stg.Open*stg.Number_of_Shares_bought).sum()
    
    st = pd.merge(st, stg[['Name','Number_of_Shares_bought']], on='Name')
    st['Daily_Value'] = st['Close'] * st['Number_of_Shares_bought']
    
    equity_curve_of_day = st.groupby(['Date']).agg(Equity_curuve_values=('Daily_Value', 'sum'))
    
    return Equally_Invested_Amount,equity_curve_of_day
    


# In[5]:


def Benchmark_stategy(start,end,investment):
    
    #Download nifty_50 data
    url = 'https://archives.nseindia.com/content/indices/ind_nifty50list.csv'
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    #select symbol from nifty_50
    symbols = list(df.Symbol)
    
    #applying normal bencmark strategy STEP1
    Benchark_Strategy_EC = Equity_curuve_values(symbols)
    return Benchark_Strategy_EC


# In[6]:


def Sample_strategy(start,end,investment,n_days_for_measuring_performance,top_n_stocks):
    
    temp_end = start 
    temp_start = end - datetime.timedelta(days=n_days_for_measuring_performance)
    
    #Download nifty_50 data and load
    url = 'https://archives.nseindia.com/content/indices/ind_nifty50list.csv'
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    symbols = list(df.Symbol)
    
    #process of measuring perforamce of last n days
    data_n_days = pd.DataFrame()

    for i in symbols:
        stock = []
        stock = yf.download(i+".ns",temp_start,temp_end)        
        stock['Name'] = i
        data_n_days = data_n_days.append(stock)
    
    t_minus_nth_day = data_n_days.sort_values('Date').groupby('Name').first()
    t_minus_1_day = data_n_days.sort_values('Date').groupby('Name').last()
    
    performance = ((t_minus_1_day.Close/t_minus_nth_day.Close)-1)*100
    
    #selecting top-n stocks
    top_n = performance.sort_values(ascending = False).iloc[0:(int(top_n_stocks))].reset_index()
    top_n.rename(columns = {'Close':'Percentage_returns_'+str(int(n_days_for_measuring_performance))+'_days'}, inplace=True)
    
    #STEP-2 Apply strategy based on performance of stocks
    I_s,sample_strategy_EC = Equity_curuve_values(list(top_n.Name))
    
    return top_n,I_s,sample_strategy_EC
    
    
    


# In[7]:


def Metrics(ec):
    CAGR = ((ec.Equity_curuve_values.iloc[-1]/ec.Equity_curuve_values.iloc[0])**(365/((end-start).days-1)) -1)*100
    
    dr  = lambda x: (x.iloc[-1]/x.iloc[0])-1

    Daily_returns = ec.Equity_curuve_values.rolling(2).apply(dr)
    
    voltality = ((Daily_returns.std())**(1/252)) * 100
    sharpe_ratio = (Daily_returns.std()/Daily_returns.mean())**(1/252)
    
    return CAGR,voltality,sharpe_ratio


# In[8]:


st.title('WELCOME ITO STOCK MARKET')


# In[9]:


st.header('Please enter the follwoing details to invest in stock market:')


# In[10]:


start = st.text_input('Enter the Start date in the format "year-month-day" (Example: 2022-10-31):', key='st')

end = st.text_input('Enter the End date in the format "year-month-day" (Example: 2022-10-31): ', key='en')

investment =st.number_input('Enter the invesment amount:')
n_days_for_measuring_performance = st.number_input('Enter number of previous days to measure performance of stocks: ')
top_n_stocks = st.number_input('Number of top stocks to select: ')


# In[11]:


if(st.button('Submit')):
    start = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    end = datetime.datetime.strptime(end, '%Y-%m-%d').date()
    st.success('Please wait, while we analyse and fetch data')
    
    #Bnechmark strategy (TASK-1)
    I_b,Benchmark_EC = Benchmark_stategy(start,end,investment)
    
    #Performance Based strategy(TASK-2)
    top_n,I_s,Sample_EC = Sample_strategy(start,end,investment,n_days_for_measuring_performance,top_n_stocks)
    
    nifty_50 = yf.download('^NSEI',start,end)
    nifty_50 = np.round(nifty_50,2)
    nifty_50['Number_of_shares_bought'] = np.floor(investment/nifty_50.sort_values('Date').iloc[0].Open)
    nifty_50['Equity_curuve_values'] = nifty_50['Number_of_shares_bought']*nifty_50['Close']
    
    #Measuring metrics(Task-3)
    Results = pd.DataFrame({'Benchmark_strategy': Metrics(Benchmark_EC),'Performance_based_strategy': Metrics(Sample_EC),'Nifty_50':Metrics(nifty_50)}).T
    
    Results.columns = ['CAGR(%)','Voltality(%)','Sharpe']
    Results = np.round(Results, 3)
    
    # Set CSS properties for th elements in dataframe
    th_props = [('font-size', '16px'),
                ('text-align', 'center'),
                ('font-weight', 'bold'),
                ('color', '#cf297c'),
                ('background-color', '#0a1212')]

    # Set CSS properties for td elements in dataframe
    td_props = [('font-size', '11px')]

    # Set table styles
    styles = [dict(selector="th", props=th_props),
              dict(selector="td", props=td_props)]

    cm = sns.light_palette("green", as_cmap=True)
    
    Results = (Results.style
               .background_gradient(cmap=cm, subset=['CAGR(%)','Voltality(%)','Sharpe'])
               .highlight_max(color='red',subset=['CAGR(%)','Voltality(%)','Sharpe'])
               .set_caption('PERFORMACE METRICS')
               .set_table_styles(styles)
                .set_precision(2))
    
    #Plot the Equity curves(Task-4)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(Benchmark_EC,label='Benchmark_strategy')
    ax.plot(Sample_EC, label='Performance_startegy')
    ax.plot(nifty_50.Equity_curuve_values,label='Nifty_50')
    ax.title.set_text('Graphical Comparission')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity Curve Values')
    ax.legend()
    
    #Using Steam lit to host app(TASK_5)
    st.pyplot(fig)
    st.dataframe(Results)
    st.dataframe(top_n.style.set_precision(2))


# In[ ]:




