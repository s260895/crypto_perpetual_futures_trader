

import os
import sys
from pprint import pprint
import ccxt
#import ccxt.async_support as ccxt
from pyti.exponential_moving_average import exponential_moving_average as ema
import pandas as pd

import datetime
import time

import numpy as np




# root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(root + '/python')

# print('CCXT Version:', ccxt.__version__)

def get_last_n_kline_closes(n=50,interval='1h',symbol='BTC/USDT'):
    
    exchange = ccxt.binance({
        'apiKey': g_api_key,
        'secret': g_secret_key,
        'enableRateLimit': True,  
        'options': {
            'defaultType': 'future',
        },
        'hedgeMode':True
    })
    # symbol = 'BTC/USDT'
    market = exchange.load_markets()
    market = exchange.market(symbol)
    closes = [elem[4] for elem in exchange.fapiPublic_get_klines({'symbol':market['id'],'interval':interval})][-n:]
    
    return closes

def generate_signal(current_input=None,strat='ema_cross_over_under',fast_ema=10,slow_ema=40):
    if current_input is None:
        return "NO INPUT"    
    closes = pd.DataFrame(current_input,columns=['close'])
    closes['close'] = closes['close'].astype(float)
    ema_diff = pd.DataFrame(ema(closes['close'].tolist(),fast_ema) - ema(closes['close'].tolist(),slow_ema),columns=['ema_diff'])

    last = ema_diff.values[-2]
    second_last = ema_diff.values[-3]
    third_last = ema_diff.values[-4]

    if strat == 'ema_diff_peak_trough':
        #short if local peak
        if last < second_last and third_last < second_last:
            return 'SHORT'
        # long if local trough 
        if last > second_last and third_last > second_last:
            return 'LONG'

    if strat == 'ema_cross_over_under':
        # long id diff cross over 0    
        if last > 0 and second_last < 0:
            return "LONG"
        # short if diff cross under 0
        if last < 0 and second_last > 0:
            return "SHORT"

    return 'NONE'

def get_open_trade(mode='paper',symbol='BTC/USDT'):
    exchange = ccxt.binance({
        'apiKey': g_api_key,
        'secret': g_secret_key,
        'enableRateLimit': True,  # https://github.com/ccxt/ccxt/wiki/Manual#rate-limit
        'options': {
            'defaultType': 'future',
        },
        'hedgeMode':True
    })

    # symbol = 'BTC/USDT'
    markets = exchange.load_markets()
    #exchange.verbose=True
    market = exchange.market(symbol)

    positions = [elem  for elem in exchange.fapiPrivate_get_positionrisk() if float(elem['positionAmt'])!=0]

    if len(positions)==0:
        return 0
    return positions

# if len(positions)==0:
#     return 0
# return np.sign(float(positons[0]['positionAmt']))

def get_balance(mode='paper',asset='USDT'):
    exchange = ccxt.binance({
        'apiKey': g_api_key,
        'secret': g_secret_key,
        'enableRateLimit': True,  # https://github.com/ccxt/ccxt/wiki/Manual#rate-limit
        'options': {
            'defaultType': 'future',
        },
        'hedgeMode':True
    })
    return float([elem for elem in exchange.fapiPrivate_get_balance({'asset':asset}) if elem['asset']== asset][0]['balance'])

def close_all_open_positions(mode='paper',symbol='BTC/USDT'):
        
    exchange = ccxt.binance({
            'apiKey': g_api_key,
            'secret': g_secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            },
            'hedgeMode':True
        })
    markets = exchange.load_markets()
    #exchange.verbose=True
    market = exchange.market(symbol)

    open_trade = get_open_trade(mode=mode,symbol=symbol)
    if open_trade == 0:
        return None
    
    if np.sign(float(open_trade[0]['positionAmt'])) == -1.0:
        opp_side = "BUY"
    if np.sign(float(open_trade[0]['positionAmt'])) == 1.0:
        opp_side = "SELL" 
    baseqty= abs(float(open_trade[0]['positionAmt']))
    order = exchange.fapiPrivatePostOrder({'symbol':market['id'], 'type':"MARKET", 'side':opp_side,'positionSide':"BOTH" ,'quantity':baseqty})

    return order

def open_market_long(mode='paper',balance=1,symbol='BTC/USDT',leverage="5"):
        
    exchange = ccxt.binance({
            'apiKey': g_api_key,
            'secret': g_secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            },
            'hedgeMode':True
        })
    # symbol = 'BTC/USDT'
    markets = exchange.load_markets()
    #exchange.verbose=True
    market = exchange.market(symbol)
    quoteqty = float([elem for elem in exchange.fapiPrivate_get_balance() if elem['asset']=='USDT'][0]['balance']) * balance
    price = float(exchange.fapiPublic_get_ticker_price({'symbol':market['id']})['price'])
    baseqty = "{:.3f}".format(quoteqty*float(leverage)/price)
    baseqty = str(float(baseqty)-0.001)
    lev_req = exchange.fapiPrivate_post_leverage({'symbol':market['id'],'leverage':leverage})
    order = exchange.fapiPrivatePostOrder({'symbol':market['id'], 'type':"MARKET", 'side':"BUY",'positionSide':"BOTH" ,'quantity':baseqty})

    return order


def open_market_short(mode='paper',balance=1,symbol='BTC/USDT',leverage="5"):
    
    exchange = ccxt.binance({
            'apiKey': g_api_key,
            'secret': g_secret_key,
            'enableRateLimit': True, 
            'options': {
                'defaultType': 'future',
            },
            'hedgeMode':True
        })
    # symbol = 'BTC/USDT'
    markets = exchange.load_markets()
    #exchange.verbose=True
    market = exchange.market(symbol)
    quoteqty = float([elem for elem in exchange.fapiPrivate_get_balance({'asset':"USDT"}) if elem['asset']=='USDT'][0]['balance']) * balance
    price = float(exchange.fapiPublic_get_ticker_price({'symbol':market['id']})['price'])
    baseqty = "{:.3f}".format(quoteqty*float(leverage)/price)
    baseqty = str(float(baseqty)-0.001)
    lev_req = exchange.fapiPrivate_post_leverage({'symbol':market['id'],'leverage':leverage})
    order = exchange.fapiPrivatePostOrder({'symbol':market['id'], 'type':"MARKET", 'side':"SELL",'positionSide':"BOTH" ,'quantity':baseqty})

    return order

def runner(mode='paper',symbol='BTC/USDT',n=50,interval='1h',asset='USDT',strat='ema_cross_over_under',fast_ema=10,slow_ema=40):
    
    allowed_intervals = ['1d','6h','4h','1h','15m','5m']
    
    if interval not in allowed_intervals:
        print("Invalid Interval")
        return None
    
    check_hours_1d = [0]
    check_hours_6h = [0,6,12,18]
    check_hours_4h = [0,4,8,12,16,20]
    check_hours_1h = [i for i in range(24)]
    check_minutes_h = [0]
    check_minutes_15m = [0,15,30,45]
    check_minutes_5m = [0,5,10,15,20,25,30,35,40,45,50,55]

    if interval == '1h':
        check_hours = check_hours_1h
        check_minutes = check_minutes_h

    if interval == '4h':
        check_hours = check_hours_4h
        check_minutes = check_minutes_h

    if interval == '6h':
        check_hours = check_hours_6h
        check_minutes = check_minutes_h

    if interval == '1d':
        check_hours = check_hours_1d
        check_minutes = check_minutes_h
    
    if interval == '15m':
        check_hours = check_hours_1h
        check_minutes = check_minutes_15m
        
    if interval == '5m':
        check_hours = check_hours_1h
        check_minutes = check_minutes_5m

    print("Starting Runner")
    print("Time Started:")
    print(datetime.datetime.utcnow())  
    
    while True:
        # os.system('cls')
        try:
            
            if datetime.datetime.utcnow().minute in check_minutes and  datetime.datetime.utcnow().hour in check_hours:
                print("New Candle Detected ")
                try:
                    balance = get_balance(mode=mode,asset=asset)
                    # break
                except Exception as e:
                    time.sleep(10)
                    print("Failed to get futures account balance,Retrying...")
                    print(e)
                    continue
                # if ctr>5:
                #     continue
                print("Current Balance:")
                print(balance)

                try:
                    current_input = get_last_n_kline_closes(n=n,interval=interval)
                    
                    # break
                except Exception as e:
                    time.sleep(10)
                    print("Failed to get last n klines,Retrying...")
                    print(e)
                    continue
                try:
                    current_signal = generate_signal(current_input,strat=strat,fast_ema=fast_ema,slow_ema=slow_ema)
                    # time.sleep(60)
                    # break
                except Exception as e:
                    print("Failed to generate signal from last n klines,Retrying...")
                    time.sleep(10)
                    print(e)
                    continue
                print("Current Signal:")
                print(current_signal)
                
                if current_signal == "NONE":
                    time.sleep(60)
                
          
                if current_signal != "NONE":
                    while True:
                        ctr = 0
                        while True:
                            ctr+=1
                                    
                            try:
                                order = get_open_trade(mode=mode,symbol=symbol)
                            
                                if order == 0:
                                    current_exposure = "NONE"
                                    # break

                                else:
                                    current_exposure = np.sign(float(order[0]['positionAmt']))
                                
                                if current_exposure == 1.0:
                                    current_exposure = "LONG"

                                if current_exposure == -1.0:
                                    current_exposure = "SHORT"

                                break

                            except Exception as e:
                                print("Failed to get current open exposure,Retrying...")
                                print(e)
                                time.sleep(10+2*ctr)

                        if current_signal == 'LONG':
                            if current_exposure != 'LONG':
                                print("Closing All Current Positions")
                                ctr=0
                                while True:
                                    ctr+=1
                                    # if ctr > 100:
                                    #     print("Exceeded 100 Retries, waiting 400 Seconds")
                                    #     time.sleep(400)
                                    #     # break
                                    try:
                                        order = close_all_open_positions(mode=mode,symbol=symbol)
                                        print("Closed Open Position Successfully")
                                        print("Details of Closed Position:")
                                        print(order)
                                        break
                                    except Exception as e:
                                        print("Failed to Close all Open Positions,Retrying...")
                                        print(e)
                                        time.sleep(10+2*ctr)

                                print("Opening Long")
                                ctr=0
                                while True:
                                    ctr+=1
                                    # if ctr > 100:
                                    #     print("Exceeded 100 Retries, waiting 400 Seconds")
                                    #     time.sleep(400)
                                    #     # break
                                    try:
                                        order = open_market_long(mode=mode,balance=1,symbol=symbol,leverage="5")
                                        print("5x Long Opened Successfully")
                                        print("Details of Opened Long:")
                                        print(order)
                                        time.sleep(60)
                                        break
                                    except Exception as e:
                                        print("Failed to Open Market Long,Retrying...")
                                        print(e)
                                        time.sleep(10+2*ctr)
                            else:
                                time.sleep(60)

                        if current_signal == 'SHORT':
                            if current_exposure != 'SHORT':
                                print("Closing All Current Positions")
                                ctr=0
                                while True:
                                    ctr+=1
                                    # if ctr > 100:
                                    #     print("Exceeded 100 Retries, waiting 400 Seconds")
                                    #     time.sleep(400)
                                    #     # break
                                    try:
                                        order = close_all_open_positions(mode=mode,symbol=symbol)
                                        print("Closed Open Position Successfully")
                                        print("Details of Closed Position:")
                                        print(order)
                                        break
                                    except Exception as e:
                                        print("Failed to Close all Open Positions,Retrying...")
                                        print(e)
                                        time.sleep(10+2*ctr)

                                print("Opening Short")
                                ctr=0
                                while True:
                                    ctr+=1
                                    # if ctr > 100:
                                    #     print("Exceeded 100 Retries, waiting 400 Seconds")
                                    #     time.sleep(400)
                                    #     # break
                                    try:
                                        order = open_market_short(mode=mode,balance=1,symbol=symbol,leverage="2")
                                        print("2x Short Opened Successfully")
                                        print("Details of Opened Short:")
                                        print(order)
                                        time.sleep(60)
                                        break
                                    except Exception as e:
                                        print("Failed to Open Market Short,Retrying...")
                                        print(e)
                                        time.sleep(10+2*ctr)
                            else:
                                time.sleep(60)            
                        break
        
        except Exception as e:
            print("Iteration Failed,Retrying...")
            print(e)
        



runner(mode='paper',symbol='BTC/USDT',n=50,interval='1d')
#intrval=4h,n=110,fast_ema=50,slow_ema=100,leverage=5


# In[ ]:


