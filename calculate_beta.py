import tushare as ts
from opdata import opdata
import os
import numpy as np
from operator import itemgetter
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import pickle
import time

stock_data_cache = { }
index_data_cache = None

cache_dir = "cache_data"
index_code = "399300"

min_sample_num = 10

def initialize(stock_code_list, start_str, end_str):
    # all data should be fetched once and cached for later useage
    # the cached data are stored in stock_data_cache and index_data
    global stock_data_cache
    global index_data_cache

    if os.path.isdir(cache_dir) == False:
        os.mkdir(cache_dir)

    if os.path.isfile(os.path.join(cache_dir, index_code)):
        f = open(os.path.join(cache_dir, index_code), "rb")
        index_data_cache = pickle.load(f)
        f.close()
    else:
        index_data_cache = ts.get_k_data(index_code, index=True, start=start_str, end=end_str)
        f = open(os.path.join(cache_dir, index_code), "wb")
        pickle.dump(index_data_cache, f)
        f.close()
    print("index data cache done")

    for stock_code in stock_code_list:
        if os.path.isfile(os.path.join(cache_dir, stock_code)):
            f = open(os.path.join(cache_dir, stock_code), "rb")
            stock_data_cache[stock_code] = pickle.load(f)
        else:
            # stock_data_cache[stock_code] = ts.get_k_data(stock_code, start=start_str, end=end_str)
            stock_data_cache[stock_code] = opdata.get_day(stock_code, start_str, end_str)
            f = open(os.path.join(cache_dir, stock_code), "wb")
            pickle.dump(stock_data_cache[stock_code], f)
        print("%s data cache done" % stock_code)

def calculate_beta(stock_code, date_str, period, point_num):
    ''' calculate beta for a specific stock at some specific date
    Argments:
        stock_code:
        date:
        period: base time for calculaing return, e.g., 5 for weekly return, 20 for monthly return
        point_num: number of periods included
        index_data: if index_data is not None, it can be reused
    Return:
        the calculated beta
    '''
    global stock_data_cache
    global index_data_cache
    # obtain the index data if not provided
    end_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days = int(period * point_num * 2))
    start_str = start_date.strftime("%Y-%m-%d")
    if index_data_cache is None:
        index_data_cache = ts.get_k_data("399300", index=True, start=start_str, end=date_str)
    else:
        index_date = index_data_cache['date'].tolist()
        if index_date[0] > start_str or index_date[-1] < date_str:
            index_data = ts.get_k_data("399300", index=True, start=start_str, end=date_str)
        else:
            start_idx = next((index_date.index(n) for n in index_date if n >= start_str), len(index_date))
            end_idx = next((index_date.index(n) for n in index_date if n > date_str), len(index_date))
            index_data = index_data_cache[start_idx:end_idx]

    # calculate the index returns
    index_date = index_data.get('date').tolist()
    index_close = index_data.get('close').tolist()
    data_len = len(index_date)

    data_indexes = []
    for i in range(point_num):
        data_indexes.append(data_len - 1 - i * period)
    data_indexes = data_indexes[::-1]

    index_target_close = itemgetter(*data_indexes)(index_close)
    index_return = [(index_target_close[i+1] / index_target_close[i]) - 1 for i in range(len(index_target_close)-1)]

    # obtain the stock data
    if stock_code not in stock_data_cache.keys():
        stock_data = ts.get_k_data(stock_code, start=start_str, end=date_str)
        # stock_data = opdata.get_day(stock_code, start_str, date_str)
    else:
        stock_date = stock_data_cache[stock_code]['date'].tolist()
        if stock_date[0] > start_str or stock_date[-1] < date_str:
            stock_data = ts.get_k_data(stock_code, start=start_str, end=date_str)
            # stock_data = opdata.get_day(stock_code, start_str, date_str)
        else:
            start_idx = next((stock_date.index(n) for n in stock_date if n >= start_str), len(stock_date))
            end_idx = next((stock_date.index(n) for n in stock_date if n > date_str), len(stock_date))
            stock_data = stock_data_cache[stock_code][start_idx:end_idx]

    # calculate the stock returns
    stock_date = stock_data.get('date').tolist()
    stock_close = stock_data.get('close').tolist()
    date_ary = itemgetter(*data_indexes)(index_date)

    stock_target_close = []
    for date in date_ary:
        if date not in stock_date:
            stock_target_close.append(None)
        else:
            index = stock_date.index(date)
            stock_target_close.append(stock_close[index])
    
    stock_return = []
    for i in range(len(stock_target_close) - 1):
        prev_close = stock_target_close[i]
        next_close = stock_target_close[i + 1]
        if prev_close == None or next_close == None:
            stock_return.append(None)
        else:
            stock_return.append(next_close / prev_close - 1)

    # until now, the index return are stored in index_return, and the stock return are stored in stock_return
    # there might be None value in stock_return
    # the None value should be removed in stock_return, as well as the corresponding values in index_return
    # and then linear regression should be adopted to get beta

    # remove the None value
    stock_return = np.asarray(stock_return)
    index_return = np.asarray(index_return)
    idx = stock_return != None
    stock_return = stock_return[idx]
    index_return = index_return[idx]
    
    if stock_return.shape[0] < min_sample_num:
        return None, None, None, None
    
    # do linear regression to find out beta
    stock_return = np.expand_dims(stock_return, 1)
    index_return = np.expand_dims(index_return, 1)
    
    model = LinearRegression(fit_intercept=True)
    
    model.fit(index_return, stock_return)
    
    alpha = model.intercept_
    beta = model.coef_

    return stock_return, index_return, alpha, beta[0][0]


if __name__ == "__main__":
    calculate_beta("000002", "2017-08-31", 5, 20)

