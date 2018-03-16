import numpy as np
import argparse
import copy
import os
import shutil
import talib
from opdata import opdata
import tushare as ts
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import random
import string
import datetime
import time
from sklearn import preprocessing

from cfgs.config import cfg
from calculate_beta import *

import pdb

def normalize_price(adj_close, sma3, ema6, ema12, atr14, mom1, mom3, tsf10, tsf20, macd, macd_sig, macd_hist, bbandupper, bbandmiddle, bbandlower):
    max_price = np.max(adj_close)

    adj_close = adj_close / max_price
    sma3 = sma3 / max_price
    ema6 = ema6 / max_price
    ema12 = ema12 / max_price
    atr14 = atr14 / max_price
    mom1 = mom1 / max_price
    mom3 = mom3 / max_price
    tsf10 = tsf10 / max_price
    tsf20 = tsf20 / max_price
    macd = macd / max_price
    macd_sig = macd_sig / max_price
    macd_hist = macd_hist / max_price
    bbandupper = bbandupper / max_price
    bbandmiddle = bbandmiddle / max_price
    bbandlower = bbandlower / max_price

    return adj_close, sma3, ema6, ema12, atr14, mom1, mom3, tsf10, tsf20, macd, macd_sig, macd_hist, bbandupper, bbandmiddle, bbandlower

def get_predictors(data):
    open_list = np.asarray(data["open"].tolist())
    close_list = np.asarray(data["close"].tolist())
    high_list = np.asarray(data["high"].tolist())
    low_list = np.asarray(data["low"].tolist())
    volume_list = np.asarray(data["volume"].tolist())

    adj_close = close_list
    obv = talib.OBV(close_list, volume_list)
    rsi6 = talib.RSI(close_list, timeperiod=6)
    rsi12 = talib.RSI(close_list, timeperiod=12)
    sma3 = talib.SMA(close_list, timeperiod=3)
    ema6 = talib.EMA(close_list, timeperiod=6)
    ema12 = talib.EMA(close_list, timeperiod=12)
    atr14 = talib.ATR(high_list, low_list, close_list, timeperiod=14)
    mfi14 = talib.MFI(high_list, low_list, close_list, volume_list, timeperiod=14)
    adx14 = talib.ADX(high_list, low_list, close_list, timeperiod=14)
    adx20 = talib.ADX(high_list, low_list, close_list, timeperiod=20)
    mom1 = talib.MOM(close_list, timeperiod=1)
    mom3 = talib.MOM(close_list, timeperiod=3)
    cci12 = talib.CCI(high_list, low_list, close_list, timeperiod=14)
    cci20 = talib.CCI(high_list, low_list, close_list, timeperiod=20)
    rocr3 = talib.ROCR(close_list, timeperiod=3)
    rocr12 = talib.ROCR(close_list, timeperiod=12)
    macd, macd_sig, macd_hist = talib.MACD(close_list)
    willr = talib.WILLR(high_list, low_list, close_list)
    tsf10 = talib.TSF(close_list, timeperiod=10)
    tsf20 = talib.TSF(close_list, timeperiod=20)
    trix = talib.TRIX(close_list)
    bbandupper, bbandmiddle, bbandlower = talib.BBANDS(close_list)
    return [adj_close, volume_list, obv, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, \
            rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower]

def run(stock_list, start_date, end_date, pred_interval, save_dir_prefix, train_size, test_size):

    # index data (hs300)
    index_data = ts.get_k_data(cfg.index_code, index=True, start=start_date, end=end_date)
    # index_data = opdata.get_day(cfg.index_code, start_date, end_date)

    hs300_adj_close, hs300_volume, hs300_obv, hs300_rsi6, hs300_rsi12, hs300_sma3, hs300_ema6, hs300_ema12, \
    hs300_atr14, hs300_mfi14, hs300_adx14, hs300_adx20, hs300_mom1, hs300_mom3, hs300_cci12, \
    hs300_cci20, hs300_rocr3, hs300_rocr12, hs300_macd, hs300_macd_sig, hs300_macd_hist, \
    hs300_willr, hs300_tsf10, hs300_tsf20, hs300_trix, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower = \
        get_predictors(index_data)

    '''
    # normalize predictors
    hs300_adj_close, hs300_sma3, hs300_ema6, hs300_ema12, hs300_atr14, hs300_mom1, hs300_mom3, hs300_tsf10, hs300_tsf20, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower = \
        normalize_price(hs300_adj_close, hs300_sma3, hs300_ema6, hs300_ema12, hs300_atr14, hs300_mom1, hs300_mom3, hs300_tsf10, hs300_tsf20, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower)

    hs300_volume = hs300_volume / np.max(hs300_volume)

    hs300_rsi6 = (hs300_rsi6 - 50) / 50
    hs300_rsi12 = (hs300_rsi12 - 50) / 50
    hs300_mfi14 = (hs300_mfi14 - 50) / 50
    hs300_adx14 = (hs300_adx14 - 50) / 50
    hs300_adx20 = (hs300_adx20 - 50) / 50
    hs300_willr = hs300_willr / 100

    hs300_cci12 = hs300_cci12 / 1000
    hs300_cci20 = hs300_cci20 / 1000
    '''

    # training data includes samples from all stocks
    stocks_y_data = []
    stocks_predictors = []
    for stock_code in stock_list:

        cache_data_path = os.path.join(cfg.cache_dir, "%s_%s_%s" % (stock_code, start_date, end_date))
        if os.path.isfile(cache_data_path):
            f = open(cache_data_path, 'rb')
            stock_data = pickle.load(f)
            f.close()
        else:
            stock_data = opdata.get_day(stock_code, start_date, end_date)
            f = open(cache_data_path, 'wb')
            pickle.dump(stock_data, f)
            f.close()

        date_list = stock_data['date'].tolist()

        adj_close, volume, obv, rsi6, rsi12, sma3, ema6, ema12, \
        atr14, mfi14, adx14, adx20, mom1, mom3, cci12, \
        cci20, rocr3, rocr12, macd, macd_sig, macd_hist, \
        willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower = \
            get_predictors(stock_data)

        '''
        # normalize predictors
        adj_close, sma3, ema6, ema12, atr14, mom1, mom3, tsf10, tsf20, macd, macd_sig, macd_hist, bbandupper, bbandmiddle, bbandlower = \
            normalize_price(adj_close, sma3, ema6, ema12, atr14, mom1, mom3, tsf10, tsf20, macd, macd_sig, macd_hist, bbandupper, bbandmiddle, bbandlower)

        volume = volume / np.max(volume)

        rsi6 = (rsi6 - 50) / 50
        rsi12 = (rsi12 - 50) / 50
        mfi14 = (mfi14 - 50) / 50
        adx14 = (adx14 - 50) / 50
        adx20 = (adx20 - 50) / 50
        willr = willr / 100

        cci12 = cci12 / 1000
        cci20 = cci20 / 1000
        '''

        # predictors = [adj_close, obv, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower, hs300_adj_close, hs300_obv, hs300_rsi6, hs300_rsi12, hs300_sma3, hs300_ema6, hs300_ema12, hs300_atr14, hs300_mfi14, hs300_adx14, hs300_adx20, hs300_mom1, hs300_mom3, hs300_cci12, hs300_cci20, hs300_rocr3, hs300_rocr12, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_willr, hs300_tsf10, hs300_tsf20, hs300_trix, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower, macro_gdp_clone, macro_cpi_clone, macro_m2_clone, macro_rrr_clone, macro_rate_clone, finance_bvps, finance_epcf, finance_eps]
        # 26 features of individual stock (obv is removed)
        # predictors = [adj_close, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower]

        # 54 features of individual stock and index (obv is replaced by volume)
        predictors = [hs300_adj_close, hs300_volume, hs300_rsi6, hs300_rsi12, hs300_sma3, hs300_ema6, hs300_ema12, hs300_atr14, hs300_mfi14, hs300_adx14, hs300_adx20, hs300_mom1, hs300_mom3, hs300_cci12, hs300_cci20, hs300_rocr3, hs300_rocr12, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_willr, hs300_tsf10, hs300_tsf20, hs300_trix, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower, adj_close, volume, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower]

        # 38 features for cnn
        # predictors = [hs300_adj_close, hs300_volume, hs300_rsi12, hs300_ema6, hs300_atr14, hs300_mfi14, hs300_adx14, hs300_mom1, hs300_cci12, hs300_rocr3, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_willr, hs300_tsf10, hs300_trix, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower, adj_close, volume, rsi12, ema6, atr14, mfi14, adx14, mom1, cci12, rocr3, macd, macd_sig, macd_hist, willr, tsf10, trix, bbandupper, bbandmiddle, bbandlower]

        # get the nan value index
        max_nan_idx = -1
        for idx, predictor in enumerate(predictors):
            nan_idxes = np.argwhere(np.isnan(predictor))
            if nan_idxes.shape[0] == 0:
                continue
            nan_idx = nan_idxes[-1, 0]
            max_nan_idx = max(max_nan_idx, nan_idx)

        # remove nan values
        cur_date_list = date_list[max_nan_idx+1:]
        for idx, predictor in enumerate(predictors):
            predictor = predictor[max_nan_idx+1:]
            predictors[idx] = predictor

        if len(set([e.shape[0] for e in predictors])) > 1:
            # predictors have different time length, there must be something wrong
            continue

        predictors = np.vstack(predictors)
        predictors = np.transpose(predictors)
        predictors = predictors[:-pred_interval]

        # get the y data
        date_list = np.asarray(stock_data["date"].tolist())
        date_list = date_list[max_nan_idx+1:]
        close_list = np.asarray(stock_data["close"].tolist())
        close_list = close_list[max_nan_idx+1:]
        future_close_list = close_list[pred_interval:]
    
        index_close_list = np.asarray(index_data["close"].tolist())
        index_close_list = index_close_list[max_nan_idx+1:]
        future_index_close_list = index_close_list[pred_interval:]
    
        y_data = []
        
        for idx, _ in enumerate(future_close_list):
            if close_list[idx] == 0:
                import pdb
                pdb.set_trace()
            stock_percent = (future_close_list[idx] / close_list[idx] - 1) * 100
            index_percent = (future_index_close_list[idx] / index_close_list[idx] - 1) * 100
            # the stock return should be neutralized by index return
            _, _, _, beta = calculate_beta(stock_code, date_list[idx], pred_interval, 20)
            # beta = 1
            if beta == None:
                y_data.append(None)
                continue
            if stock_percent - index_percent * beta >= 0:
                y_data.append(stock_percent - index_percent * beta)
            else:
                y_data.append(stock_percent - index_percent * beta)
    
        y_data = np.asarray(y_data)

        stocks_predictors.append(predictors)
        stocks_y_data.append(y_data)

    all_size = train_size + test_size
    tot_len = stocks_predictors[0].shape[0]
    
    start_idx = cfg.sample_time_len - 1
    
    acc_ary = []

    # save data to dir
    sub_data_dir = "%s_%d_%d" % (save_dir_prefix, cfg.sample_time_len, pred_interval)
    sub_data_dir_path = os.path.join(cfg.dataset_dir, sub_data_dir)
    if os.path.isdir(sub_data_dir_path):
        shutil.rmtree(sub_data_dir_path)
    os.mkdir(sub_data_dir_path)

    while start_idx + all_size <= tot_len - pred_interval:

        print(str(pred_interval) + ", " + str(start_idx))

        train_set_x = []
        for predictors in stocks_predictors:
            # for one stock
            for idx in range(start_idx, start_idx+train_size-pred_interval+1):
                train_set_x.append(np.expand_dims(predictors[idx-cfg.sample_time_len+1:idx+1], 0))
        train_set_x = np.concatenate(train_set_x)
        train_set_y = [y_data[start_idx:start_idx+train_size-pred_interval+1] for y_data in stocks_y_data]
        train_set_y = np.concatenate(train_set_y)
        
        val_idx = train_set_y != None
        train_set_x = train_set_x[val_idx,:]
        train_set_y = train_set_y[val_idx].astype(np.float)
        
        test_set_x = []
        for predictors in stocks_predictors:
            # for one stock
            for idx in range(start_idx+train_size, start_idx+train_size+test_size):
                test_set_x.append(np.expand_dims(predictors[idx-cfg.sample_time_len+1:idx+1], 0))
        test_set_x = np.concatenate(test_set_x)
        test_set_y = [y_data[start_idx+train_size:start_idx+train_size+test_size] for y_data in stocks_y_data]
        test_set_y = np.concatenate(test_set_y)
        
        val_idx = test_set_y != None
        test_set_x = test_set_x[val_idx,:]
        test_set_y = test_set_y[val_idx].astype(np.float)
        
        norm_train_set_x = train_set_x
        norm_test_set_x = test_set_x

        cur_date = date_list[start_idx + train_size]

        f_train_x = open(os.path.join(sub_data_dir_path, "%d_train_x_%s" % (start_idx, cur_date)), 'wb')
        f_train_y = open(os.path.join(sub_data_dir_path, "%d_train_y_%s" % (start_idx, cur_date)), 'wb')
        f_test_x = open(os.path.join(sub_data_dir_path, "%d_test_x_%s" % (start_idx, cur_date)), 'wb')
        f_test_y = open(os.path.join(sub_data_dir_path, "%d_test_y_%s" % (start_idx, cur_date)), 'wb')

        pickle.dump(norm_train_set_x, f_train_x)
        pickle.dump(train_set_y, f_train_y)
        pickle.dump(norm_test_set_x, f_test_x)
        pickle.dump(test_set_y, f_test_y)

        f_train_x.close()
        f_train_y.close()
        f_test_x.close()
        f_test_y.close()

        start_idx += test_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lower_weight', help='lower bound of the stock weight in hs300', default=1)
    parser.add_argument('--upper_weight', help='upper bound of the stock weight in hs300')
    parser.add_argument('--init_start_date',
                        help='start date of the initialization, should be earlier then start date for the calculation of beta',
                        default='2010-01-01')
    parser.add_argument('--start_date', help='start date of the dataset', default='2011-01-05')
    parser.add_argument('--end_date', help='end date of the dataset', default='2017-12-31')
    parser.add_argument('--train_size', help='size of each training set', default=900)
    parser.add_argument('--test_size', help='size of each test set', default=10)
    parser.add_argument('--save_dir_prefix', required=True)
    parser.add_argument('--pred_interval', default='3')
    args = parser.parse_args()

    # make sure that the dataset dir exists
    if os.path.isdir(cfg.dataset_dir) == False:
        os.mkdir(cfg.dataset_dir)

    # for different stock codes, for different prediction intervals, for different percent
    stock_list = ts.get_hs300s()
    
    if args.lower_weight != None:
        stock_list = stock_list.loc[stock_list['weight'] > args.lower_weight]
    if args.upper_weight != None:
        stock_list = stock_list.loc[stock_list['weight'] < args.upper_weight]

    stock_code_list = stock_list["code"].tolist()
    
    # ugly, remove 000333
    stock_code_list.remove('000333')
    
    initialize(stock_code_list, args.init_start_date, args.end_date)

    pred_intervals = [int(e) for e in args.pred_interval.split(',')]

    for pred_interval in pred_intervals:
        run(stock_code_list,
            start_date=args.start_date,
            end_date=args.end_date,
            pred_interval=pred_interval,
            save_dir_prefix=args.save_dir_prefix,
            train_size=int(args.train_size),
            test_size=int(args.test_size),
            )
