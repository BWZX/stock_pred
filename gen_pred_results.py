import numpy as np
import xgboost as xgb
import os
import talib
from dataio import opdata
import tushare as ts
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import random
import string
import datetime
import argparse
import pickle
 
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pdb

model = None

pred_interval = 5
year = 2017

def gen_policy(stock_code_list, start_date, end_date, start_trade_date, end_trade_date):
    sample_size = 1000
    # get predictors of hs300
    stock_code = "399300"
    index_data = ts.get_k_data(stock_code, start_date, end_date)

    # date_ary of index data can indicate trade days
    date_ary = np.asarray(index_data["date"].tolist())

    open_list = np.asarray(index_data["open"].tolist())
    close_list = np.asarray(index_data["close"].tolist())
    high_list = np.asarray(index_data["high"].tolist())
    low_list = np.asarray(index_data["low"].tolist())
    volume_list = np.asarray(index_data["volume"].tolist())

    hs300_adj_close = close_list
    hs300_obv = talib.OBV(close_list, volume_list)
    hs300_rsi6 = talib.RSI(close_list, timeperiod=6)
    hs300_rsi12 = talib.RSI(close_list, timeperiod=12)
    hs300_sma3 = talib.SMA(close_list, timeperiod=3)
    hs300_ema6 = talib.EMA(close_list, timeperiod=6)
    hs300_ema12 = talib.EMA(close_list, timeperiod=12)
    hs300_atr14 = talib.ATR(high_list, low_list, close_list, timeperiod=14)
    hs300_mfi14 = talib.MFI(high_list, low_list, close_list, volume_list, timeperiod=14)
    hs300_adx14 = talib.ADX(high_list, low_list, close_list, timeperiod=14)
    hs300_adx20 = talib.ADX(high_list, low_list, close_list, timeperiod=20)
    hs300_mom1 = talib.MOM(close_list, timeperiod=1)
    hs300_mom3 = talib.MOM(close_list, timeperiod=3)
    hs300_cci12 = talib.CCI(high_list, low_list, close_list, timeperiod=14)
    hs300_cci20 = talib.CCI(high_list, low_list, close_list, timeperiod=20)
    hs300_rocr3 = talib.ROCR(close_list, timeperiod=3)
    hs300_rocr12 = talib.ROCR(close_list, timeperiod=12)
    hs300_macd, hs300_macd_sig, hs300_macd_hist = talib.MACD(close_list)
    hs300_willr = talib.WILLR(high_list, low_list, close_list)
    hs300_tsf10 = talib.TSF(close_list, timeperiod=10)
    hs300_tsf20 = talib.TSF(close_list, timeperiod=20)
    hs300_trix = talib.TRIX(close_list)
    hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower = talib.BBANDS(close_list)

    # the following dict variables store data for each stock
    stock_date_dict = {}
    stock_predictor_dict = {}
    stock_close_dict = {}

    for stock_code in stock_code_list:
        print(stock_code)
        stock_data = opdata.get_day(stock_code, start_date, end_date)

        date_list = np.asarray(stock_data["date"].tolist())
        open_list = np.asarray(stock_data["open"].tolist())
        close_list = np.asarray(stock_data["close"].tolist())
        high_list = np.asarray(stock_data["high"].tolist())
        low_list = np.asarray(stock_data["low"].tolist())
        volume_list = np.asarray(stock_data["volume"].tolist())

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

        predictors = [adj_close, obv, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower, hs300_adj_close, hs300_obv, hs300_rsi6, hs300_rsi12, hs300_sma3, hs300_ema6, hs300_ema12, hs300_atr14, hs300_mfi14, hs300_adx14, hs300_adx20, hs300_mom1, hs300_mom3, hs300_cci12, hs300_cci20, hs300_rocr3, hs300_rocr12, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_willr, hs300_tsf10, hs300_tsf20, hs300_trix, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower]

        # get the nan value index
        max_nan_idx = -1
        for predictor in predictors:
            nan_idxes = np.argwhere(np.isnan(predictor))
            if nan_idxes.shape[0] == 0:
                continue
            nan_idx = nan_idxes[-1, 0]
            max_nan_idx = max(max_nan_idx, nan_idx)

        # remove nan values
        date_list = date_list[max_nan_idx+1:]
        close_list = close_list[max_nan_idx+1:]
        for idx, predictor in enumerate(predictors):
            predictor = predictor[max_nan_idx+1:]
            predictors[idx] = predictor
        if len(set([e.shape[0] for e in predictors])) > 1:
            # predictors have different time length, there must be something wrong
            print("ERROR!!!!!!!!!!!!!!")
            continue

        stock_date_dict[stock_code] = date_list
        stock_close_dict[stock_code] = close_list
        stock_predictor_dict[stock_code] = predictors

    start_trade_idx = np.where(date_ary == start_trade_date)[0][0]
    end_trade_idx = np.where(date_ary == end_trade_date)[0][0]

    predict_results = []

    for trade_idx in range(start_trade_idx, end_trade_idx):
        date = date_ary[trade_idx]
        print(date)
        cur_predict_result = {}
        for stock_code in stock_date_dict.keys():
            cur_idx = np.where(stock_date_dict[stock_code] == date)[0][0]
            if cur_idx <= sample_size:
                continue
            algo_start_idx = cur_idx - sample_size
            close_data = stock_close_dict[stock_code][algo_start_idx:cur_idx]
            index_close_data = np.asarray(index_data["close"].tolist()[trade_idx-sample_size:trade_idx])
            predictor_data = [e[algo_start_idx:cur_idx] for e in stock_predictor_dict[stock_code]]

            # use close_data and predictor_data to construct a model
            cur_predict_result[stock_code] = predict_one_stock(close_data, index_close_data, predictor_data, sample_size)

        predict_results.append([date, cur_predict_result])

    pickle.dump(predict_results, open("policies/policy_%s_%d_%d_low_weight" % (model, pred_interval, year), "wb"))


# predict for one stock in one specific date
# return 1 for predicting going up
# return 0 for predicting going down
def predict_one_stock(close_data, index_close_data, predictors, sample_size):

    predictors = np.vstack(predictors)
    predictors = np.transpose(predictors)

    # get the y data
    future_close_data = close_data[pred_interval:]
    future_index_close_data = index_close_data[pred_interval:]

    y_data = []
    for idx, _ in enumerate(future_close_data):
        stock_percent = (future_close_data[idx] / close_data[idx] - 1) * 100
        index_percent = (future_index_close_data[idx] / index_close_data[idx] - 1) * 100
        if stock_percent > index_percent:
            y_data.append(1)
        else:
            y_data.append(0)

    y_data = np.asarray(y_data)

    train_set_x = predictors[0:sample_size-pred_interval]
    train_set_y = y_data

    scaler = preprocessing.StandardScaler().fit(train_set_x)

    norm_train_set_x = scaler.transform(train_set_x)
    norm_test_set_x = scaler.transform(predictors[sample_size-1:])


    # model = "adaboost"
    if model == "xgb":
        dtrain = xgb.DMatrix(norm_train_set_x, label=train_set_y)
        dtest = xgb.DMatrix(norm_test_set_x)
        param = {'max_depth': 5, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        evallist = [(dtest,'eval'), (dtrain,'train')]
        num_round = 20
        bst = xgb.train(param.items(), dtrain, num_round)
        pred_prob = bst.predict(dtest)
        pred_result = (pred_prob[0] > 0.5).astype(int)
    else:
        if model == "rf":
            clf = RandomForestClassifier(n_estimators=200)
        if model == "adaboost":
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                                     n_estimators=200)

        clf.fit(norm_train_set_x, train_set_y)
        pred_result = clf.predict(norm_test_set_x)[0]

    return pred_result



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='model name.',
                        choices=['xgb', 'rf', 'adaboost'],
                        required=True)
    args = parser.parse_args()

    model = args.model

    # for different stock codes, for different prediction intervals, for different percent
    stock_list = ts.get_hs300s()
    stock_list = stock_list.loc[stock_list['weight'] < 0.1]

    stock_code_list = stock_list["code"].tolist()

    # gen_policy(stock_code_list,
    gen_policy(stock_code_list,
               start_date="2011-01-01",
               end_date="2017-09-10",
               start_trade_date="2017-01-04",
               end_trade_date="2017-08-31")


