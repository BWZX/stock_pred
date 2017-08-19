import numpy as np
import talib
from dataio import opdata
import tushare as ts
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import random
import string

from sklearn import preprocessing
from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import pdb

def random_str(N=6):
    ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))


def run(stock_code, model, start_date="2012-01-01", end_date="2017-05-31", percent=0.5, pred_interval=5):

    # start_date = "2011-01-01"
    # end_date = "2017-05-31"
    # percent = 0.3
    # pred_interval = 5
    
    # stock data
    # stock_code = "600036"
    stock_data = opdata.get_day(stock_code, start_date, end_date)
    
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
    
    # index data (hs300)
    stock_code = "399300"
    index_data = ts.get_k_data(stock_code, start_date, end_date)
    
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
    
    predictors = [adj_close, obv, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower, hs300_adj_close, hs300_obv, hs300_rsi6, hs300_rsi12, hs300_sma3, hs300_ema6, hs300_ema12, hs300_atr14, hs300_mfi14, hs300_adx14, hs300_adx20, hs300_mom1, hs300_mom3, hs300_cci12, hs300_cci20, hs300_rocr3, hs300_rocr12, hs300_macd, hs300_macd_sig, hs300_macd_hist, hs300_willr, hs300_tsf10, hs300_tsf20, hs300_trix, hs300_bbandupper, hs300_bbandmiddle, hs300_bbandlower]
    
    # predictors = [adj_close, obv, rsi6, rsi12, sma3, ema6, ema12, atr14, mfi14, adx14, adx20, mom1, mom3, cci12, cci20, rocr3, rocr12, macd, macd_sig, macd_hist, willr, tsf10, tsf20, trix, bbandupper, bbandmiddle, bbandlower]

    # get the nan value index
    max_nan_idx = -1
    for predictor in predictors:
        nan_idxes = np.argwhere(np.isnan(predictor))
        if nan_idxes.shape[0] == 0:
            continue
        nan_idx = nan_idxes[-1, 0]
        max_nan_idx = max(max_nan_idx, nan_idx)

    # remove nan values
    for idx, predictor in enumerate(predictors):
        predictor = predictor[max_nan_idx+1:]
        predictors[idx] = predictor
    if len(set([e.shape[0] for e in predictors])) > 1:
        return None
    predictors = np.vstack(predictors)
    predictors = np.transpose(predictors)

    # get the y data
    close_list = np.asarray(stock_data["close"].tolist())
    close_list = close_list[max_nan_idx+1:]
    future_close_list = close_list[pred_interval:]
    y_data = []
    for idx, _ in enumerate(future_close_list):
        if future_close_list[idx] > close_list[idx]:
            y_data.append(1)
        else:
            y_data.append(0)
    
    y_data = np.asarray(y_data)

    train_size = 950
    test_size = 50
    all_size = train_size + test_size
    
    tot_len = predictors.shape[0]
    
    start_idx = 0
    
    acc_ary = []
    idxes_ary = []
    
    while start_idx + all_size <= tot_len - pred_interval:
        train_set_x = predictors[start_idx:start_idx+train_size]
        train_set_y = y_data[start_idx:start_idx+train_size]
        test_set_x = predictors[start_idx+train_size:start_idx+train_size+test_size]
        test_set_y = y_data[start_idx+train_size:start_idx+train_size+test_size]
    
        scaler = preprocessing.StandardScaler().fit(train_set_x)
    
        norm_train_set_x = scaler.transform(train_set_x)
        norm_test_set_x = scaler.transform(test_set_x)

        if model == "extra_trees":
            clf = ExtraTreesClassifier()
        elif model == "svm":
            clf = SVC()
        elif model == "random_forrest":
            clf = RandomForestClassifier(n_estimators=100)
        else:
            n_estimators = int(model.split("_")[1])
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                                     algorithm="SAMME",
                                     n_estimators=n_estimators)

        clf.fit(norm_train_set_x, train_set_y)
        test_set_y_pred = clf.predict(norm_test_set_x)

        '''
        # feature selection with extremely randomized trees
        et = ExtraTreesClassifier()
        et.fit(norm_train_set_x, train_set_y)
    
        feat_import = et.feature_importances_
        sort_feat_import = np.sort(feat_import)
        idx_feat_import = np.argsort(feat_import)
        cum_feat_import = np.cumsum(sort_feat_import)
        start_feat_idx = np.where(cum_feat_import >= 1 - percent)[0][0] - 1
        start_feat_idx = max(start_feat_idx, 0)
        idxes = idx_feat_import[start_feat_idx:]
        idxes.sort()
        idxes_ary.append(len(idxes))
    
        _norm_train_set_x = norm_train_set_x[:, idxes]
        _norm_test_set_x = norm_test_set_x[:, idxes]
    
        # train and test model
        clf = SVC()
        clf.fit(_norm_train_set_x, train_set_y)
        test_set_y_pred = clf.predict(_norm_test_set_x)
        # train_set_y_pred = clf.predict(_norm_train_set_x)
        '''
    
        corr = np.sum((test_set_y_pred == test_set_y).astype(int))
        # corr = np.sum((train_set_y_pred == train_set_y).astype(int))
    
        acc = corr / len(test_set_y)
        # acc = corr / len(train_set_y)
        # print(acc)
        acc_ary.append(acc)
    
        start_idx += 10
    
    acc_mean = np.mean(acc_ary)
    idxes_num_mean = np.mean(idxes_ary)
    
    # print(np.mean(acc_ary))
    
    return  [acc_mean, idxes_num_mean, acc_ary]

if __name__ == "__main__":
    '''
    [acc_mean, idxes_num_mean, acc_ary] = run_svm(stock_code="000001",
                                                  start_date="2010-01-01",
                                                  end_date="2017-05-31",
                                                  percent=0.5,
                                                  pred_interval=5)
    '''
    # for different stock codes, for different prediction intervals, for different percent
    stock_list = ts.get_hs300s()
    stock_list = stock_list.loc[stock_list['weight'] > 1]

    stock_code_list = stock_list["code"].tolist()
    pred_interval = 5
    percent = 1

    # models = ["extra_trees", "svm", "random_forrest", "adaboost"]
    models = ["adaboost_50", "adaboost_100", "adaboost_200"]

    acc_result = []

    fig = plt.figure()

    for stock_code in stock_code_list[:1]:
        acc_mean_ary = []
        for model in models:
            retval = run(stock_code,
                         model,
                         start_date="2011-01-01",
                         end_date="2017-05-31",
                         percent=1,
                         pred_interval=pred_interval)
            if retval == None:
                continue
            acc_mean, _, acc_ary = retval
            acc_mean_ary.append(acc_mean)

            # save result as figure
            plt.plot(acc_ary)
            plt.ylim(0, 1)
            plt.ylabel('accuracy')
            plt.xlabel('time (10 days)')
            plt.title(stock_code + ' ' + model + ' (' + str(acc_mean) + ')')
            fig.savefig('result_imgs/' + stock_code + '_' + model + '_' + random_str() + '.jpg')
            plt.clf()

        if len(acc_mean_ary) == 0:
            continue
        acc_result.append(np.asarray(acc_mean_ary))

    pdb.set_trace()
