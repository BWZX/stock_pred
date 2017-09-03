import numpy as np
import os
import talib
from dataio import opdata
import tushare as ts
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.pylab import date2num
import matplotlib.finance as mpf
import random
import string
import datetime
 
from sklearn import preprocessing
from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import pdb

data_dir = "svm_relative_0"

def random_str(N=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0,1]


def run(stock_code, model_desc, start_date="2012-01-01", end_date="2017-05-31", pred_interval=5):

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
        # predictors have different time length, there must be something wrong
        return None
    predictors = np.vstack(predictors)
    predictors = np.transpose(predictors)

    # get the y data
    close_list = np.asarray(stock_data["close"].tolist())
    close_list = close_list[max_nan_idx+1:]
    future_close_list = close_list[pred_interval:]

    index_close_list = np.asarray(index_data["close"].tolist())
    index_close_list = index_close_list[max_nan_idx+1:]
    future_index_close_list = index_close_list[pred_interval:]

    y_data = []
    for idx, _ in enumerate(future_close_list):
        # if future_close_list[idx] / close_list[idx] >= 1.01:
        stock_percent = (future_close_list[idx] / close_list[idx] - 1) * 100
        index_percent = (future_index_close_list[idx] / index_close_list[idx] - 1) * 100
        if stock_percent - index_percent >= 0:
            # 1 indicates up more then 1 percent
            y_data.append(1)
        else:
            # other wise, it is 0
            y_data.append(0)
    
    y_data = np.asarray(y_data)

    model, param, preprocess = model_desc.split('_')

    train_size = int(param)
    # train_size = 950
    test_size = 50
    all_size = train_size + test_size
    
    tot_len = predictors.shape[0]
    
    # start_idx = 0
    start_idx = 950 - train_size
    
    acc_ary = []
    prec_ary = []
    rec_ary = []
    det_event = []
    gt_event = []
    det_gt_event = []

    while start_idx + all_size <= tot_len - pred_interval:
        print("current_idx: " + str(start_idx))
        train_set_x = predictors[start_idx:start_idx+train_size]
        train_set_y = y_data[start_idx:start_idx+train_size]
        test_set_x = predictors[start_idx+train_size:start_idx+train_size+test_size]
        test_set_y = y_data[start_idx+train_size:start_idx+train_size+test_size]

        scaler = preprocessing.StandardScaler().fit(train_set_x)

        norm_train_set_x = scaler.transform(train_set_x)
        norm_test_set_x = scaler.transform(test_set_x)

        if preprocess.startswith('ica'):
            max_iter = int(preprocess.split('+')[1])
            ica = FastICA(max_iter=max_iter)
            norm_train_set_x = ica.fit_transform(norm_train_set_x)
            norm_test_set_x = ica.transform(norm_test_set_x)
        elif preprocess.startswith('pca'):
            energy_th = float(preprocess.split('+')[1])
            pca = PCA()
            pca.fit(norm_train_set_x)
            energy_dist = pca.explained_variance_ratio_
            energy_cum = np.cumsum(energy_dist)
            n_components = np.where(energy_cum > energy_th)[0][0] + 1

            pca = PCA(n_components=n_components)
            norm_train_set_x = pca.fit_transform(norm_train_set_x)
            norm_test_set_x = pca.transform(norm_test_set_x)

        if model == "extra_trees":
            clf = ExtraTreesClassifier()
        elif model == "logistic":
            clf = LogisticRegression()
        elif model == "svm":
            clf = SVC()
        elif model == "random_forrest":
            n_estimators = int(param)
            clf = RandomForestClassifier(n_estimators=n_estimators)
        else:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                                     n_estimators=200)

        clf.fit(norm_train_set_x, train_set_y)
        test_set_y_pred = clf.predict(norm_test_set_x)

        corr = np.sum((test_set_y_pred == test_set_y).astype(int))

        acc = corr / len(test_set_y)
        acc_ary.append(acc)

        if np.sum(test_set_y_pred) > 0:
            prec = np.sum(test_set_y_pred * test_set_y) / np.sum(test_set_y_pred)
        else:
            prec = 0.5

        if np.sum(test_set_y) > 0:
            rec = np.sum(test_set_y_pred * test_set_y) / np.sum(test_set_y)
        else:
            rec = 0.5

        prec_ary.append(prec)
        rec_ary.append(rec)

        gt_event.append(np.sum(test_set_y))
        det_event.append(np.sum(test_set_y_pred))
        det_gt_event.append(np.sum(test_set_y_pred * test_set_y))

        start_idx += 10

    acc_mean = np.mean(acc_ary)
    prec_mean = np.sum(det_gt_event) / np.sum(det_event)
    rec_mean = np.sum(det_gt_event) / np.sum(gt_event)

    return  [acc_mean, acc_ary, prec_mean, prec_ary, rec_mean, rec_ary, det_event, stock_data]

if __name__ == "__main__":
    # for different stock codes, for different prediction intervals, for different percent
    stock_list = ts.get_hs300s()
    stock_list = stock_list.loc[stock_list['weight'] > 1]

    stock_code_list = stock_list["code"].tolist()[:1]
    pred_intervals = [1, 5, 10, 20]

    # models = ["extra_trees", "svm", "random_forrest", "adaboost"]
    # model_desc_ary = ["adaboost_50_pca+0.9", "adaboost_50_pca+0.95", "adaboost_50_pca+0.99"]
    # model_desc_ary = ["adaboost_950_None", "adaboost_600_None", "adaboost_250_None"]
    model_desc_ary = ["svm_950_None"]

    acc_result = []
    prec_result = []
    rec_result = []

    for stock_code in stock_code_list:
        print("stock code: " + stock_code)
        acc_mean_ary = []
        prec_mean_ary = []
        rec_mean_ary = []
        for model_idx, model_desc in enumerate(model_desc_ary):
            print("model description: " + model_desc)
            retval = run(stock_code,
                         model_desc,
                         start_date="2011-01-01",
                         end_date="2017-05-31",
                         pred_interval=5)
            if retval == None:
                continue
            acc_mean, acc_ary, prec_mean, prec_ary, rec_mean, rec_ary, det_event, stock_data = retval
            acc_mean_ary.append(acc_mean)
            prec_mean_ary.append(prec_mean)
            rec_mean_ary.append(rec_mean)

            # plot k-line chart based on stock data
            data_list = []
            date_list = []
            for row in stock_data.iterrows():
                row_data = row[1]
                date_time = datetime.datetime.strptime(row_data["date"],'%Y-%m-%d')
                t = date2num(date_time)
                dp = (t, row_data['open'], row_data['high'], row_data['low'], row_data['close'])
                data_list.append(dp)
                date_list.append(row_data['date'])

            def save_k_line_fig(stock_code, date_list, data_list, fig_type):
                fig, ax = plt.subplots()
                fig.subplots_adjust(bottom=0.2)
                ax.xaxis_date()
                plt.xticks(rotation=45)
                plt.yticks()
                plt.title(stock_code + ' ' + date_list[0] + "--" + date_list[-1])
                plt.xlabel('date')
                plt.ylabel('price (yuan)')
                mpf.candlestick_ohlc(ax, data_list, width=1.5, colorup='r', colordown='green')
                plt.grid()
                fig.savefig(data_dir + '/' + stock_code + '_' + fig_type + '.jpg')
                plt.clf()
                plt.close()

            if os.path.isfile(data_dir + '/' + stock_code + '_all.jpg') == False:
                save_k_line_fig(stock_code, date_list, data_list, "all")
                save_k_line_fig(stock_code, date_list[950:], data_list[950:], "predict")

            file_random_str = random_str()

            # calculate the auto-correlation
            auto_corr = [1]
            for lag in range(1, 20):
                auto_corr.append(autocorr(prec_ary, lag))

            # save number of gt events detected as figure
            fig = plt.figure()
            plt.plot(det_event, "o-")
            plt.ylim(0, np.max(det_event) + 1)
            plt.xlabel('number of gt events detected')
            plt.xlabel('time (10 days)')
            plt.title(stock_code + ' ' + model_desc + '(%d)' % np.sum(det_event))
            plt.grid()
            fig.savefig(data_dir + '/' + stock_code + "_DetEveNum_%d_" % np.sum(det_event) + file_random_str + '.jpg')
            plt.clf()
            plt.close()

            # save precision-recall result as figure
            fig = plt.figure()
            plt.plot(prec_ary, "o-")
            plt.plot(rec_ary, "r*-")
            plt.legend(["precision", "recall"])
            plt.ylim(0, 1)
            plt.ylabel('precision & recall')
            plt.xlabel('time (10 days)')
            plt.title(stock_code + ' ' + model_desc + '(%.3f,%.3f)' % (prec_mean, rec_mean))
            plt.grid()
            fig.savefig(data_dir + '/' + stock_code + "_PrecRec_%.3f_%.3f_" % (prec_mean, rec_mean) + file_random_str + '.jpg')
            plt.clf()
            plt.close()

            # save accuracy result as figure
            fig = plt.figure()
            plt.plot(acc_ary, "o-")
            plt.ylim(0, 1)
            plt.ylabel('accuracy')
            plt.xlabel('time (10 days)')
            plt.title(stock_code + ' ' + model_desc + '(%.3f)' % acc_mean)
            plt.grid()
            fig.savefig(data_dir + '/' + stock_code + "_Acc_%.3f_" % acc_mean + file_random_str + '.jpg')
            plt.clf()
            plt.close()

            '''
            # save auto-correlation result as figure
            fig = plt.figure()
            plt.plot(auto_corr, "*-")
            plt.ylim(-1, 1)
            plt.ylabel('auto-correlation')
            plt.xlabel('lag')
            plt.title(stock_code + ' ' + model_desc + ' (auto-correlation)')
            plt.grid()
            fig.savefig(data_dir + '/' + stock_code + '_adaboost_autocorr_' + file_random_str + '.jpg')
            plt.clf()
            plt.close()
            '''

        if len(prec_mean_ary) == 0:
            continue
        acc_result.append(np.asarray(prec_mean_ary))

    pdb.set_trace()
