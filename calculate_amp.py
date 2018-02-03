import os
import pickle
import numpy as np
from dataio import opdata

cache_data_dir = "cache_data_with_date"

def calculate_stock_ave_amp(stock_code, start_date, end_date, interval):
    cache_data_path = os.path.join(cache_data_dir, "%s_%s_%s" % (stock_code, start_date, end_date))

    if os.path.isfile(cache_data_path):
        f = open(cache_data_path, 'rb')
        stock_data = pickle.load(f)
        f.close()
    else:
        stock_data = opdata.get_day(stock_code, start_date, end_date)
        f = open(cache_data_path, 'wb')
        pickle.dump(stock_data, f)
        f.close()


    close_list = np.asarray(stock_data["close"].tolist())

    current_close = close_list[interval:]
    future_close = close_list[:-interval]

    percent = (np.asarray(future_close) / np.asarray(current_close) - 1) * 100

    return percent
