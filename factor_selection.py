import numpy as np
from scipy.stats.stats import pearsonr
import os
import scipy.stats as ss

from opdata import opdata
import pickle

'''
The factor selection includes 4 steps, as follow:
1. rank IC
2. IC decay profile
3. Factiles
4. Pure factor returns
'''

factor_code = "momentum"
start_year = 2010
start_month = 1
end_year = 2014
end_month = 12
stock_pool = "hs300"


# load of get data from interface
filename = 'factor_selection_data'

if os.path.isfile(filename):
    f = open(filename, 'rb')
    month_data_ary = pickle.load(f)
else:
    month_data_ary = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            date_str = "%d-%02d" % (year, month)
            print(date_str)
            cur_month_data = opdata.get_month(date_str)
            month_data_ary.append(cur_month_data)

    f = open(filename, 'wb')
    pickle.dump(month_data_ary, f)


rank_ic_ary = []

for month_data_idx in range(len(month_data_ary) - 1):

    month_data = month_data_ary[month_data_idx]
    next_month_data = month_data_ary[month_data_idx + 1]

    code_list = month_data['code'].tolist()
    factor_list = month_data['momentum'].tolist()
    close_list = month_data['close'].tolist()

    next_code_list = next_month_data['code'].tolist()
    next_close_list = next_month_data['close'].tolist()

    # calculate one month return for each stock in current month list
    return_list = []
    for idx, code in enumerate(code_list):
        current_close = close_list[idx]
        if code not in next_code_list:
            # this stock is not included in the next month data
            return_list.append(None)
            continue
        next_idx = next_code_list.index(code)
        next_close = next_close_list[next_idx]
        return_list.append(next_close / current_close - 1)

    # skip the first code, which should be the index code
    factor_ary = np.asarray(factor_list[1:])
    return_ary = np.asarray(return_list[1:])

    # filter the None value
    factor_filter = factor_ary != None
    return_filter = return_ary != None

    data_filter = factor_filter & return_filter

    factor_ary = factor_ary[data_filter]
    return_ary = return_ary[data_filter]

    factor_rank = np.argsort(factor_ary)
    return_rank = np.argsort(return_ary)

    rank_ic, _ = pearsonr(return_rank, factor_rank)

    import pdb
    pdb.set_trace()

    rank_ic_ary.append(rank_ic)


import pdb
pdb.set_trace()
