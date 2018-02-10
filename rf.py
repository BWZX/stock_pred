# coding: utf-8
import os
import argparse
import pickle
import pdb
import numpy as np
import uuid
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from cfgs.config import cfg


parser = argparse.ArgumentParser()
parser.add_argument('--est_num', help='number of decision trees', type=int, default=200)

args = parser.parse_args()

prefix = '52feat_900_10'

pred_interval = 3
data_dir = "%s/%s_%d" % (cfg.dataset_dir, prefix, pred_interval)
save_dir = "%s/%s_%d" % (cfg.result_dir, prefix, pred_interval)

est_num = args.est_num

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

def rf_model(norm_train_set_x, train_set_y, norm_test_set_x, test_set_y):
    clf = RandomForestClassifier(n_estimators=est_num)
    clf.fit(norm_train_set_x, train_set_y)
    test_set_y_pred = clf.predict(norm_test_set_x)
    corr = np.sum((test_set_y_pred == test_set_y).astype(int))
    acc = corr / len(test_set_y)
    return acc

data_files = os.listdir(data_dir)
start_idxes = [int(e.split('_')[0]) for e in data_files]
start_idxes = list(set(start_idxes))
start_idxes.sort()

acc_ary = []

for start_idx in start_idxes:

    # prepare training and validation set
    train_x_name =  os.path.join(data_dir, "%d_train_x" % start_idx)
    train_y_name =  os.path.join(data_dir, "%d_train_y" % start_idx)
    test_x_name =  os.path.join(data_dir, "%d_test_x" % start_idx)
    test_y_name =  os.path.join(data_dir, "%d_test_y" % start_idx)
    
    train_x_f = open(train_x_name, 'rb')
    train_y_f = open(train_y_name, 'rb')
    test_x_f = open(test_x_name, 'rb')
    test_y_f = open(test_y_name, 'rb')
    
    train_x = pickle.load(train_x_f)
    train_y = pickle.load(train_y_f)
    test_x = pickle.load(test_x_f)
    test_y = pickle.load(test_y_f)

    train_y = (train_y > 0).astype('int')
    test_y = (test_y > 0).astype('int')

    acc = rf_model(train_x, train_y, test_x, test_y)

    print("%d: %.3f" % (start_idx, acc))

    acc_ary.append(acc)

acc_mean = np.mean(acc_ary)

run_id = str(uuid.uuid4())
short_id = run_id.split('-')[-1]

acc_str = ','.join(["%.3f" % e for e in acc_ary])
f = open("%s/%s_rf%d_%.3f.txt" % (save_dir, short_id, est_num, acc_mean), 'w')
f.write(acc_str)
f.close()

fig = plt.figure()
plt.plot(acc_ary, "o-")
plt.ylim(0, 1)
plt.ylabel('accuracy')
plt.xlabel('time (10 days)')
plt.title('%dday_%.3f' % (pred_interval, acc_mean))
plt.grid()
fig.savefig("%s/%s_rf%d_%.3f.jpg" % (save_dir, short_id, est_num, acc_mean))
plt.clf()
plt.close()
    
print("mean acc: %.3f" % acc_mean)
