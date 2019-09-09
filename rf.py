import numpy as np
import pandas as pd
import pickle
import itertools
import collections
import math
import bisect
import operator
from datetime import datetime

from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_feature_importance(clf, X_train, features, feat_id):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.savefig(feat_id+'.png')
    plt.close()

def create_class_weight(labels_dict,mu):
    """Create weight based on the number of domain name in the dataset"""
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.pow(total/float(labels_dict[key]),mu)
        class_weight[key] = score

    return class_weight

def scan_roc(fpr, tpr, thresholds, fpr_value=0.01, interval=10):
    assert(len(fpr) == len(tpr))

    half_interval = interval // 2

    up_bound = len(fpr) - 1
    # t will be the first occurence greater than the desired FPR rate
    t = 0
    end = len(fpr)
    while t < end:
        if fpr[t] > fpr_value:
            break
        t = t + 1

    index_from = t - half_interval if t > half_interval else 0
    index_to = t + half_interval if t <= up_bound else up_bound

    return index_from, index_to, t

def sub_auc(fpr, tpr, percentage= 1):
    assert(len(fpr) == len(tpr))

    endValue = round(percentage / 100.0, 2)
    index = 0
    for i, (f, t) in enumerate(zip(fpr, tpr)):
        if f > endValue:
            index = i
            break

    if index == 0:
        return 0.0

    sub_fpr = fpr[:index+1]
    sub_tpr = tpr[:index+1]

    return auc(sub_fpr, sub_tpr)

def plot_roc(auc, fpr, tpr, thresh, x_max=0.01, fpr_value=1e-4, xy=None, png_name=None):
    # draw figure
    fig, ax = plt.subplots(figsize=(20, 12))
    metric_binding = []

    lw = 1

    p = ax.plot(fpr, tpr, lw=lw, label='AUC-0.1% = {:.7f}'.format(auc))

    _, _, index = scan_roc(fpr, tpr, thresh, fpr_value=fpr_value)
    if index > 0:
        index -= 1
    desired_tpr = tpr[index]
    desired_fpr = fpr[index]
    desired_thr = thresh[index]
    metric_binding.append((desired_fpr, desired_tpr, desired_thr, p[0].get_color()))

    sorted_metric_binding = sorted(metric_binding, key=lambda t: t[1], reverse=True)
    for index, m in enumerate(sorted_metric_binding):
        if xy is None:
            x_, y_ = x_max * 0.15, sorted_metric_binding[0][1] * 0.9 - index * 0.05
        else:
            x_, y_ = xy[0], xy[1] * 0.9 - index * 0.05
        ax.annotate('TPR: {:.7f} (FPR={:.6f}, threshold={:.7f})'.format(m[1], m[0], m[2]),
                    xy=(x_, y_), xytext=(10, 0),
                    textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', alpha=0.5), fontsize=14)

    # vertical line at x = 0.0001
    ax.plot([fpr_value, fpr_value], [0, 1], linestyle='--', lw=1, color='red')
    # random guess line
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    x_tick_step = x_max * 0.1
    x_max = x_max * 1.05
    ax.set_ylim([0.0, 1.05])
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.set_xlim([-0.00003, x_max])
    ax.set_xticks(np.arange(0, x_max, x_tick_step))
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC curves')
    ax.grid(True, which='both', color='0.8', linestyle='--')
    plt.legend(loc="lower right", prop={'size': 14})
    plt.savefig('auc_1.png', dpi=150)
    #plt.show()

def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in list(zip(ft[: -1], ft[1: ])):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def new_statistic(y_true, y_pred, fpr_value):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    #_, _, i = scan_roc(fpr, tpr, thr, fpr_value=fpr_value)
    #if i > 0:
    #    i -= 1
    fpr, tpr = get_fpr_tpr_for_thresh(fpr, tpr, fpr_value)
    _, _, i = scan_roc(fpr, tpr, thr, fpr_value=fpr_value)
    if i > 0:
        i -= 1

    #stats = {'TPR': tpr[i], 'FPR': fpr[i], 'AUC@1%': sub_auc(fpr, tpr, 1)}
    stats = {'TPR': tpr[i], 'FPR': fpr[i], 'THR': thr[i], 'AUC@1%': auc_from_fpr_tpr(fpr, tpr, True)}

    return stats

def old_statistic(y_true, y_pred, fpr_value):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    _, _, i = scan_roc(fpr, tpr, thr, fpr_value=fpr_value)
    if i > 0:
        i -= 1
    #stats = {'TPR': tpr[i], 'FPR': fpr[i], 'THR':thr[i], 'AUC@1%': sub_auc(fpr, tpr, 1)}
    stats = {'TPR': tpr[i], 'FPR': fpr[i], 'THR':thr[i], 'AUC@1%': sub_auc(fpr, tpr, 1)}
    return stats

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    p = bisect.bisect_left(fpr, thresh)
    fpr = fpr.copy()
    fpr[p] = thresh
    return fpr[: p + 1], tpr[: p + 1]


# Main starts here
file_name = '../datasets/ab_train_features.csv'

features = ['domain_len', 'sld_len', 'tld_len', 'ent', 'gni', 'hex', 'cer', 'vow', 'con', 'lng_con_seq', '2gram_med',
            'sym', 'cons_con_ratio', 'cons_dig_ratio', 'rep_char_ratio', '3gram_med', '3gram_cmed', '2gram_cmed',
            'dig', 'flag_dig', 'uni_domain', 'uni_sld', 'tokens_sld', 'digits_sld', 'tld_hash', 'flag_dga',
            'domain', 'label']

print('Features: ', len(features))


train = pd.read_csv(file_name)
train = train.dropna()
print('Loaded Train: ', len(train))

#features = train.columns.values
train_features = [feature for feature in features if feature not in ['domain', 'label']]
print(len(train_features))

X_train = train[train_features].values
y_train = train['label'].values

print('Training binary classifier with RF', datetime.now())
labels_dict=collections.Counter(y_train)
class_weight = create_class_weight(labels_dict, 0.1)

b_clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0, max_features=20, class_weight=class_weight)
b_clf.fit(X_train, y_train)

dump_file = '../models/b_rf.pkl'
joblib.dump(b_clf, dump_file)

file_name = '../datasets/ab_val_features.csv'
val = pd.read_csv(file_name)
val = val.dropna()
print('Loaded val: ', len(val))

X_val = val[train_features].values
y_val = val['label'].values

print('Loading binary RF clasifier')
b_clf = joblib.load('../models/b_rf.pkl')

print('Plotting ROC curve')
y_val_pred = clf.predict_proba(X_val)
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
auc_score = auc(fpr, tpr)

plt.figure()
plt.title('ROC for B-RF')
plt.plot(fpr, tpr, 'b', label='AUC=%0.4f' % auc_score)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('b_rf.png')
plt.close()

print('Plotting feature importance')
plot_feature_importance(b_clf, X_train, train_features, 'binary')

file_name = '../datasets/ab_test_features.csv'
test = pd.read_csv(file_name)
test = test.dropna()
print('Loaded Test: ', len(test))

X_test = test[train_features].values
y_test = test['label'].values

y_pred = b_clf.predict_proba(X_test)[:,1]

desired_fpr = 0.01
print('AUC: ', roc_auc_score(y_test, y_pred), roc_auc_score(y_test, y_pred, max_fpr=desired_fpr))
print('\nNew_stats with fpr 0.01: ', new_statistic(y_test, y_pred, fpr_value=desired_fpr))
print('Done!', datetime.now())
