"""Train and test LSTM classifier"""
import numpy as np
import os
import random
import csv
import collections
import math
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import model_from_json, load_model
from sklearn.metrics import precision_score, recall_score, classification_report,accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, auc
from datetime import datetime
import sys
import tldextract
import bisect
import pickle
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

def process(domain):
    ext = tldextract.extract(domain)
    if ((ext[1] == '') or (ext[2] == '')):
        return np.nan
    else:
        return ext[1]+'.'+ext[2]
    
def get_data(name): 
    df = pd.read_csv(name, usecols=['domain','dga'])
    df = df.dropna()
    df['domain'] = df['domain'].str.lower()
    df['domain'] = df.domain.apply(process)
    df = df.dropna()
    df = df.drop_duplicates()
    print('Loaded: ', len(df))
    print(df.head())
    return df

def build_binary_model(max_features, maxlen):
    """Build LSTM model for two-class classification"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop')

    return model

def build_multiclass_model(max_features, maxlen):
    """Build multiclass LSTM model for multiclass classification"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(38))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop')

    return model

def create_class_weight(labels_dict,mu):
    """Create weight based on the number of domain name in the dataset"""
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.pow(total/float(labels_dict[key]),mu)
        class_weight[key] = score

    return class_weight

def classifaction_report_csv(report,precision,recall,f1_score,fold):
    """Generate the report to data processing"""
    with open('classification_report_cost.csv', 'a') as f:
        report_data = []
        lines = report.split('\n')
        row = {}
        row['class'] =  "fold %u" % (fold+1)
        report_data.append(row)
        for line in lines[2:44]:
            row = {}
            line = " ".join(line.split())
            row_data = line.split(' ')
            if(len(row_data)>2):
                if(row_data[0]!='avg'):
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['support'] = row_data[4]
                    report_data.append(row)
                else:
                    row['class'] = row_data[0]+row_data[1]+row_data[2]
                    row['precision'] = float(row_data[3])
                    row['recall'] = float(row_data[4])
                    row['f1_score'] = float(row_data[5])
                    row['support'] = row_data[6]
                    report_data.append(row)
        row = {}
        row['class'] = 'macro'
        row['precision'] = float(precision)
        row['recall'] = float(recall)
        row['f1_score'] = float(f1_score)
        row['support'] = 0
        report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(f, index = False)
		
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

def sub_auc(fpr, tpr, percentage=1):
    assert(len(fpr) == len(tpr))

    endValue = round(percentage / 100.0, 2)
    index = 0
    for i, (f, t) in enumerate(zip(fpr, tpr)):
        if f > endValue:
            index = i
            break

    if index == 0:
        return 0.0

    sub_fpr = fpr[:index]
    sub_tpr = tpr[:index]

    return auc(sub_fpr, sub_tpr)

def get_fpr_tpr_for_thresh(fpr, tpr, thr, thresh):
    p = bisect.bisect_left(fpr, thresh)
    fpr = fpr.copy()
    fpr[p] = thresh
    return fpr[: p + 1], tpr[: p + 1], thr[: p + 1]

def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in list(zip(ft[: -1], ft[1: ])):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

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
    #plt.savefig('roc.png', dpi=150)
    plt.show()

def new_statistics(y_true, y_pred, fpr_value):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    #_, _, i = scan_roc(fpr, tpr, thr, fpr_value=fpr_value)
    #if i > 0:
    #    i -= 1
    fpr, tpr, thr = get_fpr_tpr_for_thresh(fpr, tpr, thr, fpr_value)
    _, _, i = scan_roc(fpr, tpr, thr, fpr_value=fpr_value)
    if i > 0:
        i -= 1
    #stats = {'TPR': tpr[i], 'FPR': fpr[i], 'AUC@1%': sub_auc(fpr, tpr, 1)}
    stats = {'TPR': tpr[i], 'FPR': fpr[i], 'THR': thr[i], 'AUC@1%': auc_from_fpr_tpr(fpr, tpr, False)}

    return stats

def new_statistic(y_true, y_pred, fpr_value):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    _, _, i = scan_roc(fpr, tpr, thr, fpr_value=fpr_value)
    if i > 0:
        i -= 1
    stats = {'TPR': tpr[i], 'FPR': fpr[i], 'THR':thr[i], 'AUC@1%': sub_auc(fpr, tpr, 1)}
    return stats
	
def run(max_epoch=100, nfolds=1, batch_size=128):
    #Begin preprocessing stage
    #Read data to process
    train_data = get_data('../datasets/ab_train_80.csv')
    val_data = get_data('../datasets/ab_val_10.csv')
    test_data = get_data('../datasets/ab_test_10.csv')

    # Extract data and labels
    y_train = train_data['dga'].values
    X_train = train_data['domain'].tolist()
    
    y_val = val_data['dga'].values
    X_val = val_data['domain'].tolist()
    
    y_test = test_data['dga'].values
    X_test = test_data['domain'].tolist()

    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(sorted(set(''.join(X_train+X_val+X_test))))}
    max_features = len(valid_chars) + 1
    
    maxlen = np.max([len(x) for x in X_train+X_val+X_test])
    print('valid_chars:', valid_chars)
    print('maxlen: ', maxlen)
    print('max_features: ', max_features)

    # Convert characters to int and pad
    X_train = [[valid_chars[y] for y in x] for x in X_train]
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    
    X_val = [[valid_chars[y] for y in x] for x in X_val]
    X_val = sequence.pad_sequences(X_val, maxlen=maxlen)
    
    X_test = [[valid_chars[y] for y in x] for x in X_test]
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    
    #End preprocessing stage

    #Begin two-class classification stage
    #Build the model for two-class classification stage
    model = build_binary_model(max_features, maxlen)
    
    #Create weight for two-class classification stage
    labels_dict=collections.Counter(y_train)
    class_weight = create_class_weight(labels_dict,0.1)
    
    print("Training the model for two-class classification stage...")
    model.fit(X, y_binary, batch_size=batch_size, epochs=max_epoch, class_weight=class_weight)
    
    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='model_binary_best.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epoch, callbacks=callbacks,
                  class_weight=class_weight, validation_data=(X_val, y_val))
    
    print('Loading model')
    best_model = load_model('model_binary_best.h5')
    
    print('Record metrics against test data')
    y_result = best_model.predict_proba(X_test)

    return y_test, y_result

# Main starts here	
y_test, y_result = run()

d_fpr = 0.01
print('AUC: ', roc_auc_score(y_test, y_result), roc_auc_score(y_test, y_result, max_fpr=d_fpr))
print(new_statistics(y_test, y_result, fpr_value=d_fpr))

fpr, tpr, thr = roc_curve(y_test, y_result)
plot_roc(roc_auc_score(y_test, y_result, max_fpr=d_fpr), fpr, tpr, thr, 0.01, 0.01)