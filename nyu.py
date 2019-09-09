import numpy as np
import pandas as pd
import os
import pickle
import itertools
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Flatten, Dense, Dropout
from keras.layers import ThresholdedReLU, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.layers import Input
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from sklearn.metrics import classification_report,accuracy_score

def preprocess_data(dataset, x_column, y_column):
    tld_list = load_tld()
    x = sequence.pad_sequences(
        dataset[x_column].map(lambda s: [ord(d) for d in s.lower()] if len(s) <= 75
                              else [ord(d)
                                    for d in find_sld(s, tld_list)[:(74-len(find_tld(s, tld_list)))]
                                    + '.' + find_tld(s, tld_list)]),
        maxlen=75
    )
    y = dataset[y_column].values
    return x, y
	
def build_model():
    model = Sequential()
    model.add(Embedding(128, 128, input_length = 75))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', strides=1))
    model.add(ThresholdedReLU(1e-6))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, padding='same', strides=1))
    model.add(ThresholdedReLU(1e-6))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(ThresholdedReLU(1e-6))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    #model.summary()
    return model
	
def train_model(train, val, model_dir, callbacks):
    
    X_train, y_train = preprocess_data(train, 'domain','dga')
    X_val, y_val = preprocess_data(val, 'domain', 'dga')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_train = build_model()

    history = model_train.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size = batch_size,
        verbose = 1,
        epochs = epochs,
        callbacks = callbacks,
        shuffle = "batch")
    
    model_json = model_train.to_json()
    with open(model_dir + "/nyu-model.json", "w") as json_file:
        json_file.write(model_json)

    return history
	
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, 
						  fname='cm.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Main starts here
train = pd.read_csv('../datasets/ab_train_80.csv', usecols=['domain', 'dga'])
train = train.dropna(how='any',axis=0)

val = pd.read_csv('../datasets/ab_val_10.csv', usecols=['domain', 'dga'])
val = val.dropna(how='any',axis=0)

test = pd.read_csv('../datasets/ab_test_10.csv', usecols=['domain', 'dga'])
test = test.dropna(how='any',axis=0)

batch_size = 2048
epochs = 100
model_dir = '../models'
model_file = model_dir + '/model_checkpoint_weights.{epoch:02d}.hdf5'
cbModelCheckpoint = ModelCheckpoint(
    model_file,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    period=1)

cbEarlyStop = EarlyStopping(patience=10)
csv_logger = CSVLogger('../models/log.csv', append=True, separator=';')
callbacks = [cbModelCheckpoint, cbEarlyStop, csv_logger]

history = train_model(train, val, model_dir, [cbModelCheckpoint, cbEarlyStop, csv_logger])

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('acc-curve.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('loss-curve.png')

X_test, y_test = preprocess_data(test, 'domain', 'dga')
json_file = open(model_dir + "/nyu-model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("../models/model_checkpoint_weights.02.hdf5")
y_pred = loaded_model.predict(X_test)
y_pred = (y_pred > 0.5)

print ("\nClassification Report\n")
print (classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
class_names = ['benign', 'DGA']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.show()

