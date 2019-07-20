import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import keras
import keras.backend as K
from keras.datasets import reuters
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation,BatchNormalization
from keras.regularizers import l1,l2
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,Callback
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import gc
user_app_actived = pd.read_csv('../../data/prob_file/user_app_actived.csv',names=['uId', 'appId'])
train = pd.read_csv('../../data/prob_file/input/age_train.csv',names=['uId','age_group'],dtype={'uId':np.int32, 'age_group':np.int8})
train = pd.merge(train, user_app_actived, how='left', on='uId')
y_train = train.age_group - 1
test = pd.read_csv('../../data/prob_file/age_test.csv',names=['uId'],dtype={'uId':np.int32})
test = pd.merge(test, user_app_actived, how='left', on='uId')

x_train = train.appId.str.split('#')
x_test = test.appId.str.split('#')

x_train = [" ".join(app) for app in x_train]
x_test = [" ".join(app) for app in x_test]
y_train = to_categorical(y_train, num_classes=None)
del user_app_actived
gc.collect()
c_vec = CountVectorizer(lowercase=False,ngram_range=(1,1),dtype=np.int8,min_df=0.0001)
c_vec.fit(x_train + x_test)
x_train = c_vec.transform(x_train).toarray()
x_test = c_vec.transform(x_test).toarray()

def mlp_v2():
    model = Sequential()
    model.add(Dense(1024, input_shape=(9399,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))   
    model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    
    model.add(BatchNormalization())
    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    return model

from sklearn.model_selection import train_test_split, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=False)
y_test = np.zeros((x_test.shape[0],6))
y_val = np.zeros((x_train.shape[0],6))
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, np.argmax(y_train,axis=1))):
    X_train, X_val, Y_train, Y_val = x_train[train_index],x_train[valid_index], y_train[train_index], y_train[valid_index]
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    model = mlp_v2()
    if i == 0:print(model.summary()) 
    model.fit(X_train, Y_train, batch_size=64, epochs=5, validation_data=(X_val, Y_val), verbose=1, callbacks=callbacks, 
             )
    model.load_weights(filepath)

    y_val[valid_index] = model.predict(X_val, batch_size=64, verbose=1)
    y_test +=  np.array(model.predict(x_test, batch_size=64, verbose=1))/5

y_val = pd.DataFrame(y_val,index=x_train.uId.tolist())
y_val.to_csv('../../data/prob_file/mlp_output_train_cnt.csv')
y_test = pd.DataFrame(y_test,index=x_test.uId.tolist())
y_test.to_csv('../../data/prob_file/mlp_output_test_cnt.csv')
