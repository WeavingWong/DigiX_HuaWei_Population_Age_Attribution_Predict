
import os
import copy
import datetime
import warnings

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

import pandas as pd
import numpy as np
import math
from datetime import datetime

import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, PReLU,ReLU
from keras.layers import Bidirectional, SpatialDropout1D, CuDNNGRU,CuDNNLSTM, Conv1D,Conv2D,MaxPool2D,Reshape
from keras.layers import GlobalAvgPool1D, GlobalMaxPool1D, concatenate,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
from keras.engine import Layer
from keras.layers.core import Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,BatchNormalization
from keras.regularizers import l1,l2
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,Callback
import gc
from tqdm import tqdm_notebook   
class _Data_Preprocess:
    def __init__(self):
        self.int8_max = np.iinfo(np.int8).max
        self.int8_min = np.iinfo(np.int8).min
        self.int16_max = np.iinfo(np.int16).max
        self.int16_min = np.iinfo(np.int16).min
        self.int32_max = np.iinfo(np.int32).max
        self.int32_min = np.iinfo(np.int32).min
        self.int64_max = np.iinfo(np.int64).max
        self.int64_min = np.iinfo(np.int64).min
        self.float16_max = np.finfo(np.float16).max
        self.float16_min = np.finfo(np.float16).min
        self.float32_max = np.finfo(np.float32).max
        self.float32_min = np.finfo(np.float32).min
        self.float64_max = np.finfo(np.float64).max
        self.float64_min = np.finfo(np.float64).min
    '''
    function: _get_type(self,min_val, max_val, types)
       get the correct types that our columns can trans to
    '''
    def _get_type(self, min_val, max_val, types):
        if types == 'int':
            if max_val <= self.int8_max and min_val >= self.int8_min:
                return np.int8
            elif max_val <= self.int16_max <= max_val and min_val >= self.int16_min:
                return np.int16
            elif max_val <= self.int32_max and min_val >= self.int32_min:
                return np.int32
            return None
        elif types == 'float':
            if max_val <= self.float16_max and min_val >= self.float16_min:
                return np.float16
            if max_val <= self.float32_max and min_val >= self.float32_min:
                return np.float32
            if max_val <= self.float64_max and min_val >= self.float64_min:
                return np.float64
            return None
    '''
    function: _memory_process(self,df) 
       column data types trans, to save more memory
    '''
    def _memory_process(self, df):
        init_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('Original data occupies {} GB memory.'.format(init_memory))
        df_cols = df.columns          
        for col in tqdm_notebook(df_cols):
            try:
                if 'float' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'float')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
                elif 'int' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'int')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
            except:
                print(' Can not do any process for column, {}.'.format(col)) 
        afterprocess_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('After processing, the data occupies {} GB memory.'.format(afterprocess_memory))
        return df
memory_preprocess = _Data_Preprocess()

user_app_actived = pd.read_csv('../../data/original_data/user_app_actived.csv',names=['uId', 'appId'])
x_train = pd.read_csv('../../data/original_data/age_train.csv',names=['uId','age_group'],dtype={'uId':np.int32, 'age_group':np.int8})
x_test = pd.read_csv('../data/original_data/age_test.csv',names=['uId'],dtype={'uId':np.int32})
usage_list = pd.read_csv('../../data/processed_data/usage_app_info.csv')
usage_appId = pd.read_csv('../../data/processed_data/usage_appId.csv')

train = pd.read_csv('../../data/features/base_train.csv')
test = pd.read_csv('../../data/features/base_test.csv')
train=memory_preprocess._memory_process(train)
test=memory_preprocess._memory_process(test)
print(test.info())
gc.collect()

actived_features_all = pd.read_csv('../../data/features/actived_features_all.csv')
actived_features_all=memory_preprocess._memory_process(actived_features_all)
train = pd.merge(train, actived_features_all, how='left', on='uId').fillna(0)
test = pd.merge(test, actived_features_all, how='left', on='uId').fillna(0)
del actived_features_all
gc.collect()

act_use_rnn_hide_train=pd.read_csv('../../data/features/act_use_rnn_hide_train.csv')
act_use_rnn_hide_train=memory_preprocess._memory_process(act_use_rnn_hide_train)
act_use_rnn_hide_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train = pd.merge(train, act_use_rnn_hide_train, how='left', on='uId').fillna(0)
del act_use_rnn_hide_train

act_use_rnn_hide_test=pd.read_csv('../../data/features/act_use_rnn_hide_test.csv')
act_use_rnn_hide_test=memory_preprocess._memory_process(act_use_rnn_hide_test)
act_use_rnn_hide_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
test = pd.merge(test, act_use_rnn_hide_test, how='left', on='uId').fillna(0)
print(test.info())
del act_use_rnn_hide_test
gc.collect()

user_app_actived['app_list'] = user_app_actived.appId.str.split('#')
import ast
from tqdm import tqdm
usage_train = []
for idx in tqdm(usage_list.appId):
    usage_train.append(ast.literal_eval(idx))
usage_list['app_list'] = usage_train

user_app_actived.drop('appId',axis=1,inplace=True)
usage_list.drop('appId',axis=1,inplace=True)

user_app_actived = pd.merge(user_app_actived, usage_list, how='left', on='uId')
result = []
for index,row in tqdm(user_app_actived.iterrows()):
    try:
        result.append(row['app_list_x'] + row['app_list_y'])
    except:
        result.append(row['app_list_x'])
user_app_actived['app_list'] = result
user_app_actived.drop(['app_list_x','app_list_y'],axis=1,inplace =True)

x_train = pd.merge(x_train, user_app_actived, how='left', on='uId')
x_test = pd.merge(x_test, user_app_actived, how='left', on='uId')
y_train = x_train.age_group - 1
x_train = x_train.drop('age_group',axis=1)
del user_app_actived
del usage_list
del usage_train
gc.collect()

train_uId = x_train.uId.tolist()
test_uId = x_test.uId.tolist()

test.index = test.uId.tolist()
train.index = train.uId.tolist()
test = test.loc[test_uId,:]
train = train.loc[train_uId,:]
appId = pd.read_csv('../../data/processed_data/appId.csv')
usage_appId = pd.read_csv('../../data/processed_data/usage_appId_top_num100000.csv')
usage_appId = usage_appId[-10000:]
usage_appId['id'] = np.arange(0,10000)
all_appid = list(set(appId.appId.tolist() + usage_appId.appId.tolist()))
app_dict = dict(zip(all_appid,np.arange(len(all_appid))))

x_train = [[x for x in apps if  x in app_dict] for apps in x_train.app_list]
x_test = [[x for x in apps if  x in app_dict] for apps in x_test.app_list]
x_train = [" ".join(app) for app in x_train]
x_test = [" ".join(app) for app in x_test]

c_vec1 = CountVectorizer(lowercase=False,ngram_range=(1,1),dtype=np.int8)
c_vec1.fit(x_train + x_test)
x_train = c_vec1.transform(x_train).toarray()
x_test = c_vec1.transform(x_test).toarray()
gc.collect()

train.drop(['uId','age_group'],axis=1,inplace=True)
test.drop('uId',axis=1,inplace=True)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = MinMaxScaler()
scaler.fit(pd.concat([train,test],axis=0))
train = scaler.transform(train)
test = scaler.transform(test)

train = memory_preprocess._memory_process(pd.DataFrame(train))
test = memory_preprocess._memory_process(pd.DataFrame(test))
gc.collect()

x_train = np.hstack((x_train,train.values))
x_test = np.hstack((x_test,test.values))

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=None)

def mlp_v3():
    model = Sequential()
    model.add(Dense(1024, input_shape=(13,400,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))   
 #   model.add(BatchNormalization())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))    
#    model.add(BatchNormalization())
#
    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])
    return model

from sklearn.model_selection import train_test_split, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=False)
y_test = np.zeros((x_test.shape[0],6))
y_val = np.zeros((x_train.shape[0],6))
for i, (train_index, valid_index) in enumerate(kfold.split(x_train, np.argmax(y_train,axis=1))):
    X_train, X_val, Y_train, Y_val = x_train[train_index],x_train[valid_index], y_train[train_index], y_train[valid_index]
    filepath="weights_best2.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    model = mlp_v3()
    if i == 0:print(model.summary()) 
    model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(X_val, Y_val), verbose=1, callbacks=callbacks, 
             )
    model.load_weights(filepath)

    
    y_val[valid_index] = model.predict(X_val, batch_size=128, verbose=1)
    y_test +=  np.array(model.predict(x_test, batch_size=128, verbose=1))/5
    
y_val = pd.DataFrame(y_val,index=train_uId)
y_val.to_csv('../../data/prob_file/act_all_train_mlp.csv')
y_test = pd.DataFrame(y_test,index=test_uId)
y_test.to_csv('../../data/prob_file/act_all_test_mlp.csv')