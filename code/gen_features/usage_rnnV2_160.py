import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import keras
import keras.backend as K
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,BatchNormalization
from keras.regularizers import l1,l2
from keras.preprocessing.text import Tokenizer

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,Callback
from keras.models import Model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import gc
from models import CapsuleNet_v2

usage_list = pd.read_csv('../../data/processed_data/usage_app_info.csv')
x_train = pd.read_csv('../../data/original_data/age_train.csv',names=['uId','age_group'],dtype={'uId':np.int32, 'age_group':np.int8})
x_train = pd.merge(x_train, usage_list, how='inner', on='uId')
y_train = x_train.age_group - 1
x_test = pd.read_csv('../../data/original_data/age_test.csv',names=['uId'],dtype={'uId':np.int32})
x_test = pd.merge(x_test, usage_list, how='inner', on='uId')
x_train = x_train.drop('age_group',axis=1)
gc.collect()
usage_appId = pd.read_csv('../../data/processed_data/usage_appId_top_num100000.csv')
usage_appId = usage_appId[-20000:]
usage_appId['id'] = np.arange(0,20000)
del usage_list

gc.collect()
import ast
from tqdm import tqdm
usage_train = []
for idx in tqdm(x_train.appId):
    usage_train.append(ast.literal_eval(idx))
    
usage_test = []
for idx in tqdm(x_test.appId):
    usage_test.append(ast.literal_eval(idx))
    
usage_dict = dict(zip(usage_appId.appId,usage_appId.id))
usage_train = [[usage_dict(x) for x in apps if x in usage_dict] for apps in usage_train]
usage_test = [[usage_dict(x) for x in apps if x in usage_dict] for apps in usage_test]

from keras.preprocessing import sequence
usage_train = sequence.pad_sequences(usage_train, maxlen=200)
usage_test = sequence.pad_sequences(usage_test, maxlen=200)


from sklearn.model_selection import train_test_split, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=False)
y_test = np.zeros((usage_test.shape[0],6))
y_val = np.zeros((usage_train.shape[0],6))
for i, (train_index, valid_index) in enumerate(kfold.split(usage_train, np.argmax(y_train,axis=1))):
    X_train, X_val, Y_train, Y_val = usage_train[train_index],usage_train[valid_index], y_train[train_index], y_train[valid_index]
    filepath="weights_best2.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    model = CapsuleNet_v2()
    if i == 0:print(model.summary()) 
    model.fit(X_train, Y_train, batch_size=64, epochs=5, validation_data=(X_val, Y_val), verbose=1, callbacks=callbacks, 
             )
    model.load_weights(filepath)
#     dense1_layer_model = Model(inputs=model.input,
#                                      outputs=model.get_layer('fc').output)
    
    y_val[valid_index] = model.predict(X_val, batch_size=64, verbose=1)
    y_test +=  np.array(model.predict(usage_test, batch_size=64, verbose=1))/5


y_val = pd.DataFrame(y_val)
y_val['uId'] = x_train.uId.tolist()
y_val.to_csv('../../data/features/usage_rnnV2_train_160.csv')
y_test = pd.DataFrame(y_test)
y_test['uId'] = x_test.uId.tolist()
y_test.to_csv('../../data/features/usage_rnnV2_test_160.csv')
