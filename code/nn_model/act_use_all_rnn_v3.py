import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import pandas as pd
from models import RnnVersion3 
import gc
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,Callback

from tqdm import tqdm_notebook 
user_app_actived = pd.read_csv('../../data/original_data/user_app_actived.csv',names=['uId', 'appId'])

usage_list = pd.read_csv('../../data/processed_data/usage_app_info.csv')   #重采样的usage_app
usage_appId = pd.read_csv('../../data/processed_data/usage_appId.csv')     #使用表的app词典
appId = pd.read_csv('../../data/processed_data/appId.csv')   #激活表的app词典

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
        result.append(np.sort(list(set(row['app_list_x']) | set(row['app_list_y']))))
    except:
        result.append(row['app_list_x'])
        
user_app_actived['app_list'] = result
user_app_actived.drop(['app_list_x','app_list_y'],axis=1,inplace =True)
del usage_list
gc.collect()
x_train = pd.read_csv('../../data/original_data/age_train.csv',names=['uId','age_group'],dtype={'uId':np.int32, 'age_group':np.int8})
x_test = pd.read_csv('../../data/original_data/age_test.csv',names=['uId'],dtype={'uId':np.int32})
x_train = pd.merge(x_train, user_app_actived, how='left', on='uId')
x_test = pd.merge(x_test, user_app_actived, how='left', on='uId')
y_train = x_train.age_group - 1
x_train = x_train.drop('age_group',axis=1)

del user_app_actived
gc.collect()
usage_appId = pd.read_csv('../../data/processed_data/usage_appId_top_num100000.csv')
usage_appId = usage_appId[-20000:]
usage_appId['id'] = np.arange(0,20000)
all_appid = list(set(appId.appId.tolist() + usage_appId.appId.tolist()))
app_dict = dict(zip(all_appid,np.arange(len(all_appid))))

app_list = [[app_dict[x] for x in apps if  x in app_dict] for apps in x_train.app_list]
app_test = [[app_dict[x] for x in apps if  x in app_dict] for apps in x_test.app_list]

from keras.preprocessing import sequence
app_list = sequence.pad_sequences(app_list, maxlen=170)
app_test = sequence.pad_sequences(app_test, maxlen=170)

x_train.drop('app_list',axis=1,inplace=True)
x_test.drop('app_list',axis=1,inplace=True)
gc.collect()

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

train_uId = x_train.uId.tolist()
test_uId = x_test.uId.tolist()

test.index = test.uId.tolist()
train.index = train.uId.tolist()
test = test.loc[test_uId,:]
train = train.loc[train_uId,:]

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
train = train.values
test = test.values
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=None)

from sklearn.model_selection import train_test_split, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=False)
y_testb = np.zeros((x_test.shape[0],6))
y_valb = np.zeros((x_train.shape[0],6))


for i, (train_index, valid_index) in enumerate(kfold.split(app_list, np.argmax(y_train,axis=1))):
    X_train1, X_val1, X_train2, X_val2,Y_train, Y_val = app_list[train_index],app_list[valid_index], train[train_index], train[valid_index], y_train[train_index], y_train[valid_index]
    filepath="weights_best5.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    model = RnnVersion3()
    model.fit([X_train1,X_train2], Y_train, batch_size=128, epochs=4, validation_data=([X_val1,X_val2], Y_val), verbose=1, callbacks=callbacks, 
             )
    model.load_weights(filepath)
    
    y_valb[valid_index] = model.predict([X_val1,X_val2], batch_size=128, verbose=1)
    y_testb +=  np.array(model.predict([app_test,test], batch_size=128, verbose=1))/5    
    
y_valb = pd.DataFrame(y_valb,index=train_uId)
y_valb.to_csv('../../data/prob_file/act_use_all_rnn_train_v2.csv')
y_testb = pd.DataFrame(y_testb,index=test_uId)
y_testb.to_csv('../../data/prob_file/act_use_all_rnn_test_v2.csv') 