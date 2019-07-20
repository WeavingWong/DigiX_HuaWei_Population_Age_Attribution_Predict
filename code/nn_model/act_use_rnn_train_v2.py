import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import pandas as pd
from models import RnnVersion2
import gc
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,Callback

user_app_actived = pd.read_csv('../../data/original_data/user_app_actived.csv',names=['uId', 'appId'])
usage_list = pd.read_csv('../../data/processed_data/usage_app_info.csv')   #重采样的usage_app
# usage_appId = pd.read_csv('../input/usage_appId.csv')     #使用表的app词典
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

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=None)


from sklearn.model_selection import train_test_split, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=False)
y_testb = np.zeros((x_test.shape[0],6))
y_valb = np.zeros((x_train.shape[0],6))


for i, (train_index, valid_index) in enumerate(kfold.split(app_list, np.argmax(y_train,axis=1))):
    X_train, X_val, Y_train, Y_val = app_list[train_index],app_list[valid_index], y_train[train_index], y_train[valid_index]
    filepath="weights_best5.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    model = RnnVersion2(max_len = 170,emb_size = 21099)
    model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(X_val, Y_val), verbose=1, callbacks=callbacks, 
             )
    model.load_weights(filepath)
    
    y_valb[valid_index] = model.predict(X_val, batch_size=128, verbose=1)
    y_testb +=  np.array(model.predict(app_test, batch_size=128, verbose=1))/5    
    
    dense1_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('concatenate').output)
    

y_valb = pd.DataFrame(y_valb,index=x_train.uId)
y_valb.to_csv('../../data/prob_file/act_use_rnn_train_v2.csv')
y_testb = pd.DataFrame(y_testb,index=x_test.uId)
y_testb.to_csv('../../data/prob_file/act_use_rnn_test_v2.csv') 

