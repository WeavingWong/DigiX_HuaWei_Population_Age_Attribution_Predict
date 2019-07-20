import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import random

user_basic_info = pd.read_csv('../../data/original_data/user_basic_info.csv',
                              names=['uId', 'gender', 'city', 'prodName','ramCapacity', 'ramLeftRation', 
                                     'romCapacity', 'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os'])
user_behavior_info = pd.read_csv('../../data/original_data/user_behavior_info.csv', 
                            names=['uId', 'bootTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes',
                                   'EFuncTimes', 'FFuncTimes', 'FFuncSum'])
user_app_actived = pd.read_csv('../../data/original_data/user_app_actived.csv',names=['uId', 'appId'])
app_info = pd.read_csv('../../data/original_data/app_info.csv',names=['appId', 'category'])


# 解析激活表
def user_app():
    for i in range(len(user_app_actived)):
        if i == 0:
            user_app = user_app_actived.ix[i:i+1]
            print(user_app)
            user_app = user_app.drop('appId', axis=1).join(user_app['appId'].str.split('#', expand=True).stack().reset_index(level=1, drop=True).rename('appId'))
            user_app.to_csv('../../data/processsed_data/user_app.csv', index=False, encoding='utf-8')
            print(i)
        else:
            user_app = user_app_actived.ix[i:i + 1]
            print(user_app)
            user_app = user_app.drop('appId', axis=1).join(user_app['appId'].str.split('#', expand=True).stack().reset_index(level=1, drop=True).rename('appId'))
            user_app.to_csv('../../data/processsed_data/user_app.csv', mode='a', index=False, encoding='utf-8',header=0)
            print(i)


# 将category顺序编码
enc=preprocessing.OrdinalEncoder()
enc.fit(app_info[['category']])
app_info['category']=enc.transform(app_info[['category']])
app_info['category'] = app_info['category'].astype()
app_info.to_csv('../../data/processsed_data/app_info.csv', index=False, encoding='utf-8')


#  user_behavior_info中的使用次数为负值的数据使用0填充
user_behavior_info.loc[user_behavior_info.AFuncTimes<0,'AFuncTimes']=0
user_behavior_info.loc[user_behavior_info.BFuncTimes<0,'BFuncTimes']=0
user_behavior_info.loc[user_behavior_info.CFuncTimes<0,'CFuncTimes']=0
user_behavior_info.loc[user_behavior_info.DFuncTimes<0,'DFuncTimes']=0
user_behavior_info.loc[user_behavior_info.EFuncTimes<0,'EFuncTimes']=0
user_behavior_info.loc[user_behavior_info.FFuncTimes<0,'FFuncTimes']=0


# user_basic_info 缺失值填充
user_basic_info.fontSize.fillna(user_basic_info.fontSize.mode()[0], inplace=True)
user_basic_info.ramLeftRation.fillna(user_basic_info.ramLeftRation.mode()[0], inplace=True)
user_basic_info.romLeftRation.fillna(user_basic_info.romLeftRation.mode()[0], inplace=True)
user_basic_info.ct.fillna(user_basic_info.ct.mode()[0], inplace=True)
user_basic_info.ramCapacity.fillna(user_basic_info.ramCapacity.mode()[0], inplace=True)
user_basic_info.romCapacity.fillna(user_basic_info.romCapacity.mode()[0], inplace=True)
user_basic_info.city.fillna(user_basic_info.city.mode()[0], inplace=True)
user_basic_info.os.fillna(user_basic_info.os.mode()[0], inplace=True)

user_basic_info.to_csv('../../data/processed_data/user_basic_info.csv', index=False, encoding='utf-8')
user_behavior_info.to_csv('../../data/processed_data/user_behavior_info.csv', index=False, encoding='utf-8')