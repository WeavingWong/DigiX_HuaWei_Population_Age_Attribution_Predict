import pandas as pd
import os
import numpy as np

userAll = pd.read_csv('../../data/original_data/user_app_usage.csv', names=['uId', 'appId', 'duration', 'times', 'use_date'], dtype={'uId':np.int32, 'appId':str, 'duration':np.float32, 
                                                                'times':np.float32, 'use_date':str})
app_info = pd.read_csv('../../data/processed_data/app_info.csv', dtype={'appId':str, 'category':str})
age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test = pd.read_csv('../../data/processed_data/age_test.csv',dtype={'uId':np.int32})
user_app_actived = pd.read_csv('../data/original_data/user_app_actived.csv',names=['uId', 'appId'])

## 分离使用表的训练集样本和测试集样本
age_train_usage = pd.merge(age_train,userAll[['uId']].drop_duplicates(),how='inner',on='uId')
age_test_usage = pd.merge(age_test,userAll[['uId']].drop_duplicates(),how='inner',on='uId')
age_train_usage.to_csv('../../data/processed_data/age_train_usage.csv', index=False, encoding='utf-8')
age_test_usage.to_csv('../../data/processed_data/age_test_usage.csv', index=False, encoding='utf-8')
age_test_na = pd.concat([age_test,age_test_usage])
age_test_na = age_test_na.drop_duplicates(keep=False)
age_test_na.to_csv('../../data/processed_data/age_test_na.csv', index=False, encoding='utf-8')

##通过时长采样使用表的app，形成app的使用列表
userSub = userAll.groupby(['uId', 'appId'])['duration'].sum().reset_index()
# userSub.describe()
def divide(x):
    if x < 185:
        return 1
    elif x >= 185 and x < 1000:
        return 2
    elif x >= 1000 and x < 5839:
        return 3
    elif x >= 5839:
        return 4
    
userSub['copy_nums'] = userSub['duration'].apply(lambda x: divide(x))
userSub.drop(['duration'], axis=1, inplace=True)
userSub_2 = userSub[(userSub.copy_nums == 2)].copy()
userSub_3 = userSub[(userSub.copy_nums == 3)].copy()
userSub_4 = userSub[(userSub.copy_nums == 4)].copy()
userSub = pd.concat([userSub, userSub_2])
del userSub_2
userSub = pd.concat([userSub, userSub_3])
userSub = pd.concat([userSub, userSub_3])
del userSub_3
userSub = pd.concat([userSub, userSub_4])
userSub = pd.concat([userSub, userSub_4])
userSub = pd.concat([userSub, userSub_4])
del userSub_4
userSub.drop(['copy_nums'], axis=1, inplace=True)
userSub_list = userSub.groupby(['uId'])['appId'].apply(list).reset_index()
userSub_list.to_csv('../../data/processed_data/usage_app_info.csv', index=False, encoding='utf-8')


##把激活表的app单独抽取出来
user_app_actived = user_app_actived.appId.str.split('#')
user_app_actived = list(set(user_app_actived.reshape(-1,)))
activated_appid = pd.DataFame({'appId':user_app_actived, 'num':np.arange(len(user_app_actived))})
activated_appid.to_csv('../../data/processed_data/appId.csv', index=False, encoding='utf-8')


##选取使用表TOPN的app
appId = userAll[['appId']].drop_duplicates().reset_index()
appId['app_num'] = range(len(appId))
appId.drop(['index'], axis=1, inplace=True)
appId.to_csv('../../data/processed_data/usage_appId.csv', index=False, encoding='utf-8')

# 求usage表中出现的app的次数
userSub = userAll.groupby('appId')['use_date'].count().reset_index()
userSub.rename(columns={'use_date':'app_nums'}, inplace=True)
userSub = userSub.sort_values(by=('app_nums'))
userSub.tail(100000).to_csv('../../data/processed_data/usage_appId_top_num100000.csv', index=False, encoding='utf-8')
userSub.tail(200000).to_csv('../../data/processed_data/usage_appId_top_num200000.csv', index=False, encoding='utf-8')