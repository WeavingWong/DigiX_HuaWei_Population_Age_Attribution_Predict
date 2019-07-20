import pandas as pd
import os
import numpy as np
import gc
import copy
import datetime
import warnings
from tqdm import tqdm
from scipy import sparse
from numpy import array
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

###############################################
#########数据加载
###############################################
user_app = pd.read_csv('../../data/processed_data/user_app.csv', dtype={'uId':np.int32, 'appId':str})
app_info = pd.read_csv('../../data/processed_data/app_info.csv', dtype={'appId':str, 'category':int})


###############################################
########## 压缩函数
###############################################
# 对数据进行类型压缩，节约内存
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

# 用法：
# baseSet=memory_preprocess._memory_process(baseSet)


###############################################
########## 统计用户安装app的数量和占比
###############################################
app_counts = user_app[['appId']].drop_duplicates().count()
userSub = user_app.groupby('uId')['appId'].nunique().reset_index().rename(columns={'appId': 'user_app_active_counts'})
userSub['user_app_active_ratio'] = userSub['user_app_active_counts'].apply(lambda x: x/app_counts)
del app_counts
user_app_active_counts = userSub.copy()


###############################################
########统计用户每个年龄段安装的app
###############################################

age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
userSub = pd.merge(age_train, user_app, how='left', on='uId')

userSub=pd.pivot_table(userSub, values='uId', index=['appId'],columns=['age_group'],aggfunc='count', fill_value=0)

userSub['sum']=userSub.sum(axis=1)
userSub= userSub.reset_index()
userSub.rename(columns={1:'age_1',2:'age_2',3:'age_3',4:'age_4',5:'age_5',6:'age_6'},inplace=True)
userSub.drop(axis=0, index=0, inplace=True)

userSub['age1_%']= userSub.apply(lambda x: round(x['age_1']/x['sum'],2),axis=1)
userSub['age2_%']= userSub.apply(lambda x: round(x['age_2']/x['sum'],2),axis=1)
userSub['age3_%']= userSub.apply(lambda x: round(x['age_3']/x['sum'],2),axis=1)
userSub['age4_%']= userSub.apply(lambda x: round(x['age_4']/x['sum'],2),axis=1)
userSub['age5_%']= userSub.apply(lambda x: round(x['age_5']/x['sum'],2),axis=1)
userSub['age6_%']= userSub.apply(lambda x: round(x['age_6']/x['sum'],2),axis=1)

age1 = userSub[(userSub['age1_%'] >= 0.3)][['appId']].copy()
age1['age_num1'] = 1

age2 = userSub[(userSub['age2_%'] >= 0.6)][['appId']].copy()
age2['age_num2'] = 1

age3 = userSub[(userSub['age3_%'] >= 0.6)][['appId']].copy()
age3['age_num3'] = 1

age4 = userSub[(userSub['age4_%'] >= 0.6)][['appId']].copy()
age4['age_num4'] = 1

age5 = userSub[(userSub['age5_%'] >= 0.3)][['appId']].copy()
age5['age_num5'] = 1

age6 = userSub[(userSub['age6_%'] >= 0.3)][['appId']].copy()
age6['age_num6'] = 1

userSub = pd.merge(user_app, age1, how='left', on='appId').fillna(0)
userSub = pd.merge(userSub, age2, how='left', on='appId').fillna(0)
userSub = pd.merge(userSub, age3, how='left', on='appId').fillna(0)
userSub = pd.merge(userSub, age4, how='left', on='appId').fillna(0)
userSub = pd.merge(userSub, age5, how='left', on='appId').fillna(0)
userSub = pd.merge(userSub, age6, how='left', on='appId').fillna(0)

userSub = userSub.groupby('uId').sum().reset_index()
user_active_app_age = userSub.copy()


###############################################
########## 用户安装各app类型的数量
###############################################
userSub = pd.merge(user_app, app_info, how='left', on='appId').fillna(method='pad')
userSub = pd.pivot_table(userSub, values='appId', index=['uId'],columns=['category'], aggfunc='count', fill_value=0).reset_index()
userSub['use_app_cate_nums']=0
for i in range(25):
    userSub['use_app_cate_nums']+=userSub[float(i)]
for i in range(26,30):
    userSub['use_app_cate_nums']+=userSub[float(i)]
for i in range(34,36):
    userSub['use_app_cate_nums']+=userSub[float(i)]

for i in range(25):
    userSub[str(float(i))+ '_ratio']=userSub[float(i)]/userSub['use_app_cate_nums']
for i in range(26,30):
    userSub[str(float(i))+ '_ratio']=userSub[float(i)]/userSub['use_app_cate_nums']
for i in range(34,36):
    userSub[str(float(i))+ '_ratio']=userSub[float(i)]/userSub['use_app_cate_nums']

user_active_category_counts = userSub.copy()


###############################################
########## 用户安装了多少种app类型
###############################################
userSub = pd.merge(user_app, app_info, how='left', on='appId').fillna(method='pad')
userSub = userSub[['uId', 'category']].groupby('uId')['category'].nunique().reset_index()
userSub.rename(columns={'category': 'active_cate_nums'}, inplace=True)
user_active_cate_nums = userSub.copy()


###############################################
########## 计算每个app的目标客户年龄指数
###############################################
age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
userSub = pd.merge(age_train, user_app, how='left', on='uId')

userSub=pd.pivot_table(userSub, values='uId', index=['appId'],columns=['age_group'],
                                aggfunc='count', fill_value=0)

userSub['sum']=userSub.sum(axis=1)
userSub= userSub.reset_index()
userSub.rename(columns={1:'age_1',2:'age_2',3:'age_3',4:'age_4',5:'age_5',6:'age_6'},inplace=True)
userSub.drop(axis=0, index=0, inplace=True)

userSub['age1_%']= userSub.apply(lambda x: round(x['age_1']/x['sum'],2),axis=1)
userSub['age2_%']= userSub.apply(lambda x: round(x['age_2']/x['sum'],2),axis=1)
userSub['age3_%']= userSub.apply(lambda x: round(x['age_3']/x['sum'],2),axis=1)
userSub['age4_%']= userSub.apply(lambda x: round(x['age_4']/x['sum'],2),axis=1)
userSub['age5_%']= userSub.apply(lambda x: round(x['age_5']/x['sum'],2),axis=1)
userSub['age6_%']= userSub.apply(lambda x: round(x['age_6']/x['sum'],2),axis=1)

# 计算每个app的目标客户年龄指数（计算方法 ：sum(app在该年龄段N安装比例 * 年龄段数值N * 10 / 对应年龄段样本比例))
userSub['age_weight']=userSub.apply(lambda x:(10*x['age1_%']/0.03 +20*x['age2_%']/0.2 +30*x['age3_%']/0.3 +40*x['age4_%']/0.25 +50*x['age5_%']/0.15 +60*x['age6_%']/0.075) ,axis=1)
userSub=userSub[['appId','age_weight']]
userSub=pd.merge(user_app,userSub,how='left',on='appId')

userSub=userSub.groupby('uId')['age_weight'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub['sqrt_of_mean']=userSub['mean']**2
userSub=round(userSub)
feature_activated_app_age_weight = userSub.copy()



###############################################
########## 计算每个app的在每个年龄组的概率
###############################################
age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
userSub = pd.merge(age_train, user_app, how='inner', on='uId')

userSub=pd.pivot_table(userSub, values='uId', index=['appId'],columns=['age_group'],aggfunc='count', fill_value=0)
userSub['sum']=userSub.sum(axis=1)
userSub= userSub.reset_index()
userSub.rename(columns={1:'age_1',2:'age_2',3:'age_3',4:'age_4',5:'age_5',6:'age_6'},inplace=True)
userSub.drop(axis=0, index=0, inplace=True)

userSub = userSub[(userSub['sum']>=20)].copy()

userSub['age1_%']= userSub.apply(lambda x: round(x['age_1']/x['sum'],4),axis=1)
userSub['age2_%']= userSub.apply(lambda x: round(x['age_2']/x['sum'],4),axis=1)
userSub['age3_%']= userSub.apply(lambda x: round(x['age_3']/x['sum'],4),axis=1)
userSub['age4_%']= userSub.apply(lambda x: round(x['age_4']/x['sum'],4),axis=1)
userSub['age5_%']= userSub.apply(lambda x: round(x['age_5']/x['sum'],4),axis=1)
userSub['age6_%']= userSub.apply(lambda x: round(x['age_6']/x['sum'],4),axis=1)

userSub=pd.merge(user_app,userSub,how='left',on='appId')

age_features_ratio = age_train[['uId']].copy()
age_features_use_nums = age_train[['uId']].copy()
for i in range(1,7):
    age_features=userSub.groupby('uId')['age'+str(i)+'_%'].agg({'mean','min','max','median','std','var'}).reset_index()
    age_features['sqrt_of_mean_age'+str(i)+'act_%']=age_features['mean']**2
    age_features.rename({'mean':'mean_'+str(i)+'act_%','min':'min_'+str(i)+'act_%','max':'max_'+str(i)+'act_%','median':'median_'+str(i)+'act_%','std':'std_'+str(i)+'act_%','var':'var_'+str(i)+'act_%'},axis=1,inplace=True)
    age_features_ratio = pd.merge(age_features_ratio,age_features,how='left',on='uId')
    
    age_features=userSub.groupby('uId')['age_'+str(i)].agg({'mean','min','max','median','std','var'}).reset_index()
    age_features.rename({'mean':'mean_act'+str(i),'min':'min_act'+str(i),'max':'max_act'+str(i),'median':'median_act'+str(i),'std':'std_act'+str(i),'var':'var_act'+str(i)},axis=1,inplace=True)
    age_features_use_nums = pd.merge(age_features_use_nums,age_features,how='left',on='uId')
    
feature_actived_app_age_weight_all = pd.merge(age_features_ratio,age_features_use_nums,how='left',on='uId')
feature_actived_app_age_weight_all =memory_preprocess._memory_process(feature_actived_app_age_weight_all )


###############################################
########## # 提取激活表类型的TF-IDF 特征
###############################################
app_info = pd.read_csv('../../data/processed_data/app_info.csv', dtype={'appId':str, 'category':str})
userSub = pd.merge(user_app, app_info, how='left', on='appId')
userSub = userSub.fillna(method='pad')
userSub = userSub.groupby(['uId'])['category'].apply(','.join).reset_index()

## CountVectorizer
# c_vec = CountVectorizer(lowercase=False,ngram_range=(1,1),dtype=np.int8,min_df=0,token_pattern='(?u)\\b\\w+\\b')
# activated_cate_cntVec = c_vec.fit_transform(userSub.category).toarray()
# activated_cate_cntVec = pd.DataFrame(activated_cate_cntVec)
# activated_cate_cntVec = pd.concat([userSub[['uId']],activated_cate_cntVec],axis=1)
# activated_cate_cntVec.to_csv('../../data/features_actived/activated_cate_cntVec.csv', index=False, encoding='utf-8')

# TfidfVectorizer
tf_vec = TfidfVectorizer(lowercase=False,ngram_range=(1,1),min_df=0,token_pattern='(?u)\\b\\w+\\b')
activated_cate_tfidfVec = tf_vec.fit_transform(userSub.category).toarray()
activated_cate_tfidfVec = activated_cate_tfidfVec.astype(np.float16)
activated_cate_tfidfVec = pd.DataFrame(activated_cate_tfidfVec)
activated_cate_tfidfVec = pd.concat([userSub[['uId']],activated_cate_tfidfVec],axis=1)
gc.collect()


###############################################
#########拼接各维度特征
###############################################
dataSet = pd.merge(user_app_active_counts, user_active_app_age, how='left', on='uId').fillna(0)
dataSet = pd.merge(dataSet, user_active_category_counts, how='left', on='uId').fillna(0)
dataSet = pd.merge(dataSet, user_active_cate_nums, how='left', on='uId').fillna(0)
dataSet = pd.merge(dataSet, feature_activated_app_age_weight, how='left', on='uId').fillna(0)
dataSet = pd.merge(dataSet, feature_actived_app_age_weight_all, how='left', on='uId').fillna(0)
dataSet = pd.merge(dataSet, activated_cate_tfidfVec, how='left', on='uId').fillna(0)
dataSet=memory_preprocess._memory_process(dataSet)
dataSet.to_csv('../../data/features/actived_features_all.csv', index=False, encoding='utf-8')

###############################################
#########转成csr稀疏矩阵形式存储
###############################################
dataSet.sort_values('uId', axis=0, ascending=True, inplace=True)

actived_features_train = pd.merge(age_train[['uId']],dataSet, how='left', on='uId').fillna(0)
actived_features_test = pd.merge(age_test,dataSet, how='left', on='uId').fillna(0)

actived_features_usage_train = pd.merge(age_train_usage[['uId']],dataSet, how='inner', on='uId').fillna(0)
actived_features_usage_test = pd.merge(age_test_usage,dataSet, how='inner', on='uId').fillna(0)

actived_features_train.drop('uId',axis=1,inplace=True)
actived_features_test.drop('uId',axis=1,inplace=True)
actived_features_usage_train.drop('uId',axis=1,inplace=True)
actived_features_usage_test.drop('uId',axis=1,inplace=True)


actived_features_train = csr_matrix(actived_features_train) 
actived_features_test = csr_matrix(actived_features_test)
print(actived_features_train.shape)
print(actived_features_test.shape)
gc.collect()

actived_features_usage_train = csr_matrix(actived_features_usage_train) 
actived_features_usage_test = csr_matrix(actived_features_usage_test)
print(actived_features_usage_train.shape)
print(actived_features_usage_test.shape)
gc.collect()

# 存储全量样本的训练集和测试集
sparse.save_npz('../../data/csr_features_full/actived_features_train.npz', actived_features_train)
sparse.save_npz('../../data/csr_features_full/actived_features_test.npz', actived_features_test)
# actived_features_train = sparse.load_npz('../../data/csr_features_full/actived_features_train.npz')
# actived_features_test = sparse.load_npz('../../data/csr_features_full/actived_features_test.npz')

# 存储缺失样本的训练集和测试集
sparse.save_npz('../../data/csr_features_usage/actived_features_usage_train.npz', actived_features_usage_train)
sparse.save_npz('../../data/csr_features_usage/actived_features_usage_test.npz', actived_features_usage_test)
# actived_features_usage_train = sparse.load_npz('../../data/csr_features_usage/actived_features_usage_train.npz')
# actived_features_usage_test = sparse.load_npz('../../data/csr_features_usage/actived_features_usage_test.npz')
