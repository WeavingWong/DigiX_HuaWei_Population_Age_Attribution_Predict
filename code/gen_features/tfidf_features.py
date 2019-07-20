from scipy import sparse
from numpy import array
from scipy.sparse import csr_matrix
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
import gc


###############################################
########## 数据加载
###############################################
userAll = pd.read_csv('../../data/original_data/user_app_usage.csv', names=['uId', 'appId', 'duration', 'times', 'use_date'], dtype={'uId':np.int32, 'appId':str, 'duration':np.float32, 
                                                                'times':np.float32, 'use_date':str})
user_app_actived = pd.read_csv('../data/processed_data/user_app_actived.csv')
app_info = pd.read_csv('../../data/processed_data/app_info.csv', dtype={'appId':str, 'category':str})
age_train=pd.read_csv('../../data/processed_data/age_train.csv').sort_values('uId')
age_test=pd.read_csv('../../data/processed_data/age_test.csv').sort_values('uId')
age_train_usage=pd.read_csv('../../data/processed_data/age_train_usage.csv').sort_values('uId')
age_test_usage=pd.read_csv('../../data/processed_data/age_test_usage.csv').sort_values('uId')


###############################################
########## 激活表全量样本的TFIDF特征
##############################################
app_list = user_app_actived.appId.str.split('#')
app_list = [" ".join(app) for app in app_list]

tf_vec = TfidfVectorizer(lowercase=False,ngram_range=(1,1), min_df=0.0008, token_pattern='(?u)\\b\\w+\\b')
full_tfidf = tf_vec.fit_transform(app_list).toarray()
full_tfidf = full_tfidf.astype(np.float16)
print(full_tfidf.shape)
full_tfidf = pd.DataFrame(full_tfidf,dtype='float16')
full_tfidf = pd.concat([user_app_actived[['uId']],full_tfidf],axis=1)

train = pd.merge(age_train[['uId']],full_tfidf, how='inner', on='uId').fillna(0)
test = pd.merge(age_test,full_tfidf, how='inner', on='uId').fillna(0)
train.sort_values('uId', axis=0, ascending=True, inplace=True)
test.sort_values('uId', axis=0, ascending=True, inplace=True)
train.drop('uId',axis=1,inplace=True)
test.drop('uId',axis=1,inplace=True)
del user_app_actived,full_tfidf,app_list
gc.collect()

train = csr_matrix(train) 
test = csr_matrix(test) 
print(train.shape)
print(test.shape)
gc.collect()

sparse.save_npz('../../data/csr_features_full/actived_app_tfidf_train_3000.npz', tfidf_train)
sparse.save_npz('../../data/csr_features_full/actived_app_tfidf_test_3000.npz', tfidf_test)
# actived_app_tfidf_train_3000 = sparse.load_npz('../../data/csr_features_full/actived_app_tfidf_train_3000.npz')
# actived_app_tfidf_test_3000 = sparse.load_npz('../../data/csr_features_full/actived_app_tfidf_test_3000.npz')


###############################################
########## 激活表中关于usage表样本的TFIDF特征
##############################################
app_list = user_app_actived.appId.str.split('#')
app_list = [" ".join(app) for app in app_list]

tf_vec = TfidfVectorizer(lowercase=False,ngram_range=(1,1), min_df=0.0008, token_pattern='(?u)\\b\\w+\\b')
full_tfidf = tf_vec.fit_transform(app_list).toarray()
full_tfidf = full_tfidf.astype(np.float16)
print(full_tfidf.shape)
full_tfidf = pd.DataFrame(full_tfidf,dtype='float16')
full_tfidf = pd.concat([user_app_actived[['uId']],full_tfidf],axis=1)

train = pd.merge(age_train_usage[['uId']],full_tfidf, how='inner', on='uId').fillna(0)
test = pd.merge(age_test_usage,full_tfidf, how='inner', on='uId').fillna(0)
train.sort_values('uId', axis=0, ascending=True, inplace=True)
test.sort_values('uId', axis=0, ascending=True, inplace=True)
train.drop('uId',axis=1,inplace=True)
test.drop('uId',axis=1,inplace=True)
del user_app_actived,full_tfidf,app_list
gc.collect()

train = csr_matrix(train) 
test = csr_matrix(test) 
print(train.shape)
print(test.shape)
gc.collect()

sparse.save_npz('../../data/csr_features_usage/actived_app_tfidf_train_3000_usage.npz', train)
sparse.save_npz('../../data/csr_features_usage/actived_app_tfidf_test_3000_usage.npz', test)
# actived_app_tfidf_train_3000_usage = sparse.load_npz('../../data/csr_features_usage/actived_app_tfidf_train_3000_usage.npz')
# actived_app_tfidf_test_3000_usage = sparse.load_npz('../../data/csr_features_usage/actived_app_tfidf_test_3000_usage.npz')


###############################################
########## 使用表的TFIDF特征
###############################################
train_uId_list=age_train_usage.uId.tolist()
test_uId_list=age_test_usage.uId.tolist()

user_app_usage = pd.read_csv('../../data/features_usage/usage_app_list_duration.csv').sort_values('uId').reset_index(drop=True)
# train_index
user_app_usage_train=user_app_usage.loc[user_app_usage['uId'].isin(train_uId_list)]
index_train=user_app_usage_train.index.tolist()
# test_index
user_app_usage_test=user_app_usage.loc[user_app_usage['uId'].isin(test_uId_list)]
index_test=user_app_usage_test.index.tolist()

app_list = user_app_usage.app_list.str.split(',')
app_list = [" ".join(app) for app in app_list]

tf_vec = TfidfVectorizer(lowercase=False,ngram_range=(1,1),dtype=np.float32,min_df=0.001,token_pattern='(?u)\\b\\w+\\b')
usage_tfidf = tf_vec.fit_transform(app_list)
print(usage_tfidf.shape)

tfidf_train = usage_tfidf[index_train]
tfidf_test = usage_tfidf[index_test]
print(tfidf_train.shape)
print(tfidf_test.shape)

del user_app_usage,usage_tfidf,app_list,user_app_usage_train,user_app_usage_test
gc.collect()

sparse.save_npz('../../data/csr_features_usage/usage_app_tfidf_train_2200.npz', tfidf_train)
sparse.save_npz('../../data/csr_features_usage/usage_app_tfidf_test_2200.npz', tfidf_test)
# usage_app_tfidf_train_2200 = sparse.load_npz('../../data/csr_features_usage/usage_app_tfidf_train_2200.npz')
# usage_app_tfidf_test_2200 = sparse.load_npz('../../data/csr_features_usage/usage_app_tfidf_test_2200.npz')