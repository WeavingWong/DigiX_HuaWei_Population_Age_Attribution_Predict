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
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import pyplot
import seaborn as sns
import lightgbm as lgb
import time
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.metrics import f1_score
import warnings
from sklearn.metrics import accuracy_score
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

age_train=pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test=pd.read_csv('../../data/processed_data/age_test.csv',dtype={'uId':np.int32})
age_train_usage=pd.read_csv('../../data/processed_data/age_train_usage.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test_usage=pd.read_csv('../../data/processed_data/age_test_usage.csv',dtype={'uId':np.int32})


# 激活表——rnn_feature_v1
rnn_feature_train=pd.read_csv('../../data/features/rnn_feature_train.csv')
rnn_feature_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)

rnn_feature_test=pd.read_csv('../../data/features/rnn_feature_test.csv')
rnn_feature_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)

rnn_feature_train = pd.merge(age_train[['uId']],rnn_feature_train,how='inner',on='uId')
rnn_feature_test = pd.merge(age_test,rnn_feature_test,how='inner',on='uId')

rnn_feature_train.sort_values('uId', axis=0, ascending=True, inplace=True)
rnn_feature_test.sort_values('uId', axis=0, ascending=True, inplace=True)

rnn_feature_train.drop('uId',axis=1,inplace=True)
rnn_feature_test.drop('uId',axis=1,inplace=True)

rnn_feature_train = csr_matrix(rnn_feature_train,dtype='float32') 
rnn_feature_test = csr_matrix(rnn_feature_test,dtype='float32') 
print(rnn_feature_train.shape)
print(rnn_feature_test.shape)
gc.collect()

sparse.save_npz('../../data/csr_features_full/actived_rnn_features_train_v1.npz', rnn_feature_train)
sparse.save_npz('../../data/csr_features_full/actived_rnn_features_test_v1.npz', rnn_feature_test)
# actived_rnn_features_train_v1 = sparse.load_npz('../../data/csr_features_full/actived_rnn_features_train_v1.npz')
# actived_rnn_features_test_v1 = sparse.load_npz('../../data/csr_features_full/actived_rnn_features_test_v1.npz')


# 激活表——rnn_feature_v1
rnn_feature_train=pd.read_csv('../../data/features/rnn_feature_train.csv')
rnn_feature_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)

rnn_feature_test=pd.read_csv('../../data/features/rnn_feature_test.csv')
rnn_feature_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)

rnn_feature_train = pd.merge(age_train_usage[['uId']],rnn_feature_train,how='inner',on='uId')
rnn_feature_test = pd.merge(age_test_usage,rnn_feature_test,how='inner',on='uId')

rnn_feature_train.sort_values('uId', axis=0, ascending=True, inplace=True)
rnn_feature_test.sort_values('uId', axis=0, ascending=True, inplace=True)

rnn_feature_train.drop('uId',axis=1,inplace=True)
rnn_feature_test.drop('uId',axis=1,inplace=True)

rnn_feature_train = csr_matrix(rnn_feature_train,dtype='float32') 
rnn_feature_test = csr_matrix(rnn_feature_test,dtype='float32') 
print(rnn_feature_train.shape)
print(rnn_feature_test.shape)
gc.collect()

sparse.save_npz('../../data/csr_features_usage/actived_rnn_features_train_usage_v1.npz', rnn_feature_train)
sparse.save_npz('../../data/csr_features_usage/actived_rnn_features_test_usage_v1.npz', rnn_feature_test)
# actived_rnn_features_train_usage_v1 = sparse.load_npz('../../data/csr_features_usage/actived_rnn_features_train_usage_v1.npz')
# actived_rnn_features_test_usage_v1 = sparse.load_npz('../../data/csr_features_usage/actived_rnn_features_test_usage_v1.npz')



# usage表——rnn_feature
usage_rnnV2_train_160=pd.read_csv('../../data/features/usage_rnnV2_train_160.csv')
usage_rnnV2_train_160.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)

usage_rnnV2_test_160=pd.read_csv('../../data/features/usage_rnnV2_test_160.csv')
usage_rnnV2_test_160.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)

usage_rnnV2_train_160 = pd.merge(age_train[['uId']],usage_rnnV2_train_160,how='inner',on='uId')
usage_rnnV2_test_160 = pd.merge(age_test,usage_rnnV2_test_160,how='inner',on='uId')

usage_rnnV2_train_160.sort_values('uId', axis=0, ascending=True, inplace=True)
usage_rnnV2_test_160.sort_values('uId', axis=0, ascending=True, inplace=True)

usage_rnnV2_train_160.drop('uId',axis=1,inplace=True)
usage_rnnV2_test_160.drop('uId',axis=1,inplace=True)

usage_rnnV2_train_160 = csr_matrix(usage_rnnV2_train_160,dtype='float32') 
usage_rnnV2_test_160 = csr_matrix(usage_rnnV2_test_160,dtype='float32') 
print(usage_rnnV2_train_160.shape)
print(usage_rnnV2_test_160.shape)
gc.collect()

sparse.save_npz('../../data/csr_features_usage/usage_rnnV2_train_usage_160.npz', usage_rnnV2_train_usage_160)
sparse.save_npz('../../data/csr_features_usage/usage_rnnV2_test_usage_160.npz', usage_rnnV2_test_usage_160)
# usage_rnnV2_train_usage_160 = sparse.load_npz('../../data/csr_features_usage/usage_rnnV2_train_usage_160.npz')
# usage_rnnV2_test_usage_160 = sparse.load_npz('../../data/csr_features_usage/usage_rnnV2_test_usage_160.npz')