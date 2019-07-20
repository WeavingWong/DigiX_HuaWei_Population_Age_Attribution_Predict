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

age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test = pd.read_csv('../../data/processed_data/age_test.csv',dtype={'uId':np.int32})

# 结果概率文件1
mlp_output_train_cnt=pd.read_csv('../../data/prob_file/mlp_output_train_cnt.csv')
mlp_output_train_cnt=memory_preprocess._memory_process(mlp_output_train_cnt)
mlp_output_test_cnt=pd.read_csv('../../data/prob_file/mlp_output_test_cnt.csv')
mlp_output_test_cnt=memory_preprocess._memory_process(mlp_output_test_cnt)
mlp_output_train_cnt.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
mlp_output_test_cnt.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(age_train[['uId']], mlp_output_train_cnt, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(age_test, mlp_output_test_cnt, how='left', on='uId').fillna(0)

# 结果概率文件2
mlp_output_train=pd.read_csv('../../data/prob_file/mlp_output_train.csv')
mlp_output_train=memory_preprocess._memory_process(mlp_output_train)
mlp_output_test=pd.read_csv('../../data/prob_file/mlp_output_test.csv')
mlp_output_test=memory_preprocess._memory_process(mlp_output_test)
mlp_output_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
mlp_output_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train = pd.merge(train_output_prod, mlp_output_train, how='left', on='uId').fillna(0)
test = pd.merge(test_output_prod, mlp_output_test, how='left', on='uId').fillna(0)

# 结果概率文件3
act_use_rnn_train=pd.read_csv('../../data/prob_file/act_use_rnn_train.csv')
act_use_rnn_train=memory_preprocess._memory_process(act_use_rnn_train)
act_use_rnn_test=pd.read_csv('../../data/prob_file/act_use_rnn_test.csv')
act_use_rnn_test=memory_preprocess._memory_process(act_use_rnn_test)
act_use_rnn_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train = pd.merge(train_output_prod, act_use_rnn_train, how='left', on='uId').fillna(0)
test = pd.merge(test_output_prod, act_use_rnn_test, how='left', on='uId').fillna(0)

# 结果概率文件4
act_use_rnn_train_v1=pd.read_csv('../../data/prob_file/act_use_rnn_train_v1.csv')
act_use_rnn_train_v1=memory_preprocess._memory_process(act_use_rnn_train_v1)
act_use_rnn_test_v1=pd.read_csv('../../data/prob_file/act_use_rnn_test_v1.csv')
act_use_rnn_test_v1=memory_preprocess._memory_process(act_use_rnn_test_v1)
act_use_rnn_train_v1.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test_v1.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_use_rnn_train_v1, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_use_rnn_test_v1, how='left', on='uId').fillna(0)

# 结果概率文件5
act_use_rnn_train_v2=pd.read_csv('../../data/prob_file/act_use_rnn_train_v2.csv')
act_use_rnn_train_v2=memory_preprocess._memory_process(act_use_rnn_train_v2)
act_use_rnn_test_v2=pd.read_csv('../../data/prob_file/act_use_rnn_test_v2.csv')
act_use_rnn_test_v2=memory_preprocess._memory_process(act_use_rnn_test_v2)
act_use_rnn_train_v2.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test_v2.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_use_rnn_train_v2, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_use_rnn_test_v2, how='left', on='uId').fillna(0)

# 结果概率文件6
act_use_rnn_train_mlp=pd.read_csv('../../data/prob_file/act_use_rnn_train_mlp.csv')
act_use_rnn_train_mlp=memory_preprocess._memory_process(act_use_rnn_train_mlp)
act_use_rnn_test_mlp=pd.read_csv('../../data/prob_file/act_use_rnn_test_mlp.csv')
act_use_rnn_test_mlp=memory_preprocess._memory_process(act_use_rnn_test_mlp)
act_use_rnn_train_mlp.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test_mlp.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_use_rnn_train_mlp, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_use_rnn_test_mlp, how='left', on='uId').fillna(0)

# 结果概率文件8
act_all_train_mlp=pd.read_csv('../../data/prob_file/act_all_train_mlp.csv')
act_all_train_mlp=memory_preprocess._memory_process(act_all_train_mlp)
act_all_test_mlp=pd.read_csv('../../data/prob_file/act_all_test_mlp.csv')
act_all_test_mlp=memory_preprocess._memory_process(act_all_test_mlp)
act_all_train_mlp.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_all_test_mlp.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_all_train_mlp, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_all_test_mlp, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件1(不加NN，不加usage)
lgb_valid_full_csr_v1=pd.read_csv('../../data/prob_file/lgb_valid_full_csr_v1.csv')
lgb_valid_full_csr_v1=memory_preprocess._memory_process(lgb_valid_full_csr_v1)
lgb_test_full_csr_v1=pd.read_csv('../../data/prob_file/lgb_test_full_csr_v1.csv')
lgb_test_full_csr_v1=memory_preprocess._memory_process(lgb_test_full_v1)
train_output_prod = pd.merge(train_output_prod, lgb_valid_full_csr_v1, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_full_csr_v1, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件2(加NN，不加usage)
lgb_valid_full_csr_v2=pd.read_csv('../../data/prob_file/lgb_valid_full_csr_v2.csv')
lgb_valid_full_csr_v2=memory_preprocess._memory_process(lgb_valid_full_csr_v2)
lgb_test_full_csr_v2=pd.read_csv('../../data/prob_file/lgb_test_full_csr_v2.csv')
lgb_test_full_csr_v2=memory_preprocess._memory_process(lgb_test_full_csr_v2)
train_output_prod = pd.merge(train_output_prod, lgb_valid_full_csr_v2, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_full_csr_v2, how='left', on='uId').fillna(0)


train_output_prod = csr_matrix(train_output_prod,dtype='float32') 
test_output_prod = csr_matrix(test_output_prod,dtype='float32') 
print(train_output_prod.shape)
print(test_output_prod.shape)
gc.collect()


sparse.save_npz('../../data/csr_features_full/train_output_prod_full.npz', train_output_prod)
sparse.save_npz('../../data/csr_features_full/test_output_prod_full.npz', test_output_prod)
# train_output_prod = sparse.load_npz('../../data/csr_features_full/train_output_prod_full.npz')
# test_output_prod = sparse.load_npz('../../data/csr_features_full/test_output_prod_full.npz')