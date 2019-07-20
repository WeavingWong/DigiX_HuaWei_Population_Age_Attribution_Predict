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


###############################################
#########数据加载
###############################################
age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test= pd.read_csv('../../data/processed_data/age_test.csv',dtype={'uId':np.int32})
age_train_usage = pd.read_csv('../../data/processed_data/age_train_usage.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test_usage = pd.read_csv('../../data/processed_data/age_test_usage.csv',dtype={'uId':np.int32})


###############################################
#########结果概率文件拼接
###############################################
# 结果概率文件1
mlp_output_train_cnt=pd.read_csv('../../data/model_result/mlp_output_train_cnt.csv')
mlp_output_train_cnt=memory_preprocess._memory_process(mlp_output_train_cnt)
mlp_output_test_cnt=pd.read_csv('../../data/model_result/mlp_output_test_cnt.csv')
mlp_output_test_cnt=memory_preprocess._memory_process(mlp_output_test_cnt)
mlp_output_train_cnt.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
mlp_output_test_cnt.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(age_train[['uId']], mlp_output_train_cnt, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(age_test, mlp_output_test_cnt, how='left', on='uId').fillna(0)

# 结果概率文件2
mlp_output_train=pd.read_csv('../../data/model_result/mlp_output_train.csv')
mlp_output_train=memory_preprocess._memory_process(mlp_output_train)
mlp_output_test=pd.read_csv('../../data/model_result/mlp_output_test.csv')
mlp_output_test=memory_preprocess._memory_process(mlp_output_test)
mlp_output_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
mlp_output_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train = pd.merge(train_output_prod, mlp_output_train, how='left', on='uId').fillna(0)
test = pd.merge(test_output_prod, mlp_output_test, how='left', on='uId').fillna(0)

# 结果概率文件3
act_use_rnn_train=pd.read_csv('../../data/model_result/act_use_rnn_train.csv')
act_use_rnn_train=memory_preprocess._memory_process(act_use_rnn_train)
act_use_rnn_test=pd.read_csv('../../data/model_result/act_use_rnn_test.csv')
act_use_rnn_test=memory_preprocess._memory_process(act_use_rnn_test)
act_use_rnn_train.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train = pd.merge(train_output_prod, act_use_rnn_train, how='left', on='uId').fillna(0)
test = pd.merge(test_output_prod, act_use_rnn_test, how='left', on='uId').fillna(0)

# 结果概率文件4
act_use_rnn_train_v1=pd.read_csv('../../data/model_result/act_use_rnn_train_v1.csv')
act_use_rnn_train_v1=memory_preprocess._memory_process(act_use_rnn_train_v1)
act_use_rnn_test_v1=pd.read_csv('../../data/model_result/act_use_rnn_test_v1.csv')
act_use_rnn_test_v1=memory_preprocess._memory_process(act_use_rnn_test_v1)
act_use_rnn_train_v1.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test_v1.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_use_rnn_train_v1, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_use_rnn_test_v1, how='left', on='uId').fillna(0)

# 结果概率文件5
act_use_rnn_train_v2=pd.read_csv('../../data/model_result/act_use_rnn_train_v2.csv')
act_use_rnn_train_v2=memory_preprocess._memory_process(act_use_rnn_train_v2)
act_use_rnn_test_v2=pd.read_csv('../../data/model_result/act_use_rnn_test_v2.csv')
act_use_rnn_test_v2=memory_preprocess._memory_process(act_use_rnn_test_v2)
act_use_rnn_train_v2.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test_v2.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_use_rnn_train_v2, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_use_rnn_test_v2, how='left', on='uId').fillna(0)

# 结果概率文件6
act_use_rnn_train_mlp=pd.read_csv('../../data/model_result/act_use_rnn_train_mlp.csv')
act_use_rnn_train_mlp=memory_preprocess._memory_process(act_use_rnn_train_mlp)
act_use_rnn_test_mlp=pd.read_csv('../../data/model_result/act_use_rnn_test_mlp.csv')
act_use_rnn_test_mlp=memory_preprocess._memory_process(act_use_rnn_test_mlp)
act_use_rnn_train_mlp.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
act_use_rnn_test_mlp.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, act_use_rnn_train_mlp, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, act_use_rnn_test_mlp, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件1(不加NN，不加usage)
lgb_valid_full_v1=pd.read_csv('../../data/model_result/lgb_valid_full_v1.csv')
lgb_valid_full_v1=memory_preprocess._memory_process(lgb_valid_full_v1)
lgb_test_full_v1=pd.read_csv('../../data/model_result/lgb_test_full_v1.csv')
lgb_test_full_v1=memory_preprocess._memory_process(lgb_test_full_v1)
train_output_prod = pd.merge(train_output_prod, lgb_valid_full_v1, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_full_v1, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件2(加NN，不加usage)
lgb_valid_full_2500=pd.read_csv('../../data/model_result/lgb_valid_full_2500.csv')
lgb_valid_full_2500=memory_preprocess._memory_process(lgb_valid_full_2500)
lgb_test_full_2500=pd.read_csv('../../data/model_result/lgb_test_full_2500.csv')
lgb_test_full_2500=memory_preprocess._memory_process(lgb_test_full_2500)
train_output_prod = pd.merge(train_output_prod, lgb_valid_full_2500, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_full_2500, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件3(不加NN，加usage)
lgb_valid_usage_csr_v1=pd.read_csv('../../data/model_result/lgb_valid_usage_csr_v1.csv')
lgb_valid_usage_csr_v1=memory_preprocess._memory_process(lgb_valid_usage_csr_v1)
lgb_test_usage_csr_v1=pd.read_csv('../../data/model_result/lgb_test_usage_csr_v1.csv')
lgb_test_usage_csr_v1=memory_preprocess._memory_process(lgb_test_usage_csr_v1)
train_output_prod = pd.merge(train_output_prod, lgb_valid_usage_csr_v1, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_usage_csr_v1, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件4(加NN，加usage)
lgb_valid_usage_csr_v2=pd.read_csv('../../data/model_result/lgb_valid_usage_csr_v2.csv')
lgb_valid_usage_csr_v2=memory_preprocess._memory_process(lgb_valid_usage_csr_v2)
lgb_test_usage_csr_v2=pd.read_csv('../../data/model_result/lgb_test_usage_csr_v2.csv')
lgb_test_usage_csr_v2=memory_preprocess._memory_process(lgb_test_usage_csr_v2)
train_output_prod = pd.merge(train_output_prod, lgb_valid_usage_csr_v2, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_usage_csr_v2, how='left', on='uId').fillna(0)

# LGB模型的结果概率文件5(单纯usage的手工特征)
lgb_valid_usage_v3=pd.read_csv('../../data/model_result/lgb_valid_usage_v3.csv')
lgb_valid_usage_v3=memory_preprocess._memory_process(lgb_valid_usage_v3)
lgb_test_usage_v3=pd.read_csv('../../data/model_result/lgb_test_usage_v3.csv')
lgb_test_usage_v3=memory_preprocess._memory_process(lgb_test_usage_v3)
train_output_prod = pd.merge(train_output_prod, lgb_valid_usage_v3, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, lgb_test_usage_v3, how='left', on='uId').fillna(0)

# NN模型的结果概率文件
dnn_output_train_usage=pd.read_csv('../../data/model_result/dnn_output_train_usage.csv')
dnn_output_train_usage=memory_preprocess._memory_process(dnn_output_train_usage)
dnn_output_test_usage=pd.read_csv('../../data/model_result/dnn_output_test_usage.csv')
dnn_output_test_usage=memory_preprocess._memory_process(dnn_output_test_usage)
dnn_output_train_usage.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
dnn_output_test_usage.rename(columns={'Unnamed: 0': 'uId'}, inplace=True)
train_output_prod = pd.merge(train_output_prod, dnn_output_train_usage, how='left', on='uId').fillna(0)
test_output_prod = pd.merge(test_output_prod, dnn_output_test_usage, how='left', on='uId').fillna(0)


###############################################
#########转csr稀疏矩阵形式
###############################################
train_output_prod_usage = pd.merge(age_train_usage[['uId']], train_output_prod, how='inner', on='uId')
test_output_prod_usage = pd.merge(age_test_usage, test_output_prod, how='inner', on='uId')

train_output_prod = csr_matrix(train_output_prod,dtype='float32') 
test_output_prod = csr_matrix(test_output_prod,dtype='float32') 
print(train_output_prod.shape)
print(test_output_prod.shape)
gc.collect()


train_output_prod_usage = csr_matrix(train_output_prod_usage,dtype='float32') 
test_output_prod_usage = csr_matrix(test_output_prod_usage,dtype='float32') 
print(train_output_prod_usage.shape)
print(test_output_prod_usage.shape)
gc.collect()


sparse.save_npz('../../data/csr_features_full/train_output_prod.npz', train_output_prod)
sparse.save_npz('../../data/csr_features_full/test_output_prod.npz', test_output_prod)
# train_output_prod = sparse.load_npz('../../data/csr_features_full/train_output_prod.npz')
# test_output_prod = sparse.load_npz('../../data/csr_features_full/test_output_prod.npz')


sparse.save_npz('../../data/csr_features_usage/train_output_prod_usage.npz', train_output_prod)
sparse.save_npz('../../data/csr_features_usage/test_output_prod_usage.npz', test_output_prod)
# train_output_prod = sparse.load_npz('../../data/csr_features_usage/train_output_prod_usage.npz')
# test_output_prod = sparse.load_npz('../../data/csr_features_usage/test_output_prod_usage.npz')
