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
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


###############################################
######基础表处理方式一：
######把基础表的各维度特征交叉组合，排序，独热，加减乘除
###############################################

# 以最节省内存的数据类型读取,可重复读取（调试使用）
user_basic_info_dtype={'uId':np.int32, 'gender':np.int8, 'city':str, 'prodName':str, 'ramCapacity':np.int8,
                       'ramLeftRation':np.float16, 'romCapacity':np.int16, 'romLeftRation':np.float16, 'color':str, 
                       'fontSize':np.float16, 'ct':str, 'carrier':str, 'os':np.float16}
user_behavior_info_dtype={'uId':np.int32, 'bootTimes':np.int32, 'AFuncTimes':np.float16, 'BFuncTimes':np.float16,
                        'CFuncTimes':np.float16, 'DFuncTimes':np.float16, 'EFuncTimes':np.float16, 'FFuncTimes':np.float16,
                          'FFuncSum':np.int32}
# 读取预处理后数据
user_basic_info = pd.read_csv('../../data/processed_data/user_basic_info.csv',dtype=user_basic_info_dtype)
user_behavior_info = pd.read_csv('../../data/processed_data/user_behavior_info.csv',dtype=user_behavior_info_dtype)  

"""
基础特征
"""
#  数值，以及类别过多 简单顺序编码
enc=preprocessing.OrdinalEncoder()
user_basic_info_copy=enc.fit_transform(user_basic_info[['ramCapacity','romCapacity','os','ramLeftRation','romLeftRation','fontSize','city','color','prodName']])
user_basic_info_copy=pd.DataFrame(user_basic_info_copy,columns=['ramCapacity_rank','romCapacity_rank','os_rank','ramLeftRation_rank','romLeftRation_rank','fontSize_rank','city_rank','color_rank','prodName_rank'])
user_basic_info=pd.concat([user_basic_info,user_basic_info_copy],axis=1)
del user_basic_info_copy
user_behavior_info_copy=enc.fit_transform(user_behavior_info[['bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum']])
user_behavior_info_copy=pd.DataFrame(user_behavior_info_copy,columns=['bootTimes_rank','AFuncTimes_rank','BFuncTimes_rank','CFuncTimes_rank','DFuncTimes_rank','EFuncTimes_rank','FFuncTimes_rank','FFuncSum_rank'])
user_behavior_info=pd.concat([user_behavior_info,user_behavior_info_copy],axis=1)
del user_behavior_info_copy

# 使用次数使用四舍五入
user_behavior_info.AFuncTimes=user_behavior_info.AFuncTimes * 10
user_behavior_info.BFuncTimes=user_behavior_info.BFuncTimes * 10
user_behavior_info.CFuncTimes=user_behavior_info.CFuncTimes * 10
user_behavior_info.DFuncTimes=user_behavior_info.DFuncTimes * 10
user_behavior_info.EFuncTimes=user_behavior_info.EFuncTimes * 10
user_behavior_info.FFuncTimes=user_behavior_info.FFuncTimes * 10

user_behavior_info['average_bootTimes'] = user_behavior_info['bootTimes']/30.0
user_behavior_info['average_dataFlow'] = user_behavior_info['FFuncSum']/30.0
user_behavior_info['square_bootTimes'] = user_behavior_info['bootTimes']**2
user_behavior_info['square_average_bootTimes'] = user_behavior_info['average_bootTimes']**2
user_behavior_info['square_AFuncTimes'] = user_behavior_info.AFuncTimes**2
user_behavior_info['square_BFuncTimes'] = user_behavior_info.BFuncTimes**2
user_behavior_info['square_CFuncTimes'] = user_behavior_info.CFuncTimes**2
user_behavior_info['square_DFuncTimes'] = user_behavior_info.DFuncTimes**2
user_behavior_info['square_EFuncTimes'] = user_behavior_info.EFuncTimes**2
user_behavior_info['square_FFuncTimes'] = user_behavior_info.FFuncTimes**2
user_behavior_info['behavior_sum']=user_behavior_info.AFuncTimes+user_behavior_info.BFuncTimes+user_behavior_info.CFuncTimes+user_behavior_info.DFuncTimes+user_behavior_info.EFuncTimes+user_behavior_info.FFuncTimes
user_behavior_info['camera_sum']=(user_behavior_info['BFuncTimes']+user_behavior_info['CFuncTimes'])**2
user_behavior_info['call_sum']=(user_behavior_info['DFuncTimes']+user_behavior_info['EFuncTimes'])**2


# 手机型号P，颜色C，字体F，RAM，Rom, os 交叉组合特征
user_basic_info['P_C_F']=user_basic_info[['prodName','color','fontSize']].apply(lambda x :x.prodName +'_'+ str(x.fontSize)+'_' + x.color,axis=1)
user_basic_info['R_R_O']=user_basic_info[['ramCapacity','romCapacity','os']].apply(lambda x :str(x.ramCapacity) +'_'+ str(x.romCapacity)+'_' + str(x.os),axis=1)
user_basic_info['PCFRRO']=user_basic_info[['ramCapacity','romCapacity','os','prodName','color','fontSize']].apply(lambda x :str(x.prodName)+'_'+str(x.color)+'_'+str(x.fontSize)+'_'+str(x.ramCapacity) +'_'+ str(x.romCapacity)+'_' + str(x.os),axis=1)


# 组合特征PCF
train = pd.read_csv('../../data/processed_data/age_train.csv')
user_basic_info_merge=pd.merge(train,user_basic_info,how ='left',on='uId')

user_basic_info_pivot=pd.pivot_table(user_basic_info_merge,values='uId',index=['P_C_F'],columns=['age_group'],aggfunc='count',fill_value=0)

user_basic_info_pivot['sum']=user_basic_info_pivot.sum(axis=1)
user_basic_info_pivot.rename(columns={1:'P_C_F_1',2:'P_C_F_2',3:'P_C_F_3',4:'P_C_F_4',5:'P_C_F_5',6:'P_C_F_6'},inplace=True)

# 计算每个年段在每个app中的安装比例
user_basic_info_pivot['P_C_F1_%']= user_basic_info_pivot.apply(lambda x: round(x['P_C_F_1']/x['sum'],2),axis=1)
user_basic_info_pivot['P_C_F2_%']= user_basic_info_pivot.apply(lambda x: round(x['P_C_F_2']/x['sum'],2),axis=1)
user_basic_info_pivot['P_C_F3_%']= user_basic_info_pivot.apply(lambda x: round(x['P_C_F_3']/x['sum'],2),axis=1)
user_basic_info_pivot['P_C_F4_%']= user_basic_info_pivot.apply(lambda x: round(x['P_C_F_4']/x['sum'],2),axis=1)
user_basic_info_pivot['P_C_F5_%']= user_basic_info_pivot.apply(lambda x: round(x['P_C_F_5']/x['sum'],2),axis=1)
user_basic_info_pivot['P_C_F6_%']= user_basic_info_pivot.apply(lambda x: round(x['P_C_F_6']/x['sum'],2),axis=1)
user_basic_info_pivot = user_basic_info_pivot.sort_values('sum',ascending=False)

# 筛选每个年龄段中安装量比例最高的100个左右app列表(年龄段中app可能有部分重复)
user_basic_info_select = user_basic_info_pivot[((user_basic_info_pivot['P_C_F1_%']>= 0.1) 
                                      | (user_basic_info_pivot['P_C_F2_%']>= 0.3) 
                                      | (user_basic_info_pivot['P_C_F3_%']>= 0.4)
                                      | (user_basic_info_pivot['P_C_F4_%']>= 0.35)
                                      | (user_basic_info_pivot['P_C_F5_%']>= 0.2)
                                      | (user_basic_info_pivot['P_C_F6_%']>= 0.15)
                                      ) & (user_basic_info_pivot['sum']>10000)|(user_basic_info_pivot['sum']>26000)].reset_index()[['P_C_F']]
del user_basic_info_pivot
print('选中PCF数量：',len(user_basic_info_select))
feature_P_C_F=pd.merge(user_basic_info[['uId','P_C_F']],user_basic_info_select[['P_C_F']],how='inner',on='P_C_F')
del user_basic_info_select
feature_P_C_F_dummies=pd.get_dummies(feature_P_C_F['P_C_F'],prefix='P_C_F')
feature_P_C_F=pd.concat([feature_P_C_F,feature_P_C_F_dummies],axis=1)
del  feature_P_C_F['P_C_F']
del  feature_P_C_F_dummies


# 组合特征RRO
user_basic_info_pivot=pd.pivot_table(user_basic_info_merge,values='uId',index=['R_R_O'],columns=['age_group'],aggfunc='count',fill_value=0)
user_basic_info_pivot['sum']=user_basic_info_pivot.sum(axis=1)
user_basic_info_pivot.rename(columns={1:'R_R_O_1',2:'R_R_O_2',3:'R_R_O_3',4:'R_R_O_4',5:'R_R_O_5',6:'R_R_O_6'},inplace=True)
# 计算每个年段在每个app中的安装比例
user_basic_info_pivot['R_R_O1_%']= user_basic_info_pivot.apply(lambda x: round(x['R_R_O_1']/x['sum'],2),axis=1)
user_basic_info_pivot['R_R_O2_%']= user_basic_info_pivot.apply(lambda x: round(x['R_R_O_2']/x['sum'],2),axis=1)
user_basic_info_pivot['R_R_O3_%']= user_basic_info_pivot.apply(lambda x: round(x['R_R_O_3']/x['sum'],2),axis=1)
user_basic_info_pivot['R_R_O4_%']= user_basic_info_pivot.apply(lambda x: round(x['R_R_O_4']/x['sum'],2),axis=1)
user_basic_info_pivot['R_R_O5_%']= user_basic_info_pivot.apply(lambda x: round(x['R_R_O_5']/x['sum'],2),axis=1)
user_basic_info_pivot['R_R_O6_%']= user_basic_info_pivot.apply(lambda x: round(x['R_R_O_6']/x['sum'],2),axis=1)

# 筛选每个年龄段中手机配置比例最高的20个左右app列表(年龄段中app可能有部分重复)
user_basic_info_select = user_basic_info_pivot[((user_basic_info_pivot['R_R_O1_%']>= 0.4) 
                                      | (user_basic_info_pivot['R_R_O2_%']>= 0.3) 
                                      | (user_basic_info_pivot['R_R_O3_%']>= 0.32)
                                      | (user_basic_info_pivot['R_R_O4_%']>= 0.32)
                                      | (user_basic_info_pivot['R_R_O5_%']>= 0.3)
                                      | (user_basic_info_pivot['R_R_O6_%']>= 0.3)
                                      ) & (user_basic_info_pivot['sum']>10000)|(user_basic_info_pivot['sum']>35000)].reset_index()[['R_R_O']]

del user_basic_info_pivot
print('选中RRO数量：',len(user_basic_info_select))
feature_R_R_O=pd.merge(user_basic_info[['uId','R_R_O']],user_basic_info_select[['R_R_O']],how='inner',on='R_R_O')
del user_basic_info_select
feature_R_R_O_dummies=pd.get_dummies(feature_R_R_O['R_R_O'],prefix='R_R_O')
feature_R_R_O=pd.concat([feature_R_R_O,feature_R_R_O_dummies],axis=1)
del feature_R_R_O['R_R_O']
del feature_R_R_O_dummies


# 组合特征PCFRRO
user_basic_info_pivot=pd.pivot_table(user_basic_info_merge,values='uId',index=['PCFRRO'],columns=['age_group'],aggfunc='count',fill_value=0)
user_basic_info_pivot['sum']=user_basic_info_pivot.sum(axis=1)
user_basic_info_pivot.rename(columns={1:'PCFRRO_1',2:'PCFRRO_2',3:'PCFRRO_3',4:'PCFRRO_4',5:'PCFRRO_5',6:'PCFRRO_6'},inplace=True)
# 计算每个年段在每个app中的安装比例
user_basic_info_pivot['PCFRRO1_%']= user_basic_info_pivot.apply(lambda x: round(x['PCFRRO_1']/x['sum'],2),axis=1)
user_basic_info_pivot['PCFRRO2_%']= user_basic_info_pivot.apply(lambda x: round(x['PCFRRO_2']/x['sum'],2),axis=1)
user_basic_info_pivot['PCFRRO3_%']= user_basic_info_pivot.apply(lambda x: round(x['PCFRRO_3']/x['sum'],2),axis=1)
user_basic_info_pivot['PCFRRO4_%']= user_basic_info_pivot.apply(lambda x: round(x['PCFRRO_4']/x['sum'],2),axis=1)
user_basic_info_pivot['PCFRRO5_%']= user_basic_info_pivot.apply(lambda x: round(x['PCFRRO_5']/x['sum'],2),axis=1)
user_basic_info_pivot['PCFRRO6_%']= user_basic_info_pivot.apply(lambda x: round(x['PCFRRO_6']/x['sum'],2),axis=1)

# 筛选每个年龄段中手机配置比例最高的20个左右app列表(年龄段中app可能有部分重复)
user_basic_info_select = user_basic_info_pivot[((user_basic_info_pivot['PCFRRO1_%']>= 0.4) 
                                      | (user_basic_info_pivot['PCFRRO2_%']>= 0.3) 
                                      | (user_basic_info_pivot['PCFRRO3_%']>= 0.32)
                                      | (user_basic_info_pivot['PCFRRO4_%']>= 0.32)
                                      | (user_basic_info_pivot['PCFRRO5_%']>= 0.2)
                                      | (user_basic_info_pivot['PCFRRO6_%']>= 0.3)
                                      ) & (user_basic_info_pivot['sum']>10000)].reset_index()[['PCFRRO']]
del user_basic_info_pivot
print('选中PCFRRO数量：',len(user_basic_info_select))
feature_PCFRRO=pd.merge(user_basic_info[['uId','PCFRRO']],user_basic_info_select[['PCFRRO']],how='inner',on='PCFRRO')
del user_basic_info_select
feature_PCFRRO_dummies=pd.get_dummies(feature_PCFRRO['PCFRRO'],prefix='PCFRRO')
feature_PCFRRO=pd.concat([feature_PCFRRO,feature_PCFRRO_dummies],axis=1)
del feature_PCFRRO['PCFRRO']
del feature_PCFRRO_dummies
feature_PCFRRO.to_csv('../../data/features_base/feature_PCFRRO.csv',index=False,encoding='utf8')

del user_basic_info_merge
del train


# 组合特征RRRR
user_basic_info_copy_full  = user_basic_info[['uId','ramCapacity','romCapacity','ramLeftRation','romLeftRation']].copy()
user_basic_info_copy_full['ramLeft']=user_basic_info_copy_full.apply(lambda x :x.ramCapacity*x.ramLeftRation,axis=1)
user_basic_info_copy_full['ramUsed']=user_basic_info_copy_full.apply(lambda x :x.ramCapacity-x.ramLeft,axis=1)

user_basic_info_copy_full['romLeft']=user_basic_info_copy_full.apply(lambda x :x.romCapacity*x.romLeftRation,axis=1)
user_basic_info_copy_full['romUsed']=user_basic_info_copy_full.apply(lambda x :x.romCapacity-x.romLeft,axis=1)

enc=preprocessing.OrdinalEncoder()
user_behavior_info_rank=enc.fit_transform(user_basic_info_copy_full[['ramLeft','ramUsed','romLeft','romUsed']])
user_behavior_info_rank=pd.DataFrame(user_behavior_info_rank,columns=['ramLeft_rank','ramUsed_rank','romLeft_rank','romUsed_rank'])
feature_RRRR=pd.concat([user_basic_info_copy_full[['uId','ramLeft','ramUsed','romLeft','romUsed']],user_behavior_info_rank],axis=1)
del user_basic_info_copy_full
del user_behavior_info_rank


# 拼接
# 为交叉组合特征 字符串转排序特征
from sklearn import preprocessing
enc=preprocessing.OrdinalEncoder()
user_basic_info[['P_C_F','R_R_O','PCFRRO']]=enc.fit_transform(user_basic_info[['P_C_F','R_R_O','PCFRRO']])

# 两个基础特征文件 合并
age_dataSet = pd.merge(user_basic_info, user_behavior_info, how='left', on='uId')

# 独热 性别、运营商、网络接入方式
df_gender = pd.get_dummies(age_dataSet.gender, prefix='gender')
df_ct = pd.get_dummies(age_dataSet.ct, prefix='ct')
df_carrier = pd.get_dummies(age_dataSet.carrier, prefix='carrier')

# 舍去字符串特征
age_dataSet.drop(['gender','ct','carrier','prodName','color','city'], axis=1, inplace=True) 

# 独热特征拼接 
age_dataSet = pd.concat([age_dataSet,df_gender, df_ct, df_carrier], axis=1)
del df_gender, df_ct, df_carrier

# 手机型号_颜色_字体 组合特征13维
age_dataSet = pd.merge(age_dataSet, feature_P_C_F, how='left', on='uId').fillna(0)

# 手机ram_rom_os 组合特征23维
age_dataSet = pd.merge(age_dataSet, feature_R_R_O, how='left', on='uId').fillna(0)

# PCFRRO 组合特征  维
age_dataSet = pd.merge(age_dataSet, feature_PCFRRO, how='left', on='uId').fillna(0)

# RRRR 组合特征  8维
age_dataSet = pd.merge(age_dataSet, feature_RRRR, how='left', on='uId')


age_test = pd.read_csv('../../data/processed_data/age_test.csv')
age_train = pd.read_csv('../../data/processed_data/age_train.csv')

# 训练集
base_train_1 = pd.merge(age_train[['uId']], age_dataSet, how='left', on='uId')

# 测试集
base_test_1 = pd.merge(age_test, age_dataSet, how='left', on='uId')



###############################################
######基础表处理方式一：
######把基础表的各维度特征关联标签，求对应先验概率
###############################################
user_behavior_info = pd.read_csv('../../data/processed_data/user_behavior_info.csv')
user_basic_info = pd.read_csv('../../data/processed_data/user_basic_info.csv')
age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test = pd.read_csv('../../data/processed_data/age_test.csv',dtype={'uId':np.int32})
age_train_usage = pd.read_csv('../../data/processed_data/age_train_usage.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test_usage = pd.read_csv('../../data/processed_data/age_test_usage.csv',dtype={'uId':np.int32})


train = pd.merge(user_behavior_info, user_basic_info, how='left', on='uId')
train = pd.merge(age_train, train, how='left', on='uId')


# prodName特征处理
train_prod = train[['uId', 'prodName', 'age_group']].copy()
train_prod=pd.pivot_table(train_prod, values='uId',index=['prodName'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_prod['sum_prod']=train_prod.sum(axis=1)
train_prod= train_prod.reset_index()
train_prod.rename(columns={1:'age_1_prod',2:'age_2_prod',3:'age_3_prod',4:'age_4_prod',5:'age_5_prod',6:'age_6_prod'},inplace=True)
train_prod['age1_prod_%']= train_prod.apply(lambda x: round(x['age_1_prod']/x['sum_prod'],3),axis=1)
train_prod['age2_prod_%']= train_prod.apply(lambda x: round(x['age_2_prod']/x['sum_prod'],3),axis=1)
train_prod['age3_prod_%']= train_prod.apply(lambda x: round(x['age_3_prod']/x['sum_prod'],3),axis=1)
train_prod['age4_prod_%']= train_prod.apply(lambda x: round(x['age_4_prod']/x['sum_prod'],3),axis=1)
train_prod['age5_prod_%']= train_prod.apply(lambda x: round(x['age_5_prod']/x['sum_prod'],3),axis=1)
train_prod['age6_prod_%']= train_prod.apply(lambda x: round(x['age_6_prod']/x['sum_prod'],3),axis=1)
# train_prod.to_csv('../../data/features_base/prodName_features.csv', index=False, encoding='utf-8')


# city特征处理
train_city = train[['uId', 'city', 'age_group']].copy()
train_city=pd.pivot_table(train_city, values='uId', index=['city'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_city['sum_city']=train_city.sum(axis=1)
train_city= train_city.reset_index()
train_city.rename(columns={1:'age_1_city',2:'age_2_city',3:'age_3_city',4:'age_4_city',5:'age_5_city',6:'age_6_city'},inplace=True)
train_city['age1_city_%']= train_city.apply(lambda x: round(x['age_1_city']/x['sum_city'],3),axis=1)
train_city['age2_city_%']= train_city.apply(lambda x: round(x['age_2_city']/x['sum_city'],3),axis=1)
train_city['age3_city_%']= train_city.apply(lambda x: round(x['age_3_city']/x['sum_city'],3),axis=1)
train_city['age4_city_%']= train_city.apply(lambda x: round(x['age_4_city']/x['sum_city'],3),axis=1)
train_city['age5_city_%']= train_city.apply(lambda x: round(x['age_5_city']/x['sum_city'],3),axis=1)
train_city['age6_city_%']= train_city.apply(lambda x: round(x['age_6_city']/x['sum_city'],3),axis=1)
# train_city.to_csv('../../data/features_base/city_features.csv', index=False, encoding='utf-8')


# color特征处理
train_color = train[['uId', 'color', 'age_group']].copy()
train_color=pd.pivot_table(train_color, values='uId', index=['color'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_color['sum_color']=train_color.sum(axis=1)
train_color= train_color.reset_index()
train_color.rename(columns={1:'age_1_color',2:'age_2_color',3:'age_3_color',4:'age_4_color',5:'age_5_color',6:'age_6_color'},inplace=True)
train_color['age1_color_%']= train_color.apply(lambda x: round(x['age_1_color']/x['sum_color'],3),axis=1)
train_color['age2_color_%']= train_color.apply(lambda x: round(x['age_2_color']/x['sum_color'],3),axis=1)
train_color['age3_color_%']= train_color.apply(lambda x: round(x['age_3_color']/x['sum_color'],3),axis=1)
train_color['age4_color_%']= train_color.apply(lambda x: round(x['age_4_color']/x['sum_color'],3),axis=1)
train_color['age5_color_%']= train_color.apply(lambda x: round(x['age_5_color']/x['sum_color'],3),axis=1)
train_color['age6_color_%']= train_color.apply(lambda x: round(x['age_6_color']/x['sum_color'],3),axis=1)
# train_color.to_csv('../../data/features_base/color_features.csv', index=False, encoding='utf-8')


# fontSize特征处理
train_fontSize = train[['uId', 'fontSize', 'age_group']].copy()
train_fontSize=pd.pivot_table(train_fontSize, values='uId', index=['fontSize'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_fontSize['sum_fontSize']=train_fontSize.sum(axis=1)
train_fontSize= train_fontSize.reset_index()
train_fontSize.rename(columns={1:'age_1_fontSize',2:'age_2_fontSize',3:'age_3_fontSize',4:'age_4_fontSize',5:'age_5_fontSize',6:'age_6_fontSize'},inplace=True)
train_fontSize['age1_fontSize_%']= train_fontSize.apply(lambda x: round(x['age_1_fontSize']/x['sum_fontSize'],3),axis=1)
train_fontSize['age2_fontSize_%']= train_fontSize.apply(lambda x: round(x['age_2_fontSize']/x['sum_fontSize'],3),axis=1)
train_fontSize['age3_fontSize_%']= train_fontSize.apply(lambda x: round(x['age_3_fontSize']/x['sum_fontSize'],3),axis=1)
train_fontSize['age4_fontSize_%']= train_fontSize.apply(lambda x: round(x['age_4_fontSize']/x['sum_fontSize'],3),axis=1)
train_fontSize['age5_fontSize_%']= train_fontSize.apply(lambda x: round(x['age_5_fontSize']/x['sum_fontSize'],3),axis=1)
train_fontSize['age6_fontSize_%']= train_fontSize.apply(lambda x: round(x['age_6_fontSize']/x['sum_fontSize'],3),axis=1)
# train_fontSize.to_csv('../../data/features_base/fontSize_features.csv', index=False, encoding='utf-8')


# romCapacity特征处理
train_romCapacity = train[['uId', 'romCapacity', 'age_group']].copy()
train_romCapacity=pd.pivot_table(train_romCapacity, values='uId', index=['romCapacity'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_romCapacity['sum_romCapacity']=train_romCapacity.sum(axis=1)
train_romCapacity= train_romCapacity.reset_index()
train_romCapacity.rename(columns={1:'age_1_romCapacity',2:'age_2_romCapacity',3:'age_3_romCapacity',4:'age_4_romCapacity',5:'age_5_romCapacity',6:'age_6_romCapacity'},inplace=True)
train_romCapacity['age1_romCapacity_%']= train_romCapacity.apply(lambda x: round(x['age_1_romCapacity']/x['sum_romCapacity'],3),axis=1)
train_romCapacity['age2_romCapacity_%']= train_romCapacity.apply(lambda x: round(x['age_2_romCapacity']/x['sum_romCapacity'],3),axis=1)
train_romCapacity['age3_romCapacity_%']= train_romCapacity.apply(lambda x: round(x['age_3_romCapacity']/x['sum_romCapacity'],3),axis=1)
train_romCapacity['age4_romCapacity_%']= train_romCapacity.apply(lambda x: round(x['age_4_romCapacity']/x['sum_romCapacity'],3),axis=1)
train_romCapacity['age5_romCapacity_%']= train_romCapacity.apply(lambda x: round(x['age_5_romCapacity']/x['sum_romCapacity'],3),axis=1)
train_romCapacity['age6_romCapacity_%']= train_romCapacity.apply(lambda x: round(x['age_6_romCapacity']/x['sum_romCapacity'],3),axis=1)
# train_romCapacity.to_csv('../../data/features_base/romCapacity_features.csv', index=False, encoding='utf-8')
 
    
# ramCapacity特征处理
train_ramCapacity = train[['uId', 'ramCapacity', 'age_group']].copy()
train_ramCapacity=pd.pivot_table(train_ramCapacity, values='uId', index=['ramCapacity'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_ramCapacity['sum_ramCapacity']=train_ramCapacity.sum(axis=1)
train_ramCapacity= train_ramCapacity.reset_index()
train_ramCapacity.rename(columns={1:'age_1_ramCapacity',2:'age_2_ramCapacity',3:'age_3_ramCapacity',4:'age_4_ramCapacity',5:'age_5_ramCapacity',6:'age_6_ramCapacity'},inplace=True)
train_ramCapacity['age1_ramCapacity_%']= train_ramCapacity.apply(lambda x: round(x['age_1_ramCapacity']/x['sum_ramCapacity'],3),axis=1)
train_ramCapacity['age2_ramCapacity_%']= train_ramCapacity.apply(lambda x: round(x['age_2_ramCapacity']/x['sum_ramCapacity'],3),axis=1)
train_ramCapacity['age3_ramCapacity_%']= train_ramCapacity.apply(lambda x: round(x['age_3_ramCapacity']/x['sum_ramCapacity'],3),axis=1)
train_ramCapacity['age4_ramCapacity_%']= train_ramCapacity.apply(lambda x: round(x['age_4_ramCapacity']/x['sum_ramCapacity'],3),axis=1)
train_ramCapacity['age5_ramCapacity_%']= train_ramCapacity.apply(lambda x: round(x['age_5_ramCapacity']/x['sum_ramCapacity'],3),axis=1)
train_ramCapacity['age6_ramCapacity_%']= train_ramCapacity.apply(lambda x: round(x['age_6_ramCapacity']/x['sum_ramCapacity'],3),axis=1)
# train_ramCapacity.to_csv('../../data/features_base/ramCapacity_features.csv', index=False, encoding='utf-8')


# os特征处理
train_os = train[['uId', 'os', 'age_group']].copy()
train_os=pd.pivot_table(train_os, values='uId', index=['os'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_os['sum_os']=train_os.sum(axis=1)
train_os= train_os.reset_index()
train_os.rename(columns={1:'age_1_os',2:'age_2_os',3:'age_3_os',4:'age_4_os',5:'age_5_os',6:'age_6_os'},inplace=True)
train_os['age1_os_%']= train_os.apply(lambda x: round(x['age_1_os']/x['sum_os'],3),axis=1)
train_os['age2_os_%']= train_os.apply(lambda x: round(x['age_2_os']/x['sum_os'],3),axis=1)
train_os['age3_os_%']= train_os.apply(lambda x: round(x['age_3_os']/x['sum_os'],3),axis=1)
train_os['age4_os_%']= train_os.apply(lambda x: round(x['age_4_os']/x['sum_os'],3),axis=1)
train_os['age5_os_%']= train_os.apply(lambda x: round(x['age_5_os']/x['sum_os'],3),axis=1)
train_os['age6_os_%']= train_os.apply(lambda x: round(x['age_6_os']/x['sum_os'],3),axis=1)
# train_os.to_csv('../../data/features_base/os_features.csv', index=False, encoding='utf-8')


# bootTimes特征处理
train_boot = train[['uId', 'bootTimes', 'age_group']].copy()
train_boot=pd.pivot_table(train_boot, values='uId',index=['bootTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_boot['sum_boot']=train_boot.sum(axis=1)
train_boot= train_boot.reset_index()
train_boot.rename(columns={1:'age_1_boot',2:'age_2_boot',3:'age_3_boot',4:'age_4_boot',5:'age_5_boot',6:'age_6_boot'},inplace=True)
train_boot['age1_boot_%']= train_boot.apply(lambda x: round(x['age_1_boot']/x['sum_boot'],3),axis=1)
train_boot['age2_boot_%']= train_boot.apply(lambda x: round(x['age_2_boot']/x['sum_boot'],3),axis=1)
train_boot['age3_boot_%']= train_boot.apply(lambda x: round(x['age_3_boot']/x['sum_boot'],3),axis=1)
train_boot['age4_boot_%']= train_boot.apply(lambda x: round(x['age_4_boot']/x['sum_boot'],3),axis=1)
train_boot['age5_boot_%']= train_boot.apply(lambda x: round(x['age_5_boot']/x['sum_boot'],3),axis=1)
train_boot['age6_boot_%']= train_boot.apply(lambda x: round(x['age_6_boot']/x['sum_boot'],3),axis=1)
# train_boot.to_csv('../../data/features_base/boot_features.csv', index=False, encoding='utf-8')


# AFuncTimes特征处理
train_AFunc = train[['uId', 'AFuncTimes', 'age_group']].copy()
train_AFunc=pd.pivot_table(train_AFunc, values='uId', index=['AFuncTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_AFunc['sum_AFunc']=train_AFunc.sum(axis=1)
train_AFunc= train_AFunc.reset_index()
train_AFunc.rename(columns={1:'age_1_AFunc',2:'age_2_AFunc',3:'age_3_AFunc',4:'age_4_AFunc',5:'age_5_AFunc',6:'age_6_AFunc'},inplace=True)
train_AFunc['age1_AFunc_%']= train_AFunc.apply(lambda x: round(x['age_1_AFunc']/x['sum_AFunc'],3),axis=1)
train_AFunc['age2_AFunc_%']= train_AFunc.apply(lambda x: round(x['age_2_AFunc']/x['sum_AFunc'],3),axis=1)
train_AFunc['age3_AFunc_%']= train_AFunc.apply(lambda x: round(x['age_3_AFunc']/x['sum_AFunc'],3),axis=1)
train_AFunc['age4_AFunc_%']= train_AFunc.apply(lambda x: round(x['age_4_AFunc']/x['sum_AFunc'],3),axis=1)
train_AFunc['age5_AFunc_%']= train_AFunc.apply(lambda x: round(x['age_5_AFunc']/x['sum_AFunc'],3),axis=1)
train_AFunc['age6_AFunc_%']= train_AFunc.apply(lambda x: round(x['age_6_AFunc']/x['sum_AFunc'],3),axis=1)
# train_AFunc.to_csv('../../data/features_base/AFunc_features.csv', index=False, encoding='utf-8')


# BFuncTimes特征处理
train_BFunc = train[['uId', 'BFuncTimes', 'age_group']].copy()
train_BFunc=pd.pivot_table(train_BFunc, values='uId', index=['BFuncTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_BFunc['sum_BFunc']=train_BFunc.sum(axis=1)
train_BFunc= train_BFunc.reset_index()
train_BFunc.rename(columns={1:'age_1_BFunc',2:'age_2_BFunc',3:'age_3_BFunc',4:'age_4_BFunc',5:'age_5_BFunc',6:'age_6_BFunc'},inplace=True)
train_BFunc['age1_BFunc_%']= train_BFunc.apply(lambda x: round(x['age_1_BFunc']/x['sum_BFunc'],3),axis=1)
train_BFunc['age2_BFunc_%']= train_BFunc.apply(lambda x: round(x['age_2_BFunc']/x['sum_BFunc'],3),axis=1)
train_BFunc['age3_BFunc_%']= train_BFunc.apply(lambda x: round(x['age_3_BFunc']/x['sum_BFunc'],3),axis=1)
train_BFunc['age4_BFunc_%']= train_BFunc.apply(lambda x: round(x['age_4_BFunc']/x['sum_BFunc'],3),axis=1)
train_BFunc['age5_BFunc_%']= train_BFunc.apply(lambda x: round(x['age_5_BFunc']/x['sum_BFunc'],3),axis=1)
train_BFunc['age6_BFunc_%']= train_BFunc.apply(lambda x: round(x['age_6_BFunc']/x['sum_BFunc'],3),axis=1)
# train_BFunc.to_csv('../../data/features_base/BFunc_features.csv', index=False, encoding='utf-8')


# CFuncTimes特征处理
train_CFunc = train[['uId', 'CFuncTimes', 'age_group']].copy()
train_CFunc=pd.pivot_table(train_CFunc, values='uId', index=['CFuncTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_CFunc['sum_CFunc']=train_CFunc.sum(axis=1)
train_CFunc= train_CFunc.reset_index()
train_CFunc.rename(columns={1:'age_1_CFunc',2:'age_2_CFunc',3:'age_3_CFunc',4:'age_4_CFunc',5:'age_5_CFunc',6:'age_6_CFunc'},inplace=True)
train_CFunc['age1_CFunc_%']= train_CFunc.apply(lambda x: round(x['age_1_CFunc']/x['sum_CFunc'],3),axis=1)
train_CFunc['age2_CFunc_%']= train_CFunc.apply(lambda x: round(x['age_2_CFunc']/x['sum_CFunc'],3),axis=1)
train_CFunc['age3_CFunc_%']= train_CFunc.apply(lambda x: round(x['age_3_CFunc']/x['sum_CFunc'],3),axis=1)
train_CFunc['age4_CFunc_%']= train_CFunc.apply(lambda x: round(x['age_4_CFunc']/x['sum_CFunc'],3),axis=1)
train_CFunc['age5_CFunc_%']= train_CFunc.apply(lambda x: round(x['age_5_CFunc']/x['sum_CFunc'],3),axis=1)
train_CFunc['age6_CFunc_%']= train_CFunc.apply(lambda x: round(x['age_6_CFunc']/x['sum_CFunc'],3),axis=1)
# train_CFunc.to_csv('../../data/features_base/CFunc_features.csv', index=False, encoding='utf-8')


# DFuncTimes特征处理
train_DFunc = train[['uId', 'DFuncTimes', 'age_group']].copy()
train_DFunc=pd.pivot_table(train_DFunc, values='uId', index=['DFuncTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_DFunc['sum_DFunc']=train_DFunc.sum(axis=1)
train_DFunc= train_DFunc.reset_index()
train_DFunc.rename(columns={1:'age_1_DFunc',2:'age_2_DFunc',3:'age_3_DFunc',4:'age_4_DFunc',5:'age_5_DFunc',6:'age_6_DFunc'},inplace=True)
train_DFunc['age1_DFunc_%']= train_DFunc.apply(lambda x: round(x['age_1_DFunc']/x['sum_DFunc'],3),axis=1)
train_DFunc['age2_DFunc_%']= train_DFunc.apply(lambda x: round(x['age_2_DFunc']/x['sum_DFunc'],3),axis=1)
train_DFunc['age3_DFunc_%']= train_DFunc.apply(lambda x: round(x['age_3_DFunc']/x['sum_DFunc'],3),axis=1)
train_DFunc['age4_DFunc_%']= train_DFunc.apply(lambda x: round(x['age_4_DFunc']/x['sum_DFunc'],3),axis=1)
train_DFunc['age5_DFunc_%']= train_DFunc.apply(lambda x: round(x['age_5_DFunc']/x['sum_DFunc'],3),axis=1)
train_DFunc['age6_DFunc_%']= train_DFunc.apply(lambda x: round(x['age_6_DFunc']/x['sum_DFunc'],3),axis=1)
# train_DFunc.to_csv('../../data/features_base/DFunc_features.csv', index=False, encoding='utf-8')


# EFuncTimes特征处理
train_EFunc = train[['uId', 'EFuncTimes', 'age_group']].copy()
train_EFunc=pd.pivot_table(train_EFunc, values='uId', index=['EFuncTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_EFunc['sum_EFunc']=train_EFunc.sum(axis=1)
train_EFunc= train_EFunc.reset_index()
train_EFunc.rename(columns={1:'age_1_EFunc',2:'age_2_EFunc',3:'age_3_EFunc',4:'age_4_EFunc',5:'age_5_EFunc',6:'age_6_EFunc'},inplace=True)
train_EFunc['age1_EFunc_%']= train_EFunc.apply(lambda x: round(x['age_1_EFunc']/x['sum_EFunc'],3),axis=1)
train_EFunc['age2_EFunc_%']= train_EFunc.apply(lambda x: round(x['age_2_EFunc']/x['sum_EFunc'],3),axis=1)
train_EFunc['age3_EFunc_%']= train_EFunc.apply(lambda x: round(x['age_3_EFunc']/x['sum_EFunc'],3),axis=1)
train_EFunc['age4_EFunc_%']= train_EFunc.apply(lambda x: round(x['age_4_EFunc']/x['sum_EFunc'],3),axis=1)
train_EFunc['age5_EFunc_%']= train_EFunc.apply(lambda x: round(x['age_5_EFunc']/x['sum_EFunc'],3),axis=1)
train_EFunc['age6_EFunc_%']= train_EFunc.apply(lambda x: round(x['age_6_EFunc']/x['sum_EFunc'],3),axis=1)
# train_EFunc.to_csv('../../data/features_base/EFunc_features.csv', index=False, encoding='utf-8')


# FFuncTimes特征处理
train_FFunc = train[['uId', 'FFuncTimes', 'age_group']].copy()
train_FFunc=pd.pivot_table(train_FFunc, values='uId', index=['FFuncTimes'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_FFunc['sum_FFunc']=train_FFunc.sum(axis=1)
train_FFunc= train_FFunc.reset_index()
train_FFunc.rename(columns={1:'age_1_FFunc',2:'age_2_FFunc',3:'age_3_FFunc',4:'age_4_FFunc',5:'age_5_FFunc',6:'age_6_FFunc'},inplace=True)
train_FFunc['age1_FFunc_%']= train_FFunc.apply(lambda x: round(x['age_1_FFunc']/x['sum_FFunc'],3),axis=1)
train_FFunc['age2_FFunc_%']= train_FFunc.apply(lambda x: round(x['age_2_FFunc']/x['sum_FFunc'],3),axis=1)
train_FFunc['age3_FFunc_%']= train_FFunc.apply(lambda x: round(x['age_3_FFunc']/x['sum_FFunc'],3),axis=1)
train_FFunc['age4_FFunc_%']= train_FFunc.apply(lambda x: round(x['age_4_FFunc']/x['sum_FFunc'],3),axis=1)
train_FFunc['age5_FFunc_%']= train_FFunc.apply(lambda x: round(x['age_5_FFunc']/x['sum_FFunc'],3),axis=1)
train_FFunc['age6_FFunc_%']= train_FFunc.apply(lambda x: round(x['age_6_FFunc']/x['sum_FFunc'],3),axis=1)
# train_FFunc.to_csv('../../data/features_base/FFunc_features.csv', index=False, encoding='utf-8')

# FFuncSum特征处理
train_FFuncSum = train[['uId', 'FFuncSum', 'age_group']].copy()
train_FFuncSum=pd.pivot_table(train_FFuncSum, values='uId', index=['FFuncSum'],columns=['age_group'],
                                aggfunc='count', fill_value=0)
train_FFuncSum['sum_FFuncSum']=train_FFuncSum.sum(axis=1)
train_FFuncSum= train_FFuncSum.reset_index()
train_FFuncSum.rename(columns={1:'age_1_FFuncSum',2:'age_2_FFuncSum',3:'age_3_FFuncSum',4:'age_4_FFuncSum',5:'age_5_FFuncSum',6:'age_6_FFuncSum'},inplace=True)
train_FFuncSum['age1_FFuncSum_%']= train_FFuncSum.apply(lambda x: round(x['age_1_FFuncSum']/x['sum_FFuncSum'],3),axis=1)
train_FFuncSum['age2_FFuncSum_%']= train_FFuncSum.apply(lambda x: round(x['age_2_FFuncSum']/x['sum_FFuncSum'],3),axis=1)
train_FFuncSum['age3_FFuncSum_%']= train_FFuncSum.apply(lambda x: round(x['age_3_FFuncSum']/x['sum_FFuncSum'],3),axis=1)
train_FFuncSum['age4_FFuncSum_%']= train_FFuncSum.apply(lambda x: round(x['age_4_FFuncSum']/x['sum_FFuncSum'],3),axis=1)
train_FFuncSum['age5_FFuncSum_%']= train_FFuncSum.apply(lambda x: round(x['age_5_FFuncSum']/x['sum_FFuncSum'],3),axis=1)
train_FFuncSum['age6_FFuncSum_%']= train_FFuncSum.apply(lambda x: round(x['age_6_FFuncSum']/x['sum_FFuncSum'],3),axis=1)
# train_FFuncSum.to_csv('../../data/features_base/FFuncSum_features.csv', index=False, encoding='utf-8')

# 拼接
train = pd.merge(train, train_city, how='left', on='city').fillna(0)
train = pd.merge(train, train_color, how='left', on='color').fillna(0)
train = pd.merge(train, train_fontSize, how='left', on='fontSize').fillna(0)
train = pd.merge(train, train_os, how='left', on='os').fillna(0)
train = pd.merge(train, train_prod, how='left', on='prodName').fillna(0)
train = pd.merge(train, train_ramCapacity, how='left', on='ramCapacity').fillna(0)
train = pd.merge(train, train_romCapacity, how='left', on='romCapacity').fillna(0)

train = pd.merge(train, train_boot, how='left', on='bootTimes').fillna(0)
train = pd.merge(train, train_AFunc, how='left', on='AFuncTimes').fillna(0)
train = pd.merge(train, train_BFunc, how='left', on='BFuncTimes').fillna(0)
train = pd.merge(train, train_CFunc, how='left', on='CFuncTimes').fillna(0)
train = pd.merge(train, train_DFunc, how='left', on='DFuncTimes').fillna(0)
train = pd.merge(train, train_EFunc, how='left', on='EFuncTimes').fillna(0)
train = pd.merge(train, train_FFunc, how='left', on='FFuncTimes').fillna(0)
train = pd.merge(train, train_FFuncSum, how='left', on='FFuncSum').fillna(0)

train = pd.merge(train, special_color, how='left', on='uId').fillna(0)
train = pd.merge(train, special_city, how='left', on='uId').fillna(0)
train = pd.merge(train, special_prodName, how='left', on='uId').fillna(0)

# 删除
train.drop(['gender','color','city','prodName','ct','fontSize','os','ramCapacity','romCapacity','carrier','ramLeftRation','romLeftRation'],axis=1, inplace=True)
train.drop(['bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum'],axis=1, inplace=True)
train.drop(['sum_AFunc','sum_BFunc','sum_CFunc','sum_DFunc','sum_EFunc','sum_FFunc','sum_FFuncSum'],axis=1, inplace=True)
train.drop(['sum_fontSize','sum_romCapacity','sum_ramCapacity','sum_os'],axis=1, inplace=True)


base_train_2 = pd.merge(age_train[['uId']],train,how='inner',on='uId')
base_test_2 = pd.merge(age_test,train,how='inner',on='uId')

base_train = pd.merge(base_train_1,base_train_2, how='left', on='uId').fillna(0)
base_test = pd.merge(base_test_1,base_test_2, how='left', on='uId').fillna(0)

# 存储CSV格式
base_train=memory_preprocess._memory_process(base_train)
base_test=memory_preprocess._memory_process(base_test)
base_train.to_csv('../../data/features/base_train.csv', index=False, encoding='utf-8')
base_test.to_csv('../../data/features/base_test.csv', index=False, encoding='utf-8')

# 存储CSR稀疏矩阵格式
base_train.sort_values('uId', axis=0, ascending=True, inplace=True)
base_test.sort_values('uId', axis=0, ascending=True, inplace=True)

base_train_usage = pd.merge(age_train_usage[['uId']],base_train, how='inner', on='uId').fillna(0)
base_test_usage = pd.merge(age_test_usage,base_test, how='inner', on='uId').fillna(0)

base_train.drop('uId',axis=1,inplace=True)
base_test.drop('uId',axis=1,inplace=True)
base_train_usage.drop('uId',axis=1,inplace=True)
base_test_usage.drop('uId',axis=1,inplace=True)


base_train = csr_matrix(base_train) 
base_test = csr_matrix(base_test)
print(base_train.shape)
print(base_test.shape)
gc.collect()

base_test_usage = csr_matrix(base_train_usage) 
base_test_usage = csr_matrix(base_test_usage)
print(base_train_usage.shape)
print(base_test_usage.shape)
gc.collect()

# 存储全量样本的训练集和测试集
sparse.save_npz('../../data/csr_features_full/base_train.npz', base_train)
sparse.save_npz('../../data/csr_features_full/base_test.npz', base_test)
# base_train = sparse.load_npz('../../data/csr_features_full/base_train.npz')
# base_test = sparse.load_npz('../../data/csr_features_full/base_test.npz')

# 存储缺失样本的训练集和测试集
sparse.save_npz('../../data/csr_features_usage/base_train_usage.npz', base_train_usage)
sparse.save_npz('../../data/csr_features_usage/base_test_usage.npz', base_test_usage)
# base_train_usage = sparse.load_npz('../../data/csr_features_usage/base_train_usage.npz')
# base_test_usage = sparse.load_npz('../../data/csr_features_usage/base_test_usage.npz')