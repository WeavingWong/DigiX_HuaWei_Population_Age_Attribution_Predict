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
userAll = pd.read_csv('../../data/original_data/user_app_usage.csv', names=['uId', 'appId', 'duration', 'times', 'use_date'], dtype={'uId':np.int32, 'appId':str, 'duration':np.float32, 
                                                                'times':np.float32, 'use_date':str})
app_info = pd.read_csv('../../data/processed_data/app_info.csv', dtype={'appId':str, 'category':int})
age_train = pd.read_csv('../../data/processed_data/age_train_usage.csv',dtype={'uId':np.int32, 'age_group':np.int8})
age_test = pd.read_csv('../../data/processed_data/age_test_usage.csv',dtype={'uId':np.int32}) 
user_app = pd.read_csv('../../data/processed_data/user_app.csv', dtype={'uId':np.int32, 'appId':str})


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
#########用户30天使用时长和次数维度特征
###############################################
# 用户30天使用app的数量
use_app_num = userAll.groupby(['uId'])['appId'].nunique().reset_index().rename(columns={'appId': 'use_app_nums'})
# 用户30天使用app的总时长和总次数
userSub = userAll.groupby(['uId']).agg({'use_date':'nunique','duration':'sum','times':'sum'}).reset_index().rename(columns={'use_date':'use_days_30','duration': 'use_duration_30','times':'use_times_30',})
userSub['use_days_ratio'] = round(userSub['use_days_30']/30,5)
#分桶
userSub['use_duration_30_cut']=pd.cut(userSub['use_duration_30'],range(0,int(userSub['use_duration_30'].max()),int(userSub['use_duration_30'].max()/10000)),labels=False)
userSub['use_times_30_cut']=pd.cut(userSub['use_times_30'],range(0,int(userSub['use_times_30'].max()),int(userSub['use_times_30'].max()/1000)),labels=False)
# 用户平均每天使用app的时长和次数
userSub['use_duration_average_day'] = userSub['use_duration_30'].apply(lambda x: x/30) 
userSub['use_duration_average_day'] = userSub['use_duration_average_day'].astype('int')
userSub['use_times_average_day'] = userSub['use_times_30'].apply(lambda x: x/30) 
userSub['use_times_average_day'] = userSub['use_times_average_day'].astype('int')
#分桶
userSub['use_duration_average_day_cut']=pd.cut(userSub['use_duration_average_day'],range(0,int(userSub['use_duration_average_day'].max()),int(userSub['use_duration_average_day'].max()/1000)),labels=False)
# 用户30天内平均每个app使用时长和次数
userSub = pd.merge(userSub, use_app_num, how='left', on='uId')
userSub['use_duration_average_app_30'] = userSub['use_duration_30'] / userSub['use_app_nums']
userSub['use_duration_average_app_30'] = userSub['use_duration_average_app_30'].astype('int')
userSub['use_times_average_app_30'] = userSub['use_times_30'] / userSub['use_app_nums']
userSub['use_times_average_app_30'] = userSub['use_times_average_app_30'].astype('int')
del use_app_num 

# 分桶
userSub['use_duration_average_app_30_cut']=pd.cut(userSub['use_duration_average_app_30'],range(0,int(userSub['use_duration_average_app_30'].max()),int(userSub['use_duration_average_app_30'].max()/10000)),labels=False)

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_times_duration = userSub.copy()


###############################################
#########用户30天平均每天使用app数量的均值和方差
###############################################
userSub=userAll.groupby(['uId','use_date'])['appId'].count().reset_index()
userSub=userSub.groupby('uId')['appId'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub=round(userSub)

userSub=memory_preprocess._memory_process(userSub)
use_app_nums_7_features = userSub.copy()


###############################################
#########用户30天平均每天使用app数量的均值和方差
###############################################
# 用户30天使用时长和次数的均值和方差
userSub1=userAll.groupby('uId')['duration'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub1=round(userSub1)
userSub2=userAll.groupby('uId')['times'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub2=round(userSub2)
userSub = pd.merge(userSub1,userSub2,how='left',on='uId')

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_7_features = userSub.copy()



###############################################
#########用户周末使用时长和次数维度特征
###############################################
userSub = userAll[(userAll['use_date'] == '2019-03-02') | (userAll['use_date'] == '2019-03-03') |
                          (userAll['use_date'] == '2019-03-09') | (userAll['use_date'] == '2019-03-10') |
                          (userAll['use_date'] == '2019-03-16') | (userAll['use_date'] == '2019-03-17') |
                          (userAll['use_date'] == '2019-03-23') | (userAll['use_date'] == '2019-03-24')]
# 用户周末使用app的数量
use_app_num = userSub.groupby(['uId'])['appId'].nunique().reset_index().rename(columns={'appId': 'use_app_nums_weekend'})
# 用户周末使用app的总时长和总次数
userSub = userAll.groupby(['uId']).agg({'duration':'sum','times':'sum'}).reset_index().rename(columns={'duration': 'use_duration_weekend','times':'use_times_weekend',})
#分桶
userSub['use_duration_weekend_cut']=pd.cut(userSub['use_duration_weekend'],range(0,int(userSub['use_duration_weekend'].max()),int(userSub['use_duration_weekend'].max()/10000)),labels=False)
userSub['use_times_weekend_cut']=pd.cut(userSub['use_times_weekend'],range(0,int(userSub['use_times_weekend'].max()),int(userSub['use_times_weekend'].max()/1000)),labels=False)
# 用户周末平均每天使用app的时长和次数
userSub['use_duration_average_day_weekend'] = userSub['use_duration_weekend'].apply(lambda x: x/30) 
userSub['use_duration_average_day_weekend'] = userSub['use_duration_average_day_weekend'].astype('int')
userSub['use_times_average_day_weekend'] = userSub['use_times_weekend'].apply(lambda x: x/30) 
userSub['use_times_average_day_weekend'] = userSub['use_times_average_day_weekend'].astype('int')
#分桶
userSub['use_duration_average_day_weekend_cut']=pd.cut(userSub['use_duration_average_day_weekend'],range(0,int(userSub['use_duration_average_day_weekend'].max()),int(userSub['use_duration_average_day_weekend'].max()/1000)),labels=False)
# 用户周末内平均每个app使用时长和次数
userSub = pd.merge(userSub, use_app_num, how='left', on='uId')
userSub['use_duration_average_app_weekend'] = userSub['use_duration_weekend'] / userSub['use_app_nums_weekend']
userSub['use_duration_average_app_weekend'] = userSub['use_duration_average_app_30'].astype('int')
userSub['use_times_average_app_weekend'] = userSub['use_times_weekend'] / userSub['use_app_nums_weekend']
userSub['use_times_average_app_weekend'] = userSub['use_times_average_app_weekend'].astype('int')
del use_app_num 

# 分桶
userSub['use_duration_average_app_weekend_cut']=pd.cut(userSub['use_duration_average_app_weekend'],range(0,int(userSub['use_duration_average_app_weekend'].max()),int(userSub['use_duration_average_app_weekend'].max()/10000)),labels=False)

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_times_duration_weekend = userSub.copy()



###############################################
######## 用户周末每天使用时长和次数的均值和方差
###############################################
userSub = userAll[(userAll['use_date'] == '2019-03-02') | (userAll['use_date'] == '2019-03-03') |
                          (userAll['use_date'] == '2019-03-09') | (userAll['use_date'] == '2019-03-10') |
                          (userAll['use_date'] == '2019-03-16') | (userAll['use_date'] == '2019-03-17') |
                          (userAll['use_date'] == '2019-03-23') | (userAll['use_date'] == '2019-03-24')]
userSub = userSub.groupby(['uId','use_date']).agg({'duration':'sum','times':'sum'}).reset_index()
userSub1=userSub.groupby('uId')['duration'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub1=round(userSub1)
userSub2=userAll.groupby('uId')['times'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub2=round(userSub2)
userSub = pd.merge(userSub1,userSub2,how='left',on='uId')

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_7_features_weekend = userSub.copy()


###############################################
####### 用户周末每天使用时长和次数的均值和方差
###############################################
userSub = userAll[(userAll['use_date'] == '2019-03-02') | (userAll['use_date'] == '2019-03-03') |
                          (userAll['use_date'] == '2019-03-09') | (userAll['use_date'] == '2019-03-10') |
                          (userAll['use_date'] == '2019-03-16') | (userAll['use_date'] == '2019-03-17') |
                          (userAll['use_date'] == '2019-03-23') | (userAll['use_date'] == '2019-03-24')]
userSub = userSub.groupby(['uId','use_date']).agg({'duration':'sum','times':'sum'}).reset_index()
userSub1=userSub.groupby('uId')['duration'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub1=round(userSub1)
userSub2=userAll.groupby('uId')['times'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub2=round(userSub2)
userSub = pd.merge(userSub1,userSub2,how='left',on='uId')

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_7_features_weekend = userSub.copy()



###############################################
######用户周一~周五使用时长和使用次数维度特征
###############################################
userSub = userAll[(userAll['use_date'] != '2019-03-02') | (userAll['use_date'] != '2019-03-03') |
                          (userAll['use_date'] != '2019-03-09') | (userAll['use_date'] != '2019-03-10') |
                          (userAll['use_date'] != '2019-03-16') | (userAll['use_date'] != '2019-03-17') |
                          (userAll['use_date'] != '2019-03-23') | (userAll['use_date'] != '2019-03-24')]
# 用户周末使用app的数量
use_app_num = userSub.groupby(['uId'])['appId'].nunique().reset_index().rename(columns={'appId': 'use_app_nums_weekdays'})
# 用户周末使用app的总时长和总次数
userSub = userAll.groupby(['uId']).agg({'duration':'sum','times':'sum'}).reset_index().rename(columns={'duration': 'use_duration_weekdays','times':'use_times_weekdays',})
#分桶
userSub['use_duration_weekdays_cut']=pd.cut(userSub['use_duration_weekdays'],range(0,int(userSub['use_duration_weekdays'].max()),int(userSub['use_duration_weekdays'].max()/10000)),labels=False)
userSub['use_times_weekdays_cut']=pd.cut(userSub['use_times_weekdays'],range(0,int(userSub['use_times_weekdays'].max()),int(userSub['use_times_weekdays'].max()/1000)),labels=False)
# 用户周末平均每天使用app的时长和次数
userSub['use_duration_average_day_weekdays'] = userSub['use_duration_weekdays'].apply(lambda x: x/30) 
userSub['use_duration_average_day_weekdays'] = userSub['use_duration_average_day_weekdays'].astype('int')
userSub['use_times_average_day_weekdays'] = userSub['use_times_weekdays'].apply(lambda x: x/30) 
userSub['use_times_average_day_weekdays'] = userSub['use_times_average_day_weekdays'].astype('int')
#分桶
userSub['use_duration_average_day_weekdays_cut']=pd.cut(userSub['use_duration_average_day_weekdays'],range(0,int(userSub['use_duration_average_day_weekdays'].max()),int(userSub['use_duration_average_day_weekdays'].max()/1000)),labels=False)
# 用户周末内平均每个app使用时长和次数
userSub = pd.merge(userSub, use_app_num, how='left', on='uId')
userSub['use_duration_average_app_weekdays'] = userSub['use_duration_weekdays'] / userSub['use_app_nums_weekdays']
userSub['use_duration_average_app_weekdays'] = userSub['use_duration_average_app_30'].astype('int')
userSub['use_times_average_app_weekdays'] = userSub['use_times_weekdays'] / userSub['use_app_nums_weekdays']
userSub['use_times_average_app_weekdays'] = userSub['use_times_average_app_weekdays'].astype('int')
del use_app_num 

# 分桶
userSub['use_duration_average_app_weekdays_cut']=pd.cut(userSub['use_duration_average_app_weekdays'],range(0,int(userSub['use_duration_average_app_weekdays'].max()),int(userSub['use_duration_average_app_weekdays'].max()/10000)),labels=False)

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_times_duration_weekdays = userSub.copy()



###############################################
#####用户工作日每天使用时长和次数的均值和方差
###############################################
userSub = userAll[(userAll['use_date'] != '2019-03-02') | (userAll['use_date'] != '2019-03-03') |
                          (userAll['use_date'] != '2019-03-09') | (userAll['use_date'] != '2019-03-10') |
                          (userAll['use_date'] != '2019-03-16') | (userAll['use_date'] != '2019-03-17') |
                          (userAll['use_date'] != '2019-03-23') | (userAll['use_date'] != '2019-03-24')]
userSub = userSub.groupby(['uId','use_date']).agg({'duration':'sum','times':'sum'}).reset_index()
userSub1=userSub.groupby('uId')['duration'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub1=round(userSub1)
userSub2=userAll.groupby('uId')['times'].agg({'mean','min','max','median','std','var'}).reset_index()
userSub2=round(userSub2)
userSub = pd.merge(userSub1,userSub2,how='left',on='uId')

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_7_features_weekdays = userSub.copy()



###############################################
#########用户使用app的时长方差，次数方差
###############################################
userSub = userAll.groupby(['uId', 'use_date']).agg({'duration': 'sum', 'times': 'sum'}).reset_index()
userSub = userSub.groupby(['uId']).agg({'duration': lambda x: list(x), 'times': lambda x: list(x)}).reset_index()
# 用户使用app的时长方差
userSub['duration_var'] = userSub['duration'].apply(lambda x: np.array(x).var())
userSub['duration_var'] = userSub['duration_var'].astype('int')
# 排序特征
userSub['duration_var_rank'] = userSub['duration_var'].rank(ascending=1, method='dense')
# 用户使用app的次数方差
userSub['times_var'] = userSub['times'].apply(lambda x: np.array(x).var())
userSub['times_var'] = userSub['times_var'].astype('int')
# 排序特征
userSub['times_var_rank'] = userSub['times_var'].rank(ascending=1, method='dense')
userSub.drop(['duration', 'times'], axis=1, inplace=True)

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_var = userSub.copy()



###############################################
#########用户使用app类型的时间方差，次数方差
###############################################
userSub = pd.merge(userAll,app_info,how='left',on='appId')
userSub = userSub.groupby(['uId', 'category']).agg({'duration': 'sum', 'times': 'sum'}).reset_index()
userSub = userSub.groupby(['uId']).agg({'duration': lambda x: list(x), 'times': lambda x: list(x)}).reset_index()
# 用户使用app类型的时间方差
userSub['cate_duration_var'] = userSub['duration'].apply(lambda x: np.array(x).var())
userSub['cate_duration_var'] = userSub['cate_duration_var'].astype('int')

# 用户使用app类型的次数方差
userSub['cate_times_var'] = userSub['times'].apply(lambda x: np.array(x).var())
userSub['cate_times_var'] = userSub['cate_times_var'].astype('int')
# 排序特征
userSub['cate_duration_var_rank'] = userSub['cate_duration_var'].rank(ascending=1, method='dense')
userSub['cate_times_var_rank'] = userSub['cate_times_var'].rank(ascending=1, method='dense')
userSub.drop(['duration', 'times'], axis=1, inplace=True)

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_cate_var = userSub.copy()


###############################################
#########用户使用app的最大次数（30天的总使用次数）和 用户使用app的最长时间（30天的总使用时长）
###############################################
userSub = userAll[['uId', 'appId', 'duration', 'times']].copy()
userSub = userSub.groupby(['uId', 'appId']).sum().reset_index()
userSub = userSub.groupby(['uId']).agg({'duration': 'max', 'times': 'max'}).reset_index()
userSub.rename(columns={'duration': 'duration_max_30', 'times': 'times_max_30'}, inplace=True)
# 分桶
userSub['duration_max_30_cut']=pd.cut(userSub['duration_max_30'],range(0,int(userSub['duration_max_30'].max()),int(userSub['duration_max_30'].max()/20000)),labels=False)
userSub['times_max_30_cut']=pd.cut(userSub['times_max_30'],range(0,int(userSub['times_max_30'].max()),int(userSub['times_max_30'].max()/2000)),labels=False)

userSub=memory_preprocess._memory_process(userSub)
user_app_usage_max = userSub.copy()


###############################################
#########用户最常用的app是什么？根据30天使用的天数来计算
###############################################
# 最常用的app中用户有用了多少个
userSub = userAll.groupby(['uId','appId'])['use_date'].count().reset_index()
userSub.rename(columns={'use_date': 'use_days'}, inplace=True)
userSub = userSub.sort_values(['uId', 'use_days'],ascending = False)
userSub['sort_num'] = userSub['use_days'].groupby(userSub['uId']).rank(ascending=0, method='first')

userSub = userSub[(userSub.use_days>=20)].groupby('uId').agg({'sort_num':'max'}).reset_index()
userSub.rename(columns={'sort_num': 'often_use_app_nums'}, inplace=True)
userSub['often_use_app_nums'] = userSub['often_use_app_nums'].astype('int')

userSub=memory_preprocess._memory_process(userSub)
often_use_app_nums = userSub.copy()


###############################################
####使用人数最多的top200个app的使用时间和次数，天数
###############################################
app_use_top_200 = userAll.groupby(['appId'])['uId'].nunique().reset_index().sort_values(['uId'],ascending = False)[['appId']].head(200)
userSub = pd.merge(userAll, app_use_top_200,how='inner',on='appId')
userSub = userSub.groupby(['uId','appId']).agg({'use_date':'count','duration':'sum','times':'sum'}).reset_index()

app_use_days_top_200 = pd.pivot_table(userSub, values='use_date', index=['uId'],columns=['appId'], aggfunc=lambda x:x, fill_value=0).reset_index()
app_use_days_top_200=memory_preprocess._memory_process(app_use_days_top_200)

app_use_times_top_200 = pd.pivot_table(userSub, values='times', index=['uId'],columns=['appId'], aggfunc=lambda x:x, fill_value=0).reset_index()
app_use_times_top_200=memory_preprocess._memory_process(app_use_times_top_200)

app_use_duration_top_200 = pd.pivot_table(userSub, values='duration', index=['uId'],columns=['appId'], aggfunc=lambda x:x, fill_value=0).reset_index()
app_use_duration_top_200=memory_preprocess._memory_process(app_use_duration_top_200)



###############################################
#########用户使用最长时间的app的应用类型和对应app的使用时间
###############################################
userSub = userAll[['uId', 'appId', 'duration']].copy()
userSub = userSub.groupby(['uId', 'appId']).sum().reset_index()
userSub['sort_num'] = userSub['duration'].groupby(userSub['uId']).rank(ascending=0, method='first')
userSub_1 = userSub[(userSub.sort_num == 1)].copy()
userSub_2 = userSub[(userSub.sort_num == 2)].copy()
userSub_1.rename(columns={'duration': 'duration_max_app'}, inplace=True)
userSub_2.rename(columns={'duration': 'duration_second_app'}, inplace=True)

# 分桶
userSub_1['duration_max_app_cut']=pd.cut(userSub_1['duration_max_app'],range(0,int(userSub_1['duration_max_app'].max()),int(userSub_1['duration_max_app'].max()/20000)),labels=False)
userSub_2['duration_second_app_cut']=pd.cut(userSub_2['duration_second_app'],range(0,int(userSub_2['duration_second_app'].max()),int(userSub_2['duration_second_app'].max()/2000)),labels=False)

# 拼接原始的app_info表
userSub_1 = pd.merge(userSub_1, app_info.drop_duplicates(subset=['appId'], keep='first'), how='left',on='appId').fillna(method='pad')
userSub_1.drop(['sort_num', 'appId'], axis=1, inplace=True)
userSub_2 = pd.merge(userSub_2, app_info.drop_duplicates(subset=['appId'], keep='first'), how='left',on='appId').fillna(method='pad')
userSub_2.drop(['sort_num', 'appId'], axis=1, inplace=True)
userSub_1.rename(columns={'category': 'cate_dura_max'}, inplace=True)
userSub_1['cate_dura_max'] = userSub_1['cate_dura_max'].fillna(method='pad').astype('int')
userSub_2.rename(columns={'category': 'cate_dura_second'}, inplace=True)
userSub_2['cate_dura_second'] = userSub_2['cate_dura_second'].fillna(method='pad').astype('int')
userSub = pd.merge(userSub_1, userSub_2, how='left', on='uId')
del userSub_1, userSub_2
gc.collect()

userSub=memory_preprocess._memory_process(userSub)
use_app_cate_max_duration = userSub.copy()


###############################################
#########用户使用app的时长方差，次数方差
###############################################

userSub = userAll[['uId', 'appId', 'times']].copy()
userSub = userSub.groupby(['uId', 'appId']).sum().reset_index()
userSub['sort_num'] = userSub['times'].groupby(userSub['uId']).rank(ascending=0, method='first')
userSub_1 = userSub[(userSub.sort_num == 1)].copy()
userSub_2 = userSub[(userSub.sort_num == 2)].copy()
userSub_1.rename(columns={'times': 'times_max_app'}, inplace=True)
userSub_2.rename(columns={'times': 'times_second_app'}, inplace=True)

userSub_1 = pd.merge(userSub_1, app_info.drop_duplicates(subset=['appId'], keep='first'), how='left', on='appId').fillna(method='pad')
userSub_1.drop(['sort_num', 'appId'], axis=1, inplace=True)
userSub_2 = pd.merge(userSub_2, app_info.drop_duplicates(subset=['appId'], keep='first'), how='left', on='appId').fillna(method='pad')
userSub_2.drop(['sort_num', 'appId'], axis=1, inplace=True)
userSub_1.rename(columns={'category': 'cate_times_max'}, inplace=True)
userSub_2.rename(columns={'category': 'cate_times_second'}, inplace=True)
userSub_1['cate_times_max'] = userSub_1['cate_times_max'].fillna(method='pad').astype('int')
userSub_2['cate_times_second'] = userSub_2['cate_times_second'].fillna(method='pad').astype('int')
userSub = pd.merge(userSub_1, userSub_2, how='left', on='uId')
del userSub_1, userSub_2
gc.collect()

userSub=memory_preprocess._memory_process(userSub)
use_max_times_average_app_cate = userSub.copy()


###############################################
#########用户使用app的时长方差，次数方差
###############################################
userSub = userAll.groupby(['uId', 'appId']).agg({'duration':'sum', 'times':'sum'}).reset_index()
userSub['use_max_time_average'] = userSub['duration']/userSub['times']
userSub['use_max_time_average'] = userSub['use_max_time_average'].astype('int')
userSub['use_max_time_average_rank'] = userSub['use_max_time_average'].rank(ascending=1, method='dense')

userSub = userSub[['uId', 'appId', 'use_max_time_average']].sort_values(by=['uId', 'use_max_time_average'],ascending=False)
userSub = userSub.drop_duplicates(subset=['uId'], keep='first')
userSub = pd.merge(userSub, app_info.drop_duplicates(subset=['appId'], keep='first'), on=['appId'], how='left').fillna(method='pad')
userSub.drop('appId', axis=1, inplace=True)
userSub.rename(columns={'category': 'cate_dura_average_max'}, inplace=True)
userSub['cate_dura_average_max'] = userSub['cate_dura_average_max'].fillna(method='pad').astype('int')

userSub=memory_preprocess._memory_process(userSub)
use_max_duration_average_app_cate = userSub.copy()


###############################################
#########用户使用app的时长方差，次数方差
###############################################
userSub = userAll[['uId', 'appId']].drop_duplicates()
userSub = pd.merge(userSub, app_info, on=['appId'], how='left').fillna(method='pad')
userSub = userSub.groupby(['uId'])['category'].nunique().reset_index()
userSub.rename(columns={'category': 'user_cate_nums_30'}, inplace=True)

userSub=memory_preprocess._memory_process(userSub)
user_cate_nums_30 = userSub.copy()


###############################################
#########用户对各app类型的使用时长，使用次数，对应40维+40维+40维
###############################################
userSub = userAll.groupby(['uId', 'appId']).agg({'duration':'sum', 'times':'sum'}).reset_index()
userSub = pd.merge(userSub, app_info, how='left',on='appId').fillna(method='pad')
# 每个类型的使用人数
userSub_1 = pd.pivot_table(userSub, values='appId', index=['uId'],columns=['category'], aggfunc='count', fill_value=0).reset_index()
userSub.drop('appId',axis=1,inplace=True)
userSub = userSub.groupby(['uId', 'category'], as_index=False).sum().reset_index()
# 每个类型的使用时长和次数
userSub_2 = pd.pivot_table(userSub, values='duration', index=['uId'],columns=['category'], aggfunc=np.sum, fill_value=0).reset_index()
userSub_3 = pd.pivot_table(userSub, values='times', index=['uId'],columns=['category'], aggfunc=np.sum, fill_value=0).reset_index()
userSub = pd.merge(userSub_1, userSub_2, how='left',on='uId').fillna(0.0)
userSub = pd.merge(userSub, userSub_3, how='left',on='uId').fillna(0.0)
del userSub_1,userSub_2,userSub_3
gc.collect()

userSub=memory_preprocess._memory_process(userSub)
user_cate_dura_times = userSub.copy()


###############################################
#########usage表app的权重概率
###############################################
# 计算每个app的目标客户年龄指数
age_train = pd.read_csv('../../data/processed_data/age_train.csv',dtype={'uId':np.int32, 'age_group':np.int8})
userSub = pd.merge(age_train, userAll[['uId','appId']].drop_duplicates(), how='inner', on='uId')

userSub=pd.pivot_table(userSub, values='uId', index=['appId'],columns=['age_group'],aggfunc='count', fill_value=0)
userSub['sum']=userSub.sum(axis=1)
userSub= userSub.reset_index()
userSub.rename(columns={1:'age_1',2:'age_2',3:'age_3',4:'age_4',5:'age_5',6:'age_6'},inplace=True)
userSub.drop(axis=0, index=0, inplace=True)

userSub = userSub[(userSub['sum']>=50)].copy()

userSub['age1_%']= userSub.apply(lambda x: round(x['age_1']/x['sum'],4),axis=1)
userSub['age2_%']= userSub.apply(lambda x: round(x['age_2']/x['sum'],4),axis=1)
userSub['age3_%']= userSub.apply(lambda x: round(x['age_3']/x['sum'],4),axis=1)
userSub['age4_%']= userSub.apply(lambda x: round(x['age_4']/x['sum'],4),axis=1)
userSub['age5_%']= userSub.apply(lambda x: round(x['age_5']/x['sum'],4),axis=1)
userSub['age6_%']= userSub.apply(lambda x: round(x['age_6']/x['sum'],4),axis=1)

userSub=pd.merge(userAll[['uId','appId']].drop_duplicates(),userSub,how='left',on='appId')
age_features_ratio = userSub[['uId']].drop_duplicates()
age_features_use_nums = userSub[['uId']].drop_duplicates()
for i in range(1,7):
    age_features=userSub.groupby('uId')['age'+str(i)+'_%'].agg({'mean','min','max','median','std','var'}).reset_index()
    age_features['sqrt_of_mean_age'+str(i)+'_%']=age_features['mean']**2
    age_features.rename({'mean':'mean_'+str(i)+'_%','min':'min_'+str(i)+'_%','max':'max_'+str(i)+'_%','median':'median_'+str(i)+'_%','std':'std_'+str(i)+'_%','var':'var_'+str(i)+'_%'},axis=1,inplace=True)
    age_features_ratio = pd.merge(age_features_ratio,age_features,how='left',on='uId')
    
    age_features=userSub.groupby('uId')['age_'+str(i)].agg({'mean','min','max','median','std','var'}).reset_index()
    age_features = age_features.apply(np.floor)
    age_features.rename({'mean':'mean_'+str(i),'min':'min_'+str(i),'max':'max_'+str(i),'median':'median_'+str(i),'std':'std_'+str(i),'var':'var_'+str(i)},axis=1,inplace=True)
    age_features_use_nums = pd.merge(age_features_use_nums,age_features,how='left',on='uId')
feature_usage_app_age_weight = pd.merge(age_features_ratio,age_features_use_nums,how='left',on='uId')
del age_features_ratio,age_features_use_nums,age_features
gc.collect()

feature_usage_app_age_weight =memory_preprocess._memory_process(feature_usage_app_age_weight )


###############################################
#########各维度特征拼接
###############################################

userSub = pd.merge(user_app_usage_times_duration, user_app_usage_max, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, user_app_usage_var, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, user_app_usage_cate_var, how='left', on='uId').fillna(0)

userSub = pd.merge(userSub, user_cate_dura_times, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, user_cate_nums_30, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, use_max_times_average_app_cate, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, use_max_duration_average_app_cate, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, use_app_cate_max_duration, how='left', on='uId').fillna(0)

userSub = pd.merge(userSub, user_app_usage_times_duration_weekend, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, user_app_usage_times_duration_weekdays, how='left', on='uId').fillna(0)

userSub = pd.merge(userSub, app_use_days_top_200, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, app_use_duration_top_200, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, app_use_times_top_200, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, often_use_app_nums, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, feature_usage_app_age_weight, how='left', on='uId').fillna(0)

userSub = pd.merge(userSub, user_app_usage_7_features, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, user_app_usage_7_features_weekend, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, user_app_usage_7_features_weekdays, how='left', on='uId').fillna(0)
userSub = pd.merge(userSub, use_app_nums_7_features, how='left', on='uId').fillna(0)

# 周末和工作日总的使用时长和次数比值
userSub['weekend_weekdays_duration_ratio'] = userSub.apply(lambda x: x['use_duration_weekend'] if(x['use_duration_weekdays'] == 0) else x['use_duration_weekend']/x['use_duration_weekdays'],axis=1)
userSub['weekend_weekdays_times_ratio'] = userSub.apply(lambda x:x['use_times_weekend'] if(x['use_times_weekdays'] == 0) else x['use_times_weekend']/x['use_times_weekdays'],axis=1)

# 周末和工作日平均每天的使用时长和次数比值
userSub['weekend_weekdays_duration_average_day_ratio'] = userSub.apply(lambda x: x['use_duration_average_day_weekend'] if(x['use_duration_average_day_weekdays'] == 0) else x['use_duration_average_day_weekend']/x['use_duration_average_day_weekdays'], axis=1)
userSub['weekend_weekdays_times_average_day_ratio'] = userSub.apply(lambda x: x['use_times_average_day_weekend'] if(x['use_times_average_day_weekdays'] == 0) else x['use_times_average_day_weekend']/x['use_times_average_day_weekdays'], axis=1)

# 周末和工作日平均每个app的使用时长和次数比值
userSub['weekend_weekdays_duration_average_app_ratio'] = userSub.apply(lambda x: x['use_duration_average_app_weekend'] if(x['use_duration_average_app_weekdays']==0) else x['use_duration_average_app_weekend']/x['use_duration_average_app_weekdays'], axis=1)
userSub['weekend_weekdays_times_average_app_ratio'] = userSub.apply(lambda x: x['use_times_average_app_weekend'] if(x['use_times_average_app_weekdays']==0) else x['use_times_average_app_weekend']/x['use_times_average_app_weekdays'], axis=1)

# 周末和工作日分别占30天总的使用时长比值
userSub['weekend_30_use_duration_ratio'] = userSub.apply(lambda x: x['use_duration_weekend'] if(x['use_duration_30']==0) else x['use_duration_weekend']/x['use_duration_30'], axis=1)
userSub['weekdays_30_use_duration_ratio'] = userSub.apply(lambda x: x['use_duration_weekdays'] if(x['use_duration_30']==0) else x['use_duration_weekdays']/x['use_duration_30'], axis=1)

# 周末和工作日分别占30天总的使用app数量比值
userSub['weekend_30_use_app_nums_ratio'] = userSub.apply(lambda x: x['use_app_nums_weekend'] if(x['use_app_nums']==0) else x['use_app_nums_weekend']/x['use_app_nums'], axis=1)
userSub['weekdays_30_use_app_nums_ratio'] = userSub.apply(lambda x: x['use_app_nums_weekdays'] if(x['use_app_nums'] == 0) else x['use_app_nums_weekdays']/x['use_app_nums'], axis=1)

# 平均每个app每天使用时长
userSub['use_ratio_average_app_on_one_day'] = userSub['use_duration_average_app_30'].apply(lambda x: x/(1440*30))
userSub['use_ratio_max_app_on_day'] = userSub['duration_max_app'].apply(lambda x: x/(1440*30))
userSub['use_ratio_second_app_on_day'] = userSub['duration_second_app'].apply(lambda x: x/(1440*30))

# 存储CSV格式
userSub=memory_preprocess._memory_process(userSub)
userSub.to_csv('../../data/features/usage_features_all.csv', index=False, encoding='utf-8')

###############################################
#########转成csr稀疏矩阵形式存储
###############################################

userSub.sort_values('uId', axis=0, ascending=True, inplace=True)

usage_features_all_train = pd.merge(age_train_usage[['uId']],userSub, how='inner', on='uId').fillna(0)
usage_features_all_test = pd.merge(age_test_usage,userSub, how='inner', on='uId').fillna(0)

usage_features_all_train.drop('uId',axis=1,inplace=True)
usage_features_all_test.drop('uId',axis=1,inplace=True)

usage_features_all_train = csr_matrix(usage_features_all_train) 
usage_features_all_test = csr_matrix(usage_features_all_test)
print(usage_features_all_train.shape)
print(usage_features_all_test.shape)
gc.collect()

# 存储缺失样本的训练集和测试集
sparse.save_npz('../../data/csr_features_usage/usage_features_train.npz', usage_features_all_train)
sparse.save_npz('../../data/csr_features_usage/usage_features_test.npz', usage_features_all_test)
# usage_features_all_train = sparse.load_npz('../../data/csr_features_usage/usage_features_train.npz')
# usage_features_all_test = sparse.load_npz('../../data/csr_features_usage/usage_features_test.npz')
