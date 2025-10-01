#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import dot
import glob
from statistics import mean
from pathlib import Path
from tqdm import tqdm
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
import sklearn
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, train_test_split
import datetime
from datetime import date, timedelta
import jieba
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, save_npz, load_npz, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, Trainset, SVD, SVDpp, accuracy, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, NMF, SlopeOne, CoClustering
from surprise.model_selection.validation import cross_validate
from surprise.model_selection.search import GridSearchCV
from surprise.model_selection import KFold
from surprise import PredictionImpossible
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')


# In[2]:


device = pd.read_csv('data/1_device_active.csv')
user = pd.read_csv('data/2_user_info.csv', sep='|')
#music = pd.read_csv('data/3_music_info.csv', sep='|', error_bad_lines=False)
#behaviour = pd.read_csv('data/4_user_behavior.csv', sep='|')
artist = pd.read_csv('data/5_artist_info.csv', sep='|')
device = device.drop(columns=['days'])


# In[7]:


train = device.iloc[:,:31]
train = train.merge(user.drop(columns=['topics']), on='device_id', how='left')
train_1 = train.merge(device[['device_id','day_31']], on='device_id', how='left').drop(columns=['device_id'])
train_2 = train.merge(device[['device_id','day_32']], on='device_id', how='left').drop(columns=['device_id'])
train_3 = train.merge(device[['device_id','day_33']], on='device_id', how='left').drop(columns=['device_id'])
train_7 = train.merge(device[['device_id','day_37']], on='device_id', how='left').drop(columns=['device_id'])
train_14 = train.merge(device[['device_id','day_44']], on='device_id', how='left').drop(columns=['device_id'])
train_30 = train.merge(device[['device_id','day_60']], on='device_id', how='left').drop(columns=['device_id'])


# In[8]:


test = pd.concat((device['device_id'], device.iloc[:,-30:]), axis=1)
rename_dict = {}
for i in range(31, 61):
    rename_dict['day_'+str(i)] = 'day_' + str(i-30)
test = test.rename(columns=rename_dict)
test = test.merge(user.drop(columns=['topics']), on='device_id', how='left')


# In[9]:


cat_cols = ['gender', 'age', 'device', 'city', 'is_vip']
for i in range(1, 31):
    cat_cols.append('day_'+str(i))


# In[10]:


all_trains = {'1': train_1, '2': train_2, '3': train_3,
              '7': train_7, '14': train_14, '30': train_30}


# In[11]:


val_set = []
train_set = []
kf = sklearn.model_selection.KFold(5, shuffle=True, random_state=42)
for tr_index, val_index in kf.split(train_1):
    val_set.append(train_1.iloc[val_index])
    train_set.append(train_1.iloc[tr_index])


# In[77]:


mean_auc = 0.0
for fold in range(5):
    lgb_clf = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1, 
                         n_estimators=500, objective='binary')
    lgb_clf.fit(train_set[fold].drop(columns=['day_31']), train_set[fold]['day_31'],
           eval_set=(val_set[fold].drop(columns=['day_31']), val_set[fold]['day_31']),
           eval_metric='AUC', 
           early_stopping_rounds=50,
           verbose=20, categorical_feature=cat_cols)
    mean_auc += lgb_clf.best_score_['valid_0']['auc']
    if fold == 0:
        #final_preds = cat_clf.predict(test_data.drop(columns=['watch_label']))
        proba_preds = lgb_clf.predict_proba(test.drop(columns='device_id'))
    else:
        #final_preds += cat_clf.predict(test_data.drop(columns=['watch_label']))
        proba_preds += lgb_clf.predict_proba(test.drop(columns='device_id'))
    print('\n')
proba_preds /= 5
print('Mean AUC:', mean_auc/5)


# In[12]:


# All training made here
for day_n, the_train in all_trains.items():
    val_set = []
    train_set = []
    kf = sklearn.model_selection.KFold(5, shuffle=True, random_state=42)
    for tr_index, val_index in kf.split(the_train):
        val_set.append(the_train.iloc[val_index])
        train_set.append(the_train.iloc[tr_index])
    mean_auc = 0.0
    for fold in range(5):
        lgb_clf = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1, 
                             n_estimators=500, objective='binary')
        lgb_clf.fit(train_set[fold].iloc[:,:-1], train_set[fold].iloc[:,-1],
               eval_set=(val_set[fold].iloc[:,:-1], val_set[fold].iloc[:,-1]),
               eval_metric='AUC', 
               early_stopping_rounds=50,
                verbose=50, categorical_feature=cat_cols)
        mean_auc += lgb_clf.best_score_['valid_0']['auc']
        if fold == 0:
            #final_preds = cat_clf.predict(test_data.drop(columns=['watch_label']))
            proba_preds = lgb_clf.predict_proba(test.drop(columns='device_id'))
        else:
            #final_preds += cat_clf.predict(test_data.drop(columns=['watch_label']))
            proba_preds += lgb_clf.predict_proba(test.drop(columns='device_id'))
        print('\n')
    print('Day', day_n,'Mean AUC:', mean_auc/5)
    print('\n')
    proba_preds /= 5
    if day_n == '1':
        submission = pd.concat((test['device_id'], pd.DataFrame(proba_preds[:,1]).rename(columns={0: 'label_'+day_n+'d'})), axis=1)
    else:
        submission = pd.concat((submission, pd.DataFrame(proba_preds[:,1]).rename(columns={0: 'label_'+day_n+'d'})), axis=1)


# In[16]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type) == 'datetime64[ns]':
                continue
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[18]:


submission = reduce_mem_usage(submission)


# In[15]:


submission.to_csv('submisison.csv', index=False)

