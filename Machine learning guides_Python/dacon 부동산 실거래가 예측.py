# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:21:37 2020

@author: Administrator
"""
print(list(set(train['city'])))
print(list(set(train['dong'])))
# In[ ]
city=pd.DataFrame(set(train['city']))
dong=pd.DataFrame(set(train['dong']))

a=pd.DataFrame()
a['city']=train['city']
a['dong']=train['dong']

at=pd.DataFrame()
at['city']=test['city']
at['dong']=test['dong']
# In[ ]
a['city_num']=a['city'].apply(lambda x : '00' if x=='서울특별시' else '01')
at['city_num']=at['city'].apply(lambda x : '00' if x=='서울특별시' else '01')

# In[ ]
for x in range(len(dong)):
    dong['dong_num']=str(x).zfill(3)
for x in range(len(dong)):
    dong['dong_num'][x]=str(x).zfill(3)
    
# In[ ]
dong['dong']=dong[0]
c=pd.merge(a, dong, on="dong")
ct=pd.merge(at, dong, on="dong")

# In[ ]
c['num']=0
for x in range(len(c)):
    c['num'][x]=str(c['city_num'][x])+str(c['dong_num'][x])
    
ct['num']=0
for x in range(len(ct)):
    ct['num']=[]
    ct['num'][x]=str(ct['city_num'][x])+str(ct['dong_num'][x])
# In[ ]
train[key] = c['num']
test[key] = ct['num']

# In[ ]
import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
# %matplotlib inline

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import gc
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge

train = pd.read_csv("./Apartment/train.csv")
test = pd.read_csv("./Apartment/test.csv")
test_idx = test['key']

'''
데이터 전처리 준비작업 인천 수정 Validation 구축 : 
Test가 가장 마지막 거래로만 이루어져 있어서, 
실제 제출 전 점수를 평가 할 validation도 비슷하게 구축. 
(주의 : 샘플링에 의해 Validation Score는 달라지기 때문에 동일한 샘플링 기법을 적용한것 끼리 비교해야 함 !!!) 
인천의 경우 서울보다는 부산에 가까워서 city를 부산으로 수정
'''

# In[ ]
test.loc[test['key']==1503614,'city'] = 0

'''
데이터가 시간의 순서대로 이루어져 있어서 Merge과정에서 순서가 깨지지 않도록 index컬럼을 생성해서 sort작업을 진행해줄것임.

이를 안해주면 fold에서 다른 cv값이 나옴
'''

# In[ ]
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

train_len = train.shape[0]
df_all = pd.concat([train,test])

index = []
for i in range(0,df_all.shape[0]):
    index.append(i)
df_all['index'] = index

train = df_all[:train_len].reset_index(drop=True)
df_test = df_all[train_len:].reset_index(drop=True)

df_train_busan = train[train['city']==0]
df_test_busan = df_test[df_test['city']==0]
df_train_seoul = train[train['city']==1]
df_test_seoul = df_test[df_test['city']==1]

'''
신뢰성 있는 Validation 구축을 위해서 아파트 별로 가장 마지막 거래를 Validation으로 추출
'''

# In[ ]
busan_val_idx = df_train_busan.groupby(['apartment_id']).tail(1).index
busan_valid = df_train_busan.loc[busan_val_idx,:]
busan_valid = busan_valid[['key','transaction_real_price']]

seoul_val_idx = df_train_seoul.groupby(['apartment_id']).tail(1).index
seoul_valid = df_train_seoul.loc[seoul_val_idx,:]
seoul_valid = seoul_valid[['key','transaction_real_price']]

del df_train_busan,df_train_seoul,df_test_busan,df_test_seoul
del df_test,train,df_all

'''
데이터 전처리 트레인은 날짜가 고른 반면, 
테스트는 2018년도 6월 이후가 압도적으로 많음. 
샘플링 작업이 필요. 

방과 화장실 0인 값 대체 : 
동일한 아파트에서 비슷한 크기에 값이 존재하면 그로 채워넣고, 
그렇지 않으면 비슷한 크기에서 median으로 채워 넣음. 

방과 화장실 결측치 대체 : 
동일한 아파트에서 비슷한 크기에 값이 존재하면 그로 채워넣고, 
그렇지 않으면 비슷한 크기에서 median으로 채워 넣음. 
주차장의 결측치는 0으로 대체 난방과 현관구조는 None으로 대체
'''

# In[ ]
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_idx = test['key']
test.loc[test['key']==1503614,'city'] = 0

train_up1 = train[train['transaction_year_month']&gt;201806]
train_up1['transaction_real_price'] = train_up1['transaction_real_price'] + 10000000

train_up2 = train[train['transaction_year_month']&gt;201806]
train_up2['transaction_real_price'] = train_up2['transaction_real_price'] + 5000000

train = pd.concat([train,train_up1])
train = pd.concat([train,train_up2])
train = train.reset_index(drop=True)
del train_up1,train_up2
gc.collect()

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

train_len = train.shape[0]
df_all = pd.concat([train,test])

index = []
for i in range(0,df_all.shape[0]):
    index.append(i)
df_all['index'] = index

# In[ ]
### 0인 값 대체
df_all.loc[(df_all['apartment_id']==2805) & (df_all['supply_area'] &gt; 90),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==2805) & (df_all['supply_area'] &gt; 90),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==895) & (df_all['supply_area'] &gt; 137),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==895) & (df_all['supply_area'] &gt; 137),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==903) & (df_all['supply_area'] &gt; 135),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==903) & (df_all['supply_area'] &gt; 135),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==1622) & (df_all['supply_area'] == 127.07),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==1622) & (df_all['supply_area'] == 127.07),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==14029) & (df_all['supply_area'] &gt; 100),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==14029) & (df_all['supply_area'] &gt; 100),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==14029) & (df_all['supply_area'] &lt; 100),'room_count'] = 2
df_all.loc[(df_all['apartment_id']==14029) & (df_all['supply_area'] &lt; 100),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==12067) & (df_all['supply_area'] &gt;= 95),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==12067) & (df_all['supply_area'] &gt;= 95),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==12067) & (df_all['supply_area'] == 92),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==12067) & (df_all['supply_area'] == 92),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==12067) & (df_all['supply_area'] &lt; 90),'room_count'] = 2
df_all.loc[(df_all['apartment_id']==12067) & (df_all['supply_area'] &lt; 90),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==360) & (df_all['supply_area'] == 189.99),'room_count'] = 5
df_all.loc[(df_all['apartment_id']==360) & (df_all['supply_area'] == 189.99),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==360) & (df_all['supply_area'] == 154.46),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==360) & (df_all['supply_area'] == 154.46),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==568) & (df_all['supply_area']//10 == 11.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==568) & (df_all['supply_area']//10 == 11.0),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==1543) & (df_all['supply_area'] &gt; 150 ),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==1543) & (df_all['supply_area'] &gt; 150),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==618) & (df_all['supply_area'] == 92.94),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==618) & (df_all['supply_area'] == 92.94),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==618) & (df_all['supply_area'] == 110.57),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==618) & (df_all['supply_area'] == 110.57),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==4368) & (df_all['supply_area'] &gt; 90),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==4368) & (df_all['supply_area'] &gt; 90),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==4368) & (df_all['supply_area'] &lt; 90),'room_count'] = 2
df_all.loc[(df_all['apartment_id']==4368) & (df_all['supply_area'] &lt; 90),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==3701) & (df_all['supply_area'] == 148.55),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==3701) & (df_all['supply_area'] == 148.55),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==346) & (df_all['supply_area'] &gt; 100),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==346) & (df_all['supply_area'] &gt; 100),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==1524) & (df_all['supply_area'] == 104.39),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==1524) & (df_all['supply_area'] == 104.39),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==1524) & (df_all['supply_area'] == 175.60),'room_count'] = 5
df_all.loc[(df_all['apartment_id']==1524) & (df_all['supply_area'] == 175.60),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 9.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 9.0),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 10.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 10.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 13.0),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 13.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 14.0),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area']//10 == 14.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==1112) & (df_all['supply_area']//10 == 7.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==1112) & (df_all['supply_area']//10 == 7.0),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==65) & (df_all['supply_area']//10 == 11.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==65) & (df_all['supply_area']//10 == 11.0),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==541) & (df_all['supply_area']//10 == 8.0),'room_count'] = 2
df_all.loc[(df_all['apartment_id']==541) & (df_all['supply_area']//10 == 8.0),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==184) & (df_all['supply_area'] == 66.12),'room_count'] = 2
df_all.loc[(df_all['apartment_id']==184) & (df_all['supply_area'] == 66.12),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==2601) & (df_all['supply_area'] == 104.97),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==2601) & (df_all['supply_area'] == 104.97),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==6161) & (df_all['supply_area'] == 99.91),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==6161) & (df_all['supply_area'] == 99.91),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==3685) & (df_all['supply_area'] == 115.70),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==3685) & (df_all['supply_area'] == 115.70),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==21288) & (df_all['supply_area'] == 116.03),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==21288) & (df_all['supply_area'] == 116.03),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==10636) & (df_all['supply_area'] == 112.40),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==10636) & (df_all['supply_area'] == 112.40),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==1162) & (df_all['supply_area'] == 154.71),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==1162) & (df_all['supply_area'] == 154.71),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==10989) & (df_all['supply_area'] == 110.51),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==10989) & (df_all['supply_area'] == 110.51),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==11096) & (df_all['supply_area'] == 97.09),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==11096) & (df_all['supply_area'] == 97.09),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==184) & (df_all['supply_area'] == 69.42),'room_count'] = 2
df_all.loc[(df_all['apartment_id']==184) & (df_all['supply_area'] == 69.42),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==534) & (df_all['supply_area'] //10 == 11.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==534) & (df_all['supply_area'] //10 == 11.0),'bathroom_count'] = 1

df_all.loc[(df_all['apartment_id']==17384) & (df_all['supply_area'] //10 == 11.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==17384) & (df_all['supply_area'] //10 == 11.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area'] //10 == 10.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==431) & (df_all['supply_area'] //10 == 10.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==4058) & (df_all['supply_area'] //10 == 9.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==4058) & (df_all['supply_area'] //10 == 9.0),'bathroom_count'] = 2

#df_all.loc[(df_all['apartment_id']==1388) & (df_all['room_count']==0)]
df_all.loc[(df_all['apartment_id']==1388) & (df_all['supply_area'] //10 == 14.0),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==1388) & (df_all['supply_area'] //10 == 14.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==7136) & (df_all['supply_area'] //10 == 7.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==7136) & (df_all['supply_area'] //10 == 7.0),'bathroom_count'] = 1

#df_all.loc[(df_all['apartment_id']==18737)]
df_all.loc[(df_all['apartment_id']==18737) & (df_all['supply_area'] //10 == 17.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==18737) & (df_all['supply_area'] //10 == 17.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==18741) & (df_all['supply_area'] &gt;= 160),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==18741) & (df_all['supply_area'] &gt;= 160),'bathroom_count'] = 2

#df_all.loc[(df_all['apartment_id']==18732)]

df_all.loc[(df_all['apartment_id']==18732) & (df_all['supply_area'] //10 == 11.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==18732) & (df_all['supply_area'] //10 == 11.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==18732) & (df_all['supply_area'] //10 == 18.0),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==18732) & (df_all['supply_area'] //10 == 18.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==360) & (df_all['supply_area'] //10 == 16.0),'room_count'] = 4
df_all.loc[(df_all['apartment_id']==360) & (df_all['supply_area'] //10 == 16.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==8460) & (df_all['supply_area'] //10 == 8.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==8460) & (df_all['supply_area'] //10 == 8.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==6175) & (df_all['supply_area'] &gt; 290),'room_count'] = 5
df_all.loc[(df_all['apartment_id']==6175) & (df_all['supply_area'] &gt; 290),'bathroom_count'] = 3

df_all.loc[(df_all['apartment_id']==6232)]
df_all.loc[(df_all['apartment_id']==6232) & (df_all['supply_area'] //10 == 19.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==6232) & (df_all['supply_area'] //10 == 19.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==15502) & (df_all['supply_area'] //10 == 10.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==15502) & (df_all['supply_area'] //10 == 10.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==568) & (df_all['supply_area'] //10 == 19.0),'room_count'] = 5
df_all.loc[(df_all['apartment_id']==568) & (df_all['supply_area'] //10 == 19.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==16837) & (df_all['supply_area'] //10 == 9.0),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==16837) & (df_all['supply_area'] //10 == 9.0),'bathroom_count'] = 2

df_all.loc[(df_all['apartment_id']==37468) & (df_all['supply_area']  &lt;= 200),'room_count'] = 3
df_all.loc[(df_all['apartment_id']==37468) & (df_all['supply_area']  &lt;= 200),'bathroom_count'] = 2

'''
결측치 대체 방, 화장실은 위와 동일한 방식으로 채워넣음. 
주차장의 경우 0으로 대체. 
dacon에 물어본 결과 결측치는 0이라고 했음. 
히트 및 현관의 결측치는 None으로 대체.
'''

# In[ ]
### 방과 화장실 결측치
df_all.loc[df_all['apartment_id'] == 9005, ['room_count']] = 1
df_all.loc[df_all['apartment_id'] == 9005, ['bathroom_count']] = 1

df_all.loc[df_all['apartment_id'] == 1179, ['room_count']] = 4
df_all.loc[df_all['apartment_id'] == 1179, ['bathroom_count']] = 2

df_all.loc[df_all['apartment_id'] == 10627, ['room_count']] = 3
df_all.loc[df_all['apartment_id'] == 10627, ['bathroom_count']] = 1

df_all.loc[(df_all['apartment_id'] == 10627) & (df_all['supply_area'] == 56.61), ['room_count']] = 2
df_all.loc[(df_all['apartment_id'] == 10627) & (df_all['supply_area'] == 56.61), ['bathroom_count']] = 1

df_all.loc[(df_all['apartment_id'] == 7992) , ['room_count']] = 3
df_all.loc[
(df_all['apartment_id'] == 7992) & (df_all['supply_area'] &lt;= 81), ['bathroom_count']] = 1
df_all.loc[(df_all['apartment_id'] == 7992) & (df_all['supply_area'] &gt; 81), ['bathroom_count']] = 2

df_all.loc[(df_all['apartment_id'] == 7118) & (df_all['supply_area'] == 75.55), ['room_count']] = 3
df_all.loc[(df_all['apartment_id'] == 7118) & (df_all['supply_area'] == 75.55), ['bathroom_count']] = 1

df_all.loc[(df_all['apartment_id'] == 7118) & (df_all['supply_area']//10 == 5.0), ['room_count']] = 2
df_all.loc[(df_all['apartment_id'] == 7118) & (df_all['supply_area']//10 == 5.0), ['bathroom_count']] = 1

df_all.loc[(df_all['apartment_id'] == 4047) & (df_all['supply_area']//10 == 11.0), ['room_count']] = 3
df_all.loc[(df_all['apartment_id'] == 4047) & (df_all['supply_area']//10 == 11.0), ['bathroom_count']] =2

df_all.loc[(df_all['apartment_id'] == 37175) & (df_all['supply_area'] &lt; 80), ['room_count']] = 2
df_all.loc[(df_all['apartment_id'] == 37175) & (df_all['supply_area'] &lt; 80), ['bathroom_count']] = 1

df_all.loc[(df_all['apartment_id'] == 37175) & (df_all['supply_area'] &gt; 80), ['room_count']] = 3
df_all.loc[(df_all['apartment_id'] == 37175) & (df_all['supply_area'] &gt; 80), ['bathroom_count']] = 2

# In[ ]
### 주차장 결측치
df_all.loc[(df_all['total_parking_capacity_in_site'].isnull()), ['total_parking_capacity_in_site']] = 0
### 히트 결측치
df_all.loc[(df_all['heat_type'].isnull()), ['heat_type']] = 'None'
df_all.loc[(df_all['heat_fuel'].isnull()), ['heat_fuel']] = 'None'
### 현관구조 결측치
df_all.loc[(df_all['heat_fuel'].isnull()), ['front_door_structure']] = 'None'

# In[ ]
# 파생변수 생성
### 용적률(容積率)은 건축 용어로 전체 대지면적에 대한 건물 연면적의 비율을 뜻하며 백분율로 표시한다. 
### 용적률이 높을수록 건축할 수 있는 연면적이 많아져 건축밀도가 높아지므로, 적정 주거환경을 보장하기 위하여 용적률의 상한선을 지정한다.
df_all['effective_ratio'] = (df_all['exclusive_use_area'] / df_all['supply_area']) * 100

# In[ ]
### 시간을 좀 더 세부적으로 나타냄. 
df_all['transaction_date1'] = df_all.transaction_date.apply(lambda x: x[-2:])
#convert int to date
df_all['transaction_year_month1'] = df_all['transaction_year_month'].astype(str)
#join month and date 
df_all['transaction_year_month_date'] = df_all[['transaction_year_month1', 'transaction_date1']].apply(lambda x: ''.join(x), axis=1)
#convert  month and date to datetime 
df_all['transaction_year_month_date'] = pd.to_datetime(df_all['transaction_year_month_date'] )
#reindext datetime
del df_all['transaction_date1']; del df_all['transaction_year_month1']

# In[ ]
### 계산의 편의성을 위해 날짜를 만들어 둠.
df_all['year'] = df_all['transaction_year_month_date'].dt.year
df_all['month'] = df_all['transaction_year_month_date'].dt.month
df_all['month'] = df_all['month'].apply(lambda x : x-1 if x%2 == 0 else x)
df_all['mean_year_month'] = df_all['year']*100 + df_all['month']
del df_all['year']
del df_all['month']

# In[ ]
### 동일한 아파트의 가장 최근의 거래와 현재 거래의 차이를 계산
df_all['last_month'] = df_all.groupby(['apartment_id'])['transaction_year_month'].shift(1)
df_all['diff_month'] = df_all['transaction_year_month'] - df_all['last_month']
del df_all['last_month']

### 방의 총 갯수 ( 화장실 + 방 )
df_all['total_room'] = df_all['room_count'] + df_all['bathroom_count']

### Last_price_1과 3은 동일한 아파트의 면적대비 거래액을 의미. 추가로 현재 면적을 곱해줘야 함. 
df_all['last_price_1'] = df_all['transaction_real_price']/df_all['exclusive_use_area']
df_all['last_price_1'] = df_all.groupby(['apartment_id'])['last_price_1'].shift(1)

df_all['last_price_3'] = df_all['transaction_real_price']/df_all['supply_area']
df_all['last_price_3'] = df_all.groupby(['apartment_id'])['last_price_3'].shift(1)

df_all['last_area'] = df_all.groupby(['apartment_id'])['exclusive_use_area'].shift(1)
df_all['last_transaction_year_month'] = df_all.groupby(['apartment_id'])['transaction_year_month'].shift(1)
#df_all['transaction_real_price'] = np.log1p(df_all['transaction_real_price'])

df_all['last_price_1'] = df_all['last_price_1'] * df_all['exclusive_use_area'] #현재 면적을 곱해줘서 비교를 가능하게 만듬.
df_all['last_price_3'] = df_all['last_price_3'] * df_all['supply_area'] #현재 면적을 곱해줘서 비교를 가능하게 만듬.
del df_all['last_area'],df_all['last_transaction_year_month']

### log를 씌어줘서 정규성을 띄게 만듬. 
df_all['last_price_1'] = np.log1p(df_all['last_price_1'])
df_all['last_price_3'] = np.log1p(df_all['last_price_3'])
df_all['transaction_real_price'] = np.log1p(df_all['transaction_real_price'])

### 빌딩의 간격계산
df_all['difference_building_height'] = df_all['tallest_building_in_sites'] - df_all['lowest_building_in_sites']
### 세대당 주차수 계산
df_all['capacity_per_household'] = df_all['total_parking_capacity_in_site']/df_all['total_household_count_in_sites']

### 아파트당 세대 수 계산
df_all['household_per_building'] = df_all['total_household_count_in_sites']/df_all['apartment_building_count_in_sites']

### 아파트당 타입의 비율 계산
df_all['areahousehold_per_household'] = df_all['total_household_count_of_area_type']/df_all['total_household_count_in_sites']

df_all['year'] = df_all['transaction_year_month']//100

### 거래된 기간과 완성된 년도의 차이 계산
df_all['transaction_diff_completion'] = df_all['transaction_year_month'] - df_all['year_of_completion']

### 몇번째 층인지 비율 계산
df_all['floor_ratio'] = df_all['floor']/df_all['tallest_building_in_sites']

### 재개발 예정인지 가중치 줌. 
### 35를 상한선으로 잡은것은 이 이상이 되면 재개발 될 거라는 심리가 떨어져서 임.
df_all['weight'] = 0
df_all.loc[((df_all['year']-df_all['year_of_completion']) &gt;= 25) & ((df_all['year']-df_all['year_of_completion']) &lt; 35) & (df_all['effective_ratio'] &gt;= 80) & (df_all['tallest_building_in_sites'] &lt;=5),'weight'] = 1

# In[ ]
### 아래의 파일은 따로 첨부한 코드에 계산식이 나와있습니다.
### Apartment_subway : 0.5, 1km내에 몇개의 지하철이 있냐, 몇개의 호선이 있냐
### Apartment_gd_hd : 강남 및 해운대로부터의 거리가 얼마나 되냐
### apartment_school : 0.5km 내에 초,중,고등학교가 있냐 없냐. total_0.5는 3개 중에서 몇개가 있는지
### min_distance_apartment : 가장 가까운 초,중,고,지하철의 거리
apartment = pd.read_csv("Apartment_subway.csv")
apartment1 = apartment[['apartment_id','subwayline_count_0.5','subwayline_count_1','subway_count_0.5','subway_count_1']]
apartment2 = pd.read_csv("Apartment_ga_hd.csv")
apartment2 = apartment2[['apartment_id','gangnam_dist']]
apartment3 = pd.read_csv("apartment_school.csv")
apartment3 = apartment3[['apartment_id','elementary_0.5','middle_0.5','high_0.5','total_0.5']]
apartment4 = pd.read_csv("min_distance_apartment.csv")
apartment4 = apartment4[['apartment_id','subway_min_distance','min_distance_ele','min_distance_middle','min_distance_high']]
#apartment4 = pd.read_csv("apartment_bub.csv")
#apartment4 = apartment4[['apartment_id','gu','dong']]
df_all = pd.merge(df_all,apartment1,on='apartment_id').reset_index(drop=True)
df_all = pd.merge(df_all,apartment2,on='apartment_id').reset_index(drop=True)
df_all = pd.merge(df_all,apartment3,on='apartment_id').reset_index(drop=True)
df_all = pd.merge(df_all,apartment4,on='apartment_id').reset_index(drop=True).sort_values('index') #index

# In[ ]
# 공공데이터 사용목록과 코드는 따로 첨부하였습니다.
df_all = df_all.reset_index(drop=True)
### 구청
public = pd.read_csv('apartment_public.csv')
public = public[['apartment_id','public_1']]
public['public_1'] = public['public_1'].apply(lambda x: 1 if x&gt;1 else x)
df_all = pd.merge(df_all,public,on='apartment_id').reset_index(drop=True).sort_values('index') #index

# In[ ]
# PCA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA,NMF,LatentDirichletAllocation,IncrementalPCA,MiniBatchSparsePCA
from sklearn.decomposition import TruncatedSVD,FactorAnalysis,KernelPCA

train_df = df_all.loc[df_all['transaction_real_price'] != 0]
test_df = df_all.loc[df_all['transaction_real_price'] == 0]

train_len = train_df.shape[0]

### 날짜와 object, 공공데이터는 제거. 
train_columns = [c for c in train_df.columns if c not in ['key','transaction_real_price','transaction_year_month_date','transaction_date','heat_type','heat_fuel',
                                                          'front_door_structure','shop_count_0.5','shop_count_1','univ_1,2','public_1','coffee_count_0.5']]
train_columns

# PCA
n_comp = 1

# ICA
ica = FastICA(n_components=n_comp, random_state=2019)
ica2_results_train = ica.fit_transform(train_df[train_columns].fillna(-1))
ica2_results_test = ica.transform(test_df[train_columns].fillna(-1))

for i in range(1, n_comp+1):
    train_df['ica_' + str(i)] = ica2_results_train[:,i-1]
    test_df['ica_' + str(i)] = ica2_results_test[:, i-1]
    train_columns.append('ica_' + str(i))

df_all = pd.concat([train_df,test_df])
df_all = df_all.sort_values('index').reset_index(drop=True)

# In[ ]
### 날짜 형식변경
df_all['transaction_date1'] = df_all.transaction_date.apply(lambda x: x[-2:])
#convert int to date
df_all['transaction_year_month1'] = df_all['transaction_year_month'].astype(str)
#join month and date 
df_all['transaction_year_month_date'] = df_all[['transaction_year_month1', 'transaction_date1']].apply(lambda x: ''.join(x), axis=1)
df_all['transaction_year_month_date'] = df_all['transaction_year_month_date'].astype(int)
del df_all['transaction_date1']; del df_all['transaction_year_month1']
del df_all['transaction_year_month']; del df_all['transaction_date'] ; del df_all['year']

# In[ ]
### One-hot-encoding
df_all = pd.get_dummies(df_all)
train = df_all[:train_len]
df_test = df_all[train_len:]
del df_all
train = train.sort_values('index')
df_test = df_test.sort_values('index')

### 거리의 경우 너무 크면 의미가 없어서 상관관계를 통해서 파악한 16을 기준으로 잘라버림. 
train['gangnam_dist'] = train['gangnam_dist'].apply(lambda x: 16 if x &gt; 16 else x)
df_test['gangnam_dist'] = df_test['gangnam_dist'].apply(lambda x: 16 if x &gt; 16 else x)

# In[ ]
# LIGHTGBM 모델
excluded_features = [
    'transaction_real_price'
]

categorical_features = [
    'apartment_id'
]

# In[ ]
del train['index']
del df_test['index']
del train['mean_year_month']
del df_test['mean_year_month']

# In[ ]
df_train_busan = train[train['city']==0].reset_index(drop=True)
df_test_busan = df_test[df_test['city']==0].reset_index(drop=True)
df_train_seoul = train[train['city']==1].reset_index(drop=True)
df_test_seoul = df_test[df_test['city']==1].reset_index(drop=True)

# In[ ]
import time

y_train = df_train_busan['transaction_real_price'].reset_index(drop=True)

x_train = df_train_busan.copy().reset_index(drop=True)
del x_train['city']; del x_train['transaction_real_price']; del x_train['public_1']; 


excluded_features = ['key']
train_features = [_f for _f in x_train.columns if _f not in excluded_features]

busan_key = df_test_busan['key'].values
x_test = df_test_busan[train_features].reset_index(drop=True)

# In[ ]
# LightGBM
folds = KFold(n_splits=5,random_state=6,shuffle=True)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0

feature_importance_df = pd.DataFrame()

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train[train_features].iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train[train_features].iloc[val_idx], y_train[val_idx]  
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)   
    
    params = {"objective" : "regression", "metric" : "quantile", 'n_estimators':15000, 'early_stopping_rounds':133,
              "num_leaves" : 20, "learning_rate" : 0.18, "bagging_fraction" : 0.8,
               "bagging_seed" : 0, 'min_data_in_leaf': 1144, 'max_depth': 6}
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_preds += sub_pred
    
    #print('Fold %2d rmse : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))))
    valid_score += mean_squared_error(val_y, oof_preds[val_idx])
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = train_features
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    gc.collect()
    
print('Full rmse score %.6f\n' % np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_preds))))

# In[ ]
x_train['transaction_real_price'] = oof_preds
x_train.to_csv("Lightgbm_Bestmodel_busan_train_not_quantile.csv",index=False)

x_test['transaction_real_price'] = sub_preds
x_test['key'] = busan_key
x_test.to_csv("Lightgbm_Bestmodel_busan_test_not_quantile.csv",index=False)

sub_busan_not = x_test[['key','transaction_real_price']]
busan_valid.columns = ['key','valid_price']
busan_valid = pd.merge(busan_valid,x_train,on='key',how='left')


print('Full rmse score %.6f\n' % np.sqrt(mean_squared_error(np.expm1(busan_valid['transaction_real_price']), busan_valid['valid_price'])))
busan_score_not = np.sqrt(mean_squared_error(np.expm1(busan_valid['transaction_real_price']), busan_valid['valid_price']))

# In[ ]
import time 
y_train = df_train_seoul['transaction_real_price'].reset_index(drop=True)

x_train = df_train_seoul.copy().reset_index(drop=True)
del x_train['city']; del x_train['transaction_real_price']; del x_train['last_price_3'];  del x_train['ica_1']


excluded_features = ['key','floor']
train_features = [_f for _f in x_train.columns if _f not in excluded_features]

seoul_key = df_test_seoul['key'].values
seoul_floor = df_test_seoul['floor'].values
x_test = df_test_seoul[train_features].reset_index(drop=True)

# In[ ]
# LightGBM
import time
folds = KFold(n_splits=5,random_state=6,shuffle=True)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0

feature_importance_df = pd.DataFrame()

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train[train_features].iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train[train_features].iloc[val_idx], y_train[val_idx]  
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)   
    
    params = {"objective" : "regression", "metric" : "quantile", 'n_estimators': 20000, 'early_stopping_rounds':110,
              "num_leaves" : 30, "learning_rate" : 0.15, "bagging_fraction" : 0.9, "lambda_l1" : 0.1,
               "bagging_seed" : 0}
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_preds += sub_pred
    
    #print('Fold %2d rmse : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))))
    valid_score += mean_squared_error(val_y, oof_preds[val_idx])
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = train_features
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    gc.collect()
    
print('Full rmse score %.6f\n' % np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_preds))))

# In[ ]
x_train['transaction_real_price'] = oof_preds
x_train.to_csv("Lightgbm_Bestmodel_seoul_train_not_quantile.csv",index=False)

x_test['transaction_real_price'] = sub_preds
x_test['key'] = seoul_key
x_test['floor'] = seoul_floor

x_test.to_csv("Lightgbm_Bestmodel_seoul_test_not_quantile.csv",index=False)

sub_seoul_not = x_test[['key','transaction_real_price']]
seoul_valid.columns = ['key','valid_price']
seoul_valid = pd.merge(seoul_valid,x_train,on='key',how='left')


print('Full rmse score %.6f\n' % np.sqrt(mean_squared_error(np.expm1(seoul_valid['transaction_real_price']), seoul_valid['valid_price'])))
busan_score_not = np.sqrt(mean_squared_error(np.expm1(seoul_valid['transaction_real_price']), seoul_valid['valid_price']))

# In[ ]
sub_not_deep = pd.concat([sub_busan_not,sub_seoul_not])
sub_not_deep.head()

# In[ ]
sub_not_deep['transaction_real_price'] = np.expm1(sub_not_deep['transaction_real_price'])
sub_not_deep = sub_not_deep.sort_values('key')
sub_not_deep = sub_not_deep.reset_index(drop=True)
sub_not_deep.to_csv("[190130]LGB_Quantile_not_deep.csv",index=False)

