# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:19:18 2020

@author: 동현중
"""

# In[ ]:
import pandas as pd

# In[ ]:
titanic_df=pd.read_csv(r'C:\Users\Administrator\파이썬 머신러닝\titanic_train.csv')
titanic_df.head(3)

# In[ ]:
titanic_df = pd.read_csv('titanic_train.csv')
print('titanic 변수 type:', type(titanic_df))
titanic_df
titanic_df.head(3)

# In[ ]:
print('DataFrame 크기:', titanic_df.shape)

# In[ ]:
titanic_df.info()

# In[ ]:
titanic_df.describe()

# In[ ]:
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)

# In[ ]:
titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))

titanic_pclass.head()
# In[ ]:
vaLue_counts=titanic_df['Pclass'].value_counts
print(type(value_counts))
print(value_counts)

# In[ ]:
import numpy as np

col_name1=['col1']
list1=[1,2,3]
array1 = np.array(list1)
print('array1 shape:', array1.shape)

#리스트를 이용해 DataFrame 생성.
df_list1=pd.DataFrame(list1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)

#넘파이 ndarray를  이용해 DataFrame 생성.
df_array1=pd.DataFrame(array1, columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)

# In[ ]:
col_name2=['col1', 'col2', 'col3']

#2행x3열 형태의 리스트와 ndarray 생성한 뒤 이를 DataFrame으로 변환.
list2=[[1,2,3],
       [11,12,13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape)

#리스트를 이용해 DataFrame 생성.
df_list2=pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)

#넘파이 ndarray를  이용해 DataFrame 생성.
df_array2=pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)

# In[ ]:
# key는 문자열 칼럼영으로 매핑, Value는 리스트 형(또는 ndarray)칼럼 데이터로 매핑
dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict=pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n',df_dict)

# In[ ]:
#DataFrame을 ndarray로 변환
array3=df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.value shape:', array3.shape)
print(array3)

# In[ ]:
#DataFrame을 리스트로 변환
list3=df_dict.values.tolist()
print('df_dict.values.tolist() 타입:', type(list3))
print(list3)

#DataFrame을 딕셔너리 변환
dict3=df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입:', type(dict3))
print(dict3)

# In[]
titanic_df['Age_0']=0
titanic_df.head(3)

# In[ ]:
titanic_df['Age_by_10']=titanic_df['Age']*10
titanic_df['Family_No']=titanic_df['SibSp']+titanic_df['Parch']+1
titanic_df.head(3)

# In[ ]:
titanic_df['Age_by_10']=titanic_df['Age_by_10']+100
titanic_df.head(3)

# In[ ]:
titanic_drop_df = titanic_df.drop('Age_0', axis=1)
titanic_drop_df.head(3)
titanic_df.head(3)

# In[ ]:
drop_result = titanic_df.drop(['Age_0','Age_by_10', 'Family_No'], axis=1, inplace=True)
print('inplace=True로 drop 후 반환된 값:', drop_result)
titanic_df.head(3)

# In[ ]:
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(3))

titanic_df.drop([0,1,2], axis=0, inplace=True)

print('#### after axis 0 drop ####')
print(titanic_df.head(3))

# In[ ]:
#원본 파일 다시 로딩
titanic_df=pd.read_csv(r'C:\Users\Administrator\파이썬 머신러닝\titanic_train.csv')

# Index 객체 추출
indexes = titanic_df.index
print(indexes)

# Index 객체를 실제 값 array로 변환
print('Index 객체 array값:\n', indexes.values)

# In[ ]:
print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])

# In[ ]:
# DataFrame 및 Series의 한 번 만들어진 Index 객체는 변경불가
indexes[0]=5

# In[ ]:
series_fair = titanic_df['Fare']
print('Fair Series max 값:',series_fair.max())
print('Fair Series sum 값:',series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series +3:\n', (series_fair+3).head(3))

# In[ ]:
titanic_reset_df=titanic_df.reset_index(inplace=False)
print(titanic_reset_df.head(3))

# In[ ]:
print('#### before reset_index ####')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:', type(value_counts))
new_value_counts=value_counts.reset_index(inplace=False)
print('#### after reset_index ####')
print(new_value_counts)
print('new_vaLue_counts 객체 변수 타입:', type(new_value_counts))

# In[ ]:
print('단일 칼럼 데이터 추출:\n', titanic_df['Pclass'].head(3))
print('\n여러 칼럼의 데이터 추출:\n', titanic_df[['Survived', 'Pclass']].head(3))
print('[] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])

# In[ ]:
titanic_df[0:2]
print(titanic_df[(titanic_df['Pclass']==3)].head(3))

# In[ ]:
# 지금은 ix 사라짐
print('칼럼 위치 기반 인덱싱 데이터 추출:', titanic_df.ix[0,2])
print('칼럼 명 기반 인덱싱 데이터 추출:', titanic_df.ix[0,'Pclass'])

# In[ ]:
data={'Name':['Chulmin','Eunkyung','Jinwoong','Soobeom'],
      'Year':[2001,2016,2015,2015],
      'Gender':['Male','Female','Male','Male']
      }
data_df=pd.DataFrame(data, index=['one','two','three','four'])
print(data_df)

# In[ ]:
# data_df를 reset_index()로 새로운 숫자형 인덱스를 생성    
data_df_reset= data_df.reset_index()
data_df_reset= data_df_reset.rename(columns={'index':'old_index'})

# 인덱스값에 1을 더해서 1부터 시작하는 새로운 인덱스값 생성
data_df_reset.index= data_df_reset.index+1
print(data_df_reset)
data_df_reset.ix[1,1]

# In[ ]:
#iloc는 위치기반
print(data_df.iloc[0,0])

# In[ ]:
# 다음 코드는 오류를 발생합니다.
data_df.iloc[0,'Name']

# In[ ]:
# 다음 코드는 오류를 발생합니다.
data_df.iloc['one',0]

# In[ ]:
print(data_df_reset.iloc[0,1])

# In[ ]:
#loc는 명칭기반
print(data_df.loc['one','Name'])

# In[ ]:
print(data_df_reset.loc[1,'Name'])
    
# In[ ]:
# 인덱스가 0인 값이 없으므로 오류 반환
print(data_df_reset.loc[0,'Name'])


# In[ ]:
print('명칭기반 ix slicing\n', data_df.ix['one':'two', 'Name'], '\n')
print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0], '\n')
print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name'])

# In[ ]:
#명칭기반   
print(data_df_reset.loc[1:2,'Name'])

#위치기반으로 사용
print(data_df.ix[1:2,'Name'])

# In[ ]:
titanic_df=pd.read_csv(r'C:\Users\Administrator\파이썬 머신러닝\titanic_train.csv')
titanic_boolean=titanic_df[titanic_df['Age']>60]
print((type(titanic_boolean)))
print(titanic_boolean)

# In[ ]:
print(titanic_df[titanic_df['Age']>60][['Name','Age']].head(3))
print(titanic_df.loc[titanic_df['Age']>60][['Name','Age']].head(3))

# In[ ]:
# and = & / or = | / Not = ~
print(titanic_df[(titanic_df['Age']>60)&(titanic_df['Pclass']==1)&
           (titanic_df['Sex']=='female')])

# In[ ]:
cond1=titanic_df['Age']>60
cond2=titanic_df['Pclass']==1
cond3=titanic_df['Sex']=='female'
print(titanic_df[ cond1&cond2&cond3])

# In[ ]:
titanic_sorted=titanic_df.sort_values(by=['Name'])
print(titanic_sorted.head(3))

# In[ ]:
titanic_sorted=titanic_df.sort_values(by=['Pclass','Name'],ascending=False)
print(titanic_sorted.head(3))

# In[ ]:
# 칼럼의 개수 결측치 제외    
print(titanic_df.count())

# 대상 칼럼의 평균
print(titanic_df[['Age','Fare']].mean())

# In[ ]:
# Pclass 칼럼 기준으로 GroupBy됨
titanic_groupby=titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))

# In[ ]:
titanic_groupby=titanic_df.groupby(by='Pclass').count()
print(titanic_groupby)
    
# In[ ]:
titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId','Survived']].count()
print(titanic_groupby)    

# In[ ]:
print(titanic_df.groupby('Pclass')['Age'].agg([max,min]))
    
# In[ ]:
agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
print(titanic_df.groupby('Pclass').agg(agg_format))    

# In[ ]:
print(titanic_df.isna().head(3))
    
# In[ ]:
# 결측치 갯수
print(titanic_df.isna().sum())
    
# In[ ]:
titanic_df['Cabin']=titanic_df['Cabin'].fillna('C000')
print(titanic_df.head(3))

# In[ ]:
titanic_df['Age']=titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')
print(titanic_df.isna().sum())

# In[ ]:
def get_square(a):
    return a**2

print('3의 제곱은:', get_square(3))
    
# In[ ]:
lambda_square = lambda x : x**2
print('3의 제곱은:', lambda_square(3))    

# In[ ]:
a=[1,2,3]    
squares=map(lambda x : x**2, a)
print(list(squares))
    
# In[ ]:
titanic_df['Name_len']=titanic_df['Name'].apply(lambda x : len(x))
print(titanic_df[['Name', 'Name_len']].head(3))

# In[ ]:
titanic_df['Child_Adult']=titanic_df['Age'].apply(lambda x : 'Child' if x <= 15 else 'Adult')
print(titanic_df[['Age','Child_Adult']].head(8))

# In[ ]:
# lambda는 else if를 지원하지 않음
titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x<=60 else 'Elderly'))
print(titanic_df['Age_cat'].value_counts())
    
# In[ ]:
# 나이에 따라 세분화된 분류를 수행하는 함수 생성
def get_category(age):
    cat=''
    if age <=5: cat='Baby'
    elif age <=12: cat='Child'
    elif age <=18: cat='Teenager'
    elif age <=25: cat='Student'
    elif age <=35: cat='Young Adult'
    elif age <=60: cat='Adult'
    else : cat='Elderly'
    
    return cat

# lambda 식 위에서 생성한 get_category() 함수를 반환값으로 지정.
# get_category(X)는 입력값으로 'Age' 칼럼 값을 받아서 해당하는 cat 변환
titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : get_category(x))
print(titanic_df[['Age', 'Age_cat']].head()) 
    