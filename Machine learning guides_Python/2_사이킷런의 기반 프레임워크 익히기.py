# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:38:48 2020

@author: Administrator
"""
# In[ ]
# 사이킷런의 기반 프레임워크 익히기
! conda install scikit-learn

# In[ ]:
! pip install scikit-learn
    
# In[ ]:
import sklearn
print(sklearn.__version__)

# In[ ]:
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier    
from sklearn.model_selection import train_test_split

# In[ ]:
import pandas as pd

# 붓꽃 데이터 세트를 로딩합니다.    
iris = load_iris()

# iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다.
iris_data=iris.data

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를numpy로 가지고 있습니다.
iris_label=iris.target
print('iris target값:', iris_label)
print('iris target명:', iris.target_names)

#붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
iris_df=pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target
print(iris_df.head(3))

# In[ ]:
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                    test_size=0.2, random_state=11)
    
# In[ ]:
# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

#학습 수행    
dt_clf.fit(X_train, y_train)

# In[ ]:
# 학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행.
pred=dt_clf.predict(X_test)    

# In[ ]:
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
    
# In[ ]:
from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data)) 

# In[ ]:
keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:', keys)    

# In[ ]:
print('\n feature_names의 type:',type(iris_data.feature_names))
print('feature_names의 shape:',len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names의 type:',type(iris_data.target_names))
print('target_names의 shape:',len(iris_data.target_names))
print(iris_data.target_names)

print('\n data의 type:',type(iris_data.data))
print('data의 shape:',iris_data.data.shape)
print(iris_data.data)

print('\n target의 type:',type(iris_data.target))
print('target의 shape:',iris_data.target.shape)
print(iris_data.target)
