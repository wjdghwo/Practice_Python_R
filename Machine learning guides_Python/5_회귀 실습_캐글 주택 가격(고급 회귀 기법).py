# -*- coding: utf-8 -*-
'''
Competition Description


주택 구입자에게 꿈의 집을 설명해달라고 요청하면, 지하 천장 높이나 동서 철도와의 근접성으로 시작하지 않을 것입니다.
그러나 이 대회의 데이터 세트는 침실 수나 흰 울타리보다 가격 협상에 훨씬 더 많은 영향을 미치는 어떤것을 증명합니다.

아이오와 주 에임스에 있는 주거용 주택의 거의 모든 측면을 설명하는 79개의 설명 변수가있는 이 경쟁에서는 각 주택의 최종 가격을 예측해야 합니다.

Practice Skills

Creative feature engineering 
Advanced regression techniques like random forest and gradient boosting
'''

"""
SalePrice : 부동산 판매가격. 예측하려는 목표 변수입니다.
MSSubClass: 건물 등급
MSZoning : 일반 구역 분류
LotFrontage : 사유지에 연결된 도로의 직선 피트
LotArea : 평방 피트 단위의 부지 크기
Street : 도로 접근 유형
Alley : 골목 접근 유형
LotShape : 사유지의 일반적인 모양
LandContour : 사유지의 평탄도
Utilities : 사용 가능한 유틸리티 유형
LotConfig : 품목 구성
LandSlope : 사유지의 경사
Neighborhood : 에임스시 경계 내의 물리적 위치
Condition1 : 주요 도로 또는 철도와의 근접성
Condition2 : 주요 도로 또는 철도와의 근접성 (짧은거리일 경우)
BldgType : 주거 유형
HouseStyle : 주거 스타일
OverallQual : 전체 재료 및 마감 품질
OverallCond : 전체 상태 등급
YearBuilt : 원래 건축 날짜
YearRemodAdd : 리모델링 날짜
RoofStyle : 지붕 유형
RoofMatl : 지붕 재료
Exterior1st : 집 외부 자제
Exterior2nd : 집 외부 자제 (하나 이상의 재료인 경우)
MasVnrType : 벽돌 베니어 유형
MasVnrArea : 벽돌 베니어 면적 (평방 피트)
ExterQual : 외장재 품질
ExterCond : 외장재의 현황
Foundation : 토대 유형
BsmtQual : 지하 높이
BsmtCond : 지하실의 일반 상태
BsmtExposure : 워크 아웃 또는 정원 수준의 지하 벽
BsmtFinType1 : 지하실 마감면의 품질
BsmtFinSF1 : 유형 1 마감 평방 피트
BsmtFinType2 : 두 번째 완성 된 영역의 품질 (있는 경우)
BsmtFinSF2 : 유형 2 마감 평방 피트
BsmtUnfSF : 미완성 된 지하실 면적
TotalBsmtSF : 지하 총 평방 피트
Heating : 난방 유형
HeatingQC : 난방 품질 및 상태
CentralAir : 중앙 에어컨
Electrical : 전기 시스템
1stFlrSF : 1 층 평방 피트
2ndFlrSF : 2 층 평방 피트
LowQualFinSF : 저품질 마감 평방 피트 (모든 층)
GrLivArea : 지상 (지상) 거실 면적 평방 피트
BsmtFullBath : 지하 전체 욕실
BsmtHalfBath : 지하 반 욕실
FullBath : 상급 화장실
HalfBath : 상급 화장실(세면대와 변기만 있는)
Bedroom : 지하층 이상의 침실 수
Kitchen : 주방 수
KitchenQual : 주방 품질
TotRmsAbvGrd : 상급 전체 방 (화장실 제외)
Functional : 홈 기능 등급
Fireplaces : 벽난로 수
FireplaceQu : 벽난로 품질
GarageType : 차고 위치
GarageYrBlt : 차고 건설 연도
GarageFinish : 차고 내부 마감
GarageCars : 차량 수용 가능 차고 크기
GarageArea : 차고 크기 (평방 피트)
GarageQual : 차고 품질
GarageCond : 차고 상태
PavedDrive : 포장 된 진입로
WoodDeckSF : 목재 데크 면적 (평방 피트)
OpenPorchSF : 평방 피트 단위의 열린 현관 영역
EnclosedPorch : 닫힌 현관 영역 (평방 피트)
3SsnPorch : 평방 피트 단위의 3 계절 현관 면적
ScreenPorch : 스크린 현관 영역 (평방 피트)
PoolArea : 수영장 면적 (평방 피트)
PoolQC : 수영장 품질
Fence : 울타리 품질
MiscFeature : 다른 카테고리에서 다루지 않는 기타 기능
MiscVal : 기타 기능의 달러 가치
MoSold : 월 판매
YrSold : 판매 연도
SaleType : 판매 유형
SaleCondition : 판매 조건
"""

# In[ ]
import warnings
warnings.filterwarnings('ignore') # 경고메시지 숨기기
import pandas as pd
import numpy as np
import seaborn as sns # 데이터 시각화
import matplotlib.pyplot as plt # 데이터 시각화
# %matplotlib inline

house_df_org = pd.read_csv('house_price.csv')
house_df = house_df_org.copy() # 원본데이터 복사
print(house_df.head(3))

# In[ ]
print('데이터 세트의 Shape:', house_df.shape) # 데이터 형태(size) 파악
print('\n전체 feature 들의 type \n',house_df.dtypes.value_counts()) # 피처들의 타입 파악
# 문자형 43개, 정수형 35개, 실수형 3개
isnull_series = house_df.isnull().sum() # 각 피처별 Null값 갯수 파악
print('\nNull 컬럼과 그 건수:\n ', isnull_series[isnull_series > 0].sort_values(ascending=False)) # Null 개수 파악

# In[ ]
plt.title('Original Sale Price Histogram') # 그래프 제목
sns.distplot(house_df['SalePrice']) # 판매가격에 대한 히스토그램 확인

# In[ ]
plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['SalePrice']) # 데이터가 치우져처 로그변환 시킴
# np.log를 사용하게 되면 언더플로우(취급 범위보다 더 작아짐)가 발생하기 쉬워 1+log()=log1p() 함수를 사용
sns.distplot(log_SalePrice)

# In[ ]
# SalePrice 로그 변환
original_SalePrice = house_df['SalePrice']
house_df['SalePrice'] = np.log1p(house_df['SalePrice'])

# Null 이 너무 많은 컬럼들과 불필요한 컬럼 삭제
house_df.drop(['Id','PoolQC' , 'MiscFeature', 'Alley', 'Fence','FireplaceQu'], axis=1 , inplace=True)
# Drop 하지 않는 숫자형 Null컬럼들은 평균값으로 대체
house_df.fillna(house_df.mean(),inplace=True)

# Null 값이 있는 피처명과 타입을 추출
null_column_count = house_df.isnull().sum()[house_df.isnull().sum() > 0]
print('## Null 피처의 Type :\n', house_df.dtypes[null_column_count.index])

# In[ ]
print('get_dummies() 수행 전 데이터 Shape:', house_df.shape)
house_df_ohe = pd.get_dummies(house_df) # 자동으로 문자열 피처를 원-핫 인코딩 변환하면서 Null 값은 'None' 칼럼으로 대체
print('get_dummies() 수행 후 데이터 Shape:', house_df_ohe.shape)

null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum() > 0]
print('## Null 피처의 Type :\n', house_df_ohe.dtypes[null_column_count.index])

# In[ ]
'''
선형 회귀 모델 학습/예측/평가
'''

def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test , pred) # mse 계산
    rmse = np.sqrt(mse) # 루트 계산
    print('{0} 로그 변환된 RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))
    return rmse

def get_rmses(models):
    rmses = [ ]
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

# In[ ]
from sklearn.linear_model import LinearRegression, Ridge, Lasso # LinearRegression, Ridge, Lasso 기법
from sklearn.model_selection import train_test_split # train, test set 분할 함수
from sklearn.metrics import mean_squared_error # mse 구하는 함수

y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

# LinearRegression, Ridge, Lasso 학습, 예측, 평가
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

'''
Lasso의 경우 회귀 성능이 떨어져 최적 하이퍼 파라미터 튜닝 필요
'''

# In[ ]
def get_top_bottom_coef(model):
    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명. 
    coef = pd.Series(model.coef_, index=X_features.columns)
    
    # 회귀 계수값의 + 상위 10개 , - 하위 10개 coefficient 추출하여 반환.
    coef_high = coef.sort_values(ascending=False).head(10)
    coef_low = coef.sort_values(ascending=False).tail(10)
    return coef_high, coef_low

# In[ ]
def visualize_coefficient(models):
    # 3개 회귀 모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=3) # fig 전체 그래프, axs sub 그래프 생성
    fig.tight_layout() 
    
    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화. 
    for i_num, model in enumerate(models): # enumerate : 나열
        # 상위 10개, 하위 10개 회귀 계수를 구하고, 이를 판다스 concat으로 결합. 
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat( [coef_high , coef_low] ) # 동일한 형태의 dataframe 합침
        
        # 순차적으로 ax subplot에 barchar로 표현. 한 화면에 표현하기 위해 tick label 위치와 font 크기 조정. 
        axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)
        axs[i_num].tick_params(axis="y",direction="in", pad=-120)
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 회귀 계수 시각화.    
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)

'''
라쏘의 경우 전체적으로 회귀 계수 값이 매우 작음.
YearBuilt가 가장 크고 다른 피처의 회귀계수는 너무 작음

학습데이터의 데이터 분할에 문제가 있을 수도 있으므로 
전체 데이터 세트인 X_features와 y_target을 5개의 교차 검ㄷ증 폴드 세트로 분할해 평균 RMSE 측정
'''
# In[ ]
from sklearn.model_selection import cross_val_score

def get_avg_rmse_cv(models):
    for model in models:
        # 분할하지 않고 전체 데이터로 cross_val_score( ) 수행. 모델별 CV RMSE값과 평균 RMSE 출력
        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target,
                                             scoring="neg_mean_squared_error", cv = 5))
        rmse_avg = np.mean(rmse_list)
        print('\n{0} CV RMSE 값 리스트: {1}'.format( model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV 평균 RMSE 값: {1}'.format( model.__class__.__name__, np.round(rmse_avg, 3)))

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 CV RMSE값 출력           
models = [lr_reg, ridge_reg, lasso_reg]
get_avg_rmse_cv(models)

# In[ ]
'''
릿지와 라쏘 모델의 alpha 하이퍼 파라미터 최적값 찾기
'''
from sklearn.model_selection import GridSearchCV

def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, 
                              scoring='neg_mean_squared_error', cv=5) # params 파라미터를 train, test set fold로 나누어 mse를 구하는 테스트 수행 결정
    grid_model.fit(X_features, y_target) # 하이퍼 파라미터 순차적으로 학습/평가
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                                              np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_rige = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)

'''
Ridge 5 CV 시 최적 평균 RMSE 값: 0.1418, 최적 alpha:{'alpha': 12}
Lasso 5 CV 시 최적 평균 RMSE 값: 0.142, 최적 alpha:{'alpha': 0.001}
'''

# In[ ]
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge(alpha=12)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)

# In[ ]
'''
살펴볼 것
1. 피처 데이터 세트 데이터 분포도
2. 이상치 데이터 처리
'''

from scipy.stats import skew

# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.
features_index = house_df.dtypes[house_df.dtypes != 'object'].index # 문자형이 아닌 객체들만 추출
# house_df에 컬럼 index를 [ ]로 입력하면 해당하는 컬럼 데이터 셋 반환. apply lambda로 skew( )호출 
skew_features = house_df[features_index].apply(lambda x : skew(x))
# skew 정도가 5 이상인 컬럼들만 추출. 
skew_features_top = skew_features[skew_features > 1] # 피처들의 왜곡된 정도 확인
print(skew_features_top.sort_values(ascending=False))

# In[ ]
house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index]) # 왜곡도 높은 피처들 로그 변환

# In[ ]
# Skew가 높은 피처들을 로그 변환 했으므로 다시 원-핫 인코딩 적용 및 피처/타겟 데이터 셋 생성,
house_df_ohe = pd.get_dummies(house_df)
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

# 피처들을 로그 변환 후 다시 최적 하이퍼 파라미터와 RMSE 출력
ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_ridge = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)

# In[ ]
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge(alpha=10)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)

'''
이상치 데이터 처리 ㄱㄱ
가장 큰 회귀 계수를 가지는 GrLivArea 피처 분포를 살펴봄
'''
# In[ ]
plt.scatter(x = house_df_org['GrLivArea'], y = house_df_org['SalePrice'])
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel('GrLivArea', fontsize=15)
plt.show()

'''
4000평방피트 이상에도 500,000달러 이하인 데이터는 이상치로 간주
'''
# In[ ]
# GrLivArea와 SalePrice 모두 로그 변환되었으므로 이를 반영한 조건 생성. 
cond1 = house_df_ohe['GrLivArea'] > np.log1p(4000)
cond2 = house_df_ohe['SalePrice'] < np.log1p(500000)
outlier_index = house_df_ohe[cond1 & cond2].index # 이상치 전환

print('아웃라이어 레코드 index :', outlier_index.values)
print('아웃라이어 삭제 전 house_df_ohe shape:', house_df_ohe.shape)
# DataFrame의 index를 이용하여 아웃라이어 레코드 삭제. 
house_df_ohe.drop(outlier_index , axis=0, inplace=True)
print('아웃라이어 삭제 후 house_df_ohe shape:', house_df_ohe.shape)

# In[ ]
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_ridge = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)

# In[ ]
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge(alpha=8)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)

# In[ ]
from xgboost import XGBRegressor

xgb_params = {'n_estimators':[1000]}
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, 
                       colsample_bytree=0.5, subsample=0.8)
# n_estimators-1000까지만 반복 learning_rate-학습률 조정(순차적으로 오류 값 보정하는데 쓰는 계수)
# colsample_bytree-트리 생성에 필요한 피처를 임의로 샘플링/subsample-트리가 복잡하게 생성되는 것을 막음
best_xgb = get_best_params(xgb_reg, xgb_params)

# In[ ]
from lightgbm import LGBMRegressor

lgbm_params = {'n_estimators':[1000]}
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
# num_leaves-개별트리가 가질수 있는 최대 리프 개수(과적합 조정)/
# reg_lambda-L2계수 적용값 값이 클수록 과적합 감소/n_jobs-병렬 흐름 수
best_lgbm = get_best_params(lgbm_reg, lgbm_params)

# In[ ]
# 모델의 중요도 상위 20개의 피처명과 그때의 중요도값을 Series로 반환.
def get_top_features(model):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=X_features.columns  )
    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
    return ftr_top20

def visualize_ftr_importances(models):
    # 2개 회귀 모델의 시각화를 위해 2개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=2)
    fig.tight_layout() 
    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 피처 중요도 시각화. 
    for i_num, model in enumerate(models):
        # 중요도 상위 20개의 피처명과 그때의 중요도값 추출 
        ftr_top20 = get_top_features(model)
        axs[i_num].set_title(model.__class__.__name__+' Feature Importances', size=25)
        #font 크기 조정.
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=ftr_top20.values, y=ftr_top20.index , ax=axs[i_num])

# 앞 예제에서 get_best_params( )가 반환한 GridSearchCV로 최적화된 모델의 피처 중요도 시각화    
models = [best_xgb, best_lgbm]
visualize_ftr_importances(models)

# In[ ]
def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test , pred_value)
        rmse = np.sqrt(mse)
        print('{0} 모델의 RMSE: {1}'.format(key, rmse))

# 개별 모델의 학습
ridge_reg = Ridge(alpha=8)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)
# 개별 모델 예측
ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

# 개별 모델 예측값 혼합으로 최종 예측값 도출
pred = 0.4 * ridge_pred + 0.6 * lasso_pred # Lasso에 더 가중치를 둠
preds = {'최종 혼합': pred,
         'Ridge': ridge_pred,
         'Lasso': lasso_pred}
#최종 혼합 모델, 개별모델의 RMSE 값 출력
get_rmse_pred(preds)

# In[ ]
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, 
                       colsample_bytree=0.5, subsample=0.8)
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
xgb_reg.fit(X_train, y_train)
lgbm_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)
lgbm_pred = lgbm_reg.predict(X_test)

pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
preds = {'최종 혼합': pred,
         'XGBM': xgb_pred,
         'LGBM': lgbm_pred}
        
get_rmse_pred(preds)

# In[ ]
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수. 
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds ):
    # 지정된 n_folds값으로 KFold 생성.
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
    #추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화 
    train_fold_pred = np.zeros((X_train_n.shape[0] ,1 )) # 행의 개수로 된 train 벡터 생성 
    test_pred = np.zeros((X_test_n.shape[0],n_folds))
    print(model.__class__.__name__ , ' model 시작 ')
    
    for folder_counter , (train_index, valid_index) in enumerate(kf.split(X_train_n)): # 문자열을 세서 반복시행
        #입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 셋 추출 
        print('\t 폴드 세트: ',folder_counter,' 시작 ')
        X_tr = X_train_n[train_index] 
        y_tr = y_train_n[train_index] 
        X_te = X_train_n[valid_index]  
        
        #폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행.
        model.fit(X_tr , y_tr)       
        #폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장.
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)
        #입력된 원본 테스트 데이터를 폴드 세트내 학습된 기반 모델에서 예측 후 데이터 저장. 
        test_pred[:, folder_counter] = model.predict(X_test_n)
            
    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성 
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)    
    
    #train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred , test_pred_mean

# In[ ]
# get_stacking_base_datasets( )은 넘파이 ndarray를 인자로 사용하므로 DataFrame을 넘파이로 변환. 
X_train_n = X_train.values
X_test_n = X_test.values
y_train_n = y_train.values

# 각 개별 기반(Base)모델이 생성한 학습용/테스트용 데이터 반환. 
ridge_train, ridge_test = get_stacking_base_datasets(ridge_reg, X_train_n, y_train_n, X_test_n, 5)
lasso_train, lasso_test = get_stacking_base_datasets(lasso_reg, X_train_n, y_train_n, X_test_n, 5)
xgb_train, xgb_test = get_stacking_base_datasets(xgb_reg, X_train_n, y_train_n, X_test_n, 5)  
lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_test_n, 5)

# In[ ]
# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 Stacking 형태로 결합.  
Stack_final_X_train = np.concatenate((ridge_train, lasso_train, 
                                      xgb_train, lgbm_train), axis=1)
Stack_final_X_test = np.concatenate((ridge_test, lasso_test, 
                                     xgb_test, lgbm_test), axis=1)

# 최종 메타 모델은 라쏘 모델을 적용. 
meta_model_lasso = Lasso(alpha=0.0005)

#기반 모델의 예측값을 기반으로 새롭게 만들어진 학습 및 테스트용 데이터로 예측하고 RMSE 측정.
meta_model_lasso.fit(Stack_final_X_train, y_train)
final = meta_model_lasso.predict(Stack_final_X_test)
mse = mean_squared_error(y_test , final)
rmse = np.sqrt(mse)
print('스태킹 회귀 모델의 최종 RMSE 값은:', rmse)
