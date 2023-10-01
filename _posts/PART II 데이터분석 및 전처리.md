<h1>데이터 전처리</h1>



 인공지능 모델이 좋은 성능을 발휘하기 위해서는 좋은 데이터들로 학습되어야 한다. 따라서 학습 데이터를 모델이 적절하게 학습하도록 전처리를 해야한다. 전처리의 종류에는 많은 것들이 있지만 거의 필수적으로 해야하는 전처리는 3가지 정도가 있다.



1. 결측치 처리 및 필요없는 데이터 삭제
2. 카테고리형 데이터 및 텍스트 데이터 더미처리
3. 스케일링



```python
import pandas as pd
import numpy as np

housing = pd.read_csv('/content/drive/MyDrive/housing.csv')
```



<strong> 결측치 처리 및 데이터 삭제</strong>

```python
housing.isnull().sum()
```

출력 :  ![image-20231001162515155](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20231001162515155.png)

total_bedrooms의 207개의 결측치를 처리해야 함을 알 수 있다.

3가지 처리방법

```python
housing.dropna(subset = ['total_bedrooms']) #해당 구역을 제거(행을 제거)
housing.drop('total_bedrooms',axis = 1) #해당 피쳐값을 아예 제거(열을 제거)
housing['total_bedrooms'].fillna(housing['total_bedrooms'].median(),inplace = True) #다른 값들의 중앙값으로 대체
```



207개의 행을 삭제하여 결측치를 없애는 방법, total_bedrooms를 아예 데이터에서 지워버리는 방법 그리고 결측치있는 부분을 total_bedrooms의 중앙값(어떤 경우에는 평균값 또는 0)으로 대체하는 방법이 있다. 지금은 중앙값이 가장 적절해 보이므로 3번 방법을 선택한다.



데이터를 전반적으로 훑어본 후에 특정 피쳐값은 모델 학습에 전혀 도움이 될 것 같지 않은 특성이거나, 결측치가 너무 많아서 평균값이나 0으로 대체하기에 무리가 있다고 판단되는 데이터들은 결측치를 다른값으로 대체하기 보다는 그 피쳐값 자체를 제거하는 것이 유리할 때가 많다.



sklearn 에는 결측치 처리를 위한 클래스가 존재한다.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')
```



중앙값은 수치에만 적용할 수 있으므로 텍스트형 데이터는 제거하고 다뤄야한다.

```python
housing_num = housing.drop('ocean_proximity',axis = 1)

imputer.fit(housing_num)
X = imputer.transform(housing_num) #학습이 된 imputer를 housing_num에 적용 후 X에 저장
housing_tr = pd.DataFrame(X,columns = housing_num.columns,index = housing_num.index)
```

imputer를 학습시킨 후(각 피쳐값들의 중앙값 학습됨) X에 적용하여 결측치를 중앙값으로 대체한다. 전처리 후에는 X에 numpy에 array형태로 저장되므로 데이터 프레임으로 바꿔주려면 pd.DataFrame 함수를 사용해야 한다.



<strong>텍스트와 카테고리 특성 처리</strong>



위에서 결측치를 다뤘는데 ocean_proximity는 텍스트형 카테고리 피쳐값이다. 인공지능은 텍스트를 처리할 수 없으므로 적절한 숫자로 바꿔줘야 한다.

이는 OrdinalEncoder와 조금더 발전된 OneHotEncoder로 쉽게 구현할 수 있다.

```python
from sklearn.preprocessing import OrdinalEncoder #카테고리를 텍스트 -> 숫자로 변환해주는 클래스

ordinal_encoder = OrdinalEncoder() #문제점 : 인덱스가 가까운 두 값이 멀리 떨어진 두 값보다 비슷하다고 머신러닝 알고리즘이 인식할 수 있음.
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
```



Ordinal Encoder는 OCEAN , INLAND , ISLAND , NEAR BAY , NEAR OCEAN 을 각각 0,1,2,3,4로 변환해 준다. 그러나 학습모델은 인덱스가 가까운(예를들어 2번과 3번인 ISLAND와 NEAR BAY)피쳐 값을 연관성이 있는 것으로 본다. 이는 실제로 그런것이 아니기 때문에 이를 해결하기 위해 '더미처리' 를 한다.



더미처리는 인덱스를 0 , 1 , 2  ... 이런식으로 분류하는 것이 아닌 희소행렬을 만들어 분류한다.

예를들면 'OCEAN' 은 [1 0 0 0 0]  , 'NEAR BAY' 는 [0 0 0 1 0] 이런식으로 해당되는 피쳐값만 1로 표시하고 나머지는 0으로 표시하는 샘이다.  이렇게 인코딩 하면 위와 같이 연관성이 없는데 연관성이 있는 것 처럼 판별하는 상황을 방지할 수 있다.

```python
from sklearn.preprocessing import OneHotEncoder #OrdinalEncoder의 문제점을 해결하기 위한 OneHot Encoding
cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```



<strong>사용자 설정 변환기 만들기</strong>



여태까지는 파이썬의 sklearn에서 제공하는 표준 클래스들을 가져다가 사용하였지만, 실제 모델을 만들 때는 모델에 꼭 필요한 사용자 설정 변환기를 직접 제작해야 하는 상황이 있을 수 있다. 그래서 sklearn에서는 sklearn의 기존 변환기 클래스 들과 같은 내장함수(fit, transform, fit_transform)들을 사용할 수 있게 BaseEstimator와 TransformerMixin이 존재한다.



다음은 이를 활용하여 'rooms_per_household' 와 'population_per_house_hold' 값을 데이터에 추가하고 선택하기에 따라서 'bedrooms_per_room'을 추가할 수 있는 클래스이다. 

```python
from sklearn.base import BaseEstimator,TransformerMixin

rooms_ix,bedrooms_ix,population_ix,households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin) :
    def __init__(self,add_bedrooms_per_room = True) : #생성자 (default = add_bedrooms_per_room)
        self.add_bedrooms_per_room = add_bedrooms_per_room #단위 방개수 당 침실수
    def fit(self,X,y = None) :
        return self
    def transform(self,X) :
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix] # 단위 가구수당 방 개수
        population_per_household = X[:,population_ix]/X[:,rooms_ix] #단위 방개수 당 인구수
        
        if self.add_bedrooms_per_room :
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room] # 단위 방개수 당 침실수를 True로 선택한 경우 리턴
        else :
            return np.c_[X,rooms_per_household,population_per_household] # 단위 방개수 당 침실수를 False로 선택한 경우 리턴

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False) #객체생성
housing_extra_attribs = attr_adder.transform(housing.values)
```



<strong> 스케일링</strong>



모델을 학습시킬 때 각 피쳐값들의 스케일이 심하게 다르면 제대로 학습이 이루어 지지 않는다. 따라서 스케일을 맞춰줘야 한다. 일반적으로 StandardScaler와 MinMaxScaler를 많이 쓴다.



1. StandardScaler(정규화)

   정규화 스케일러는 기존 변수의 범위를 정규분포로 변환하는 것이다 따라서 데이터들의 평균을 0 분산을 1로 만들어 준다. 데이터의 최대 최소를 모를 때 사용하기 좋으며 이상치에 매우 민감하므로 이상치를 최대한 제거하고 사용하는 것이 좋다.

```python
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
std_data = std.fit_transform(data)
```

2. MinMaxScaler(최대최소변환)

   최대최소 스케일러는 데이터의 값들을 최소값과 최대값을 기준으로 0~1사이로 맞추는 것이다.

   정규화 스케일러보다는 이상치에 덜 민감하지만 그래도 이상치가 너무 많을 경우 학습이 제대로 이루어지지 않을 수 있으므로 최대한 이상치를 제거하고 사용해야 한다. 만약 0~1로 맞추고 싶지 않다면 feature_range 매개변수를 이용하여 값을 바꿔줄 수 있다.

```python
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
mm_data = mm.fit_transform(data)
```





<strong> 변환 파이프라인 제작</strong>



인공지능 모델을 완성하여 배포했을 경우 시간이 지남에 따라 과거의 데이터로만 학습한 모델은 예측 적중률이 떨어질 수 밖에 없다. 따라서 주기적으로 새로운 데이터를 입력하여 학습시켜야 하는데 이때마다 전처리 코드를 새로 작성할 수 없다. 따라서 새로운 데이터가 들어올 때 마다 자동으로 정해진 프로세스를 거치는 전처리 클래스가 필요하다. sklearn의 pipeline은 이런 전처리 과정을 하나의 클래스를 묶을 수 있는 클래스를 제공한다.



```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = 'median')), #객체이름(아무거나) , estimator
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num) # 결측치처리 fit_transform 후 CombinedAttributesAdder fit_transform 후 StandartScaler 의 fit_transform을 진행
```



파이프라인 안의 각 클래스들은 위에서 부터 순서대로 데이터를 fit_transform 하여 다음 클래스로 또 fit_transform하여 다음 클래스로 넘긴다.



그러나 위의 num_pipeline은 숫자형 피쳐값들에 대한 전처리 만들 제공한다.(실제로 10번째 줄에 housing_num을 매개변수로 넣었다.) 모델의 학습에 필요한 텍스트 데이터는 파이프라인을 하나 더 만들어서 따로 전처리를 해줘야 하는데, 이는 효율적이지 못하다. 따라서 sklearn에서 텍스트형 카테고리 데이터와 숫자 데이터를 동시에 전처리 할 수 있는 파이프라인을 제공한다.

```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num) #housing_num의 피쳐이름들
cat_attribs = ['ocean_proximity'] #housing 텍스트 데이터의 피쳐이름(한개임)

full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_attribs), #튜플이름(아무거나), 그에적용할파이프라인,피쳐이름
    ('cat',OneHotEncoder(),cat_attribs) 
])

housing_prepared = full_pipeline.fit_transform(housing)
```



파이프라인 내에 이름(아무거나쓰면된다), 적용할 파이프라인 이름, 피쳐이름(들) 을 차례로 쓰면 된다. 8번째 줄 같은 경우는 전처리를 할 때 OneHotEncoding(더미처리) 만 하면 되므로 굳이 파이프라인을 만들지 않고 하나의 전처리 클래스만 적어주면 된다.

cf) 파이프라인으로 전처리된 데이터는 pandas의 데이터 프레임이 아니라 numpy의 어레이 이므로 데이터 프레임으로 사용하고 싶다면 다시 변환을 해줘야 한다.