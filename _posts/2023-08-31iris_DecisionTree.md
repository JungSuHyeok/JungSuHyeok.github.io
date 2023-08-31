<h1>붓꽃 품종분류</h1>

### sklearn의 DecisionTreeClassifier를 이용하여 붓꽃의 데이터를 학습시켜 3가지 품종중 어떤 품종인지를 예측하는 모델

```python
from sklearn.datasets import load_iris #붓꽃 Dataset을 사용하기 위함
from sklearn.tree import DecisionTreeClassifier #결정트리를 사용하기 위함
from sklearn.model_selection import train_test_split #Train set 과 Test set을 구분하기 위함
import pandas as pd
```



###### 붓꽃데이터 다운로드

```python
iris = load_iris() #sklearn에서 붓꽃 데이터를 받음
iris_data = iris.data #붓꽃의 여러가지 특성값들에 대한 수치
iris_label = iris.target #위의 수치에 맵핑되는 붓꽃종류(결정 값)
```

붓꽃의 4가지 feature값들이 들어있는 iris.data

붓꽃의 label값들이 들어있는 iris.target



###### sklearn.datasets에 있는 내장 함수들

iris.data , iris.label , iris.feature_names , iris.target_names

**iris.data** : iris의 특성값 데이터 (2차원 array)

**iris.label** : iris의 결정값 데이터 (1차원 array)

**iris.feature_names** : iris의 특성값 이름 ex) ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

**iris.target_names** : iris의 결정값 이름 ex) array(['setosa', 'versicolor', 'virginica'], dtype='<U10')



###### Numpy의 array로된 붓꽃 데이터를 Pandas의 데이터프레임으로 바꾸기

```python
iris_df = pd.DataFrame(data = iris_data,columns = iris.feature_names)
```



데이터프레임 마지막 열에 결정값을 추가

```python
iris_df['labels'] = iris.target
```



###### Train set과 Test set 나누기

```python
ftr_df = iris_df.iloc[:,:-1] #특성 데이터    
tgt_df = iris_df.iloc[:,-1] #결과 데이터

X_train,X_test,y_train,y_test = train_test_split(ftr_df,tgt_df,test_size = 0.2,random_state = 11)
```

test_size : Test set을 전체의 몇 %로 할 것인지를 정해줌

random_state : 어떤 Data를 Test set으로 할지 랜덤하게 정하는 것에 대한 seed



###### DecisionTreeClassifier로 모델 학습시키기

```
dt_clf = DecisionTreeClassifier(random_state = 11)

dt_clf.fit(X_train,y_train)
```

X_train , y_train 데이터로 fit함수로 학습시킴



###### 예측 결과 및 정확도

```python
pred = dt_clf.predict(X_test)
print(pred)
```

[2 2 1 1 2 0 1 0 0 1 1 1 1 2 2 0 2 1 2 2 1 0 0 1 0 0 2 1 0 1]



predict 함수를 이용해 X_test에 대한 결과값을 예측

```python
from sklearn.metrics import accuracy_score
print('예측 정확도 : {0:.4f}'.format(accuracy_score(y_test,pred)))
```

예측 정확도 : 0.9333



정확도 93.33%로 적합하게 학습된 모델임 알 수 있음.
