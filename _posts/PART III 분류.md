<h1>분류 시스템</h1>

0~9를 손글씨로 쓴 70000개의 Dataset을 바탕으로 어떤 숫자인지 분류하는 모델 제작한다.



<h3>0~9를 써놓은 손글씨 중 5인지 아닌지를 구분하는 모델 제작</h3>

Dataset 다운받기

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784',version = 1,as_frame = False)
```



mnist 데이터에서 feature값과 label값 나누기

```python
X,y = mnist['data'] , mnist['target']
```



mnist 데이터들 중 하나 이미지로 띄워보기

```python
import matplotlib.pyplot as plt

some_digit = X[0] #70000개의 이미지 중 0번째 이미지
some_digit_image = some_digit.reshape(28,28) #784개의 1차원 리스트로 돼있는 데이터를 28*28 데이터로 재생성

plt.imshow(some_digit_image,cmap = 'binary')
plt.show()
```

<img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20230917175337335.png" alt="image-20230917175337335" style="zoom: 67%;" />

```python
print(y[0])
```

출력 : '5'



label 데이터 (y값) 처리를 용이하게 하도록 문자열로 돼있는 데이터들을 정수형으로 바꾸기

```python
import numpy as np

y = y.astype(np.uint8)
```



Train Set 과 Test Set 을 나누기(sklearn의 mnist 데이터셋은 앞의 60000개의 데이터가 train set 뒤의 10000개의 데이터가 test set으로 이미 분류되어 있다.)

```python
X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]
# [:60000] = [0:60000] , [60000:] = [60000:70000] 을 의미함
```



label set(y)에서 5인 데이터와 5가아닌 데이터를 구분한 새로운 label set 제작

```python
y_train_5 = (y_train == 5) #y_train 중 5인 데이터는 1로 5가 아닌 데이터는 0으로 바뀐 label set 생성
y_test_5 = (y_test == 5)
```



분류기 객체인 SGDClassifier 선언(SGDClassifier는 선, 면, 초평면을 통하여 이진분류, 다진분류 모두 가능하다.)

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier() #모델 객체선언
sgd_clf.fit(X_train,y_train_5) #학습

sgd_clf.predict([some_digit]) #5인 데이터 입력후 예측실행
```

출력 : array([ True])



모든 Train Set을 다시 Train Set 과 Test Set으로 나누어 교차검증을 실시

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits = 3,shuffle = True) #교차검증 객체선언
#적절한 데이터들로 자동 선정되어 3겹 교차검증을 실시하도록 데이터의 index를 반환해주는 객체
for train_index , test_index in skfolds.split(X_train,y_train_5) :
    #train_index 와 test_index를 3번 다르게 지정해주는 반복문(교차검증용 인덱스)
    
    clone_clf = clone(sgd_clf) #sgd_clf 객체 복사(학습되기 전)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds,y_train_folds) #학습
    y_pred = clone_clf.predict(X_test_folds) #교차검증에서 test set으로 분류된 데이터로 예측
    
    n_correct = sum(y_pred == y_test_folds) #정답을 맞힌 데이터 개수의 합
    print(n_correct/len(y_pred)) #정확도를 출력(맞힌데이터 / 전체데이터개수)
```

출력 : 0.96575 0.9606 0.95705



위의 교차검증을 편하게 하기(단, 세부적인 튜닝을 할 때는 위의 방법을 사용해야함)

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf,X_train,y_train_5,cv = 3,scoring = 'accuracy')
#cross_val_score(모델객체 , feature data , label data , fold횟수 , scoring='accuracy')
```

출력 : array([0.95035, 0.96035, 0.9604 ])



<strong> 정확도를 분류기 모델의 성능 지표로 선호하지 않는 이유</strong>

 위의 전략적kfold , cross_val_score 로 모델을 평가해본 결과 정확도가 95%이상이다. 이를 보면 위의 모델은 완벽에 가까운 고성능 모델이라고 볼 수 있다. 하지만 다음 예시를 보면 그렇지 않음을 알 수 있다.

다음은 어떤 X,y가 들어 오더라고 무조건 '5가 아니다' 라고 예측하는 모델 클래스를 사용자 정의 클래스로 만들어 보겠다.

```python
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator) :
    def fit(self,X,y = None) :
        return self
    def predict(self,X) :
        return np.zeros((len(X),1),dtype = bool)
    
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf,X_train,y_train_5,cv = 3,scoring = 'accuracy')
```

출력 : array([0.91125, 0.90855, 0.90915])



전체 Data set에서 5가아닌 데이터가 애초에 90%정도 되기 때문에 어떤 입력이 들어와도 학습도 하지 않은 채 '5가 아닙니다' 라고 하면 이에 대한 accuracy는 90% 이다. 따라서 다른 평가 지표가 필요함을 알 수 있다.



오차가 어떤식으로 발생하는지 한눈에 보기 위해 오차행렬을 제작한다

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5,y_train_pred)
```

출력 : array([[53892,   687],       

​						[ 1891,  3530]])

이 오차행렬은 
$$
M = \begin{pmatrix} 진짜음성 & 거짓양성 \\ 거짓음성 & 진짜양성
\end{pmatrix}
$$
이렇게 나타난다 . 이상적인 모델(100%정확한)의 오차행렬은 대각행렬이 된다.(주대각원소를 제외한 모든 원소 = 0)



다음은 위에서 말한 정확도 대신 분류기의 성능을 평가하기 위한 지표 2가지인 <strong>정밀도</strong>와 <strong>재현율</strong>이다.

```python
from sklearn.metrics import precision_score ,recall_score #정밀도와 재현율

print(precision_score(y_train_5,y_train_pred)) #정밀도
print(recall_score(y_train_5,y_train_pred)) #재현율
```

출력 : 0.8370879772350012

​		   0.6511713705958311



<strong>정밀도 , 재현율 , 정확도를 오차행렬의 원소값으로 수식정리 및 의미부여</strong>
$$
(precision) = \frac{TP}{TP + FP} , (Recall) = \frac{TP}{TP + FN}

, (Accuracy) = \frac{TP + TN}{TP + FN + FP + TN}
$$
(T = True , F = False , P = Positive , N = Negative)

정밀도가 높다 : 진짜 5를 구분을 못해내더라도 5라고 판단한 집단 중에서는 오류가 거의 없는 경우

재현율이 높다 : 5라고 판단한 집단에 오류가 있더라고 5를 최대한 많이 포함시킨 경우



정밀도와 재현율은 트레이드 오프 관계(한쪽이 높아지면 한쪽이 낮아짐)이다.



정밀도가 높아야 하는 경우는 어린아이에게 유익한 동영상을 선정해주는 모델이 있다고 하면, 어린아이에게 도움이 되는 영상이 많이 배제가 된다고 할 지언정 어린아이에게 적합하다고 판단되는 영상의 집합에는 폭력적이거나 선정적인 영상은 반드시 배제되어야 한다. 이럴경우 재현율보다는 정밀도가 높은 모델이 좋은 모델이다.



재현율이 높아야 하는 경우는 CCTV로 도둑을 잡아내는 모델이 있다고 하면, 많은 사람들 중 도둑이 아닌 사람을 도둑으로 오인하는 경우가 많다고 하더라도 일단 도둑으로 의심되는 사람들을 모두 도둑으로 체크하여 확인하는 것이 범죄를 막을 확률이 높다. 이럴경우 정밀도 보다는 재현율이 높은 모델이 좋은 모델이다.



정밀도와 재현율을 같이 사용하여 모델의 성능을 평가하는 F1 지표가 있다. 이는 정밀도와 재현율의 조화평균으로 다음과 같은 방법으로 계산한다.
$$
F1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}}
$$
f1_score라는 함수를 임포트하여 출력 가능하다.

```python
from sklearn.metrics import f1_score
y_train_knn_pred = cross_val_predict(knn_clf,X_train,y_multilabel,cv = 3)
f1_score(y_multilabel,y_train_knn_pred,average = 'macro')
```

출력 : 0.742962043663375



Support vector machine (다중분류기) 를 통하여 임의의 숫자를 0~9의 숫자로 예측하기

```python
from sklearn.svm import SVC #서포트 벡터 머신 -> 다중분류기 0~9까지는 분류해냄 (다음과 같은 경우 정확히 5로 분류)

svm_clf = SVC()
svm_clf.fit(X_train,y_train)
svm_clf.predict([some_digit]) #5라고 정확히 예측함
```

출력 : array([5], dtype=uint8)



<strong> OvA(OvR) 전략 , OvO전략</strong>

OvA는 one-verse-all 라는 뜻으로 이미지를 분류할 때 각 클래스중 가장 큰 가중치를 가진 클래스를 답으로 채택하는 전략입니다.



OvO 는 one-verse-one 이라는 뜻으로 이미지를 분류할 때 각 클래스 별로 이진 분류를 진행하는 것 으로 손글씨 분류 모델을 예로 들면, 0과 1구별 0과 2구별 0과 3구별 ... 5와 8구별 ... 8과 9를 이진분류하여 클래스가 N개일 경우 N*(N-1)/2번의 분류를 사용해야 합니다.



SVC는 OvA 로 학습을 할지 OvO 로 학습을 할지 데이터의 특성에 따라 알아서 현명하게 적용을 해줍니다. 그러나, 다음과 같은 방법으로 OvO 나 OvA 전략을 사용자 설정으로 강요할 수 있습니다.



```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC()) #SVC분류기를 OVO 베이스가 아닌 OVR 베이스로함

ovr_clf.fit(X_train,y_train)
ovr_clf.predict([some_digit])
```

출력 : array([5], dtype=uint8)



SVC 같은 경우 다중 분류기이기 때문에 각각의 예측을 하게된 근거인 가중치를 가진다. 다음은 이를 출력하는 함수이다.

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
```

출력 : array([[ 1.72501977,  2.72809088,  7.2510018 ,  8.3076379 , -0.31087254,         9.3132482 ,  1.70975103,  2.76765202,  6.23049537,  4.84771048]])



5가 9.31로 가장 큰 가중치를 가졌다는 것을 볼 수 있고, 그다음은 3이 7.25로 두번째로 높은 것을 알 수 있다. 따라서 이 모델은 some_digit가 5라고 판단했고, 3은 의심하고 있다는 것을 알 수 있다.



다중분류는 SVC 뿐만 아니라 SGD , RandomForest 를 통해서도 가능하다.



Scaler를 통하여 데이터를 조정한 후의 정확도

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #정규화를 하기 위한 '자' 선언
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
#X_train.astype(np.float64) 를 기준으로 X_train_scaled 에 정규화를 적용
cross_val_score(sgd_clf,X_train_scaled,y_train,cv = 3,scoring = 'accuracy')
```

출력 : array([0.8983, 0.891 , 0.9018]) 

정규화 전 : array([0.87365, 0.85835, 0.8689 ]) 이랑 비교했을 때 정확도가 많이 개선됐음을 알 수 있다.



직관적인 오차분석을 위한 오차행렬 만들기

```python
y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv = 3)

conf_mx = confusion_matrix(y_train,y_train_pred)
conf_mx
```

출력 : <img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20230919200240082.png" alt="image-20230919200240082" style="zoom: 80%;" />



n행 m열 원소의 값이 숫자n을 m으로 예측한 가중치이다.(이상적인 경우 대각행렬이된다.)

오차행렬 이미지로 띄우기

```python
plt.matshow(conf_mx,cmap = plt.cm.gray)
plt.show()
```

<img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20230919200408940.png" alt="image-20230919200408940" style="zoom:67%;" />

다른 항목은 정규화만 진행 후 주대각원소값만 0으로 바꾸어 오차행렬 재출력(이 경우 어떤 수를 어떤 수로 모델이 헷갈려하는 지 파악하기 쉽다.)

```python
row_sums = conf_mx.sum(axis = 1,keepdims = True)
norm_conf_mx = conf_mx/row_sums

np.fill_diagonal(norm_conf_mx,0) #주대각원소 0 인 행렬 만들기
plt.matshow(norm_conf_mx,cmap = plt.cm.gray)
plt.show()
```

<img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20230919201629004.png" alt="image-20230919201629004" style="zoom: 67%;" />

주대각원소를 0으로 바꾸고 정규화를 하면 여러 숫자들(특히 5)이 8로 많이 혼동되었음을 알 수 있다. 이 경우 임의의 숫자를 8로 혼동하지 않도록 8데이터를 증식시켜서 추가 학습을 시키거나 새로운 알고리즘을 도입하여 어떤 숫자를 8로 혼동하지 않도록 하는 것 모델 성능 향상에 도움이 된다.
