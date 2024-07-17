# 데이터 분석과 머신러닝 강의 정리

## 목차
1. [소개](#소개)
2. [강의 일정](#강의-일정)
3. [학습 내용](#학습-내용)
    - [1장: 데이터 분석과 머신러닝 개요](#1장-데이터-분석과-머신러닝-개요)
    - [2장: 주성분분석](#2장-주성분분석)
    - [3장: K-평균 군집화](#3장-k-평균-군집화)
    - [4장: Apriori 알고리즘](#4장-apriori-알고리즘)
    - [5장: K-최근접 이웃 알고리즘](#5장-k-최근접-이웃-알고리즘)
    - [6장: 서포트 벡터 머신](#6장-서포트-벡터-머신)
    - [7장: C5.0](#7장-c50)
    - [8장: 경사하강법](#8장-경사하강법)
    - [9장: 인공신경망과 퍼셉트론](#9장-인공신경망과-퍼셉트론)
    - [10장: 다층 퍼셉트론과 딥러닝](#10장-다층-퍼셉트론과-딥러닝)
    - [11장: 딥러닝: 회귀분석](#11장-딥러닝-회귀분석)
    - [12장: 딥러닝: 분류분석](#12장-딥러닝-분류분석)
    - [13장: 합성곱 신경망의 이미지 분류](#13장-합성곱-신경망의-이미지-분류)
4. [참고자료](#참고자료)

## 소개
이 문서는 데이터 분석과 머신러닝 강의에서 배운 내용을 정리한 것입니다. 각 장에서는 주요 주제와 내용을 요약하고, 중요 개념과 예제 코드를 포함하여 학습 내용을 체계적으로 정리했습니다.

## 강의 일정
강의는 총 13장으로 구성되어 있으며, 각 장마다 다양한 데이터 분석 및 머신러닝 기법을 다룹니다. 주차별로 학습할 내용을 다음과 같이 정리했습니다.

## 학습 내용

### 1장: 데이터 분석과 머신러닝 개요
- **주제**: 데이터 분석과 머신러닝의 기본 개념과 흐름
- **내용 요약**:
  - 데이터 분석과 머신러닝의 역사, 종류, 응용 분야
  - 데이터 전처리, 모델 훈련, 검증, 평가
- **중요 개념**:
  - 데이터 전처리: 결측치 처리, 스케일링
  - 모델 훈련: 학습 데이터, 테스트 데이터
  - 모델 검증: 교차 검증, 성능 평가 지표

---

### 2장: 주성분분석
- **주제**: 데이터 차원 축소 기법
- **내용 요약**:
  - 주성분분석(PCA)의 원리와 활용
  - 고차원 데이터를 저차원으로 변환하여 데이터 시각화 및 분석
- **중요 개념**:
  - 공분산 행렬: 데이터의 분산과 공분산을 나타내는 행렬
  - 고유벡터와 고유값: 공분산 행렬을 분해하여 주요 성분을 추출
  - 차원 축소: 데이터의 중요한 특징을 보존하면서 차원을 줄임

---

### 3장: K-평균 군집화
- **주제**: 데이터 군집화 기법
- **내용 요약**:
  - K-평균 알고리즘의 원리와 활용
  - 데이터를 K개의 군집으로 나누어 군집 중심점을 찾음
- **중요 개념**:
  - 중심점: 각 군집의 중심을 나타내는 점
  - 거리 측정: 데이터 포인트와 중심점 사이의 거리 계산
  - 군집 할당: 데이터 포인트를 가장 가까운 중심점에 할당
- **예제코드**:
    ```python
    # 군집의 수를 2개로 하는 군집화 객체
    kmeans = KMeans(n_clusters=2, n_init=10)
    # 2, 3열을 이용한
    kmeans.fit(data_std)
    # 군집 라벨
    print(kmeans.labels_)
    # 군집별 군집 중심
    print(kmeans.cluster_centers_)
    ```
---

### 4장: Apriori 알고리즘
- **주제**: 연관 규칙 학습 기법
- **내용 요약**:
  - Apriori 알고리즘의 원리와 활용
  - 데이터 내의 항목 간의 연관 관계를 발견하여 규칙을 생성
- **중요 개념**:
  - 지지도: 특정 항목 집합이 전체 데이터에서 나타나는 빈도
  - 신뢰도: 규칙의 신뢰성을 나타내는 지표
  - 향상도: 항목 간의 연관성을 나타내는 지표
- **예제코드**:
    ```python
    # 데이터 프레임으로 변환
    df = pd.DataFrame(tran_ar, columns=te.columns_)
    print(df)

    # 각 상품별 거래 빈도
    freq = df.sum().to_frame('Frequency')
    # 빈도 역순으로 정렬
    freq_sort = freq.sort_values('Frequency', ascending=False)
    print(freq_sort)

    # 지지도
    freq_sort['Support'] = freq_sort['Frequency'] / len(freq_sort)
    print(freq_sort)
    ```
---

### 5장: K-최근접 이웃 알고리즘
- **주제**: 분류 기법
- **내용 요약**:
  - K-최근접 이웃(KNN) 알고리즘의 원리와 활용
  - 새로운 데이터 포인트를 가장 가까운 K개의 이웃으로 분류
- **중요 개념**:
  - 거리 측정: 데이터 포인트 간의 유사성 계산
  - K 값: 이웃의 개수를 나타내는 파라미터
  - 다수결 투표: 가장 많은 이웃이 속한 클래스로 분류
- **예제코드**:
    ```python
    # 모형화
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    # 학습
    knn.fit(x_train_std, y_train)
    ```
---

### 6장: 서포트 벡터 머신
- **주제**: 분류 및 회귀 기법
- **내용 요약**:
  - 서포트 벡터 머신(SVM)의 원리와 활용
  - 데이터 포인트를 분류하는 최적의 초평면을 찾음
- **중요 개념**:
  - 초평면: 데이터를 두 개의 클래스로 나누는 선형 경계
  - 서포트 벡터: 초평면에 가장 가까운 데이터 포인트
  - 커널 함수: 비선형 분류를 위한 함수 변환

---

### 7장: C5.0
- **주제**: 결정 트리 알고리즘
- **내용 요약**:
  - C5.0 알고리즘의 원리와 활용
  - 결정 트리를 사용하여 데이터 분류 및 예측
- **중요 개념**:
  - 정보 이득: 트리 분할 시 얻는 정보의 양
  - 가지치기: 트리의 복잡도를 줄이는 과정
  - 불순도: 데이터의 혼합 정도를 나타내는 지표
- **예제코드**:
    ```python
    # C5.0 모형 설정
    clf = tree.DecisionTreeClassifier(criterion="entropy")

    # 학습
    clf = clf.fit(X_train, Y_train)
    ```
---

### 8장: 경사하강법
- **주제**: 최적화 기법
- **내용 요약**:
  - 경사하강법의 원리와 활용
  - 비용 함수를 최소화하여 모델 파라미터를 최적화
- **중요 개념**:
  - 학습률: 파라미터 업데이트 시의 스텝 크기
  - 비용 함수: 모델의 예측 오차를 나타내는 함수
  - 기울기: 비용 함수의 변화율을 나타내는 벡터
- **예제코드**:
    ```python
    # 확률적 경사하강법 객체 생성
    model = SGDRegressor(verbose=1)

    # 모형 학습
    model.fit(X_train, Y_train)
    ```
---

### 9장: 인공신경망과 퍼셉트론
- **주제**: 신경망 기초
- **내용 요약**:
  - 인공신경망과 퍼셉트론의 원리와 활용
  - 뉴런과 가중치를 사용하여 데이터의 패턴을 학습
- **중요 개념**:
  - 뉴런: 신경망의 기본 단위
  - 활성화 함수: 뉴런의 출력을 결정하는 함수
  - 퍼셉트론: 단층 신경망 모델
- **예제코드**:
    ```python
    
    ```
---

### 10장: 다층 퍼셉트론과 딥러닝
- **주제**: 심층 신경망
- **내용 요약**:
  - 다층 퍼셉트론(MLP)과 딥러닝의 원리와 활용
  - 여러 개의 은닉층을 가진 신경망 모델
- **중요 개념**:
  - 은닉층: 입력과 출력을 연결하는 중간 계층
  - 역전파: 오류를 출력에서 입력 방향으로 전파하여 가중치 업데이트
  - 활성화 함수: 비선형성을 도입하여 복잡한 패턴 학습
- **예제코드**:
    ```python
    # 모형화
    model = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', solver='lbfgs', max_iter=100)
    # 학습
    model.fit(X, y)
    ```
---

### 11장: 딥러닝: 회귀분석
- **주제**: 딥러닝을 활용한 회귀분석
- **내용 요약**:
  - 회귀분석을 위한 딥러닝 모델 구축 및 평가
  - 연속형 변수 예측을 위한 딥러닝 기법
- **중요 개념**:
  - 손실 함수: 예측값과 실제값 간의 차이를 나타내는 함수
  - 최적화 알고리즘: 모델 파라미터를 조정하는 방법
  - 모델 평가: 예측 성능을 평가하는 지표
- **예제코드**:
    ```python
    # 모형 구조
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(12,)))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))

    # 모형 구성
    model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mse'])

    # 학습
    results = model.fit(X_train_norm, y_train_norm,
                       validation_data=(X_test_norm, y_train_norm),
                       epochs=200, batch_size=32)
    ```
---

### 12장: 딥러닝: 분류분석
- **주제**: 딥러닝을 활용한 분류분석
- **내용 요약**:
  - 분류를 위한 딥러닝 모델 구축 및 평가
  - 이진 분류 및 다중 분류 문제 해결
- **중요 개념**:
  - 분류 손실 함수: 분류 문제에서 사용하는 손실 함수
  - 최적화 알고리즘: 모델 파라미터를 조정하는 방법
  - 모델 평가: 분류 성능을 평가하는 지표
- **예제코드**:
    ```python
    # 모형 구조
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 모형 구성
    model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

        
    # 학습
    results = model.fit(X_train_norm, y_train_class,
                       validation_data=(X_test_norm, y_test_class),
                       epochs=100, batch_size=128)
    ```
---

### 13장: 합성곱 신경망의 이미지 분류
- **주제**: 합성곱 신경망(CNN) 기법을 활용한 이미지 분류
- **내용 요약**:
  - 합성곱 신경망의 구조와 이미지 분류 응용
  - 이미지 데이터의 특징 추출 및 분류
- **중요 개념**:
  - 합성곱 층: 이미지의 특징을 추출하는 층
  - 풀링 층: 이미지 크기를 축소하여 계산량을 줄이는 층
  - 이미지 전처리: 이미지 데이터를 신경망에 입력하기 전에 처리하는 방법
- **예제코드**:
    ```python
    # 모형 구조
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),
                    activation='relu',
                    input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0, 5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0, 5))
    model.add(Dense(10, activation='softmax'))

    # 모형 구성
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    # 학습
    results = model.fit(X_train_norm, y_train_class,
                   validation_data=(X_test_norm, y_test_class),
                   epochs=50, batch_size=128)
    ```
---

## 참고자료
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Keras Documentation](https://keras.io/api/)
- 파이썬 데이터분석 교재
