# 1) 💡 Convolutional Neural Network
저번 MNIST 예제 구현 때는 dense layer로만 추론했었는데, (그럼에도 97퍼센트의 accuracy.. 😅)  
이번에는 CNN을 한번 사용해보려고 한다.  
솔직히 CNN은 아직 배우지 않았는데 예제 구현 정도는 해보는 게 좋을 거 같고 Incepction v1 등 기초 논문들도 구현해보려면 진도 기다리다가 한 세월일 거 같아서 먼저 시작한다.  

![image](https://github.com/dpcivl/playground/assets/95332280/7fdab492-dd1e-4123-8786-f08b7b1b4be2)  
(출처: https://www.analyticsvidhya.com/blog/2022/01/convolutional-neural-network-an-overview/)

## 1.1> 출력 크기 계산 공식
- 입력 크기(I) : 입력 레이어의 너비
- 필터 크기(F) : 컨볼루션 필터(커널)의 너비
- 스트라이드(S) : 필터가 입력 데이터를 통해 이동하는 간격
- 패딩(P) : 입력 데이터 주변에 추가된 0의 두께
- 출력 크기(O) : 필터를 통해 생성되는 출력 레이어의 너비

$$
O = {{I - F + 2P} \over S} +1
$$  

## 1.2> 🤔 컨볼루션 필터는 어떻게 장만할까?
### 1.2.1] 필터 크기
일반적으로 사용되는 크기는 $3\times3$, $5\times5$이다.  
작은 필터를 여러 층에 겹쳐 쌓는 것은 큰 필터를 한 개 사용하는 것보다 더 많은 비선형성을 모델에 추가하고, 더 복잡한 특징을 학습할 수 있게 한다.  

프레임워크를 사용하지 않고 필터를 만든다면 아래의 과정을 거친다.  
```
1. 필터의 가중치를 수동으로 초기화
2. 입력 이미지에 대해 합성곱 연산
3. 필터 적용 및 특징맵 생성
```
🔼 위의 과정은 기본적인 필터를 세팅하는 것이고, 최적화까지 하려면 더 많은 과정들이 포함된다.  

### 1.2.2] 스트라이드
주로 1이나 2를 사용한다.  
스트라이드 1은 입력과 출력의 공간적 크기를 유지하려고 할 때 사용된다.  
스트라이드 2는 feature map의 크기를 줄이고 모델의 계산량을 감소시킬 때 사용된다.  

💡 과도한 스트라이드는 정보 손실을 초래할 수 있다.  

### 1.2.3] 패딩
- same 패딩 : 입력과 출력의 공간적 크기를 동일하게 유지할 때 사용
- valid 패딩 : 패딩 없이 순수한 입력 데이터만 사용하고자 할 때 사용

# 2) 구현 예제 네트워크 분석
```python
# 모델 구성
model = models.Sequential([
    # Conv2D와 MaxPooling2D 레이어들
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense 레이어로 네트워크 완성
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```
## 2.1> ✅ 레이어 크기 계산하기
### 2.1.1] 1st conv layer
첫번째 conv layer의 필터 크기는 $3\times3$이고 32개의 필터를 가진다. 스트라이드나 패딩은 따로 적용하지 않았고, 활성화 함수로는 relu를 사용했다.  

출력 feature map의 크기를 계산하면 아래와 같다.  
$$
O = {{28 - 3 + 0} \over 1} +1 = 26
$$

즉, $26\times26\times32$개의 feature map이 생성된다.  

### 2.1.2] 1st max pooling layer
첫번째 max pooling layer에서 풀링 윈도우 크기는 $2\times2$이다.  
conv layer의 feature map 크기를 계산한 것과 같이 계산해주면 출력 feature map을 알 수 있다.  

$$
O = {{26 - 2}\over 2} + 1 = 13
$$

💡 여기서 중요한 것은 max pooling layer를 구성할 때 별다른 정의가 없으면, 풀링 윈도우의 크기와 같은 값으로 스트라이드를 정의한다는 것이다. 이로 인해서 pooling layer는 feature map의 spatial size를 반으로 줄이는 효과를 가진다.  

어쨌거나, max pooling layer를 통과한 후의 feature map의 크기는 $13\times13\times26$이 된다.  

### 2.1.3] 이후 레이어
- 2번째 conv layer를 지난 후 ➡️ $10 \times 10 \times 64$
$$
O = {{13 - 3} \over 1} + 1 = 10
$$
- 2번째 max pooling layer를 지난 후 ➡️ $5 \times 5 \times 64$
$$
O = {{10 - 2} \over 2} + 1 = 5
$$
- 3번째 conv layer를 지난 후 ➡️ $3 \times 3 \times 64$
$$
O = {{5 - 3} \over 1} + 1 = 3
$$
- flattening ➡️ 576개의 뉴런이 생성
$$
3 \times 3 \times 64 = 576
$$
- 1번째 dense layer를 지난 후 ➡️ 64개의 뉴런이 생성
- 2번째 dense layer를 지난 후 ➡️ 10개의 뉴런이 생성

# 3) Dense Network와 CNN 비교
# 3.1> 파라미터 개수 계산
1. Dense layer  
🔽 아래는 Dense layer의 파라미터 개수를 구하는 공식이다.  
$$
P = (I + 1) \times O \\
P: 파라미터\space 개수 \\
I : 입력\space유닛\space수 \\
O : 출력\space유닛\space수
$$

- 첫번째 Dense layer
  - 입력 유닛 수:784(input_shape)
  - 출력 유닛 수:256
  - 파라미터 개수:$(784+1) \times 256=200960$
- 두번째 Dense layer
  - 입력 유닛 수:256(이전 레이어 출력 유닛 수)
  - 출력 유닛 수:128
  - 파라미터 개수:$(256+1) \times 128 = 32896$
- 세번째 Dense layer
  - 입력 유닛 수:128
  - 출력 유닛 수:10
  - 파라미터 개수:$(128+1) \times 10 = 1290$

총 235,146개의 파라미터.  

2. Convolutional layer  
🔽 아래는 Convolutional layer의 파라미터 개수를 구하는 공식이다.  
$$

$$

