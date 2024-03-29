import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

dataset_path = 'C:/Users/dpciv/Desktop/hmt_hi/5/workspace/playground/mnist/dataset/'

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # 첫 16바이트는 매직 넘버, 이미지 수, 행 수, 열 수 정보를 담고 있습니다.
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 이미지 데이터를 [이미지 수, 행, 열] 형태로 재구성합니다.
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # 첫 8바이트는 매직 넘버와 레이블 수 정보를 담고 있습니다.
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 파일 경로 설정
# MNIST 데이터셋 파일들을 해당 경로에 위치시키세요.
train_images_path = dataset_path + 'train-images.idx3-ubyte'
train_labels_path = dataset_path + 'train-labels.idx1-ubyte'
test_images_path = dataset_path + 't10k-images.idx3-ubyte'
test_labels_path = dataset_path + 't10k-labels.idx1-ubyte'

# 데이터 로드
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# 모델 구성
model = models.Sequential([
    # Conv2D와 MaxPooling2D 레이어들
    layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='sigmoid'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='sigmoid'),
    
    # Dense 레이어로 네트워크 완성
    layers.Flatten(),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 이미지 데이터의 차원을 조정해야 합니다. MNIST 데이터는 28x28 크기이며, 채널이 1입니다.
# 따라서 데이터를 (28, 28, 1)의 형태로 reshape해야 합니다.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 이미지 데이터 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델 훈련
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)