- - -
# 1) 개요
해당 레포지토리에서는 딥러닝의 기초를 배우기 위한 예제 프로젝트를 진행하려고 한다.  
유명한 데이터셋이나 네트워크를 직접 구현하면서 여러 경험들을 할 수 있을 것으로 기대된다.  

## 1.1> 시도해봐야 하는 것
- [x] MNIST
- [ ] Oxford-IIIT Pet Dataset
- [ ] COCO 
- [ ] Inception V1
- [ ] Object Detection
- [ ] Segmentation
- [ ] Labeling to img
- [ ] 시각화 도구 사용해보기

# 2) 수행 예제
## 2.1> MNIST dataset & Image Classification
- Kaggle에서 MNIST 데이터셋 다운로드
- 다운로드한 데이터셋을 불러와서 shape 확인
- MNIST 데이터를 연산할 네트워크 만들기
- 이미지 데이터 평탄화(`Dense` layer는 2차원 입력이 필요)
- `Sequential`을 통해 네트워크 구성
- Test accuracy: 97.8%

### 2.1.1] CNN 사용해보기
- [CNN 관련 공부 내용 정리](./mnist/docs/wil.cnn.md)
- Dense layer와 CNN layer 파라미터 개수 비교

## 2.2> Oxford-IIIT Pet dataset & Inception V1
- Kaggle에서 Oxford-IIIT Pet dataset 다운로드
- 

# 기록
2024년 2월 6일 -> 레포지토리 생성, MNIST 예제 실행