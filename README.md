<img width="1919" height="813" alt="image" src="https://github.com/user-attachments/assets/98921ea8-5fd7-4164-925a-d20463cc1bb1" /># Fashion MNIST 의류 분류 프로젝트

Fashion MNIST 데이터셋을 사용한 의류 이미지 분류 시스템입니다. FastAPI 백엔드와 Gradio 프론트엔드로 구성되어 있습니다.

## 프로젝트 구조

```
FashionMNIST/
├── hw_fashion_classifier.py    # 모델 훈련 코드
├── hw_fashion_server.py        # FastAPI 서버
├── hw_fashion_client.py        # Gradio 클라이언트
├── hw_fashion_imagemake.py     # CSV를 이미지로 변환
├── model_fashion_weights.pth   # 훈련된 모델 가중치
└── fashion/                    # Fashion MNIST 데이터셋
```

## 기능

- **10개 의류 카테고리 분류**:

  - T-shirt/top, Trouser, Pullover, Dress, Coat
  - Sandal, Shirt, Sneaker, Bag, Ankle boot

- **웹 인터페이스**: Gradio를 사용한 사용자 친화적 인터페이스
- **실시간 분류**: 이미지 업로드 시 즉시 분류 결과 제공
- **신뢰도 표시**: 예측 결과의 신뢰도를 백분율로 표시

## 웹 인터페이스
<img width="1919" height="813" alt="image" src="https://github.com/user-attachments/assets/24a174a0-01b7-4f90-9b49-60fc004fc157" />


웹 인터페이스는 다음과 같은 기능을 제공합니다:

- 의류 이미지 업로드 및 드래그 앤 드롭
- 실시간 분류 결과 표시
- 신뢰도 및 클래스 ID 표시
- 예제 이미지 제공

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install torch torchvision fastapi uvicorn gradio pillow pandas numpy
```

### 2. 서버 실행

```bash
uvicorn hw_fashion_server:app --reload --host 0.0.0.0 --port 8000
```

### 3. 클라이언트 실행

```bash
python hw_fashion_client.py
```

## 모델 구조

- **CNN 아키텍처**: 2개의 Conv2d 블록과 MaxPool2d, Dropout 사용
- **입력**: 28x28 그레이스케일 이미지
- **출력**: 10개 클래스에 대한 확률 분포
- **정규화**: Fashion MNIST 데이터셋에 맞는 정규화 적용

## API 엔드포인트

- `POST /classify`: 이미지 분류
- `GET /`: 서버 상태 확인
- `GET /classes`: 지원하는 클래스 목록

## 사용 예시

1. 웹 브라우저에서 Gradio 인터페이스 접속
2. 의류 이미지 업로드
3. 자동으로 분류 결과 확인
4. 신뢰도와 클래스 정보 확인

## 기술 스택

- **백엔드**: FastAPI, PyTorch
- **프론트엔드**: Gradio
- **모델**: CNN (Convolutional Neural Network)
- **데이터셋**: Fashion MNIST

## 라이선스

MIT License
