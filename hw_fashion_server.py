from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.Conv2d(28, 28, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(28, 56, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.Conv2d(56, 56, kernel_size=3, padding='same'),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.Linear = nn.Linear(56 * 7 * 7, 10)  # Fashion MNIST는 10개 클래스
    
    def forward(self, x):
        x = self.classifier(x)
        x = self.flatten(x)
        output = self.Linear(x)
        return output
    
model = ConvNeuralNetwork()
state_dict = torch.load('./model_fashion_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Fashion MNIST 클래스
CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860,), std=(0.3530,))  # Fashion MNIST 정규화 값
    ])
    
    # io.BytesIO: 바이너리 이미지 데이터를 파일처럼 다룰 수 있게 가상 메모리 객체로 변환
    # convert('L'): 이미지 색상 모드를 변경. L: 그레이 스케일
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform(image).unsqueeze(0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
        
        input_tensor = preprocess_image(image_bytes)
        print(f"input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(input_tensor)
            print(f"Model outputs: {outputs}")
            
            _, predicted = torch.max(outputs, 1)
            label = CLASSES[predicted.item()]
            confidence = torch.softmax(outputs, dim=1).max().item()
            print(f"Predicted label: {label}, Confidence: {confidence:.3f}")
        
        return JSONResponse(content={
            "label": label,
            "confidence": round(confidence, 3),
            "class_id": predicted.item()
        })
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Fashion MNIST Classification Server"}

@app.get("/classes")
async def get_classes():
    return {"classes": CLASSES}
