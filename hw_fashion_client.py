# Gradio frontend for Fashion MNIST
import gradio as gr
import requests
import io

def classify_with_backend(image):
    url = "http://127.0.0.1:8000/classify"
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    
    try:
        response = requests.post(url, files={"file": ("image.png", image_bytes, "image/png")})
        if response.status_code == 200:
            result = response.json()
            label = result.get("label", "Error")
            confidence = result.get("confidence", 0)
            class_id = result.get("class_id", -1)
            
            return f"예측 결과: {label}\n신뢰도: {confidence:.1%}\n클래스 ID: {class_id}"
        else:
            return f"오류 발생: {response.status_code}"
    except Exception as e:
        return f"연결 오류: {str(e)}"

# Fashion MNIST 클래스 정보
fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class_info = "\n".join([f"{i}: {cls}" for i, cls in enumerate(fashion_classes)])

iface = gr.Interface(
    fn=classify_with_backend,
    inputs=gr.Image(type="pil", label="의류 이미지를 업로드하세요"),
    outputs=gr.Textbox(label="분류 결과", lines=3),
    title="Fashion MNIST 의류 분류기",
    description="의류 이미지를 업로드하면 해당 의류의 종류를 분류해드립니다!",
    article="""
    ## 지원하는 의류 종류:
    0: T-shirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot
    
    ## 사용 방법:
    1. 의류 이미지를 업로드하거나 드래그 앤 드롭하세요
    2. 자동으로 분류가 진행됩니다
    3. 예측 결과와 신뢰도를 확인하세요
    """
)

if __name__ == "__main__":
    iface.launch(share=True)
