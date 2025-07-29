import pandas as pd
import numpy as np
from PIL import Image
import os

# test_images 폴더 생성
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# CSV 파일 읽기
test_data = pd.read_csv('fashion/fashion-mnist_test.csv')

# 클래스 매핑
class_map = {
    0: 'T-shirt_top', 
    1: 'Trouser', 
    2: 'Pullover', 
    3: 'Dress', 
    4: 'Coat', 
    5: 'Sandal', 
    6: 'Shirt', 
    7: 'Sneaker', 
    8: 'Bag', 
    9: 'Ankle_boot'
}

print(f"총 {len(test_data)}개의 테스트 이미지를 변환합니다...")

for idx, row in test_data.iterrows():
    # 레이블 추출
    label = int(row['label'])
    class_name = class_map[label]
    
    # 픽셀 데이터 추출 (label 컬럼 제외)
    pixels = row.drop('label').values.astype(np.uint8)
    
    # 28x28 이미지로 reshape
    image_array = pixels.reshape(28, 28)
    
    # PIL Image로 변환
    image = Image.fromarray(image_array, mode='L')  # 'L'은 그레이스케일
    
    # 클래스별 폴더 생성
    class_folder = os.path.join('test_images', class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    # 이미지 저장
    image_path = os.path.join(class_folder, f'{class_name}_{idx:05d}.png')
    image.save(image_path)
    
    # 진행상황 출력 (100개마다)
    if (idx + 1) % 100 == 0:
        print(f"진행률: {idx + 1}/{len(test_data)} ({((idx + 1)/len(test_data)*100):.1f}%)")

print("이미지 변환 완료!")
print(f"총 {len(test_data)}개의 이미지가 test_images 폴더에 저장되었습니다.")

# 클래스별 이미지 개수 출력
print("\n클래스별 이미지 개수:")
for class_id, class_name in class_map.items():
    class_folder = os.path.join('test_images', class_name)
    if os.path.exists(class_folder):
        image_count = len([f for f in os.listdir(class_folder) if f.endswith('.png')])
        print(f"{class_name}: {image_count}개")
