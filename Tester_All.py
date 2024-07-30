import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 모델 로드
model = load_model('C:/Users/user/Desktop/coding/Painting_Creativity_Tester/model/model15.keras')

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 차원을 추가하여 (1, height, width, channels) 형식으로 만듦
    img_array /= 255.0  # 0-1 스케일링
    return img_array

# 예측 및 점수 계산 함수
def predict_and_score(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    total_score = np.sum(predictions[0])
    
    return predictions[0], total_score

# 이미지 디렉토리 경로
image_dir = 'C:/Users/user/Desktop/coding/Painting_Creativity_Tester/images'
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

# 결과 저장을 위한 리스트
results = []

# 모든 이미지에 대해 예측 수행
for img_path in image_paths:
    predictions, total_score = predict_and_score(img_path)
    results.append([os.path.basename(img_path), *predictions, total_score])

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(results, columns=['FileName', 'Delicate', 'Storytelling', 'Diversity of Objects', 'Utilize space', 'Expressive', 'Total Score'])

# 엑셀 파일로 저장
output_path = 'C:/Users/user/Desktop/coding/images/all.xlsx'
df.to_excel(output_path, index=False)

print(f"Predictions saved to {output_path}")
