import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 모델 로드
model = load_model('C:/Users/user/Desktop/agics/model/model.keras')

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 차원을 추가하여 (1, height, width, channels) 형식으로 만듦
    img_array /= 255.0  # 0-1 스케일링
    return img_array, img

# 예측할 이미지 경로
img_path = 'C:/Users/user/Desktop/agics/test/test07.png'

# 이미지 로드 및 전처리
img_array, original_img = load_and_preprocess_image(img_path)

# 모델을 사용하여 예측 수행
predictions = model.predict(img_array)
print("Predictions:", predictions)

# 원본 이미지와 예측 결과 시각화
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(len(predictions[0])), predictions[0])
plt.title("Predictions")
plt.xlabel("Class")
plt.ylabel("Probability")

plt.show()
