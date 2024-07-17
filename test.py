import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 모델 로드
model = load_model('C:/Users/user/Desktop/coding/Painting_Creativity_Tester/model/model13.keras')

# 이미지 로드 및 전처리 함수
#target_size=(224, 224)는 VGG16의 기본 입력 크기에 맞춤
#target_size=(299, 299)는 InceptionV3의 기본 입력 크기에 맞춤
#target_size=(240, 240)는 EfficientNetB0의 기본 입력 크기에 맞춤
def load_and_preprocess_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 차원을 추가하여 (1, height, width, channels) 형식으로 만듦
    img_array /= 255.0  # 0-1 스케일링
    return img_array, img

# 예측할 이미지 경로
img_path = 'C:/Users/user/Desktop/coding/Painting_Creativity_Tester/test/test07.png'

# 이미지 로드 및 전처리
img_array, original_img = load_and_preprocess_image(img_path)

# 모델을 사용하여 예측 수행
predictions = model.predict(img_array)
total_score = np.sum(predictions[0])

if total_score <= 8:
    score_category = 'Low'
elif total_score <= 17:
    score_category = 'Medium'
else:
    score_category = 'High'

print("Predictions:", predictions)
print("Total Score:", total_score, "Category:", score_category)

# 원본 이미지와 예측 결과 시각화
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(len(predictions[0])), predictions[0], tick_label=['Delicate', 'Storytelling', 'Diversity of Objects', 'Utilize space', 'Expressive'])
plt.title(f"Predictions - Total: {total_score:.2f} ({score_category})")
plt.xlabel("Class")
plt.ylabel("Probability")

ax = plt.gca()

# 레이블 위치 조정
labels = ['Delicate', 'Storytelling', 'Diversity of Objects', 'Utilize space', 'Expressive']
for i, label in enumerate(ax.xaxis.get_majorticklabels()):
    if i % 2 != 0:
        label.set_y(-0.04)  # 홀수번째 레이블을 위로 조정

plt.tight_layout()

plt.show()
