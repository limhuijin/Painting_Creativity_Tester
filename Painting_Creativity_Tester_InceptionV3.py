import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리
data1 = pd.read_csv('C:/Users/user/Desktop/agics/csv/Painting_Creativity_Tester_05.csv')
data2 = pd.read_csv('C:/Users/user/Desktop/agics/csv/Painting_Creativity_Tester_06.csv')
data = pd.concat([data1, data2], ignore_index=True)

# 이미지 파일 경로 설정
image_dir = 'C:/Users/user/Desktop/agics/images/'
data['image_path'] = [os.path.join(image_dir, f) + '.png' for f in data['FileName']]

# 훈련 데이터와 검증 데이터 분할
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 데이터 증강을 위한 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 생성기 준비
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col=['Delicate', 'Storytelling', 'Diversity of Objects', 'Utilize space', 'Expressive'],
    target_size=(299, 299),  # InceptionV3의 기본 입력 크기에 맞춤
    batch_size=32,
    class_mode='raw'
)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col=['Delicate', 'Storytelling', 'Diversity of Objects', 'Utilize space', 'Expressive'],
    target_size=(299, 299),
    batch_size=32,
    class_mode='raw'
)

# 모델 구성
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # 학습 중 가중치를 고정

x = GlobalAveragePooling2D()(base_model.output)  # 평균 풀링 레이어
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='linear')(x)  # 5개 창의성 기준 점수 예측

model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=100,
    verbose=1
)

# 모델 평가
model.evaluate(test_generator)

# 모델 저장
model.save('C:/Users/user/Desktop/agics/model/model.keras')
