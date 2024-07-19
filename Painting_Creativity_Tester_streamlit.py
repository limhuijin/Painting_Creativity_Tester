import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from io import BytesIO

# 모델 로드
model_path = 'C:/Users/user/Desktop/coding/Painting_Creativity_Tester/model/model15.keras'
model = load_model(model_path)

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(uploaded_file, target_size=(224, 224)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 차원을 추가하여 (1, height, width, channels) 형식으로 만듦
    img_array /= 255.0  # 0-1 스케일링
    return img_array, img

# 그래프를 PNG 이미지로 변환하는 함수
def fig_to_img(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# 스트림릿 애플리케이션
st.title("Painting Creativity Tester")

# 파일 업로드
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 로드 및 전처리
    img_array, img = load_and_preprocess_image(uploaded_file)
    
    # 이미지 표시
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # 모델을 사용하여 예측 수행
    predictions = model.predict(img_array)
    total_score = np.sum(predictions[0])

    if total_score <= 8:
        score_category = 'Low'
    elif total_score <= 17:
        score_category = 'Medium'
    else:
        score_category = 'High'
    
    # 예측 결과 출력
    st.write("Predictions:", predictions)
    st.write("Total Score:", total_score)
    st.write("Category:", score_category)

    # 예측 결과 시각화
    labels = ['Delicate', 'Storytelling', 'Diversity of Objects', 'Utilize space', 'Expressive']
    fig, ax = plt.subplots()
    ax.bar(range(len(predictions[0])), predictions[0], tick_label=labels)
    ax.set_title(f"Predictions - Total: {total_score:.2f} ({score_category})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")

    # 레이블 위치 조정
    for i, label in enumerate(ax.xaxis.get_majorticklabels()):
        if i % 2 != 0:
            label.set_y(-0.04)  # 홀수번째 레이블을 위로 조정

    st.pyplot(fig)

    # 그래프를 다운로드할 수 있는 버튼 추가
    img_buf = fig_to_img(fig)
    st.download_button(
        label="Download Graph as PNG",
        data=img_buf,
        file_name="graph.png",
        mime="image/png"
    )
