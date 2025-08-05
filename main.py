import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from PIL import Image
import streamlit as st
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_model(model_path='Best_model.pth', num_classes=5):
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    model = model.to('cpu')

    # strict=False로 파라미터 일부 무시하며 로드
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # [1, C, H, W]

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]
    return pred

st.set_page_config(page_title="Segmentation Viewer", layout="wide")

# 타이틀
st.title("🧠 Segmentation 모델 웹 데모")
st.write("이미지를 업로드하면 Segmentation 결과를 시각화합니다.")

model = load_model()
# 모델 로딩
uploaded_file = st.file_uploader("이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file).convert("RGB")
    
    # 추론 버튼
    if st.button("Segmentation 실행"):
        with st.spinner("모델 추론 중..."):
            input_tensor = preprocess(image)
            mask = predict(model, input_tensor)

            # 시각화
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title(f"Image")
            axes[0].axis('off')

            axes[1].imshow(image)
            axes[1].imshow(mask, cmap='jet', alpha=0.5)
            axes[1].set_title(f"Image with Predicted Mask")
            axes[1].axis('off')

            plt.tight_layout()
            st.pyplot(fig)
