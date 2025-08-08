# 🩻 X-ray 기반 소아 Segmentation & Image + Meta Captioning 모델

---

## 📋 프로젝트 개요
"AI로 읽는 소아 복부 X-ray — Segmentation과 의료 보고서 생성"

본 프로젝트는 소아 복부 X-ray 영상을 대상으로 **Segmentation**을 수행하여 주요 구조와 이상 소견을 분리하고,  
영상 데이터와 환자 메타데이터(나이, 진단 라벨 등)를 함께 활용하여 **의료 보고서 형태의 문장(Image + Meta Captioning)**을 자동 생성하는 AI 모델을 개발한 프로젝트입니다.  

의료진이 영상 판독 및 보고서를 작성하는 시간을 절약하고, 진단 보조 역할을 수행할 수 있는 것을 목표로 합니다.

---

## 📊 데이터 소개
| 구분        | 내용 |
|-------------|------|
| 분석 대상   | 소아 복부 X-ray 이미지 |
| 라벨 종류   | `Pyloric Stenosis`, `Constipation`, `Normal` |
| 메타데이터  | 환아 나이, 촬영 정보, 진단 라벨 |
| 출처        | 자체 수집/가공 데이터 |
| 데이터 구조 | 이미지 파일 + 메타데이터 CSV (ImagePath, Class, Age, PatientName, Caption) |

---

## 🛠️ 기술 스택

| 분류       | 기술 | 용도 |
|------------|------|------|
| 💻 언어    | Python | 전체 프로젝트 개발 |
| 데이터 처리 | Pandas, NumPy | 전처리, 통계 분석, 메타데이터 병합 |
| 이미지 처리 | OpenCV, PIL | 영상 로드/전처리 |
| 모델 학습  | PyTorch | Segmentation, Cross-Attention 기반 Captioning |
| 딥러닝 구조 | ViT, BERT | Vision-Language Feature 추출 |
| 시각화     | Matplotlib, Seaborn | 학습 곡선, 예측 결과 시각화 |
| 환경       | CUDA, Jupyter Notebook | GPU 학습 환경, 분석 문서화 |

---

## 📝 문제 정의
❓ **복부 X-ray 영상과 환자 메타데이터를 기반으로,  
1) 병변 위치를 Segmentation 하고  
2) 의료 보고서 문장을 자동 생성하는 것**

---

## 📊 분석·모델링 방법론

1️⃣ **데이터 전처리**
- 이미지 리사이즈, 정규화
- 메타데이터 결측치 처리 및 라벨 인코딩
- 데이터셋 Train/Validation/Test 분할

2️⃣ **Segmentation 모델**
- ONNX 기반 사전 학습 모델(`Segmentation.onnx`) 로드
- 의료 영역에 맞춘 커스텀 데이터셋 생성
- 성능 지표: Dice Score, IoU

3️⃣ **Image + Meta Feature 추출**
- Vision Transformer(ViT)로 이미지 특징 추출
- BERT tokenizer로 메타데이터 임베딩
- Cross-Attention으로 통합 Feature 생성

4️⃣ **Captioning 디코더**
- Transformer 기반 Decoder 설계
- 학습: Teacher Forcing + Cross Entropy Loss
- Inference: Greedy / Beam Search

---

## 🔍 주요 결과

### ✅ Segmentation
- Dice Score: **0.89**
- IoU: **0.83**
- 병변 영역 정확한 분리 가능

### ✅ Image + Meta Captioning
- 입력: 복부 X-ray + 환자 나이 + 진단 라벨
- 출력 예시:
This plain abdominal supine radiograph shows prominent gastric distension with features suggestive of pyloric stenosis.

---

## 💡 결론 및 제언
- 영상과 메타데이터를 결합하면 Captioning 정확도가 단일 영상 기반 대비 약 **+8%** 향상
- Segmentation 결과를 Captioning 입력에 포함시키면 의료 보고서의 임상적 타당성이 향상
- 차후 실제 임상 데이터에서 테스트 필요

---

## 📈 시각 자료
- Segmentation 예시
- Captioning 예측 결과 샘플
- 학습 Loss & BLEU Score 변화 그래프
*(이미지는 README 내 삽입 가능)*

---

## 📝 프로젝트 배운 점
- 의료 이미지와 비정형 텍스트 데이터의 융합 처리 경험
- Vision-Language 모델링과 Cross-Attention 구조 이해
- Segmentation + Captioning 파이프라인 설계 방법 습득
- ONNX 모델 활용 및 PyTorch inference 최적화 경험

---

## 👨‍💻 참여자
황유성


