# 🌱 마이크로그린 맞춤형 추천 시스템

마이크로그린 판매 데이터를 분석하여 고객별, 계절별 맞춤형 상품을 추천하는 시스템입니다.

## 주요 기능

- 👤 **고객 맞춤 추천**: 고객의 구매 이력을 분석하여 개인화된 상품 추천
- 🌞 **계절 추천**: 현재 계절에 인기 있는 마이크로그린 상품 추천
- 📦 **번들 추천**: 함께 구매하면 좋은 상품 조합 추천
- 🆕 **신규 고객 추천**: 새로운 고객을 위한 일반적인 인기 상품 추천
- 📊 **상품 분석**: 개별 상품에 대한 판매 추세 및 특성 분석
- 🏢 **업체 분석**: 고객사별 구매 패턴 및 선호도 분석
- 📈 **고객 세분화 (RFM)**: RFM 분석을 통한 고객 세분화 및 맞춤 전략 수립

## 설치 및 실행 방법

### 1. 로컬 환경에서 직접 실행

1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

2. 앱 실행
```bash
streamlit run app.py
```

### 2. 백그라운드에서 실행 (서버 유지)

```bash
./start_streamlit.sh
```

중지하려면:
```bash
./stop_streamlit.sh
```

## Streamlit Cloud에 배포하기

1. GitHub에 코드 푸시하기
```bash
# 코드를 GitHub에 푸시 (처음 설정 시)
git add app.py app_cloud.py streamlit_cloud_data.py requirements.txt README.md .gitignore
git commit -m "Initial commit"
git push -u origin main

# 이후 업데이트 시
git add .
git commit -m "Update app"
git push
```

2. Streamlit Cloud에서 배포
   - https://streamlit.io/cloud 에 접속하여 로그인
   - "New app" 클릭
   - GitHub 저장소 URL(https://github.com/Kyu-Sung-Cho/KCK)을 입력
   - 브랜치: main
   - 메인 파일 경로: app_cloud.py
   - "Deploy" 클릭

## 데이터 파일

실행을 위해 다음 데이터 파일이 필요합니다:
- `merged_2023_2024_2025.xlsx`: 판매 데이터
- `merged_returns_2024_2025.xlsx`: 반품 데이터

데이터 파일이 없는 경우 앱 내에서 파일 업로더를 통해 업로드할 수 있습니다.

## 주요 개발자

- KCK 마이크로그린팀

## 라이센스

이 프로젝트는 사내용으로 개발되었으며, 무단 복제 및 배포를 금지합니다.
 