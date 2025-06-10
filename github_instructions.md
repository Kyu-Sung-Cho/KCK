# GitHub에 프로젝트 올리는 방법

## 1. GitHub 계정 및 저장소 생성

1. [GitHub](https://github.com/) 계정이 없다면 가입하세요.
2. GitHub에 로그인한 후 오른쪽 상단의 "+" 아이콘을 클릭하고 "New repository"를 선택합니다.
3. 저장소 이름(예: "microgreens-recommendation-system")을 입력하고 필요에 따라 설명을 추가합니다.
4. "Public" 또는 "Private" 중에서 선택합니다.
5. "Initialize this repository with a README"는 체크하지 마세요(이미 README.md 파일을 만들었습니다).
6. "Create repository"를 클릭하여 저장소를 생성합니다.

## 2. 로컬 Git 저장소 설정 및 코드 업로드

터미널(맥/리눅스) 또는 명령 프롬프트(윈도우)를 열고 다음 명령어들을 차례대로 실행하세요:

```bash
# 1. 현재 디렉토리로 이동 (이미 프로젝트 폴더에 있다면 생략)
cd /Users/kyusungcho/Desktop/kck/reco

# 2. Git 저장소 초기화
git init

# 3. 파일들을 스테이징 영역에 추가
git add app.py README.md requirements.txt .gitignore sort_data.py

# 4. 데이터 파일 추가 (대용량 파일이므로 GitHub LFS를 사용하는 것이 좋습니다)
git add merged_2023_2024_2025.xlsx merged_returns_2024_2025.xlsx

# 5. 변경사항 커밋
git commit -m "Initial commit: Microgreens recommendation system"

# 6. GitHub 원격 저장소 추가 (YOUR_USERNAME과 YOUR_REPO_NAME을 실제 값으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 7. 코드를 GitHub에 푸시
git push -u origin master    # 또는 git push -u origin main (GitHub 기본 브랜치 설정에 따라 다름)
```

> 참고: 대용량 파일(예: Excel 파일)을 GitHub에 올릴 때 문제가 발생한다면 [Git LFS](https://git-lfs.github.com/)를 사용하거나, 데이터 파일은 제외하고 코드만 업로드하는 것을 고려하세요.

## 3. Streamlit Cloud에 배포하기

1. [Streamlit Cloud](https://streamlit.io/cloud)에 접속하여 로그인합니다(GitHub 계정으로 로그인 가능).
2. "New app" 버튼을 클릭합니다.
3. GitHub 저장소, 브랜치, 메인 파일 경로(여기서는 "app.py")를 입력합니다.
4. "Deploy"를 클릭하여 배포합니다.
5. 배포가 완료되면 공개 URL이 생성됩니다. 이 URL을 통해 누구나 웹에서 앱에 접근할 수 있습니다.

## 대용량 파일 처리 방법 (선택사항)

엑셀 파일과 같은 대용량 데이터 파일은 GitHub의 일반적인 용량 제한(일반적으로 100MB)을 초과할 수 있습니다. 이 경우 다음 방법을 고려하세요:

1. **Git LFS 사용**: 대용량 파일 관리를 위한 Git 확장 기능
2. **데이터 파일 제외**: .gitignore 파일에 데이터 파일을 추가하여 GitHub에 올리지 않음
3. **Google Drive 또는 Dropbox** 같은 클라우드 스토리지에 데이터 파일을 저장하고, 앱에서는 해당 파일을 다운로드하여 사용

## Streamlit Secrets 관리 (필요한 경우)

앱에 API 키나 비밀번호 등이 필요한 경우:

1. Streamlit Cloud 대시보드에서 앱 설정으로 이동
2. "Secrets" 섹션에 필요한 정보를 추가
3. 앱에서는 `st.secrets`를 통해 이 정보에 접근할 수 있음 