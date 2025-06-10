#!/bin/bash

# 이미 실행 중인 streamlit 프로세스 확인 및 종료
pid=$(ps -ef | grep "[s]treamlit run app.py" | awk '{print $2}')
if [ ! -z "$pid" ]; then
  echo "기존에 실행 중인 Streamlit 프로세스($pid)를 종료합니다..."
  kill $pid
  sleep 2
fi

# 로그 디렉토리 생성
mkdir -p logs

# 현재 시간을 파일명에 포함
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/streamlit_$timestamp.log"

echo "마이크로그린 추천 시스템을 시작합니다..."
echo "로그는 $log_file 에 저장됩니다."

# 무한 루프로 앱 실행 (죽으면 자동 재시작)
while true; do
  echo "$(date): Streamlit 앱을 시작합니다..." >> $log_file
  streamlit run app.py >> $log_file 2>&1
  echo "$(date): Streamlit 앱이 종료되었습니다. 5초 후 재시작합니다..." >> $log_file
  sleep 5
done 