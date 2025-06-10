#!/bin/bash

# 실행 중인 스크립트와 streamlit 프로세스 모두 종료
pid_script=$(ps -ef | grep "[s]tart_streamlit.sh" | awk '{print $2}')
pid_streamlit=$(ps -ef | grep "[s]treamlit run app.py" | awk '{print $2}')

# 스크립트 프로세스 종료
if [ ! -z "$pid_script" ]; then
  echo "start_streamlit.sh 스크립트($pid_script)를 종료합니다..."
  kill $pid_script
  sleep 1
fi

# streamlit 프로세스 종료
if [ ! -z "$pid_streamlit" ]; then
  echo "Streamlit 프로세스($pid_streamlit)를 종료합니다..."
  kill $pid_streamlit
  sleep 1
fi

echo "마이크로그린 추천 시스템이 중지되었습니다." 