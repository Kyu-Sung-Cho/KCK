import streamlit as st
import pandas as pd
import os

def load_data():
    """
    로컬 파일 또는 사용자 업로드를 통해 데이터를 로드합니다.
    로컬 파일이 없는 경우 사용자에게 파일 업로드를 요청합니다.
    """
    # 판매 데이터와 반품 데이터 파일 경로 설정
    sales_file = "merged_2023_2024_2025.xlsx"
    refund_file = "merged_returns_2024_2025.xlsx"
    
    # 판매 데이터 로드 시도
    sales_data = None
    refund_data = None
    
    # 사이드바에 파일 업로드 정보 표시 섹션 추가
    with st.sidebar.expander("📂 데이터 파일 정보", expanded=True):
        # 로컬 파일 로드 시도
        try:
            if os.path.exists(sales_file):
                sales_data = pd.read_excel(sales_file)
                st.success(f"로컬 판매 데이터 로드 완료: {len(sales_data)}개 레코드")
            else:
                st.warning(f"로컬 파일 '{sales_file}'을 찾을 수 없습니다.")
                
            if os.path.exists(refund_file):
                refund_data = pd.read_excel(refund_file)
                st.success(f"로컬 반품 데이터 로드 완료: {len(refund_data)}개 레코드")
            else:
                st.warning(f"로컬 파일 '{refund_file}'을 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"로컬 파일 로드 중 오류 발생: {e}")
        
        # 로컬 파일이 없으면 업로드 옵션 제공
        if sales_data is None:
            st.info("판매 데이터 파일을 업로드해주세요.")
            uploaded_sales = st.file_uploader("판매 데이터 파일 업로드", type=["xlsx", "xls", "csv"])
            if uploaded_sales is not None:
                try:
                    if uploaded_sales.name.endswith('.csv'):
                        sales_data = pd.read_csv(uploaded_sales)
                    else:
                        sales_data = pd.read_excel(uploaded_sales)
                    st.success(f"판매 데이터 업로드 완료: {len(sales_data)}개 레코드")
                except Exception as e:
                    st.error(f"판매 데이터 업로드 중 오류 발생: {e}")
        
        if refund_data is None:
            st.info("반품 데이터 파일을 업로드해주세요.")
            uploaded_refund = st.file_uploader("반품 데이터 파일 업로드", type=["xlsx", "xls", "csv"])
            if uploaded_refund is not None:
                try:
                    if uploaded_refund.name.endswith('.csv'):
                        refund_data = pd.read_csv(uploaded_refund)
                    else:
                        refund_data = pd.read_excel(uploaded_refund)
                    st.success(f"반품 데이터 업로드 완료: {len(refund_data)}개 레코드")
                except Exception as e:
                    st.error(f"반품 데이터 업로드 중 오류 발생: {e}")
    
    return sales_data, refund_data 