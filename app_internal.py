import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="마이크로그린 관리자 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 폰트 설정
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2c5aa0;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

class MicrogreenAnalysisSystem:
    def __init__(self, sales_data, refund_data=None):
        self.original_sales_data = sales_data.copy()  # 원본 데이터 보존
        self.original_refund_data = refund_data.copy() if refund_data is not None else None
        self.sales_data = sales_data
        self.refund_data = refund_data
        self.customer_product_matrix = None
        
        # 데이터 전처리
        self.preprocess_data()
    
    def filter_data_by_date_range(self, start_date, end_date):
        """날짜 범위로 데이터 필터링"""
        # 원본 데이터에서 시작
        filtered_sales_data = self.original_sales_data.copy()
        filtered_refund_data = self.original_refund_data.copy() if self.original_refund_data is not None else None
        
        if '날짜' in filtered_sales_data.columns:
            try:
                # 날짜 컬럼을 datetime으로 변환
                filtered_sales_data['날짜'] = pd.to_datetime(filtered_sales_data['날짜'], errors='coerce')
                
                # 날짜 범위로 필터링
                mask = (filtered_sales_data['날짜'] >= start_date) & (filtered_sales_data['날짜'] <= end_date)
                filtered_sales_data = filtered_sales_data[mask]
                
            except Exception as e:
                st.warning(f"날짜 필터링 중 오류 발생: {str(e)}")
        
        # 반품 데이터도 필터링 (날짜 컬럼이 있는 경우)
        if filtered_refund_data is not None and '날짜' in filtered_refund_data.columns:
            try:
                # 반품 데이터도 같은 방식으로 날짜 변환
                refund_dates = filtered_refund_data['날짜'].astype(str).apply(
                    lambda x: f"20{x}" if len(str(x).split('.')[0]) == 2 else x
                )
                filtered_refund_data['날짜'] = pd.to_datetime(refund_dates, errors='coerce')
                
                # 2023년 이후 데이터만 유지
                cutoff_date = pd.to_datetime('2023-01-01')
                filtered_refund_data = filtered_refund_data[
                    (filtered_refund_data['날짜'].isna()) | 
                    (filtered_refund_data['날짜'] >= cutoff_date)
                ]
                
                # 날짜 범위 필터링
                mask = (filtered_refund_data['날짜'] >= start_date) & (filtered_refund_data['날짜'] <= end_date)
                filtered_refund_data = filtered_refund_data[mask]
            except Exception as e:
                st.warning(f"반품 데이터 날짜 필터링 중 오류 발생: {str(e)}")
        
        # 필터링된 데이터로 새로운 인스턴스 생성
        return MicrogreenAnalysisSystem(filtered_sales_data, filtered_refund_data)
    
    def get_date_range(self):
        """데이터의 날짜 범위 반환 (2023년 이후만)"""
        if '날짜' in self.original_sales_data.columns:
            try:
                # 원본 데이터의 날짜를 같은 방식으로 변환
                original_dates = self.original_sales_data['날짜'].astype(str).apply(
                    lambda x: f"20{x}" if len(str(x).split('.')[0]) == 2 else x
                )
                dates = pd.to_datetime(original_dates, errors='coerce')
                valid_dates = dates.dropna()
                
                # 2023년 이후 데이터만 필터링
                if not valid_dates.empty:
                    cutoff_date = pd.to_datetime('2023-01-01')
                    filtered_dates = valid_dates[valid_dates >= cutoff_date]
                    if not filtered_dates.empty:
                        return filtered_dates.min(), filtered_dates.max()
            except Exception as e:
                st.warning(f"날짜 범위 처리 중 오류: {str(e)}")
        return None, None
    
    def get_available_dates(self):
        """데이터에 실제로 존재하는 날짜들 반환 (2023년 이후만)"""
        if '날짜' in self.original_sales_data.columns:
            try:
                # 원본 데이터의 날짜를 같은 방식으로 변환
                original_dates = self.original_sales_data['날짜'].astype(str).apply(
                    lambda x: f"20{x}" if len(str(x).split('.')[0]) == 2 else x
                )
                dates = pd.to_datetime(original_dates, errors='coerce')
                valid_dates = dates.dropna()
                
                # 2023년 이후 데이터만 필터링
                if not valid_dates.empty:
                    cutoff_date = pd.to_datetime('2023-01-01')
                    filtered_dates = valid_dates[valid_dates >= cutoff_date]
                    if not filtered_dates.empty:
                        # 유니크한 날짜들을 정렬하여 반환
                        unique_dates = sorted(filtered_dates.dt.date.unique())
                        return unique_dates
            except Exception as e:
                st.warning(f"날짜 처리 중 오류: {str(e)}")
        return []

    def preprocess_data(self):
        """데이터 전처리"""
        # 데이터 타입 확인 및 변환
        try:
            # 데이터프레임 완전 복사본 생성 (SettingWithCopyWarning 방지)
            self.sales_data = self.sales_data.copy(deep=True)
            
            # 수량 컬럼을 숫자로 변환
            if '수량' in self.sales_data.columns:
                self.sales_data['수량'] = pd.to_numeric(self.sales_data['수량'], errors='coerce')
                self.sales_data = self.sales_data.dropna(subset=['수량'])
            
            # 금액 컬럼을 숫자로 변환 (있는 경우)
            if '금액' in self.sales_data.columns:
                self.sales_data['금액'] = pd.to_numeric(self.sales_data['금액'], errors='coerce')
            
            # 문자열 컬럼의 NaN 값을 빈 문자열로 변환
            string_columns = ['상품', '고객명']
            for col in string_columns:
                if col in self.sales_data.columns:
                    self.sales_data[col] = self.sales_data[col].fillna('').astype(str)
            
            # 상품명 정규화 (접미사 제거)
            if '상품' in self.sales_data.columns:
                self.sales_data['상품'] = self.sales_data['상품'].apply(self.normalize_product_name)
            
            # 배송료 관련 키워드 정의
            delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
            delivery_pattern = '|'.join(delivery_keywords)
            
            # 유효한 판매 데이터 필터링 (배송료, 세트상품 등 제외)
            self.sales_data = self.sales_data[
                ~self.sales_data['상품'].str.contains('세트상품|증정품', na=False) &
                ~self.sales_data['상품'].str.contains(delivery_pattern, case=False, na=False)
            ]
            
            # 재고조정, 창고 등 제외
            self.sales_data = self.sales_data[
                ~self.sales_data['고객명'].str.contains('재고조정|문정창고|창고', na=False, regex=True)
            ]
            
            # 빈 값들 제거
            self.sales_data = self.sales_data[
                (self.sales_data['상품'] != '') & 
                (self.sales_data['고객명'] != '') &
                (self.sales_data['수량'] > 0)
            ]
            
            # 상품명 정규화 후 데이터 재집계 (같은 상품으로 통합)
            if not self.sales_data.empty:
                # 정규화된 상품명으로 데이터 재집계
                # 날짜별로 별도 집계하여 날짜 정보 손실 방지
                aggregation_dict = {'수량': 'sum'}
                if '금액' in self.sales_data.columns:
                    aggregation_dict['금액'] = 'sum'
                
                # 비고 정보 유지를 위해 첫 번째 비고 사용
                if '비고' in self.sales_data.columns:
                    aggregation_dict['비고'] = 'first'
                
                # 고객명, 상품명, 날짜로 그룹화하여 재집계 (날짜별로 분리)
                group_columns = ['고객명', '상품']
                if '날짜' in self.sales_data.columns:
                    group_columns.append('날짜')
                
                self.sales_data = self.sales_data.groupby(group_columns).agg(aggregation_dict).reset_index()
            
            # 고객-상품 매트릭스 생성
            if not self.sales_data.empty:
                self.customer_product_matrix = self.sales_data.groupby(['고객명', '상품'])['수량'].sum().unstack(fill_value=0)
            else:
                # 빈 매트릭스 생성
                self.customer_product_matrix = pd.DataFrame()
            
            # 날짜 변환
            if '날짜' in self.sales_data.columns:
                try:
                    # 날짜 형식이 'YY.MM.DD'인 경우 처리
                    self.sales_data['날짜'] = pd.to_datetime(self.sales_data['날짜'].astype(str).apply(
                        lambda x: f"20{x}" if len(str(x).split('.')[0]) == 2 else x
                    ), errors='coerce')
                    
                    # 2023년 이전 데이터 제외 (2022년 데이터 무시)
                    if not self.sales_data.empty:
                        valid_dates_mask = self.sales_data['날짜'].notna()
                        if valid_dates_mask.any():
                            # 2023년 1월 1일 이후 데이터만 유지
                            cutoff_date = pd.to_datetime('2023-01-01')
                            self.sales_data = self.sales_data[
                                (self.sales_data['날짜'].isna()) | 
                                (self.sales_data['날짜'] >= cutoff_date)
                            ]
                            
                            # 월 정보 추가 (유효한 날짜만)
                            valid_dates_mask = self.sales_data['날짜'].notna()
                            if valid_dates_mask.any():
                                self.sales_data.loc[valid_dates_mask, 'month'] = self.sales_data.loc[valid_dates_mask, '날짜'].dt.month
                except Exception as e:
                    st.warning(f"날짜 변환 중 오류 발생: {str(e)}")
                    # 오류 발생 시 기본 처리만 수행
            
        except Exception as e:
            st.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            # 기본적인 전처리만 수행
            self.sales_data = self.sales_data.copy(deep=True)
            if '수량' in self.sales_data.columns:
                self.sales_data['수량'] = pd.to_numeric(self.sales_data['수량'], errors='coerce')
                self.sales_data = self.sales_data.dropna(subset=['수량'])
            
            if not self.sales_data.empty:
                self.customer_product_matrix = self.sales_data.groupby(['고객명', '상품'])['수량'].sum().unstack(fill_value=0)
            else:
                self.customer_product_matrix = pd.DataFrame()

    def normalize_product_name(self, product_name):
        """상품명 정규화 - 접미사 제거하여 같은 상품으로 통합"""
        if pd.isna(product_name) or product_name == '':
            return product_name
        
        # 문자열로 변환
        product_name = str(product_name).strip()
        
        # 제거할 접미사 패턴들
        suffixes_to_remove = [
            '_상온', '_냉장', '_냉동', '_실온',
            '(상온)', '(냉장)', '(냉동)', '(실온)',
            ' 상온', ' 냉장', ' 냉동', ' 실온',
            '-상온', '-냉장', '-냉동', '-실온'
        ]
        
        # 각 접미사 패턴 제거
        normalized_name = product_name
        for suffix in suffixes_to_remove:
            if normalized_name.endswith(suffix):
                normalized_name = normalized_name[:-len(suffix)].strip()
                break  # 첫 번째 매칭되는 접미사만 제거
        
        return normalized_name

    def analyze_product_details(self, product_name):
        """상품 상세 분석"""
        # 상품 판매 데이터 필터링
        product_sales = self.sales_data[self.sales_data['상품'] == product_name]
        
        if product_sales.empty:
            return {
                '상태': '실패',
                '메시지': f"상품 '{product_name}'의 판매 데이터를 찾을 수 없습니다."
            }
        
        # 기본 통계
        total_quantity = int(product_sales['수량'].sum())
        total_customers = product_sales['고객명'].nunique()
        
        # 금액 정보 (있는 경우)
        has_amount = '금액' in product_sales.columns
        total_amount = int(product_sales['금액'].sum()) if has_amount else 0
        avg_price = total_amount / total_quantity if total_quantity > 0 and has_amount else 0
        
        # 월별 판매 패턴
        monthly_sales = {}
        yearmonth_sales = {}
        
        if 'month' in product_sales.columns:
            monthly_data = product_sales.groupby('month').agg({
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            
            for _, row in monthly_data.iterrows():
                month = int(row['month'])
                monthly_sales[month] = {
                    '수량': int(row['수량']),
                    '고객수': int(row['고객명'])
                }
        
        # 연도-월별 판매 패턴 추가
        if '날짜' in product_sales.columns and not product_sales.empty:
            try:
                # 날짜 컬럼을 datetime으로 변환
                product_sales_copy = product_sales.copy(deep=True)
                product_sales_copy['날짜'] = pd.to_datetime(product_sales_copy['날짜'], errors='coerce')
                
                # 유효한 날짜만 필터링
                valid_data = product_sales_copy[product_sales_copy['날짜'].notna()].copy()
                
                if not valid_data.empty:
                    # 연도, 월, 연-월 정보 추출
                    valid_data.loc[:, 'year'] = valid_data['날짜'].dt.year
                    valid_data.loc[:, 'month'] = valid_data['날짜'].dt.month
                    valid_data.loc[:, 'yearmonth'] = valid_data['날짜'].dt.strftime('%Y-%m')
                    
                    # 연-월 기준 집계
                    agg_dict = {'수량': 'sum', '고객명': 'nunique'}
                    if has_amount:
                        agg_dict['금액'] = 'sum'
                    
                    # 연-월 기준으로 집계
                    yearmonth_data = valid_data.groupby(['year', 'month', 'yearmonth']).agg(agg_dict).reset_index()
                    
                    for _, row in yearmonth_data.iterrows():
                        year = int(row['year'])
                        month = int(row['month'])
                        yearmonth = row['yearmonth']
                        
                        month_data = {
                            '수량': int(row['수량']),
                            '고객수': int(row['고객명'])
                        }
                        
                        if has_amount:
                            month_data['금액'] = int(row['금액'])
                        else:
                            month_data['금액'] = 0
                        
                        # 연-월별 데이터 저장
                        yearmonth_sales[yearmonth] = month_data
                        
                        # 기존 월별 판매 패턴도 업데이트 (누적)
                        if month not in monthly_sales:
                            monthly_sales[month] = {'수량': 0, '고객수': 0}
                        
                        monthly_sales[month]['수량'] += month_data['수량']
                        monthly_sales[month]['고객수'] = max(monthly_sales[month]['고객수'], month_data['고객수'])
            except Exception as e:
                # 오류 발생 시 기존 로직 유지
                pass
        
        # 고객별 구매 패턴
        customer_purchases = product_sales.groupby('고객명')['수량'].sum().sort_values(ascending=False)
        top_customers = customer_purchases.head(10).to_dict()
        
        # 계절별 분석
        seasonal_sales = {'봄': 0, '여름': 0, '가을': 0, '겨울': 0}
        if 'month' in product_sales.columns:
            for month, group in product_sales.groupby('month'):
                month = int(month)
                quantity = group['수량'].sum()
                
                if month in [3, 4, 5]:
                    seasonal_sales['봄'] += quantity
                elif month in [6, 7, 8]:
                    seasonal_sales['여름'] += quantity
                elif month in [9, 10, 11]:
                    seasonal_sales['가을'] += quantity
                else:
                    seasonal_sales['겨울'] += quantity
        
        return {
            '상태': '성공',
            '상품명': product_name,
            '총_판매량': total_quantity,
            '총_판매금액': total_amount,
            '평균_단가': round(avg_price, 2),
            '구매_고객수': total_customers,
            '월별_판매': monthly_sales,
            '주요_고객': top_customers,
            '계절별_판매': seasonal_sales,
            '연월별_판매': yearmonth_sales
        }

    def analyze_product_details_exclude_fourseasons(self, product_name):
        """상품 상세 분석 (포시즌스 호텔 제외)"""
        # 포시즌스 호텔 관련 고객 제외
        filtered_sales_data = self.sales_data[
            ~self.sales_data['고객명'].str.contains('포시즌스', na=False, case=False)
        ]
        
        # 상품 판매 데이터 필터링
        product_sales = filtered_sales_data[filtered_sales_data['상품'] == product_name]
        
        if product_sales.empty:
            return {
                '상태': '실패',
                '메시지': f"상품 '{product_name}'의 판매 데이터를 찾을 수 없습니다. (포시즌스 호텔 제외)"
            }
        
        # 기본 통계
        total_quantity = int(product_sales['수량'].sum())
        total_customers = product_sales['고객명'].nunique()
        
        # 금액 정보 (있는 경우)
        has_amount = '금액' in product_sales.columns
        total_amount = int(product_sales['금액'].sum()) if has_amount else 0
        avg_price = total_amount / total_quantity if total_quantity > 0 and has_amount else 0
        
        # 월별 판매 패턴
        monthly_sales = {}
        yearmonth_sales = {}
        
        if 'month' in product_sales.columns:
            monthly_data = product_sales.groupby('month').agg({
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            
            for _, row in monthly_data.iterrows():
                month = int(row['month'])
                monthly_sales[month] = {
                    '수량': int(row['수량']),
                    '고객수': int(row['고객명'])
                }
        
        # 연도-월별 판매 패턴 추가
        if '날짜' in product_sales.columns and not product_sales.empty:
            try:
                # 날짜 컬럼을 datetime으로 변환
                product_sales_copy = product_sales.copy(deep=True)
                product_sales_copy['날짜'] = pd.to_datetime(product_sales_copy['날짜'], errors='coerce')
                
                # 유효한 날짜만 필터링
                valid_data = product_sales_copy[product_sales_copy['날짜'].notna()].copy()
                
                if not valid_data.empty:
                    # 연도, 월, 연-월 정보 추출
                    valid_data.loc[:, 'year'] = valid_data['날짜'].dt.year
                    valid_data.loc[:, 'month'] = valid_data['날짜'].dt.month
                    valid_data.loc[:, 'yearmonth'] = valid_data['날짜'].dt.strftime('%Y-%m')
                    
                    # 연-월 기준 집계
                    agg_dict = {'수량': 'sum', '고객명': 'nunique'}
                    if has_amount:
                        agg_dict['금액'] = 'sum'
                    
                    # 연-월 기준으로 집계
                    yearmonth_data = valid_data.groupby(['year', 'month', 'yearmonth']).agg(agg_dict).reset_index()
                    
                    for _, row in yearmonth_data.iterrows():
                        year = int(row['year'])
                        month = int(row['month'])
                        yearmonth = row['yearmonth']
                        
                        month_data = {
                            '수량': int(row['수량']),
                            '고객수': int(row['고객명'])
                        }
                        
                        if has_amount:
                            month_data['금액'] = int(row['금액'])
                        else:
                            month_data['금액'] = 0
                        
                        # 연-월별 데이터 저장
                        yearmonth_sales[yearmonth] = month_data
                        
                        # 기존 월별 판매 패턴도 업데이트 (누적)
                        if month not in monthly_sales:
                            monthly_sales[month] = {'수량': 0, '고객수': 0}
                        
                        monthly_sales[month]['수량'] += month_data['수량']
                        monthly_sales[month]['고객수'] = max(monthly_sales[month]['고객수'], month_data['고객수'])
            except Exception as e:
                # 오류 발생 시 기존 로직 유지
                pass
        
        # 고객별 구매 패턴
        customer_purchases = product_sales.groupby('고객명')['수량'].sum().sort_values(ascending=False)
        top_customers = customer_purchases.head(10).to_dict()
        
        # 계절별 분석
        seasonal_sales = {'봄': 0, '여름': 0, '가을': 0, '겨울': 0}
        if 'month' in product_sales.columns:
            for month, group in product_sales.groupby('month'):
                month = int(month)
                quantity = group['수량'].sum()
                
                if month in [3, 4, 5]:
                    seasonal_sales['봄'] += quantity
                elif month in [6, 7, 8]:
                    seasonal_sales['여름'] += quantity
                elif month in [9, 10, 11]:
                    seasonal_sales['가을'] += quantity
                else:
                    seasonal_sales['겨울'] += quantity
        
        return {
            '상태': '성공',
            '상품명': product_name,
            '총_판매량': total_quantity,
            '총_판매금액': total_amount,
            '평균_단가': round(avg_price, 2),
            '구매_고객수': total_customers,
            '월별_판매': monthly_sales,
            '주요_고객': top_customers,
            '계절별_판매': seasonal_sales,
            '연월별_판매': yearmonth_sales,
            '제외_조건': '포시즌스 호텔 제외'
        }

    def analyze_customer_details(self, customer_name):
        """특정 업체(고객)의 상세 정보 분석"""
        if customer_name not in self.customer_product_matrix.index:
            return {
                '상태': '실패',
                '메시지': f"고객 '{customer_name}'을(를) 찾을 수 없습니다."
            }
        
        # 재고조정, 창고 제외
        if '재고조정' in customer_name or '문정창고' in customer_name or '창고' in customer_name:
            return {
                '상태': '실패',
                '메시지': f"'{customer_name}'은(는) 분석에서 제외됩니다."
            }
        
        # 업체 코드 분석 (앞 3자리)
        customer_code = ""
        customer_category = "일반"
        try:
            code_match = re.match(r'^(\d{3})', customer_name)
            if code_match:
                customer_code = code_match.group(1)
                if customer_code in ['001', '005']:
                    customer_category = "호텔"
        except:
            pass
            
        # 총 구매량 및 금액
        customer_purchases = self.sales_data[self.sales_data['고객명'] == customer_name].copy(deep=True)
        total_quantity = int(customer_purchases['수량'].sum())
        
        # 금액 컬럼이 있는지 확인하고 처리
        has_amount = '금액' in customer_purchases.columns
        total_amount = int(customer_purchases['금액'].sum()) if has_amount else 0
        
        # 연도-월별 구매 패턴 (연도 정보 추가)
        monthly_purchases = {}
        yearmonth_purchases = {}
        
        if '날짜' in customer_purchases.columns and not customer_purchases.empty:
            try:
                # 날짜 컬럼을 datetime으로 변환
                customer_purchases_copy = customer_purchases.copy(deep=True)
                customer_purchases_copy['날짜'] = pd.to_datetime(customer_purchases_copy['날짜'], errors='coerce')
                
                # 유효한 날짜만 필터링
                valid_data = customer_purchases_copy[customer_purchases_copy['날짜'].notna()].copy()
                
                if not valid_data.empty:
                    # 연도, 월, 연-월 정보 추출
                    valid_data.loc[:, 'year'] = valid_data['날짜'].dt.year
                    valid_data.loc[:, 'month'] = valid_data['날짜'].dt.month
                    valid_data.loc[:, 'yearmonth'] = valid_data['날짜'].dt.strftime('%Y-%m')
                    
                    # customer_purchases를 유효한 데이터로 교체
                    customer_purchases = valid_data
                    
                    # 연-월 기준 집계
                    agg_dict = {'수량': 'sum'}
                    if has_amount:
                        agg_dict['금액'] = 'sum'
                    
                    # 연-월 기준으로 집계
                    yearmonth_data = customer_purchases.groupby(['year', 'month', 'yearmonth']).agg(agg_dict).reset_index()
                    
                    for _, row in yearmonth_data.iterrows():
                        year = int(row['year'])
                        month = int(row['month'])
                        yearmonth = row['yearmonth']
                        
                        month_data = {'수량': int(row['수량'])}
                        
                        if has_amount:
                            month_data['금액'] = int(row['금액'])
                        else:
                            month_data['금액'] = 0
                        
                        # 날짜 정보를 키로 저장 (연-월)
                        yearmonth_purchases[yearmonth] = month_data
                        
                        # 기존 월별 구매 패턴도 유지 (이전 코드와의 호환성)
                        if month not in monthly_purchases:
                            monthly_purchases[month] = {'수량': 0, '금액': 0}
                        
                        monthly_purchases[month]['수량'] += month_data['수량']
                        monthly_purchases[month]['금액'] += month_data['금액']
            except Exception as e:
                st.warning(f"날짜 처리 중 오류 발생: {str(e)}")
                # 오류 발생 시 빈 딕셔너리로 초기화
                monthly_purchases = {}
                yearmonth_purchases = {}
                
        elif 'month' in customer_purchases.columns:
            # 기존 month 컬럼만 있는 경우 (이전 버전 호환)
            try:
                agg_dict = {'수량': 'sum'}
                if has_amount:
                    agg_dict['금액'] = 'sum'
                
                monthly_data = customer_purchases.groupby('month').agg(agg_dict).reset_index()
                
                for _, row in monthly_data.iterrows():
                    month = int(row['month'])
                    month_data = {'수량': int(row['수량'])}
                    
                    if has_amount:
                        month_data['금액'] = int(row['금액'])
                    else:
                        month_data['금액'] = 0
                        
                    monthly_purchases[month] = month_data
            except Exception as e:
                st.warning(f"월별 데이터 처리 중 오류 발생: {str(e)}")
                monthly_purchases = {}
        
        # 연-월별 상품 구매 내역 및 날짜별 내역
        yearmonth_product_purchases = {}
        yearmonth_purchase_dates = {}
        
        if 'yearmonth' in customer_purchases.columns:
            try:
                # 연-월 기준으로 그룹화
                for yearmonth, yearmonth_group in customer_purchases.groupby('yearmonth'):
                    if pd.isna(yearmonth):
                        continue
                        
                    # 해당 연-월의 모든 상품 구매 내역
                    all_products = yearmonth_group.groupby('상품')['수량'].sum().sort_values(ascending=False)
                    yearmonth_product_purchases[yearmonth] = all_products.to_dict()
                    
                    # 해당 연-월의 날짜별 구매 기록
                    date_purchases = {}
                    if '날짜' in yearmonth_group.columns:
                        for date, date_group in yearmonth_group.groupby('날짜'):
                            if pd.isna(date):
                                continue
                            date_str = date.strftime('%Y-%m-%d')
                            date_products = {}
                            for _, row in date_group.iterrows():
                                product = row['상품']
                                quantity = row['수량']
                                date_products[product] = int(quantity)
                            date_purchases[date_str] = date_products
                        yearmonth_purchase_dates[yearmonth] = date_purchases
            except Exception as e:
                st.warning(f"연월별 데이터 처리 중 오류 발생: {str(e)}")
        
        # 월별 상품 구매 내역 (이전 버전 호환)
        monthly_product_purchases = {}
        monthly_purchase_dates = {}
        
        if 'month' in customer_purchases.columns:
            try:
                for month, month_group in customer_purchases.groupby('month'):
                    if pd.isna(month):
                        continue
                    month = int(month)
                    # 모든 구매 상품 포함
                    all_products = month_group.groupby('상품')['수량'].sum().sort_values(ascending=False)
                    monthly_product_purchases[month] = all_products.to_dict()
                    
                    # 해당 월의 날짜별 구매 기록 추가
                    if '날짜' in month_group.columns:
                        date_purchases = {}
                        for date, date_group in month_group.groupby('날짜'):
                            if pd.isna(date):
                                continue
                            date_str = date.strftime('%Y-%m-%d')
                            date_products = {}
                            for _, row in date_group.iterrows():
                                product = row['상품']
                                quantity = row['수량']
                                date_products[product] = int(quantity)
                            date_purchases[date_str] = date_products
                        monthly_purchase_dates[month] = date_purchases
            except Exception as e:
                st.warning(f"월별 데이터 처리 중 오류 발생: {str(e)}")
        
        # 구매 상품 TOP 5
        top_products = customer_purchases.groupby('상품')['수량'].sum().sort_values(ascending=False).head(5)
        
        # 계절별 선호도 분석
        seasonal_preference = {
            '봄': 0,  # 3-5월
            '여름': 0,  # 6-8월
            '가을': 0,  # 9-11월
            '겨울': 0   # 12-2월
        }
        
        if 'month' in customer_purchases.columns:
            try:
                for month, group in customer_purchases.groupby('month'):
                    if pd.isna(month):
                        continue
                    month = int(month)
                    quantity = group['수량'].sum()
                    
                    if month in [3, 4, 5]:
                        seasonal_preference['봄'] += quantity
                    elif month in [6, 7, 8]:
                        seasonal_preference['여름'] += quantity
                    elif month in [9, 10, 11]:
                        seasonal_preference['가을'] += quantity
                    else:  # 12, 1, 2
                        seasonal_preference['겨울'] += quantity
            except Exception as e:
                st.warning(f"계절별 분석 중 오류 발생: {str(e)}")
        
        # 분기별 선호도 분석
        quarterly_preference = {
            '1분기': 0,  # 1-3월
            '2분기': 0,  # 4-6월
            '3분기': 0,  # 7-9월
            '4분기': 0   # 10-12월
        }
        
        if 'month' in customer_purchases.columns:
            try:
                for month, group in customer_purchases.groupby('month'):
                    if pd.isna(month):
                        continue
                    month = int(month)
                    quantity = group['수량'].sum()
                    
                    if month in [1, 2, 3]:
                        quarterly_preference['1분기'] += quantity
                    elif month in [4, 5, 6]:
                        quarterly_preference['2분기'] += quantity
                    elif month in [7, 8, 9]:
                        quarterly_preference['3분기'] += quantity
                    else:  # 10, 11, 12
                        quarterly_preference['4분기'] += quantity
            except Exception as e:
                st.warning(f"분기별 분석 중 오류 발생: {str(e)}")
        
        # 반품 정보
        refund_info = {}
        if self.refund_data is not None:
            # 반품 데이터에 '고객명' 컬럼이 있는지 확인
            if '고객명' in self.refund_data.columns:
                customer_refunds = self.refund_data[self.refund_data['고객명'] == customer_name]
                refund_qty = customer_refunds['수량'].sum()
                refund_ratio = abs(refund_qty) / (total_quantity + 0.1) * 100
                
                # 반품 사유별 집계 (반품사유 컬럼이 있는 경우만)
                if '반품사유' in self.refund_data.columns:
                    refund_types = customer_refunds.groupby('반품사유')['수량'].sum().abs()
                    refund_reasons = refund_types.to_dict() if not refund_types.empty else {}
                else:
                    refund_reasons = {}
                
                refund_info = {
                    '반품_수량': abs(refund_qty),
                    '반품_비율': refund_ratio,
                    '반품_이유': refund_reasons
                }
            else:
                # 고객명 컬럼이 없으면 빈 정보 반환
                refund_info = {
                    '반품_수량': 0,
                    '반품_비율': 0,
                    '반품_이유': {}
                }
        
        # 최근 구매일
        latest_purchase = None
        if '날짜' in customer_purchases.columns:
            try:
                temp_dates = pd.to_datetime(customer_purchases['날짜'], errors='coerce')
                valid_dates = temp_dates.dropna()
                if not valid_dates.empty:
                    latest_date = valid_dates.max()
                    if not pd.isna(latest_date):
                        latest_purchase = latest_date.strftime('%Y-%m-%d')
            except Exception as e:
                st.warning(f"최근 구매일 계산 중 오류 발생: {str(e)}")
        
        # 구매 빈도 계산 (날짜 기반)
        purchase_frequency = 0
        unique_days = 0
        purchase_dates = []
        
        if '날짜' in customer_purchases.columns:
            try:
                # 날짜 컬럼을 datetime으로 변환
                temp_dates = pd.to_datetime(customer_purchases['날짜'], errors='coerce')
                valid_dates = temp_dates.dropna()
                
                if not valid_dates.empty:
                    unique_days = valid_dates.dt.date.nunique()
                    purchase_dates = sorted(valid_dates.dt.date.unique())
                    first_date = valid_dates.min()
                    last_date = valid_dates.max()
                    
                    if not pd.isna(first_date) and not pd.isna(last_date):
                        total_days = (last_date - first_date).days + 1
                        purchase_frequency = unique_days / max(total_days, 1) * 100
            except Exception as e:
                st.warning(f"구매 빈도 계산 중 오류 발생: {str(e)}")
        
        # 제품별 구매일 변화 추적 (제품 구매 패턴 파악)
        product_purchase_history = {}
        if '날짜' in customer_purchases.columns:
            try:
                # 날짜 컬럼을 datetime으로 변환
                temp_purchases = customer_purchases.copy()
                temp_purchases['날짜'] = pd.to_datetime(temp_purchases['날짜'], errors='coerce')
                
                for product, product_group in temp_purchases.groupby('상품'):
                    dates = []
                    quantities = []
                    for _, row in product_group.iterrows():
                        if not pd.isna(row['날짜']):
                            date_str = row['날짜'].strftime('%Y-%m-%d')
                            dates.append(date_str)
                            quantities.append(int(row['수량']))
                    
                    if dates:  # 유효한 날짜가 있는 경우만 추가
                        product_purchase_history[product] = {
                            '구매일': dates,
                            '구매량': quantities
                        }
            except Exception as e:
                st.warning(f"제품별 구매 이력 계산 중 오류 발생: {str(e)}")
        
        return {
            '상태': '성공',
            '고객명': customer_name,
            '고객_카테고리': customer_category,
            '고객_코드': customer_code,
            '총_구매량': total_quantity,
            '총_구매금액': total_amount,
            '월별_구매': monthly_purchases,
            '연월별_구매': yearmonth_purchases,  # 연-월 정보가 포함된 구매 데이터
            '월별_상품_구매': monthly_product_purchases,
            '연월별_상품_구매': yearmonth_product_purchases,  # 연-월 정보가 포함된 상품 구매 데이터
            '월별_구매_날짜': monthly_purchase_dates,
            '연월별_구매_날짜': yearmonth_purchase_dates,  # 연-월 정보가 포함된 날짜별 구매 데이터
            '주요_구매상품': top_products.to_dict(),
            '계절별_선호도': seasonal_preference,
            '분기별_선호도': quarterly_preference,
            '반품_정보': refund_info,
            '구매_빈도': purchase_frequency,
            '구매_날짜': [d.strftime('%Y-%m-%d') for d in purchase_dates],
            '구매일수': unique_days,
            '제품별_구매_이력': product_purchase_history,
            '최근_구매일': latest_purchase
        }

    def get_customer_categories(self):
        """고객을 카테고리별로 분류"""
        customer_categories = {'호텔': [], '일반': []}
        
        for customer in self.customer_product_matrix.index:
            if '재고조정' in customer or '문정창고' in customer or '창고' in customer:
                continue
                
            try:
                code_match = re.match(r'^(\d{3})', customer)
                if code_match:
                    customer_code = code_match.group(1)
                    if customer_code in ['001', '005']:
                        customer_categories['호텔'].append(customer)
                    else:
                        customer_categories['일반'].append(customer)
                else:
                    customer_categories['일반'].append(customer)
            except:
                customer_categories['일반'].append(customer)
                
        return customer_categories

    def perform_rfm_analysis(self, customer_type='전체', selected_month=None):
        """RFM 고객 세분화 분석"""
        if '날짜' not in self.sales_data.columns or self.sales_data['날짜'].isnull().all():
            return {
                '상태': '실패',
                '메시지': "날짜 정보가 없어 RFM 분석을 수행할 수 없습니다."
            }
        
        valid_sales = self.sales_data.copy()
        
        # 데이터 타입 확인 및 변환
        try:
            # 금액 컬럼이 없거나 모두 NaN인 경우 수량으로 대체
            if '금액' not in valid_sales.columns or valid_sales['금액'].isnull().all():
                valid_sales['금액'] = valid_sales['수량'] * 1000  # 임시 단가 적용
                st.warning("금액 정보가 없어 수량 기반으로 RFM 분석을 수행합니다.")
            else:
                # 금액 컬럼을 숫자로 변환
                valid_sales['금액'] = pd.to_numeric(valid_sales['금액'], errors='coerce')
                # NaN 값을 0으로 대체
                valid_sales['금액'] = valid_sales['금액'].fillna(0)
            
            # 수량 컬럼도 숫자로 변환
            valid_sales['수량'] = pd.to_numeric(valid_sales['수량'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['수량'])
            
        except Exception as e:
            st.error(f"데이터 타입 변환 중 오류: {str(e)}")
            return {
                '상태': '실패',
                '메시지': "데이터 타입 변환에 실패했습니다."
            }
        
        # 고객 유형 필터링
        customer_categories = self.get_customer_categories()
        if customer_type == '호텔':
            valid_sales = valid_sales[valid_sales['고객명'].isin(customer_categories['호텔'])]
        elif customer_type == '일반':
            valid_sales = valid_sales[valid_sales['고객명'].isin(customer_categories['일반'])]
        
        if valid_sales.empty:
            return {
                '상태': '실패',
                '메시지': f"{customer_type} 고객 유형에 대한 유효한 판매 데이터가 없습니다."
            }
        
        # 특정 월 필터링
        if selected_month is not None:
            # month 컬럼이 있는지 확인
            if 'month' in valid_sales.columns:
                valid_sales = valid_sales[valid_sales['month'] == selected_month]
            else:
                try:
                    # 날짜 컬럼을 안전하게 처리
                    temp_sales = valid_sales.copy()
                    temp_sales['날짜'] = pd.to_datetime(temp_sales['날짜'], errors='coerce')
                    valid_sales = temp_sales[temp_sales['날짜'].dt.month == selected_month]
                except Exception as e:
                    st.warning(f"월별 필터링 중 오류 발생: {str(e)}")
                    return {
                        '상태': '실패',
                        '메시지': f"월별 필터링에 실패했습니다."
                    }
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': f"{selected_month}월에 대한 유효한 판매 데이터가 없습니다."
                }
        
        # 최근 날짜 계산
        max_date = valid_sales['날짜'].max()
        
        # RFM 분석을 위한 고객별 지표 계산
        try:
            rfm_data = valid_sales.groupby('고객명').agg({
                '날짜': lambda x: (max_date - x.max()).days,
                '상품': 'count',
                '금액': 'sum'
            }).reset_index()
            
            rfm_data.rename(columns={
                '날짜': 'Recency',
                '상품': 'Frequency',
                '금액': 'Monetary'
            }, inplace=True)
            
            # 데이터 타입 확인
            rfm_data['Recency'] = pd.to_numeric(rfm_data['Recency'], errors='coerce')
            rfm_data['Frequency'] = pd.to_numeric(rfm_data['Frequency'], errors='coerce')
            rfm_data['Monetary'] = pd.to_numeric(rfm_data['Monetary'], errors='coerce')
            
            # NaN 값 제거
            rfm_data = rfm_data.dropna()
            
            if rfm_data.empty:
                return {
                    '상태': '실패',
                    '메시지': "RFM 분석을 위한 유효한 데이터가 없습니다."
                }
            
        except Exception as e:
            st.error(f"RFM 데이터 집계 중 오류: {str(e)}")
            return {
                '상태': '실패',
                '메시지': "RFM 데이터 집계에 실패했습니다."
            }
        
        # RFM 점수 계산
        try:
            if len(rfm_data) >= 4:
                rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
                rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
                rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
            else:
                r_median = rfm_data['Recency'].median()
                f_median = rfm_data['Frequency'].median()
                m_median = rfm_data['Monetary'].median()
                
                rfm_data['R_Score'] = rfm_data['Recency'].apply(lambda x: 4 if x <= r_median else 1)
                rfm_data['F_Score'] = rfm_data['Frequency'].apply(lambda x: 4 if x >= f_median else 1)
                rfm_data['M_Score'] = rfm_data['Monetary'].apply(lambda x: 4 if x >= m_median else 1)
        except Exception as e:
            st.warning(f"RFM 점수 계산 중 오류 발생: {e}")
            return {'상태': '실패', '메시지': "RFM 점수 계산에 실패했습니다."}
        
        # 고객 세그먼트 정의
        def segment_customer(row):
            try:
                r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
                
                if r >= 4 and f >= 4 and m >= 4:
                    return "Champions"
                elif r >= 3 and f >= 3 and m >= 3:
                    return "Loyal Customers"
                elif r >= 3 and f <= 2:
                    return "Potential Loyalists"
                elif r <= 2 and f >= 3:
                    return "At Risk"
                elif r <= 2 and f <= 2 and m >= 3:
                    return "Can't Lose Them"
                elif r <= 2 and f <= 2 and m <= 2:
                    return "Lost"
                else:
                    return "Others"
            except:
                return "Others"
        
        rfm_data['Segment'] = rfm_data.apply(segment_customer, axis=1)
        
        # 세그먼트별 통계
        try:
            segment_stats = rfm_data.groupby('Segment').agg({
                'Recency': ['mean', 'min', 'max'],
                'Frequency': ['mean', 'min', 'max'],
                'Monetary': ['mean', 'min', 'max'],
                '고객명': 'count'
            }).round(2)
        except Exception as e:
            st.warning(f"세그먼트 통계 계산 중 오류: {str(e)}")
            segment_stats = {}
        
        return {
            '상태': '성공',
            'RFM_데이터': rfm_data,
            '세그먼트_통계': segment_stats.to_dict() if hasattr(segment_stats, 'to_dict') else {},
            '고객_유형': customer_type,
            '분석_월': selected_month
        }

    def analyze_dining_vip_metrics(self):
        """다이닝 VIP 지표 분석 - 지정된 8개 업체"""
        try:
            # 지정된 8개 업체 리스트 (실제 데이터에 존재하는 업체명)
            target_customers = [
                "002_알라프리마(Alla prima)",
                "002_주식회사 콘피에르", 
                "002_주식회사 스와니예",
                "002_*신금유통",
                "002_정식당",
                "002_#구찌오스테리아",
                "002_콘피에르셀렉션"
            ]
            
            # 금액 컬럼이 있는지 확인
            if '금액' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "금액 정보가 없어 매출 분석을 수행할 수 없습니다."
                }
            
            # 날짜 정보가 있는지 확인
            if '날짜' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "날짜 정보가 없어 매출 분석을 수행할 수 없습니다."
                }
            
            # 유효한 데이터 필터링
            valid_sales = self.sales_data.copy()
            valid_sales['금액'] = pd.to_numeric(valid_sales['금액'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['금액', '날짜'])
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': "유효한 매출 데이터가 없습니다."
                }
            
            # 날짜 처리
            valid_sales['날짜'] = pd.to_datetime(valid_sales['날짜'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['날짜'])
            valid_sales['연월'] = valid_sales['날짜'].dt.strftime('%Y-%m')
            valid_sales['월'] = valid_sales['날짜'].dt.month
            
            # 지정된 8개 업체만 필터링
            valid_sales = valid_sales[valid_sales['고객명'].isin(target_customers)]
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': "지정된 8개 업체의 매출 데이터가 없습니다."
                }
            
            # 업체별 매출 계산
            customer_revenue = valid_sales.groupby('고객명')['금액'].sum().sort_values(ascending=False)
            selected_customers = customer_revenue.index.tolist()
            
            # 선택된 고객들의 월별 매출 추이
            selected_data = valid_sales[valid_sales['고객명'].isin(selected_customers)]
            
            # 연월별 매출 집계
            monthly_revenue = {}
            product_revenue = {}
            yearmonth_product_data = {}
            
            # 업체별 연월별 매출
            yearmonth_revenue = {}
            
            for customer in selected_customers:
                customer_data = selected_data[selected_data['고객명'] == customer]
                
                # 월별 매출 (기존)
                monthly_data = customer_data.groupby('월')['금액'].sum().to_dict()
                monthly_revenue[customer] = monthly_data
                
                # 연월별 매출 (새로 추가)
                yearmonth_data = customer_data.groupby('연월')['금액'].sum().to_dict()
                yearmonth_revenue[customer] = yearmonth_data
                
                # 품목별 매출
                product_data = customer_data.groupby('상품')['금액'].sum().sort_values(ascending=False).head(10).to_dict()
                product_revenue[customer] = product_data
                
                # 연월별 상품 구매 데이터
                yearmonth_products = customer_data.groupby(['연월', '상품']).agg({
                    '수량': 'sum',
                    '금액': 'sum'
                }).reset_index()
                yearmonth_product_data[customer] = yearmonth_products
            
            # 전체 7개 업체 통합 연월별 TOP 10 상품
            all_yearmonth_products = selected_data.groupby(['연월', '상품']).agg({
                '수량': 'sum',
                '금액': 'sum'
            }).reset_index()
            
            # 각 연월별로 TOP 10 상품 선정
            monthly_top10_products = {}
            for yearmonth in all_yearmonth_products['연월'].unique():
                month_data = all_yearmonth_products[all_yearmonth_products['연월'] == yearmonth]
                top10 = month_data.nlargest(10, '금액')[['상품', '수량', '금액']].to_dict('records')
                monthly_top10_products[yearmonth] = top10
            
            return {
                '상태': '성공',
                'selected_customers': selected_customers,
                'customer_total_revenue': customer_revenue.to_dict(),
                'monthly_revenue': monthly_revenue,
                'yearmonth_revenue': yearmonth_revenue,
                'product_revenue': product_revenue,
                'yearmonth_product_data': yearmonth_product_data,
                'monthly_top10_products': monthly_top10_products
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"다이닝 VIP 분석 중 오류 발생: {str(e)}"
            }

    def analyze_hotel_vip_metrics(self):
        """호텔 VIP 지표 분석 - 지정된 5개 호텔"""
        try:
            # 지정된 5개 호텔 키워드 리스트
            hotel_keywords = ["포시즌스", "소피텔", "인스파이어", "조선팰리스", "웨스틴조선"]
            
            # 금액 컬럼이 있는지 확인
            if '금액' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "금액 정보가 없어 호텔 VIP 분석을 수행할 수 없습니다."
                }
            
            # 날짜 정보가 있는지 확인
            if '날짜' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "날짜 정보가 없어 호텔 VIP 분석을 수행할 수 없습니다."
                }
            
            # 유효한 데이터 필터링
            valid_sales = self.sales_data.copy()
            valid_sales['금액'] = pd.to_numeric(valid_sales['금액'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['금액', '날짜'])
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': "유효한 매출 데이터가 없습니다."
                }
            
            # 날짜 처리
            valid_sales['날짜'] = pd.to_datetime(valid_sales['날짜'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['날짜'])
            valid_sales['연월'] = valid_sales['날짜'].dt.strftime('%Y-%m')
            valid_sales['월'] = valid_sales['날짜'].dt.month
            
            # 호텔 키워드가 포함된 고객명 찾기
            hotel_customers = []
            for keyword in hotel_keywords:
                matching_customers = valid_sales[valid_sales['고객명'].str.contains(keyword, case=False, na=False)]['고객명'].unique()
                hotel_customers.extend(matching_customers)
            
            # 중복 제거
            hotel_customers = list(set(hotel_customers))
            
            if not hotel_customers:
                return {
                    '상태': '실패',
                    '메시지': "지정된 5개 호텔의 매출 데이터가 없습니다."
                }
            
            # 호텔 고객만 필터링
            valid_sales = valid_sales[valid_sales['고객명'].isin(hotel_customers)]
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': "호텔 고객의 매출 데이터가 없습니다."
                }
            
            # 업체별 매출 계산
            customer_revenue = valid_sales.groupby('고객명')['금액'].sum().sort_values(ascending=False)
            selected_customers = customer_revenue.index.tolist()
            
            # 선택된 고객들의 월별 매출 추이
            selected_data = valid_sales[valid_sales['고객명'].isin(selected_customers)]
            
            # 연월별 매출 집계
            monthly_revenue = {}
            product_revenue = {}
            yearmonth_product_data = {}
            
            # 업체별 연월별 매출
            yearmonth_revenue = {}
            
            for customer in selected_customers:
                customer_data = selected_data[selected_data['고객명'] == customer]
                
                # 월별 매출 (기존)
                monthly_data = customer_data.groupby('월')['금액'].sum().to_dict()
                monthly_revenue[customer] = monthly_data
                
                # 연월별 매출 (새로 추가)
                yearmonth_data = customer_data.groupby('연월')['금액'].sum().to_dict()
                yearmonth_revenue[customer] = yearmonth_data
                
                # 품목별 매출
                product_data = customer_data.groupby('상품')['금액'].sum().sort_values(ascending=False).head(10).to_dict()
                product_revenue[customer] = product_data
                
                # 연월별 상품 구매 데이터
                yearmonth_products = customer_data.groupby(['연월', '상품']).agg({
                    '수량': 'sum',
                    '금액': 'sum'
                }).reset_index()
                yearmonth_product_data[customer] = yearmonth_products
            
            # 전체 호텔 통합 연월별 TOP 10 상품
            all_yearmonth_products = selected_data.groupby(['연월', '상품']).agg({
                '수량': 'sum',
                '금액': 'sum'
            }).reset_index()
            
            # 각 연월별로 TOP 10 상품 선정
            monthly_top10_products = {}
            for yearmonth in all_yearmonth_products['연월'].unique():
                month_data = all_yearmonth_products[all_yearmonth_products['연월'] == yearmonth]
                top10 = month_data.nlargest(10, '금액')[['상품', '수량', '금액']].to_dict('records')
                monthly_top10_products[yearmonth] = top10
            
            return {
                '상태': '성공',
                'selected_customers': selected_customers,
                'customer_total_revenue': customer_revenue.to_dict(),
                'monthly_revenue': monthly_revenue,
                'yearmonth_revenue': yearmonth_revenue,
                'product_revenue': product_revenue,
                'yearmonth_product_data': yearmonth_product_data,
                'monthly_top10_products': monthly_top10_products
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"호텔 VIP 분석 중 오류 발생: {str(e)}"
            }

    def analyze_banquet_metrics(self):
        """
        비고 컬럼에 BANQUET 키워드가 포함된 고객들의 매출 지표 분석
        다이닝 VIP와 동일한 구조로 분석 결과 반환
        """
        try:
            print("BANQUET 분석 시작...")
            
            # 데이터 유효성 확인
            if self.sales_data.empty:
                return {
                    '상태': '실패',
                    '메시지': 'BANQUET 분석을 위한 데이터가 없습니다.'
                }
            
            print(f"데이터 크기: {len(self.sales_data)}개 레코드")
            print(f"컬럼: {self.sales_data.columns.tolist()}")
            
            # 비고 컬럼 확인
            if '비고' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': '비고 컬럼이 없어 BANQUET 분석을 수행할 수 없습니다.'
                }
            
            # 유효한 데이터만 필터링
            valid_data = self.sales_data.dropna(subset=['고객명', '상품', '날짜', '금액'])
            print(f"유효한 데이터: {len(valid_data)}개 레코드")
            
            # BANQUET 관련 키워드로 데이터 필터링
            print("BANQUET 관련 고객 검색 중...")
            banquet_keywords = ['banquet', 'BANQUET', '연회', '웨딩', '파티', '행사']
            
            banquet_data = pd.DataFrame()
            for keyword in banquet_keywords:
                keyword_data = valid_data[valid_data['비고'].str.contains(keyword, case=False, na=False)]
                print(f"'{keyword}' 키워드: {len(keyword_data)}개 레코드")
                if len(keyword_data) > 0:
                    print(f"  예시 비고: {keyword_data['비고'].head(3).tolist()}")
                banquet_data = pd.concat([banquet_data, keyword_data], ignore_index=True)
            
            # 중복 제거
            banquet_data = banquet_data.drop_duplicates()
            print(f"BANQUET 관련 데이터: {len(banquet_data)}개 레코드")
            
            if banquet_data.empty:
                return {
                    '상태': '실패',
                    '메시지': 'BANQUET 관련 데이터를 찾을 수 없습니다.'
                }
            
            # 발견된 BANQUET 고객들
            banquet_customers = banquet_data['고객명'].unique().tolist()
            print(f"발견된 BANQUET 고객: {banquet_customers}")
            
            # 각 고객별 분석
            customer_data = {}
            customer_total_revenue = {}
            product_revenue = {}
            yearmonth_revenue = {}
            
            for customer in banquet_customers:
                print(f"고객 '{customer}' 분석 중...")
                customer_records = banquet_data[banquet_data['고객명'] == customer]
                print(f"  -> {len(customer_records)}개 레코드")
                
                # 총 매출
                total_revenue = customer_records['금액'].sum()
                print(f"  -> 총 매출: {total_revenue:,.0f}원")
                
                customer_total_revenue[customer] = total_revenue
                
                # 상품별 매출
                product_sales = customer_records.groupby('상품')['금액'].sum().sort_values(ascending=False)
                product_revenue[customer] = product_sales.to_dict()
                
                # 연월별 매출
                customer_records_copy = customer_records.copy()
                customer_records_copy['연월'] = customer_records_copy['날짜'].dt.strftime('%Y-%m')
                monthly_sales = customer_records_copy.groupby('연월')['금액'].sum()
                yearmonth_revenue[customer] = monthly_sales.to_dict()
                
                print(f"  -> 고객 데이터 저장 완료")
            
            # 연월별 통합 TOP 10 상품 분석
            monthly_top10_products = {}
            banquet_data_copy = banquet_data.copy()
            banquet_data_copy['연월'] = banquet_data_copy['날짜'].dt.strftime('%Y-%m')
            
            for month in banquet_data_copy['연월'].unique():
                month_data = banquet_data_copy[banquet_data_copy['연월'] == month]
                product_summary = month_data.groupby('상품').agg({
                    '수량': 'sum',
                    '금액': 'sum'
                }).sort_values('금액', ascending=False).head(10)
                
                monthly_top10_products[month] = [
                    {
                        '상품': product,
                        '수량': row['수량'],
                        '금액': row['금액']
                    }
                    for product, row in product_summary.iterrows()
                ]
            
            print(f"BANQUET 분석 완료: {len(banquet_customers)}개 고객 발견")
            
            return {
                '상태': '성공',
                'found_customers': banquet_customers,
                'customer_total_revenue': customer_total_revenue,
                'product_revenue': product_revenue,
                'yearmonth_revenue': yearmonth_revenue,
                'monthly_top10_products': monthly_top10_products,
                'customer_data': customer_data
            }
            
        except Exception as e:
            print(f"BANQUET 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return {
                '상태': '실패',
                '메시지': f"BANQUET 분석 중 오류 발생: {str(e)}"
            }

    def analyze_monthly_dining_sales(self):
        """월별 다이닝(호텔 제외) 매출 분석"""
        try:
            # 금액 컬럼이 있는지 확인
            if '금액' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "금액 정보가 없어 매출 분석을 수행할 수 없습니다."
                }
            
            # 유효한 데이터 필터링
            valid_sales = self.sales_data.copy()
            valid_sales['금액'] = pd.to_numeric(valid_sales['금액'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['금액', '날짜'])
            
            # 다이닝 고객만 필터링 (호텔 제외)
            customer_categories = self.get_customer_categories()
            dining_customers = customer_categories['일반']
            dining_sales = valid_sales[valid_sales['고객명'].isin(dining_customers)]
            
            if dining_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': "다이닝 매출 데이터가 없습니다."
                }
            
            # 연월 정보 추가
            dining_sales = dining_sales.copy()
            dining_sales['날짜'] = pd.to_datetime(dining_sales['날짜'], errors='coerce')
            dining_sales = dining_sales.dropna(subset=['날짜'])
            dining_sales['연월'] = dining_sales['날짜'].dt.strftime('%Y-%m')
            dining_sales['연도'] = dining_sales['날짜'].dt.year
            dining_sales['월'] = dining_sales['날짜'].dt.month
            
            # 월별 총 매출
            monthly_total = dining_sales.groupby('연월')['금액'].sum().reset_index()
            
            # 월별 업체별 매출
            monthly_customer = dining_sales.groupby(['연월', '고객명'])['금액'].sum().reset_index()
            
            # 월별 품목별 매출
            monthly_product = dining_sales.groupby(['연월', '상품'])['금액'].sum().reset_index()
            
            # Heatmap용 데이터 준비
            # 업체별 월별 매출 피벗 테이블
            customer_heatmap = dining_sales.pivot_table(
                index='고객명', 
                columns='연월', 
                values='금액', 
                aggfunc='sum', 
                fill_value=0
            )
            
            # 품목별 월별 매출 피벗 테이블 (상위 20개 품목만)
            top_products = dining_sales.groupby('상품')['금액'].sum().nlargest(20).index
            product_sales_filtered = dining_sales[dining_sales['상품'].isin(top_products)]
            product_heatmap = product_sales_filtered.pivot_table(
                index='상품', 
                columns='연월', 
                values='금액', 
                aggfunc='sum', 
                fill_value=0
            )
            
            return {
                '상태': '성공',
                'monthly_total': monthly_total,
                'monthly_customer': monthly_customer,
                'monthly_product': monthly_product,
                'customer_heatmap': customer_heatmap,
                'product_heatmap': product_heatmap
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"다이닝 매출 분석 중 오류 발생: {str(e)}"
            }

    def analyze_monthly_hotel_sales(self):
        """월별 호텔 매출 분석"""
        try:
            # 금액 컬럼이 있는지 확인
            if '금액' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "금액 정보가 없어 매출 분석을 수행할 수 없습니다."
                }
            
            # 유효한 데이터 필터링
            valid_sales = self.sales_data.copy()
            valid_sales['금액'] = pd.to_numeric(valid_sales['금액'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['금액', '날짜'])
            
            # 호텔 고객만 필터링
            customer_categories = self.get_customer_categories()
            hotel_customers = customer_categories['호텔']
            hotel_sales = valid_sales[valid_sales['고객명'].isin(hotel_customers)]
            
            if hotel_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': "호텔 매출 데이터가 없습니다."
                }
            
            # 연월 정보 추가
            hotel_sales = hotel_sales.copy()
            hotel_sales['날짜'] = pd.to_datetime(hotel_sales['날짜'], errors='coerce')
            hotel_sales = hotel_sales.dropna(subset=['날짜'])
            hotel_sales['연월'] = hotel_sales['날짜'].dt.strftime('%Y-%m')
            hotel_sales['연도'] = hotel_sales['날짜'].dt.year
            hotel_sales['월'] = hotel_sales['날짜'].dt.month
            
            # 월별 총 매출
            monthly_total = hotel_sales.groupby('연월')['금액'].sum().reset_index()
            
            # 월별 업체별 매출
            monthly_customer = hotel_sales.groupby(['연월', '고객명'])['금액'].sum().reset_index()
            
            # 월별 품목별 매출
            monthly_product = hotel_sales.groupby(['연월', '상품'])['금액'].sum().reset_index()
            
            # Heatmap용 데이터 준비
            # 업체별 월별 매출 피벗 테이블
            customer_heatmap = hotel_sales.pivot_table(
                index='고객명', 
                columns='연월', 
                values='금액', 
                aggfunc='sum', 
                fill_value=0
            )
            
            # 품목별 월별 매출 피벗 테이블 (상위 20개 품목만)
            top_products = hotel_sales.groupby('상품')['금액'].sum().nlargest(20).index
            product_sales_filtered = hotel_sales[hotel_sales['상품'].isin(top_products)]
            product_heatmap = product_sales_filtered.pivot_table(
                index='상품', 
                columns='연월', 
                values='금액', 
                aggfunc='sum', 
                fill_value=0
            )
            
            return {
                '상태': '성공',
                'monthly_total': monthly_total,
                'monthly_customer': monthly_customer,
                'monthly_product': monthly_product,
                'customer_heatmap': customer_heatmap,
                'product_heatmap': product_heatmap
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"호텔 매출 분석 중 오류 발생: {str(e)}"
            }

    def analyze_yearly_sales_comparison(self):
        """연별 다이닝/호텔 매출 비교 분석"""
        try:
            # 금액 컬럼이 있는지 확인
            if '금액' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "금액 정보가 없어 매출 분석을 수행할 수 없습니다."
                }
            
            # 유효한 데이터 필터링
            valid_sales = self.sales_data.copy()
            valid_sales['금액'] = pd.to_numeric(valid_sales['금액'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['금액', '날짜'])
            
            # 날짜 정보 추가
            valid_sales['날짜'] = pd.to_datetime(valid_sales['날짜'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['날짜'])
            valid_sales['연도'] = valid_sales['날짜'].dt.year
            valid_sales['월'] = valid_sales['날짜'].dt.month
            
            # 고객 카테고리 분류
            customer_categories = self.get_customer_categories()
            
            # 다이닝과 호텔 데이터 분리
            dining_sales = valid_sales[valid_sales['고객명'].isin(customer_categories['일반'])]
            hotel_sales = valid_sales[valid_sales['고객명'].isin(customer_categories['호텔'])]
            
            # 연별 매출 집계
            yearly_dining = dining_sales.groupby('연도')['금액'].sum().reset_index()
            yearly_dining['카테고리'] = '다이닝'
            
            yearly_hotel = hotel_sales.groupby('연도')['금액'].sum().reset_index()
            yearly_hotel['카테고리'] = '호텔'
            
            # 연월별 매출 집계 (Heatmap용)
            dining_sales['연월'] = dining_sales['날짜'].dt.strftime('%Y-%m')
            hotel_sales['연월'] = hotel_sales['날짜'].dt.strftime('%Y-%m')
            
            # 다이닝 연도별 월별 매출 피벗
            dining_yearly_monthly = dining_sales.pivot_table(
                index='연도',
                columns='월',
                values='금액',
                aggfunc='sum',
                fill_value=0
            )
            
            # 호텔 연도별 월별 매출 피벗
            hotel_yearly_monthly = hotel_sales.pivot_table(
                index='연도',
                columns='월',
                values='금액',
                aggfunc='sum',
                fill_value=0
            )
            
            return {
                '상태': '성공',
                'yearly_dining': yearly_dining,
                'yearly_hotel': yearly_hotel,
                'dining_yearly_monthly': dining_yearly_monthly,
                'hotel_yearly_monthly': hotel_yearly_monthly
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"연별 매출 분석 중 오류 발생: {str(e)}"
            }

    def get_michelin_restaurants(self):
        """미슐랭 레스토랑 분류"""
        michelin_restaurants = {
            '3_STAR': ['밍글스'],
            '2_STAR': ['알렌&컨티뉴움', '미토우', '스와니예', '알라프리마', '정식당'],
            '1_STAR': ['강민철 레스토랑', '라망시크레', '비채나', '빈호', '소설한남', '소울', '솔밤', 
                      '익스퀴진 에스콘디도', '체로컴플렉스', '익스퀴진'],
            'SELECTED': ['줄라이', '페리지', '보르고한남', '홍연', '알레즈', '류니끄', '구찌오스테리아', 
                        '소바쥬 산로', '본앤브레드', '트리드', '일 베키오', '쉐시몽', '물랑']
        }
        return michelin_restaurants
    
    def classify_michelin_customers(self):
        """고객명을 기반으로 미슐랭 레스토랑 분류"""
        try:
            michelin_restaurants = self.get_michelin_restaurants()
            classified_customers = {}
            
            # 모든 고객명 가져오기
            all_customers = self.sales_data['고객명'].unique()
            
            for customer in all_customers:
                customer_str = str(customer).lower()
                
                # 각 미슐랭 등급별로 확인
                for grade, restaurants in michelin_restaurants.items():
                    for restaurant in restaurants:
                        if restaurant.lower() in customer_str:
                            if grade not in classified_customers:
                                classified_customers[grade] = []
                            classified_customers[grade].append(customer)
                            break
            
            return classified_customers
            
        except Exception as e:
            return {}
    
    def analyze_michelin_overview(self):
        """미슐랭 레스토랑 전체 개요 분석"""
        try:
            classified_customers = self.classify_michelin_customers()
            
            if not classified_customers:
                return {
                    '상태': '실패',
                    '메시지': '미슐랭 레스토랑 데이터를 찾을 수 없습니다.'
                }
            
            overview_data = {}
            total_sales = 0
            total_quantity = 0
            
            for grade, customers in classified_customers.items():
                grade_data = self.sales_data[self.sales_data['고객명'].isin(customers)]
                
                if not grade_data.empty:
                    # 금액 컬럼을 숫자로 변환
                    grade_data_clean = grade_data.copy()
                    grade_data_clean['금액'] = pd.to_numeric(grade_data_clean['금액'], errors='coerce')
                    grade_data_clean['수량'] = pd.to_numeric(grade_data_clean['수량'], errors='coerce')
                    
                    grade_sales = grade_data_clean['금액'].sum()
                    grade_quantity = grade_data_clean['수량'].sum()
                    
                    overview_data[grade] = {
                        '레스토랑_수': len(customers),
                        '총_매출': grade_sales,
                        '총_구매량': grade_quantity,
                        '평균_매출': grade_sales / len(customers) if len(customers) > 0 else 0,
                        '레스토랑_목록': customers
                    }
                    
                    total_sales += grade_sales
                    total_quantity += grade_quantity
            
            return {
                '상태': '성공',
                '등급별_데이터': overview_data,
                '전체_매출': total_sales,
                '전체_구매량': total_quantity,
                '분류된_고객수': sum(len(customers) for customers in classified_customers.values())
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"미슐랭 개요 분석 중 오류 발생: {str(e)}"
            }
    
    def analyze_michelin_by_grade(self, grade):
        """특정 미슐랭 등급의 상세 분석"""
        try:
            classified_customers = self.classify_michelin_customers()
            
            if grade not in classified_customers:
                return {
                    '상태': '실패',
                    '메시지': f'{grade} 등급의 레스토랑을 찾을 수 없습니다.'
                }
            
            customers = classified_customers[grade]
            grade_data = self.sales_data[self.sales_data['고객명'].isin(customers)]
            
            if grade_data.empty:
                return {
                    '상태': '실패',
                    '메시지': f'{grade} 등급의 판매 데이터가 없습니다.'
                }
            
            # 데이터 정리
            grade_data_clean = grade_data.copy()
            grade_data_clean['금액'] = pd.to_numeric(grade_data_clean['금액'], errors='coerce')
            grade_data_clean['수량'] = pd.to_numeric(grade_data_clean['수량'], errors='coerce')
            grade_data_clean = grade_data_clean.dropna(subset=['금액', '수량'])
            
            # 날짜 처리
            if '날짜' in grade_data_clean.columns:
                grade_data_clean['날짜'] = pd.to_datetime(grade_data_clean['날짜'], errors='coerce')
                grade_data_clean = grade_data_clean.dropna(subset=['날짜'])
                grade_data_clean['년월'] = grade_data_clean['날짜'].dt.to_period('M')
            
            # 레스토랑별 분석
            restaurant_analysis = {}
            for customer in customers:
                customer_data = grade_data_clean[grade_data_clean['고객명'] == customer]
                if not customer_data.empty:
                    restaurant_analysis[customer] = {
                        '총_매출': customer_data['금액'].sum(),
                        '총_구매량': customer_data['수량'].sum(),
                        '구매_품목수': customer_data['상품'].nunique(),
                        '거래_횟수': len(customer_data),
                        '평균_주문금액': customer_data['금액'].mean(),
                        '주요_품목': customer_data.groupby('상품')['수량'].sum().nlargest(5).to_dict()
                    }
            
            # 월별 매출 추이
            monthly_sales = {}
            if '년월' in grade_data_clean.columns:
                monthly_data = grade_data_clean.groupby('년월').agg({
                    '금액': 'sum',
                    '수량': 'sum'
                }).reset_index()
                monthly_data['년월_str'] = monthly_data['년월'].astype(str)
                monthly_sales = dict(zip(monthly_data['년월_str'], monthly_data['금액']))
            
            # 품목별 분석
            product_analysis = grade_data_clean.groupby('상품').agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            product_analysis = product_analysis.sort_values('금액', ascending=False)
            top_products = product_analysis.head(10).to_dict('records')
            
            return {
                '상태': '성공',
                '등급': grade,
                '레스토랑_분석': restaurant_analysis,
                '월별_매출': monthly_sales,
                '인기_품목': top_products,
                '총_매출': grade_data_clean['금액'].sum(),
                '총_구매량': grade_data_clean['수량'].sum(),
                '레스토랑_수': len(customers)
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"{grade} 등급 분석 중 오류 발생: {str(e)}"
            }
    
    def analyze_michelin_comparison(self):
        """미슐랭 등급간 비교 분석"""
        try:
            classified_customers = self.classify_michelin_customers()
            comparison_data = {}
            
            for grade in ['3_STAR', '2_STAR', '1_STAR', 'SELECTED']:
                if grade in classified_customers:
                    customers = classified_customers[grade]
                    grade_data = self.sales_data[self.sales_data['고객명'].isin(customers)]
                    
                    if not grade_data.empty:
                        # 데이터 정리
                        grade_data_clean = grade_data.copy()
                        grade_data_clean['금액'] = pd.to_numeric(grade_data_clean['금액'], errors='coerce')
                        grade_data_clean['수량'] = pd.to_numeric(grade_data_clean['수량'], errors='coerce')
                        grade_data_clean = grade_data_clean.dropna(subset=['금액', '수량'])
                        
                        comparison_data[grade] = {
                            '총_매출': grade_data_clean['금액'].sum(),
                            '총_구매량': grade_data_clean['수량'].sum(),
                            '평균_주문금액': grade_data_clean['금액'].mean(),
                            '레스토랑당_평균매출': grade_data_clean['금액'].sum() / len(customers) if len(customers) > 0 else 0,
                            '품목_다양성': grade_data_clean['상품'].nunique(),
                            '레스토랑_수': len(customers)
                        }
            
            return {
                '상태': '성공',
                '비교_데이터': comparison_data
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"미슐랭 비교 분석 중 오류 발생: {str(e)}"
            }

    def analyze_customer_sales_details(self, customer_name):
        """고객의 상세 판매 정보 분석"""
        customer_sales = self.sales_data[self.sales_data['고객명'] == customer_name]
        
        if customer_sales.empty:
            return {
                '상태': '실패',
                '메시지': f"고객 '{customer_name}'의 판매 데이터를 찾을 수 없습니다."
            }
        
        # 기본 통계
        total_quantity = int(customer_sales['수량'].sum())
        total_amount = int(customer_sales['금액'].sum()) if '금액' in customer_sales.columns else 0
        unique_products = customer_sales['상품'].nunique()
        
        # 최근 구매일
        if '날짜' in customer_sales.columns:
            try:
                customer_sales_copy = customer_sales.copy()
                customer_sales_copy['날짜'] = pd.to_datetime(customer_sales_copy['날짜'], errors='coerce')
                valid_dates = customer_sales_copy['날짜'].dropna()
                last_purchase_date = valid_dates.max().strftime('%Y-%m-%d') if not valid_dates.empty else None
            except:
                last_purchase_date = None
        else:
            last_purchase_date = None
        
        # 상품별 구매량
        product_quantities = customer_sales.groupby('상품')['수량'].sum().sort_values(ascending=False).head(10).to_dict()
        
        # 월별 구매 패턴
        monthly_pattern = {}
        if 'month' in customer_sales.columns:
            monthly_data = customer_sales.groupby('month')['수량'].sum()
            monthly_pattern = monthly_data.to_dict()
        
        return {
            '상태': '성공',
            '고객명': customer_name,
            '총_구매량': total_quantity,
            '총_구매금액': total_amount,
            '구매_상품수': unique_products,
            '최근_구매일': last_purchase_date,
            '주요_상품': product_quantities,
            '월별_구매': monthly_pattern
        }
    
    def analyze_churned_customers(self):
        """이탈 업체 관리: 최근 3개월간 구매 이력 없는 업체"""
        from datetime import datetime, timedelta
        
        today = datetime.now()
        three_months_ago = today - timedelta(days=90)
        
        try:
            # 날짜 컬럼을 datetime으로 변환
            sales_data_copy = self.sales_data.copy()
            sales_data_copy['날짜'] = pd.to_datetime(sales_data_copy['날짜'], errors='coerce')
            
            # 유효한 날짜만 필터링
            valid_sales = sales_data_copy[sales_data_copy['날짜'].notna()]
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': '유효한 날짜 데이터가 없습니다.'
                }
            
            # 최근 3개월간 구매한 고객들
            recent_customers = valid_sales[valid_sales['날짜'] >= three_months_ago]['고객명'].unique()
            
            # 전체 고객 중 최근 3개월간 구매하지 않은 고객들
            all_customers = valid_sales['고객명'].unique()
            churned_customers = [customer for customer in all_customers if customer not in recent_customers]
            
            # 이탈 고객들의 마지막 구매일과 총 구매 정보
            churned_details = []
            for customer in churned_customers:
                customer_data = valid_sales[valid_sales['고객명'] == customer]
                last_purchase = customer_data['날짜'].max()
                total_quantity = customer_data['수량'].sum()
                total_amount = customer_data['금액'].sum() if '금액' in customer_data.columns else 0
                
                churned_details.append({
                    '고객명': customer,
                    '마지막_구매일': last_purchase.strftime('%Y-%m-%d'),
                    '총_구매량': int(total_quantity),
                    '총_구매금액': int(total_amount),
                    '이탈_일수': (today - last_purchase).days
                })
            
            # 이탈 일수 기준으로 정렬
            churned_details.sort(key=lambda x: x['이탈_일수'], reverse=True)
            
            return {
                '상태': '성공',
                '오늘_날짜': today.strftime('%Y-%m-%d'),
                '기준_날짜': three_months_ago.strftime('%Y-%m-%d'),
                '이탈_업체수': len(churned_customers),
                '이탈_업체_목록': churned_details
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f'이탈 업체 분석 중 오류 발생: {str(e)}'
            }
    
    def analyze_claim_customers(self):
        """클레임 발생 업체 관리: 최근 3개월간 클레임 발생한 업체"""
        from datetime import datetime, timedelta
        
        if self.refund_data is None or self.refund_data.empty:
            return {
                '상태': '실패',
                '메시지': '반품(클레임) 데이터가 없습니다.'
            }
        
        today = datetime.now()
        three_months_ago = today - timedelta(days=90)
        
        try:
            # 반품 데이터의 날짜 컬럼을 datetime으로 변환
            refund_data_copy = self.refund_data.copy()
            refund_data_copy['날짜'] = pd.to_datetime(refund_data_copy['날짜'], errors='coerce')
            
            # 유효한 날짜만 필터링
            valid_refunds = refund_data_copy[refund_data_copy['날짜'].notna()]
            
            if valid_refunds.empty:
                return {
                    '상태': '실패',
                    '메시지': '유효한 반품 날짜 데이터가 없습니다.'
                }
            
            # 최근 3개월간 클레임 발생 업체
            recent_claims = valid_refunds[valid_refunds['날짜'] >= three_months_ago]
            
            if recent_claims.empty:
                return {
                    '상태': '성공',
                    '오늘_날짜': today.strftime('%Y-%m-%d'),
                    '기준_날짜': three_months_ago.strftime('%Y-%m-%d'),
                    '클레임_업체수': 0,
                    '클레임_업체_목록': []
                }
            
            # 업체별 클레임 정보 집계
            claim_details = []
            for customer in recent_claims['고객명'].unique():
                customer_claims = recent_claims[recent_claims['고객명'] == customer]
                
                claim_count = len(customer_claims)
                total_refund_quantity = customer_claims['수량'].sum()
                total_refund_amount = customer_claims['금액'].sum() if '금액' in customer_claims.columns else 0
                last_claim_date = customer_claims['날짜'].max()
                
                # 클레임 사유 (비고 컬럼이 있는 경우)
                claim_reasons = []
                if '비고' in customer_claims.columns:
                    reasons = customer_claims['비고'].dropna().unique()
                    claim_reasons = [reason for reason in reasons if reason and str(reason).strip()]
                
                claim_details.append({
                    '고객명': customer,
                    '클레임_횟수': claim_count,
                    '총_반품량': int(total_refund_quantity),
                    '총_반품금액': int(total_refund_amount),
                    '최근_클레임일': last_claim_date.strftime('%Y-%m-%d'),
                    '클레임_사유': ', '.join(claim_reasons) if claim_reasons else '사유 없음'
                })
            
            # 클레임 횟수 기준으로 정렬
            claim_details.sort(key=lambda x: x['클레임_횟수'], reverse=True)
            
            return {
                '상태': '성공',
                '오늘_날짜': today.strftime('%Y-%m-%d'),
                '기준_날짜': three_months_ago.strftime('%Y-%m-%d'),
                '클레임_업체수': len(claim_details),
                '클레임_업체_목록': claim_details
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f'클레임 업체 분석 중 오류 발생: {str(e)}'
            }
    
    def analyze_new_customers_2025(self):
        """신규 업체 관리: 2025년 기준 초도 구매 이루어진 업체"""
        try:
            # 날짜 컬럼을 datetime으로 변환
            sales_data_copy = self.sales_data.copy()
            sales_data_copy['날짜'] = pd.to_datetime(sales_data_copy['날짜'], errors='coerce')
            
            # 유효한 날짜만 필터링
            valid_sales = sales_data_copy[sales_data_copy['날짜'].notna()]
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': '유효한 날짜 데이터가 없습니다.'
                }
            
            # 각 고객의 첫 구매일 찾기
            customer_first_purchase = valid_sales.groupby('고객명')['날짜'].min().reset_index()
            customer_first_purchase.columns = ['고객명', '첫_구매일']
            
            # 2025년에 첫 구매한 고객들 필터링
            new_customers_2025 = customer_first_purchase[
                customer_first_purchase['첫_구매일'].dt.year == 2025
            ]
            
            if new_customers_2025.empty:
                return {
                    '상태': '성공',
                    '신규_업체수': 0,
                    '신규_업체_목록': []
                }
            
            # 신규 고객들의 상세 정보
            new_customer_details = []
            for _, row in new_customers_2025.iterrows():
                customer_name = row['고객명']
                first_purchase_date = row['첫_구매일']
                
                # 해당 고객의 총 구매 정보
                customer_data = valid_sales[valid_sales['고객명'] == customer_name]
                total_quantity = customer_data['수량'].sum()
                total_amount = customer_data['금액'].sum() if '금액' in customer_data.columns else 0
                purchase_count = len(customer_data)
                unique_products = customer_data['상품'].nunique()
                
                # 최근 구매일
                last_purchase_date = customer_data['날짜'].max()
                
                new_customer_details.append({
                    '고객명': customer_name,
                    '첫_구매일': first_purchase_date.strftime('%Y-%m-%d'),
                    '최근_구매일': last_purchase_date.strftime('%Y-%m-%d'),
                    '총_구매량': int(total_quantity),
                    '총_구매금액': int(total_amount),
                    '구매_횟수': purchase_count,
                    '구매_상품수': unique_products
                })
            
            # 첫 구매일 기준으로 정렬 (최신순)
            new_customer_details.sort(key=lambda x: x['첫_구매일'], reverse=True)
            
            return {
                '상태': '성공',
                '신규_업체수': len(new_customer_details),
                '신규_업체_목록': new_customer_details
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f'신규 업체 분석 중 오류 발생: {str(e)}'
            }

    def analyze_customer_characteristics(self):
        """업체 특성 분석"""
        try:
            # 고객별 기본 통계 계산
            customer_stats = self.sales_data.groupby('고객명').agg({
                '수량': 'sum',
                '금액': 'sum',
                '상품': 'nunique',
                '날짜': ['count', 'min', 'max']
            }).round(2)
            
            customer_stats.columns = ['총구매량', '총매출', '구매품목수', '거래횟수', '첫구매일', '최근구매일']
            customer_stats = customer_stats.reset_index()
            
            # 날짜 처리
            customer_stats['첫구매일'] = pd.to_datetime(customer_stats['첫구매일'])
            customer_stats['최근구매일'] = pd.to_datetime(customer_stats['최근구매일'])
            customer_stats['거래기간'] = (customer_stats['최근구매일'] - customer_stats['첫구매일']).dt.days + 1
            
            # 평균 주문 금액
            customer_stats['평균주문금액'] = (customer_stats['총매출'] / customer_stats['거래횟수']).round(0)
            
            # 구매 빈도 (일 단위)
            customer_stats['구매빈도'] = (customer_stats['거래기간'] / customer_stats['거래횟수']).round(1)
            
            # 구매 패턴 분류
            def classify_purchase_pattern(row):
                if row['구매빈도'] <= 7:  # 주 1회 이상
                    return '정기구매형'
                elif row['구매빈도'] <= 30:  # 월 1회 이상
                    return '일반구매형'
                else:
                    return '비정기구매형'
            
            customer_stats['구매패턴'] = customer_stats.apply(classify_purchase_pattern, axis=1)
            
            # 품목 다양성 분류
            def classify_product_diversity(row):
                if row['구매품목수'] >= 50:
                    return '고다양성'
                elif row['구매품목수'] >= 20:
                    return '중다양성'
                elif row['구매품목수'] >= 10:
                    return '저다양성'
                else:
                    return '단일품목형'
            
            customer_stats['품목다양성'] = customer_stats.apply(classify_product_diversity, axis=1)
            
            # 고객 활성도 점수 (0-100)
            max_revenue = customer_stats['총매출'].max()
            max_frequency = customer_stats['거래횟수'].max()
            max_diversity = customer_stats['구매품목수'].max()
            
            customer_stats['활성도점수'] = (
                (customer_stats['총매출'] / max_revenue * 40) +
                (customer_stats['거래횟수'] / max_frequency * 30) +
                (customer_stats['구매품목수'] / max_diversity * 30)
            ).round(1)
            
            # 고객 등급 분류
            def classify_customer_grade(score):
                if score >= 80:
                    return 'VIP'
                elif score >= 60:
                    return 'Gold'
                elif score >= 40:
                    return 'Silver'
                else:
                    return 'Bronze'
            
            customer_stats['고객등급'] = customer_stats['활성도점수'].apply(classify_customer_grade)
            
            return {
                '상태': '성공',
                '고객특성데이터': customer_stats,
                '구매패턴분포': customer_stats['구매패턴'].value_counts(),
                '품목다양성분포': customer_stats['품목다양성'].value_counts(),
                '고객등급분포': customer_stats['고객등급'].value_counts()
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f'업체 특성 분석 중 오류가 발생했습니다: {str(e)}'
            }
    
    def analyze_sales_trends_advanced(self):
        """고급 매출 트렌드 분석"""
        try:
            # 날짜 컬럼 처리
            sales_data = self.sales_data.copy()
            sales_data['날짜'] = pd.to_datetime(sales_data['날짜'])
            sales_data['연도'] = sales_data['날짜'].dt.year
            sales_data['월'] = sales_data['날짜'].dt.month
            sales_data['요일'] = sales_data['날짜'].dt.day_name()
            sales_data['주차'] = sales_data['날짜'].dt.isocalendar().week
            
            # 월별 매출 트렌드
            monthly_trend = sales_data.groupby(['연도', '월']).agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            monthly_trend['연월'] = monthly_trend['연도'].astype(str) + '-' + monthly_trend['월'].astype(str).str.zfill(2)
            
            # 요일별 매출 패턴
            weekday_pattern = sales_data.groupby('요일').agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            
            # 요일 순서 정렬
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_pattern['요일'] = pd.Categorical(weekday_pattern['요일'], categories=weekday_order, ordered=True)
            weekday_pattern = weekday_pattern.sort_values('요일').reset_index(drop=True)
            
            # 상품 카테고리별 매출 (상품명 기반 간단 분류)
            def categorize_product(product_name):
                product_name = str(product_name).lower()
                if any(keyword in product_name for keyword in ['샐러드', '채소', '야채', '상추', '케일']):
                    return '채소류'
                elif any(keyword in product_name for keyword in ['허브', 'herb', '바질', '로즈마리']):
                    return '허브류'
                elif any(keyword in product_name for keyword in ['마이크로그린', 'microgreen']):
                    return '마이크로그린'
                elif any(keyword in product_name for keyword in ['꽃', 'flower', '에디블']):
                    return '에디블플라워'
                else:
                    return '기타'
            
            sales_data['상품카테고리'] = sales_data['상품'].apply(categorize_product)
            category_sales = sales_data.groupby('상품카테고리').agg({
                '금액': 'sum',
                '수량': 'sum'
            }).reset_index()
            
            # 고객 유형별 매출 (고객명 기반 분류)
            def categorize_customer(customer_name):
                customer_name = str(customer_name).lower()
                if any(keyword in customer_name for keyword in ['호텔', 'hotel']):
                    return '호텔'
                elif any(keyword in customer_name for keyword in ['레스토랑', '식당', 'restaurant']):
                    return '레스토랑'
                elif any(keyword in customer_name for keyword in ['카페', 'cafe', 'coffee']):
                    return '카페'
                elif any(keyword in customer_name for keyword in ['마트', '마켓', 'market']):
                    return '마트/마켓'
                else:
                    return '기타'
            
            sales_data['고객유형'] = sales_data['고객명'].apply(categorize_customer)
            customer_type_sales = sales_data.groupby('고객유형').agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            
            # 매출 성장률 계산
            monthly_growth = monthly_trend.copy()
            monthly_growth['매출성장률'] = monthly_growth['금액'].pct_change() * 100
            monthly_growth['수량성장률'] = monthly_growth['수량'].pct_change() * 100
            
            return {
                '상태': '성공',
                '월별트렌드': monthly_trend,
                '요일별패턴': weekday_pattern,
                '상품카테고리별매출': category_sales,
                '고객유형별매출': customer_type_sales,
                '매출성장률': monthly_growth
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f'고급 매출 트렌드 분석 중 오류가 발생했습니다: {str(e)}'
            }
    
    def get_all_customers_for_analysis(self):
        """분석 가능한 모든 고객 목록 반환"""
        try:
            customers = self.sales_data['고객명'].unique().tolist()
            return sorted(customers)
        except:
            return []

    def analyze_michelin_vs_non_michelin(self):
        """미슐랭 등급별 vs 비미슐랭 업장 특징 비교 분석"""
        try:
            classified_customers = self.classify_michelin_customers()
            
            # 모든 미슐랭 고객 리스트
            all_michelin_customers = []
            for customers in classified_customers.values():
                all_michelin_customers.extend(customers)
            
            # 비미슐랭 고객 식별
            all_customers = self.sales_data['고객명'].unique()
            non_michelin_customers = [customer for customer in all_customers 
                                    if customer not in all_michelin_customers]
            
            # 비미슐랭 데이터 분석
            non_michelin_data = self.sales_data[self.sales_data['고객명'].isin(non_michelin_customers)]
            
            if non_michelin_data.empty:
                return {
                    '상태': '실패',
                    '메시지': '비미슐랭 데이터를 찾을 수 없습니다.'
                }
            
            # 데이터 정리
            non_michelin_clean = non_michelin_data.copy()
            non_michelin_clean['금액'] = pd.to_numeric(non_michelin_clean['금액'], errors='coerce')
            non_michelin_clean['수량'] = pd.to_numeric(non_michelin_clean['수량'], errors='coerce')
            non_michelin_clean = non_michelin_clean.dropna(subset=['금액', '수량'])
            
            # 비미슐랭 기본 지표 계산
            non_michelin_stats = {
                '총_매출': non_michelin_clean['금액'].sum(),
                '총_구매량': non_michelin_clean['수량'].sum(),
                '평균_주문금액': non_michelin_clean['금액'].mean(),
                '업장당_평균매출': non_michelin_clean['금액'].sum() / len(non_michelin_customers) if len(non_michelin_customers) > 0 else 0,
                '품목_다양성': non_michelin_clean['상품'].nunique(),
                '업장_수': len(non_michelin_customers),
                '평균_거래횟수': len(non_michelin_clean) / len(non_michelin_customers) if len(non_michelin_customers) > 0 else 0,
                '단위당_평균가격': non_michelin_clean['금액'].sum() / non_michelin_clean['수량'].sum() if non_michelin_clean['수량'].sum() > 0 else 0
            }
            
            # 비미슐랭 인기 품목
            non_michelin_products = non_michelin_clean.groupby('상품').agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).reset_index()
            non_michelin_products = non_michelin_products.sort_values('금액', ascending=False)
            non_michelin_top_products = non_michelin_products.head(20)['상품'].tolist()
            
            # 비미슐랭 계절별/분기별 선호도 분석
            non_michelin_seasonal = {'봄': 0, '여름': 0, '가을': 0, '겨울': 0}
            non_michelin_quarterly = {'1분기': 0, '2분기': 0, '3분기': 0, '4분기': 0}
            
            if 'month' in non_michelin_clean.columns:
                for month, group in non_michelin_clean.groupby('month'):
                    if pd.isna(month):
                        continue
                    month = int(month)
                    quantity = group['수량'].sum()
                    
                    # 계절별
                    if month in [3, 4, 5]:
                        non_michelin_seasonal['봄'] += quantity
                    elif month in [6, 7, 8]:
                        non_michelin_seasonal['여름'] += quantity
                    elif month in [9, 10, 11]:
                        non_michelin_seasonal['가을'] += quantity
                    else:
                        non_michelin_seasonal['겨울'] += quantity
                    
                    # 분기별
                    if month in [1, 2, 3]:
                        non_michelin_quarterly['1분기'] += quantity
                    elif month in [4, 5, 6]:
                        non_michelin_quarterly['2분기'] += quantity
                    elif month in [7, 8, 9]:
                        non_michelin_quarterly['3분기'] += quantity
                    else:
                        non_michelin_quarterly['4분기'] += quantity
            
            # 전체 미슐랭 통합 분석 추가
            all_michelin_data = self.sales_data[self.sales_data['고객명'].isin(all_michelin_customers)]
            
            if not all_michelin_data.empty:
                # 데이터 정리
                all_michelin_clean = all_michelin_data.copy()
                all_michelin_clean['금액'] = pd.to_numeric(all_michelin_clean['금액'], errors='coerce')
                all_michelin_clean['수량'] = pd.to_numeric(all_michelin_clean['수량'], errors='coerce')
                all_michelin_clean = all_michelin_clean.dropna(subset=['금액', '수량'])
                
                # 전체 미슐랭 기본 지표
                all_michelin_stats = {
                    '총_매출': all_michelin_clean['금액'].sum(),
                    '총_구매량': all_michelin_clean['수량'].sum(),
                    '평균_주문금액': all_michelin_clean['금액'].mean(),
                    '업장당_평균매출': all_michelin_clean['금액'].sum() / len(all_michelin_customers) if len(all_michelin_customers) > 0 else 0,
                    '품목_다양성': all_michelin_clean['상품'].nunique(),
                    '업장_수': len(all_michelin_customers),
                    '평균_거래횟수': len(all_michelin_clean) / len(all_michelin_customers) if len(all_michelin_customers) > 0 else 0,
                    '단위당_평균가격': all_michelin_clean['금액'].sum() / all_michelin_clean['수량'].sum() if all_michelin_clean['수량'].sum() > 0 else 0
                }
                
                # 전체 미슐랭 인기 품목
                all_michelin_products = all_michelin_clean.groupby('상품').agg({
                    '금액': 'sum',
                    '수량': 'sum',
                    '고객명': 'nunique'
                }).reset_index()
                all_michelin_products = all_michelin_products.sort_values('금액', ascending=False)
                all_michelin_top_products = all_michelin_products.head(20)['상품'].tolist()
                
                # 전체 미슐랭 계절별/분기별 선호도 분석
                all_michelin_seasonal = {'봄': 0, '여름': 0, '가을': 0, '겨울': 0}
                all_michelin_quarterly = {'1분기': 0, '2분기': 0, '3분기': 0, '4분기': 0}
                
                if 'month' in all_michelin_clean.columns:
                    for month, group in all_michelin_clean.groupby('month'):
                        if pd.isna(month):
                            continue
                        month = int(month)
                        quantity = group['수량'].sum()
                        
                        # 계절별
                        if month in [3, 4, 5]:
                            all_michelin_seasonal['봄'] += quantity
                        elif month in [6, 7, 8]:
                            all_michelin_seasonal['여름'] += quantity
                        elif month in [9, 10, 11]:
                            all_michelin_seasonal['가을'] += quantity
                        else:
                            all_michelin_seasonal['겨울'] += quantity
                        
                        # 분기별
                        if month in [1, 2, 3]:
                            all_michelin_quarterly['1분기'] += quantity
                        elif month in [4, 5, 6]:
                            all_michelin_quarterly['2분기'] += quantity
                        elif month in [7, 8, 9]:
                            all_michelin_quarterly['3분기'] += quantity
                        else:
                            all_michelin_quarterly['4분기'] += quantity
                
                # 전체 미슐랭 vs 비미슐랭 비교
                all_michelin_comparison = {
                    '평균_주문금액_배수': all_michelin_stats['평균_주문금액'] / non_michelin_stats['평균_주문금액'] if non_michelin_stats['평균_주문금액'] > 0 else 0,
                    '업장당_매출_배수': all_michelin_stats['업장당_평균매출'] / non_michelin_stats['업장당_평균매출'] if non_michelin_stats['업장당_평균매출'] > 0 else 0,
                    '거래횟수_배수': all_michelin_stats['평균_거래횟수'] / non_michelin_stats['평균_거래횟수'] if non_michelin_stats['평균_거래횟수'] > 0 else 0,
                    '단위가격_배수': all_michelin_stats['단위당_평균가격'] / non_michelin_stats['단위당_평균가격'] if non_michelin_stats['단위당_평균가격'] > 0 else 0,
                }
                
                # 품목 차이 분석
                michelin_unique_products = [p for p in all_michelin_top_products if p not in non_michelin_top_products]
                common_products = [p for p in all_michelin_top_products if p in non_michelin_top_products]
                non_michelin_unique_products = [p for p in non_michelin_top_products if p not in all_michelin_top_products]
            
            # 각 미슐랭 등급별 vs 비미슐랭 비교
            comparison_results = {}
            
            for grade in ['3_STAR', '2_STAR', '1_STAR', 'SELECTED']:
                if grade in classified_customers:
                    customers = classified_customers[grade]
                    grade_data = self.sales_data[self.sales_data['고객명'].isin(customers)]
                    
                    if not grade_data.empty:
                        # 데이터 정리
                        grade_data_clean = grade_data.copy()
                        grade_data_clean['금액'] = pd.to_numeric(grade_data_clean['금액'], errors='coerce')
                        grade_data_clean['수량'] = pd.to_numeric(grade_data_clean['수량'], errors='coerce')
                        grade_data_clean = grade_data_clean.dropna(subset=['금액', '수량'])
                        
                        # 미슐랭 등급 기본 지표
                        michelin_stats = {
                            '총_매출': grade_data_clean['금액'].sum(),
                            '총_구매량': grade_data_clean['수량'].sum(),
                            '평균_주문금액': grade_data_clean['금액'].mean(),
                            '업장당_평균매출': grade_data_clean['금액'].sum() / len(customers) if len(customers) > 0 else 0,
                            '품목_다양성': grade_data_clean['상품'].nunique(),
                            '업장_수': len(customers),
                            '평균_거래횟수': len(grade_data_clean) / len(customers) if len(customers) > 0 else 0,
                            '단위당_평균가격': grade_data_clean['금액'].sum() / grade_data_clean['수량'].sum() if grade_data_clean['수량'].sum() > 0 else 0
                        }
                        
                        # 인기 품목
                        michelin_products = grade_data_clean.groupby('상품').agg({
                            '금액': 'sum',
                            '수량': 'sum',
                            '고객명': 'nunique'
                        }).reset_index()
                        michelin_products = michelin_products.sort_values('금액', ascending=False)
                        michelin_top_products = michelin_products.head(10)['상품'].tolist()
                        
                        # 특징 비교 분석
                        comparison_analysis = {
                            '평균_주문금액_배수': michelin_stats['평균_주문금액'] / non_michelin_stats['평균_주문금액'] if non_michelin_stats['평균_주문금액'] > 0 else 0,
                            '업장당_매출_배수': michelin_stats['업장당_평균매출'] / non_michelin_stats['업장당_평균매출'] if non_michelin_stats['업장당_평균매출'] > 0 else 0,
                            '거래횟수_배수': michelin_stats['평균_거래횟수'] / non_michelin_stats['평균_거래횟수'] if non_michelin_stats['평균_거래횟수'] > 0 else 0,
                            '단위가격_배수': michelin_stats['단위당_평균가격'] / non_michelin_stats['단위당_평균가격'] if non_michelin_stats['단위당_평균가격'] > 0 else 0,
                        }
                        
                        # 차별화된 특징 식별
                        unique_features = []
                        
                        if comparison_analysis['평균_주문금액_배수'] > 1.5:
                            unique_features.append(f"주문금액이 비미슐랭 대비 {comparison_analysis['평균_주문금액_배수']:.1f}배 높음")
                        elif comparison_analysis['평균_주문금액_배수'] < 0.7:
                            unique_features.append(f"주문금액이 비미슐랭 대비 {comparison_analysis['평균_주문금액_배수']:.1f}배 낮음")
                            
                        if comparison_analysis['업장당_매출_배수'] > 2.0:
                            unique_features.append(f"업장당 매출이 비미슐랭 대비 {comparison_analysis['업장당_매출_배수']:.1f}배 높음")
                        elif comparison_analysis['업장당_매출_배수'] < 0.5:
                            unique_features.append(f"업장당 매출이 비미슐랭 대비 {comparison_analysis['업장당_매출_배수']:.1f}배 낮음")
                            
                        if comparison_analysis['거래횟수_배수'] > 1.3:
                            unique_features.append(f"거래빈도가 비미슐랭 대비 {comparison_analysis['거래횟수_배수']:.1f}배 높음")
                        elif comparison_analysis['거래횟수_배수'] < 0.8:
                            unique_features.append(f"거래빈도가 비미슐랭 대비 {comparison_analysis['거래횟수_배수']:.1f}배 낮음")
                            
                        if comparison_analysis['단위가격_배수'] > 1.2:
                            unique_features.append(f"고가제품 선호 (단위가격 {comparison_analysis['단위가격_배수']:.1f}배)")
                        elif comparison_analysis['단위가격_배수'] < 0.8:
                            unique_features.append(f"저가제품 선호 (단위가격 {comparison_analysis['단위가격_배수']:.1f}배)")
                        
                        # 품목 차이 분석
                        unique_products = []
                        common_products_grade = []
                        
                        for product in michelin_top_products:
                            if product in non_michelin_top_products:
                                common_products_grade.append(product)
                            else:
                                unique_products.append(product)
                        
                        if unique_products:
                            unique_features.append(f"독특한 선호품목: {', '.join(unique_products[:3])}")
                        
                        comparison_results[grade] = {
                            '미슐랭_지표': michelin_stats,
                            '비교_배수': comparison_analysis,
                            '차별화_특징': unique_features,
                            '독특한_품목': unique_products,
                            '공통_품목': common_products_grade,
                            '인기_품목_TOP5': michelin_top_products[:5]
                        }
            
            return {
                '상태': '성공',
                '비미슐랭_기준지표': non_michelin_stats,
                '비미슐랭_인기품목': non_michelin_top_products,
                '비미슐랭_계절별_선호도': non_michelin_seasonal,
                '비미슐랭_분기별_선호도': non_michelin_quarterly,
                '전체_미슐랭_지표': all_michelin_stats if 'all_michelin_stats' in locals() else {},
                '전체_미슐랭_인기품목': all_michelin_top_products if 'all_michelin_top_products' in locals() else [],
                '전체_미슐랭_계절별_선호도': all_michelin_seasonal if 'all_michelin_seasonal' in locals() else {},
                '전체_미슐랭_분기별_선호도': all_michelin_quarterly if 'all_michelin_quarterly' in locals() else {},
                '전체_미슐랭_비교': all_michelin_comparison if 'all_michelin_comparison' in locals() else {},
                '미슐랭_독특한_품목': michelin_unique_products if 'michelin_unique_products' in locals() else [],
                '공통_품목': common_products if 'common_products' in locals() else [],
                '비미슐랭_독특한_품목': non_michelin_unique_products if 'non_michelin_unique_products' in locals() else [],
                '등급별_비교': comparison_results
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"미슐랭 vs 비미슐랭 비교 분석 중 오류 발생: {str(e)}"
            }
    
    def analyze_michelin_comparison(self):
        """미슐랭 등급간 비교 분석"""
        try:
            classified_customers = self.classify_michelin_customers()
            comparison_data = {}
            
            for grade in ['3_STAR', '2_STAR', '1_STAR', 'SELECTED']:
                if grade in classified_customers:
                    customers = classified_customers[grade]
                    grade_data = self.sales_data[self.sales_data['고객명'].isin(customers)]
                    
                    if not grade_data.empty:
                        # 데이터 정리
                        grade_data_clean = grade_data.copy()
                        grade_data_clean['금액'] = pd.to_numeric(grade_data_clean['금액'], errors='coerce')
                        grade_data_clean['수량'] = pd.to_numeric(grade_data_clean['수량'], errors='coerce')
                        grade_data_clean = grade_data_clean.dropna(subset=['금액', '수량'])
                        
                        comparison_data[grade] = {
                            '총_매출': grade_data_clean['금액'].sum(),
                            '총_구매량': grade_data_clean['수량'].sum(),
                            '평균_주문금액': grade_data_clean['금액'].mean(),
                            '레스토랑당_평균매출': grade_data_clean['금액'].sum() / len(customers) if len(customers) > 0 else 0,
                            '품목_다양성': grade_data_clean['상품'].nunique(),
                            '레스토랑_수': len(customers)
                        }
            
            return {
                '상태': '성공',
                '비교_데이터': comparison_data
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"미슐랭 비교 분석 중 오류 발생: {str(e)}"
            }
    
    def get_bakery_restaurants(self):
        """베이커리 & 디저트 레스토랑 목록 반환"""
        bakery_keywords = [
            "파리크라상 Passion5", "파리크라상 도곡점", "파리크라상(양재연구실)", "파리크라상 신세계백화점본점", "터치", "라뜰리에 이은", "노틀던", "파티세리 폰드", 
            "앨리스 프로젝트", "카페꼼마", "문화시민 서울", "소나(SONA)",
            "사색연희", "알디프", "클레어파티시에", "슬로운", "바 오쁘띠베르"
        ]
        
        all_customers = self.sales_data['고객명'].unique()
        bakery_customers = []
        
        for customer in all_customers:
            for keyword in bakery_keywords:
                if keyword in str(customer):
                    bakery_customers.append(customer)
                    break
        
        return bakery_customers
    
    def classify_bakery_customers(self):
        """베이커리 & 디저트 레스토랑 분류"""
        try:
            bakery_customers = self.get_bakery_restaurants()
            
            # 키워드별 분류
            classified = {}
            bakery_keywords = [
                "파리크라상 Passion5", "파리크라상 도곡점", "파리크라상(양재연구실)", "파리크라상 신세계백화점본점", "터치", "라뜰리에 이은", "노틀던", "파티세리 폰드", 
                "앨리스 프로젝트", "카페꼼마", "문화시민 서울", "소나(SONA)",
                "사색연희", "알디프", "클레어파티시에", "슬로운", "바 오쁘띠베르"
            ]
            
            all_customers = self.sales_data['고객명'].unique()
            
            for keyword in bakery_keywords:
                classified[keyword] = []
                for customer in all_customers:
                    if keyword in str(customer):
                        classified[keyword].append(customer)
            
            return classified
            
        except Exception as e:
            print(f"베이커리 분류 중 오류: {e}")
            return {}
    
    def analyze_bakery_overview(self):
        """베이커리 & 디저트 전체 현황 분석"""
        try:
            bakery_customers = self.get_bakery_restaurants()
            
            if not bakery_customers:
                return {
                    '상태': '실패',
                    '메시지': '베이커리 & 디저트 레스토랑을 찾을 수 없습니다.'
                }
            
            # 베이커리 데이터 필터링
            bakery_data = self.sales_data[self.sales_data['고객명'].isin(bakery_customers)].copy()
            
            if bakery_data.empty:
                return {
                    '상태': '실패',
                    '메시지': '베이커리 & 디저트 레스토랑 데이터를 찾을 수 없습니다.'
                }
            
            # 데이터 정리
            bakery_data['금액'] = pd.to_numeric(bakery_data['금액'], errors='coerce')
            bakery_data['수량'] = pd.to_numeric(bakery_data['수량'], errors='coerce')
            bakery_data = bakery_data.dropna(subset=['금액', '수량'])
            
            # 기본 통계
            total_sales = bakery_data['금액'].sum()
            total_quantity = bakery_data['수량'].sum()
            avg_order = bakery_data['금액'].mean()
            customer_count = len(bakery_customers)
            
            # 월별 매출
            if '날짜' in bakery_data.columns:
                bakery_data['날짜'] = pd.to_datetime(bakery_data['날짜'], errors='coerce')
                bakery_data['연월'] = bakery_data['날짜'].dt.to_period('M')
                monthly_sales = bakery_data.groupby('연월')['금액'].sum()
            else:
                monthly_sales = pd.Series(dtype='float64')
            
            # 업체별 매출
            customer_sales = bakery_data.groupby('고객명').agg({
                '금액': 'sum',
                '수량': 'sum',
                '상품': 'nunique'
            }).round(2)
            
            # 상품별 매출
            product_sales = bakery_data.groupby('상품').agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).round(2)
            
            return {
                '상태': '성공',
                '총매출': total_sales,
                '총구매량': total_quantity,
                '평균주문금액': avg_order,
                '업체수': customer_count,
                '월별매출': monthly_sales.to_dict() if not monthly_sales.empty else {},
                '업체별매출': customer_sales.to_dict('index'),
                '상품별매출': product_sales.to_dict('index'),
                '베이커리_업체목록': bakery_customers
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"베이커리 분석 중 오류 발생: {str(e)}"
            }
    
    def analyze_bakery_by_store(self, store_keyword):
        """특정 베이커리 업체 상세 분석"""
        try:
            # 해당 키워드를 포함한 업체들 찾기
            all_customers = self.sales_data['고객명'].unique()
            matching_customers = [customer for customer in all_customers if store_keyword in str(customer)]
            
            if not matching_customers:
                return {
                    '상태': '실패',
                    '메시지': f'{store_keyword}에 해당하는 업체를 찾을 수 없습니다.'
                }
            
            # 해당 업체들의 데이터 필터링
            store_data = self.sales_data[self.sales_data['고객명'].isin(matching_customers)].copy()
            
            if store_data.empty:
                return {
                    '상태': '실패',
                    '메시지': f'{store_keyword} 업체의 데이터를 찾을 수 없습니다.'
                }
            
            # 데이터 정리
            store_data['금액'] = pd.to_numeric(store_data['금액'], errors='coerce')
            store_data['수량'] = pd.to_numeric(store_data['수량'], errors='coerce')
            store_data = store_data.dropna(subset=['금액', '수량'])
            
            # 기본 통계
            total_sales = store_data['금액'].sum()
            total_quantity = store_data['수량'].sum()
            avg_order = store_data['금액'].mean()
            
            # 지점별 분석
            branch_analysis = store_data.groupby('고객명').agg({
                '금액': ['sum', 'mean', 'count'],
                '수량': 'sum',
                '상품': 'nunique'
            }).round(2)
            
            # 월별 매출 추이
            if '날짜' in store_data.columns:
                store_data['날짜'] = pd.to_datetime(store_data['날짜'], errors='coerce')
                store_data['연월'] = store_data['날짜'].dt.to_period('M')
                # 지점별 월별 추이
                branch_monthly_trend = store_data.groupby(['고객명', '연월']).agg({
                    '금액': 'sum',
                    '수량': 'sum'
                }).reset_index()
                
                monthly_trend = store_data.groupby('연월').agg({
                    '금액': 'sum',
                    '수량': 'sum'
                })
            else:
                monthly_trend = pd.DataFrame()
                branch_monthly_trend = pd.DataFrame()
            
            # 상품별 분석
            product_analysis = store_data.groupby('상품').agg({
                '금액': 'sum',
                '수량': 'sum',
                '고객명': 'nunique'
            }).sort_values('금액', ascending=False)
            
            # 상위 상품 (매출 기준)
            top_products = product_analysis.head(10)
            
            return {
                '상태': '성공',
                '업체명': store_keyword,
                '매칭_업체들': matching_customers,
                '총매출': total_sales,
                '총구매량': total_quantity,
                '평균주문금액': avg_order,
                '지점별분석': branch_analysis.to_dict('index'),
                '월별추이': monthly_trend.to_dict('index') if not monthly_trend.empty else {}, '지점별월별추이': branch_monthly_trend.to_dict('records') if not branch_monthly_trend.empty else [],
                '상품별분석': product_analysis.to_dict('index'),
                '상위상품': top_products.to_dict('index')
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"{store_keyword} 분석 중 오류 발생: {str(e)}"
            }
    
    def analyze_bakery_comparison(self):
        """베이커리 업체간 비교 분석"""
        try:
            classified_customers = self.classify_bakery_customers()
            comparison_data = {}
            
            bakery_keywords = [
                "파리크라상 Passion5", "파리크라상 도곡점", "파리크라상(양재연구실)", "파리크라상 신세계백화점본점", "터치", "라뜰리에 이은", "노틀던", "파티세리 폰드", 
                "앨리스 프로젝트", "카페꼼마", "문화시민 서울", "소나(SONA)",
                "사색연희", "알디프", "클레어파티시에", "슬로운", "바 오쁘띠베르"
            ]
            
            for keyword in bakery_keywords:
                if keyword in classified_customers and classified_customers[keyword]:
                    customers = classified_customers[keyword]
                    keyword_data = self.sales_data[self.sales_data['고객명'].isin(customers)]
                    
                    if not keyword_data.empty:
                        # 데이터 정리
                        keyword_data_clean = keyword_data.copy()
                        keyword_data_clean['금액'] = pd.to_numeric(keyword_data_clean['금액'], errors='coerce')
                        keyword_data_clean['수량'] = pd.to_numeric(keyword_data_clean['수량'], errors='coerce')
                        keyword_data_clean = keyword_data_clean.dropna(subset=['금액', '수량'])
                        
                        comparison_data[keyword] = {
                            '총_매출': keyword_data_clean['금액'].sum(),
                            '총_구매량': keyword_data_clean['수량'].sum(),
                            '평균_주문금액': keyword_data_clean['금액'].mean(),
                            '지점당_평균매출': keyword_data_clean['금액'].sum() / len(customers) if len(customers) > 0 else 0,
                            '품목_다양성': keyword_data_clean['상품'].nunique(),
                            '지점_수': len(customers),
                            '지점_목록': customers
                        }
            
            return {
                '상태': '성공',
                '비교_데이터': comparison_data
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"베이커리 비교 분석 중 오류 발생: {str(e)}"
            }

    def analyze_product_performance_heatmap(self, end_date):
        """종료날짜 기준으로 해당 월과 전월 비교 히트맵 데이터 생성"""
        try:
            if '날짜' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': '날짜 정보가 없습니다.'
                }
            
            # 종료 날짜를 기준으로 해당 월과 전월 계산
            end_date = pd.to_datetime(end_date)
            
            # 종료날짜의 월 (current_month)
            current_month_start = end_date.replace(day=1)
            current_month_end = end_date  # 종료날짜까지만
            
            # 전월 (previous_month)
            prev_month_end = current_month_start - pd.Timedelta(days=1)
            prev_month_start = prev_month_end.replace(day=1)
            
            # 날짜가 있는 데이터만 필터링
            sales_with_date = self.sales_data[self.sales_data['날짜'].notna()].copy()
            
            if sales_with_date.empty:
                return {
                    '상태': '실패',
                    '메시지': '유효한 날짜 데이터가 없습니다.'
                }
            
            # 해당 월 데이터 (종료날짜까지)
            current_month_data = sales_with_date[
                (sales_with_date['날짜'] >= current_month_start) & 
                (sales_with_date['날짜'] <= current_month_end)
            ]
            
            # 전월 데이터 (전체 월)
            prev_month_data = sales_with_date[
                (sales_with_date['날짜'] >= prev_month_start) & 
                (sales_with_date['날짜'] <= prev_month_end)
            ]
            
            # 상품별 현재월 판매량
            current_sales = current_month_data.groupby('상품')['수량'].sum()
            
            # 상품별 전월 판매량
            prev_sales = prev_month_data.groupby('상품')['수량'].sum()
            
            # 모든 상품 목록
            all_products = set(current_sales.index) | set(prev_sales.index)
            
            heatmap_data = []
            
            for product in all_products:
                current_qty = current_sales.get(product, 0)
                prev_qty = prev_sales.get(product, 0)
                
                # 변화율 계산
                if prev_qty > 0:
                    change_rate = ((current_qty - prev_qty) / prev_qty) * 100
                elif current_qty > 0:
                    change_rate = 100  # 신규 상품
                else:
                    change_rate = 0
                
                # 총 판매량 (크기 결정용)
                total_qty = current_qty + prev_qty
                
                if total_qty > 0:  # 판매량이 있는 상품만 포함
                    heatmap_data.append({
                        '상품': product,
                        '현재월_판매량': current_qty,
                        '전월_판매량': prev_qty,
                        '변화율': change_rate,
                        '총_판매량': total_qty,
                        '크기': total_qty  # 히트맵 크기용
                    })
            
            # 데이터프레임 생성
            heatmap_df = pd.DataFrame(heatmap_data)
            
            if heatmap_df.empty:
                return {
                    '상태': '실패',
                    '메시지': '분석할 상품 데이터가 없습니다.'
                }
            
            # 크기 정규화 (20-100 범위)
            if heatmap_df['크기'].max() > heatmap_df['크기'].min():
                heatmap_df['정규화_크기'] = 20 + (heatmap_df['크기'] - heatmap_df['크기'].min()) / (heatmap_df['크기'].max() - heatmap_df['크기'].min()) * 80
            else:
                heatmap_df['정규화_크기'] = 50
            
            return {
                '상태': '성공',
                '히트맵_데이터': heatmap_df,
                '현재월': current_month_start.strftime('%Y-%m'),
                '전월': prev_month_start.strftime('%Y-%m'),
                '분석_기준일': end_date.strftime('%Y-%m-%d'),
                '분석_설명': f"{end_date.strftime('%Y-%m-%d')}를 기준으로 {current_month_start.strftime('%Y-%m')}월과 {prev_month_start.strftime('%Y-%m')}월 비교"
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"히트맵 분석 중 오류 발생: {str(e)}"
            }

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">📊 마이크로그린 관리자 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">상품 분석, 고객 분석, RFM 세분화를 통한 비즈니스 인사이트</p>', unsafe_allow_html=True)
    
    # 현재 디렉토리의 파일들을 확인
    sales_file = "merged_with_remarks_final.xlsx"
    refund_file = "merged_returns_2024_2025.xlsx"
    
    # 데이터 로드 시도
    try:
        # 판매 데이터 로드
        sales_data = pd.read_excel(sales_file)
        st.sidebar.success(f"판매 데이터 로드 완료: {len(sales_data)}개 레코드")
        
        # 반품 데이터 로드
        refund_data = pd.read_excel(refund_file)
        st.sidebar.success(f"반품 데이터 로드 완료: {len(refund_data)}개 레코드")
        
        # 컬럼명 공백 제거
        sales_data.columns = sales_data.columns.str.strip()
        refund_data.columns = refund_data.columns.str.strip()
        
        # 판매 데이터 컬럼 매핑
        sales_data = sales_data.rename(columns={
            '거래처': '고객명',
            '월일': '날짜',
            '품목': '상품',
            '합계': '금액',  # '합계' 컬럼을 '금액'으로 매핑
        })
        
        # 반품 데이터 컬럼 매핑
        refund_data = refund_data.rename(columns={
            '반품유형': '반품사유',
            '품목': '상품',
        })
        
        # 데이터 미리보기
        with st.sidebar.expander("판매 데이터 미리보기"):
            st.dataframe(sales_data.head())
            # 전체 데이터 다운로드 버튼 추가
            csv = sales_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 전체 판매 데이터 다운로드",
                data=csv,
                file_name="sales_data_full.csv",
                mime="text/csv",
                key="download_sales"
            )
        
        with st.sidebar.expander("반품 데이터 미리보기"):
            st.dataframe(refund_data.head())
            # 전체 데이터 다운로드 버튼 추가
            csv = refund_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 전체 반품 데이터 다운로드",
                data=csv,
                file_name="refund_data_full.csv",
                mime="text/csv",
                key="download_refund"
            )
        
        # 분석 시스템 초기화
        analyzer = MicrogreenAnalysisSystem(sales_data, refund_data)
        
        # 날짜 범위 선택 UI 추가
        st.markdown("---")
        st.markdown("### 📅 분석 기간 선택")
        
        # 데이터의 날짜 범위 및 실제 존재하는 날짜들 확인
        min_date, max_date = analyzer.get_date_range()
        available_dates = analyzer.get_available_dates()
        
        if min_date and max_date and available_dates:
            # 실제 존재하는 날짜들 정보 표시
            st.info(f"📅 데이터 기간: {min_date.date()} ~ {max_date.date()} (총 {len(available_dates)}일)")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # 실제 존재하는 날짜들만 선택 가능하도록 selectbox 사용
                start_date_options = available_dates
                start_date_index = 0  # 기본값: 첫 번째 날짜
                
                start_date = st.selectbox(
                    "시작 날짜",
                    options=start_date_options,
                    index=start_date_index,
                    key="start_date_select",
                    format_func=lambda x: x.strftime('%Y-%m-%d (%a)')
                )
            
            with col2:
                # 종료 날짜는 시작 날짜 이후의 날짜들만 선택 가능
                end_date_options = [d for d in available_dates if d >= start_date]
                end_date_index = len(end_date_options) - 1  # 기본값: 마지막 날짜
                
                end_date = st.selectbox(
                    "종료 날짜",
                    options=end_date_options,
                    index=end_date_index,
                    key="end_date_select",
                    format_func=lambda x: x.strftime('%Y-%m-%d (%a)')
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)  # 수직 정렬을 위한 공간
                apply_filter = st.button("🔍 기간 적용", type="primary")
            
            # 선택된 날짜 범위 표시
            selected_dates_count = len([d for d in available_dates if start_date <= d <= end_date])
            st.info(f"📊 선택된 분석 기간: {start_date} ~ {end_date} ({selected_dates_count}일)")
            
            # 날짜 필터 적용
            if apply_filter:
                if start_date <= end_date:
                    with st.spinner('선택된 기간으로 데이터를 필터링하고 있습니다...'):
                        # 날짜를 datetime으로 변환
                        start_datetime = pd.to_datetime(start_date)
                        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        
                        # 필터링된 분석기 생성
                        filtered_analyzer = analyzer.filter_data_by_date_range(start_datetime, end_datetime)
                        st.session_state.date_filtered_analyzer = filtered_analyzer
                        
                        # 필터링된 데이터 정보 표시
                        filtered_records = len(filtered_analyzer.sales_data)
                        total_records = len(analyzer.sales_data)
                        
                        st.success(f"✅ 필터링 완료: {total_records:,}개 중 {filtered_records:,}개 레코드가 선택되었습니다.")
                        st.info("👆 위에서 원하는 분석 카테고리를 선택하여 필터링된 데이터로 분석을 진행하세요.")
                else:
                    st.error("시작 날짜는 종료 날짜보다 이전이어야 합니다.")
            
            # 초기 로드 시에만 기본 필터 적용
            elif 'date_filtered_analyzer' not in st.session_state:
                # 기본적으로 전체 데이터 사용
                st.session_state.date_filtered_analyzer = analyzer
            
            # 필터링된 분석기 사용
            if 'date_filtered_analyzer' in st.session_state:
                analyzer = st.session_state.date_filtered_analyzer
        else:
            st.warning("날짜 정보를 찾을 수 없어 전체 데이터로 분석을 진행합니다.")
        
        st.markdown("---")
        
        # 탭 상태 관리를 위한 session_state 초기화
        if 'selected_tab' not in st.session_state:
            st.session_state.selected_tab = 0
        
        # 메인 탭 구성
        tab_names = [
            "📈 상품 분석", 
            "👥 업체 분석", 
            "🏢 고객관리",
            "💰 매출 지표",
            "📊 매출분석",
            "⭐ 미슐랭 분석",
            "🧁 베이커리 & 디저트"
        ]
        
        # 탭 선택 (라디오 버튼으로 변경하여 상태 유지)
        selected_tab_name = st.radio(
            "분석 카테고리를 선택하세요:",
            options=tab_names,
            index=st.session_state.selected_tab,
            horizontal=True,
            key="main_tab_selector"
        )
        
        # 선택된 탭 인덱스 업데이트
        st.session_state.selected_tab = tab_names.index(selected_tab_name)
        
        # 탭 1: 상품 분석
        if st.session_state.selected_tab == 0:
            st.markdown('<h2 class="sub-header">📈 상품 분석</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                # 필터링된 데이터의 날짜 범위 확인
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            # 상품 성과 히트맵 표시
            st.subheader("📊 상품 성과 히트맵 (전월 대비)")
            
            # 히트맵 분석 실행
            if 'date_filtered_analyzer' in st.session_state:
                # 필터링된 분석기의 날짜 범위 확인
                filtered_min, filtered_max = st.session_state.date_filtered_analyzer.get_date_range()
                if filtered_max:
                    heatmap_result = st.session_state.date_filtered_analyzer.analyze_product_performance_heatmap(filtered_max)
                else:
                    heatmap_result = {'상태': '실패', '메시지': '날짜 범위를 확인할 수 없습니다.'}
            else:
                # 전체 데이터의 최대 날짜 사용
                min_date, max_date = analyzer.get_date_range()
                if max_date:
                    heatmap_result = analyzer.analyze_product_performance_heatmap(max_date)
                else:
                    heatmap_result = {'상태': '실패', '메시지': '날짜 데이터가 없습니다.'}
            
            if heatmap_result['상태'] == '성공':
                heatmap_df = heatmap_result['히트맵_데이터']
                
                # 히트맵 정보 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("분석 상품 수", len(heatmap_df))
                with col2:
                    positive_change = len(heatmap_df[heatmap_df['변화율'] > 0])
                    st.metric("성장 상품", positive_change, delta=f"{positive_change}/{len(heatmap_df)}")
                with col3:
                    avg_change = heatmap_df['변화율'].mean()
                    st.metric("평균 성장률", f"{avg_change:.1f}%")
                
                st.info(f"📅 비교 기간: {heatmap_result['전월']} vs {heatmap_result['현재월']} (기준일: {heatmap_result['분석_기준일']})")
                
                # 트리맵 스타일 히트맵 생성
                import plotly.graph_objects as go
                import math
                
                # 색상 설정 (미국 주식 스타일)
                def get_color(change_rate):
                    if change_rate > 0:
                        # 초록색 계열 (성장)
                        intensity = min(abs(change_rate) / 100, 1.0)  # 0-1 사이로 정규화
                        return f'rgba(34, 139, 34, {0.3 + intensity * 0.7})'  # 연한 초록에서 진한 초록
                    elif change_rate < 0:
                        # 빨간색 계열 (하락)
                        intensity = min(abs(change_rate) / 100, 1.0)
                        return f'rgba(220, 20, 60, {0.3 + intensity * 0.7})'  # 연한 빨강에서 진한 빨강
                    else:
                        return 'rgba(128, 128, 128, 0.5)'  # 회색 (변화 없음)
                
                # 트리맵 데이터 준비
                fig = go.Figure(go.Treemap(
                    labels=heatmap_df['상품'],
                    parents=[""] * len(heatmap_df),  # 모든 항목이 루트 레벨
                    values=heatmap_df['총_판매량'],
                    text=[f"{row['상품']}<br>{row['변화율']:+.1f}%<br>판매량: {row['총_판매량']:,}" 
                          for _, row in heatmap_df.iterrows()],
                    textinfo="text",
                    textfont=dict(size=12, color="white"),
                    marker=dict(
                        colors=[get_color(rate) for rate in heatmap_df['변화율']],
                        line=dict(width=2, color="white")
                    ),
                    hovertemplate="<b>%{label}</b><br>" +
                                  "변화율: %{customdata[0]:+.1f}%<br>" +
                                  "현재월 판매량: %{customdata[1]:,}<br>" +
                                  "전월 판매량: %{customdata[2]:,}<br>" +
                                  "총 판매량: %{value:,}<extra></extra>",
                    customdata=heatmap_df[['변화율', '현재월_판매량', '전월_판매량']].values
                ))
                
                fig.update_layout(
                    title=f"상품 성과 히트맵 - {heatmap_result['전월']} vs {heatmap_result['현재월']} (기준일: {heatmap_result['분석_기준일']})",
                    font_size=12,
                    height=600,
                    margin=dict(t=50, l=0, r=0, b=0)
                )
                
                # 히트맵 표시
                st.plotly_chart(fig, use_container_width=True, key="product_heatmap")
                
                # 히트맵 상품 클릭 시뮬레이션을 위한 검색 가능한 선택 박스
                st.markdown("**🎯 히트맵에서 상품 선택 및 분석**")
                
                # 검색 가능한 상품 선택
                col1, col2 = st.columns([3, 1])
                with col1:
                    # 히트맵 데이터를 변화율 순으로 정렬
                    sorted_products = heatmap_df.sort_values('변화율', ascending=False)['상품'].tolist()
                    
                    # 상품명과 성과 정보를 함께 표시하는 옵션 생성
                    product_options = []
                    for _, row in heatmap_df.sort_values('변화율', ascending=False).iterrows():
                        change_rate = row['변화율']
                        emoji = "🚀" if change_rate > 50 else "📈" if change_rate > 0 else "📉"
                        option = f"{emoji} {row['상품']} ({change_rate:+.1f}%)"
                        product_options.append(option)
                    
                    selected_option = st.selectbox(
                        "히트맵에서 상품을 선택하세요 (성과순 정렬):",
                        options=product_options,
                        key="heatmap_product_select",
                        help="상품을 선택하면 우측 버튼으로 바로 분석할 수 있습니다"
                    )
                    
                    # 선택된 상품명 추출
                    if selected_option:
                        # 이모지와 변화율 정보를 제거하고 상품명만 추출
                        selected_product_name = selected_option.split(' (')[0][2:].strip()  # 이모지 제거
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # 수직 정렬
                    if st.button("🔍 선택한 상품 분석", type="primary", key="analyze_selected_product"):
                        if selected_option:
                            # 선택된 상품을 session_state에 저장하고 분석 실행
                            st.session_state.selected_product_from_heatmap = selected_product_name
                            st.session_state.trigger_product_analysis = True
                            st.success(f"🎯 '{selected_product_name}' 상품 분석을 시작합니다!")
                            st.rerun()
                
                # 성과 상위 상품들의 빠른 분석 버튼들
                st.markdown("**🚀 성과 상위 상품 빠른 분석**")
                top_performers = heatmap_df.nlargest(6, '변화율')  # 6개로 줄임
                
                # 2열로 버튼 배치 (더 깔끔하게)
                for i in range(0, len(top_performers), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(top_performers):
                            product = top_performers.iloc[i + j]
                            change_rate = product['변화율']
                            emoji = "🚀" if change_rate > 50 else "📈" if change_rate > 0 else "📉"
                            
                            with col:
                                if st.button(f"{emoji} {product['상품'][:20]}{'...' if len(product['상품']) > 20 else ''}", 
                                           key=f"quick_analysis_{i+j}",
                                           help=f"변화율: {change_rate:+.1f}% | 총 판매량: {product['총_판매량']:,}"):
                                    # 선택된 상품을 session_state에 저장하고 분석 실행
                                    st.session_state.selected_product_from_heatmap = product['상품']
                                    st.session_state.trigger_product_analysis = True
                                    st.rerun()
                
                # 분석 설명 및 범례 추가
                st.info(f"📊 **분석 설명**: {heatmap_result['분석_설명']}")
                
                st.markdown("""
                **📋 히트맵 사용법:**
                - 🟢 **초록색**: 전월 대비 판매량 증가
                - 🔴 **빨간색**: 전월 대비 판매량 감소  
                - 📦 **박스 크기**: 총 판매량 (클수록 많이 팔림)
                - 🎯 **상품 선택**: 위의 선택박스에서 원하는 상품을 찾아 분석
                - 🚀 **빠른 분석**: 성과 상위 상품들을 바로 분석 가능
                """)
                
                # 성과 요약 테이블
                with st.expander("📈 상품별 성과 상세 데이터"):
                    # 데이터 정렬 (변화율 기준 내림차순)
                    display_df = heatmap_df.sort_values('변화율', ascending=False).copy()
                    display_df['변화율'] = display_df['변화율'].apply(lambda x: f"{x:+.1f}%")
                    display_df = display_df[['상품', '현재월_판매량', '전월_판매량', '변화율', '총_판매량']]
                    display_df.columns = ['상품명', f'{heatmap_result["현재월"]} 판매량', f'{heatmap_result["전월"]} 판매량', '변화율', '총 판매량']
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning(f"히트맵을 생성할 수 없습니다: {heatmap_result['메시지']}")
            
            st.markdown("---")
            
            # 분석 유형 선택
            analysis_type = st.radio(
                "분석 유형을 선택하세요:",
                ["전체 상품 분석", "상품 분석 (포시즌스 호텔 제외)"],
                horizontal=True
            )
            
            # 상품 선택
            if not analyzer.sales_data.empty:
                products = sorted(analyzer.sales_data['상품'].unique())
            else:
                products = []
            
            if products:
                # 히트맵에서 선택된 상품이 있는지 확인하고 기본값 설정
                default_index = 0
                auto_analysis = False
                
                if 'selected_product_from_heatmap' in st.session_state and st.session_state.selected_product_from_heatmap:
                    # 히트맵에서 선택된 상품이 상품 목록에 있는지 확인
                    if st.session_state.selected_product_from_heatmap in products:
                        default_index = products.index(st.session_state.selected_product_from_heatmap)
                        
                        # 자동 분석 실행 여부 확인
                        if 'trigger_product_analysis' in st.session_state and st.session_state.trigger_product_analysis:
                            auto_analysis = True
                            st.session_state.trigger_product_analysis = False  # 플래그 리셋
                
                # 상품 선택 박스 (히트맵에서 선택된 상품이 기본값으로 설정됨)
                if 'selected_product_from_heatmap' in st.session_state and st.session_state.selected_product_from_heatmap:
                    st.info(f"🎯 히트맵에서 선택된 상품: **{st.session_state.selected_product_from_heatmap}**")
                
                selected_product = st.selectbox(
                    "분석할 상품을 선택하세요:", 
                    products, 
                    index=default_index,
                    key="main_product_select",
                    help="히트맵에서 상품을 선택하면 자동으로 해당 상품이 선택됩니다"
                )
                
                if analysis_type == "전체 상품 분석":
                    button_text = "상품 분석 실행"
                    button_key = "product_analysis_full"
                else:
                    button_text = "상품 분석 실행 (포시즌스 호텔 제외)"
                    button_key = "product_analysis_exclude_fourseasons"
                
                # 버튼 클릭 또는 자동 분석 실행
                if st.button(button_text, type="primary", key=button_key) or auto_analysis:
                    if auto_analysis:
                        st.info(f"🚀 히트맵에서 선택된 '{selected_product}' 상품을 자동으로 분석합니다!")
                    
                    with st.spinner('상품을 분석하고 있습니다...'):
                        if analysis_type == "전체 상품 분석":
                            result = analyzer.analyze_product_details(selected_product)
                        else:
                            result = analyzer.analyze_product_details_exclude_fourseasons(selected_product)
                    
                    if result['상태'] == '성공':
                        # 분석 유형 표시
                        if analysis_type == "상품 분석 (포시즌스 호텔 제외)":
                            st.info("🏨 포시즌스 호텔 관련 고객을 제외한 분석 결과입니다.")
                        
                        st.success(f"✅ {selected_product} 분석이 완료되었습니다!")
                        
                        # 기본 정보
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 판매량", f"{result['총_판매량']:,}개")
                        with col2:
                            st.metric("총 판매금액", f"{result['총_판매금액']:,}원")
                        with col3:
                            st.metric("평균 단가", f"{result['평균_단가']:,}원")
                        with col4:
                            st.metric("구매 고객수", f"{result['구매_고객수']:,}명")
                        
                        # 월별 판매 패턴
                        if result['월별_판매'] and len(result['월별_판매']) > 0:
                            st.subheader("📅 월별 판매 패턴")
                            
                            monthly_df = pd.DataFrame.from_dict(result['월별_판매'], orient='index')
                            monthly_df.index.name = '월'
                            monthly_df = monthly_df.reset_index()
                            
                            if not monthly_df.empty and '수량' in monthly_df.columns:
                                fig = px.line(monthly_df, x='월', y='수량', 
                                            title="월별 판매량 추이",
                                            markers=True)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("월별 판매 데이터가 충분하지 않습니다.")
                        
                        # 연월별 판매 패턴 (상세)
                        if result['연월별_판매'] and len(result['연월별_판매']) > 0:
                            st.subheader("📅 연월별 판매 패턴 (상세)")
                            
                            yearmonth_df = pd.DataFrame.from_dict(result['연월별_판매'], orient='index')
                            yearmonth_df.index.name = '연월'
                            yearmonth_df = yearmonth_df.reset_index()
                            yearmonth_df = yearmonth_df.sort_values('연월')
                            
                            if not yearmonth_df.empty and '수량' in yearmonth_df.columns:
                                # 연월별 판매량 추이 차트
                                fig = px.line(yearmonth_df, x='연월', y='수량',
                                            title="연월별 판매량 추이", markers=True)
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 금액 정보가 있는 경우 매출 차트도 표시
                                if '금액' in yearmonth_df.columns and yearmonth_df['금액'].sum() > 0:
                                    fig2 = px.bar(yearmonth_df, x='연월', y='금액',
                                                title="연월별 매출 추이")
                                    fig2.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                # 연월별 데이터 테이블
                                st.dataframe(yearmonth_df, use_container_width=True)
                            else:
                                st.info("연월별 판매 데이터가 충분하지 않습니다.")
                        
                        # 주요 고객
                        if result['주요_고객'] and len(result['주요_고객']) > 0:
                            st.subheader("👥 주요 구매 고객 TOP 10")
                            
                            customer_df = pd.DataFrame.from_dict(result['주요_고객'], orient='index', columns=['구매량'])
                            customer_df.index.name = '고객명'
                            customer_df = customer_df.reset_index()
                            
                            if not customer_df.empty:
                                fig = px.bar(customer_df, x='고객명', y='구매량',
                                           title="고객별 구매량")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("고객 구매 데이터가 없습니다.")
                        
                        # 계절별 분석
                        if result['계절별_판매'] and any(v > 0 for v in result['계절별_판매'].values()):
                            st.subheader("🌱 계절별 판매 분석")
                            
                            # 계절 순서 고정: 봄, 여름, 가을, 겨울
                            seasonal_order = ['봄', '여름', '가을', '겨울']
                            seasonal_data = []
                            for season in seasonal_order:
                                if season in result['계절별_판매']:
                                    seasonal_data.append({'계절': season, '판매량': result['계절별_판매'][season]})
                            
                            seasonal_df = pd.DataFrame(seasonal_data)
                            
                            if not seasonal_df.empty and seasonal_df['판매량'].sum() > 0:
                                fig = px.pie(seasonal_df, values='판매량', names='계절',
                                           title="계절별 판매 비중",
                                           category_orders={'계절': seasonal_order})
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("계절별 판매 데이터가 없습니다.")
                    else:
                        st.error(result['메시지'])
            else:
                st.warning("분석 가능한 상품이 없습니다.")
        
        # 탭 2: 업체 분석
        elif st.session_state.selected_tab == 1:
            st.markdown('<h2 class="sub-header">👥 업체 분석</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            # 고객 선택
            if not analyzer.customer_product_matrix.empty:
                customers = [c for c in analyzer.customer_product_matrix.index 
                            if not any(keyword in c for keyword in ['재고조정', '문정창고', '창고'])]
            else:
                customers = []
            
            if customers:
                selected_customer = st.selectbox("분석할 업체를 선택하세요:", customers, key="customer_select")
                
                # session_state 초기화
                if 'customer_analysis_result' not in st.session_state:
                    st.session_state.customer_analysis_result = None
                if 'analyzed_customer' not in st.session_state:
                    st.session_state.analyzed_customer = None
                
                # 업체가 변경되었거나 분석 결과가 없는 경우에만 분석 버튼 표시
                if (st.session_state.analyzed_customer != selected_customer or 
                    st.session_state.customer_analysis_result is None):
                    
                    if st.button("업체 분석 실행", type="primary"):
                        with st.spinner('업체를 분석하고 있습니다...'):
                            result = analyzer.analyze_customer_details(selected_customer)
                        
                        # 분석 결과를 session_state에 저장
                        st.session_state.customer_analysis_result = result
                        st.session_state.analyzed_customer = selected_customer
                        st.rerun()
                
                # 저장된 분석 결과가 있고 같은 업체인 경우 결과 표시
                if (st.session_state.customer_analysis_result is not None and 
                    st.session_state.analyzed_customer == selected_customer):
                    
                    result = st.session_state.customer_analysis_result
                    
                    if result['상태'] == '성공':
                        st.success(f"✅ {selected_customer} 분석이 완료되었습니다!")
                        
                        # 분석 결과 초기화 버튼 추가
                        if st.button("🔄 새로 분석하기", key="reset_analysis"):
                            st.session_state.customer_analysis_result = None
                            st.session_state.analyzed_customer = None
                            st.rerun()
                        
                        # 기본 정보
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 구매량", f"{result['총_구매량']:,}개")
                        with col2:
                            st.metric("총 구매금액", f"{result['총_구매금액']:,}원")
                        with col3:
                            st.metric("고객 카테고리", result['고객_카테고리'])
                        with col4:
                            st.metric("구매 빈도", f"{result['구매_빈도']:.1f}%")
                        
                        # 추가 상세 정보
                        col5, col6, col7, col8 = st.columns(4)
                        with col5:
                            st.metric("최근 구매일", result['최근_구매일'] or "정보 없음")
                        with col6:
                            st.metric("구매일수", f"{result['구매일수']:,}일")
                        with col7:
                            st.metric("고객 코드", result['고객_코드'] or "없음")
                        with col8:
                            if result['반품_정보']:
                                st.metric("반품 비율", f"{result['반품_정보']['반품_비율']:.1f}%")
                            else:
                                st.metric("반품 비율", "0.0%")
                        
                        # 연월별 구매 패턴 (상세)
                        if result['연월별_구매'] and len(result['연월별_구매']) > 0:
                            st.subheader("📅 연월별 구매 패턴 (상세)")
                            
                            yearmonth_df = pd.DataFrame.from_dict(result['연월별_구매'], orient='index')
                            yearmonth_df.index.name = '연월'
                            yearmonth_df = yearmonth_df.reset_index()
                            yearmonth_df = yearmonth_df.sort_values('연월')
                            
                            if not yearmonth_df.empty and '수량' in yearmonth_df.columns:
                                # 연월별 매출 추이 차트 (금액 정보가 있는 경우)
                                if '금액' in yearmonth_df.columns and yearmonth_df['금액'].sum() > 0:
                                    fig = px.line(yearmonth_df, x='연월', y='금액',
                                                title="연월별 매출 추이", markers=True)
                                    fig.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # 금액 정보가 없는 경우 구매량 추이로 대체
                                    fig = px.line(yearmonth_df, x='연월', y='수량',
                                                title="연월별 구매량 추이 (매출 정보 없음)", markers=True)
                                    fig.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # 연월별 구매량 차트 (보조 차트)
                                if '금액' in yearmonth_df.columns and yearmonth_df['금액'].sum() > 0:
                                    fig2 = px.bar(yearmonth_df, x='연월', y='수량',
                                                title="연월별 구매량")
                                    fig2.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                # 연월별 데이터 테이블
                                st.dataframe(yearmonth_df, use_container_width=True)
                            else:
                                st.info("연월별 구매 데이터가 없습니다.")
                        
                        # 월별 구매 패턴 (기본)
                        elif result['월별_구매'] and len(result['월별_구매']) > 0:
                            st.subheader("📅 월별 구매 패턴")
                            
                            monthly_df = pd.DataFrame.from_dict(result['월별_구매'], orient='index')
                            monthly_df.index.name = '월'
                            monthly_df = monthly_df.reset_index()
                            
                            if not monthly_df.empty and '수량' in monthly_df.columns:
                                fig = px.bar(monthly_df, x='월', y='수량',
                                           title="월별 구매량")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("월별 구매 데이터가 없습니다.")
                        
                        # 연월별 상품 구매 내역
                        if result['연월별_상품_구매'] and len(result['연월별_상품_구매']) > 0:
                            st.subheader("🛒 연월별 상품 구매 내역")
                            
                            # 연월 선택
                            available_yearmonths = sorted(result['연월별_상품_구매'].keys())
                            selected_yearmonth = st.selectbox("연월 선택:", available_yearmonths, key="yearmonth_products_select")
                            
                            if selected_yearmonth:
                                yearmonth_products = result['연월별_상품_구매'][selected_yearmonth]
                                
                                if yearmonth_products:
                                    products_df = pd.DataFrame.from_dict(yearmonth_products, orient='index', columns=['구매량'])
                                    products_df.index.name = '상품명'
                                    products_df = products_df.reset_index()
                                    products_df = products_df.sort_values('구매량', ascending=False)
                                    
                                    # 상위 10개 상품 차트
                                    top_products_df = products_df.head(10)
                                    fig = px.bar(top_products_df, x='상품명', y='구매량',
                                               title=f"{selected_yearmonth} 상품별 구매량 TOP 10")
                                    fig.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 전체 상품 데이터 테이블
                                    st.dataframe(products_df, use_container_width=True)
                                else:
                                    st.info(f"{selected_yearmonth}에 구매한 상품이 없습니다.")
                        
                        # 연월별 날짜별 구매 기록
                        if result['연월별_구매_날짜'] and len(result['연월별_구매_날짜']) > 0:
                            st.subheader("📆 연월별 날짜별 구매 기록")
                            
                            # 연월 선택
                            available_yearmonths_dates = sorted(result['연월별_구매_날짜'].keys())
                            selected_yearmonth_dates = st.selectbox("날짜별 기록을 볼 연월 선택:", available_yearmonths_dates, key="yearmonth_dates_select")
                            
                            if selected_yearmonth_dates:
                                date_records = result['연월별_구매_날짜'][selected_yearmonth_dates]
                                
                                if date_records:
                                    # 날짜별 구매 기록을 데이터프레임으로 변환
                                    date_data = []
                                    for date, products in date_records.items():
                                        for product, quantity in products.items():
                                            date_data.append({
                                                '날짜': date,
                                                '상품': product,
                                                '수량': quantity
                                            })
                                    
                                    if date_data:
                                        date_df = pd.DataFrame(date_data)
                                        
                                        # 날짜별 총 구매량
                                        daily_total = date_df.groupby('날짜')['수량'].sum().reset_index()
                                        fig = px.bar(daily_total, x='날짜', y='수량',
                                                   title=f"{selected_yearmonth_dates} 날짜별 총 구매량")
                                        fig.update_layout(xaxis_tickangle=45)
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # 상세 구매 기록 테이블
                                        st.dataframe(date_df, use_container_width=True)
                                else:
                                    st.info(f"{selected_yearmonth_dates}에 구매 기록이 없습니다.")
                        
                        # 제품별 구매 이력
                        if result['제품별_구매_이력'] and len(result['제품별_구매_이력']) > 0:
                            st.subheader("📈 제품별 구매 이력")
                            
                            # 제품 선택
                            available_products = list(result['제품별_구매_이력'].keys())
                            selected_product = st.selectbox("제품 선택:", available_products, key="product_history_select")
                            
                            if selected_product:
                                product_history = result['제품별_구매_이력'][selected_product]
                                
                                if product_history['구매일'] and product_history['구매량']:
                                    history_df = pd.DataFrame({
                                        '날짜': product_history['구매일'],
                                        '구매량': product_history['구매량']
                                    })
                                    history_df['날짜'] = pd.to_datetime(history_df['날짜'])
                                    history_df = history_df.sort_values('날짜')
                                    
                                    # 제품별 구매 추이 차트
                                    fig = px.line(history_df, x='날짜', y='구매량',
                                                title=f"{selected_product} 구매 추이", markers=True)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 구매 이력 테이블
                                    st.dataframe(history_df, use_container_width=True)
                        
                        # 주요 구매 상품
                        if result['주요_구매상품'] and len(result['주요_구매상품']) > 0:
                            st.subheader("🛒 주요 구매 상품 TOP 5")
                            
                            products_df = pd.DataFrame.from_dict(result['주요_구매상품'], orient='index', columns=['구매량'])
                            products_df.index.name = '상품명'
                            products_df = products_df.reset_index()
                            
                            if not products_df.empty and products_df['구매량'].sum() > 0:
                                fig = px.pie(products_df, values='구매량', names='상품명',
                                           title="주요 구매 상품 비중")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 상품별 구매량 바 차트
                                fig2 = px.bar(products_df, x='상품명', y='구매량',
                                            title="주요 구매 상품별 구매량")
                                fig2.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("구매 상품 데이터가 없습니다.")
                        
                        # 계절별 선호도
                        if result['계절별_선호도'] and any(v > 0 for v in result['계절별_선호도'].values()):
                            st.subheader("🌱 계절별 구매 패턴")
                            
                            # 계절 순서 고정: 봄, 여름, 가을, 겨울
                            seasonal_order = ['봄', '여름', '가을', '겨울']
                            seasonal_data = []
                            for season in seasonal_order:
                                if season in result['계절별_선호도']:
                                    seasonal_data.append({'계절': season, '구매량': result['계절별_선호도'][season]})
                            
                            seasonal_df = pd.DataFrame(seasonal_data)
                            
                            if not seasonal_df.empty and seasonal_df['구매량'].sum() > 0:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(seasonal_df, x='계절', y='구매량',
                                               title="계절별 구매량",
                                               color='구매량',
                                               color_continuous_scale='Viridis',
                                               category_orders={'계절': seasonal_order})
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig2 = px.pie(seasonal_df, values='구매량', names='계절',
                                                title="계절별 구매 비중",
                                                category_orders={'계절': seasonal_order})
                                    st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("계절별 구매 데이터가 없습니다.")
                        
                        # 분기별 선호도
                        if result['분기별_선호도'] and any(v > 0 for v in result['분기별_선호도'].values()):
                            st.subheader("📊 분기별 구매 패턴")
                            
                            # 분기 순서 고정: 1분기, 2분기, 3분기, 4분기
                            quarterly_order = ['1분기', '2분기', '3분기', '4분기']
                            quarterly_data = []
                            for quarter in quarterly_order:
                                if quarter in result['분기별_선호도']:
                                    quarterly_data.append({'분기': quarter, '구매량': result['분기별_선호도'][quarter]})
                            
                            quarterly_df = pd.DataFrame(quarterly_data)
                            
                            if not quarterly_df.empty and quarterly_df['구매량'].sum() > 0:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(quarterly_df, x='분기', y='구매량',
                                               title="분기별 구매량",
                                               color='구매량',
                                               color_continuous_scale='Blues',
                                               category_orders={'분기': quarterly_order})
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig2 = px.pie(quarterly_df, values='구매량', names='분기',
                                                title="분기별 구매 비중",
                                                category_orders={'분기': quarterly_order})
                                    st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("분기별 구매 데이터가 없습니다.")
                        
                        # 구매 날짜 분석
                        if result['구매_날짜'] and len(result['구매_날짜']) > 0:
                            st.subheader("📅 구매 날짜 분석")
                            
                            try:
                                # 구매 날짜 데이터 처리
                                purchase_dates_raw = result['구매_날짜']
                                
                                # 문자열 리스트를 datetime으로 변환
                                if isinstance(purchase_dates_raw, list):
                                    # 리스트인 경우 Series로 변환 후 datetime 변환
                                    purchase_dates_series = pd.Series(purchase_dates_raw)
                                    purchase_dates = pd.to_datetime(purchase_dates_series, errors='coerce')
                                else:
                                    # 이미 Series나 다른 형태인 경우
                                    purchase_dates = pd.to_datetime(purchase_dates_raw, errors='coerce')
                                
                                # 유효한 날짜만 필터링
                                valid_purchase_dates = purchase_dates.dropna()
                                
                                if not valid_purchase_dates.empty and len(valid_purchase_dates) > 0:
                                    # 요일별 구매 패턴
                                    try:
                                        weekday_purchases = valid_purchase_dates.dt.day_name().value_counts()
                                        if not weekday_purchases.empty:
                                            weekday_df = pd.DataFrame({
                                                '요일': weekday_purchases.index,
                                                '구매횟수': weekday_purchases.values
                                            })
                                            
                                            fig = px.bar(weekday_df, x='요일', y='구매횟수',
                                                       title="요일별 구매 횟수")
                                            st.plotly_chart(fig, use_container_width=True)
                                    except Exception as weekday_error:
                                        st.warning(f"요일별 분석 중 오류: {str(weekday_error)}")
                                    
                                    # 월별 구매 횟수
                                    try:
                                        monthly_purchases_count = valid_purchase_dates.dt.month.value_counts().sort_index()
                                        if not monthly_purchases_count.empty:
                                            monthly_count_df = pd.DataFrame({
                                                '월': monthly_purchases_count.index,
                                                '구매횟수': monthly_purchases_count.values
                                            })
                                            
                                            fig2 = px.line(monthly_count_df, x='월', y='구매횟수',
                                                         title="월별 구매 횟수", markers=True)
                                            st.plotly_chart(fig2, use_container_width=True)
                                    except Exception as monthly_error:
                                        st.warning(f"월별 분석 중 오류: {str(monthly_error)}")
                                else:
                                    st.info("유효한 구매 날짜 데이터가 없습니다.")
                            except Exception as e:
                                st.warning(f"구매 날짜 분석 중 오류 발생: {str(e)}")
                                st.info("날짜 데이터 형식을 확인해주세요.")
                        
                        # 반품 정보
                        if result['반품_정보'] and result['반품_정보'].get('반품_수량', 0) > 0:
                            st.subheader("↩️ 반품 정보")
                            
                            refund_info = result['반품_정보']
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("반품 수량", f"{refund_info['반품_수량']:,}개")
                            with col2:
                                st.metric("반품 비율", f"{refund_info['반품_비율']:.1f}%")
                            
                            # 반품 사유별 분석
                            if refund_info['반품_이유']:
                                st.subheader("반품 사유별 분석")
                                refund_reasons_df = pd.DataFrame.from_dict(refund_info['반품_이유'], orient='index', columns=['반품수량'])
                                refund_reasons_df.index.name = '반품사유'
                                refund_reasons_df = refund_reasons_df.reset_index()
                                
                                fig = px.pie(refund_reasons_df, values='반품수량', names='반품사유',
                                           title="반품 사유별 비중")
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(result['메시지'])
            else:
                st.warning("분석 가능한 업체가 없습니다.")
        
        # 탭 3: 고객관리
        elif st.session_state.selected_tab == 2:
            st.markdown('<h2 class="sub-header">🏢 고객관리</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            # 관리 카테고리 선택
            management_type = st.selectbox(
                "관리 유형을 선택하세요:",
                ["이탈 업체 관리", "클레임 발생 업체 관리", "신규 업체 관리"]
            )
            
            if management_type == "이탈 업체 관리":
                st.subheader("📉 이탈 업체 관리")
                st.info("최근 3개월간 구매 이력이 없는 업체를 분석합니다.")
                
                if st.button("이탈 업체 분석 실행", type="primary"):
                    with st.spinner('이탈 업체를 분석하고 있습니다...'):
                        result = analyzer.analyze_churned_customers()
                    
                    if result['상태'] == '성공':
                        st.success("✅ 이탈 업체 분석이 완료되었습니다!")
                        
                        # 기본 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("분석 기준일", result['기준_날짜'])
                        with col2:
                            st.metric("오늘 날짜", result['오늘_날짜'])
                        with col3:
                            st.metric("이탈 업체 수", f"{result['이탈_업체수']:,}개")
                        
                        if result['이탈_업체_목록']:
                            st.subheader("📋 이탈 업체 목록")
                            
                            # 데이터프레임으로 변환
                            churned_df = pd.DataFrame(result['이탈_업체_목록'])
                            
                            # 이탈 일수별 분포 차트
                            if not churned_df.empty:
                                fig = px.histogram(churned_df, x='이탈_일수', nbins=20,
                                                 title="이탈 일수 분포")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 상위 이탈 업체 (이탈 일수 기준)
                                top_churned = churned_df.head(20)
                                fig2 = px.bar(top_churned, x='고객명', y='이탈_일수',
                                            title="상위 20개 이탈 업체 (이탈 일수 기준)")
                                fig2.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # 전체 이탈 업체 테이블
                                st.dataframe(churned_df, use_container_width=True)
                        else:
                            st.info("이탈 업체가 없습니다.")
                    else:
                        st.error(result['메시지'])
            
            elif management_type == "클레임 발생 업체 관리":
                st.subheader("⚠️ 클레임 발생 업체 관리")
                st.info("최근 3개월간 클레임(반품)이 발생한 업체를 분석합니다.")
                
                if st.button("클레임 업체 분석 실행", type="primary"):
                    with st.spinner('클레임 발생 업체를 분석하고 있습니다...'):
                        result = analyzer.analyze_claim_customers()
                    
                    if result['상태'] == '성공':
                        st.success("✅ 클레임 업체 분석이 완료되었습니다!")
                        
                        # 기본 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("분석 기준일", result['기준_날짜'])
                        with col2:
                            st.metric("오늘 날짜", result['오늘_날짜'])
                        with col3:
                            st.metric("클레임 업체 수", f"{result['클레임_업체수']:,}개")
                        
                        if result['클레임_업체_목록']:
                            st.subheader("📋 클레임 발생 업체 목록")
                            
                            # 데이터프레임으로 변환
                            claim_df = pd.DataFrame(result['클레임_업체_목록'])
                            
                            # 클레임 횟수별 분포 차트
                            if not claim_df.empty:
                                fig = px.histogram(claim_df, x='클레임_횟수', nbins=10,
                                                 title="클레임 횟수 분포")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 상위 클레임 업체
                                top_claims = claim_df.head(15)
                                fig2 = px.bar(top_claims, x='고객명', y='클레임_횟수',
                                            title="상위 15개 클레임 발생 업체")
                                fig2.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # 클레임 사유 분석
                                if '클레임_사유' in claim_df.columns:
                                    st.subheader("📊 클레임 사유 분석")
                                    all_reasons = []
                                    for reasons in claim_df['클레임_사유']:
                                        if reasons and reasons != '사유 없음':
                                            all_reasons.extend([r.strip() for r in reasons.split(',')])
                                    
                                    if all_reasons:
                                        reason_counts = pd.Series(all_reasons).value_counts()
                                        fig3 = px.pie(values=reason_counts.values, names=reason_counts.index,
                                                    title="클레임 사유별 분포")
                                        st.plotly_chart(fig3, use_container_width=True)
                                
                                # 전체 클레임 업체 테이블
                                st.dataframe(claim_df, use_container_width=True)
                        else:
                            st.info("최근 3개월간 클레임이 발생한 업체가 없습니다.")
                    else:
                        st.error(result['메시지'])
            
            elif management_type == "신규 업체 관리":
                st.subheader("🆕 신규 업체 관리")
                st.info("2025년 기준으로 초도 구매가 이루어진 신규 업체를 분석합니다.")
                
                if st.button("신규 업체 분석 실행", type="primary"):
                    with st.spinner('신규 업체를 분석하고 있습니다...'):
                        result = analyzer.analyze_new_customers_2025()
                    
                    if result['상태'] == '성공':
                        st.success("✅ 신규 업체 분석이 완료되었습니다!")
                        
                        # 기본 정보
                        st.metric("2025년 신규 업체 수", f"{result['신규_업체수']:,}개")
                        
                        if result['신규_업체_목록']:
                            st.subheader("📋 2025년 신규 업체 목록")
                            
                            # 데이터프레임으로 변환
                            new_df = pd.DataFrame(result['신규_업체_목록'])
                            
                            if not new_df.empty:
                                # 월별 신규 업체 등록 추이
                                new_df['첫_구매월'] = pd.to_datetime(new_df['첫_구매일']).dt.month
                                monthly_new = new_df['첫_구매월'].value_counts().sort_index()
                                
                                # 월 데이터를 문자열로 변환하여 카테고리형으로 처리
                                monthly_chart_df = pd.DataFrame({
                                    '월': [f"{month}월" for month in monthly_new.index],
                                    '신규_업체수': monthly_new.values
                                })
                                
                                fig = px.bar(monthly_chart_df, x='월', y='신규_업체수',
                                           title="2025년 월별 신규 업체 등록 추이")
                                fig.update_layout(
                                    xaxis_title="월",
                                    yaxis_title="신규 업체 수",
                                    xaxis={'categoryorder': 'array', 'categoryarray': [f"{i}월" for i in range(1, 13)]}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 신규 업체 구매 규모 분포
                                fig2 = px.scatter(new_df, x='총_구매량', y='총_구매금액',
                                                size='구매_상품수', hover_data=['고객명'],
                                                title="신규 업체 구매 규모 분포")
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # 상위 신규 업체 (구매금액 기준)
                                top_new = new_df.nlargest(15, '총_구매금액')
                                fig3 = px.bar(top_new, x='고객명', y='총_구매금액',
                                            title="상위 15개 신규 업체 (구매금액 기준)")
                                fig3.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig3, use_container_width=True)
                                
                                # 전체 신규 업체 테이블
                                st.dataframe(new_df, use_container_width=True)
                        else:
                            st.info("2025년 신규 업체가 없습니다.")
                    else:
                        st.error(result['메시지'])
        
        # 탭 4: 매출 지표
        elif st.session_state.selected_tab == 3:
            st.markdown('<h2 class="sub-header">💰 매출 지표</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            # 매출 지표 카테고리 선택
            # 매출 지표 상태 관리를 위한 session_state 초기화
            if 'sales_metric_result' not in st.session_state:
                st.session_state.sales_metric_result = None
            if 'sales_metric_category' not in st.session_state:
                st.session_state.sales_metric_category = "다이닝 VIP 지표"
                
            # 매출 지표 카테고리 선택
            metric_category = st.selectbox(
                "매출 지표 카테고리를 선택하세요:",
                ["다이닝 VIP 지표", "호텔 VIP 지표", "BANQUET 지표"],
                index=0 if st.session_state.sales_metric_category == "다이닝 VIP 지표" else (1 if st.session_state.sales_metric_category == "호텔 VIP 지표" else 2),
                key="metric_category_selector"
            )
            
            # 카테고리가 변경된 경우 결과 초기화
            if metric_category != st.session_state.sales_metric_category:
                st.session_state.sales_metric_result = None
                st.session_state.sales_metric_category = metric_category
            
            if metric_category == "다이닝 VIP 지표":
                st.info("📊 다이닝 VIP 지표 분석을 제공합니다 (선별 7개 업체)")
            elif metric_category == "호텔 VIP 지표":
                st.info("🏨 호텔 VIP 지표 분석을 제공합니다 (선별 5개 호텔)")
            else:
                st.info("🎉 BANQUET 지표 분석을 제공합니다 (비고란에 BANQUET 키워드 포함)")
                
            # 분석 실행
            if st.session_state.sales_metric_result is None:
                
                try:
                    if metric_category == "다이닝 VIP 지표":
                        with st.spinner('다이닝 VIP 지표 분석을 수행하고 있습니다...'):
                            result = analyzer.analyze_dining_vip_metrics()
                    elif metric_category == "호텔 VIP 지표":
                        with st.spinner('호텔 VIP 지표 분석을 수행하고 있습니다...'):
                            result = analyzer.analyze_hotel_vip_metrics()
                    else:  # BANQUET 지표
                        with st.spinner('BANQUET 지표 분석을 수행하고 있습니다...'):
                            result = analyzer.analyze_banquet_metrics()
                    
                    st.session_state.sales_metric_result = result
                    
                    if result and result.get('상태') == '성공':
                        if metric_category == "다이닝 VIP 지표":
                            st.success("✅ 다이닝 VIP 지표 분석이 완료되었습니다!")
                        elif metric_category == "호텔 VIP 지표":
                            st.success("✅ 호텔 VIP 지표 분석이 완료되었습니다!")
                        else:
                            st.success("✅ BANQUET 지표 분석이 완료되었습니다!")
                    elif result and result.get('상태') == '실패':
                        st.error(f"❌ {metric_category} 분석 실패: {result.get('메시지', '알 수 없는 오류')}")
                    else:
                        st.error(f"❌ {metric_category} 분석 중 예상치 못한 오류가 발생했습니다.")
                
                except Exception as e:
                    print(f"{metric_category} 분석 중 예외 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_result = {
                        '상태': '실패',
                        '메시지': f"분석 중 예외 발생: {str(e)}"
                    }
                    st.session_state.sales_metric_result = error_result
                    st.error(f"❌ {metric_category} 분석 중 오류 발생: {str(e)}")
                
                # session_state에서 결과 가져오기
                result = st.session_state.sales_metric_result
                
                # 결과가 없거나 상태가 없는 경우 처리
                if result is None:
                    st.warning("분석 결과가 없습니다. 다시 시도해 주세요.")
                elif result.get('상태') != '성공':
                    error_msg = result.get('메시지', '알 수 없는 오류가 발생했습니다.')
                    st.error(f"데이터 처리 중 오류가 발생했습니다: {error_msg}")
                    
                    # 디버깅 정보 표시
                    if st.checkbox("디버깅 정보 표시"):
                        st.json(result)
                elif result.get('상태') == '성공':
                    
                    # 선택된 지표에 따른 결과 표시
                    if metric_category == "다이닝 VIP 지표":
                        st.subheader("🍽️ 다이닝 VIP 매출 분석 (선별 7개 업체)")
                    elif metric_category == "호텔 VIP 지표":
                        st.subheader("🏨 호텔 VIP 매출 분석 (선별 5개 호텔)")
                    else:
                        st.subheader("🎉 BANQUET 매출 분석")
                        
                    # 총 매출
                    if result['customer_total_revenue']:
                        revenue_df = pd.DataFrame.from_dict(result['customer_total_revenue'], orient='index', columns=['총매출'])
                        revenue_df.index.name = '고객명'
                        revenue_df = revenue_df.reset_index()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if metric_category == "다이닝 VIP 지표":
                                title = "선별 7개 업체 총 매출"
                            elif metric_category == "호텔 VIP 지표":
                                title = "선별 5개 호텔 총 매출"
                            else:
                                title = "BANQUET 고객 총 매출"
                            fig = px.bar(revenue_df, x='고객명', y='총매출', title=title)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if metric_category == "다이닝 VIP 지표":
                                title = "선별 7개 업체 매출 비중"
                            elif metric_category == "호텔 VIP 지표":
                                title = "선별 5개 호텔 매출 비중"
                            else:
                                title = "BANQUET 고객 매출 비중"
                            fig_pie = px.pie(revenue_df, values='총매출', names='고객명', title=title)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        st.dataframe(revenue_df, use_container_width=True)
                        
                    # 연월별 매출 추이
                    if result['yearmonth_revenue']:
                        if metric_category == "다이닝 VIP 지표":
                            st.subheader("📅 선별 7개 업체 연월별 매출 추이")
                        elif metric_category == "호텔 VIP 지표":
                            st.subheader("📅 선별 5개 호텔 연월별 매출 추이")
                        else:
                            st.subheader("📅 BANQUET 고객 연월별 매출 추이")
                            
                        # 연월별 매출 데이터 준비
                        yearmonth_data = []
                        for customer, yearmonth_sales in result['yearmonth_revenue'].items():
                            for yearmonth, amount in yearmonth_sales.items():
                                yearmonth_data.append({
                                    '고객명': customer,
                                    '연월': yearmonth,
                                    '매출': amount
                                })
                        
                        if yearmonth_data:
                            yearmonth_df = pd.DataFrame(yearmonth_data)
                            
                            # 연월별 매출 추이 그래프
                            fig = px.line(yearmonth_df, x='연월', y='매출', color='고객명',
                                        title="연월별 매출 추이", markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 연월별 매출 표 (피벗 테이블)
                            st.subheader("📊 연월별 매출 상세 표")
                            pivot_df = yearmonth_df.pivot(index='고객명', columns='연월', values='매출').fillna(0)
                            
                            # 숫자 포맷팅 (천단위 구분자)
                            pivot_formatted = pivot_df.applymap(lambda x: f"{int(x):,}" if x != 0 else "0")
                            st.dataframe(pivot_formatted, use_container_width=True)
                            
                            # 총계 행 추가
                            total_row = pivot_df.sum().to_frame().T
                            total_row.index = ['총계']
                            total_formatted = total_row.applymap(lambda x: f"{int(x):,}")
                            st.write("**월별 총계:**")
                            st.dataframe(total_formatted, use_container_width=True)
                        
                    # 연월별 통합 TOP 10 상품
                    if result['monthly_top10_products']:
                        if metric_category == "다이닝 VIP 지표":
                            st.subheader("🏆 7개 업체 통합 연월별 TOP 10 상품")
                            session_key = 'dining_vip_selected_month'
                            selector_key = "dining_vip_month_selector"
                        elif metric_category == "호텔 VIP 지표":
                            st.subheader("🏆 5개 호텔 통합 연월별 TOP 10 상품")
                            session_key = 'hotel_vip_selected_month'
                            selector_key = "hotel_vip_month_selector"
                        else:
                            st.subheader("🏆 BANQUET 고객 통합 연월별 TOP 10 상품")
                            session_key = 'banquet_selected_month'
                            selector_key = "banquet_month_selector"
                        
                        # 연월 선택 (session_state로 상태 유지)
                        available_months = sorted(result['monthly_top10_products'].keys())
                        
                        # 기본 선택값 설정 (최신 월)
                        if session_key not in st.session_state:
                            st.session_state[session_key] = available_months[-1] if available_months else None
                        
                        # 현재 선택된 월이 available_months에 없으면 기본값으로 리셋
                        if st.session_state[session_key] not in available_months:
                            st.session_state[session_key] = available_months[-1] if available_months else None
                        
                        # 현재 선택된 월의 인덱스 찾기
                        try:
                            current_index = available_months.index(st.session_state[session_key])
                        except (ValueError, AttributeError):
                            current_index = len(available_months)-1 if available_months else 0
                        
                        selected_month = st.selectbox(
                            "연월을 선택하세요:",
                            available_months,
                            index=current_index,
                            key=selector_key
                        )
                        
                        # 선택된 월을 session_state에 저장
                        st.session_state[session_key] = selected_month
                        
                        if selected_month and selected_month in result['monthly_top10_products']:
                            top10_data = result['monthly_top10_products'][selected_month]
                            
                            if top10_data:
                                # 데이터프레임으로 변환
                                top10_df = pd.DataFrame(top10_data)
                                top10_df['순위'] = range(1, len(top10_df) + 1)
                                top10_df = top10_df[['순위', '상품', '수량', '금액']]
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 바 차트
                                    fig_bar = px.bar(top10_df.head(10), x='상품', y='금액',
                                                   title=f"{selected_month} TOP 10 상품 매출")
                                    fig_bar.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig_bar, use_container_width=True)
                                
                                with col2:
                                    # 파이차트
                                    fig_pie = px.pie(top10_df.head(10), values='금액', names='상품',
                                                   title=f"{selected_month} TOP 10 상품 비중")
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # 상세 표
                                st.write(f"**{selected_month} TOP 10 상품 상세:**")
                                # 숫자 포맷팅
                                display_df = top10_df.copy()
                                display_df['수량'] = display_df['수량'].apply(lambda x: f"{int(x):,}")
                                display_df['금액'] = display_df['금액'].apply(lambda x: f"{int(x):,}")
                                st.dataframe(display_df, use_container_width=True)
                            
                            # 모든 연월의 TOP 10 상품 요약
                            st.subheader("📋 전체 연월별 TOP 10 상품 요약")
                            
                            # 모든 월의 데이터를 하나의 표로 만들기
                            all_months_data = []
                            for month, products in result['monthly_top10_products'].items():
                                for i, product in enumerate(products[:5], 1):  # 상위 5개만
                                    all_months_data.append({
                                        '연월': month,
                                        '순위': i,
                                        '상품': product['상품'],
                                        '수량': f"{int(product['수량']):,}",
                                        '금액': f"{int(product['금액']):,}"
                                    })
                            
                            if all_months_data:
                                summary_df = pd.DataFrame(all_months_data)
                                st.dataframe(summary_df, use_container_width=True)
                        
                        # 개별 업체별 주요 품목 매출
                        if result['product_revenue']:
                            st.subheader("🛒 개별 업체별 주요 품목 매출")
                            
                            for customer, products in result['product_revenue'].items():
                                with st.expander(f"{customer} 주요 품목"):
                                    if products:
                                        product_df = pd.DataFrame.from_dict(products, orient='index', columns=['매출'])
                                        product_df.index.name = '상품명'
                                        product_df = product_df.reset_index()
                                        
                                        # 상위 10개 품목만 표시
                                        top_products = product_df.head(10)
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # 파이차트
                                            fig_pie = px.pie(top_products, values='매출', names='상품명',
                                                           title=f"{customer} Top 10 품목 매출 비중")
                                            st.plotly_chart(fig_pie, use_container_width=True)
                                        
                                        with col2:
                                            # 바 차트
                                            fig_bar = px.bar(top_products, x='상품명', y='매출',
                                                           title=f"{customer} Top 10 품목 매출")
                                            fig_bar.update_layout(xaxis_tickangle=45)
                                            st.plotly_chart(fig_bar, use_container_width=True)
                                        
                                        # 데이터 테이블 (숫자 포맷팅)
                                        display_products = top_products.copy()
                                        display_products['매출'] = display_products['매출'].apply(lambda x: f"{int(x):,}")
                                        st.dataframe(display_products, use_container_width=True)
                        

        
        # 탭 5: 매출분석
        elif st.session_state.selected_tab == 4:
            st.markdown('<h2 class="sub-header">📊 매출분석</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            # 분석 카테고리 선택
            analysis_category = st.selectbox(
                "분석 카테고리를 선택하세요:",
                ["월별 다이닝 매출 분석", "월별 호텔 매출 분석", "연별 다이닝/호텔 매출 비교"]
            )
            
            if st.button("매출분석 실행", type="primary"):
                with st.spinner('매출을 분석하고 있습니다...'):
                    if analysis_category == "월별 다이닝 매출 분석":
                        result = analyzer.analyze_monthly_dining_sales()
                    elif analysis_category == "월별 호텔 매출 분석":
                        result = analyzer.analyze_monthly_hotel_sales()
                    else:  # 연별 다이닝/호텔 매출 비교
                        result = analyzer.analyze_yearly_sales_comparison()
                
                if result['상태'] == '성공':
                    st.success(f"✅ {analysis_category} 분석이 완료되었습니다!")
                    
                    # 월별 다이닝 매출 분석 결과
                    if analysis_category == "월별 다이닝 매출 분석":
                        st.subheader("🍽️ 월별 다이닝 매출 분석")
                        
                        # 월별 총 매출 추이
                        if not result['monthly_total'].empty:
                            st.subheader("📈 월별 총 매출 추이")
                            fig = px.line(result['monthly_total'], x='연월', y='금액',
                                        title="월별 다이닝 총 매출", markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 업체별 월별 매출 히트맵
                        if not result['customer_heatmap'].empty:
                            st.subheader("🔥 업체별 월별 매출 히트맵")
                            
                            # 상위 20개 업체만 표시
                            top_customers = result['customer_heatmap'].sum(axis=1).nlargest(20).index
                            heatmap_data = result['customer_heatmap'].loc[top_customers]
                            
                            fig = px.imshow(heatmap_data.values,
                                          x=heatmap_data.columns,
                                          y=heatmap_data.index,
                                          title="업체별 월별 매출 히트맵 (상위 20개 업체)",
                                          color_continuous_scale='Blues')
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 품목별 월별 매출 히트맵
                        if not result['product_heatmap'].empty:
                            st.subheader("🛒 품목별 월별 매출 히트맵")
                            
                            fig = px.imshow(result['product_heatmap'].values,
                                          x=result['product_heatmap'].columns,
                                          y=result['product_heatmap'].index,
                                          title="품목별 월별 매출 히트맵 (상위 20개 품목)",
                                          color_continuous_scale='Greens')
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 월별 업체별 매출 상세
                        if not result['monthly_customer'].empty:
                            st.subheader("📊 월별 업체별 매출 상세")
                            
                            # 상위 10개 업체 선택
                            top_customers_monthly = result['monthly_customer'].groupby('고객명')['금액'].sum().nlargest(10).index
                            filtered_data = result['monthly_customer'][result['monthly_customer']['고객명'].isin(top_customers_monthly)]
                            
                            fig = px.line(filtered_data, x='연월', y='금액', color='고객명',
                                        title="월별 업체별 매출 추이 (상위 10개 업체)", markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 월별 호텔 매출 분석 결과
                    elif analysis_category == "월별 호텔 매출 분석":
                        st.subheader("🏨 월별 호텔 매출 분석")
                        
                        # 월별 총 매출 추이
                        if not result['monthly_total'].empty:
                            st.subheader("📈 월별 총 매출 추이")
                            fig = px.line(result['monthly_total'], x='연월', y='금액',
                                        title="월별 호텔 총 매출", markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 호텔별 월별 매출 히트맵
                        if not result['customer_heatmap'].empty:
                            st.subheader("🔥 호텔별 월별 매출 히트맵")
                            
                            fig = px.imshow(result['customer_heatmap'].values,
                                          x=result['customer_heatmap'].columns,
                                          y=result['customer_heatmap'].index,
                                          title="호텔별 월별 매출 히트맵",
                                          color_continuous_scale='Reds')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 품목별 월별 매출 히트맵
                        if not result['product_heatmap'].empty:
                            st.subheader("🛒 품목별 월별 매출 히트맵")
                            
                            fig = px.imshow(result['product_heatmap'].values,
                                          x=result['product_heatmap'].columns,
                                          y=result['product_heatmap'].index,
                                          title="품목별 월별 매출 히트맵 (상위 20개 품목)",
                                          color_continuous_scale='Oranges')
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 월별 호텔별 매출 상세
                        if not result['monthly_customer'].empty:
                            st.subheader("📊 월별 호텔별 매출 상세")
                            
                            fig = px.line(result['monthly_customer'], x='연월', y='금액', color='고객명',
                                        title="월별 호텔별 매출 추이", markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 연별 다이닝/호텔 매출 비교 결과
                    else:  # 연별 다이닝/호텔 매출 비교
                        st.subheader("📅 연별 다이닝/호텔 매출 비교")
                        
                        # 연별 매출 비교 차트
                        if not result['yearly_dining'].empty and not result['yearly_hotel'].empty:
                            # 데이터 결합
                            combined_yearly = pd.concat([result['yearly_dining'], result['yearly_hotel']])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.bar(combined_yearly, x='연도', y='금액', color='카테고리',
                                           title="연별 다이닝 vs 호텔 매출 비교", barmode='group')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 비율 차트
                                yearly_total = combined_yearly.groupby('연도')['금액'].sum().reset_index()
                                yearly_total['카테고리'] = '전체'
                                
                                fig = px.pie(combined_yearly, values='금액', names='카테고리',
                                           title="전체 기간 다이닝 vs 호텔 매출 비율")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # 다이닝 연도별 월별 매출 히트맵
                        if not result['dining_yearly_monthly'].empty:
                            st.subheader("🍽️ 다이닝 연도별 월별 매출 히트맵")
                            
                            fig = px.imshow(result['dining_yearly_monthly'].values,
                                          x=result['dining_yearly_monthly'].columns,
                                          y=result['dining_yearly_monthly'].index,
                                          title="다이닝 연도별 월별 매출 히트맵",
                                          color_continuous_scale='Blues')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 호텔 연도별 월별 매출 히트맵
                        if not result['hotel_yearly_monthly'].empty:
                            st.subheader("🏨 호텔 연도별 월별 매출 히트맵")
                            
                            fig = px.imshow(result['hotel_yearly_monthly'].values,
                                          x=result['hotel_yearly_monthly'].columns,
                                          y=result['hotel_yearly_monthly'].index,
                                          title="호텔 연도별 월별 매출 히트맵",
                                          color_continuous_scale='Reds')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 연도별 성장률 분석
                        if not result['yearly_dining'].empty and not result['yearly_hotel'].empty:
                            st.subheader("📈 연도별 성장률 분석")
                            
                            # 성장률 계산
                            dining_growth = result['yearly_dining'].copy()
                            dining_growth['성장률'] = dining_growth['금액'].pct_change() * 100
                            
                            hotel_growth = result['yearly_hotel'].copy()
                            hotel_growth['성장률'] = hotel_growth['금액'].pct_change() * 100
                            
                            growth_data = []
                            for _, row in dining_growth.iterrows():
                                if not pd.isna(row['성장률']):
                                    growth_data.append({
                                        '연도': row['연도'],
                                        '성장률': row['성장률'],
                                        '카테고리': '다이닝'
                                    })
                            
                            for _, row in hotel_growth.iterrows():
                                if not pd.isna(row['성장률']):
                                    growth_data.append({
                                        '연도': row['연도'],
                                        '성장률': row['성장률'],
                                        '카테고리': '호텔'
                                    })
                            
                            if growth_data:
                                growth_df = pd.DataFrame(growth_data)
                                
                                fig = px.line(growth_df, x='연도', y='성장률', color='카테고리',
                                            title="연도별 매출 성장률 비교", markers=True)
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(result['메시지'])
        
        # 탭 6: 미슐랭 분석
        elif st.session_state.selected_tab == 5:
            st.markdown('<h2 class="sub-header">⭐ 미슐랭 분석</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            # 미슐랭 등급별 vs 비미슐랭 업장 특징 비교 분석 (맨 위로 이동)
            st.subheader("🆚 미슐랭 vs 비미슐랭 업장 특징 비교")
            if st.button("미슐랭 vs 비미슐랭 비교 분석 실행", type="primary"):
                with st.spinner('미슐랭 vs 비미슐랭 비교 분석을 수행하고 있습니다...'):
                    vs_result = analyzer.analyze_michelin_vs_non_michelin()
                
                if vs_result['상태'] == '성공':
                    st.success("✅ 미슐랭 vs 비미슐랭 비교 분석이 완료되었습니다!")
                    
                    # 전체 미슐랭 통합 vs 비미슐랭 비교 섹션 추가
                    if vs_result.get('전체_미슐랭_지표'):
                        st.subheader("🌟 전체 미슐랭 통합 vs 비미슐랭 비교")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🌟 전체 미슐랭 업체 (통합)**")
                            michelin_stats = vs_result['전체_미슐랭_지표']
                            st.metric("업장 수", f"{michelin_stats['업장_수']:,}개")
                            st.metric("총 매출", f"{michelin_stats['총_매출']:,.0f}원")
                            st.metric("평균 주문금액", f"{michelin_stats['평균_주문금액']:,.0f}원")
                            st.metric("업장당 평균매출", f"{michelin_stats['업장당_평균매출']:,.0f}원")
                            st.metric("품목 다양성", f"{michelin_stats['품목_다양성']:,}개")
                        
                        with col2:
                            st.markdown("**🏪 비미슐랭 업체**")
                            non_michelin_stats = vs_result['비미슐랭_기준지표']
                            st.metric("업장 수", f"{non_michelin_stats['업장_수']:,}개")
                            st.metric("총 매출", f"{non_michelin_stats['총_매출']:,.0f}원")
                            st.metric("평균 주문금액", f"{non_michelin_stats['평균_주문금액']:,.0f}원")
                            st.metric("업장당 평균매출", f"{non_michelin_stats['업장당_평균매출']:,.0f}원")
                            st.metric("품목 다양성", f"{non_michelin_stats['품목_다양성']:,}개")
                        
                        # 통합 비교 배수 차트
                        if vs_result.get('전체_미슐랭_비교'):
                            st.subheader("📊 전체 미슐랭 vs 비미슐랭 비교 배수")
                            comparison_data = []
                            for metric, value in vs_result['전체_미슐랭_비교'].items():
                                comparison_data.append({
                                    '지표': metric.replace('_배수', '').replace('_', ' '),
                                    '배수': value
                                })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                fig = px.bar(comparison_df, x='지표', y='배수',
                                           title="전체 미슐랭 vs 비미슐랭 비교 (배수)",
                                           color='배수',
                                           color_continuous_scale='RdYlBu_r')
                                fig.add_hline(y=1, line_dash="dash", line_color="red", 
                                            annotation_text="비미슐랭 기준선")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # 품목 차이 분석
                        st.subheader("🛒 품목 선호도 차이 분석")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if vs_result.get('미슐랭_독특한_품목'):
                                st.markdown("**🌟 미슐랭만의 독특한 품목**")
                                unique_products = vs_result['미슐랭_독특한_품목'][:10]
                                for i, product in enumerate(unique_products, 1):
                                    st.write(f"{i}. {product}")
                        
                        with col2:
                            if vs_result.get('공통_품목'):
                                st.markdown("**🤝 공통 인기 품목**")
                                common_products = vs_result['공통_품목'][:10]
                                for i, product in enumerate(common_products, 1):
                                    st.write(f"{i}. {product}")
                        
                        with col3:
                            if vs_result.get('비미슐랭_독특한_품목'):
                                st.markdown("**🏪 비미슐랭만의 독특한 품목**")
                                non_michelin_unique = vs_result['비미슐랭_독특한_품목'][:10]
                                for i, product in enumerate(non_michelin_unique, 1):
                                    st.write(f"{i}. {product}")
                        
                        # 계절별/분기별 선호도 비교
                        if vs_result.get('전체_미슐랭_계절별_선호도') and vs_result.get('비미슐랭_계절별_선호도'):
                            st.subheader("🌱 계절별 구매 패턴 비교")
                            
                            # 계절별 데이터 준비
                            seasonal_order = ['봄', '여름', '가을', '겨울']
                            seasonal_comparison_data = []
                            
                            for season in seasonal_order:
                                michelin_qty = vs_result['전체_미슐랭_계절별_선호도'].get(season, 0)
                                non_michelin_qty = vs_result['비미슐랭_계절별_선호도'].get(season, 0)
                                
                                seasonal_comparison_data.append({
                                    '계절': season,
                                    '미슐랭': michelin_qty,
                                    '비미슐랭': non_michelin_qty
                                })
                            
                            seasonal_df = pd.DataFrame(seasonal_comparison_data)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.bar(seasonal_df, x='계절', y=['미슐랭', '비미슐랭'],
                                           title="계절별 구매량 비교",
                                           barmode='group',
                                           category_orders={'계절': seasonal_order})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 계절별 비율 비교
                                michelin_total = sum(vs_result['전체_미슐랭_계절별_선호도'].values())
                                non_michelin_total = sum(vs_result['비미슐랭_계절별_선호도'].values())
                                
                                seasonal_ratio_data = []
                                for season in seasonal_order:
                                    michelin_ratio = (vs_result['전체_미슐랭_계절별_선호도'].get(season, 0) / michelin_total * 100) if michelin_total > 0 else 0
                                    non_michelin_ratio = (vs_result['비미슐랭_계절별_선호도'].get(season, 0) / non_michelin_total * 100) if non_michelin_total > 0 else 0
                                    
                                    seasonal_ratio_data.append({
                                        '계절': season,
                                        '미슐랭 비율': michelin_ratio,
                                        '비미슐랭 비율': non_michelin_ratio
                                    })
                                
                                seasonal_ratio_df = pd.DataFrame(seasonal_ratio_data)
                                fig2 = px.line(seasonal_ratio_df, x='계절', y=['미슐랭 비율', '비미슐랭 비율'],
                                             title="계절별 구매 비율 비교 (%)", markers=True,
                                             category_orders={'계절': seasonal_order})
                                st.plotly_chart(fig2, use_container_width=True)
                        
                        if vs_result.get('전체_미슐랭_분기별_선호도') and vs_result.get('비미슐랭_분기별_선호도'):
                            st.subheader("📊 분기별 구매 패턴 비교")
                            
                            # 분기별 데이터 준비
                            quarterly_order = ['1분기', '2분기', '3분기', '4분기']
                            quarterly_comparison_data = []
                            
                            for quarter in quarterly_order:
                                michelin_qty = vs_result['전체_미슐랭_분기별_선호도'].get(quarter, 0)
                                non_michelin_qty = vs_result['비미슐랭_분기별_선호도'].get(quarter, 0)
                                
                                quarterly_comparison_data.append({
                                    '분기': quarter,
                                    '미슐랭': michelin_qty,
                                    '비미슐랭': non_michelin_qty
                                })
                            
                            quarterly_df = pd.DataFrame(quarterly_comparison_data)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.bar(quarterly_df, x='분기', y=['미슐랭', '비미슐랭'],
                                           title="분기별 구매량 비교",
                                           barmode='group',
                                           category_orders={'분기': quarterly_order})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 분기별 비율 비교  
                                michelin_total = sum(vs_result['전체_미슐랭_분기별_선호도'].values())
                                non_michelin_total = sum(vs_result['비미슐랭_분기별_선호도'].values())
                                
                                quarterly_ratio_data = []
                                for quarter in quarterly_order:
                                    michelin_ratio = (vs_result['전체_미슐랭_분기별_선호도'].get(quarter, 0) / michelin_total * 100) if michelin_total > 0 else 0
                                    non_michelin_ratio = (vs_result['비미슐랭_분기별_선호도'].get(quarter, 0) / non_michelin_total * 100) if non_michelin_total > 0 else 0
                                    
                                    quarterly_ratio_data.append({
                                        '분기': quarter,
                                        '미슐랭 비율': michelin_ratio,
                                        '비미슐랭 비율': non_michelin_ratio
                                    })
                                
                                quarterly_ratio_df = pd.DataFrame(quarterly_ratio_data)
                                fig2 = px.line(quarterly_ratio_df, x='분기', y=['미슐랭 비율', '비미슐랭 비율'],
                                             title="분기별 구매 비율 비교 (%)", markers=True,
                                             category_orders={'분기': quarterly_order})
                                st.plotly_chart(fig2, use_container_width=True)
                    
                    # 등급별 세부 비교 분석 (기존 코드도 유지)
                    st.divider()
                    st.subheader("🔍 등급별 세부 비교 분석")
                    
                    # 등급별 비교 차트
                    if vs_result.get('등급별_비교'):
                        comparison_data = []
                        for grade, data in vs_result['등급별_비교'].items():
                            michelin_stats = data['미슐랭_지표']
                            comparison_data.append({
                                '등급': grade,
                                '업장수': michelin_stats['업장_수'],
                                '총_매출': michelin_stats['총_매출'],
                                '평균_주문금액': michelin_stats['평균_주문금액'],
                                '업장당_평균매출': michelin_stats['업장당_평균매출'],
                                '품목_다양성': michelin_stats['품목_다양성']
                            })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # 등급별 총 매출 비교
                            fig1 = px.bar(comparison_df, x='등급', y='총_매출',
                                         title="미슐랭 등급별 총 매출 비교")
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # 등급별 평균 주문금액 비교
                            fig2 = px.bar(comparison_df, x='등급', y='평균_주문금액',
                                         title="미슐랭 등급별 평균 주문금액 비교")
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # 등급별 업장당 평균매출 비교
                            fig3 = px.bar(comparison_df, x='등급', y='업장당_평균매출',
                                         title="미슐랭 등급별 업장당 평균매출 비교")
                            st.plotly_chart(fig3, use_container_width=True)
                            
                            # 비미슐랭 기준 대비 배수 비교
                            if vs_result.get('비미슐랭_기준지표'):
                                non_michelin_avg_order = vs_result['비미슐랭_기준지표']['평균_주문금액']
                                non_michelin_avg_revenue = vs_result['비미슐랭_기준지표']['업장당_평균매출']
                                
                                comparison_df['주문금액_배수'] = comparison_df['평균_주문금액'] / non_michelin_avg_order
                                comparison_df['매출_배수'] = comparison_df['업장당_평균매출'] / non_michelin_avg_revenue
                                
                                fig4 = px.bar(comparison_df, x='등급', y=['주문금액_배수', '매출_배수'],
                                             title="비미슐랭 대비 배수 비교",
                                             barmode='group')
                                fig4.add_hline(y=1, line_dash="dash", line_color="red", 
                                              annotation_text="비미슐랭 기준선")
                                st.plotly_chart(fig4, use_container_width=True)
                            
                            # 상세 비교 테이블
                            st.subheader("📊 상세 비교 데이터")
                            st.dataframe(comparison_df, use_container_width=True)
                
                else:
                    st.error(vs_result['메시지'])
            
            st.divider()
            
            # 미슐랭 전체 개요
            st.subheader("📊 미슐랭 레스토랑 전체 개요")
            if st.button("전체 개요 분석", type="primary"):
                with st.spinner('미슐랭 레스토랑 전체 개요를 분석하고 있습니다...'):
                    overview_result = analyzer.analyze_michelin_overview()
                
                if overview_result['상태'] == '성공':
                    st.success("✅ 미슐랭 레스토랑 전체 개요 분석이 완료되었습니다!")
                    
                    # 전체 통계
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 매출", f"{overview_result['전체_매출']:,}원")
                    with col2:
                        st.metric("전체 구매량", f"{overview_result['전체_구매량']:,}개")
                    with col3:
                        st.metric("분류된 고객수", f"{overview_result['분류된_고객수']:,}개")
                    
                    # 등급별 데이터
                    st.subheader("등급별 상세 정보")
                    for grade, data in overview_result['등급별_데이터'].items():
                        with st.expander(f"🌟 {grade} 등급"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("레스토랑 수", f"{data['레스토랑_수']}개")
                            with col2:
                                st.metric("총 매출", f"{data['총_매출']:,}원")
                            with col3:
                                st.metric("총 구매량", f"{data['총_구매량']:,}개")
                            with col4:
                                st.metric("평균 매출", f"{data['평균_매출']:,.0f}원")
                            
                            st.write("**레스토랑 목록:**")
                            for restaurant in data['레스토랑_목록']:
                                st.write(f"- {restaurant}")
                else:
                    st.error(overview_result['메시지'])
            
            st.divider()
            
            # 등급별 상세 분석
            st.subheader("🔍 등급별 상세 분석")
            grade_options = {
                '3_STAR': '⭐⭐⭐ 3 STAR (밍글스)',
                '2_STAR': '⭐⭐ 2 STAR (알렌&컨티뉴움, 미토우, 스와니예, 알라프리마, 정식당)',
                '1_STAR': '⭐ 1 STAR (강민철 레스토랑, 라망시크레, 비채나, 빈호, 소설한남, 소울, 솔밤, 익스퀴진 에스콘디도, 체로컴플렉스, 익스퀴진)',
                'SELECTED': '🍽️ SELECTED (줄라이, 페리지, 보르고한남, 홍연, 알레즈, 류니끄, 구찌오스테리아, 소바쥬 산로, 본앤브레드, 트리드, 일 베키오, 쉐시몽, 물랑)'
            }
            
            selected_grade = st.selectbox("분석할 미슐랭 등급을 선택하세요:", 
                                        options=list(grade_options.keys()),
                                        format_func=lambda x: grade_options[x])
            
            if st.button("등급별 상세 분석 실행", type="primary"):
                with st.spinner(f'{selected_grade} 등급을 분석하고 있습니다...'):
                    result = analyzer.analyze_michelin_by_grade(selected_grade)
                
                if result['상태'] == '성공':
                    st.success(f"✅ {selected_grade} 등급 분석이 완료되었습니다!")
                    
                    # 기본 정보
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("레스토랑 수", f"{result['레스토랑_수']}개")
                    with col2:
                        st.metric("총 매출", f"{result['총_매출']:,}원")
                    with col3:
                        st.metric("총 구매량", f"{result['총_구매량']:,}개")
                    
                    # 레스토랑별 분석
                    if result['레스토랑_분석']:
                        st.subheader("🏪 레스토랑별 상세 분석")
                        
                        # 레스토랑별 매출 비교 차트
                        restaurant_data = []
                        for restaurant, analysis in result['레스토랑_분석'].items():
                            restaurant_data.append({
                                '레스토랑': restaurant,
                                '총_매출': analysis['총_매출'],
                                '총_구매량': analysis['총_구매량'],
                                '평균_주문금액': analysis['평균_주문금액']
                            })
                        
                        if restaurant_data:
                            restaurant_df = pd.DataFrame(restaurant_data)
                            
                            # 매출 비교 차트
                            fig = px.bar(restaurant_df, x='레스토랑', y='총_매출',
                                       title=f"{selected_grade} 등급 레스토랑별 총 매출")
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 구매량 비교 차트
                            fig2 = px.bar(restaurant_df, x='레스토랑', y='총_구매량',
                                        title=f"{selected_grade} 등급 레스토랑별 총 구매량")
                            fig2.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # 레스토랑별 상세 정보
                        for restaurant, analysis in result['레스토랑_분석'].items():
                            with st.expander(f"🏪 {restaurant} 상세 정보"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("총 매출", f"{analysis['총_매출']:,}원")
                                with col2:
                                    st.metric("총 구매량", f"{analysis['총_구매량']:,}개")
                                with col3:
                                    st.metric("구매 품목수", f"{analysis['구매_품목수']}개")
                                with col4:
                                    st.metric("거래 횟수", f"{analysis['거래_횟수']}회")
                                
                                st.metric("평균 주문금액", f"{analysis['평균_주문금액']:,.0f}원")
                                
                                if analysis['주요_품목']:
                                    st.write("**주요 구매 품목 TOP 5:**")
                                    for product, quantity in analysis['주요_품목'].items():
                                        st.write(f"- {product}: {quantity:,}개")
                    
                    # 월별 매출 추이
                    if result['월별_매출']:
                        st.subheader("📅 월별 매출 추이")
                        monthly_data = []
                        for month, sales in result['월별_매출'].items():
                            monthly_data.append({
                                '월': month,
                                '매출': sales
                            })
                        
                        if monthly_data:
                            monthly_df = pd.DataFrame(monthly_data)
                            monthly_df = monthly_df.sort_values('월')
                            
                            fig = px.line(monthly_df, x='월', y='매출',
                                        title=f"{selected_grade} 등급 월별 매출 추이",
                                        markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 인기 품목
                    if result['인기_품목']:
                        st.subheader("🔥 인기 품목 TOP 10")
                        products_df = pd.DataFrame(result['인기_품목'])
                        
                        if not products_df.empty:
                            # 2열 레이아웃으로 pie chart 표시
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 품목별 매출 pie chart
                                fig = px.pie(products_df.head(10), values='금액', names='상품',
                                           title=f"{selected_grade} 등급 인기 품목별 매출 비중")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 품목별 구매량 pie chart
                                fig2 = px.pie(products_df.head(10), values='수량', names='상품',
                                            title=f"{selected_grade} 등급 인기 품목별 구매량 비중")
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # 상세 테이블
                            st.dataframe(products_df, use_container_width=True)
                
                else:
                    st.error(result['메시지'])
            
            st.divider()
            
            # 등급간 비교 분석
            st.subheader("⚖️ 미슐랭 등급간 비교 분석")
            if st.button("등급간 비교 분석 실행", type="primary"):
                with st.spinner('미슐랭 등급간 비교를 분석하고 있습니다...'):
                    comparison_result = analyzer.analyze_michelin_comparison()
                
                if comparison_result['상태'] == '성공':
                    st.success("✅ 미슐랭 등급간 비교 분석이 완료되었습니다!")
                    
                    comparison_data = comparison_result['비교_데이터']
                    
                    # 비교 차트 데이터 준비
                    chart_data = []
                    for grade, data in comparison_data.items():
                        chart_data.append({
                            '등급': grade,
                            '총_매출': data['총_매출'],
                            '총_구매량': data['총_구매량'],
                            '평균_주문금액': data['평균_주문금액'],
                            '레스토랑당_평균매출': data['레스토랑당_평균매출'],
                            '품목_다양성': data['품목_다양성'],
                            '레스토랑_수': data['레스토랑_수']
                        })
                    
                    if chart_data:
                        chart_df = pd.DataFrame(chart_data)
                        
                        # 등급별 총 매출 비교
                        fig1 = px.bar(chart_df, x='등급', y='총_매출',
                                     title="미슐랭 등급별 총 매출 비교")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # 등급별 레스토랑당 평균 매출 비교
                        fig2 = px.bar(chart_df, x='등급', y='레스토랑당_평균매출',
                                     title="미슐랭 등급별 레스토랑당 평균 매출 비교")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # 등급별 평균 주문금액 비교
                        fig3 = px.bar(chart_df, x='등급', y='평균_주문금액',
                                     title="미슐랭 등급별 평균 주문금액 비교")
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # 등급별 품목 다양성 비교
                        fig4 = px.bar(chart_df, x='등급', y='품목_다양성',
                                     title="미슐랭 등급별 품목 다양성 비교")
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # 상세 비교 테이블
                        st.subheader("📊 상세 비교 데이터")
                        st.dataframe(chart_df, use_container_width=True)
                
                else:
                    st.error(comparison_result['메시지'])
            

        
        # 탭 7: 베이커리 & 디저트
        elif st.session_state.selected_tab == 6:
            st.markdown('<h2 class="sub-header">🧁 베이커리 & 디저트 분석</h2>', unsafe_allow_html=True)
            
            # 선택된 분석 기간 정보 표시
            if 'date_filtered_analyzer' in st.session_state:
                filtered_min, filtered_max = analyzer.get_date_range()
                if filtered_min and filtered_max:
                    st.info(f"🗓️ 현재 분석 기간: {filtered_min.date()} ~ {filtered_max.date()}")
                else:
                    st.info("🗓️ 선택된 기간으로 필터링된 데이터로 분석합니다.")
            else:
                st.info("🗓️ 전체 데이터 기간으로 분석합니다.")
            
            st.markdown("베이커리 & 디저트 레스토랑들의 특성과 매출 패턴을 분석합니다.")
            
            # 서브탭 구성
            subtab1, subtab2, subtab3 = st.tabs(["📊 전체 현황", "🏪 업체별 분석", "📈 업체간 비교"])
            
            # 서브탭 1: 전체 현황
            with subtab1:
                st.subheader("📊 베이커리 & 디저트 전체 현황")
                
                if st.button("전체 현황 분석 실행", type="primary", key="bakery_overview"):
                    with st.spinner('베이커리 & 디저트 현황을 분석하고 있습니다...'):
                        result = analyzer.analyze_bakery_overview()
                    
                    if result['상태'] == '성공':
                        st.success("✅ 베이커리 & 디저트 현황 분석이 완료되었습니다!")
                        
                        # 기본 지표
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 매출", f"{result['총매출']:,.0f}원")
                        with col2:
                            st.metric("총 구매량", f"{result['총구매량']:,.0f}개")
                        with col3:
                            st.metric("평균 주문금액", f"{result['평균주문금액']:,.0f}원")
                        with col4:
                            st.metric("업체 수", f"{result['업체수']:,}개")
                        
                        # 베이커리 업체 목록 표시
                        st.subheader("🏪 분석 대상 베이커리 & 디저트 업체")
                        if result['베이커리_업체목록']:
                            bakery_list_df = pd.DataFrame(result['베이커리_업체목록'], columns=['업체명'])
                            st.dataframe(bakery_list_df, use_container_width=True)
                        
                        # 월별 매출 추이
                        if result['월별매출']:
                            st.subheader("📅 월별 매출 추이")
                            monthly_df = pd.DataFrame.from_dict(result['월별매출'], orient='index', columns=['매출'])
                            monthly_df.index.name = '연월'
                            monthly_df = monthly_df.reset_index()
                            monthly_df['연월'] = monthly_df['연월'].astype(str)
                            monthly_df = monthly_df.sort_values('연월')
                            
                            fig = px.line(monthly_df, x='연월', y='매출',
                                        title="베이커리 & 디저트 월별 매출 추이", markers=True)
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 업체별 매출 분포
                        if result['업체별매출']:
                            st.subheader("🏪 업체별 매출 분포")
                            customer_sales_df = pd.DataFrame.from_dict(result['업체별매출'], orient='index')
                            customer_sales_df = customer_sales_df.sort_values('금액', ascending=False).head(10)
                            
                            fig = px.bar(customer_sales_df, x=customer_sales_df.index, y='금액',
                                       title="상위 10개 업체 매출")
                            fig.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(customer_sales_df, use_container_width=True)
                        
                        # 상품별 매출 분포
                        if result['상품별매출']:
                            st.subheader("🥧 인기 상품 분석")
                            product_sales_df = pd.DataFrame.from_dict(result['상품별매출'], orient='index')
                            product_sales_df = product_sales_df.sort_values('금액', ascending=False).head(15)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(product_sales_df.head(10), values='금액', names=product_sales_df.head(10).index,
                                           title="상위 10개 상품 매출 비중")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.write("**상위 15개 인기 상품**")
                                st.dataframe(product_sales_df[['금액', '수량', '고객명']], use_container_width=True)
                    else:
                        st.error(result['메시지'])
            
            # 서브탭 2: 업체별 분석
            with subtab2:
                st.subheader("🏪 베이커리 업체별 상세 분석")
                
                # 분석할 업체 선택
                bakery_keywords = [
                    "파리크라상 Passion5", "파리크라상 도곡점", "파리크라상(양재연구실)", "파리크라상 신세계백화점본점", "터치", "라뜰리에 이은", "노틀던", "파티세리 폰드", 
                    "앨리스 프로젝트", "카페꼼마", "문화시민 서울", "소나(SONA)",
                    "사색연희", "알디프", "클레어파티시에", "슬로운", "바 오쁘띠베르"
                ]
                
                selected_bakery = st.selectbox("분석할 베이커리 업체를 선택하세요:", bakery_keywords)
                
                if st.button("업체별 분석 실행", type="primary", key="bakery_by_store"):
                    with st.spinner(f'{selected_bakery} 업체를 분석하고 있습니다...'):
                        result = analyzer.analyze_bakery_by_store(selected_bakery)
                    
                    if result['상태'] == '성공':
                        st.success(f"✅ {selected_bakery} 업체 분석이 완료되었습니다!")
                        
                        # 매칭된 업체들 표시
                        if result['매칭_업체들']:
                            st.subheader(f"📋 {selected_bakery} 관련 업체 목록")
                            matching_df = pd.DataFrame(result['매칭_업체들'], columns=['업체명'])
                            st.dataframe(matching_df, use_container_width=True)
                        
                        # 기본 지표
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("총 매출", f"{result['총매출']:,.0f}원")
                        with col2:
                            st.metric("총 구매량", f"{result['총구매량']:,.0f}개")
                        with col3:
                            st.metric("평균 주문금액", f"{result['평균주문금액']:,.0f}원")
                        
                        # 상위 상품
                        if result['상위상품']:
                            st.subheader("🏆 인기 상품 TOP 10")
                        # 연월별 판매 추이 그래프 추가
                        if result['월별추이']:
                            st.subheader("📈 연월별 판매 추이")
                            monthly_trend_df = pd.DataFrame.from_dict(result['월별추이'], orient='index')
                            monthly_trend_df.index.name = '연월'
                            monthly_trend_df = monthly_trend_df.reset_index()
                            monthly_trend_df['연월'] = monthly_trend_df['연월'].astype(str)
                            monthly_trend_df = monthly_trend_df.sort_values('연월')
                            
                            # 매출과 구매량 추이를 함께 표시
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if '금액' in monthly_trend_df.columns:
                                    fig_revenue = px.line(monthly_trend_df, x='연월', y='금액',
                                                        title=f"{selected_bakery} 월별 매출 추이", markers=True)
                                    fig_revenue.update_layout(xaxis_tickangle=45)
                                    fig_revenue.update_traces(line_color='#1f77b4')
                                    st.plotly_chart(fig_revenue, use_container_width=True)
                            
                            with col2:
                                if '수량' in monthly_trend_df.columns:
                                    fig_quantity = px.line(monthly_trend_df, x='연월', y='수량',
                                                         title=f"{selected_bakery} 월별 구매량 추이", markers=True)
                                    fig_quantity.update_layout(xaxis_tickangle=45)
                                    fig_quantity.update_traces(line_color='#ff7f0e')
                                    st.plotly_chart(fig_quantity, use_container_width=True)
                            
                            # 월별 추이 데이터 테이블
                            st.subheader("📊 월별 상세 데이터")
                            display_trend_df = monthly_trend_df.copy()
                            if '금액' in display_trend_df.columns:
                                display_trend_df['금액'] = display_trend_df['금액'].apply(lambda x: f"{x:,.0f}원")
                            if '수량' in display_trend_df.columns:
                                display_trend_df['수량'] = display_trend_df['수량'].apply(lambda x: f"{x:,.0f}개")
                            st.dataframe(display_trend_df, use_container_width=True)
                        
                        # 지점별 연월별 추이 (여러 지점이 있는 경우)
                        if result.get('지점별월별추이') and len(result['매칭_업체들']) > 1:
                            st.subheader("🏪 지점별 연월별 매출 비교")
                            branch_trend_df = pd.DataFrame(result['지점별월별추이'])
                            
                            if not branch_trend_df.empty and '연월' in branch_trend_df.columns:
                                branch_trend_df['연월'] = branch_trend_df['연월'].astype(str)
                                branch_trend_df = branch_trend_df.sort_values('연월')
                                
                                # 지점별 매출 추이 라인 차트
                                fig_branch_trend = px.line(branch_trend_df, x='연월', y='금액', color='고객명',
                                                         title=f"{selected_bakery} 지점별 월별 매출 추이", markers=True)
                                fig_branch_trend.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig_branch_trend, use_container_width=True)
                                
                                # 지점별 구매량 추이 라인 차트
                                fig_branch_quantity = px.line(branch_trend_df, x='연월', y='수량', color='고객명',
                                                            title=f"{selected_bakery} 지점별 월별 구매량 추이", markers=True)
                                fig_branch_quantity.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig_branch_quantity, use_container_width=True)
                        
                        # 상위 상품 데이터 표시
                        if result['상위상품']:
                            top_products_df = pd.DataFrame.from_dict(result['상위상품'], orient='index')
                            st.dataframe(top_products_df, use_container_width=True)
                    else:
                        st.error(result['메시지'])
            
            # 서브탭 3: 업체간 비교
            with subtab3:
                st.subheader("📈 베이커리 업체간 비교 분석")
                
                if st.button("업체간 비교 분석 실행", type="primary", key="bakery_comparison"):
                    with st.spinner('베이커리 업체들을 비교 분석하고 있습니다...'):
                        result = analyzer.analyze_bakery_comparison()
                    
                    if result['상태'] == '성공':
                        st.success("✅ 베이커리 업체간 비교 분석이 완료되었습니다!")
                        
                        comparison_data = result['비교_데이터']
                        
                        if comparison_data:
                            # 비교 데이터를 DataFrame으로 변환
                            comparison_df = pd.DataFrame.from_dict(comparison_data, orient='index')
                            comparison_df = comparison_df.sort_values('총_매출', ascending=False)
                            
                            # 주요 지표 비교 차트
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.bar(comparison_df, x=comparison_df.index, y='총_매출',
                                           title="업체별 총 매출 비교")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.bar(comparison_df, x=comparison_df.index, y='평균_주문금액',
                                           title="업체별 평균 주문금액 비교")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # 상세 비교 테이블
                            st.subheader("📊 상세 비교 데이터")
                            display_df = comparison_df.copy()
                            display_df['총_매출'] = display_df['총_매출'].apply(lambda x: f"{x:,.0f}원")
                            display_df['총_구매량'] = display_df['총_구매량'].apply(lambda x: f"{x:,.0f}개")
                            display_df['평균_주문금액'] = display_df['평균_주문금액'].apply(lambda x: f"{x:,.0f}원")
                            display_df['지점당_평균매출'] = display_df['지점당_평균매출'].apply(lambda x: f"{x:,.0f}원")
                            
                            st.dataframe(display_df[['총_매출', '총_구매량', '평균_주문금액', '지점당_평균매출', '품목_다양성', '지점_수']], 
                                       use_container_width=True)
                        else:
                            st.warning("비교할 수 있는 베이커리 업체 데이터가 없습니다.")
                    else:
                        st.error(result['메시지'])
    
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {str(e)}")
        st.info("📁 다음 파일들이 필요합니다:")
        st.markdown("""
        - `merged_2023_2024_2025.xlsx` (판매 데이터)
        - `merged_returns_2024_2025.xlsx` (반품 데이터)
        """)
        
        # 파일 업로드 옵션 제공
        st.markdown("### 📋 파일 업로드")
        sales_file_upload = st.file_uploader("판매 데이터 업로드", type=['csv', 'xlsx'])
        refund_file_upload = st.file_uploader("반품 데이터 업로드 (선택사항)", type=['csv', 'xlsx'])
        
        if sales_file_upload is not None:
            try:
                # 업로드된 파일로 데이터 로드
                if sales_file_upload.name.endswith('.csv'):
                    sales_data = pd.read_csv(sales_file_upload)
                else:
                    sales_data = pd.read_excel(sales_file_upload)
                
                refund_data = None
                if refund_file_upload is not None:
                    if refund_file_upload.name.endswith('.csv'):
                        refund_data = pd.read_csv(refund_file_upload)
                    else:
                        refund_data = pd.read_excel(refund_file_upload)
                
                # 날짜 컬럼 처리
                if '날짜' in sales_data.columns:
                    sales_data['날짜'] = pd.to_datetime(sales_data['날짜'], errors='coerce')
                
                # 분석 시스템 초기화
                with st.spinner('분석 시스템을 초기화하고 있습니다...'):
                    analyzer = MicrogreenAnalysisSystem(sales_data, refund_data)
                
                st.success("✅ 업로드된 데이터로 분석 시스템이 초기화되었습니다!")
                st.rerun()  # 페이지 새로고침
                
            except Exception as upload_error:
                st.error(f"업로드된 데이터 처리 중 오류가 발생했습니다: {str(upload_error)}")
    
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        st.info("데이터 형식을 확인해주세요.")

if __name__ == "__main__":
    main() 