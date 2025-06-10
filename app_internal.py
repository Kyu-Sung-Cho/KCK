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
        self.sales_data = sales_data
        self.refund_data = refund_data
        self.customer_product_matrix = None
        
        # 데이터 전처리
        self.preprocess_data()

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
        """다이닝 VIP 지표 분석 - 전년도 매출 Top 5 업체"""
        try:
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
            
            # 호텔 고객 제외 (다이닝만)
            customer_categories = self.get_customer_categories()
            dining_customers = customer_categories['일반']
            valid_sales = valid_sales[valid_sales['고객명'].isin(dining_customers)]
            
            # 전년도 매출 Top 5 계산
            customer_revenue = valid_sales.groupby('고객명')['금액'].sum().sort_values(ascending=False)
            top5_customers = customer_revenue.head(5).index.tolist()
            
            # Top 5 고객들의 월별 매출 추이
            top5_data = valid_sales[valid_sales['고객명'].isin(top5_customers)]
            
            # 월별 매출 집계
            monthly_revenue = {}
            product_revenue = {}
            yearmonth_product_data = {}
            
            for customer in top5_customers:
                customer_data = top5_data[top5_data['고객명'] == customer]
                
                # 월별 매출
                monthly_data = customer_data.groupby('월')['금액'].sum().to_dict()
                monthly_revenue[customer] = monthly_data
                
                # 품목별 매출
                product_data = customer_data.groupby('상품')['금액'].sum().sort_values(ascending=False).head(10).to_dict()
                product_revenue[customer] = product_data
                
                # 연월별 상품 구매 데이터
                yearmonth_products = customer_data.groupby(['연월', '상품']).agg({
                    '수량': 'sum',
                    '금액': 'sum'
                }).reset_index()
                yearmonth_product_data[customer] = yearmonth_products
            
            return {
                '상태': '성공',
                'top5_customers': top5_customers,
                'customer_total_revenue': customer_revenue.head(5).to_dict(),
                'monthly_revenue': monthly_revenue,
                
                'yearmonth_product_data': yearmonth_product_data
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"다이닝 VIP 분석 중 오류 발생: {str(e)}"
            }

    def analyze_hotel_vip_metrics(self):
        """호텔 VIP 지표 분석 - 지정된 호텔 5곳"""
        try:
            # 지정된 호텔 리스트
            target_hotels = ['포시즌스', '소피텔', '인스파이어인티그레이티드리조트', '조선팰리스', '웨스틴조선']
            
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
            
            # 날짜 처리
            valid_sales['날짜'] = pd.to_datetime(valid_sales['날짜'], errors='coerce')
            valid_sales = valid_sales.dropna(subset=['날짜'])
            valid_sales['연월'] = valid_sales['날짜'].dt.strftime('%Y-%m')
            valid_sales['월'] = valid_sales['날짜'].dt.month
            
            # 지정된 호텔들 찾기 (부분 문자열 매칭)
            found_hotels = []
            hotel_data = {}
            
            for hotel in target_hotels:
                # 고객명에 호텔명이 포함된 고객들 찾기
                matching_customers = valid_sales[valid_sales['고객명'].str.contains(hotel, case=False, na=False)]['고객명'].unique()
                
                if len(matching_customers) > 0:
                    found_hotels.append(hotel)
                    hotel_sales = valid_sales[valid_sales['고객명'].isin(matching_customers)]
                    
                    # 월별 매출
                    monthly_revenue = hotel_sales.groupby('월')['금액'].sum().to_dict()
                    
                    # 품목별 매출
                    product_revenue = hotel_sales.groupby('상품')['금액'].sum().sort_values(ascending=False).head(10).to_dict()
                    
                    # 연월별 상품 구매 데이터
                    yearmonth_products = hotel_sales.groupby(['연월', '상품']).agg({
                        '수량': 'sum',
                        '금액': 'sum'
                    }).reset_index()
                    
                    # 총 매출
                    total_revenue = hotel_sales['금액'].sum()
                    
                    hotel_data[hotel] = {
                        'customers': matching_customers.tolist(),
                        'revenue_data': {'total_revenue': total_revenue, 'product_revenue': product_revenue},
                        'monthly_revenue': monthly_revenue,
                        
                        'yearmonth_product_data': yearmonth_products
                    }
            
            return {
                '상태': '성공',
                'found_hotels': found_hotels,
                'hotel_data': hotel_data
            }
            
        except Exception as e:
            return {
                '상태': '실패',
                '메시지': f"호텔 VIP 분석 중 오류 발생: {str(e)}"
            }

    def analyze_banquet_metrics(self):
        """BANQUET 지표 분석 - 비고 컬럼에 banquet 관련 키워드가 있는 데이터"""
        try:
            # 비고 컬럼이 있는지 확인
            if '비고' not in self.sales_data.columns:
                return {
                    '상태': '실패',
                    '메시지': "비고 컬럼이 없어 BANQUET 분석을 수행할 수 없습니다."
                }
            
            # 유효한 데이터 필터링
            valid_sales = self.sales_data.copy()
            
            # 비고 컬럼에서 banquet 관련 데이터 필터링 (대소문자 구분 없이)
            banquet_data = valid_sales[
                valid_sales['비고'].str.contains('banquet', case=False, na=False) |
                valid_sales['비고'].str.contains('banquets', case=False, na=False)
            ]
            
            if banquet_data.empty:
                return {
                    '상태': '실패',
                    '메시지': "BANQUET 데이터가 없습니다."
                }
            
            # 날짜 처리
            banquet_data['날짜'] = pd.to_datetime(banquet_data['날짜'], errors='coerce')
            banquet_data = banquet_data.dropna(subset=['날짜'])
            banquet_data['연월'] = banquet_data['날짜'].dt.strftime('%Y-%m')
            banquet_data['월'] = banquet_data['날짜'].dt.month
            
            # 호텔별 BANQUET 분석
            banquet_customers = banquet_data['고객명'].unique()
            banquet_data_result = {}
            
            # 각 고객별로 BANQUET 데이터 분석
            for customer in banquet_customers:
                customer_banquet = banquet_data[banquet_data['고객명'] == customer]
                
                # 품목별 판매량
                product_sales = customer_banquet.groupby('상품')['수량'].sum().sort_values(ascending=False)
                
                # 월별 판매 추이
                monthly_sales = customer_banquet.groupby('월')['수량'].sum().to_dict()
                
                # 연월별 상품 구매 데이터
                yearmonth_products = customer_banquet.groupby(['연월', '상품']).agg({
                    '수량': 'sum'
                }).reset_index()
                
                # 금액 정보가 있는 경우
                revenue_data = {}
                monthly_revenue = {}
                if '금액' in customer_banquet.columns:
                    customer_banquet['금액'] = pd.to_numeric(customer_banquet['금액'], errors='coerce')
                    revenue_data = {
                        'total_revenue': customer_banquet['금액'].sum(),
                        'product_revenue': customer_banquet.groupby('상품')['금액'].sum().sort_values(ascending=False).to_dict()
                    }
                    monthly_revenue = customer_banquet.groupby('월')['금액'].sum().to_dict()
                    
                    # 연월별 상품 구매 데이터에 금액 추가
                    yearmonth_products_with_amount = customer_banquet.groupby(['연월', '상품']).agg({
                        '수량': 'sum',
                        '금액': 'sum'
                    }).reset_index()
                    yearmonth_products = yearmonth_products_with_amount
                
                banquet_data_result[customer] = {
                    'product_sales': product_sales.to_dict(),
                    'monthly_sales': monthly_sales,
                    'monthly_revenue': monthly_revenue,
                    'total_quantity': customer_banquet['수량'].sum(),
                    'revenue_data': revenue_data,
                    'yearmonth_product_data': yearmonth_products
                }
            
            # 전체 BANQUET 품목별 분석
            total_product_sales = banquet_data.groupby('상품')['수량'].sum().sort_values(ascending=False)
            
            return {
                '상태': '성공',
                'found_banquet_customers': banquet_customers.tolist(),
                'banquet_data': banquet_data_result,
                'total_product_sales': total_product_sales.to_dict(),
                'total_banquet_quantity': banquet_data['수량'].sum()
            }
            
        except Exception as e:
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
            non_michelin_top_products = non_michelin_products.head(10)['상품'].tolist()
            
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
                        common_products = []
                        
                        for product in michelin_top_products:
                            if product in non_michelin_top_products:
                                common_products.append(product)
                            else:
                                unique_products.append(product)
                        
                        if unique_products:
                            unique_features.append(f"독특한 선호품목: {', '.join(unique_products[:3])}")
                        
                        comparison_results[grade] = {
                            '미슐랭_지표': michelin_stats,
                            '비교_배수': comparison_analysis,
                            '차별화_특징': unique_features,
                            '독특한_품목': unique_products,
                            '공통_품목': common_products,
                            '인기_품목_TOP5': michelin_top_products[:5]
                        }
            
            return {
                '상태': '성공',
                '비미슐랭_기준지표': non_michelin_stats,
                '비미슐랭_인기품목': non_michelin_top_products,
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
                monthly_trend = store_data.groupby('연월').agg({
                    '금액': 'sum',
                    '수량': 'sum'
                })
            else:
                monthly_trend = pd.DataFrame()
            
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
                '월별추이': monthly_trend.to_dict('index') if not monthly_trend.empty else {},
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

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">📊 마이크로그린 관리자 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">상품 분석, 고객 분석, RFM 세분화를 통한 비즈니스 인사이트</p>', unsafe_allow_html=True)
    
    # 현재 디렉토리의 파일들을 확인
    sales_file = "merged_2023_2024_2025.xlsx"
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
        
        # 메인 탭 구성
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 상품 분석", 
            "👥 업체 분석", 
            "🏢 고객관리",
            "💰 매출 지표",
            "📊 매출분석",
            "⭐ 미슐랭 분석",
            "🧁 베이커리 & 디저트"
        ])
        
        # 탭 1: 상품 분석
        with tab1:
            st.markdown('<h2 class="sub-header">📈 상품 분석</h2>', unsafe_allow_html=True)
            
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
                selected_product = st.selectbox("분석할 상품을 선택하세요:", products)
                
                if analysis_type == "전체 상품 분석":
                    button_text = "상품 분석 실행"
                    button_key = "product_analysis_full"
                else:
                    button_text = "상품 분석 실행 (포시즌스 호텔 제외)"
                    button_key = "product_analysis_exclude_fourseasons"
                
                if st.button(button_text, type="primary", key=button_key):
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
                            
                            seasonal_df = pd.DataFrame.from_dict(result['계절별_판매'], orient='index', columns=['판매량'])
                            seasonal_df.index.name = '계절'
                            seasonal_df = seasonal_df.reset_index()
                            
                            if not seasonal_df.empty and seasonal_df['판매량'].sum() > 0:
                                fig = px.pie(seasonal_df, values='판매량', names='계절',
                                           title="계절별 판매 비중")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("계절별 판매 데이터가 없습니다.")
                    else:
                        st.error(result['메시지'])
            else:
                st.warning("분석 가능한 상품이 없습니다.")
        
        # 탭 2: 업체 분석
        with tab2:
            st.markdown('<h2 class="sub-header">👥 업체 분석</h2>', unsafe_allow_html=True)
            
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
                            
                            seasonal_df = pd.DataFrame.from_dict(result['계절별_선호도'], orient='index', columns=['구매량'])
                            seasonal_df.index.name = '계절'
                            seasonal_df = seasonal_df.reset_index()
                            
                            if not seasonal_df.empty and seasonal_df['구매량'].sum() > 0:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(seasonal_df, x='계절', y='구매량',
                                               title="계절별 구매량",
                                               color='구매량',
                                               color_continuous_scale='Viridis')
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig2 = px.pie(seasonal_df, values='구매량', names='계절',
                                                title="계절별 구매 비중")
                                    st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("계절별 구매 데이터가 없습니다.")
                        
                        # 분기별 선호도
                        if result['분기별_선호도'] and any(v > 0 for v in result['분기별_선호도'].values()):
                            st.subheader("📊 분기별 구매 패턴")
                            
                            quarterly_df = pd.DataFrame.from_dict(result['분기별_선호도'], orient='index', columns=['구매량'])
                            quarterly_df.index.name = '분기'
                            quarterly_df = quarterly_df.reset_index()
                            
                            if not quarterly_df.empty and quarterly_df['구매량'].sum() > 0:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(quarterly_df, x='분기', y='구매량',
                                               title="분기별 구매량",
                                               color='구매량',
                                               color_continuous_scale='Blues')
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig2 = px.pie(quarterly_df, values='구매량', names='분기',
                                                title="분기별 구매 비중")
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
        with tab3:
            st.markdown('<h2 class="sub-header">🏢 고객관리</h2>', unsafe_allow_html=True)
            
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
        with tab4:
            st.markdown('<h2 class="sub-header">💰 매출 지표</h2>', unsafe_allow_html=True)
            
            # 분석 유형 선택
            analysis_type = st.radio(
                "분석 유형을 선택하세요:",
                ["기본 매출 지표", "업체 특성 분석"],
                horizontal=True
            )
            
            if analysis_type == "기본 매출 지표":
                # 매출 지표 카테고리 선택
                metric_category = st.selectbox(
                    "분석할 매출 지표를 선택하세요:",
                    ["다이닝 VIP 지표", "호텔 VIP 지표", "BANQUET 지표"]
                )
                
                if st.button("매출 지표 분석 실행", type="primary"):
                    with st.spinner(f'{metric_category} 분석을 수행하고 있습니다...'):
                        if metric_category == "다이닝 VIP 지표":
                            result = analyzer.analyze_dining_vip_metrics()
                        elif metric_category == "호텔 VIP 지표":
                            result = analyzer.analyze_hotel_vip_metrics()
                        else:  # BANQUET 지표
                            result = analyzer.analyze_banquet_metrics()
                    
                    if result['상태'] == '성공':
                        st.success(f"✅ {metric_category} 분석이 완료되었습니다!")
                        
                        # 다이닝 VIP 지표 결과 표시
                        if metric_category == "다이닝 VIP 지표":
                            st.subheader("🍽️ 다이닝 VIP 매출 Top 5")
                            
                            # Top 5 고객 총 매출
                            if result['customer_total_revenue']:
                                revenue_df = pd.DataFrame.from_dict(result['customer_total_revenue'], orient='index', columns=['총매출'])
                                revenue_df.index.name = '고객명'
                                revenue_df = revenue_df.reset_index()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = px.bar(revenue_df, x='고객명', y='총매출',
                                               title="다이닝 VIP Top 5 총 매출")
                                    fig.update_layout(xaxis_tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    fig_pie = px.pie(revenue_df, values='총매출', names='고객명',
                                                   title="다이닝 VIP 매출 비중")
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                st.dataframe(revenue_df, use_container_width=True)
                            
                            # 월별 매출 추이
                            if result['monthly_revenue']:
                                st.subheader("📅 Top 5 고객 월별 매출 추이")
                                
                                monthly_data = []
                                for customer, monthly_sales in result['monthly_revenue'].items():
                                    for month, amount in monthly_sales.items():
                                        monthly_data.append({
                                            '고객명': customer,
                                            '월': month,
                                            '매출': amount
                                        })
                                
                                if monthly_data:
                                    monthly_df = pd.DataFrame(monthly_data)
                                    
                                    fig = px.line(monthly_df, x='월', y='매출', color='고객명',
                                                title="월별 매출 추이", markers=True)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # 품목별 매출
                            if result['product_revenue']:
                                st.subheader("🛒 고객별 주요 품목 매출")
                                
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
                                            
                                            # 데이터 테이블
                                            st.dataframe(top_products, use_container_width=True)
                        
                        # 호텔 VIP 지표 결과 표시
                        elif metric_category == "호텔 VIP 지표":
                            st.subheader("🏨 호텔 VIP 매출 분석")
                            
                            if result['found_hotels']:
                                st.info(f"발견된 호텔: {', '.join(result['found_hotels'])}")
                                
                                # 호텔별 총 매출
                                hotel_revenue_data = []
                                for hotel, data in result['hotel_data'].items():
                                    hotel_revenue_data.append({
                                        '호텔': hotel,
                                        '총매출': data['revenue_data'].get('total_revenue', 0) if data['revenue_data'] else 0,
                                        '고객수': len(data['customers'])
                                    })
                                
                                if hotel_revenue_data:
                                    hotel_df = pd.DataFrame(hotel_revenue_data)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        fig = px.bar(hotel_df, x='호텔', y='총매출',
                                                   title="호텔별 총 매출")
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col2:
                                        fig = px.bar(hotel_df, x='호텔', y='고객수',
                                                   title="호텔별 고객 수")
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col3:
                                        fig_pie = px.pie(hotel_df, values='총매출', names='호텔',
                                                       title="호텔별 매출 비중")
                                        st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                    st.dataframe(hotel_df, use_container_width=True)
                                
                                # 각 호텔별 상세 분석
                                for hotel, data in result['hotel_data'].items():
                                    with st.expander(f"{hotel} 상세 분석"):
                                        st.write(f"**고객명:** {', '.join(data['customers'])}")
                                        st.metric("총 매출", f"{data['revenue_data'].get('total_revenue', 0) if data['revenue_data'] else 0:,}원")
                                        
                                        # 월별 매출
                                        if data['monthly_revenue']:
                                            monthly_df = pd.DataFrame.from_dict(data['monthly_revenue'], orient='index', columns=['매출'])
                                            monthly_df.index.name = '월'
                                            monthly_df = monthly_df.reset_index()
                                            
                                            fig = px.line(monthly_df, x='월', y='매출',
                                                        title=f"{hotel} 월별 매출 추이", markers=True)
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # 품목별 매출
                                        if data['revenue_data'] and data['revenue_data'].get('product_revenue'):
                                            product_df = pd.DataFrame.from_dict(data['revenue_data'].get('product_revenue', {}) if data['revenue_data'] else {}, orient='index', columns=['매출'])
                                            product_df.index.name = '상품명'
                                            product_df = product_df.reset_index()
                                            
                                            # 상위 10개 품목 선택
                                            top_products = product_df.head(10)
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                fig_pie = px.pie(top_products, values='매출', names='상품명',
                                                               title=f"{hotel} Top 10 품목 매출 비중")
                                                st.plotly_chart(fig_pie, use_container_width=True)
                                            
                                            with col2:
                                                fig_bar = px.bar(top_products, x='상품명', y='매출',
                                                               title=f"{hotel} Top 10 품목 매출")
                                                fig_bar.update_layout(xaxis_tickangle=45)
                                                st.plotly_chart(fig_bar, use_container_width=True)
                                            
                                            # 데이터 테이블
                                            st.dataframe(top_products, use_container_width=True)
                            else:
                                st.warning("호텔 고객 데이터를 찾을 수 없습니다.")
                        
                        # BANQUET 지표 결과 표시
                        else:  # BANQUET 지표
                            st.subheader("🎉 BANQUET 매출 분석")
                            
                            if result['found_banquet_customers']:
                                st.info(f"발견된 BANQUET 고객: {', '.join(result['found_banquet_customers'])}")
                                
                                # BANQUET 고객별 총 매출
                                banquet_revenue_data = []
                                for customer, data in result['banquet_data'].items():
                                    banquet_revenue_data.append({
                                        '고객명': customer,
                                        '총매출': data['revenue_data'].get('total_revenue', 0) if data['revenue_data'] else 0
                                    })
                                
                                if banquet_revenue_data:
                                    banquet_df = pd.DataFrame(banquet_revenue_data)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig = px.bar(banquet_df, x='고객명', y='총매출',
                                                   title="BANQUET 고객별 총 매출")
                                        fig.update_layout(xaxis_tickangle=45)
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col2:
                                        fig_pie = px.pie(banquet_df, values='총매출', names='고객명',
                                                       title="BANQUET 매출 비중")
                                        st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                    st.dataframe(banquet_df, use_container_width=True)
                                
                                # 각 BANQUET 고객별 상세 분석
                                for customer, data in result['banquet_data'].items():
                                    with st.expander(f"{customer} 상세 분석"):
                                        st.metric("총 매출", f"{data['revenue_data'].get('total_revenue', 0) if data['revenue_data'] else 0:,}원")
                                        
                                        # 월별 매출
                                        if data['monthly_revenue']:
                                            monthly_df = pd.DataFrame.from_dict(data['monthly_revenue'], orient='index', columns=['매출'])
                                            monthly_df.index.name = '월'
                                            monthly_df = monthly_df.reset_index()
                                            
                                            fig = px.line(monthly_df, x='월', y='매출',
                                                        title=f"{customer} 월별 매출 추이", markers=True)
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # 품목별 매출
                                        if data['revenue_data'] and data['revenue_data'].get('product_revenue'):
                                            product_df = pd.DataFrame.from_dict(data['revenue_data'].get('product_revenue', {}) if data['revenue_data'] else {}, orient='index', columns=['매출'])
                                            product_df.index.name = '상품명'
                                            product_df = product_df.reset_index()
                                            
                                            # 상위 10개 품목 선택
                                            top_products = product_df.head(10)
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                fig_pie = px.pie(top_products, values='매출', names='상품명',
                                                               title=f"{customer} Top 10 품목 매출 비중")
                                                st.plotly_chart(fig_pie, use_container_width=True)
                                            
                                            with col2:
                                                fig_bar = px.bar(top_products, x='상품명', y='매출',
                                                               title=f"{customer} Top 10 품목 매출")
                                                fig_bar.update_layout(xaxis_tickangle=45)
                                                st.plotly_chart(fig_bar, use_container_width=True)
                                            
                                            # 데이터 테이블
                                            st.dataframe(top_products, use_container_width=True)
                            else:
                                st.warning("BANQUET 고객 데이터를 찾을 수 없습니다.")
                    else:
                        st.error(result['메시지'])
            
            elif analysis_type == "업체 특성 분석":
                st.subheader("🏢 업체 특성 분석")
                st.markdown("업체별 구매 패턴, 품목 다양성, 고객 등급 등을 종합적으로 분석합니다.")
                
                # 세션 상태 초기화
                if 'company_analysis_result' not in st.session_state:
                    st.session_state.company_analysis_result = None
                
                if st.button("업체 특성 분석 실행", type="primary"):
                    with st.spinner('업체 특성을 분석하고 있습니다...'):
                        result = analyzer.analyze_customer_characteristics()
                    
                    if result['상태'] == '성공':
                        # 결과를 세션 상태에 저장
                        st.session_state.company_analysis_result = result
                        st.success("✅ 업체 특성 분석이 완료되었습니다!")
                    else:
                        st.error(result['메시지'])
                
                # 분석 결과가 있을 때만 표시
                if st.session_state.company_analysis_result is not None:
                    result = st.session_state.company_analysis_result
                    
                    # 기본 통계
                    st.subheader("📊 업체 특성 분포")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("구매 패턴 분포")
                        pattern_dist = result['구매패턴분포']
                        fig = px.pie(values=pattern_dist.values, names=pattern_dist.index,
                                   title="구매 패턴별 업체 분포")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("품목 다양성 분포")
                        diversity_dist = result['품목다양성분포']
                        fig = px.pie(values=diversity_dist.values, names=diversity_dist.index,
                                   title="품목 다양성별 업체 분포")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        st.subheader("고객 등급 분포")
                        grade_dist = result['고객등급분포']
                        fig = px.pie(values=grade_dist.values, names=grade_dist.index,
                                   title="고객 등급별 업체 분포")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 상관관계 분석
                    st.subheader("📈 업체 특성 상관관계 분석")
                    
                    numeric_data = result['고객특성데이터'][['총매출', '거래횟수', '구매품목수', '활성도점수']]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 활성도 점수 vs 총 매출
                        fig = px.scatter(result['고객특성데이터'], x='활성도점수', y='총매출',
                                       color='고객등급', size='구매품목수',
                                       hover_data=['고객명'],
                                       title="활성도 점수 vs 총 매출")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 구매품목수 vs 총매출
                        fig = px.scatter(result['고객특성데이터'], x='구매품목수', y='총매출',
                                       color='품목다양성', size='거래횟수',
                                       hover_data=['고객명'],
                                       title="구매 품목 수 vs 총 매출")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 필터링 옵션 - 세션 상태로 관리
                    st.subheader("🔍 업체 특성별 필터링")
                    
                    # 세션 상태 키 초기화
                    if 'selected_pattern' not in st.session_state:
                        st.session_state.selected_pattern = '전체'
                    if 'selected_diversity' not in st.session_state:
                        st.session_state.selected_diversity = '전체'
                    if 'selected_grade' not in st.session_state:
                        st.session_state.selected_grade = '전체'
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pattern_options = ['전체'] + list(result['구매패턴분포'].index)
                        pattern_index = 0
                        if st.session_state.selected_pattern in pattern_options:
                            pattern_index = pattern_options.index(st.session_state.selected_pattern)
                        
                        selected_pattern = st.selectbox("구매 패턴", pattern_options,
                                                      key='pattern_filter',
                                                      index=pattern_index)
                        st.session_state.selected_pattern = selected_pattern
                    
                    with col2:
                        diversity_options = ['전체'] + list(result['품목다양성분포'].index)
                        diversity_index = 0
                        if st.session_state.selected_diversity in diversity_options:
                            diversity_index = diversity_options.index(st.session_state.selected_diversity)
                        
                        selected_diversity = st.selectbox("품목 다양성", diversity_options,
                                                        key='diversity_filter',
                                                        index=diversity_index)
                        st.session_state.selected_diversity = selected_diversity
                    
                    with col3:
                        grade_options = ['전체'] + list(result['고객등급분포'].index)
                        grade_index = 0
                        if st.session_state.selected_grade in grade_options:
                            grade_index = grade_options.index(st.session_state.selected_grade)
                        
                        selected_grade = st.selectbox("고객 등급", grade_options,
                                                    key='grade_filter',
                                                    index=grade_index)
                        st.session_state.selected_grade = selected_grade
                    
                    # 필터링된 데이터
                    filtered_data = result['고객특성데이터'].copy()
                    
                    if selected_pattern != '전체':
                        filtered_data = filtered_data[filtered_data['구매패턴'] == selected_pattern]
                    
                    if selected_diversity != '전체':
                        filtered_data = filtered_data[filtered_data['품목다양성'] == selected_diversity]
                    
                    if selected_grade != '전체':
                        filtered_data = filtered_data[filtered_data['고객등급'] == selected_grade]
                    
                    st.info(f"필터링된 업체 수: {len(filtered_data)}개")
                    
                    # 상위 업체 분석
                    if not filtered_data.empty:
                        st.subheader("🏆 상위 업체 분석")
                        
                        tab_top1, tab_top2, tab_top3 = st.tabs(["매출 상위", "활성도 상위", "다양성 상위"])
                        
                        with tab_top1:
                            top_revenue = filtered_data.nlargest(10, '총매출')[['고객명', '총매출', '고객등급', '활성도점수']]
                            st.dataframe(top_revenue, use_container_width=True)
                        
                        with tab_top2:
                            top_activity = filtered_data.nlargest(10, '활성도점수')[['고객명', '활성도점수', '총매출', '거래횟수', '구매품목수']]
                            st.dataframe(top_activity, use_container_width=True)
                        
                        with tab_top3:
                            top_diversity = filtered_data.nlargest(10, '구매품목수')[['고객명', '구매품목수', '총매출', '품목다양성', '고객등급']]
                            st.dataframe(top_diversity, use_container_width=True)
                        
                        # 전체 데이터 테이블
                        st.subheader("📋 전체 업체 특성 데이터")
                        st.dataframe(filtered_data, use_container_width=True)
        
        # 탭 5: 매출분석
        with tab5:
            st.markdown('<h2 class="sub-header">📊 매출분석</h2>', unsafe_allow_html=True)
            
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
        with tab6:
            st.markdown('<h2 class="sub-header">⭐ 미슐랭 분석</h2>', unsafe_allow_html=True)
            
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
            
            # 미슐랭 등급별 vs 비미슐랭 업장 특징 비교 분석
            st.divider()
            st.subheader("🆚 미슐랭 vs 비미슐랭 업장 특징 비교")
            if st.button("미슐랭 vs 비미슐랭 비교 분석 실행", type="primary"):
                with st.spinner('미슐랭 vs 비미슐랭 비교 분석을 수행하고 있습니다...'):
                    vs_result = analyzer.analyze_michelin_vs_non_michelin()
                
                if vs_result['상태'] == '성공':
                    st.success("✅ 미슐랭 vs 비미슐랭 비교 분석이 완료되었습니다!")
                    
                    # 비미슐랭 기준 지표 표시
                    st.subheader("📊 비미슐랭 업장 기준 지표")
                    non_michelin_stats = vs_result['비미슐랭_기준지표']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("업장 수", f"{non_michelin_stats['업장_수']:,}개")
                    with col2:
                        st.metric("총 매출", f"{non_michelin_stats['총_매출']:,.0f}원")
                    with col3:
                        st.metric("평균 주문금액", f"{non_michelin_stats['평균_주문금액']:,.0f}원")
                    with col4:
                        st.metric("업장당 평균매출", f"{non_michelin_stats['업장당_평균매출']:,.0f}원")
                    
                    # 등급별 비교 결과
                    for grade, comparison in vs_result['등급별_비교'].items():
                        st.subheader(f"🌟 {grade} 등급 vs 비미슐랭 비교")
                        
                        # 차별화 특징 표시
                        if comparison['차별화_특징']:
                            st.info("**주요 차별화 특징:**")
                            for feature in comparison['차별화_특징']:
                                st.write(f"• {feature}")
                        
                        # 비교 배수 차트
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 배수 비교 바 차트
                            comparison_metrics = comparison['비교_배수']
                            metrics_data = []
                            for metric, value in comparison_metrics.items():
                                metrics_data.append({
                                    '지표': metric.replace('_', ' '),
                                    '배수': value
                                })
                            
                            if metrics_data:
                                metrics_df = pd.DataFrame(metrics_data)
                                fig = px.bar(metrics_df, x='지표', y='배수',
                                           title=f"{grade} vs 비미슐랭 비교 배수")
                                fig.add_hline(y=1, line_dash="dash", line_color="red", 
                                            annotation_text="비미슐랭 기준선")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 지표 비교 테이블
                            michelin_stats = comparison['미슐랭_지표']
                            comparison_table = []
                            
                            for key in ['총_매출', '평균_주문금액', '업장당_평균매출', '평균_거래횟수']:
                                if key in michelin_stats and key.replace('업장당', '업장당') in non_michelin_stats:
                                    non_key = key.replace('업장당', '업장당')
                                    comparison_table.append({
                                        '지표': key.replace('_', ' '),
                                        f'{grade}': f"{michelin_stats[key]:,.0f}",
                                        '비미슐랭': f"{non_michelin_stats[non_key]:,.0f}",
                                        '배수': f"{comparison['비교_배수'].get(key+'_배수', 0):.1f}x"
                                    })
                            
                            if comparison_table:
                                comparison_df = pd.DataFrame(comparison_table)
                                st.dataframe(comparison_df, use_container_width=True)
                        
                        # 품목 차이 분석
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if comparison['독특한_품목']:
                                st.write("**독특한 선호 품목:**")
                                for i, product in enumerate(comparison['독특한_품목'][:5], 1):
                                    st.write(f"{i}. {product}")
                        
                        with col2:
                            if comparison['공통_품목']:
                                st.write("**비미슐랭과 공통 품목:**")
                                for i, product in enumerate(comparison['공통_품목'][:5], 1):
                                    st.write(f"{i}. {product}")
                        
                        # 인기 품목 TOP 5
                        if comparison['인기_품목_TOP5']:
                            st.write(f"**{grade} 등급 인기 품목 TOP 5:**")
                            for i, product in enumerate(comparison['인기_품목_TOP5'], 1):
                                st.write(f"{i}. {product}")
                        
                        st.divider()
                    
                    # 전체 미슐랭 vs 비미슐랭 종합 비교 차트
                    st.subheader("📈 종합 비교 차트")
                    
                    # 모든 등급의 비교 배수 데이터 수집
                    all_comparison_data = []
                    for grade, comparison in vs_result['등급별_비교'].items():
                        for metric, value in comparison['비교_배수'].items():
                            all_comparison_data.append({
                                '등급': grade,
                                '지표': metric.replace('_배수', '').replace('_', ' '),
                                '배수': value
                            })
                    
                    if all_comparison_data:
                        all_comparison_df = pd.DataFrame(all_comparison_data)
                        
                        # 등급별 지표 비교 히트맵
                        pivot_data = all_comparison_df.pivot(index='등급', columns='지표', values='배수')
                        fig = px.imshow(pivot_data.values,
                                      x=pivot_data.columns,
                                      y=pivot_data.index,
                                      title="미슐랭 등급별 vs 비미슐랭 비교 히트맵 (배수)",
                                      color_continuous_scale='RdYlBu_r',
                                      aspect="auto")
                        
                        # 텍스트 주석 추가
                        for i in range(len(pivot_data.index)):
                            for j in range(len(pivot_data.columns)):
                                fig.add_annotation(
                                    x=j, y=i,
                                    text=f"{pivot_data.iloc[i, j]:.1f}x",
                                    showarrow=False,
                                    font=dict(color="white" if pivot_data.iloc[i, j] > 1.5 else "black")
                                )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 라인 차트로도 표시
                        fig2 = px.line(all_comparison_df, x='지표', y='배수', color='등급',
                                     title="미슐랭 등급별 비미슐랭 대비 배수 비교", markers=True)
                        fig2.add_hline(y=1, line_dash="dash", line_color="red",
                                     annotation_text="비미슐랭 기준선")
                        fig2.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # 비미슐랭 인기 품목 참고 정보
                    st.subheader("📋 비미슐랭 업장 인기 품목 (참고)")
                    non_michelin_products = vs_result['비미슐랭_인기품목'][:10]
                    if non_michelin_products:
                        cols = st.columns(5)
                        for i, product in enumerate(non_michelin_products):
                            with cols[i % 5]:
                                st.write(f"{i+1}. {product}")
                
                else:
                    st.error(vs_result['메시지'])
        
        # 탭 7: 베이커리 & 디저트
        with tab7:
            st.markdown('<h2 class="sub-header">🧁 베이커리 & 디저트 분석</h2>', unsafe_allow_html=True)
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