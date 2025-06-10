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
    page_title="마이크로그린 추천 시스템",
    page_icon="🌱",
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
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #228B22;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

class MicrogreenRecommendationSystem:
    def __init__(self, sales_data, refund_data=None):
        self.sales_data = sales_data
        self.refund_data = refund_data
        self.customer_product_matrix = None
        self.product_similarity = None
        self.seasonal_products = None
        self.frequent_pairs = None
        
        # 데이터 전처리
        self.preprocess_data()
        
        # 분석 수행
        self.calculate_product_similarity()
        self.identify_seasonal_products()
        self.identify_frequent_pairs()

    def preprocess_data(self):
        """데이터 전처리"""
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
        
        # 고객-상품 매트릭스 생성
        self.customer_product_matrix = self.sales_data.groupby(['고객명', '상품'])['수량'].sum().unstack(fill_value=0)
        
        # 날짜 컬럼이 있으면 월 정보 추가
        if '날짜' in self.sales_data.columns:
            self.sales_data['month'] = self.sales_data['날짜'].dt.month

    def calculate_product_similarity(self):
        """상품 간 유사도 계산"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 상품별 고객 구매 패턴으로 유사도 계산
        product_matrix = self.customer_product_matrix.T  # 상품 x 고객
        similarity_matrix = cosine_similarity(product_matrix)
        
        self.product_similarity = pd.DataFrame(
            similarity_matrix,
            index=product_matrix.index,
            columns=product_matrix.index
        )

    def identify_seasonal_products(self):
        """계절별 상품 식별"""
        if 'month' not in self.sales_data.columns:
            self.seasonal_products = {}
            return
            
        # 계절 정의
        def calculate_seasonality(sales_array):
            if len(sales_array) < 4:
                return 0
            return np.std(sales_array) / (np.mean(sales_array) + 1)
        
        seasonal_data = {}
        
        for product in self.sales_data['상품'].unique():
            product_sales = self.sales_data[self.sales_data['상품'] == product]
            monthly_sales = product_sales.groupby('month')['수량'].sum()
            
            # 12개월 데이터로 확장 (없는 월은 0)
            full_monthly = pd.Series(0, index=range(1, 13))
            full_monthly.update(monthly_sales)
            
            # 계절별 집계
            spring = full_monthly[[3, 4, 5]].sum()  # 봄
            summer = full_monthly[[6, 7, 8]].sum()  # 여름
            fall = full_monthly[[9, 10, 11]].sum()  # 가을
            winter = full_monthly[[12, 1, 2]].sum()  # 겨울
            
            seasonal_sales = [spring, summer, fall, winter]
            seasonality_score = calculate_seasonality(seasonal_sales)
            
            # 가장 높은 계절 찾기
            seasons = ['봄', '여름', '가을', '겨울']
            peak_season = seasons[np.argmax(seasonal_sales)]
            
            seasonal_data[product] = {
                '계절성_점수': seasonality_score,
                '주요_계절': peak_season,
                '계절별_판매량': dict(zip(seasons, seasonal_sales))
            }
        
        self.seasonal_products = seasonal_data

    def identify_frequent_pairs(self):
        """자주 함께 구매되는 상품 쌍 식별"""
        def get_product_pairs(products):
            pairs = []
            for i in range(len(products)):
                for j in range(i+1, len(products)):
                    pairs.append((products[i], products[j]))
            return pairs
        
        # 고객별 구매 상품 리스트
        customer_products = self.sales_data.groupby('고객명')['상품'].apply(list).reset_index()
        
        # 모든 상품 쌍 수집
        all_pairs = []
        for products in customer_products['상품']:
            if len(products) > 1:
                pairs = get_product_pairs(list(set(products)))  # 중복 제거
                all_pairs.extend(pairs)
        
        # 쌍별 빈도 계산
        pair_counts = pd.Series(all_pairs).value_counts()
        
        # 최소 2번 이상 함께 구매된 쌍만 선택
        self.frequent_pairs = pair_counts[pair_counts >= 2].to_dict()

    def recommend_for_customer(self, customer_id, n=5, current_month=None):
        """특정 고객을 위한 상품 추천"""
        if customer_id not in self.customer_product_matrix.index:
            return {
                '상태': '실패',
                '메시지': f"고객 '{customer_id}'을(를) 찾을 수 없습니다."
            }
        
        # 재고조정, 창고 제외
        if '재고조정' in customer_id or '문정창고' in customer_id or '창고' in customer_id:
            return {
                '상태': '실패',
                '메시지': f"'{customer_id}'은(는) 추천에서 제외됩니다."
            }
        
        # 고객의 구매 이력
        customer_purchases = self.customer_product_matrix.loc[customer_id]
        purchased_products = customer_purchases[customer_purchases > 0].index.tolist()
        
        if not purchased_products:
            return {
                '상태': '실패',
                '메시지': "구매 이력이 없어 추천할 수 없습니다."
            }
        
        # 추천 점수 계산
        recommendations = {}
        
        # 1. 유사 상품 기반 추천
        for product in purchased_products:
            if product in self.product_similarity.index:
                similar_products = self.product_similarity[product].sort_values(ascending=False)
                
                for similar_product, similarity in similar_products.items():
                    if similar_product not in purchased_products and similarity > 0.1:
                        if similar_product not in recommendations:
                            recommendations[similar_product] = 0
                        recommendations[similar_product] += similarity * customer_purchases[product]
        
        # 2. 계절성 고려 (현재 월이 주어진 경우)
        if current_month and self.seasonal_products:
            current_season = None
            if current_month in [3, 4, 5]:
                current_season = '봄'
            elif current_month in [6, 7, 8]:
                current_season = '여름'
            elif current_month in [9, 10, 11]:
                current_season = '가을'
            else:
                current_season = '겨울'
            
            for product, data in self.seasonal_products.items():
                if (product not in purchased_products and 
                    data['주요_계절'] == current_season and 
                    data['계절성_점수'] > 0.5):
                    if product not in recommendations:
                        recommendations[product] = 0
                    recommendations[product] += data['계절성_점수'] * 2
        
        # 3. 자주 함께 구매되는 상품
        for (prod1, prod2), count in self.frequent_pairs.items():
            if prod1 in purchased_products and prod2 not in purchased_products:
                if prod2 not in recommendations:
                    recommendations[prod2] = 0
                recommendations[prod2] += count * 0.5
            elif prod2 in purchased_products and prod1 not in purchased_products:
                if prod1 not in recommendations:
                    recommendations[prod1] = 0
                recommendations[prod1] += count * 0.5
        
        # 상위 N개 추천
        if not recommendations:
            return {
                '상태': '실패',
                '메시지': "추천할 상품이 없습니다."
            }
        
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # 추천 이유 생성
        recommendation_details = []
        for product, score in top_recommendations:
            reasons = []
            
            # 유사 상품 이유
            for purchased in purchased_products:
                if (purchased in self.product_similarity.index and 
                    product in self.product_similarity.columns):
                    similarity = self.product_similarity.loc[purchased, product]
                    if similarity > 0.3:
                        reasons.append(f"'{purchased}'와 유사한 상품")
                        break
            
            # 계절성 이유
            if current_month and product in self.seasonal_products:
                season_data = self.seasonal_products[product]
                current_season = None
                if current_month in [3, 4, 5]:
                    current_season = '봄'
                elif current_month in [6, 7, 8]:
                    current_season = '여름'
                elif current_month in [9, 10, 11]:
                    current_season = '가을'
                else:
                    current_season = '겨울'
                
                if season_data['주요_계절'] == current_season:
                    reasons.append(f"{current_season} 시즌 인기 상품")
            
            # 함께 구매 이유
            for (prod1, prod2), count in self.frequent_pairs.items():
                if ((prod1 in purchased_products and prod2 == product) or 
                    (prod2 in purchased_products and prod1 == product)):
                    other_product = prod1 if prod2 == product else prod2
                    reasons.append(f"'{other_product}'와 자주 함께 구매")
                    break
            
            recommendation_details.append({
                '상품': product,
                '점수': round(score, 2),
                '추천_이유': ', '.join(reasons) if reasons else '구매 패턴 기반'
            })
        
        return {
            '상태': '성공',
            '고객명': customer_id,
            '추천_상품': recommendation_details,
            '구매_이력': purchased_products
        }

    def recommend_for_season(self, season=None, n=10):
        """계절별 추천 상품"""
        if not self.seasonal_products:
            return {
                '상태': '실패',
                '메시지': "계절별 데이터가 없습니다."
            }
        
        # 현재 계절 자동 감지
        if season is None:
            current_month = datetime.now().month
            if current_month in [3, 4, 5]:
                season = '봄'
            elif current_month in [6, 7, 8]:
                season = '여름'
            elif current_month in [9, 10, 11]:
                season = '가을'
            else:
                season = '겨울'
        
        # 해당 계절 상품 필터링 및 정렬
        seasonal_recommendations = []
        
        for product, data in self.seasonal_products.items():
            if data['주요_계절'] == season:
                seasonal_score = data['계절별_판매량'][season]
                seasonality_score = data['계절성_점수']
                
                seasonal_recommendations.append({
                    '상품': product,
                    '계절_판매량': seasonal_score,
                    '계절성_점수': round(seasonality_score, 3),
                    '종합_점수': seasonal_score * (1 + seasonality_score)
                })
        
        # 종합 점수로 정렬
        seasonal_recommendations.sort(key=lambda x: x['종합_점수'], reverse=True)
        
        return {
            '상태': '성공',
            '계절': season,
            '추천_상품': seasonal_recommendations[:n]
        }

    def recommend_bundles(self, n=5, current_month=None):
        """번들 상품 추천"""
        if not self.frequent_pairs:
            return {
                '상태': '실패',
                '메시지': "번들 추천을 위한 데이터가 부족합니다."
            }
        
        # 상품별 총 판매량 계산
        product_sales = self.sales_data.groupby('상품')['수량'].sum()
        
        # 번들 점수 계산
        bundle_scores = []
        
        for (prod1, prod2), pair_count in self.frequent_pairs.items():
            # 각 상품이 유효한지 확인
            if prod1 in product_sales.index and prod2 in product_sales.index:
                # 기본 번들 점수 (함께 구매 빈도)
                bundle_score = pair_count
                
                # 개별 상품 인기도 고려
                popularity_score = (product_sales[prod1] + product_sales[prod2]) / 2
                
                # 계절성 고려 (현재 월이 주어진 경우)
                seasonal_bonus = 0
                if current_month and self.seasonal_products:
                    current_season = None
                    if current_month in [3, 4, 5]:
                        current_season = '봄'
                    elif current_month in [6, 7, 8]:
                        current_season = '여름'
                    elif current_month in [9, 10, 11]:
                        current_season = '가을'
                    else:
                        current_season = '겨울'
                    
                    for product in [prod1, prod2]:
                        if (product in self.seasonal_products and 
                            self.seasonal_products[product]['주요_계절'] == current_season):
                            seasonal_bonus += 1
                
                # 최종 점수 계산
                final_score = bundle_score * 10 + popularity_score * 0.01 + seasonal_bonus * 5
                
                bundle_scores.append({
                    '상품1': prod1,
                    '상품2': prod2,
                    '함께_구매_횟수': pair_count,
                    '상품1_총판매량': int(product_sales[prod1]),
                    '상품2_총판매량': int(product_sales[prod2]),
                    '번들_점수': round(final_score, 2)
                })
        
        # 번들 점수로 정렬
        bundle_scores.sort(key=lambda x: x['번들_점수'], reverse=True)
        
        return {
            '상태': '성공',
            '추천_번들': bundle_scores[:n]
        }

    def recommend_for_new_customer(self, n=5, current_month=None):
        """신규 고객을 위한 추천"""
        # 전체 상품별 인기도 계산
        product_popularity = self.sales_data.groupby('상품')['수량'].sum().sort_values(ascending=False)
        
        # 고객 수 기준 인기도
        product_customer_count = self.sales_data.groupby('상품')['고객명'].nunique().sort_values(ascending=False)
        
        # 종합 점수 계산 (판매량 + 고객 수)
        recommendations = []
        
        for product in product_popularity.index:
            total_sales = product_popularity[product]
            customer_count = product_customer_count.get(product, 0)
            
            # 기본 점수 (판매량 + 고객 수)
            base_score = total_sales * 0.7 + customer_count * 0.3
            
            # 계절성 보너스
            seasonal_bonus = 0
            if current_month and product in self.seasonal_products:
                current_season = None
                if current_month in [3, 4, 5]:
                    current_season = '봄'
                elif current_month in [6, 7, 8]:
                    current_season = '여름'
                elif current_month in [9, 10, 11]:
                    current_season = '가을'
                else:
                    current_season = '겨울'
                
                season_data = self.seasonal_products[product]
                if season_data['주요_계절'] == current_season:
                    seasonal_bonus = season_data['계절별_판매량'][current_season] * 0.2
            
            final_score = base_score + seasonal_bonus
            
            recommendations.append({
                '상품': product,
                '총_판매량': int(total_sales),
                '구매_고객수': int(customer_count),
                '추천_점수': round(final_score, 2)
            })
        
        # 점수순 정렬
        recommendations.sort(key=lambda x: x['추천_점수'], reverse=True)
        
        # 추천 이유 추가
        for i, rec in enumerate(recommendations[:n]):
            reasons = []
            if rec['총_판매량'] > product_popularity.median():
                reasons.append("높은 판매량")
            if rec['구매_고객수'] > product_customer_count.median():
                reasons.append("다양한 고객층 선호")
            
            if current_month and rec['상품'] in self.seasonal_products:
                current_season = None
                if current_month in [3, 4, 5]:
                    current_season = '봄'
                elif current_month in [6, 7, 8]:
                    current_season = '여름'
                elif current_month in [9, 10, 11]:
                    current_season = '가을'
                else:
                    current_season = '겨울'
                
                if self.seasonal_products[rec['상품']]['주요_계절'] == current_season:
                    reasons.append(f"{current_season} 시즌 인기")
            
            rec['추천_이유'] = ', '.join(reasons) if reasons else '전반적 인기'
        
        return {
            '상태': '성공',
            '추천_상품': recommendations[:n]
        }

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">🌱 마이크로그린 추천 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">신선하고 건강한 마이크로그린을 추천해드립니다</p>', unsafe_allow_html=True)
    
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
        
        # 날짜 변환
        if '날짜' in sales_data.columns:
            # 날짜 형식이 'YY.MM.DD'인 경우 처리
            sales_data['날짜'] = pd.to_datetime(sales_data['날짜'].astype(str).apply(
                lambda x: f"20{x}" if len(str(x).split('.')[0]) == 2 else x
            ), errors='coerce')
            
            # 월 정보 추가
            sales_data['month'] = sales_data['날짜'].dt.month
        
        # 추천 시스템 초기화
        with st.spinner('추천 시스템을 초기화하고 있습니다...'):
            recommender = MicrogreenRecommendationSystem(sales_data, refund_data)
        
        st.success("✅ 데이터가 성공적으로 로드되었습니다!")
        
        # 메인 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 고객 맞춤 추천", 
            "🌱 계절 추천", 
            "📦 번들 추천", 
            "✨ 신규 고객 추천"
        ])
        
        # 탭 1: 고객 맞춤 추천
        with tab1:
            st.markdown('<h2 class="sub-header">🎯 고객 맞춤 추천</h2>', unsafe_allow_html=True)
            
            # 고객 선택
            customers = [c for c in recommender.customer_product_matrix.index 
                        if not any(keyword in c for keyword in ['재고조정', '문정창고', '창고'])]
            
            if customers:
                selected_customer = st.selectbox("고객을 선택하세요:", customers)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    recommendation_count = st.slider("추천 상품 수", 1, 10, 5)
                with col2:
                    current_month = st.selectbox("현재 월 (계절성 고려)", 
                                               [None] + list(range(1, 13)), 
                                               format_func=lambda x: "자동 감지" if x is None else f"{x}월")
                
                if st.button("추천 받기", type="primary"):
                    with st.spinner('맞춤 추천을 생성하고 있습니다...'):
                        result = recommender.recommend_for_customer(
                            selected_customer, 
                            n=recommendation_count, 
                            current_month=current_month
                        )
                    
                    if result['상태'] == '성공':
                        st.success(f"✅ {selected_customer}님을 위한 추천이 완료되었습니다!")
                        
                        # 구매 이력
                        with st.expander("📋 구매 이력"):
                            if result['구매_이력']:
                                purchase_df = pd.DataFrame({
                                    '구매한 상품': result['구매_이력']
                                })
                                st.dataframe(purchase_df, use_container_width=True)
                            else:
                                st.info("구매 이력이 없습니다.")
                        
                        # 추천 상품
                        st.subheader("🎯 추천 상품")
                        for i, rec in enumerate(result['추천_상품'], 1):
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 2])
                                with col1:
                                    st.markdown(f"**{i}. {rec['상품']}**")
                                with col2:
                                    st.metric("점수", rec['점수'])
                                with col3:
                                    st.caption(rec['추천_이유'])
                    else:
                        st.error(result['메시지'])
            else:
                st.warning("추천 가능한 고객이 없습니다.")
        
        # 탭 2: 계절 추천
        with tab2:
            st.markdown('<h2 class="sub-header">🌱 계절 추천</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                season_option = st.selectbox("계절 선택", 
                                           ["자동 감지", "봄", "여름", "가을", "겨울"])
            with col2:
                season_count = st.slider("추천 상품 수", 1, 20, 10)
            
            season = None if season_option == "자동 감지" else season_option
            
            if st.button("계절 추천 받기", type="primary"):
                with st.spinner('계절별 추천을 생성하고 있습니다...'):
                    result = recommender.recommend_for_season(season=season, n=season_count)
                
                if result['상태'] == '성공':
                    st.success(f"✅ {result['계절']} 시즌 추천이 완료되었습니다!")
                    
                    if result['추천_상품']:
                        # 차트로 시각화
                        chart_data = pd.DataFrame(result['추천_상품'])
                        
                        fig = px.bar(
                            chart_data.head(10), 
                            x='상품', 
                            y='계절_판매량',
                            title=f"{result['계절']} 시즌 인기 상품",
                            color='계절성_점수',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 상세 테이블
                        st.subheader("📊 상세 정보")
                        display_df = chart_data[['상품', '계절_판매량', '계절성_점수', '종합_점수']].copy()
                        display_df.columns = ['상품명', '계절 판매량', '계절성 점수', '종합 점수']
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info(f"{result['계절']} 시즌에 특화된 상품이 없습니다.")
                else:
                    st.error(result['메시지'])
        
        # 탭 3: 번들 추천
        with tab3:
            st.markdown('<h2 class="sub-header">📦 번들 추천</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                bundle_count = st.slider("추천 번들 수", 1, 10, 5)
            with col2:
                bundle_month = st.selectbox("현재 월 (계절성 고려)", 
                                          [None] + list(range(1, 13)), 
                                          format_func=lambda x: "고려 안함" if x is None else f"{x}월",
                                          key="bundle_month")
            
            if st.button("번들 추천 받기", type="primary"):
                with st.spinner('번들 추천을 생성하고 있습니다...'):
                    result = recommender.recommend_bundles(n=bundle_count, current_month=bundle_month)
                
                if result['상태'] == '성공':
                    st.success("✅ 번들 추천이 완료되었습니다!")
                    
                    # 번들 추천 표시
                    for i, bundle in enumerate(result['추천_번들'], 1):
                        with st.container():
                            st.markdown(f"### 📦 번들 {i}")
                            
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.markdown(f"**상품 1:** {bundle['상품1']}")
                                st.caption(f"총 판매량: {bundle['상품1_총판매량']:,}개")
                            with col2:
                                st.markdown(f"**상품 2:** {bundle['상품2']}")
                                st.caption(f"총 판매량: {bundle['상품2_총판매량']:,}개")
                            with col3:
                                st.metric("번들 점수", bundle['번들_점수'])
                                st.caption(f"함께 구매: {bundle['함께_구매_횟수']}회")
                            
                            st.divider()
                    
                    # 번들 인기도 차트
                    if result['추천_번들']:
                        chart_data = pd.DataFrame(result['추천_번들'])
                        chart_data['번들명'] = chart_data['상품1'] + ' + ' + chart_data['상품2']
                        
                        fig = px.bar(
                            chart_data, 
                            x='번들명', 
                            y='번들_점수',
                            title="번들 추천 점수",
                            color='함께_구매_횟수',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=400)
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(result['메시지'])
        
        # 탭 4: 신규 고객 추천
        with tab4:
            st.markdown('<h2 class="sub-header">✨ 신규 고객 추천</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                new_customer_count = st.slider("추천 상품 수", 1, 15, 8)
            with col2:
                new_customer_month = st.selectbox("현재 월 (계절성 고려)", 
                                                [None] + list(range(1, 13)), 
                                                format_func=lambda x: "고려 안함" if x is None else f"{x}월",
                                                key="new_customer_month")
            
            if st.button("신규 고객 추천 받기", type="primary"):
                with st.spinner('신규 고객 추천을 생성하고 있습니다...'):
                    result = recommender.recommend_for_new_customer(
                        n=new_customer_count, 
                        current_month=new_customer_month
                    )
                
                if result['상태'] == '성공':
                    st.success("✅ 신규 고객 추천이 완료되었습니다!")
                    
                    # 추천 상품 카드 형태로 표시
                    for i in range(0, len(result['추천_상품']), 2):
                        cols = st.columns(2)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(result['추천_상품']):
                                rec = result['추천_상품'][i + j]
                                with col:
                                    with st.container():
                                        st.markdown(f"### {i + j + 1}. {rec['상품']}")
                                        
                                        metric_col1, metric_col2 = st.columns(2)
                                        with metric_col1:
                                            st.metric("총 판매량", f"{rec['총_판매량']:,}개")
                                        with metric_col2:
                                            st.metric("구매 고객수", f"{rec['구매_고객수']:,}명")
                                        
                                        st.metric("추천 점수", rec['추천_점수'])
                                        st.caption(f"💡 {rec['추천_이유']}")
                                        st.divider()
                    
                    # 인기도 차트
                    chart_data = pd.DataFrame(result['추천_상품'])
                    
                    fig = px.scatter(
                        chart_data, 
                        x='총_판매량', 
                        y='구매_고객수',
                        size='추천_점수',
                        hover_name='상품',
                        title="상품 인기도 분석",
                        labels={
                            '총_판매량': '총 판매량',
                            '구매_고객수': '구매 고객 수'
                        }
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
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
                
                # 추천 시스템 초기화
                with st.spinner('추천 시스템을 초기화하고 있습니다...'):
                    recommender = MicrogreenRecommendationSystem(sales_data, refund_data)
                
                st.success("✅ 업로드된 데이터로 추천 시스템이 초기화되었습니다!")
                st.rerun()  # 페이지 새로고침
                
            except Exception as upload_error:
                st.error(f"업로드된 데이터 처리 중 오류가 발생했습니다: {str(upload_error)}")
    
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        st.info("데이터 형식을 확인해주세요.")

if __name__ == "__main__":
    main() 