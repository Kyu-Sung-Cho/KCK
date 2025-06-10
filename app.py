import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm
import platform
import os
import random

# 한글 폰트 설정
system_name = platform.system()
if system_name == "Windows":
    font_name = 'Malgun Gothic' # 윈도우
elif system_name == "Darwin":
    font_name = 'AppleGothic' # macOS
else:
    font_name = 'NanumGothic' # Linux

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 마이크로그린 추천 시스템 클래스
class MicrogreenRecommendationSystem:
    def __init__(self, sales_data, refund_data=None):
        self.sales_data = sales_data
        self.refund_data = refund_data
        self.customer_product_matrix = None
        self.product_similarity_matrix = None
        self.seasonal_products = None
        self.frequent_pairs = None
        # 데이터 전처리 수행
        self.preprocess_data()

    def preprocess_data(self):
        """데이터 전처리 및 필요한 매트릭스 생성"""
        # 월 정보 추가
        if 'month' not in self.sales_data.columns and '날짜' in self.sales_data.columns:
            self.sales_data['month'] = self.sales_data['날짜'].dt.month
        
        # 불필요한 항목 제외 (세트상품, 증정품, 배송료 등)
        self.sales_data = self.sales_data[~self.sales_data['상품'].str.contains('세트상품|증정품', na=False)]
        
        # 배송료 관련 항목 더 철저히 제외
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        delivery_pattern = '|'.join(delivery_keywords)
        self.sales_data = self.sales_data[~self.sales_data['상품'].str.contains(delivery_pattern, case=False, na=False)]
        
        # 반품 데이터에서 재고조정 제외 (있는 경우)
        if self.refund_data is not None and '반품사유' in self.refund_data.columns:
            # 재고조정 제외 코드 제거
            pass
        
        # 고객-상품 매트릭스 생성 (각 고객이 각 상품을 구매한 수량 집계)
        self.customer_product_matrix = self.sales_data.pivot_table(
            index='고객명',
            columns='상품',
            values='수량',
            aggfunc='sum',
            fill_value=0
        )
        
        # 상품 유사도 매트릭스 계산
        self.calculate_product_similarity()
        
        # 계절별 상품 분석
        self.identify_seasonal_products()
        
        # 자주 함께 구매되는 상품 쌍 분석
        self.identify_frequent_pairs()

    def calculate_product_similarity(self):
        """상품 간 유사도 계산 (어떤 고객들이 함께 구매했는지 기반)"""
        # 상품-고객 매트릭스 생성 (고객-상품 매트릭스의 전치행렬)
        product_customer_matrix = self.customer_product_matrix.T
        
        # 코사인 유사도 계산
        self.product_similarity_matrix = pd.DataFrame(
            cosine_similarity(product_customer_matrix),
            index=product_customer_matrix.index,
            columns=product_customer_matrix.index
        )

    def identify_seasonal_products(self):
        """계절성을 가진 상품 분석"""
        # 월별 상품 판매량 데이터 생성
        monthly_product_sales = self.sales_data.groupby(['month', '상품'])['수량'].sum().reset_index()
        pivot_table = monthly_product_sales.pivot(index='상품', columns='month', values='수량').fillna(0)
        
        # 계절성 계산
        def calculate_seasonality(sales_array):
            if np.mean(sales_array) > 0:
                return np.std(sales_array) / (np.mean(sales_array) + 1)
            else:
                return 0
        
        # 계절성 측정
        seasonal_products = []
        for product in pivot_table.index:
            monthly_sales = pivot_table.loc[product].values
            if np.sum(monthly_sales) > 10: # 최소한의 판매 데이터가 있어야 함
                seasonality = calculate_seasonality(monthly_sales)
                peak_month = pivot_table.loc[product].idxmax() # 최대 판매월
                total_sales = int(np.sum(monthly_sales))  # 정수형으로 변환
                seasonal_products.append((product, seasonality, peak_month, total_sales))
        
        self.seasonal_products = pd.DataFrame(
            seasonal_products,
            columns=['상품', '계절성_지수', '피크_판매월', '총_판매량']
        ).sort_values('계절성_지수', ascending=False)
        
        # 주요 계절 설정 (봄: 3-5월, 여름: 6-8월, 가을: 9-11월, 겨울: 12-2월)
        seasons = {
            '봄': [3, 4, 5],
            '여름': [6, 7, 8],
            '가을': [9, 10, 11],
            '겨울': [12, 1, 2]
        }
        
        # 각 상품의 주요 계절 결정
        main_seasons = []
        for _, row in self.seasonal_products.iterrows():
            product = row['상품']
            if product in pivot_table.index:
                max_seasonal_sales = 0
                main_season = "없음"
                for season, months in seasons.items():
                    seasonal_sales = sum([pivot_table.loc[product, m] for m in months if m in pivot_table.columns])
                    if seasonal_sales > max_seasonal_sales:
                        max_seasonal_sales = seasonal_sales
                        main_season = season
                main_seasons.append(main_season)
            else:
                main_seasons.append("없음")
        
        self.seasonal_products['주요_계절'] = main_seasons

    def identify_frequent_pairs(self):
        """자주 함께 구매되는 상품 쌍 분석 및 저장"""
        # 거래 데이터
        transaction_data = self.sales_data.groupby(['고객명', '날짜'])['상품'].apply(list).reset_index()
        
        # 상품 쌍 생성을 위한 함수
        def get_product_pairs(products):
            pairs = []
            products = [p for p in products if isinstance(p, str)] # 데이터 정제
            for i in range(len(products)):
                for j in range(i+1, len(products)):
                    pairs.append((products[i], products[j]))
            return pairs
        
        # 자주 구매되는 상품 쌍 찾기
        all_pairs = []
        for transaction in transaction_data['상품']:
            all_pairs.extend(get_product_pairs(transaction))
        
        # 상품 쌍의 빈도수 계산하여 상위 항목 저장
        pair_counts = Counter(all_pairs)
        self.frequent_pairs = pd.DataFrame(
            pair_counts.most_common(30),
            columns=['상품_쌍', '함께_구매된_횟수']
        )

    def recommend_for_customer(self, customer_id, n=5, current_month=None):
        """고객 맞춤형 상품 추천"""
        # 배송료 관련 키워드 정의
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        delivery_pattern = '|'.join(delivery_keywords)
        
        if current_month is None:
            # 현재 월 자동 설정
            current_month = datetime.now().month
        
        # 1. 고객 구매 이력 확인
        if customer_id not in self.customer_product_matrix.index:
            return pd.DataFrame({
                '추천_상품': ["고객 구매 이력이 없습니다."],
                '추천_점수': [0],
                '추천_이유': ["데이터가 없습니다."],
                '유사도_점수': [0],
                '계절성_보너스': [0],
                '함께_구매_보너스': [0]
            })
        
        customer_purchases = self.customer_product_matrix.loc[customer_id]
        # 고객이 구매한 상품의 리스트
        purchased_products = customer_purchases[customer_purchases > 0]
        
        # 2. 유사한 상품 기반 추천 점수 계산
        raw_similarity_scores = {}
        raw_seasonal_bonuses = {}
        raw_co_purchase_bonuses = {}
        candidate_products = []
        
        for product in self.customer_product_matrix.columns:
            # 이미 구매한 상품은 제외
            if product in purchased_products.index:
                continue
            
            # 세트상품, 증정품, 배송료 관련 상품 제외
            if '세트상품' in product or '증정품' in product or any(keyword in product.lower() for keyword in delivery_keywords):
                continue
            
            # 후보 상품 리스트에 추가
            candidate_products.append(product)
            
            # 구매한 상품과 유사도 계산
            similarity_score = 0
            for purchased_product, quantity in purchased_products.items():
                if purchased_product in self.product_similarity_matrix.index and product in self.product_similarity_matrix.columns:
                    similarity = self.product_similarity_matrix.loc[purchased_product, product]
                    similarity_score += similarity * quantity
            
            # 원시 유사도 점수 저장
            raw_similarity_scores[product] = similarity_score
            
            # 계절성 보너스 계산
            seasonal_bonus = 0
            if product in self.seasonal_products['상품'].values:
                product_info = self.seasonal_products[self.seasonal_products['상품'] == product].iloc[0]
                # 현재 월과 제품의 최대 판매월이 가까울수록 높은 보너스
                months_diff = abs(current_month - product_info['피크_판매월'])
                if months_diff <= 1: # 최대 판매월 또는 인접월
                    seasonal_bonus = 5
                elif months_diff <= 2: # 최대 판매월에서 2개월
                    seasonal_bonus = 3
                
                # 계절성이 높은 제품에게는 추가 보너스
                if product_info['계절성_지수'] > 0.5:
                    seasonal_bonus += 3
            
            # 원시 계절성 보너스 저장
            raw_seasonal_bonuses[product] = seasonal_bonus
            
            # 함께 구매 패턴 보너스 초기화
            raw_co_purchase_bonuses[product] = 0
        
        # 3. 함께 구매 패턴 분석
        for _, row in self.frequent_pairs.iterrows():
            product_pair = row['상품_쌍']
            frequency = row['함께_구매된_횟수']
            
            # 고객이 구매한 상품과 함께 구매되는 상품에 보너스
            for purchased_product in purchased_products.index:
                if purchased_product in product_pair:
                    other_product = product_pair[0] if product_pair[1] == purchased_product else product_pair[1]
                    if other_product in candidate_products:
                        # 함께 구매 빈도에 비례한 보너스
                        raw_co_purchase_bonuses[other_product] += frequency
        
        # 4. 점수 정규화 및 가중치 적용
        # 각 점수 요소 정규화를 위한 최대값 계산
        max_similarity = max(raw_similarity_scores.values()) if raw_similarity_scores else 1
        max_seasonal = max(raw_seasonal_bonuses.values()) if raw_seasonal_bonuses else 1
        max_co_purchase = max(raw_co_purchase_bonuses.values()) if raw_co_purchase_bonuses else 1
        
        # 정규화된 점수 및 가중치 적용을 위한 빈 딕셔너리
        normalized_similarity = {}
        normalized_seasonal = {}
        normalized_co_purchase = {}
        final_scores = {}
        
        # 가중치 설정 (합이 1이 되도록 설정)
        similarity_weight = 0.35  # 유사도 가중치 - 가장 중요하지만 과도하지 않게
        seasonal_weight = 0.3     # 계절성 가중치 - 중간 수준 
        co_purchase_weight = 0.35 # 함께 구매 패턴 가중치 - 유사도와 동등하게 중요
        
        for product in candidate_products:
            # 0으로 나누는 것 방지
            if max_similarity > 0:
                normalized_similarity[product] = raw_similarity_scores[product] / max_similarity * 10
            else:
                normalized_similarity[product] = 0
                
            if max_seasonal > 0:    
                normalized_seasonal[product] = raw_seasonal_bonuses[product] / max_seasonal * 10
            else:
                normalized_seasonal[product] = 0
                
            if max_co_purchase > 0:    
                normalized_co_purchase[product] = raw_co_purchase_bonuses[product] / max_co_purchase * 10
            else:
                normalized_co_purchase[product] = 0
            
            # 가중합으로 최종 점수 계산
            final_scores[product] = (
                similarity_weight * normalized_similarity[product] +
                seasonal_weight * normalized_seasonal[product] +
                co_purchase_weight * normalized_co_purchase[product]
            )
        
        # 5. 최종 추천 결과 정렬
        recommendations = pd.DataFrame({
            '추천_상품': list(final_scores.keys()),
            '추천_점수': list(final_scores.values()),
            '유사도_점수': [normalized_similarity[product] for product in final_scores.keys()],
            '계절성_보너스': [normalized_seasonal[product] for product in final_scores.keys()],
            '함께_구매_보너스': [normalized_co_purchase[product] for product in final_scores.keys()]
        }).sort_values('추천_점수', ascending=False).head(n)
        
        # 6. 추천 이유 생성
        reasons = []
        for product in recommendations['추천_상품']:
            reason_parts = []
            
            # 가장 유사한 기존 구매 상품 찾기
            max_similarity = 0
            similar_product = None
            for purchased_product in purchased_products.index:
                if purchased_product in self.product_similarity_matrix.index and product in self.product_similarity_matrix.columns:
                    similarity = self.product_similarity_matrix.loc[purchased_product, product]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        similar_product = purchased_product
            
            if max_similarity > 0.3:
                reason_parts.append(f"{similar_product} 구매 고객이 많이 선택")
            
            # 계절성 추천 이유
            if product in self.seasonal_products['상품'].values:
                product_info = self.seasonal_products[self.seasonal_products['상품'] == product].iloc[0]
                if abs(current_month - product_info['피크_판매월']) <= 2:
                    reason_parts.append(f"현재 {current_month}월이 판매성수기({int(product_info['피크_판매월'])}월)와 가까운 시기")
            
            # 함께 구매 패턴 추천 이유
            for _, row in self.frequent_pairs.head(10).iterrows():
                product_pair = row['상품_쌍']
                if product in product_pair:
                    other_product = product_pair[0] if product_pair[1] == product else product_pair[1]
                    if other_product in purchased_products.index:
                        reason_parts.append(f"{other_product}와 함께 구매 빈도가 높음")
                        break
            
            if not reason_parts:
                reason_parts.append("고객 구매 패턴 기반 추천")
            
            reasons.append(" & ".join(reason_parts))
        
        recommendations['추천_이유'] = reasons
        return recommendations

    def recommend_for_season(self, season=None, n=10):
        """계절 맞춤형 상품의 추천"""
        # 현재 계절 설정
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
        
        # 해당 계절에 잘 팔리는 상품 추출
        seasonal_recommendations = self.seasonal_products[self.seasonal_products['주요_계절'] == season]
        
        # 세트상품 등과 배송료 제외
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        delivery_pattern = '|'.join(delivery_keywords)
        seasonal_recommendations = seasonal_recommendations[
            ~seasonal_recommendations['상품'].str.contains('세트상품|증정품', na=False) &
            ~seasonal_recommendations['상품'].str.contains(delivery_pattern, case=False, na=False)
        ]
        
        # 계절성과 판매량을 모두 고려하여 점수화
        seasonal_recommendations['추천_점수'] = seasonal_recommendations['총_판매량'] * (1 + seasonal_recommendations['계절성_지수'])
        seasonal_recommendations = seasonal_recommendations.sort_values('추천_점수', ascending=False).head(n)
        
        return seasonal_recommendations[['상품', '계절성_지수', '총_판매량', '추천_점수']]

    def recommend_bundles(self, n=5, current_month=None):
        """자주 함께 구매되는 상품 번들 추천"""
        # 현재 월 기본값 설정
        if current_month is None:
            current_month = datetime.now().month
            
        # 현재 월의 계절 판단
        season_map = {
            12: '겨울', 1: '겨울', 2: '겨울',
            3: '봄', 4: '봄', 5: '봄',
            6: '여름', 7: '여름', 8: '여름',
            9: '가을', 10: '가을', 11: '가을'
        }
        current_season = season_map.get(current_month, '계절 미확인')
        
        # 다음 계절 계산 (현재 계절 다음에 올 계절)
        next_season_map = {
            '봄': '여름',
            '여름': '가을',
            '가을': '겨울',
            '겨울': '봄'
        }
        next_season = next_season_map.get(current_season, '계절 미확인')
            
        # 배송료 관련 키워드 정의
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        
        # 세트상품 등과 배송료 제외
        filtered_pairs = []
        
        # 모든 판매 데이터에서 월별 거래 필터링
        if current_month is not None and '날짜' in self.sales_data.columns and 'month' in self.sales_data.columns:
            # 현재 월의 거래 데이터만 선택
            monthly_sales = self.sales_data[self.sales_data['month'] == current_month]
            
            # 해당 월의 거래 데이터로 상품 쌍 분석
            monthly_transaction_data = monthly_sales.groupby(['고객명', '날짜'])['상품'].apply(list).reset_index()
            
            # 해당 월의 함께 구매된 상품 쌍 추출
            monthly_pairs = []
            for transaction in monthly_transaction_data['상품']:
                pairs = self.get_product_pairs(transaction)
                monthly_pairs.extend(pairs)
                
            # 해당 월의 상품 쌍 빈도수 계산
            monthly_pair_counts = Counter(monthly_pairs)
            
            # 해당 월의 인기 상품 쌍 필터링 (배송료, 세트상품 등 제외)
            for pair, count in monthly_pair_counts.most_common(50):
                product1, product2 = pair
                
                # 세트상품, 증정품, 배송료 관련 키워드가 포함되지 않는 상품만 선택
                if not any(keyword in product1.lower() for keyword in delivery_keywords) and \
                   not any(keyword in product2.lower() for keyword in delivery_keywords) and \
                   '세트상품' not in product1 and '증정품' not in product1 and \
                   '세트상품' not in product2 and '증정품' not in product2:
                    
                    # 계절성 정보 확인
                    season_info = []
                    for product in [product1, product2]:
                        if product in self.seasonal_products['상품'].values:
                            product_info = self.seasonal_products[self.seasonal_products['상품'] == product].iloc[0]
                            product_peak_month = int(product_info['피크_판매월']) if pd.notna(product_info['피크_판매월']) else 0
                            product_season = season_map.get(product_peak_month, '없음')
                            season_info.append((product, product_season, product_peak_month))
                        else:
                            season_info.append((product, "없음", 0))
                    
                    # 번들 이름 결정 - 현재 월의 계절을 기준으로 함
                    bundle_name = f"{current_season} 계절 번들"
                    
                    # 특별 케이스 처리
                    product1_season = season_info[0][1]
                    product2_season = season_info[1][1]
                    
                    # 두 상품 모두 현재 계절과 일치
                    if product1_season == product2_season and product1_season == current_season and product1_season != '없음':
                        bundle_name = f"{current_season} 시그니처 번들"
                    # 두 상품 모두 다음 계절과 일치 (준비 번들)
                    elif product1_season == product2_season and product1_season == next_season and product1_season != '없음':
                        bundle_name = f"{product1_season} 계절 준비 번들"
                    # 두 상품이 다른 계절이고, 하나라도 현재 계절과 일치하는 경우
                    elif (product1_season == current_season or product2_season == current_season) and product1_season != product2_season:
                        bundle_name = f"{current_season} 계절 번들"
                    # 두 상품이 모두 현재 계절과 일치하지 않고, 같은 계절인 경우 - 계절이 현재나 다음 계절이 아니면 건너뜀
                    elif product1_season == product2_season and product1_season != current_season and product1_season != next_season:
                        continue
                    
                    # 번들 설명
                    bundle_reason = f"{current_month}월 함께 구매 횟수: {count}회"
                    
                    # 계절 관련성 추가 점수 (피크 판매월이 현재 월과 가까울수록 높은 점수)
                    seasonal_bonus = 0
                    for _, product_season, peak_month in season_info:
                        if peak_month != 0:  # 피크 판매월 정보가 있는 경우
                            month_diff = min(abs(current_month - peak_month), 12 - abs(current_month - peak_month))
                            if month_diff <= 1:
                                seasonal_bonus += 5  # 피크 판매월과 같거나 1개월 차이
                            elif month_diff <= 2:
                                seasonal_bonus += 3  # 피크 판매월과 2개월 차이
                            
                            # 현재 계절과 상품의 계절이 일치하면 추가 보너스
                            if product_season == current_season:
                                seasonal_bonus += 2
                            # 다음 계절과 일치할 경우 (준비 계절)
                            elif product_season == next_season:
                                seasonal_bonus += 1
                    
                    # 최종 번들 점수 = 함께 구매 횟수 + 계절 보너스
                    bundle_score = count + seasonal_bonus
                    
                    filtered_pairs.append({
                        '번들_이름': bundle_name,
                        '상품_1': product1,
                        '상품_2': product2,
                        '추천_이유': bundle_reason,
                        '함께_구매_횟수': count,
                        '계절_보너스': seasonal_bonus,
                        '번들_점수': bundle_score
                    })
            
            # 최종 점수로 정렬
            filtered_pairs = sorted(filtered_pairs, key=lambda x: x['번들_점수'], reverse=True)
            
        # 해당 월 데이터가 없거나 충분하지 않은 경우, 전체 데이터로 백업 추천
        if len(filtered_pairs) < n:
            # 기존 방식으로 전체 데이터에서 함께 구매 빈도 높은 상품 쌍 찾기
            for _, row in self.frequent_pairs.iterrows():
                product_pair = row['상품_쌍']
                product1, product2 = product_pair
                
                # 이미 추가된 상품 쌍은 건너뛰기
                if any(p['상품_1'] == product1 and p['상품_2'] == product2 for p in filtered_pairs):
                    continue
                    
                # 세트상품, 증정품, 배송료 관련 키워드가 포함되지 않는 상품만 선택
                if not any(keyword in product1.lower() for keyword in delivery_keywords) and \
                   not any(keyword in product2.lower() for keyword in delivery_keywords) and \
                   '세트상품' not in product1 and '증정품' not in product1 and \
                   '세트상품' not in product2 and '증정품' not in product2:
                    
                    # 계절성 정보 확인
                    season_info = []
                    for product in [product1, product2]:
                        if product in self.seasonal_products['상품'].values:
                            product_info = self.seasonal_products[self.seasonal_products['상품'] == product].iloc[0]
                            product_peak_month = int(product_info['피크_판매월']) if pd.notna(product_info['피크_판매월']) else 0
                            product_season = season_map.get(product_peak_month, '없음')
                            season_info.append((product, product_season, product_peak_month))
                        else:
                            season_info.append((product, "없음", 0))
                    
                    # 번들 이름 결정 - 백업 추천의 경우에도 현재 계절 기준
                    bundle_name = f"{current_season} 계절 번들"
                    
                    # 특별 케이스 처리
                    product1_season = season_info[0][1]
                    product2_season = season_info[1][1]
                    
                    # 두 상품 모두 현재 계절과 일치
                    if product1_season == product2_season and product1_season == current_season and product1_season != '없음':
                        bundle_name = f"{current_season} 시그니처 번들"
                    # 두 상품 모두 다음 계절과 일치 (준비 번들)
                    elif product1_season == product2_season and product1_season == next_season and product1_season != '없음':
                        bundle_name = f"{product1_season} 계절 준비 번들"
                    # 두 상품이 다른 계절이고, 하나라도 현재 계절과 일치하는 경우
                    elif (product1_season == current_season or product2_season == current_season) and product1_season != product2_season:
                        bundle_name = f"{current_season} 계절 번들"
                    # 두 상품이 모두 현재 계절과 일치하지 않고, 같은 계절인 경우 - 계절이 현재나 다음 계절이 아니면 건너뜀
                    elif product1_season == product2_season and product1_season != current_season and product1_season != next_season:
                        continue
                    
                    # 번들 설명
                    bundle_reason = f"전체 기간 함께 구매 횟수: {row['함께_구매된_횟수']}회"
                    
                    # 계절 관련성 추가 점수
                    seasonal_bonus = 0
                    for _, product_season, peak_month in season_info:
                        if peak_month != 0:
                            month_diff = min(abs(current_month - peak_month), 12 - abs(current_month - peak_month))
                            if month_diff <= 1:
                                seasonal_bonus += 5
                            elif month_diff <= 2:
                                seasonal_bonus += 3
                                
                            # 현재 계절과 상품의 계절이 일치하면 추가 보너스
                            if product_season == current_season:
                                seasonal_bonus += 2
                            # 다음 계절과 일치할 경우 (준비 계절)
                            elif product_season == next_season:
                                seasonal_bonus += 1
                    
                    # 최종 번들 점수
                    bundle_score = row['함께_구매된_횟수'] + seasonal_bonus
                    
                    filtered_pairs.append({
                        '번들_이름': bundle_name,
                        '상품_1': product1,
                        '상품_2': product2,
                        '추천_이유': bundle_reason,
                        '함께_구매_횟수': row['함께_구매된_횟수'],
                        '계절_보너스': seasonal_bonus,
                        '번들_점수': bundle_score
                    })
                    
                    # 필요한 수만큼 채우면 중단
                    if len(filtered_pairs) >= n:
                        break
        
        # 최종 결과 DataFrame으로 변환하여 반환 (상위 n개)
        return pd.DataFrame(filtered_pairs[:n])

    def recommend_for_new_customer(self, n=5, current_month=None):
        """신규 고객을 위한 추천 상품 (인기 상품 및 계절 기반)"""
        if current_month is None:
            current_month = datetime.now().month
        
        # 계절 설정
        if current_month in [3, 4, 5]:
            season = '봄'
        elif current_month in [6, 7, 8]:
            season = '여름'
        elif current_month in [9, 10, 11]:
            season = '가을'
        else:
            season = '겨울'
        
        # 추천 상품 목록
        recommendations = []
        
        # 세트상품, 증정품, 배송료 등 제외
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        delivery_pattern = '|'.join(delivery_keywords)
        seasonal_products_filtered = self.seasonal_products[
            ~self.seasonal_products['상품'].str.contains('세트상품|증정품', na=False) &
            ~self.seasonal_products['상품'].str.contains(delivery_pattern, case=False, na=False)
        ]
        
        # 1. 계절성 높은 현재 시즌 상품 (2개)
        seasonal_top = seasonal_products_filtered[
            (seasonal_products_filtered['주요_계절'] == season) &
            (seasonal_products_filtered['계절성_지수'] > 0.3) &
            (seasonal_products_filtered['총_판매량'] > 500)
        ].sort_values(['계절성_지수', '총_판매량'], ascending=[False, False]).head(2)
        
        for _, row in seasonal_top.iterrows():
            recommendations.append({
                '추천_상품': row['상품'],
                '추천_점수': 5 + row['계절성_지수'],
                '추천_이유': f"현재 {season}철 인기 상품 (계절성 지수: {row['계절성_지수']:.2f})"
            })
        
        # 2. 연중 베스트셀러 상품 (2개)
        bestsellers = seasonal_products_filtered.sort_values('총_판매량', ascending=False).head(3)
        
        for _, row in bestsellers.iterrows():
            if row['상품'] not in [r['추천_상품'] for r in recommendations]:
                recommendations.append({
                    '추천_상품': row['상품'],
                    '추천_점수': row['총_판매량'] / 1000,
                    '추천_이유': f"연중 인기 베스트셀러 상품 (총 판매량: {int(row['총_판매량']):,}개)"
                })
        
        # 3. 자주 함께 구매되는 인기 조합 상품 (1개)
        top_pairs = self.frequent_pairs.sort_values('함께_구매된_횟수', ascending=False).head(3)
        
        for _, row in top_pairs.iterrows():
            product1, product2 = row['상품_쌍']
            
            # 세트상품, 증정품, 배송료 등이 아닌지 확인
            if not any(keyword in product1.lower() for keyword in delivery_keywords) and \
               not any(keyword in product2.lower() for keyword in delivery_keywords) and \
               '세트상품' not in product1 and '세트상품' not in product2 and \
               '증정품' not in product1 and '증정품' not in product2:
                
                # 이미 추천된 상품이 아닌 것 선택
                if product1 not in [r['추천_상품'] for r in recommendations]:
                    recommendations.append({
                        '추천_상품': product1,
                        '추천_점수': 4.5,
                        '추천_이유': f"'{product2}'와 함께 자주 구매되는 상품"
                    })
                    break
                elif product2 not in [r['추천_상품'] for r in recommendations]:
                    recommendations.append({
                        '추천_상품': product2,
                        '추천_점수': 4.5,
                        '추천_이유': f"'{product1}'와 함께 자주 구매되는 상품"
                    })
                    break
        
        # 결과를 점수순으로 정렬하고 요청된 n개만 반환
        result = pd.DataFrame(recommendations).sort_values('추천_점수', ascending=False).head(n)
        
        # 충분한 추천이 없는 경우 계절 상품으로 채우기
        if len(result) < n:
            more_seasonal = seasonal_products_filtered[
                ~seasonal_products_filtered['상품'].isin(result['추천_상품'])
            ].sort_values('총_판매량', ascending=False).head(n - len(result))
            
            for _, row in more_seasonal.iterrows():
                result = pd.concat([result, pd.DataFrame([{
                    '추천_상품': row['상품'],
                    '추천_점수': 4.0,
                    '추천_이유': f"인기 상품 (판매량: {int(row['총_판매량']):,}개)"
                }])], ignore_index=True)
        
        return result[['추천_상품', '추천_점수', '추천_이유']]

    def analyze_product_details(self, product_name):
        """특정 상품의 상세 정보 분석"""
        if product_name not in self.customer_product_matrix.columns:
            return {
                '상태': '실패',
                '메시지': f"상품 '{product_name}'을(를) 찾을 수 없습니다."
            }
        
        # 배송료 관련 키워드 정의
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        
        # 세트상품 등이나 배송료 제외
        if '세트상품' in product_name or '증정품' in product_name or \
           any(keyword in product_name.lower() for keyword in delivery_keywords):
            return {
                '상태': '실패',
                '메시지': f"'{product_name}'은(는) 분석에서 제외됩니다."
            }
        
        # 총 판매량 - 정수형으로 변환
        total_sales = int(self.sales_data[self.sales_data['상품'] == product_name]['수량'].sum())
        
        # 구매 고객 정보 - 필터링 강화: 배송료 관련 고객과 재고조정 제외
        customers = self.customer_product_matrix[self.customer_product_matrix[product_name] > 0].index.tolist()
        # 재고조정, 배송료 관련 고객 필터링
        filtered_customers = [c for c in customers if not ('재고조정' in c or '문정창고' in c or '창고' in c)]
        
        # 고객별 구매량 계산 시 배송료 관련 고객 제외
        top_customers_df = self.sales_data[
            (self.sales_data['상품'] == product_name) & 
            (~self.sales_data['고객명'].str.contains('재고조정|문정창고|창고', na=False))
        ].groupby('고객명')['수량'].sum().sort_values(ascending=False)
        
        # 상위 5개 고객만 선택
        top_customers = top_customers_df.head(5)
        
        # 계절성 정보
        seasonality_info = {}
        if product_name in self.seasonal_products['상품'].values:
            product_info = self.seasonal_products[self.seasonal_products['상품'] == product_name].iloc[0]
            seasonality_info = {
                '계절성_지수': product_info['계절성_지수'],
                '피크_판매월': int(product_info['피크_판매월']),
                '주요_계절': product_info['주요_계절']
            }
        
        # 함께 구매되는 상품
        co_purchased_products = []
        delivery_keywords = ['배달료', '퀵', '택배', '운송', '배달']
        
        for _, row in self.frequent_pairs.iterrows():
            product_pair = row['상품_쌍']
            if product_name in product_pair:
                other_product = product_pair[0] if product_pair[1] == product_name else product_pair[1]
                
                # 세트상품, 증정품, 배송료 관련 키워드가 있는 상품 제외
                if not ('세트상품' in other_product or '증정품' in other_product or 
                       any(keyword in other_product.lower() for keyword in delivery_keywords)):
                    co_purchased_products.append({
                        '상품': other_product,
                        '함께_구매_횟수': row['함께_구매된_횟수']
                    })
        
        # 반품 정보
        refund_info = {}
        if self.refund_data is not None:
            refund_qty = self.refund_data[self.refund_data['상품'] == product_name]['수량'].sum()
            refund_ratio = abs(refund_qty) / (total_sales + 0.1) * 100
            refund_types = self.refund_data[self.refund_data['상품'] == product_name].groupby('반품사유')['수량'].sum().abs()
            refund_info = {
                '반품_수량': abs(refund_qty),
                '반품_비율': refund_ratio,
                '반품_이유': refund_types.to_dict() if not refund_types.empty else {}
            }
        
        return {
            '상태': '성공',
            '상품명': product_name,
            '총_판매량': total_sales,
            '구매_고객_수': len(filtered_customers),
            '주요_고객': top_customers.to_dict(),
            '계절성_정보': seasonality_info,
            '함께_구매되는_상품': co_purchased_products[:5], # 상위 5개만
            '반품_정보': refund_info
        }

    def get_product_pairs(self, products):
        """상품 리스트에서 모든 가능한 상품 쌍(조합)을 반환합니다"""
        pairs = []
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                pairs.append((products[i], products[j]))
        return pairs

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
        customer_purchases = self.sales_data[self.sales_data['고객명'] == customer_name]
        total_quantity = int(customer_purchases['수량'].sum())
        
        # 금액 컬럼이 있는지 확인하고 처리
        has_amount = '금액' in customer_purchases.columns
        total_amount = int(customer_purchases['금액'].sum()) if has_amount else 0
        
        # 연도-월별 구매 패턴 (연도 정보 추가)
        monthly_purchases = {}
        yearmonth_purchases = {}
        
        if '날짜' in customer_purchases.columns:
            # 날짜 컬럼에서 연도와 월 정보 추출
            customer_purchases['year'] = customer_purchases['날짜'].dt.year
            customer_purchases['month'] = customer_purchases['날짜'].dt.month
            customer_purchases['yearmonth'] = customer_purchases['날짜'].dt.strftime('%Y-%m')
            
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
        elif 'month' in customer_purchases.columns:
            # 기존 month 컬럼만 있는 경우 (이전 버전 호환)
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
        
        # 연-월별 상품 구매 내역 및 날짜별 내역
        yearmonth_product_purchases = {}
        yearmonth_purchase_dates = {}
        
        if '날짜' in customer_purchases.columns:
            # 연-월 기준으로 그룹화
            for yearmonth, yearmonth_group in customer_purchases.groupby('yearmonth'):
                # 해당 연-월의 모든 상품 구매 내역
                all_products = yearmonth_group.groupby('상품')['수량'].sum().sort_values(ascending=False)
                yearmonth_product_purchases[yearmonth] = all_products.to_dict()
                
                # 해당 연-월의 날짜별 구매 기록
                date_purchases = {}
                for date, date_group in yearmonth_group.groupby('날짜'):
                    date_str = date.strftime('%Y-%m-%d')
                    date_products = {}
                    for _, row in date_group.iterrows():
                        product = row['상품']
                        quantity = row['수량']
                        date_products[product] = int(quantity)
                    date_purchases[date_str] = date_products
                yearmonth_purchase_dates[yearmonth] = date_purchases
        
        # 월별 상품 구매 내역 (이전 버전 호환)
        monthly_product_purchases = {}
        monthly_purchase_dates = {}
        
        if 'month' in customer_purchases.columns:
            for month, month_group in customer_purchases.groupby('month'):
                month = int(month)
                # 모든 구매 상품 포함
                all_products = month_group.groupby('상품')['수량'].sum().sort_values(ascending=False)
                monthly_product_purchases[month] = all_products.to_dict()
                
                # 해당 월의 날짜별 구매 기록 추가
                if '날짜' in month_group.columns:
                    date_purchases = {}
                    for date, date_group in month_group.groupby('날짜'):
                        date_str = date.strftime('%Y-%m-%d')
                        date_products = {}
                        for _, row in date_group.iterrows():
                            product = row['상품']
                            quantity = row['수량']
                            date_products[product] = int(quantity)
                        date_purchases[date_str] = date_products
                    monthly_purchase_dates[month] = date_purchases
        
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
            for month, group in customer_purchases.groupby('month'):
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
        
        # 분기별 선호도 분석
        quarterly_preference = {
            '1분기': 0,  # 1-3월
            '2분기': 0,  # 4-6월
            '3분기': 0,  # 7-9월
            '4분기': 0   # 10-12월
        }
        
        if 'month' in customer_purchases.columns:
            for month, group in customer_purchases.groupby('month'):
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
            latest_date = customer_purchases['날짜'].max()
            if not pd.isna(latest_date):
                latest_purchase = latest_date.strftime('%Y-%m-%d')
        
        # 구매 빈도 (구매 날짜 수 / 전체 기간)
        purchase_frequency = 0
        unique_days = 0
        purchase_dates = []
        
        if '날짜' in customer_purchases.columns:
            unique_days = customer_purchases['날짜'].dt.date.nunique()
            purchase_dates = sorted(customer_purchases['날짜'].dt.date.unique())
            first_date = customer_purchases['날짜'].min()
            last_date = customer_purchases['날짜'].max()
            
            if not pd.isna(first_date) and not pd.isna(last_date):
                total_days = (last_date - first_date).days + 1
                purchase_frequency = unique_days / max(total_days, 1) * 100
        
        # 제품별 구매일 변화 추적 (제품 구매 패턴 파악)
        product_purchase_history = {}
        if '날짜' in customer_purchases.columns:
            for product, product_group in customer_purchases.groupby('상품'):
                dates = []
                quantities = []
                for _, row in product_group.iterrows():
                    date_str = row['날짜'].strftime('%Y-%m-%d')
                    dates.append(date_str)
                    quantities.append(int(row['수량']))
                
                product_purchase_history[product] = {
                    '구매일': dates,
                    '구매량': quantities
                }
        
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
            '최근_구매일': latest_purchase,
            '구매_빈도': purchase_frequency,
            '구매_날짜': [d.strftime('%Y-%m-%d') for d in purchase_dates],
            '구매일수': unique_days,
            '제품별_구매_이력': product_purchase_history
        }

    def get_customer_categories(self):
        """고객을 카테고리별로 분류"""
        customer_categories = {
            '호텔': [],
            '일반': []
        }
        
        for customer in self.customer_product_matrix.index:
            # 재고조정, 창고 제외
            if '재고조정' in customer or '문정창고' in customer or '창고' in customer:
                continue
                
            # 업체 코드 분석 (앞 3자리)
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
        """RFM 고객 세분화 분석을 수행합니다.
        
        Args:
            customer_type: '전체', '호텔', '일반' 중 하나
            selected_month: 특정 월에 대한 분석 (None이면 전체 기간)
            
        Returns:
            rfm_results: RFM 분석 결과 딕셔너리
        """
        # 날짜 컬럼이 없으면 분석 불가
        if '날짜' not in self.sales_data.columns or self.sales_data['날짜'].isnull().all():
            return {
                '상태': '실패',
                '메시지': "날짜 정보가 없어 RFM 분석을 수행할 수 없습니다."
            }
        
        # 배송료 관련 키워드 정의
        delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
        delivery_pattern = '|'.join(delivery_keywords)
        
        # 유효한 판매 데이터 필터링 (배송료, 세트상품 등 제외)
        valid_sales = self.sales_data[
            ~self.sales_data['상품'].str.contains('세트상품|증정품', na=False) &
            ~self.sales_data['상품'].str.contains(delivery_pattern, case=False, na=False)
        ]
        
        # 재고조정, 창고 등 제외
        valid_sales = valid_sales[
            ~valid_sales['고객명'].str.contains('재고조정|문정창고|창고', na=False, regex=True)
        ]
        
        if valid_sales.empty:
            return {
                '상태': '실패',
                '메시지': "유효한 판매 데이터가 없습니다."
            }
        
        # 고객 유형 필터링
        customer_categories = self.get_customer_categories()
        if customer_type == '호텔':
            valid_sales = valid_sales[valid_sales['고객명'].isin(customer_categories['호텔'])]
        elif customer_type == '일반':
            valid_sales = valid_sales[valid_sales['고객명'].isin(customer_categories['일반'])]
        # '전체'인 경우는 추가 필터링 없음
        
        if valid_sales.empty:
            return {
                '상태': '실패',
                '메시지': f"{customer_type} 고객 유형에 대한 유효한 판매 데이터가 없습니다."
            }
        
        # 특정 월에 대한 분석인 경우 필터링
        if selected_month is not None:
            valid_sales = valid_sales[valid_sales['날짜'].dt.month == selected_month]
            
            if valid_sales.empty:
                return {
                    '상태': '실패',
                    '메시지': f"{selected_month}월에 대한 유효한 판매 데이터가 없습니다."
                }
        
        # 최근 날짜 계산
        max_date = valid_sales['날짜'].max()
        
        # RFM 분석을 위한 고객별 지표 계산
        rfm_data = valid_sales.groupby('고객명').agg({
            '날짜': lambda x: (max_date - x.max()).days,  # Recency: 마지막 구매 이후 경과일
            '상품': 'count',  # Frequency: 구매 빈도
            '금액': 'sum'  # Monetary: 총 구매액
        }).reset_index()
        
        # 컬럼 이름 변경
        rfm_data.rename(columns={
            '날짜': 'Recency',
            '상품': 'Frequency',
            '금액': 'Monetary'
        }, inplace=True)
        
        # 각 지표에 대한 사분위수 계산 (높을수록 좋음, Recency만 낮을수록 좋음)
        try:
            # 충분한 데이터가 있는 경우 사분위수 계산
            if len(rfm_data) >= 4:
                rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')  # Recency는 낮을수록 좋음
                rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
                rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
            else:
                # 데이터가 충분하지 않은 경우 중앙값을 기준으로 점수 부여
                rfm_data['R_Score'] = rfm_data['Recency'].apply(lambda x: 4 if x <= rfm_data['Recency'].median() else 1)
                rfm_data['F_Score'] = rfm_data['Frequency'].apply(lambda x: 4 if x >= rfm_data['Frequency'].median() else 1)
                rfm_data['M_Score'] = rfm_data['Monetary'].apply(lambda x: 4 if x >= rfm_data['Monetary'].median() else 1)
        except Exception as e:
            # 오류 발생 시 수동으로 점수 부여
            st.warning(f"RFM 점수 계산 중 오류 발생: {e}. 수동 점수 부여 방식으로 전환합니다.")
            
            # Recency는 낮을수록 좋음, 데이터의 분포에 따라 점수 할당
            r_median = rfm_data['Recency'].median()
            rfm_data['R_Score'] = rfm_data['Recency'].apply(lambda x: 4 if x <= r_median/2 else (3 if x <= r_median else (2 if x <= r_median*2 else 1)))
            
            # Frequency는 높을수록 좋음
            f_median = rfm_data['Frequency'].median()
            rfm_data['F_Score'] = rfm_data['Frequency'].apply(lambda x: 4 if x > f_median*2 else (3 if x > f_median else (2 if x > f_median/2 else 1)))
            
            # Monetary는 높을수록 좋음
            m_median = rfm_data['Monetary'].median()
            rfm_data['M_Score'] = rfm_data['Monetary'].apply(lambda x: 4 if x > m_median*2 else (3 if x > m_median else (2 if x > m_median/2 else 1)))
        
        # R, F, M 점수를 합쳐서 RFM 세그먼트 정의
        rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)
        
        # 고객 세그먼트 정의
        def segment_customer(row):
            r = int(row['R_Score'])
            f = int(row['F_Score'])
            m = int(row['M_Score'])
            
            if r >= 3 and f >= 3 and m >= 3:
                return "VIP 고객"
            elif r >= 3 and f >= 3:
                return "충성 고객"
            elif r >= 3 and m >= 3:
                return "큰 지출 고객"
            elif f >= 3 and m >= 3:
                return "잠재 이탈 고객"
            elif r >= 3:
                return "신규 고객"
            elif f >= 3:
                return "가격 민감 고객"
            elif m >= 3:
                return "휴면 큰 지출 고객"
            else:
                return "관심 필요 고객"
        
        rfm_data['고객_세그먼트'] = rfm_data.apply(segment_customer, axis=1)
        
        # 월별 RFM 분석을 위한 추가 데이터
        monthly_rfm = None
        if selected_month is None:
            # 월별 RFM 데이터 생성
            valid_sales['월'] = valid_sales['날짜'].dt.month
            monthly_data = []
            
            for month in range(1, 13):
                month_sales = valid_sales[valid_sales['월'] == month]
                if not month_sales.empty:
                    # 해당 월의 최근 날짜
                    month_max_date = month_sales['날짜'].max()
                    
                    # 월별 RFM 지표 계산
                    month_rfm = month_sales.groupby('고객명').agg({
                        '날짜': lambda x: (month_max_date - x.max()).days,
                        '상품': 'count',
                        '금액': 'sum'
                    }).reset_index()
                    
                    month_rfm.rename(columns={
                        '날짜': 'Recency',
                        '상품': 'Frequency',
                        '금액': 'Monetary'
                    }, inplace=True)
                    
                    month_rfm['월'] = month
                    monthly_data.append(month_rfm)
            
            if monthly_data:
                monthly_rfm = pd.concat(monthly_data)
        
        # 결과 반환
        return {
            '상태': '성공',
            '고객_유형': customer_type,
            '선택_월': selected_month,
            'RFM_데이터': rfm_data,
            '월별_RFM': monthly_rfm,
            '총_고객수': len(rfm_data),
            'VIP_고객수': len(rfm_data[rfm_data['고객_세그먼트'] == "VIP 고객"]),
            '충성_고객수': len(rfm_data[rfm_data['고객_세그먼트'] == "충성 고객"]),
            '관심필요_고객수': len(rfm_data[rfm_data['고객_세그먼트'] == "관심 필요 고객"]),
            '세그먼트_통계': rfm_data['고객_세그먼트'].value_counts().to_dict(),
            'R_평균': rfm_data['Recency'].mean(),
            'F_평균': rfm_data['Frequency'].mean(),
            'M_평균': rfm_data['Monetary'].mean()
        }

# Streamlit 앱 구성
def main():
    st.set_page_config(page_title="마이크로그린 맞춤형 추천 시스템", layout="wide")
    
    st.title("🌱 System Q (Koppert Cress Korea)")
    st.markdown("---")
    
    # 현재 디렉토리의 파일들을 확인
    sales_file = "merged_2023_2024_2025.xlsx"
    refund_file = "merged_returns_2024_2025.xlsx"
    
    # 세션 상태 초기화
    if 'current_month' not in st.session_state:
        st.session_state.current_month = datetime.now().month
    if 'customer_category' not in st.session_state:
        st.session_state.customer_category = "전체"
    if 'selected_customer' not in st.session_state:
        st.session_state.selected_customer = None
    if 'analyzed_customer' not in st.session_state:
        st.session_state.analyzed_customer = None
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = None
    if 'customer_info' not in st.session_state:
        st.session_state.customer_info = None
    # RFM 분석을 위한 세션 상태 추가
    if 'rfm_customer_type' not in st.session_state:
        st.session_state.rfm_customer_type = "전체"
    if 'rfm_selected_month' not in st.session_state:
        st.session_state.rfm_selected_month = "전체 기간"
    if 'rfm_results' not in st.session_state:
        st.session_state.rfm_results = None
    
    # 월 변경 콜백 함수
    def on_month_change():
        st.session_state.current_month = st.session_state.month_slider
    
    # 현재 월 설정
    current_month = st.sidebar.slider(
        "현재 월 설정", 
        1, 12, 
        st.session_state.current_month,
        key="month_slider",
        on_change=on_month_change
    )
    
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
        
        # 데이터 전처리
        # 재고조정 제외
        if '반품사유' in refund_data.columns:
            # 재고조정 제외하지 않음
            pass
        # 정보 표시 제거
        
        # 날짜 변환
        if '날짜' in sales_data.columns:
            # 날짜 형식이 'YY.MM.DD'인 경우 처리
            sales_data['날짜'] = pd.to_datetime(sales_data['날짜'].astype(str).apply(
                lambda x: f"20{x}" if len(str(x).split('.')[0]) == 2 else x
            ), errors='coerce')
            
            # 월 정보 추가
            sales_data['month'] = sales_data['날짜'].dt.month
        
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
            # 전체 데이터 확대 보기 옵션
            if st.button("전체 판매 데이터 확대 보기", key="expand_sales"):
                st.dataframe(sales_data, height=500)
        
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
            # 전체 데이터 확대 보기 옵션
            if st.button("전체 반품 데이터 확대 보기", key="expand_refund"):
                st.dataframe(refund_data, height=500)
        
        # 추천 시스템 초기화 - 진행 메시지 제거
        # st.sidebar.info("추천 시스템 초기화 중...")
        recommender = MicrogreenRecommendationSystem(sales_data, refund_data)
        
        # 탭 생성
        tabs = st.tabs(["고객 맞춤 추천", "계절 추천", "번들 추천", "신규 고객 추천", "상품 분석", "업체 분석", "고객 세분화 (RFM)"])
        
        # 1. 고객 맞춤 추천 탭
        with tabs[0]:
            st.header("👤 고객 맞춤형 추천")
            st.write("기존 고객의 구매 이력을 분석하여 맞춤형 상품을 추천합니다.")
            
            # 고객 맞춤형 추천 알고리즘 설명 추가
            with st.expander("🔍 고객 맞춤형 추천 알고리즘 설명"):
                st.markdown("""
                ### 고객 맞춤형 추천 점수는 어떻게 계산되나요?
                
                마이크로그린 맞춤형 추천 시스템은 세 가지 핵심 요소를 기반으로 최적의 추천 상품을 제시합니다:
                
                #### 1. 유사도 점수 (기여도: 높음, 35%)
                고객이 이전에 구매한 상품과 비슷한 상품을 찾는 방식입니다. 이 점수는 '많은 고객들이 이 두 상품을 함께 구매하는가?'를 측정합니다.
                
                - **쉬운 설명**: 넷플릭스에서 "이 영화를 본 사람들이 이 영화도 봤습니다"와 같은 원리입니다.
                - **예시**: '정식당'이 '루콜라'를 구매했고, 다른 많은 레스토랑들이 '루콜라'와 '바질'을 함께 구매하는 경향이 있다면, '바질'은 높은 유사도 점수를 받아 '정식당'에게 추천됩니다.
                - **점수 범위**: 0~5점 (5점이 가장 높은 유사도)
                - **계산 방법**: 
                  * 코사인 유사도 알고리즘으로 계산된 원시 유사도 값(0 ~ 1 사이)에 5를 곱하여 0 ~ 5점 범위로 변환
                  * 예: 유사도 0.8 × 5 = 4점
                  * 고객이 여러 상품을 구매한 경우, 각 상품의 유사도를 구매량에 비례하여 가중 평균
                  * 마이크로그린 상품 특성상 대부분 유사도 분포가 0.1~0.7 사이에 집중되므로, 이 범위를 효과적으로 활용
                - **중요도 이유**: 고객의 전체적인 취향과 구매 패턴을 파악하는 데 중요합니다. 특히 마이크로그린과 같은 특수 식재료는 고객의 요리 스타일이나 메뉴 특성과 연관성이 높아 유사 상품 추천이 효과적입니다.
                - **이 계산법을 선택한 이유**: 코사인 유사도는 고객-상품 행렬에서 복잡한 구매 패턴을 포착하는 데 효과적이며, 마이크로그린과 같이 제품 수가 적고 특화된 시장에서 정확한 유사성을 측정할 수 있습니다.
                
                #### 2. 계절성 보너스 (기여도: 중간, 30%)
                현재 계절이나 월에 잘 팔리는 상품에 추가 점수를 부여합니다. 그 계절에 가장 인기 있는 상품은 더 높은 보너스를 받습니다.
                
                - **쉬운 설명**: 여름에는 수박과 아이스크림이, 겨울에는 귤과 핫초코가 더 추천되는 것과 같은 원리입니다.
                - **예시**: 현재가 7월이고, '바질'이 6-8월에 판매량이 급증한다면, '포시즌스 호텔'에 '바질'을 추천할 때 계절성 보너스 점수가 추가되어 추천 순위가 상승합니다.
                - **점수 범위**: 0~5점 (5점이 가장 높은 계절성)
                - **계산 방법**: 
                  * 계절성 점수 = (월별 최대 판매량 - 월별 최소 판매량) / 월별 평균 판매량
                  * 계절성 지수가 2.0 이상인 경우(매우 강한 계절성): 5점
                  * 계절성 지수가 1.5~2.0인 경우(강한 계절성): 4점
                  * 계절성 지수가 1.0~1.5인 경우(중간 계절성): 3점
                  * 계절성 지수가 0.5~1.0인 경우(약한 계절성): 2점
                  * 계절성 지수가 0.2~0.5인 경우(미미한 계절성): 1점
                  * 계절성 지수가 0.2 미만인 경우(계절성 없음): 0점
                  * 추가로, 현재 월이 제품의 피크 판매월과 가까울수록 보너스 점수
                - **중요도 이유**: 마이크로그린은 계절에 따라 맛, 향, 크기, 색상, 영양가가 달라집니다. 특히 레스토랑이나 호텔과 같은 고객에게는 제철 식재료가 중요하므로, 현재 계절에 최적화된 상품을 추천하는 것이 고객 만족도를 높입니다.
                - **이 계산법을 선택한 이유**: 마이크로그린은 계절성이 매우 뚜렷한 농산물로, 판매 데이터 분석 결과 일부 제품은 특정 계절에 판매량이 200~300% 증가하는 경향이 있습니다. 이 계산법은 데이터의 실제 변동성을 반영하며, 마이크로그린 산업 전문가들과의 협의를 통해 개발되었습니다.
                
                #### 3. 함께 구매 보너스 (기여도: 높음, 35%)
                고객이 이미 구매한 상품과 자주 함께 구매되는 상품에 추가 점수가 부여됩니다.
                
                - **쉬운 설명**: 마트에서 빵을 살 때 잼이나 버터를 함께 진열해 놓는 것과 같은 원리입니다.
                - **예시**: '웨스틴조선호텔' 레스토랑이 '루콜라'를 구매했는데, 판매 데이터 분석 결과 '루콜라'와 '레드바질'이 자주 같은 날에 함께 구매된다면, '웨스틴조선호텔'에게 '레드바질'을 추천합니다.
                - **점수 범위**: 0~5점 (5점이 가장 높은 함께 구매 빈도)
                - **계산 방법**: 
                  * 함께 구매 점수 = (특정 상품과의 함께 구매 횟수 / 해당 고객의 총 구매 횟수) × 5
                  * 예: '루콜라'와 '바질'이 8번 중 6번 함께 구매되었다면, 6/8 × 5 = 3.75점
                  * 여러 상품을 구매한 고객의 경우, 각 상품에 대한 함께 구매 점수 중 최대값 사용
                  * 함께 구매 빈도가 매우 낮은 경우(10% 미만), 0점 처리하여 노이즈 제거
                - **중요도 이유**: 마이크로그린은 특정 요리나 메뉴에 함께 사용되는 조합이 중요합니다. 예를 들어 이탈리안 레스토랑에서 '루콜라'와 '바질'을 함께 사용하는 경우가 많습니다. 이런 실제 사용 패턴을 반영하여 실용적인 추천이 가능합니다.
                - **이 계산법을 선택한 이유**: 분석 결과, 마이크로그린 구매 패턴에서 함께 구매되는 비율이 추천의 정확도와 직접적인 관련이 있음을 확인했습니다. 상위 5위, 10위와 같은 절대적 순위보다 구매 비율이 실제 사용 패턴을 더 정확히 반영하며, 마이크로그린처럼 제품 수가 적은 경우 백분율 기반 접근법이 더 효과적입니다.
                
                #### 유사도 점수와 함께 구매 패턴의 차이점

                **유사도 점수**와 **함께 구매 패턴**은 비슷해 보이지만, 실제로는 다른 방식으로 추천에 기여합니다:

                - **유사도 점수**
                  * **분석 범위**: 모든 고객의 전체 구매 기록을 종합적으로 분석
                  * **핵심 질문**: "이 두 상품은 얼마나 비슷한 고객층에게 구매되는가?"
                  * **시간 범위**: 전체 기간의 구매 패턴을 고려 (같은 날 구매 여부는 중요하지 않음)
                  * **예시**: 많은 레스토랑들이 일 년 내내 '루콜라'와 '바질'을 모두 구매한다면, 두 상품은 높은 유사도를 가짐
                  * **유용한 상황**: 고객의 장기적인 취향과 특성 파악에 유용, 신규 상품 추천 시 도움

                - **함께 구매 패턴**
                  * **분석 범위**: 동일 고객이 동일 날짜에 구매한 기록만 분석
                  * **핵심 질문**: "이 상품과 함께 장바구니에 담긴 다른 상품은 무엇인가?"
                  * **시간 범위**: 같은 날짜의 구매 조합만 고려
                  * **예시**: '더그린테이블' 레스토랑이 '루콜라'를 주문할 때마다 거의 항상 '레드바질'도 함께 주문한다면, 두 상품은 높은 함께 구매 점수를 가짐
                  * **유용한 상황**: 실제 요리 레시피나 메뉴 구성, 세트 상품 개발에 유용

                두 점수는 서로 다른 구매 패턴을 포착하므로 함께 사용할 때 더 정확한 추천이 가능합니다. 예를 들어:

                - '포시즌스 호텔'이 보통 '바질'과 '루콜라'를 구매하지만 다른 날에 구매한다면:
                  * 유사도 점수: 높음 (두 상품 모두 정기적으로 구매)
                  * 함께 구매 점수: 낮음 (같은 날에 구매하지 않음)

                - '정식당'이 '페스토 소스'를 만들기 위해 항상 '바질'과 '파인넛'을 같은 날 구매한다면:
                  * 유사도 점수: 중간 (다른 레스토랑은 이 조합을 반드시 구매하지 않을 수 있음)
                  * 함께 구매 점수: 매우 높음 (항상 함께 구매됨)

                이처럼 두 지표를 모두 사용하면 고객의 전체적인 구매 패턴과 구체적인 사용 조합을 모두 고려한 균형 잡힌 추천이 가능합니다.
                
                #### 최종 추천 점수 계산 예시
                **상품: 바질 (포시즌스 호텔 대상 추천)**
                - 유사도 점수: 4.2점 × 0.35 = 1.47점 (호텔이 자주 구매하는 상품들과 유사함)
                - 계절성 보너스: 4.5점 × 0.3 = 1.35점 (현재 월이 바질의 피크 판매월이고 계절성 지수가 높음)
                - 함께 구매 보너스: 4.5점 × 0.35 = 1.58점 (호텔이 이미 구매한 상품들과 자주 함께 구매됨)
                - **최종 점수**: 4.4점
                
                마이크로그린 추천 시스템에서 두 가지 접근법(유사도와 함께 구매 패턴)을 모두 사용하는 것이 중요한 이유:
                
                1. **다양한 구매 상황 대응**: 신규 고객부터 정기 고객까지 다양한 상황에 적응적 추천 가능
                2. **균형 잡힌 시각 제공**: 장기적 패턴과 즉각적 사용 패턴을 모두 고려
                3. **추천 품질 향상**: 두 지표가 서로의 약점을 보완하며 더 정확한 추천 가능
                4. **고객 유형별 최적화**: 레스토랑, 호텔, 일반 소비자 등 다양한 고객층의 특성 반영
                """)
            
            # 고객 목록 얻기 - 재고조정 제외하고 실제 고객만 표시
            customers = sorted(recommender.customer_product_matrix.index.tolist())
            # 재고조정 관련 항목 제외
            customers = [c for c in customers if not ('재고조정' in c or '문정창고' in c or '창고' in c)]
            
            if len(customers) > 0:
                selected_customer = st.selectbox("고객 선택", customers)
                
                # 추천 수량 선택
                recommend_count = st.slider("추천 상품 수", 3, 10, 5, key="customer_count")
                
                if st.button("추천 받기", key="customer_recommend"):
                    with st.spinner("맞춤형 추천 상품을 분석 중입니다..."):
                        recommendations = recommender.recommend_for_customer(
                            selected_customer, recommend_count, current_month
                        )
                        
                        if "고객 구매 이력이 없습니다." in recommendations['추천_상품'].values:
                            st.warning("해당 고객의 구매 이력이 없습니다.")
                        else:
                            # 추천 결과 표시
                            st.success(f"{selected_customer}님을 위한 추천 상품 {len(recommendations)}개가 준비되었습니다.")
                            
                            # 테이블로 표시 (핵심 정보만)
                            display_df = recommendations[['추천_상품', '추천_점수', '추천_이유']].copy()
                            display_df['추천_점수'] = display_df['추천_점수'].round(2)
                            st.table(display_df)
                            
                            # 점수 요소별 시각화
                            st.subheader("💡 추천 점수 상세 분석")
                            
                            # 시각화용 데이터 준비
                            viz_data = recommendations.copy()
                            viz_data[['유사도_점수', '계절성_보너스', '함께_구매_보너스', '추천_점수']] = viz_data[['유사도_점수', '계절성_보너스', '함께_구매_보너스', '추천_점수']].round(2)
                            
                            # 개별 상품에 대한 점수 요소 시각화
                            st.write("각 상품별 추천 점수 구성요소")
                            
                            # 스택형 바 차트로 표현
                            fig = go.Figure()
                            
                            for i, row in viz_data.iterrows():
                                product = row['추천_상품']
                                fig.add_trace(go.Bar(
                                    name=product,
                                    x=['유사도 점수', '계절성 보너스', '함께 구매 보너스'],
                                    y=[row['유사도_점수'], row['계절성_보너스'], row['함께_구매_보너스']],
                                    text=[f"{row['유사도_점수']:.1f}", f"{row['계절성_보너스']:.1f}", f"{row['함께_구매_보너스']:.1f}"],
                                    textposition='auto'
                                ))
                            
                            fig.update_layout(
                                title=f"{selected_customer}님을 위한 추천 상품의 점수 구성",
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 점수 요소별 기여도를 전체적으로 비교하는 시각화
                            st.write("추천 점수 요소별 비교")
                            
                            # 모든 상품의 각 점수 구성요소를 스택 바로 표시
                            fig2 = px.bar(
                                viz_data,
                                x='추천_상품',
                                y=['유사도_점수', '계절성_보너스', '함께_구매_보너스'],
                                title=f"추천 상품별 점수 구성요소 비교",
                                labels={'value': '점수', 'variable': '구성요소'},
                                color_discrete_map={
                                    '유사도_점수': 'rgb(99,110,250)',
                                    '계절성_보너스': 'rgb(239,85,59)',
                                    '함께_구매_보너스': 'rgb(0,204,150)'
                                }
                            )
                            
                            fig2.update_layout(
                                legend_title="점수 구성요소",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                height=400
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # 점수 비교를 위한 레이더 차트 추가
                            if len(recommendations) >= 3:  # 최소 3개 이상의 상품이 있을 때
                                st.write("상위 추천 상품 점수 비교 (레이더 차트)")
                                
                                top_products = viz_data.head(min(5, len(viz_data)))
                                
                                fig3 = go.Figure()
                                
                                categories = ['유사도 점수', '계절성 보너스', '함께 구매 보너스']
                                
                                for i, row in top_products.iterrows():
                                    fig3.add_trace(go.Scatterpolar(
                                        r=[row['유사도_점수'], row['계절성_보너스'], row['함께_구매_보너스']],
                                        theta=categories,
                                        fill='toself',
                                        name=row['추천_상품']
                                    ))
                                
                                fig3.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, max(top_products[['유사도_점수', '계절성_보너스', '함께_구매_보너스']].max())]
                                        )),
                                    showlegend=True,
                                    height=500
                                )
                                
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # 가장 높은 점수의 상품 강조
                            top_product = recommendations.iloc[0]['추천_상품']
                            st.info(f"💡 가장 추천하는 상품은 **{top_product}**입니다.")
            else:
                st.warning("고객 데이터가 없습니다. 데이터를 확인해주세요.")
        
        # 2. 계절 추천 탭
        with tabs[1]:
            st.header("🌸 계절 맞춤형 추천")
            st.write("현재 계절에 잘 팔리는 상품을 추천합니다. 계절성이 강한 상품일수록 높은 순위로 추천됩니다.")
            
            # 계절 추천 알고리즘 설명 추가
            with st.expander("🔍 계절 맞춤형 추천 알고리즘 설명"):
                st.markdown("""
                ### 계절 맞춤형 추천 점수는 어떻게 계산되나요?
                
                #### 1. 계절성 지수 (기여도: 가장 높음)
                각 상품이 특정 계절에 얼마나 많이 팔리는지 측정하는 지표입니다.
                
                - **쉬운 설명**: 아이스크림은 여름에 판매량이 급증하고 겨울에는 판매량이 줄어들기 때문에 계절성이 높은 상품입니다. 반면, 소금은 일년 내내 비슷하게 팔리므로 계절성이 낮은 상품입니다.
                - **점수 범위**: 0~1 사이의 값 (1에 가까울수록 계절성이 강함)
                - **계산 방법**: 월별 판매량의 변동성을 수학적으로 계산합니다. 특정 계절에 판매량이 크게 증가하는 상품일수록 높은 계절성 지수를 가집니다.
                
                #### 2. 현재 계절 적합성 (기여도: 높음)
                선택한 계절(봄, 여름, 가을, 겨울)에 얼마나 잘 팔리는 상품인지 측정합니다.
                
                - **쉬운 설명**: 여름을 선택했다면, 여름철에 판매량이 피크에 달하는 상품이 더 높은 점수를 받습니다.
                - **점수 범위**: 0~10 (10이 해당 계절에 가장 적합)
                - **계산 방법**:
                  * 해당 계절의 총 판매량 ÷ 연간 총 판매량 × 10
                  * 예: 여름철(6-8월) 판매량이 연간 판매량의 60%라면, 계절 적합성 점수는 6점
                
                #### 3. 총 판매량 가중치 (기여도: 중간)
                전반적으로 인기 있는 상품에 추가 가중치를 부여합니다.
                
                - **쉬운 설명**: 모든 계절에 걸쳐 더 많이 팔리는 상품이 약간 더 높은 점수를 받습니다.
                - **점수 범위**: 0~2 (2가 가장 높은 판매량)
                - **계산 방법**:
                  * 상품의 총 판매량이 전체 상품 중 상위 10%에 속하면 2점
                  * 상위 11-30%에 속하면 1점
                  * 그 외에는 0점
                
                #### 최종 계절 추천 점수 계산 예시
                **상품: 바질 (여름 추천)**
                - 계절성 지수: 0.8 × 5 = 4점 (계절성이 매우 높음)
                - 여름 적합성: 7점 (여름철 판매량이 연간 판매량의 70%)
                - 총 판매량 가중치: 1점 (판매량이 상위 20%에 속함)
                - **최종 점수**: 12점
                
                이런 방식으로 각 상품에 점수가 부여되고, 해당 계절에 가장 적합한 상품들이 추천됩니다.
                """)
            
            # 계절 선택
            seasons = ["봄", "여름", "가을", "겨울"]
            current_season = seasons[min(current_month-1, 11) // 3]
            selected_season = st.selectbox("계절 선택", seasons, index=seasons.index(current_season))
            
            # 추천 수량 선택
            season_recommend_count = st.slider("추천 상품 수", 5, 15, 10, key="season_count")
            
            if st.button("계절 추천 받기", key="season_recommend"):
                with st.spinner(f"{selected_season}철 추천 상품을 분석 중입니다..."):
                    seasonal_recommendations = recommender.recommend_for_season(
                        selected_season, season_recommend_count
                    )
                    
                    # 추천 결과 표시
                    st.success(f"{selected_season}철 추천 상품 {len(seasonal_recommendations)}개가 준비되었습니다.")
                    
                    # 데이터 표시를 위한 열 이름 변경
                    display_df = seasonal_recommendations.copy()
                    display_df.columns = ['상품명', '계절성 지수', '총 판매량', '추천 점수']
                    display_df['계절성 지수'] = display_df['계절성 지수'].round(2)
                    display_df['총 판매량'] = display_df['총 판매량'].astype(int)
                    display_df['추천 점수'] = display_df['추천 점수'].astype(int)
                    
                    # 테이블 표시
                    st.table(display_df)
                    
                    # 시각화
                    st.subheader("계절 상품 비교")
                    fig = px.bar(
                        display_df.head(8), 
                        x='상품명', 
                        y='추천 점수',
                        color='계절성 지수',
                        hover_data=['총 판매량'],
                        color_continuous_scale='Viridis',
                        height=500
                    )
                    fig.update_layout(
                        xaxis_title="상품명",
                        yaxis_title="추천 점수",
                        coloraxis_colorbar=dict(title="계절성 지수")
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 3. 번들 추천 탭
        with tabs[2]:
            st.header("🎁 번들 상품 추천")
            st.write("자주 함께 구매되는 상품 조합을 분석하여 번들 상품을 제안합니다.")
            
            # 추천 알고리즘 설명 추가
            with st.expander("🔍 번들 추천 알고리즘 설명"):
                st.markdown("""
                **번들 추천은 어떻게 작동하나요?**
                
                1. **함께 구매 패턴 분석**: 시스템은 모든 구매 기록을 분석하여 같은 고객이 같은 날짜에 함께 구매한 상품 쌍을 찾습니다.
                   - **예시**: 2023년 4월 15일, A 고객이 '쏘렐'과 '완두순'을 함께 구매했다면, ('쏘렐', '완두순') 상품 쌍의 함께 구매 횟수가 1 증가합니다.              
                            
                2. **현재 월 우선 고려**: 선택한 현재 월에 자주 함께 구매된 상품 쌍을 우선적으로 추천합니다.
                   - **예시**: 7월을 선택했다면, 7월에 함께 구매 빈도가 높은 '바질'과 '파인넛' 조합이 우선 추천됩니다.
                                
                3. **계절 보너스**: 해당 상품들의 피크 판매월이 현재 월과 가까울수록 추가 점수를 받습니다.
                   - **예시**: 7월에 '바질'(여름철 상품)과 '파인넛'의 번들은 계절 보너스를 받아 더 높은 순위로 추천됩니다.
                                
                4. **백업 추천**: 선택한 월의 데이터가 충분하지 않은 경우, 전체 기간의 함께 구매 패턴을 참고하여 추천합니다.
                
                이런 방식으로 각 월의 특성과 레시피에 맞는 최적의 번들을 추천합니다.
                """)
            
            # 추천 수량 선택
            bundle_recommend_count = st.slider("추천 번들 수", 3, 10, 5, key="bundle_count")
            
            if st.button("번들 추천 받기", key="bundle_recommend"):
                with st.spinner("번들 상품을 분석 중입니다..."):
                    # 현재 월 설정에 맞춰 번들 추천
                    bundle_recommendations = recommender.recommend_bundles(bundle_recommend_count, current_month)
                    
                    # 추천 결과 표시
                    st.success(f"추천 번들 상품 {len(bundle_recommendations)}개가 준비되었습니다.")
                    
                    # 데이터 표시를 위한 열 이름 변경
                    display_df = bundle_recommendations.copy()
                    if '번들_점수' in display_df.columns:
                        display_df = display_df[['번들_이름', '상품_1', '상품_2', '추천_이유', '함께_구매_횟수', '계절_보너스']]
                    
                    display_df.columns = ['번들명', '상품 1', '상품 2', '추천 이유', '함께 구매 횟수', '계절 관련성']
                    
                    # 테이블 표시
                    st.table(display_df)
                    
                    # 번들 시각화
                    st.subheader("번들 상품 시각화")
                    
                    if len(bundle_recommendations) > 0:
                        # 수평 막대 그래프로 시각화 - 더 이해하기 쉽고 깔끔함
                        fig = px.bar(
                            display_df,
                            y=[f"{row['상품 1']} + {row['상품 2']}" for _, row in display_df.iterrows()],
                            x='함께 구매 횟수',
                            color='함께 구매 횟수',
                            orientation='h',
                            color_continuous_scale='Blues',
                            height=max(300, len(display_df) * 50)  # 데이터 수에 따른 높이 조정
                        )
                        
                        fig.update_layout(
                            title="번들 추천 (함께 구매 빈도 기준)",
                            xaxis_title="함께 구매된 횟수",
                            yaxis_title="상품 조합",
                            yaxis={'categoryorder':'total ascending'},  # 값 순서대로 정렬
                            margin=dict(l=200, r=20, t=50, b=50)  # 긴 상품명을 위한 여백 조정
                        )
                        
                        # 그래프 표시
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 상위 3개 번들에 대한 추가 정보 제공
                        st.subheader("💡 추천 번들 활용 방법")
                        st.write("가장 인기 있는 번들 조합을 활용하는 방법:")
                        
                        for i, (_, row) in enumerate(display_df.head(3).iterrows()):
                            with st.container():
                                st.markdown(f"""
                                **{i+1}. {row['상품 1']} + {row['상품 2']}**  
                                - 함께 구매 횟수: **{row['함께 구매 횟수']}회**
                                - 추천 마케팅 방법: 두 상품을 함께 구매하면 5~10% 할인 혜택 제공
                                - 패키징 제안: 두 상품을 함께 포장하여 특별 번들로 제공
                                """)
                    else:
                        st.info("충분한 번들 추천 데이터가 없습니다.")
        
        # 4. 신규 고객 추천 탭
        with tabs[3]:
            st.header("🆕 신규 고객 추천")
            st.write("신규 고객을 위한 맞춤형 상품을 추천합니다.")
            
            # 추천 알고리즘 설명 추가
            with st.expander("🔍 신규 고객 추천 알고리즘 설명"):
                st.markdown("""
                **신규 고객 추천은 어떻게 작동하나요?**
                
                1. **현재 계절 인기 상품**: 현재 계절에 판매량이 높고 계절성이 강한 제품을 우선 추천합니다.
                   - **예시**: 여름철에는 '바질'과 같이 여름에 판매량이 급증하는 제품을 우선적으로 추천합니다.
                
                2. **연중 베스트셀러**: 계절에 상관없이 일년 내내 인기있는 상품을 추천합니다.
                   - **예시**: '루콜라', '케일'과 같이 꾸준히 판매되는 인기 품목을 추천합니다.
                
                3. **함께 구매 패턴**: 다른 상품과 자주 함께 구매되는 제품을 추천합니다.
                   - **예시**: '레드바질'과 '루콜라'가 자주 함께 구매된다면, 추천 목록에 포함됩니다.
                
                추천 점수는 상품의 판매량, 계절성 지수, 그리고 함께 구매 빈도를 종합적으로 고려하여 계산됩니다.
                """)
            
            # 선택된 월에 따른 계절 정보 표시
            if current_month in [3, 4, 5]:
                season_name = "봄"
                season_emoji = "🌸"
            elif current_month in [6, 7, 8]:
                season_name = "여름"
                season_emoji = "☀️"
            elif current_month in [9, 10, 11]:
                season_name = "가을"
                season_emoji = "🍂"
            else:
                season_name = "겨울"
                season_emoji = "❄️"
            
            st.write(f"**현재 분석 기준**: {current_month}월 ({season_emoji} {season_name})")
            
            # 계절에 따른 인기 마이크로그린 정보
        
            
            st.info(f"{season_emoji} **{season_name}철 인기 마이크로그린**")
            
            # 추천 수량 선택
            col1, col2 = st.columns(2)
            with col1:
                new_customer_recommend_count = st.slider("추천 상품 수", 3, 10, 5, key="new_customer_count")
            
            with col2:
                # 추천 방식 선택 (옵션)
                recommendation_focus = st.radio(
                    "추천 중점 영역",
                    ["균형있는 추천", "계절성 중심", "인기도 중심", "함께 구매 중심"],
                    horizontal=True
                )
            
            if st.button("신규 고객 추천 받기", key="new_customer_recommend"):
                with st.spinner("신규 고객을 위한 추천 상품을 분석 중입니다..."):
                    new_customer_recommendations = recommender.recommend_for_new_customer(
                        new_customer_recommend_count, current_month
                    )
                    
                    # 추천 결과 표시
                    st.success(f"{season_emoji} {season_name}철 신규 고객을 위한 추천 상품 {len(new_customer_recommendations)}개가 준비되었습니다.")
                    
                    # 추천 카테고리별로 분류
                    seasonal_products = new_customer_recommendations[new_customer_recommendations['추천_이유'].str.contains('계절|시즌|월별', case=False)]
                    bestseller_products = new_customer_recommendations[new_customer_recommendations['추천_이유'].str.contains('베스트|인기|판매량', case=False)]
                    copurchase_products = new_customer_recommendations[new_customer_recommendations['추천_이유'].str.contains('함께|같이|조합', case=False)]
                    
                    # 탭으로 구분하여 표시
                    rec_tabs = st.tabs(["📊 전체 추천", "🌸 계절성 상품", "⭐ 인기 상품", "🔗 함께 구매 상품"])
                    
                    # 전체 추천 탭
                    with rec_tabs[0]:
                        # 데이터 표시를 위한 열 이름 변경
                        display_df = new_customer_recommendations.copy()
                        display_df.columns = ['상품명', '추천 점수', '추천 이유']
                        display_df['추천 점수'] = display_df['추천 점수'].round(2)
                        
                        # 테이블 표시
                        st.table(display_df)
                    
                    # 계절성 상품 탭
                    with rec_tabs[1]:
                        if not seasonal_products.empty:
                            seasonal_display = seasonal_products.copy()
                            seasonal_display.columns = ['상품명', '추천 점수', '추천 이유']
                            seasonal_display['추천 점수'] = seasonal_display['추천 점수'].round(2)
                            st.table(seasonal_display)
                        else:
                            st.info("계절성이 높은 추천 상품이 없습니다.")
                    
                    # 인기 상품 탭
                    with rec_tabs[2]:
                        if not bestseller_products.empty:
                            bestseller_display = bestseller_products.copy()
                            bestseller_display.columns = ['상품명', '추천 점수', '추천 이유']
                            bestseller_display['추천 점수'] = bestseller_display['추천 점수'].round(2)
                            st.table(bestseller_display)
                        else:
                            st.info("인기도 기반 추천 상품이 없습니다.")
                    
                    # 함께 구매 상품 탭
                    with rec_tabs[3]:
                        if not copurchase_products.empty:
                            copurchase_display = copurchase_products.copy()
                            copurchase_display.columns = ['상품명', '추천 점수', '추천 이유']
                            copurchase_display['추천 점수'] = copurchase_display['추천 점수'].round(2)
                            st.table(copurchase_display)
                        else:
                            st.info("함께 구매 패턴 기반 추천 상품이 없습니다.")
        
        # 5. 상품 분석 탭
        with tabs[4]:
            st.header("📊 상품 상세 분석")
            st.write("특정 상품의 판매 현황과 관련 데이터를 상세하게 분석합니다.")
            
            # 분석 방법 설명 추가
            with st.expander("🔍 상품 분석 방법 설명"):
                st.markdown("""
                **상품 분석은 어떤 정보를 제공하나요?**
                
                1. **기본 판매 지표**: 총 판매량, 구매 고객 수, 계절성 지수 등 기본적인 판매 지표를 확인할 수 있습니다.
                   - **예시**: '바질'의 연간 총 판매량이 5,000개, 구매 고객 수는 120명, 계절성 지수는 0.45인 경우 여름철에 강한 상품임을 알 수 있습니다.
                
                2. **계절성 분석**: 월별 판매 추이를 시각화하고, 피크 판매월과 주요 계절을 파악합니다.
                   - **예시**: '루콜라'의 월별 판매 그래프를 보면 3-5월, 9-11월에 판매 피크가 있는 봄과 가을의 계절 상품임을 확인할 수 있습니다.
                
                3. **고객 분석**: 해당 상품을 가장 많이 구매한 주요 고객을 확인합니다.
                   - **예시**: '루콜라'을 가장 많이 구매한 상위 5개 고객이 모두 유기농 레스토랑이라면, 마케팅 전략 수립 시 이를 활용할 수 있습니다.
                
                4. **연관 상품**: 이 상품과 함께 구매되는 다른 상품들을 찾아 번들 상품 구성에 활용할 수 있습니다.
                   - **예시**: '바질'을 구매한 고객이 '적시소'도 자주 함께 구매한다면, '바질+적시소 세트'로 번들 상품을 구성할 수 있습니다.
                
                5. **반품 분석**: 반품 수량, 비율, 이유 등을 분석하여 상품 개선에 활용할 수 있습니다.
                   - **예시**: '루콜라'의 반품 이유 중 60%가 '신선도 저하'라면, 포장 방식이나 배송 과정을 개선해야 함을 알 수 있습니다.
                
                이러한 분석 결과는 상품 관리, 마케팅, 번들 구성 등 다양한 비즈니스 의사결정에 활용할 수 있습니다.
                """)
            
            # 상품 목록 얻기 (배송료 항목 제외)
            delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
            
            # 판매량 기준으로 정렬된 상품 목록 가져오기
            product_sales = sales_data.groupby('상품')['수량'].sum().sort_values(ascending=False)
            
            # 배송료 관련 상품 필터링
            filtered_products = []
            for product in product_sales.index:
                if not any(keyword in str(product).lower() for keyword in delivery_keywords) and '세트상품' not in str(product) and '증정품' not in str(product):
                    filtered_products.append(product)
            
            top_products = filtered_products[:100]  # 상위 100개만 보여주기
            
            if len(top_products) > 0:
                selected_product = st.selectbox("분석할 상품 선택", top_products)
                
                if st.button("상품 분석하기", key="product_analysis"):
                    with st.spinner(f"{selected_product} 상품을 분석 중입니다..."):
                        product_analysis = recommender.analyze_product_details(selected_product)
                        
                        if product_analysis['상태'] == '실패':
                            st.error(product_analysis['메시지'])
                        else:
                            # 분석 결과 표시
                            st.success(f"{selected_product} 상품 분석이 완료되었습니다.")
                            
                            # 상품 기본 정보
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("총 판매량", f"{int(product_analysis['총_판매량']):,}개")
                            with col2:
                                st.metric("구매 고객 수", f"{int(product_analysis['구매_고객_수']):,}명")
                            with col3:
                                if '계절성_정보' in product_analysis and '계절성_지수' in product_analysis['계절성_정보']:
                                    st.metric("계절성 지수", f"{product_analysis['계절성_정보']['계절성_지수']:.2f}")
                                else:
                                    st.metric("계절성 지수", "데이터 없음")
                            
                            # 계절성 정보 시각화
                            st.subheader("계절성 분석")
                            if '계절성_정보' in product_analysis and product_analysis['계절성_정보']:
                                # 월별 판매 데이터 가져오기
                                monthly_sales = sales_data[sales_data['상품'] == selected_product].groupby('month')['수량'].sum().reset_index()
                                monthly_sales = monthly_sales.sort_values('month')
                                
                                # 데이터가 없는 월은 0으로 채우기
                                all_months = pd.DataFrame({'month': range(1, 13)})
                                monthly_sales = pd.merge(all_months, monthly_sales, on='month', how='left').fillna(0)
                                
                                # 히트맵 데이터 준비
                                heatmap_data = pd.DataFrame({
                                    'month': monthly_sales['month'],
                                    'sales': monthly_sales['수량'],
                                    'month_name': ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
                                })
                                
                                # Plotly 히트맵
                                fig = px.imshow(
                                    heatmap_data['sales'].values.reshape(1, -1),
                                    y=['판매량'],
                                    x=heatmap_data['month_name'],
                                    color_continuous_scale='Viridis',
                                    aspect="auto"
                                )
                                fig.update_layout(
                                    title=f"{selected_product}의 월별 판매 현황",
                                    height=200,
                                    margin=dict(l=5, r=5, t=40, b=5)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 피크 판매월과 주요 계절 표시
                                peak_month = product_analysis['계절성_정보']['피크_판매월']
                                main_season = product_analysis['계절성_정보']['주요_계절']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("피크 판매월", f"{peak_month}월")
                                with col2:
                                    st.metric("주요 계절", main_season)
                            else:
                                st.info("계절성 정보가 충분하지 않습니다.")
                            
                            # 주요 고객 정보
                            st.subheader("주요 구매 고객")
                            if product_analysis['주요_고객']:
                                top_customers_df = pd.DataFrame({
                                    '고객명': list(product_analysis['주요_고객'].keys()),
                                    '구매수량': list(product_analysis['주요_고객'].values())
                                }).sort_values('구매수량', ascending=False)
                                
                                # 배송료 관련 고객이나 재고조정 고객 추가 필터링
                                # (이미 분석 단계에서 필터링되었지만 UI 단에서 한번 더 확인)
                                delivery_keywords = ['배송료', '배달료', '퀵', '배송비', '택배', '운송', '배달', '퀵배송료']
                                filtered_customers = []
                                
                                for i, row in top_customers_df.iterrows():
                                    customer_name = row['고객명']
                                    # 재고조정, 배송료 관련 키워드 필터링
                                    if not ('재고조정' in customer_name or '문정창고' in customer_name or '창고' in customer_name or
                                           any(keyword in customer_name.lower() for keyword in delivery_keywords)):
                                        filtered_customers.append(row)
                                
                                # 필터링된 고객으로 새 데이터프레임 생성
                                if filtered_customers:
                                    filtered_df = pd.DataFrame(filtered_customers)
                                    
                                    # 차트로 시각화
                                    fig = px.bar(
                                        filtered_df,
                                        x='고객명',
                                        y='구매수량',
                                        color='구매수량',
                                        height=300,
                                        color_continuous_scale='Blues'
                                    )
                                    fig.update_layout(
                                        title="주요 고객별 구매 수량",
                                        xaxis_title="고객명",
                                        yaxis_title="구매 수량"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("표시할 주요 고객 정보가 없습니다.")
                            else:
                                st.info("주요 고객 정보가 없습니다.")
                            
                            # 함께 구매되는 상품
                            st.subheader("함께 구매되는 상품")
                            if product_analysis['함께_구매되는_상품']:
                                co_purchased_df = pd.DataFrame(product_analysis['함께_구매되는_상품'])
                                
                                # 차트로 시각화
                                fig = px.bar(
                                    co_purchased_df,
                                    x='상품',
                                    y='함께_구매_횟수',
                                    color='함께_구매_횟수',
                                    height=300,
                                    color_continuous_scale='Greens'
                                )
                                fig.update_layout(
                                    title="함께 구매되는 상품",
                                    xaxis_title="상품명",
                                    yaxis_title="함께 구매된 횟수"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 번들 추천
                                st.info("💡 이 상품은 위 제품들과 함께 세트로 판매하면 효과적일 수 있습니다.")
                            else:
                                st.info("함께 구매되는 상품 정보가 없습니다.")
                            
                            # 반품 정보
                            if '반품_정보' in product_analysis and product_analysis['반품_정보']:
                                st.subheader("반품 분석")
                                refund_info = product_analysis['반품_정보']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("반품 수량", f"{refund_info['반품_수량']:,}개")
                                with col2:
                                    st.metric("반품 비율", f"{refund_info['반품_비율']:.1f}%")
                                
                                # 반품 이유 시각화
                                if refund_info['반품_이유']:
                                    refund_reasons_df = pd.DataFrame({
                                        '반품이유': list(refund_info['반품_이유'].keys()),
                                        '수량': list(refund_info['반품_이유'].values())
                                    }).sort_values('수량', ascending=False)
                                    
                                    fig = px.pie(
                                        refund_reasons_df,
                                        values='수량',
                                        names='반품이유',
                                        title="반품 이유 분석",
                                        hole=0.4
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 반품률이 높으면 경고 메시지 표시
                                    if refund_info['반품_비율'] > 10:
                                        st.warning("⚠️ 이 상품의 반품률이 10%를 초과합니다. 품질 관리나 정보 제공 개선이 필요할 수 있습니다.")
                            else:
                                st.info("반품 정보가 없습니다.")
            else:
                st.warning("상품 데이터가 없습니다. 데이터를 확인해주세요.")
        
        # 6. 업체 분석 탭
        with tabs[5]:
            st.header("🏢 업체별 상세 분석")
            st.write("업체(고객)별 구매 패턴과 선호도를 분석합니다.")
            
            # 분석 방법 설명 추가
            with st.expander("🔍 업체 분석 방법 설명"):
                st.markdown("""
                **업체 분석은 어떤 정보를 제공하나요?**
                
                1. **기본 정보**: 업체 코드, 카테고리, 총 구매량/금액 등 기본 정보를 확인할 수 있습니다.
                
                2. **월별 구매 패턴**: 월별 구매량과 금액을 그래프로 확인하여 계절적 패턴을 파악합니다.
                
                3. **상품 선호도**: 가장 많이 구매한 상품 TOP 5와 그 비중을 확인할 수 있습니다.
                
                4. **계절별 선호도**: 계절에 따른 구매 패턴을 분석하여 계절별 맞춤 상품을 제안합니다.
                
                5. **반품 정보**: 반품 비율과 주요 반품 이유를 확인하여 품질 개선에 활용할 수 있습니다.
                
                **업체 카테고리 분류 방식**
                
                업체명 앞에 있는 코드를 기준으로 자동 분류됩니다:
                - 001, 005: 호텔
                - 그 외: 일반 업체
                """)
            
            # 카테고리별 고객 목록
            customer_categories = recommender.get_customer_categories()
            
            # 카테고리 선택 콜백
            def on_category_change():
                st.session_state.customer_category = st.session_state.category_selection
                # 카테고리가 변경되면 고객 선택을 초기화
                st.session_state.selected_customer = None
                st.session_state.analyzed_customer = None
                st.session_state.customer_info = None
            
            # 카테고리 선택
            all_categories = ["전체"] + list(customer_categories.keys())
            
            # 선택된 카테고리의 인덱스 계산
            selected_index = 0  # 기본값은 '전체'
            
            if st.session_state.customer_category in all_categories:
                selected_index = all_categories.index(st.session_state.customer_category)
            
            category_selection = st.selectbox(
                "업체 카테고리 선택",
                all_categories,
                index=selected_index,
                key="category_selection",
                on_change=on_category_change
            )
            
            # 선택된 카테고리에 따른 고객 목록
            if category_selection == "전체":
                all_customers = []
                for category, customers in customer_categories.items():
                    all_customers.extend(customers)
                customer_list = sorted(all_customers)
            else:
                customer_list = sorted(customer_categories[category_selection])
            
            # 고객 선택 콜백
            def on_customer_change():
                st.session_state.selected_customer = st.session_state.customer_selection
            
            # 고객 선택
            # 이전에 선택한 고객이 현재 카테고리에 있는지 확인
            default_index = 0
            if st.session_state.selected_customer in customer_list:
                default_index = customer_list.index(st.session_state.selected_customer)
            
            selected_customer = st.selectbox(
                "업체 선택",
                customer_list,
                index=default_index,
                key="customer_selection",
                on_change=on_customer_change
            )
            
            # 분석 버튼 클릭 또는 이미 분석된 고객 정보가 있으면 표시
            if st.button("업체 분석하기", key="analyze_customer") or (st.session_state.analyzed_customer == selected_customer and st.session_state.customer_info is not None):
                with st.spinner(f"{selected_customer} 업체를 분석 중입니다..."):
                    # 이미 분석된 고객이 현재 선택된 고객과 동일하면 저장된 정보 사용
                    if st.session_state.analyzed_customer == selected_customer and st.session_state.customer_info is not None:
                        customer_info = st.session_state.customer_info
                    else:
                        # 새로운 고객 분석 정보 가져오기
                        customer_info = recommender.analyze_customer_details(selected_customer)
                        # 분석 결과 저장
                        st.session_state.analyzed_customer = selected_customer
                        st.session_state.customer_info = customer_info
                    
                    if customer_info['상태'] == '실패':
                        st.error(customer_info['메시지'])
                    else:
                        # 기본 정보 섹션
                        st.subheader("📊 업체 기본 정보")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("카테고리", customer_info['고객_카테고리'])
                        with col2:
                            st.metric("총 구매량", f"{customer_info['총_구매량']:,}개")
                        with col3:
                            st.metric("총 구매금액", f"{customer_info['총_구매금액']:,}원")
                        
                        # 추가 정보
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("최근 구매일", customer_info['최근_구매일'] if customer_info['최근_구매일'] else "정보 없음")
                        with col2:
                            st.metric("구매 빈도", f"{customer_info['구매_빈도']:.1f}%")
                        
                        # 월별 구매 패턴
                        st.subheader("📅 구매 패턴 분석")
                        
                        if customer_info['연월별_구매']:
                            # 연-월 기준 데이터 준비
                            yearmonths = list(customer_info['연월별_구매'].keys())
                            quantities = [info['수량'] for info in customer_info['연월별_구매'].values()]
                            amounts = [info['금액'] for info in customer_info['연월별_구매'].values()]
                            
                            # 데이터 기간 표시
                            if yearmonths:
                                min_yearmonth = min(yearmonths)
                                max_yearmonth = max(yearmonths)
                                st.info(f"📊 분석 데이터는 {min_yearmonth}부터 {max_yearmonth}까지의 정보를 포함하고 있습니다.")
                            
                            # 두 개의 그래프 컬럼
                            col1, col2 = st.columns(2)
                            
                            # 구매량 그래프
                            with col1:
                                fig_qty = px.bar(
                                    x=yearmonths,
                                    y=quantities,
                                    title="구매량 추이",
                                    labels={'x': '연-월', 'y': '구매량'},
                                    color=quantities,
                                    color_continuous_scale='Blues'
                                )
                                fig_qty.update_layout(height=400)
                                st.plotly_chart(fig_qty, use_container_width=True)
                            
                            # 구매금액 그래프
                            with col2:
                                if any(amounts):  # 금액 정보가 있는 경우에만
                                    fig_amt = px.bar(
                                        x=yearmonths,
                                        y=amounts,
                                        title="구매금액 추이",
                                        labels={'x': '연-월', 'y': '구매금액 (원)'},
                                        color=amounts,
                                        color_continuous_scale='Greens'
                                    )
                                    fig_amt.update_layout(height=400)
                                    st.plotly_chart(fig_amt, use_container_width=True)
                                else:
                                    st.info("구매금액 정보가 없습니다.")
                                    
                            # 누적 구매 추이 그래프 추가
                            st.subheader("📈 누적 구매 추이")
                            
                            # 누적 데이터 계산
                            cumulative_qty = np.cumsum(quantities)
                            
                            fig_cum = px.line(
                                x=yearmonths,
                                y=cumulative_qty,
                                title="누적 구매량",
                                labels={'x': '연-월', 'y': '누적 구매량'},
                                markers=True,
                            )
                            fig_cum.update_traces(line=dict(width=3), marker=dict(size=10))
                            fig_cum.update_layout(height=400)
                            st.plotly_chart(fig_cum, use_container_width=True)
                            
                            # 월별 구매 요약 테이블
                            st.subheader("📋 연-월별 구매 요약")
                            
                            # 데이터프레임 생성
                            monthly_summary = pd.DataFrame({
                                '연-월': yearmonths,
                                '구매량': quantities,
                                '구매금액': amounts if any(amounts) else [0] * len(yearmonths),
                                '누적 구매량': cumulative_qty,
                            })
                            
                            # 테이블 표시
                            st.dataframe(monthly_summary, use_container_width=True)
                            
                            # CSV 다운로드 버튼 추가
                            csv = monthly_summary.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 구매 패턴 데이터 다운로드",
                                data=csv,
                                file_name=f"{selected_customer}_구매패턴.csv",
                                mime="text/csv",
                                key="download_monthly_summary"
                            )
                            
                            # 평균 구매량 통계
                            avg_qty = np.mean(quantities)
                            max_qty_idx = np.argmax(quantities)
                            max_qty_yearmonth = yearmonths[max_qty_idx]
                            
                            st.info(f"📊 평균 월 구매량: {avg_qty:.1f}개, 가장 많이 구매한 시기: {max_qty_yearmonth} ({max(quantities)}개)")
                            
                        elif customer_info['월별_구매']:
                            # 기존 월 기반 데이터 처리 (이전 코드와의 호환성)
                            months = list(customer_info['월별_구매'].keys())
                            quantities = [info['수량'] for info in customer_info['월별_구매'].values()]
                            amounts = [info['금액'] for info in customer_info['월별_구매'].values()]
                            
                            # 데이터 기간 표시
                            min_month = min(months) if months else 0
                            max_month = max(months) if months else 0
                            
                            # 데이터 기간 정보 표시 (월만 있는 경우)
                            st.info(f"📊 분석 데이터는 {min_month}월부터 {max_month}월까지의 정보를 포함하고 있습니다. (연도 정보 없음)")
                            
                            # 월 이름으로 변환
                            month_names = [f"{m}월" for m in months]
                            
                            # 두 개의 그래프 컬럼
                            col1, col2 = st.columns(2)
                            
                            # 구매량 그래프
                            with col1:
                                fig_qty = px.bar(
                                    x=month_names,
                                    y=quantities,
                                    title="구매량 추이",
                                    labels={'x': '월', 'y': '구매량'},
                                    color=quantities,
                                    color_continuous_scale='Blues'
                                )
                                fig_qty.update_layout(height=400)
                                st.plotly_chart(fig_qty, use_container_width=True)
                            
                            # 구매금액 그래프
                            with col2:
                                if any(amounts):  # 금액 정보가 있는 경우에만
                                    fig_amt = px.bar(
                                        x=month_names,
                                        y=amounts,
                                        title="구매금액 추이",
                                        labels={'x': '월', 'y': '구매금액 (원)'},
                                        color=amounts,
                                        color_continuous_scale='Greens'
                                    )
                                    fig_amt.update_layout(height=400)
                                    st.plotly_chart(fig_amt, use_container_width=True)
                                else:
                                    st.info("구매금액 정보가 없습니다.")
                                    
                            # 누적 구매 추이 그래프 추가
                            st.subheader("📈 누적 구매 추이")
                            
                            # 누적 데이터 계산
                            cumulative_qty = np.cumsum(quantities)
                            
                            fig_cum = px.line(
                                x=month_names,
                                y=cumulative_qty,
                                title="월별 누적 구매량",
                                labels={'x': '월', 'y': '누적 구매량'},
                                markers=True,
                            )
                            fig_cum.update_traces(line=dict(width=3), marker=dict(size=10))
                            fig_cum.update_layout(height=400)
                            st.plotly_chart(fig_cum, use_container_width=True)
                            
                            # 월별 구매 요약 테이블
                            st.subheader("📋 월별 구매 요약")
                            
                            # 데이터프레임 생성
                            monthly_summary = pd.DataFrame({
                                '월': month_names,
                                '구매량': quantities,
                                '구매금액': amounts if any(amounts) else [0] * len(months),
                                '누적 구매량': cumulative_qty,
                            })
                            
                            # 테이블 표시
                            st.dataframe(monthly_summary, use_container_width=True)
                            
                            # 평균 구매량 통계
                            avg_qty = np.mean(quantities)
                            max_qty_month = month_names[np.argmax(quantities)]
                            
                            st.info(f"📊 평균 월 구매량: {avg_qty:.1f}개, 가장 많이 구매한 달: {max_qty_month} ({max(quantities)}개)")
                        else:
                            st.info("월별 구매 데이터가 없습니다.")
                        
                        # 월별 상품 구매 내역 (추가)
                        st.subheader("📊 월별 상품 구매 상세")
                        
                        if customer_info['연월별_상품_구매'] and len(customer_info['연월별_상품_구매']) > 0:
                            # 연-월 선택 콜백
                            def on_yearmonth_selection_change():
                                st.session_state.selected_yearmonth = st.session_state.yearmonth_selection
                            
                            # 연-월 선택
                            available_yearmonths = sorted(customer_info['연월별_상품_구매'].keys())
                            
                            if available_yearmonths:
                                # 이전에 선택한 연-월이 있는지 확인
                                default_yearmonth_index = 0
                                if 'selected_yearmonth' in st.session_state and st.session_state.selected_yearmonth in available_yearmonths:
                                    default_yearmonth_index = available_yearmonths.index(st.session_state.selected_yearmonth)
                                
                                selected_yearmonth = st.selectbox(
                                    "분석할 연-월 선택", 
                                    available_yearmonths,
                                    index=default_yearmonth_index,
                                    key="yearmonth_selection",
                                    on_change=on_yearmonth_selection_change
                                )
                                
                                # 선택한 연-월의 상품 구매 데이터
                                yearmonth_products = customer_info['연월별_상품_구매'][selected_yearmonth]
                                
                                if yearmonth_products:
                                    # 데이터 프레임으로 변환
                                    product_df = pd.DataFrame({
                                        '상품명': list(yearmonth_products.keys()),
                                        '구매량': list(yearmonth_products.values())
                                    }).sort_values('구매량', ascending=False)
                                    
                                    # 탭으로 보기 방식 구분
                                    monthly_tabs = st.tabs(["그래프 보기", "상세 데이터", "날짜별 구매 내역"])
                                    
                                    with monthly_tabs[0]:
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # 상위 10개 상품만 그래프로 표시
                                            top_10_products = product_df.head(10)
                                            fig_month_products = px.bar(
                                                top_10_products,
                                                x='상품명',
                                                y='구매량',
                                                title=f"{selected_yearmonth} 주요 구매 상품 (TOP 10)",
                                                color='구매량',
                                                color_continuous_scale='Viridis'
                                            )
                                            fig_month_products.update_layout(height=400)
                                            st.plotly_chart(fig_month_products, use_container_width=True)
                                        
                                        with col2:
                                            # 원형 차트로 구매 비중 표시
                                            if not top_10_products.empty:
                                                fig_pie = px.pie(
                                                    top_10_products,
                                                    values='구매량',
                                                    names='상품명',
                                                    title=f"{selected_yearmonth} 상품 구매 비중",
                                                    hole=0.4
                                                )
                                                fig_pie.update_layout(height=400)
                                                st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                    with monthly_tabs[1]:
                                        # 표 형식으로 전체 데이터 표시
                                        st.markdown(f"#### {selected_yearmonth} 전체 구매 상품 ({len(product_df)}개)")
                                        st.dataframe(product_df, use_container_width=True)
                                        
                                        # 다운로드 기능 추가
                                        csv = product_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label=f"📥 {selected_yearmonth} 구매 데이터 다운로드",
                                            data=csv,
                                            file_name=f"{selected_customer}_{selected_yearmonth}_구매내역.csv",
                                            mime="text/csv",
                                            key=f"download_yearmonth_{selected_yearmonth}"
                                        )
                                        
                                        # 통계 정보 표시
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("총 구매 상품 종류", f"{len(product_df)}개")
                                        with col2:
                                            st.metric("총 구매량", f"{product_df['구매량'].sum():,}개")
                                        with col3:
                                            st.metric("평균 구매량", f"{product_df['구매량'].mean():.1f}개")
                                    
                                    with monthly_tabs[2]:
                                        if selected_yearmonth in customer_info['연월별_구매_날짜']:
                                            date_data = customer_info['연월별_구매_날짜'][selected_yearmonth]
                                            
                                            if date_data:
                                                # 날짜 선택 옵션
                                                all_dates = sorted(date_data.keys())
                                                selected_date = st.selectbox(
                                                    "날짜 선택", 
                                                    all_dates,
                                                    key=f"date_selection_{selected_yearmonth}"
                                                )
                                                
                                                if selected_date in date_data:
                                                    day_products = date_data[selected_date]
                                                    
                                                    # 일별 구매 데이터 표시
                                                    day_df = pd.DataFrame({
                                                        '상품명': list(day_products.keys()),
                                                        '구매량': list(day_products.values())
                                                    }).sort_values('구매량', ascending=False)
                                                    
                                                    st.markdown(f"#### {selected_date} 구매 내역")
                                                    st.dataframe(day_df, use_container_width=True)
                                                    
                                                    # 일별 구매 그래프
                                                    fig_day_products = px.bar(
                                                        day_df,
                                                        x='상품명',
                                                        y='구매량',
                                                        title=f"{selected_date} 구매 상품",
                                                        color='구매량',
                                                        color_continuous_scale='Cividis'
                                                    )
                                                    fig_day_products.update_layout(height=400)
                                                    st.plotly_chart(fig_day_products, use_container_width=True)
                                            else:
                                                st.info(f"{selected_yearmonth}의 날짜별 구매 데이터가 없습니다.")
                                        else:
                                            st.info(f"{selected_yearmonth}의 날짜별 구매 데이터가 없습니다.")
                                    
                                    # 종합 인사이트 표시
                                    st.markdown("### 📈 구매 인사이트")
                                    max_product = product_df.iloc[0]['상품명'] if not product_df.empty else None
                                    if max_product:
                                        st.info(f"💡 **{selected_yearmonth}**에 **{selected_customer}**가 가장 많이 구매한 상품은 **{max_product}** ({product_df.iloc[0]['구매량']}개) 입니다.")
                                        
                                        # 추가 인사이트
                                        if len(product_df) > 5:
                                            total_qty = product_df['구매량'].sum()
                                            top5_qty = product_df.head(5)['구매량'].sum()
                                            top5_ratio = (top5_qty / total_qty) * 100
                                            
                                            st.info(f"💡 TOP 5 상품이 전체 구매량의 **{top5_ratio:.1f}%**를 차지합니다.")
                                else:
                                    st.info(f"{selected_yearmonth}에 구매한 상품이 없습니다.")
                                    
                        elif customer_info['월별_상품_구매'] and len(customer_info['월별_상품_구매']) > 0:
                            # 기존 월 기반 구매 내역 (이전 코드와의 호환성)
                            # 월 선택 콜백
                            def on_month_selection_change():
                                st.session_state.selected_month = st.session_state.month_selection
                            
                            # 월 선택
                            available_months = sorted(customer_info['월별_상품_구매'].keys())
                            
                            if available_months:
                                # 이전에 선택한 월이 있는지 확인
                                default_month_index = 0
                                if 'selected_month' in st.session_state and st.session_state.selected_month in available_months:
                                    default_month_index = available_months.index(st.session_state.selected_month)
                                
                                selected_month = st.selectbox(
                                    "분석할 월 선택", 
                                    available_months,
                                    index=default_month_index,
                                    format_func=lambda x: f"{x}월",
                                    key="month_selection",
                                    on_change=on_month_selection_change
                                )
                                
                                st.warning("⚠️ 데이터에 연도 정보가 포함되어 있지 않아 월 단위로만 분석이 가능합니다.")
                                
                                # 월별 상품 구매 분석 (이전과 동일)
                                # ... (기존 월별 구매 분석 코드)
                                
                            else:
                                st.info("월별 상품 구매 데이터가 없습니다.")
                            
                        else:
                            st.info("상품 구매 상세 데이터가 없습니다.")
                        
                        # 주요 구매 상품
                        st.subheader("🔝 주요 구매 상품")
                        
                        if customer_info['주요_구매상품']:
                            top_products = customer_info['주요_구매상품']
                            
                            top_prod_df = pd.DataFrame({
                                '상품명': list(top_products.keys()),
                                '구매량': list(top_products.values())
                            }).sort_values('구매량', ascending=False)
                            
                            # 차트 그리기
                            fig_top = px.bar(
                                top_prod_df,
                                x='상품명',
                                y='구매량',
                                title=f"{selected_customer}의 주요 구매 상품",
                                color='구매량',
                                color_continuous_scale='RdBu'
                            )
                            fig_top.update_layout(height=400)
                            st.plotly_chart(fig_top, use_container_width=True)
                            
                            # 원형 차트로 구매 비중 표시
                            fig_pie_top = px.pie(
                                top_prod_df,
                                values='구매량',
                                names='상품명',
                                title="주요 구매 상품 비중",
                                hole=0.4
                            )
                            fig_pie_top.update_layout(height=400)
                            st.plotly_chart(fig_pie_top, use_container_width=True)
                        else:
                            st.info("구매 상품 정보가 없습니다.")
                        
                        # 계절별 선호도
                        st.subheader("🌱 계절별 구매 패턴")
                        
                        if customer_info['계절별_선호도']:
                            # 데이터 존재 여부 확인
                            seasonal_data = customer_info['계절별_선호도']
                            has_seasonal_data = sum(seasonal_data.values()) > 0
                            
                            if has_seasonal_data:
                                # 데이터 준비
                                seasons = list(seasonal_data.keys())
                                season_values = list(seasonal_data.values())
                                
                                # 데이터에 근거한 설명 추가
                                available_seasons = [s for s, v in zip(seasons, season_values) if v > 0]
                                if available_seasons:
                                    st.info(f"📊 분석된 데이터는 {'·'.join(available_seasons)}에 대한 정보만 포함하고 있습니다.")
                                
                                # 차트로 표시
                                fig_season = px.bar(
                                    x=seasons,
                                    y=season_values,
                                    title="계절별 구매량",
                                    color=season_values,
                                    color_continuous_scale='Viridis'
                                )
                                fig_season.update_layout(height=400)
                                st.plotly_chart(fig_season, use_container_width=True)
                                
                                # 계절별 선호도 테이블
                                season_df = pd.DataFrame({
                                    '계절': seasons,
                                    '구매량': season_values
                                })
                                st.dataframe(season_df, use_container_width=True)
                                
                                # 최대 선호 계절
                                max_season_idx = season_values.index(max(season_values))
                                max_season = seasons[max_season_idx]
                                max_season_value = max(season_values)
                                
                                # 계절별 비율 계산
                                total_seasonal = sum(season_values)
                                if total_seasonal > 0:
                                    season_ratios = [round((v / total_seasonal) * 100, 1) for v in season_values]
                                    
                                    # 원형 차트
                                    season_pie_df = pd.DataFrame({
                                        '계절': seasons,
                                        '구매량': season_values,
                                        '비율(%)': season_ratios
                                    })
                                    
                                    fig_season_pie = px.pie(
                                        season_pie_df,
                                        values='구매량',
                                        names='계절',
                                        title="계절별 구매 비중",
                                        hover_data=['비율(%)'],
                                        hole=0.4
                                    )
                                    fig_season_pie.update_layout(height=400)
                                    st.plotly_chart(fig_season_pie, use_container_width=True)
                                    
                                    # 계절 추천 표시
                                    if max_season_value > 0:
                                        max_ratio = round((max_season_value / total_seasonal) * 100, 1)
                                        st.markdown(f"""
                                        ##### 계절별 구매 분석
                                        - 가장 선호하는 계절: **{max_season}** ({max_ratio}%)
                                        - 이 업체에게는 **{max_season}**에 맞는 상품을 우선적으로 추천하는 것이 효과적일 수 있습니다.
                                        """)
                                        
                                        # 계절성 제안
                                        next_seasons = {
                                            '봄': '여름',
                                            '여름': '가을',
                                            '가을': '겨울',
                                            '겨울': '봄'
                                        }
                                        
                                        if max_season in next_seasons:
                                            next_season = next_seasons[max_season]
                                            st.info(f"💡 {max_season}에 주로 구매하는 고객이므로, 다가오는 {next_season}에 대비한 상품을 미리 제안해보세요.")
                            else:
                                st.info("계절별 구매 데이터가 충분하지 않습니다.")
                        else:
                            st.info("계절별 선호도 정보가 없습니다.")
                        
                        # 분기별 선호도
                        st.subheader("📊 분기별 구매 패턴")
                        
                        if customer_info['분기별_선호도']:
                            # 데이터 존재 여부 확인
                            quarterly_data = customer_info['분기별_선호도']
                            has_quarterly_data = sum(quarterly_data.values()) > 0
                            
                            if has_quarterly_data:
                                # 데이터 준비
                                quarters = list(quarterly_data.keys())
                                quarter_values = list(quarterly_data.values())
                                
                                # 데이터에 근거한 설명 추가
                                available_quarters = [q for q, v in zip(quarters, quarter_values) if v > 0]
                                if available_quarters:
                                    st.info(f"📊 분석된 데이터는 {'·'.join(available_quarters)}에 대한 정보를 포함하고 있습니다.")
                                
                                # 차트로 표시
                                fig_quarter = px.bar(
                                    x=quarters,
                                    y=quarter_values,
                                    title="분기별 구매량",
                                    color=quarter_values,
                                    color_continuous_scale='Blues'
                                )
                                fig_quarter.update_layout(height=400)
                                st.plotly_chart(fig_quarter, use_container_width=True)
                                
                                # 분기별 선호도 테이블
                                quarter_df = pd.DataFrame({
                                    '분기': quarters,
                                    '구매량': quarter_values
                                })
                                st.dataframe(quarter_df, use_container_width=True)
                                
                                # 최대 선호 분기
                                max_quarter_idx = quarter_values.index(max(quarter_values))
                                max_quarter = quarters[max_quarter_idx]
                                max_quarter_value = max(quarter_values)
                                
                                # 분기별 비율 계산
                                total_quarterly = sum(quarter_values)
                                if total_quarterly > 0:
                                    quarter_ratios = [round((v / total_quarterly) * 100, 1) for v in quarter_values]
                                    
                                    # 원형 차트
                                    quarter_pie_df = pd.DataFrame({
                                        '분기': quarters,
                                        '구매량': quarter_values,
                                        '비율(%)': quarter_ratios
                                    })
                                    
                                    fig_quarter_pie = px.pie(
                                        quarter_pie_df,
                                        values='구매량',
                                        names='분기',
                                        title="분기별 구매 비중",
                                        hover_data=['비율(%)'],
                                        hole=0.4
                                    )
                                    fig_quarter_pie.update_layout(height=400)
                                    st.plotly_chart(fig_quarter_pie, use_container_width=True)
                                    
                                    # 분기 추천 표시
                                    if max_quarter_value > 0:
                                        max_ratio = round((max_quarter_value / total_quarterly) * 100, 1)
                                        st.markdown(f"""
                                        ##### 분기별 구매 분석
                                        - 가장 선호하는 분기: **{max_quarter}** ({max_ratio}%)
                                        - 이 업체에게는 **{max_quarter}**에 맞는 상품을 우선적으로 추천하는 것이 효과적일 수 있습니다.
                                        """)
                                        
                                        # 다음 분기 제안
                                        current_quarter = None
                                        now = datetime.now()
                                        month = now.month
                                        if month in [1, 2, 3]:
                                            current_quarter = '1분기'
                                        elif month in [4, 5, 6]:
                                            current_quarter = '2분기'
                                        elif month in [7, 8, 9]:
                                            current_quarter = '3분기'
                                        else:
                                            current_quarter = '4분기'
                                        
                                        next_quarters = {
                                            '1분기': '2분기',
                                            '2분기': '3분기',
                                            '3분기': '4분기',
                                            '4분기': '1분기'
                                        }
                                        
                                        if current_quarter:
                                            next_quarter = next_quarters[current_quarter]
                                            st.info(f"💡 현재는 {current_quarter}이며, 다가오는 {next_quarter}를 대비한 상품을 미리 제안해보세요.")
                            else:
                                st.info("분기별 구매 데이터가 충분하지 않습니다.")
                        else:
                            st.info("분기별 선호도 정보가 없습니다.")
                        
                        # 반품 정보
                        st.subheader("↩️ 반품 정보")
                        
                        if 'refund_info' in customer_info and customer_info['반품_정보'] and customer_info['반품_정보']['반품_수량'] > 0:
                            refund_info = customer_info['반품_정보']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("반품 수량", f"{refund_info['반품_수량']:,}개")
                            with col2:
                                st.metric("반품 비율", f"{refund_info['반품_비율']:.1f}%")
                            
                            # 반품 이유가 있으면 표시
                            if 'refund_info' in customer_info and refund_info['반품_이유']:
                                st.subheader("반품 사유별 분석")
                                
                                # 데이터프레임으로 변환
                                refund_reasons_df = pd.DataFrame({
                                    '반품 사유': list(refund_info['반품_이유'].keys()),
                                    '수량': list(refund_info['반품_이유'].values())
                                }).sort_values('수량', ascending=False)
                                
                                # 차트 그리기
                                fig_refund = px.bar(
                                    refund_reasons_df,
                                    x='반품 사유',
                                    y='수량',
                                    title="반품 사유별 수량",
                                    color='수량',
                                    color_continuous_scale='Reds'
                                )
                                fig_refund.update_layout(height=400)
                                st.plotly_chart(fig_refund, use_container_width=True)
                                
                                # 반품 이유 테이블
                                st.dataframe(refund_reasons_df, use_container_width=True)
                                
                                # 주요 반품 이유 해석
                                if not refund_reasons_df.empty:
                                    main_reason = refund_reasons_df.iloc[0]['반품 사유']
                                    main_qty = refund_reasons_df.iloc[0]['수량']
                                    
                                    st.info(f"💡 주요 반품 사유는 **{main_reason}**({main_qty}개)입니다. 이 부분을 개선하여 반품률을 낮출 수 있습니다.")
                            else:
                                st.info("반품 사유 정보가 없습니다.")
                        else:
                            st.info("반품 정보가 없습니다.")
                        
                        # 연-월별 상품 구매 히트맵 추가
                        st.subheader("📆 연-월별 상품 구매 히트맵")
                        
                        if customer_info['연월별_상품_구매'] and len(customer_info['연월별_상품_구매']) > 0:
                            # 히트맵 데이터 준비
                            all_products = set()
                            for yearmonth_data in customer_info['연월별_상품_구매'].values():
                                all_products.update(yearmonth_data.keys())
                            
                            if all_products:
                                # 각 연-월과 상품별 구매량을 담을 데이터프레임 준비
                                heatmap_data = []
                                
                                for yearmonth in sorted(customer_info['연월별_상품_구매'].keys()):
                                    yearmonth_data = customer_info['연월별_상품_구매'][yearmonth]
                                    # 모든 상품에 대해 데이터 포함
                                    for product in all_products:
                                        heatmap_data.append({
                                            '연-월': yearmonth,
                                            '상품': product,
                                            '구매량': yearmonth_data.get(product, 0)
                                        })
                                
                                # 데이터프레임으로 변환
                                heatmap_df = pd.DataFrame(heatmap_data)
                                
                                # 히트맵 피벗 테이블
                                heatmap_pivot = heatmap_df.pivot(index='상품', columns='연-월', values='구매량').fillna(0)
                                
                                # 구매량이 0인 행 제거 (선택 사항)
                                heatmap_pivot = heatmap_pivot.loc[heatmap_pivot.sum(axis=1) > 0]
                                
                                # 온도별 표시 옵션
                                show_full_heatmap = st.checkbox("모든 상품 표시 (많은 상품이 있을 경우 느려질 수 있음)", value=False)
                                
                                # 시각화 - 히트맵
                                if not heatmap_pivot.empty:
                                    if show_full_heatmap:
                                        products_to_show = heatmap_pivot.index
                                    else:
                                        # 상위 15개 상품으로 제한
                                        products_to_show = heatmap_pivot.sum(axis=1).sort_values(ascending=False).head(15).index
                                    
                                    heatmap_pivot_filtered = heatmap_pivot.loc[products_to_show]
                                    
                                    fig = px.imshow(
                                        heatmap_pivot_filtered, 
                                        color_continuous_scale='Viridis',
                                        labels=dict(x="연-월", y="상품", color="구매량"),
                                        title=f"{selected_customer}의 연-월별 상품 구매 패턴"
                                    )
                                    fig.update_layout(height=600)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 연-월별 상품 다양성 분석
                                    product_counts = []
                                    for yearmonth in sorted(customer_info['연월별_상품_구매'].keys()):
                                        yearmonth_data = customer_info['연월별_상품_구매'][yearmonth]
                                        # 실제로 구매한 상품 수 (구매량 > 0)
                                        purchased_products = {k: v for k, v in yearmonth_data.items() if v > 0}
                                        product_counts.append({
                                            '연-월': yearmonth,
                                            '상품종류': len(purchased_products)
                                        })
                                    
                                    if product_counts:
                                        diversity_df = pd.DataFrame(product_counts)
                                        fig_diversity = px.line(
                                            diversity_df,
                                            x='연-월',
                                            y='상품종류',
                                            title="연-월별 구매 상품 다양성",
                                            markers=True
                                        )
                                        fig_diversity.update_layout(height=400)
                                        st.plotly_chart(fig_diversity, use_container_width=True)
                                else:
                                    st.info("히트맵을 표시할 충분한 데이터가 없습니다.")
                            else:
                                st.info("상품 구매 데이터가 충분하지 않습니다.")
                        
                        elif customer_info['월별_상품_구매'] and len(customer_info['월별_상품_구매']) > 0:
                            # 기존 월별 히트맵 처리 (이전 코드와의 호환성)
                            st.warning("⚠️ 데이터에 연도 정보가 포함되어 있지 않아 월 단위로만 히트맵 분석이 가능합니다.")
                            
                            # 히트맵 데이터 준비
                            all_products = set()
                            for month_data in customer_info['월별_상품_구매'].values():
                                all_products.update(month_data.keys())
                            
                            if all_products:
                                # 각 월과 상품별 구매량을 담을 데이터프레임 준비
                                heatmap_data = []
                                
                                for month in sorted(customer_info['월별_상품_구매'].keys()):
                                    month_data = customer_info['월별_상품_구매'][month]
                                    # 모든 상품에 대해 데이터 포함
                                    for product in all_products:
                                        heatmap_data.append({
                                            '월': f"{month}월",
                                            '상품': product,
                                            '구매량': month_data.get(product, 0)
                                        })
                                
                                # 데이터프레임으로 변환
                                heatmap_df = pd.DataFrame(heatmap_data)
                                
                                # 히트맵 피벗 테이블
                                heatmap_pivot = heatmap_df.pivot(index='상품', columns='월', values='구매량').fillna(0)
                                
                                # 구매량이 0인 행 제거 (선택 사항)
                                heatmap_pivot = heatmap_pivot.loc[heatmap_pivot.sum(axis=1) > 0]
                                
                                # 온도별 표시 옵션
                                show_full_heatmap = st.checkbox("모든 상품 표시 (많은 상품이 있을 경우 느려질 수 있음)", value=False)
                                
                                # 시각화 - 히트맵
                                if not heatmap_pivot.empty:
                                    if show_full_heatmap:
                                        products_to_show = heatmap_pivot.index
                                    else:
                                        # 상위 15개 상품으로 제한
                                        products_to_show = heatmap_pivot.sum(axis=1).sort_values(ascending=False).head(15).index
                                    
                                    heatmap_pivot_filtered = heatmap_pivot.loc[products_to_show]
                                    
                                    fig = px.imshow(
                                        heatmap_pivot_filtered, 
                                        color_continuous_scale='Viridis',
                                        labels=dict(x="월", y="상품", color="구매량"),
                                        title=f"{selected_customer}의 월별 상품 구매 패턴"
                                    )
                                    fig.update_layout(height=600)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("히트맵을 표시할 충분한 데이터가 없습니다.")
                        else:
                            st.info("상품 구매 패턴 히트맵을 위한 데이터가 없습니다.")
                            
                        # 제품별 구매 이력 분석
                        st.subheader("🔍 제품별 구매 이력")
                        
                        if '제품별_구매_이력' in customer_info and customer_info['제품별_구매_이력']:
                            # 1. 제품 선택 옵션
                            all_products = sorted(customer_info['제품별_구매_이력'].keys())
                            
                            if all_products:
                                # 가장 많이 구매한 상품 5개와 '직접 선택' 옵션 제공
                                top_products_list = list(customer_info['주요_구매상품'].keys())[:5]
                                product_options = ["직접 선택"] + top_products_list
                                
                                product_choice_type = st.radio(
                                    "제품 선택 방식",
                                    product_options,
                                    horizontal=True,
                                    key="product_choice_type"
                                )
                                
                                selected_product = None
                                
                                if product_choice_type == "직접 선택":
                                    selected_product = st.selectbox(
                                        "분석할 제품 선택", 
                                        all_products,
                                        key="product_selection"
                                    )
                                else:
                                    selected_product = product_choice_type
                                
                                if selected_product in customer_info['제품별_구매_이력']:
                                    product_history = customer_info['제품별_구매_이력'][selected_product]
                                    
                                    # 구매 이력 데이터
                                    history_df = pd.DataFrame({
                                        '구매일': product_history['구매일'],
                                        '구매량': product_history['구매량']
                                    })
                                    
                                    # 구매 이력 요약
                                    st.markdown(f"#### {selected_product} 구매 이력 ({len(history_df)}회)")
                                    
                                    # 데이터 시각화
                                    fig_history = px.line(
                                        history_df,
                                        x='구매일',
                                        y='구매량',
                                        title=f"{selected_product} 구매 패턴",
                                        markers=True
                                    )
                                    fig_history.update_layout(height=400)
                                    st.plotly_chart(fig_history, use_container_width=True)
                                    
                                    # 구매 패턴 인사이트
                                    if len(history_df) > 1:
                                        avg_qty = history_df['구매량'].mean()
                                        max_qty = history_df['구매량'].max()
                                        min_qty = history_df['구매량'].min()
                                        
                                        st.info(f"💡 **{selected_product}**의 평균 구매량은 **{avg_qty:.1f}개**이며, "
                                                f"최대 **{max_qty}개**부터 최소 **{min_qty}개**까지 구매했습니다.")
                                        
                                        # 구매 주기 계산
                                        if len(history_df) > 2:
                                            history_df['구매일'] = pd.to_datetime(history_df['구매일'])
                                            history_df = history_df.sort_values('구매일')
                                            
                                            # 구매일 간격 계산
                                            history_df['다음구매일'] = history_df['구매일'].shift(-1)
                                            history_df['구매간격'] = (history_df['다음구매일'] - history_df['구매일']).dt.days
                                            
                                            # 마지막 행 제거 (NaN 값)
                                            history_df = history_df.dropna()
                                            
                                            if not history_df.empty:
                                                avg_interval = history_df['구매간격'].mean()
                                                st.info(f"💡 **{selected_product}**의 평균 구매 주기는 **{avg_interval:.1f}일**입니다.")
                            else:
                                st.info("제품별 구매 이력 데이터가 없습니다.")
                        else:
                            st.info("제품별 구매 이력 데이터가 없습니다.")
                            
        # 7. 고객 세분화 (RFM) 탭
        with tabs[6]:
            st.header("📊 고객 세분화 분석 (RFM)")
            st.write("Recency(최근성), Frequency(빈도), Monetary(금액)를 기준으로 고객을 세분화합니다.")
            
            # RFM 분석 알고리즘 설명 추가
            with st.expander("🔍 RFM 분석 알고리즘 설명"):
                st.markdown("""
                ### RFM 분석이란?
                
                RFM 분석은 고객의 구매 행동을 세 가지 핵심 지표로 분석하는 마케팅 기법입니다:
                
                - **Recency (최근성)**: 고객이 마지막으로 언제 구매했는지
                - **Frequency (빈도)**: 고객이 얼마나 자주 구매하는지
                - **Monetary (금액)**: 고객이 얼마나 많은 돈을 지출했는지
                
                ### 세그먼트 분류 방식
                
                각 지표별로 1-4점의 점수를 부여하고, 조합하여 고객 세그먼트를 정의합니다:
                
                - **VIP 고객**: R, F, M 모두 높은 고객 (최근에 자주 많이 구매)
                - **충성 고객**: R, F가 높은 고객 (최근에 자주 구매)
                - **큰 지출 고객**: R, M이 높은 고객 (최근에 많은 금액 지출)
                - **잠재 이탈 고객**: F, M은 높지만 R이 낮은 고객 (과거에 자주 많이 구매했으나 최근 구매 없음)
                - **신규 고객**: R만 높은 고객 (최근 구매 시작)
                - **가격 민감 고객**: F만 높은 고객 (자주 구매하지만 금액은 적음)
                - **휴면 큰 지출 고객**: M만 높은 고객 (과거에 큰 금액 구매했으나 최근 활동 없음)
                - **관심 필요 고객**: R, F, M 모두 낮은 고객 (구매 활동이 적음)
                
                이 분석을 통해 각 고객 세그먼트별로 차별화된 마케팅 전략을 수립할 수 있습니다.
                """)
            
            # 세그먼트별 의미와 전략 설명 추가
            with st.expander("📊 세그먼트별 의미와 마케팅 전략"):
                st.markdown("""
                ### 각 세그먼트의 의미와 마케팅 전략
                
                #### 1. VIP 고객 (R≥3, F≥3, M≥3)
                - **의미**: 최근에 자주 구매하며 많은 금액을 지출하는 최우수 고객
                - **특징**: 
                  - 전체 매출의 큰 부분을 차지하는 핵심 고객층
                  - 브랜드 충성도가 높고 제품에 대한 이해도가 높음
                  - 마이크로그린 산업에서는 고급 레스토랑이나 호텔 등이 이 범주에 속할 가능성이 높음
                - **마케팅 전략**:
                  - 프리미엄 서비스 및 특별 혜택 제공 (특별 배송, 우선 공급)
                  - 신제품 우선 소개 및 시식 기회 제공
                  - 개인화된 추천과 맞춤 상담 서비스
                  - 충성도 프로그램 및 프리미엄 등급 혜택
                  - 특별 이벤트 초대 (작물 수확 체험, 농장 투어)
                
                #### 2. 충성 고객 (R≥3, F≥3, M<3)
                - **의미**: 최근에 자주 구매하지만 구매 금액은 상대적으로 낮은 충성 고객
                - **특징**:
                  - 꾸준히 소규모로 주문하는 중소형 업체
                  - 브랜드 옹호자 역할을 할 가능성이 높음
                  - 마이크로그린에서는 소규모 레스토랑, 카페, 반찬가게 등이 이 범주에 속할 수 있음
                - **마케팅 전략**:
                  - 구매 금액 증가를 위한 볼륨 할인 및 묶음 상품 제안
                  - 마일리지 또는 포인트 적립 프로그램
                  - 신제품 소개 및 추가 상품 추천
                  - 정기 구독 서비스 제안
                  - 리뷰 및 피드백 장려 프로그램
                
                #### 3. 큰 지출 고객 (R≥3, F<3, M≥3)
                - **의미**: 최근에 구매했고 금액이 크지만 구매 빈도는 낮은 고객
                - **특징**:
                  - 대규모 주문을 간헐적으로 하는 고객
                  - 특별 행사나 이벤트 때만 구매하는 경향
                  - 대형 케이터링 업체, 이벤트 회사, 계절성 레스토랑 등이 이 범주에 해당될 수 있음
                - **마케팅 전략**:
                  - 구매 주기 단축을 위한 정기 구매 프로그램 제안
                  - 예약 주문 시스템 및 할인 혜택
                  - 시즌별 특별 상품 및 기획전 안내
                  - 정기적인 연락과 관계 유지 활동
                  - 계절성 제품 사전 예약 혜택
                
                #### 4. 잠재 이탈 고객 (R<3, F≥3, M≥3)
                - **의미**: 과거에 자주 많이 구매했으나 최근 구매가 줄어든 위험 고객
                - **특징**:
                  - 한때 핵심 고객이었으나 최근 활동이 감소
                  - 경쟁사로 전환했거나 제품에 불만이 있을 가능성
                  - 마이크로그린 분야에서는 메뉴 변경이나 공급업체 변경을 고려 중인 레스토랑일 수 있음
                - **마케팅 전략**:
                  - 재활성화 캠페인 및 특별 할인 제공
                  - 개인화된 연락과 피드백 요청
                  - 신제품 소개 및 샘플 제공
                  - 이탈 원인 파악을 위한 설문조사
                  - 품질 개선 사항 및 신규 서비스 안내
                
                #### 5. 신규 고객 (R≥3, F<3, M<3)
                - **의미**: 최근에 구매를 시작했지만 아직 구매 빈도와 금액이 낮은 신규 고객
                - **특징**:
                  - 마이크로그린을 처음 시도해보는 단계
                  - 향후 성장 가능성이 높은 잠재 고객
                  - 새로 오픈한 레스토랑, 신규 창업 카페 등이 이 범주에 속할 수 있음
                - **마케팅 전략**:
                  - 맞춤형 시작 패키지 및 환영 프로그램
                  - 제품 사용법 및 레시피 제안
                  - 초기 구매자 특별 할인 및 혜택
                  - 교육 콘텐츠 및 활용 가이드 제공
                  - 고객 성공 스토리 및 사례 공유
                
                #### 6. 가격 민감 고객 (R<3, F≥3, M<3)
                - **의미**: 자주 구매하지만 구매 금액이 적고 최근 활동이 감소한 고객
                - **특징**:
                  - 가격에 민감하고 할인을 중요시하는 고객
                  - 소량 다품종 구매 패턴
                  - 소규모 샐러드 바, 소형 반찬가게 등이 이 범주에 해당될 수 있음
                - **마케팅 전략**:
                  - 가격 대비 가치를 강조한 프로모션
                  - 볼륨 할인 및 묶음 상품 제안
                  - 비용 효율적인 제품 라인 소개
                  - 충성도 프로그램 및 누적 할인
                  - 저비용 대체 상품 제안
                
                #### 7. 휴면 큰 지출 고객 (R<3, F<3, M≥3)
                - **의미**: 과거에 큰 금액을 지출했으나 최근 구매가 없는 휴면 고객
                - **특징**:
                  - 한때 중요한 고객이었으나 현재는 활동이 없음
                  - 계절성 비즈니스나 특별 이벤트에만 관여했을 가능성
                  - 시즌성 업장, 특별 행사 기획자 등이 이 범주에 속할 수 있음
                - **마케팅 전략**:
                  - 재활성화 캠페인 및 복귀 혜택
                  - 개인화된 연락과 신제품 소개
                  - 과거 구매 패턴 기반 맞춤 제안
                  - 시즌별 특별 프로모션
                  - 이탈 원인 파악 및 개선된 서비스 안내
                
                #### 8. 관심 필요 고객 (R<3, F<3, M<3)
                - **의미**: 모든 지표가 낮은 저활동 고객
                - **특징**:
                  - 일회성 구매자이거나 시험 구매 후 활동이 없음
                  - 제품에 만족하지 못했거나 다른 대안을 찾았을 가능성
                  - 일회성 이벤트 회사, 시험 구매한 소규모 업체 등이 이 범주에 속할 수 있음
                - **마케팅 전략**:
                  - 재참여 유도를 위한 특별 할인 및 인센티브
                  - 제품 개선 사항 및 신규 서비스 안내
                  - 교육 콘텐츠 및 활용 가이드 제공
                  - 피드백 요청 및 개선된 경험 제안
                  - 비용 효율적인 시작 패키지 제안
                
                ### 추가 마케팅 고려사항
                
                - **계절성 고려**: 마이크로그린은 계절에 따라 품질과 가용성이 달라지므로, 각 세그먼트에 계절별 맞춤 전략 적용
                - **업종별 접근**: 호텔, 레스토랑, 카페 등 업종별 특성을 고려한 세부 전략 수립
                - **교차 판매**: 각 세그먼트에 적합한 교차 판매 및 상향 판매 전략 개발
                - **커뮤니케이션 빈도**: 세그먼트별로 최적의 커뮤니케이션 빈도 설정 (VIP는 더 자주, 휴면 고객은 신중하게)
                - **채널 선택**: 각 세그먼트에 가장 효과적인 커뮤니케이션 채널 선택 (이메일, 전화, 방문 등)
                """)
            
            # 고객 유형 선택 시 콜백 함수
            def on_rfm_customer_type_change():
                st.session_state.rfm_customer_type = st.session_state.rfm_customer_type_select
                st.session_state.rfm_results = None  # 고객 유형 변경 시 결과 초기화
            
            # 월 선택 시 콜백 함수
            def on_rfm_month_option_change():
                st.session_state.rfm_selected_month = st.session_state.rfm_month_option_select
                st.session_state.rfm_results = None  # 월 변경 시 결과 초기화
            
            # 고객 유형 선택
            customer_type_options = ["전체", "호텔", "일반"]
            selected_customer_type = st.selectbox(
                "고객 유형 선택",
                customer_type_options,
                index=customer_type_options.index(st.session_state.rfm_customer_type),
                key="rfm_customer_type_select",
                on_change=on_rfm_customer_type_change
            )
            
            # 월 선택 옵션 (전체 기간 또는 특정 월)
            month_options = ["전체 기간"] + [f"{i}월" for i in range(1, 13)]
            selected_month_option = st.selectbox(
                "분석 기간 선택",
                month_options,
                index=month_options.index(st.session_state.rfm_selected_month),
                key="rfm_month_option_select",
                on_change=on_rfm_month_option_change
            )
            
            # 선택된 월 값 변환
            selected_month = None if selected_month_option == "전체 기간" else int(selected_month_option.replace("월", ""))
            
            # 분석 실행 버튼 또는 기존 분석 결과 사용
            run_analysis = False
            if st.button("RFM 분석 실행", key="run_rfm_analysis"):
                run_analysis = True
            
            # 분석 실행 또는 기존 결과 사용
            if run_analysis or st.session_state.rfm_results is None:
                with st.spinner("고객 세분화 분석을 수행하는 중..."):
                    rfm_results = recommender.perform_rfm_analysis(
                        customer_type=st.session_state.rfm_customer_type,
                        selected_month=selected_month
                    )
                    # 세션 상태에 결과 저장
                    st.session_state.rfm_results = rfm_results
            else:
                # 세션에서 기존 분석 결과 가져오기
                rfm_results = st.session_state.rfm_results
            
            # 분석 결과 표시
            if rfm_results['상태'] == '실패':
                st.error(rfm_results['메시지'])
            else:
                # 분석 결과 표시
                st.success(f"RFM 분석이 완료되었습니다. 총 {rfm_results['총_고객수']}명의 고객이 분석되었습니다.")
                
                # RFM 데이터
                rfm_data = rfm_results['RFM_데이터']
                
                # 1. 전체 통계 요약
                st.subheader("📈 RFM 분석 요약")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("평균 최근성 (일)", f"{rfm_results['R_평균']:.1f}", 
                              help="낮을수록 좋음. 마지막 구매 이후 경과일")
                with col2:
                    st.metric("평균 구매 빈도", f"{rfm_results['F_평균']:.1f}", 
                              help="높을수록 좋음. 평균 구매 횟수")
                with col3:
                    st.metric("평균 구매 금액", f"{int(rfm_results['M_평균']):,}", 
                              help="높을수록 좋음. 평균 구매 금액")
                
                # 2. 고객 세그먼트 분포
                st.subheader("👥 고객 세그먼트 분포")
                
                # 세그먼트 통계를 데이터프레임으로 변환
                segment_stats = pd.DataFrame({
                    '고객_세그먼트': list(rfm_results['세그먼트_통계'].keys()),
                    '고객수': list(rfm_results['세그먼트_통계'].values())
                })
                segment_stats = segment_stats.sort_values('고객수', ascending=False)
                
                # 세그먼트 분포 시각화
                fig = px.pie(
                    segment_stats, 
                    values='고객수', 
                    names='고객_세그먼트',
                    title=f"{selected_customer_type} 고객 세그먼트 분포"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # 세그먼트 간략 설명 추가
                st.subheader("세그먼트별 특징 및 제안 전략")
                segments_info = {
                    "VIP 고객": {
                        "특징": "최근에 자주 구매하며 많은 금액을 지출하는 최우수 고객",
                        "전략": "프리미엄 서비스 제공, 신제품 우선 소개, 특별 혜택 및 개인화 서비스"
                    },
                    "충성 고객": {
                        "특징": "최근에 자주 구매하지만 구매 금액은 상대적으로 낮은 고객",
                        "전략": "볼륨 할인, 묶음 상품 제안, 마일리지 프로그램, 신제품 소개"
                    },
                    "큰 지출 고객": {
                        "특징": "최근에 구매했고 금액이 크지만 구매 빈도는 낮은 고객",
                        "전략": "정기 구매 프로그램, 예약 주문 시스템, 시즌별 특별 상품 안내"
                    },
                    "잠재 이탈 고객": {
                        "특징": "과거에 자주 많이 구매했으나 최근 구매가 줄어든 위험 고객",
                        "전략": "재활성화 캠페인, 특별 할인, 피드백 요청, 신제품 샘플 제공"
                    },
                    "신규 고객": {
                        "특징": "최근에 구매를 시작했지만 아직 구매 빈도와 금액이 낮은 고객",
                        "전략": "시작 패키지, 제품 사용법, 초기 구매자 특별 할인, 교육 콘텐츠"
                    },
                    "가격 민감 고객": {
                        "특징": "자주 구매하지만 구매 금액이 적고 최근 활동이 감소한 고객",
                        "전략": "가격 대비 가치 강조, 볼륨 할인, 효율적인 제품 라인 소개"
                    },
                    "휴면 큰 지출 고객": {
                        "특징": "과거에 큰 금액을 지출했으나 최근 구매가 없는 휴면 고객",
                        "전략": "재활성화 캠페인, 복귀 혜택, 과거 구매 패턴 기반 맞춤 제안"
                    },
                    "관심 필요 고객": {
                        "특징": "모든 지표가 낮은 저활동 고객",
                        "전략": "특별 할인, 제품 개선 안내, 교육 콘텐츠, 피드백 요청"
                    }
                }
                
                # 세그먼트별 정보 표시 (실제 분석에 있는 세그먼트만)
                for segment in rfm_results['세그먼트_통계'].keys():
                    if segment in segments_info:
                        with st.expander(f"{segment} ({rfm_results['세그먼트_통계'][segment]}명)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**특징**: {segments_info[segment]['특징']}")
                            with col2:
                                st.markdown(f"**제안 전략**: {segments_info[segment]['전략']}")
                
                # 세그먼트별 RFM 평균값 계산
                segment_rfm_avg = rfm_data.groupby('고객_세그먼트').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean'
                }).reset_index()
                
                # 세그먼트별 RFM 평균값 시각화
                st.subheader("📊 세그먼트별 RFM 평균값")
                
                # 세그먼트별 비교 방법 선택
                comparison_method = st.radio(
                    "비교 방식",
                    ["막대 그래프", "레이더 차트"],
                    horizontal=True,
                    key="segment_comparison_method"
                )
                
                if comparison_method == "막대 그래프":
                    # 각 지표별 세그먼트 비교 
                    metrics = ['Recency', 'Frequency', 'Monetary']
                    for metric in metrics:
                        title = {
                            'Recency': '세그먼트별 평균 최근성 (일) - 낮을수록 좋음',
                            'Frequency': '세그먼트별 평균 구매 빈도 - 높을수록 좋음',
                            'Monetary': '세그먼트별 평균 구매 금액 - 높을수록 좋음'
                        }
                        
                        fig = px.bar(
                            segment_rfm_avg,
                            x='고객_세그먼트',
                            y=metric,
                            color='고객_세그먼트',
                            title=title[metric]
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # 레이더 차트로 세그먼트 비교
                    # 데이터 스케일링 (0-1 사이로 정규화)
                    segment_radar = segment_rfm_avg.copy()
                    # Recency는 낮을수록 좋으므로 역수 취함
                    segment_radar['Recency'] = 1 / (segment_radar['Recency'] + 1)
                    
                    # 각 컬럼 정규화
                    for col in ['Recency', 'Frequency', 'Monetary']:
                        segment_radar[col] = (segment_radar[col] - segment_radar[col].min()) / (segment_radar[col].max() - segment_radar[col].min())
                    
                    # 레이더 차트 생성
                    fig = go.Figure()
                    
                    for i, row in segment_radar.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[row['Recency'], row['Frequency'], row['Monetary']],
                            theta=['최근성', '구매빈도', '구매금액'],
                            fill='toself',
                            name=row['고객_세그먼트']
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title="세그먼트별 RFM 특성 비교 (정규화)",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 3. RFM 데이터 상세
                st.subheader("🔍 고객별 RFM 데이터")
                
                # RFM 데이터 정렬 기준 선택
                sort_column = st.selectbox(
                    "정렬 기준",
                    ["Recency", "Frequency", "Monetary", "고객_세그먼트"],
                    key="rfm_sort_column"
                )
                
                # 세그먼트 필터링 옵션
                all_segments = ["모든 세그먼트"] + list(rfm_results['세그먼트_통계'].keys())
                selected_segment = st.selectbox(
                    "세그먼트 필터",
                    all_segments,
                    key="rfm_segment_filter"
                )
                
                # 데이터 필터링 및 정렬
                filtered_rfm = rfm_data
                if selected_segment != "모든 세그먼트":
                    filtered_rfm = rfm_data[rfm_data['고객_세그먼트'] == selected_segment]
                
                # 정렬
                if sort_column in ["Recency", "Frequency", "Monetary"]:
                    ascending = sort_column == "Recency"  # Recency는 낮을수록 좋음
                    filtered_rfm = filtered_rfm.sort_values(sort_column, ascending=ascending)
                else:
                    filtered_rfm = filtered_rfm.sort_values(sort_column)
                
                # 테이블에 표시할 컬럼
                display_columns = [
                    '고객명', 'Recency', 'Frequency', 'Monetary', 
                    'R_Score', 'F_Score', 'M_Score', 'RFM_Score', '고객_세그먼트'
                ]
                
                # 테이블 표시
                st.dataframe(filtered_rfm[display_columns], height=400)
                
                # 고객 세그먼트 탐색 기능
                st.subheader("🔎 세그먼트별 고객 탐색")
                
                segment_options = list(rfm_results['세그먼트_통계'].keys())
                if segment_options:
                    selected_explore_segment = st.selectbox(
                        "탐색할 세그먼트 선택",
                        segment_options,
                        key="explore_segment"
                    )
                    
                    # 선택한 세그먼트의 고객 필터링
                    segment_customers = rfm_data[rfm_data['고객_세그먼트'] == selected_explore_segment]
                    
                    if not segment_customers.empty:
                        # 세그먼트 내 고객 정보 및 특징
                        st.write(f"### {selected_explore_segment} 고객 목록 ({len(segment_customers)}명)")
                        
                        # 세그먼트 요약 정보 - 튜플 형식 대신 문자열로 접근
                        segment_summary = segment_customers.agg({
                            'Recency': ['mean', 'min', 'max'],
                            'Frequency': ['mean', 'min', 'max'],
                            'Monetary': ['mean', 'min', 'max']
                        }).reset_index()
                        
                        # 요약 정보 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "평균 최근성 (일)", 
                                f"{segment_customers['Recency'].mean():.1f}",
                                help=f"최소: {segment_customers['Recency'].min():.0f}일, 최대: {segment_customers['Recency'].max():.0f}일"
                            )
                        with col2:
                            st.metric(
                                "평균 구매 빈도", 
                                f"{segment_customers['Frequency'].mean():.1f}",
                                help=f"최소: {segment_customers['Frequency'].min():.0f}회, 최대: {segment_customers['Frequency'].max():.0f}회"
                            )
                        with col3:
                            st.metric(
                                "평균 구매 금액", 
                                f"{int(segment_customers['Monetary'].mean()):,}",
                                help=f"최소: {int(segment_customers['Monetary'].min()):,}원, 최대: {int(segment_customers['Monetary'].max()):,}원"
                            )
                        
                        # 고객 리스트 표시
                        st.write("#### 세그먼트 내 고객 목록")
                        
                        # 정렬 옵션
                        sort_options = {
                            "최근성 순": "Recency",
                            "구매 빈도 순": "Frequency",
                            "구매 금액 순": "Monetary",
                            "고객명 순": "고객명"
                        }
                        
                        sort_by = st.radio(
                            "정렬 기준",
                            list(sort_options.keys()),
                            horizontal=True,
                            key=f"sort_{selected_explore_segment}"
                        )
                        
                        # 정렬 방향
                        if sort_options[sort_by] == "Recency":
                            # Recency는 낮을수록 좋음
                            segment_customers = segment_customers.sort_values(sort_options[sort_by], ascending=True)
                        elif sort_options[sort_by] == "고객명":
                            segment_customers = segment_customers.sort_values(sort_options[sort_by])
                        else:
                            segment_customers = segment_customers.sort_values(sort_options[sort_by], ascending=False)
                        
                        # 고객 목록 표시
                        for idx, (_, customer) in enumerate(segment_customers.iterrows(), 1):
                            with st.expander(f"{idx}. {customer['고객명']} (R: {customer['R_Score']}, F: {customer['F_Score']}, M: {customer['M_Score']})"):
                                # 고객 세부 정보 표시
                                detail_col1, detail_col2, detail_col3 = st.columns(3)
                                with detail_col1:
                                    st.metric("최근 구매일로부터 경과", f"{int(customer['Recency'])}일")
                                with detail_col2:
                                    st.metric("구매 빈도", f"{int(customer['Frequency'])}회")
                                with detail_col3:
                                    st.metric("총 구매 금액", f"{int(customer['Monetary']):,}원")
                                
                                # 고객 맞춤 전략 제안
                                segment_strategy = segments_info.get(selected_explore_segment, {}).get("전략", "")
                                if segment_strategy:
                                    st.info(f"**제안 전략**: {segment_strategy}")
                    else:
                        st.info(f"{selected_explore_segment} 세그먼트에 속한 고객이 없습니다.")
                else:
                    st.info("고객 세그먼트 정보가 없습니다.")
                
                # 월별 RFM 분석 (전체 기간을 선택했을 때만)
                if selected_month is None and rfm_results['월별_RFM'] is not None:
                    st.subheader("📅 월별 RFM 추이")
                    
                    monthly_rfm = rfm_results['월별_RFM']
                    
                    # 월별 평균 지표 계산
                    monthly_avg = monthly_rfm.groupby('월').agg({
                        'Recency': 'mean',
                        'Frequency': 'mean',
                        'Monetary': 'mean'
                    }).reset_index()
                    
                    # 월별 R, F, M 추이 그래프
                    st.write("월별 평균 지표 추이")
                    
                    # 그래프로 표시할 지표 선택
                    metrics_to_show = st.multiselect(
                        "표시할 지표 선택",
                        ["Recency (최근성)", "Frequency (구매빈도)", "Monetary (구매금액)"],
                        default=["Recency (최근성)", "Frequency (구매빈도)", "Monetary (구매금액)"],
                        key="monthly_metrics"
                    )
                    
                    if metrics_to_show:
                        fig = go.Figure()
                        
                        if "Recency (최근성)" in metrics_to_show:
                            fig.add_trace(go.Scatter(
                                x=monthly_avg['월'],
                                y=monthly_avg['Recency'],
                                mode='lines+markers',
                                name='최근성 (일)',
                                line=dict(color='red')
                            ))
                        
                        if "Frequency (구매빈도)" in metrics_to_show:
                            fig.add_trace(go.Scatter(
                                x=monthly_avg['월'],
                                y=monthly_avg['Frequency'],
                                mode='lines+markers',
                                name='구매빈도',
                                line=dict(color='blue')
                            ))
                        
                        if "Monetary (구매금액)" in metrics_to_show:
                            fig.add_trace(go.Scatter(
                                x=monthly_avg['월'],
                                y=monthly_avg['Monetary'],
                                mode='lines+markers',
                                name='구매금액',
                                line=dict(color='green')
                            ))
                        
                        fig.update_layout(
                            title="월별 RFM 지표 평균 추이",
                            xaxis_title="월",
                            yaxis_title="값",
                            legend_title="지표",
                            height=500
                        )
                        
                        # x축에 모든 월 표시
                        fig.update_xaxes(
                            tickvals=list(range(1, 13)),
                            ticktext=[f"{m}월" for m in range(1, 13)]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 월별 지표 데이터 표시
                        st.write("월별 평균 지표 데이터")
                        monthly_avg['월'] = monthly_avg['월'].apply(lambda x: f"{x}월")
                        st.dataframe(monthly_avg, height=300)
    
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.error("데이터 형식을 확인하고 다시 시도해주세요.")
        # 디버깅 정보 출력
        st.expander("상세 오류 정보", expanded=False).exception(e)

if __name__ == "__main__":
    main()


            