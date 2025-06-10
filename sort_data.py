import pandas as pd

# 파일 경로 - 판매 데이터
sales_input_file = "merged_2023_2024_2025.xlsx"
sales_output_file = "sorted_sales_data.xlsx"

# 파일 경로 - 반품 데이터
refund_input_file = "merged_returns_2024_2025.xlsx"
refund_output_file = "sorted_refund_data.xlsx"

def sort_sales_data():
    """판매 데이터를 정렬하고 새 파일로 저장"""
    try:
        # 파일 읽기
        print("판매 데이터 파일 읽는 중...")
        sales_data = pd.read_excel(sales_input_file)
        print(f"판매 데이터 로드 완료: {len(sales_data)}개 레코드")
        
        # 컬럼명 공백 제거
        sales_data.columns = sales_data.columns.str.strip()
        print("판매 데이터 컬럼:", list(sales_data.columns))
        
        # 컬럼 매핑
        sales_data_renamed = sales_data.rename(columns={
            '거래처': '고객명',
            '품  목': '상품',
            '품목': '상품',
            '합 계': '금액',
            '합계': '금액',
            '월일': '날짜'
        })
        
        # "재고조정"이 포함된 데이터 제외
        if '고객명' in sales_data_renamed.columns:
            before_count = len(sales_data_renamed)
            # "재고조정"이 포함된 모든 거래처 제외
            sales_data_renamed = sales_data_renamed[~sales_data_renamed['고객명'].str.contains("재고조정", na=False)]
            filtered_count = before_count - len(sales_data_renamed)
            print(f"'재고조정'이 포함된 항목 {filtered_count}개가 제외되었습니다.")
        elif '거래처' in sales_data_renamed.columns:
            before_count = len(sales_data_renamed)
            # "재고조정"이 포함된 모든 거래처 제외
            sales_data_renamed = sales_data_renamed[~sales_data_renamed['거래처'].str.contains("재고조정", na=False)]
            filtered_count = before_count - len(sales_data_renamed)
            print(f"'재고조정'이 포함된 항목 {filtered_count}개가 제외되었습니다.")
        
        # 정렬 컬럼 확인 - 거래처 우선 정렬
        sort_columns = []
        
        # 거래처/고객명 컬럼 확인
        if '고객명' in sales_data_renamed.columns:
            sort_columns.append('고객명')
            print("'고객명' 컬럼으로 1차 정렬합니다")
        elif '거래처' in sales_data_renamed.columns:
            sort_columns.append('거래처')
            print("'거래처' 컬럼으로 1차 정렬합니다")
        
        # 날짜 컬럼 확인
        if '날짜' in sales_data_renamed.columns:
            sort_columns.append('날짜')
            print("'날짜' 컬럼으로 2차 정렬합니다")
        elif '월일' in sales_data_renamed.columns:
            sort_columns.append('월일')
            print("'월일' 컬럼으로 2차 정렬합니다")
            
        # 상품 컬럼 확인
        if '상품' in sales_data_renamed.columns:
            sort_columns.append('상품')
            print("'상품' 컬럼으로 3차 정렬합니다")
        
        # 정렬 수행
        if sort_columns:
            print(f"다음 컬럼으로 정렬: {sort_columns}")
            sales_data_sorted = sales_data_renamed.sort_values(by=sort_columns)
            print(f"정렬 완료! 총 {len(sales_data_sorted)}개 행이 정렬되었습니다.")
        else:
            print("정렬할 컬럼을 찾지 못했습니다. 원본 데이터 그대로 저장합니다.")
            sales_data_sorted = sales_data_renamed
        
        # 정렬된 데이터 저장
        sales_data_sorted.to_excel(sales_output_file, index=False)
        print(f"판매 데이터 정렬 완료: {sales_output_file} 파일로 저장됨")
        
        # 거래처 목록 출력 (처음 10개)
        if '고객명' in sales_data_sorted.columns:
            unique_customers = sales_data_sorted['고객명'].unique()
            print(f"총 {len(unique_customers)}개의 고유 거래처가 있습니다.")
            print("처음 10개 거래처:")
            for i, customer in enumerate(unique_customers[:10]):
                print(f"  {i+1}. {customer}")
        elif '거래처' in sales_data_sorted.columns:
            unique_customers = sales_data_sorted['거래처'].unique()
            print(f"총 {len(unique_customers)}개의 고유 거래처가 있습니다.")
            print("처음 10개 거래처:")
            for i, customer in enumerate(unique_customers[:10]):
                print(f"  {i+1}. {customer}")
        
        return True
    except Exception as e:
        print(f"판매 데이터 정렬 중 오류 발생: {e}")
        return False

def sort_refund_data():
    """반품 데이터를 정렬하고 새 파일로 저장"""
    try:
        # 파일 읽기
        print("반품 데이터 파일 읽는 중...")
        refund_data = pd.read_excel(refund_input_file)
        print(f"반품 데이터 로드 완료: {len(refund_data)}개 레코드")
        
        # 컬럼명 공백 제거
        refund_data.columns = refund_data.columns.str.strip()
        print("반품 데이터 컬럼:", list(refund_data.columns))
        
        # 컬럼 매핑
        refund_data_renamed = refund_data.rename(columns={
            '품       목': '상품',
            '품목': '상품',
            '반품유형': '반품사유',
            '반품일자': '날짜'
        })
        
        # "재고조정"이 포함된 데이터 제외
        if '고객명' in refund_data_renamed.columns:
            before_count = len(refund_data_renamed)
            # "재고조정"이 포함된 모든 거래처 제외
            refund_data_renamed = refund_data_renamed[~refund_data_renamed['고객명'].str.contains("재고조정", na=False)]
            filtered_count = before_count - len(refund_data_renamed)
            print(f"'재고조정'이 포함된 항목 {filtered_count}개가 제외되었습니다.")
        elif '거래처' in refund_data_renamed.columns:
            before_count = len(refund_data_renamed)
            # "재고조정"이 포함된 모든 거래처 제외
            refund_data_renamed = refund_data_renamed[~refund_data_renamed['거래처'].str.contains("재고조정", na=False)]
            filtered_count = before_count - len(refund_data_renamed)
            print(f"'재고조정'이 포함된 항목 {filtered_count}개가 제외되었습니다.")
        
        # 정렬 컬럼 확인 - 거래처 우선 정렬
        sort_columns = []
        
        # 거래처/고객명 컬럼 확인
        if '고객명' in refund_data_renamed.columns:
            sort_columns.append('고객명')
            print("'고객명' 컬럼으로 1차 정렬합니다")
        elif '거래처' in refund_data_renamed.columns:
            sort_columns.append('거래처')
            print("'거래처' 컬럼으로 1차 정렬합니다")
        
        # 날짜 컬럼 확인
        if '날짜' in refund_data_renamed.columns:
            sort_columns.append('날짜')
            print("'날짜' 컬럼으로 2차 정렬합니다")
        elif '반품일자' in refund_data_renamed.columns:
            sort_columns.append('반품일자')
            print("'반품일자' 컬럼으로 2차 정렬합니다")
            
        # 상품 컬럼 확인
        if '상품' in refund_data_renamed.columns:
            sort_columns.append('상품')
            print("'상품' 컬럼으로 3차 정렬합니다")
        
        # 정렬 수행
        if sort_columns:
            print(f"다음 컬럼으로 정렬: {sort_columns}")
            refund_data_sorted = refund_data_renamed.sort_values(by=sort_columns)
            print(f"정렬 완료! 총 {len(refund_data_sorted)}개 행이 정렬되었습니다.")
        else:
            print("정렬할 컬럼을 찾지 못했습니다. 원본 데이터 그대로 저장합니다.")
            refund_data_sorted = refund_data_renamed
        
        # 정렬된 데이터 저장
        refund_data_sorted.to_excel(refund_output_file, index=False)
        print(f"반품 데이터 정렬 완료: {refund_output_file} 파일로 저장됨")
        
        return True
    except Exception as e:
        print(f"반품 데이터 정렬 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    print("엑셀 데이터 정렬 시작...")
    print("=" * 50)
    
    print("\n[1/2] 판매 데이터 정렬 중...")
    sales_result = sort_sales_data()
    
    print("\n" + "=" * 50)
    
    print("\n[2/2] 반품 데이터 정렬 중...")
    refund_result = sort_refund_data()
    
    print("\n" + "=" * 50)
    
    if sales_result and refund_result:
        print("\n모든 데이터 정렬 완료!")
    else:
        print("\n일부 데이터 정렬에 실패했습니다.")
    
    print("\n모든 작업이 완료되었습니다.") 