import pandas as pd

# 1. 테스트 데이터 불러오기
test_df = pd.read_csv("ratings_train.txt", sep='\t', encoding='utf-8')

# 2. 결측치 제거
test_df = test_df.dropna()

# 3. 전체 데이터 개수
print("총 데이터 수:", len(test_df))

# 4. 레이블 분포 확인
label_counts = test_df['label'].value_counts()
print("\n레이블 분포:")
print(f"부정(0): {label_counts[0]}개")
print(f"긍정(1): {label_counts[1]}개")
