import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt
import pickle

# 데이터 불러오기
train_df = pd.read_csv("ratings_train.txt", sep='\t', encoding='utf-8')
test_df = pd.read_csv("ratings_test.txt", sep='\t', encoding='utf-8')

# 결측치 및 공백 제거
train_df = train_df.dropna()
test_df = test_df.dropna()

# 형태소 분석기 정의
okt = Okt()

def tokenize(text):
    return okt.morphs(text, stem=True)

# 벡터화 (형태소 분석 포함)
vectorizer = CountVectorizer(tokenizer=tokenize)
X_train = vectorizer.fit_transform(train_df['document'])
y_train = train_df['label']

X_test = vectorizer.transform(test_df['document'])
y_test = test_df['label']

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 정확도 출력 # 0.8459! 정확도가 제일 좋았으니 이걸로 모델, 벡터 저장!
predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print("정확도:", accuracy)

# 예측이 틀린 문장 6개만 출력

print("\n예측이 틀린 문장들\n")

# 틀린 문장들이 너무 많아 6개 정도만 추출
count = 0

for i in range(len(y_test)):
    actual = y_test.iloc[i]
    prediction = predicted[i]

    if actual != prediction:
        original_text = test_df.iloc[i]["document"]
        real_label = "긍정" if actual == 1 else "부정"
        predicted_label = "긍정" if prediction == 1 else "부정"

        print(f"- 문장: {original_text}")
        print(f"  실제: {real_label} / 예측: {predicted_label}\n")

        count += 1
        if count == 6:
            break

# 모델 및 벡터 저장
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)