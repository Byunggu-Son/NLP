import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# csv파일 불러오기 및 데이터 확인
df = pd.read_csv("ko_sentiment_1000.csv")
print(df.head())


# 학습용 데이터 준비
X = df["text"]
y = df["label"]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 벡터화
vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 모델 학습
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# 테스트 예측 및 정확도 평가
y_pred = model.predict(X_test_vector)
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 정확도: {accuracy * 100:.2f}%")


predicted = model.predict(X_test_vector)
print("\n 예측이 틀린 문장들:\n")

for i in range(len(y_test)):
    actual = y_test.iloc[i]
    prediction = predicted[i]
    
    if actual != prediction:
        original_text = df.iloc[y_test.index[i]]["text"]  # 원본 문장
        real_label = "긍정" if actual == 1 else "부정"
        predicted_label = "긍정" if prediction == 1 else "부정"
        
        print(f"- 문장: {original_text}")
        print(f"  실제: {real_label} / 예측: {predicted_label}\n")