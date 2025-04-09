import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv("ko_sentiment_1000.csv")

# 형태소 분석기 준비
okt = Okt()

# 형태소 분석 함수 정의
def tokenize(text):
    tokens = okt.morphs(text, stem=True)  # 어간 추출 포함
    return ' '.join(tokens)

# 형태소 분석 적용
df["text_tokenized"] = df["text"].apply(tokenize)

# 훈련/테스트 데이터 나누기
X = df["text_tokenized"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 벡터화
vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 모델 학습
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# 정확도 평가
y_pred = model.predict(X_test_vector)
accuracy = accuracy_score(y_test, y_pred)
print(f"형태소 분석 적용 후 테스트 정확도: {accuracy * 100:.2f}%")

# 예측 틀린 문장들 찾기
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
