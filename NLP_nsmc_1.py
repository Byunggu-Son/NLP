import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 파일 불러오기
train_df = pd.read_csv("ratings_train.txt", sep='\t', encoding='utf-8')
test_df = pd.read_csv("ratings_test.txt", sep='\t', encoding='utf-8')

# 결측치 제거
train_df = train_df.dropna()
test_df = test_df.dropna()

# 학습/테스트 데이터 준비
X_train = train_df["document"]
y_train = train_df["label"]
X_test = test_df["document"]
y_test = test_df["label"]

# 벡터화 (형태소 분석 없이)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 모델 학습(나이브 베이즈)
model = MultinomialNB()
model.fit(X_train_vec, y_train)



# 정확도 평가
y_pred = model.predict(X_test_vec)

print("나이브 베이즈 정확도:", accuracy_score(y_test, y_pred))


# 감성성 예측 함수
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "긍정" if prediction == 1 else "부정"

# 함수로 예제 예측
print(predict_sentiment("이 영화 ㅈㄴㄴ 최고네요"))
print(predict_sentiment("굳이 이딴거 왜 만든거?"))



# 예측 틀린 문장들 찾기
predicted = model.predict(X_test_vec)
print("\n 예측이 틀린 문장들:\n")

# 틀린 문장들이 너무 많아 6개 정도만 추출
count = 0

for i in range(len(y_test)):
    actual = y_test.iloc[i]
    prediction = predicted[i]
    
    if actual != prediction:
        original_text = test_df.iloc[y_test.index[i]]["document"]
        real_label = "긍정" if actual == 1 else "부정"
        predicted_label = "긍정" if prediction == 1 else "부정"
        
        print(f"- 문장: {original_text}")
        print(f"  실제: {real_label} / 예측: {predicted_label}\n")
        
        count += 1
        if count == 6:
            break
