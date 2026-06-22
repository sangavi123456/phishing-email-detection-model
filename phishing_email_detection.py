from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [
    "Congratulations! You won a free iPhone",
    "Click here to claim your prize",
    "Your bank account is suspended",
    "Meeting at 10 AM tomorrow",
    "Project submission deadline is today",
    "Let's have lunch together"
]

labels = [
    "Phishing",
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Safe"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

new_email = ["Click here to win money now"]
new_email_vector = vectorizer.transform(new_email)

result = model.predict(new_email_vector)

print("Prediction:", result[0])