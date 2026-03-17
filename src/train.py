import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    print("🚀 Loading corrected & cleaned data...")
    # Load the file created by your updated preprocess.py
    df = pd.read_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\processed\\cleaned_tickets.csv')
    
    # Remove any rows where cleaning might have failed
    df = df.dropna(subset=['cleaned_text'])

    # 1. Feature Engineering (Vectorization)
    print("📊 Translating text to numbers (Vectorizing)...")
    # ngram_range=(1,2) means it looks at single words AND pairs of words
    # min_df=5 ignores rare typos that appear in fewer than 5 tickets
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=5)
    
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['category']

    # 2. Train/Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Training
    print("🧠 Training the Brain (LinearSVC)...")
    model = LinearSVC(C=1.0, class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. Evaluation
    print("\n✅ --- Model Performance Report ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.2%}")
    print("\nDetailed breakdown per category:")
    print(classification_report(y_test, y_pred))

    # 5. Saving the Model and Translator
    joblib.dump(model, 'C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\models\\ticket_classifier.pkl')
    joblib.dump(tfidf, 'C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\models\\vectorizer.pkl')
    print("\n💾 Model saved! Ready for Phase 5.")

if __name__ == "__main__":
    train_model()