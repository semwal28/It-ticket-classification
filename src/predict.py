import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# 1. Setup - We need the same cleaning tools used in training
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# 2. Load the Saved "Brain"
print("🔄 Loading the AI model and translator...")
model = joblib.load(r'C:\Users\semwa\OneDrive\Desktop\IT-TICKET CLASSIFICATION\models\ticket_classifier.pkl')
tfidf = joblib.load(r'C:\Users\semwa\OneDrive\Desktop\IT-TICKET CLASSIFICATION\models\vectorizer.pkl')

def classify_new_ticket():
    print("\n--- IT Ticket Classifier ---")
    print("Type 'exit' to quit.")
    
    while True:
        # Take input from user
        user_input = input("\nEnter ticket description: ")
        
        if user_input.lower() == 'exit':
            break
            
        # 3. Clean and Transform the input
        cleaned = clean_text(user_input)
        vectorized_text = tfidf.transform([cleaned])
        
        # 4. Make the prediction
        prediction = model.predict(vectorized_text)[0]
        
        print(f"🤖 AI Prediction: {prediction.upper()}")

if __name__ == "__main__":
    classify_new_ticket()
        



