import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

def run_preprocessing():
    print("🧹 Loading and cleaning data...")
    df = pd.read_csv(r'C:\Users\semwa\OneDrive\Desktop\IT-TICKET CLASSIFICATION\data\raw\corrected_tickets.csv', usecols=['issue_description', 'category'])
    
    tqdm.pandas()
    df['cleaned_text'] = df['issue_description'].progress_apply(clean_text)
    
    df.to_csv(r'C:\Users\semwa\OneDrive\Desktop\IT-TICKET CLASSIFICATION\data\processed\cleaned_tickets.csv', index=False)
    print("✅ Cleaning complete. Saved to data/processed/cleaned_tickets.csv")

if __name__ == "__main__":
    run_preprocessing()