import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Setup NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Lowercase and remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # 2. Tokenize and remove stopwords + Lemmatize
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

def run_preprocessing():
    print("Loading raw data...")
    # Load only necessary columns to save memory
    df = pd.read_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\raw\\customer_support_tickets_200k.csv', usecols=['issue_description', 'category'])
    
    # Use tqdm to see progress on 200k rows
    print("Starting text cleaning (this may take 2-5 minutes)...")
    tqdm.pandas()
    df['cleaned_text'] = df['issue_description'].progress_apply(clean_text)
    
    # Save processed data
    output_path = 'C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\processed\\cleaned_tickets.csv'
    df.to_csv(output_path, index=False)
    print(f"Success! Cleaned data saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()