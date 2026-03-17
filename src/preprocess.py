import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLP data
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # 1. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # 2. Convert to lowercase
    text = text.lower()
    # 3. Tokenize (split into words)
    words = text.split()
    # 4. Remove Stopwords and Lemmatize (e.g., "crashing" -> "crash")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# Load the dataset
print("Loading data...")
df = pd.read_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\raw\\corrected_tickets.csv')
# Apply cleaning to the first 10,000 rows (to start fast)
# You can increase this to the full 200k later
print("Cleaning text (this may take a minute)...")
df_subset = df[['issue_description', 'category']].head(10000).copy()
df_subset['cleaned_description'] = df_subset['issue_description'].apply(clean_text)

# Save the cleaned version for Phase 4
df_subset.to_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\processed\\cleaned_tickets.csv', index=False)
print("Phase 3 Complete: Cleaned data saved to data/cleaned_tickets.csv")