import pandas as pd

def fix_dataset_balanced():
    print("🚀 Loading original dataset...")
    # Load raw data
    df = pd.read_csv(r'C:\Users\semwa\OneDrive\Desktop\IT-TICKET CLASSIFICATION\data\raw\customer_support_tickets_200k.csv')
    
    # Mapping logic to fix wrong labels in the original file
    keywords = {
        'Payment Problem': ['payment', 'billing', 'charge', 'transaction', 'invoice', 'refund', 'money'],
        'Login Issue': ['login', 'password', 'credentials', 'account', 'access', 'authentication'],
        'Performance Issue': ['slow', 'performance', 'latency', 'lag', 'loading', 'speed'],
        'Bug Report': ['bug', 'crash', 'error', 'crashes', 'fault'],
        'Subscription': ['subscription', 'cancel', 'membership', 'renewal'],
        'Data Sync': ['sync', 'synchronization', 'devices', 'pairing']
    }

    def map_category(text):
        text = str(text).lower()
        for category, terms in keywords.items():
            if any(term in text for term in terms):
                return category
        return None

    print("🛠️ Fixing labels based on ticket text...")
    df['category'] = df['issue_description'].apply(map_category)
    df = df.dropna(subset=['category'])

    # Balancing to 5,000 samples each to prevent the model from becoming biased
    print("⚖️ Balancing categories to 5,000 samples each...")
    balanced_df = df.groupby('category').apply(
        lambda x: x.sample(n=min(len(x), 5000), random_state=42)
    ).reset_index(drop=True)
    
    balanced_df.to_csv(r'C:\Users\semwa\OneDrive\Desktop\IT-TICKET CLASSIFICATION\data\raw\corrected_tickets.csv', index=False)
    print(f"✅ Success! Saved {len(balanced_df)} balanced rows to data/raw/corrected_tickets.csv")

if __name__ == "__main__":
    fix_dataset_balanced()