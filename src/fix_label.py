import pandas as pd

def fix_dataset():
    print("Loading original dataset...")
    df = pd.read_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\raw\\customer_support_tickets_200k.csv')
    
    # Rules to match text to the RIGHT category
    keywords = {
        'Payment Problem': ['payment', 'billing', 'charge', 'transaction', 'deducted', 'invoice', 'refund'],
        'Login Issue': ['login', 'access', 'password', 'credentials', 'account', 'authentication', 'two-factor'],
        'Performance Issue': ['slow', 'performance', 'latency', 'lag', 'loading', 'speed'],
        'Bug Report': ['bug', 'crash', 'error', 'crashes', 'report generation'],
        'Subscription Cancellation': ['subscription', 'cancel', 'membership', 'renewal'],
        'Data Sync Issue': ['sync', 'synchronization', 'devices', 'data sync']
    }

    def map_category(text):
        text = str(text).lower()
        for category, terms in keywords.items():
            if any(term in text for term in terms):
                return category
        return "Other"

    print("Fixing labels based on actual ticket text...")
    df['category'] = df['issue_description'].apply(map_category)
    
    # Filter out rows that don't match our IT categories
    df = df[df['category'] != "Other"]
    
    df.to_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\raw\\corrected_tickets.csv', index=False)
    print(f"Success! Fixed {len(df)} tickets. Saved to data/raw/corrected_tickets.csv")

if __name__ == "__main__":
    fix_dataset()