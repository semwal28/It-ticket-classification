import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\data\\raw\\customer_support_tickets_200k.csv')

# 1. Category Distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='category', data=df, order=df['category'].value_counts().index, palette='viridis')
plt.title('Distribution of Ticket Categories')
plt.xlabel('Number of Tickets')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\notebooks\\category_distribution.png')
print("Saved category distribution plot.")

# 2. Priority vs Category
plt.figure(figsize=(12, 6))
sns.countplot(x='category', hue='priority', data=df, palette='magma')
plt.xticks(rotation=45)
plt.title('Ticket Priority by Category')
plt.tight_layout()
plt.savefig('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\notebooks\\priority_by_category.png')
print("Saved priority/category plot.")

# 3. Analyze Issue Description Length
# This helps us decide on 'max_length' if we use Deep Learning later
df['desc_len'] = df['issue_description'].str.split().str.len()
plt.figure(figsize=(10, 5))
sns.histplot(df['desc_len'], bins=50, kde=True, color='blue')
plt.title('Distribution of Word Counts in Ticket Descriptions')
plt.xlabel('Number of Words')
plt.savefig('C:\\Users\\semwa\\OneDrive\\Desktop\\IT-TICKET CLASSIFICATION\\notebooks\\text_length_dist.png')
print("Saved text length distribution plot.")

# 4. Summary Statistics
print("\n--- Data Summary ---")
print(f"Total Tickets: {len(df)}")
print(f"Unique Categories: {df['category'].nunique()}")
print(f"Average Words per Ticket: {df['desc_len'].mean():.2f}")