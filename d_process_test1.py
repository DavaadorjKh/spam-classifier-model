import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned_text'] = df['v2' if 'v2' in df.columns else 'Message'].apply(clean_text)

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)  # Limit to top 100 features
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
feature_names = vectorizer.get_feature_names_out()

# Create a more readable display
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# 1. Show only non-zero entries for first 5 documents
print("Non-zero TF-IDF values in first 5 documents:")
for i in range(5):
    non_zero = tfidf_df.iloc[i][tfidf_df.iloc[i] > 0]
    if len(non_zero) > 0:
        print(f"\nDocument {i}:")
        print(non_zero)
    else:
        print(f"\nDocument {i}: All zeros")

# 2. Show top words for each document
print("\nTop words per document:")
for i in range(5):
    top_words = tfidf_df.iloc[i].sort_values(ascending=False)[:5]  # Top 5 words
    print(f"\nDocument {i}:")
    print(top_words[top_words > 0])  # Only show non-zero

# 3. Show document where 'free' appears most strongly
if 'free' in feature_names:
    free_scores = tfidf_df['free']
    max_idx = free_scores.idxmax()
    print(f"\nDocument with highest 'free' score (index {max_idx}, score {free_scores[max_idx]:.4f}):")
    print(df['cleaned_text'].iloc[max_idx])