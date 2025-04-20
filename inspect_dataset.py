import pandas as pd

# Load the dataset
df = pd.read_csv('preprocessed_leetcode.csv')

# Print basic info
print("Dataset Info:")
print(df.info())
print("\nUnique difficulties:", df['difficulty'].unique())
print("\nSample of related_topics:")
print(df['related_topics'].head(10))