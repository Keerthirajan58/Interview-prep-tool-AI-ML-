import pandas as pd

# Load the dataset
df = pd.read_csv('leetcode_dataset.csv')

# Select relevant columns
df = df[['title', 'description', 'difficulty', 'related_topics']]

# Handle missing values (drop rows with any missing data in these columns)
df = df.dropna()

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_leetcode.csv', index=False)

print("Preprocessing complete. Preprocessed data saved to 'preprocessed_leetcode.csv'.")