import pandas as pd

# Replace 'leetcode_problems.csv' with your actual filename if different
df = pd.read_csv('leetcode_dataset.csv')

# Print all column names
print("Column names in the dataset:", df.columns.tolist())