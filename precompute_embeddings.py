import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the dataset
df = pd.read_csv('preprocessed_leetcode.csv')

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for the problem descriptions
embeddings = model.encode(df['description'].tolist(), convert_to_tensor=False)

# Save the embeddings to a NumPy file
np.save('problem_embeddings.npy', embeddings)

# Optionally, save the DataFrame without embeddings
df.to_csv('preprocessed_leetcode.csv', index=False)