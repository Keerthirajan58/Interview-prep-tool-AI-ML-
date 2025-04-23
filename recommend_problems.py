import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load your dataset (adjust the path as needed)
df = pd.read_csv('preprocessed_leetcode.csv')

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_problems(jd_text, difficulty=None, top_n=10):
    """
    Recommends coding problems based on the job description and optional difficulty.
    
    Args:
        jd_text (str): The job description text.
        difficulty (str, optional): Desired difficulty ('easy', 'medium', 'hard').
        top_n (int, optional): Number of problems to recommend.
    
    Returns:
        DataFrame: Recommended problems with titles, difficulties, topics, and similarity scores.
    """
    # Filter by difficulty if specified
    if difficulty:
        filtered_df = df[df['difficulty'].str.lower() == difficulty.lower()]
    else:
        filtered_df = df
    
    if filtered_df.empty:
        print(f"No problems found for difficulty: {difficulty}")
        return pd.DataFrame()
    
    # Encode the job description
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    
    # Encode problem descriptions
    problem_embeddings = model.encode(filtered_df['description'].tolist(), convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(jd_embedding, problem_embeddings)[0]
    
    # Get the top N indices as a tensor
    top_indices_tensor = torch.topk(similarities, k=min(top_n, len(similarities))).indices
    
    # Convert tensor indices to NumPy for Pandas
    top_indices = top_indices_tensor.cpu().numpy()
    
    # Get recommended problems
    recommended = filtered_df.iloc[top_indices].copy()
    
    # Get similarity scores using the tensor indices
    similarity_scores = similarities[top_indices_tensor].cpu().numpy()
    
    # Assign to the DataFrame
    recommended['similarity'] = similarity_scores
    
    # Return selected columns
    return recommended[['title', 'difficulty', 'related_topics', 'similarity']]

if __name__ == "__main__":
    sample_jd = """
    We are looking for a software engineer with experience in Python, SQL, and machine learning.
    The ideal candidate should be familiar with data structures, algorithms, and cloud computing.
    Experience with AWS or Azure cloud is a plus.
    """
    recommended = recommend_problems(sample_jd, difficulty='Medium')
    print(recommended)