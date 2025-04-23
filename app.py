import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import yake
import re

# Load the dataset
df = pd.read_csv('preprocessed_leetcode.csv')

# Load precomputed embeddings
problem_embeddings = np.load('problem_embeddings.npy')

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize YAKE keyword extractor
kw_extractor = yake.KeywordExtractor(top=10, n=3)  # Extract top 10 keywords, up to 3-grams

# Recommendation function
def recommend_problems(jd_text, difficulty=None, top_n=10):
    """
    Recommends coding problems based on the job description and optional difficulty.
    Includes per-problem explanations based on JD keywords.
    
    Args:
        jd_text (str): The job description text.
        difficulty (str, optional): Desired difficulty ('easy', 'medium', 'hard').
        top_n (int, optional): Number of problems to recommend.
    
    Returns:
        DataFrame: Recommended problems with titles, difficulties, topics, similarity scores, and explanations.
        list: Keywords extracted from the JD.
    """
    # Filter by difficulty if specified
    if difficulty:
        filtered_df = df[df['difficulty'].str.lower() == difficulty.lower()]
    else:
        filtered_df = df
    
    if filtered_df.empty:
        st.write(f"No problems found for difficulty: {difficulty}")
        return pd.DataFrame(), []
    
    # Encode the job description into an embedding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jd_embedding = model.encode(jd_text, convert_to_tensor=True, device=device)
    
    # Get embeddings for filtered problems
    filtered_indices = filtered_df.index
    filtered_embeddings = problem_embeddings[filtered_indices]
    
    # Convert filtered_embeddings to a tensor on the same device
    filtered_embeddings_tensor = torch.tensor(filtered_embeddings, device=jd_embedding.device)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(jd_embedding, filtered_embeddings_tensor)[0]
    
    # Get the top N indices
    top_indices_tensor = torch.topk(similarities, k=min(top_n, len(similarities))).indices
    top_indices = top_indices_tensor.cpu().numpy()
    
    # Get recommended problems
    recommended = filtered_df.iloc[top_indices].copy()
    
    # Add similarity scores
    similarity_scores = similarities[top_indices_tensor].cpu().numpy()
    recommended['similarity'] = similarity_scores
    
    # Extract keywords from JD
    keywords = [kw for kw, score in kw_extractor.extract_keywords(jd_text)]
    
    # Generate per-problem explanations based on keyword matching in related_topics
    def generate_explanation(topics, keywords):
        topic_list = [t.strip().lower() for t in topics.split(',')]
        matching_keywords = [kw for kw in keywords if kw.lower() in topic_list]
        if matching_keywords:
            return f"Recommended because it relates to: {', '.join(matching_keywords)}"
    
    # Add explanations to the dataframe
    recommended['explanation'] = recommended['related_topics'].apply(lambda topics: generate_explanation(topics, keywords))
    
    return recommended[['title', 'difficulty', 'related_topics', 'similarity', 'explanation']], keywords

# Function to generate a detailed LLM-based explanation
def generate_detailed_explanation(jd_text, keywords, recommended):
    """
    Generates a detailed explanation using specific JD details and problem difficulty distribution.
    
    Args:
        jd_text (str): The job description text.
        keywords (list): Keywords extracted from the JD.
        recommended (DataFrame): The recommended problems.
    
    Returns:
        str: A detailed explanation paragraph.
    """
    # Extract JD details
    company_name = re.search(r'at (\w+)', jd_text, re.IGNORECASE)
    role_type = re.search(r'(engineer|developer|data scientist|analyst)', jd_text, re.IGNORECASE)
    industry = re.search(r'(fintech|healthcare|tech|finance|e-commerce)', jd_text, re.IGNORECASE)
    
    company_name = company_name.group(1) if company_name else "the company"
    role_type = role_type.group(1) if role_type else "the role"
    industry = industry.group(1) if industry else "the industry"
    
    # Calculate difficulty distribution
    difficulty_counts = recommended['difficulty'].value_counts().to_dict()
    difficulty_str = ', '.join([f"{count} {diff}" for diff, count in difficulty_counts.items()])
    
    # Aggregate topics from recommended problems
    all_topics = []
    for topics in recommended['related_topics']:
        topic_list = [t.strip().lower() for t in topics.split(',')]
        all_topics.extend(topic_list)
    unique_topics = sorted(set(all_topics))
    
    # Generate explanation
    keywords_str = ', '.join(keywords).lower()
    topics_str = ', '.join(unique_topics)
    
    explanation = (
        f"The recommended LeetCode problems are tailored for the {role_type} role at {company_name} in the {industry} sector, "
        f"aligning with the job description’s focus on {keywords_str}. These problems cover key topics like {topics_str[:5] + (' and more' if len(unique_topics) > 5 else '')}, "
        f"and include {difficulty_str} challenges to stretch your skills. This selection ensures you build both foundational and advanced coding abilities relevant to the position."
    )
    
    return explanation

# Streamlit interface
st.title("Interview Prep Tool")
st.write("Enter a job description to get coding problem recommendations for your Technical Interview.")

# Input job description
jd_text = st.text_area("Job Description", height=200)

# Difficulty filter
difficulty = st.selectbox("Select Difficulty", ["All", "Easy", "Medium", "Hard"])

# Number of recommendations
top_n = st.slider("Number of Recommendations", 1, 20, 10)

if st.button("Get Recommendations"):
    if jd_text:
        difficulty = None if difficulty == "All" else difficulty
        recommended, keywords = recommend_problems(jd_text, difficulty, top_n)
        if not recommended.empty:
            st.write("### Recommended Problems")
            st.dataframe(recommended)
            
            # Add detailed LLM-generated explanation section
            st.write("### Why These Problems Were Recommended")
            detailed_explanation = generate_detailed_explanation(jd_text, keywords, recommended)
            st.write(detailed_explanation)
        else:
            st.write("No problems found for the selected difficulty.")
    else:
        st.write("Please enter a job description.")