Intelligent Coding Problem Recommendation System

Overview

This tool recommends LeetCode problems based on job descriptions, helping users prepare for technical interviews using ML techniques and Streamlit.

Features

Input a job description to get tailored problem recommendations.
Filter by difficulty (Easy, Medium, Hard, or All).
Detailed explanation of why problems were recommended.

Setup Instructions

Clone the Repository:
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


Install Dependencies:
pip install -r requirements.txt


Run the App:
streamlit run app.py


Visit http://localhost:8501 in your browser.



Files

app.py: Main Streamlit app.
preprocessed_leetcode.csv: Dataset of LeetCode problems.
problem_embeddings.npy: Precomputed embeddings.
requirements.txt: Python dependencies.

Usage

Enter a job description.
Select difficulty and number of recommendations.
Click "Get Recommendations."

