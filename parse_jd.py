from transformers import pipeline

def extract_skills_from_jd(jd_text):
    """
    Extracts key skills from a job description using keyword matching.
    
    Args:
        jd_text (str): The job description text.
    
    Returns:
        list: A list of extracted skills.
    """
    # Predefined list of common technical skills
    skills_list = ['Python', 'SQL', 'machine learning', 'data structures', 'algorithms', 
                   'cloud computing', 'AWS']
    
    # Convert job description to lowercase for case-insensitive matching
    jd_text_lower = jd_text.lower()
    
    # Extract skills present in the job description
    extracted_skills = [skill for skill in skills_list if skill.lower() in jd_text_lower]
    
    return extracted_skills

if __name__ == "__main__":
    sample_jd = """
    We are looking for a software engineer with experience in Python, SQL, and machine learning.
    The ideal candidate should be familiar with data structures, algorithms, and cloud computing.
    Experience with AWS or Azure is a plus.
    """
    skills = extract_skills_from_jd(sample_jd)
    print("Extracted Skills:", skills)