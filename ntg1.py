import tkinter as tk
from tkinter import messagebox
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Load pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract and preprocess CV sections
def extract_sections(cv):
    sections = [cv["skills"], cv["work_experience"], cv["education"], cv["certifications"], cv["summary"]]
    return sections

# Function to compute section similarity
def compute_section_similarity(cv_embedding, job_embedding):
    # Compute cosine similarity for each section of the CV with the job description
    section_scores = util.pytorch_cos_sim(cv_embedding, job_embedding).squeeze(1).cpu().numpy()
    return section_scores

# Function to process the CV and display rankings
def process_cv():
    # Retrieve input from user
    cv_data = {
        "skills": skills_entry.get(),
        "work_experience": work_experience_entry.get(),
        "education": education_entry.get(),
        "certifications": certifications_entry.get(),
        "summary": summary_entry.get()
    }

    job_description = job_description_entry.get()

    if not job_description or any(not value for value in cv_data.values()):
        messagebox.showwarning("Input Error", "Please fill out all fields.")
        return

    # Preprocess Job Description and CV sections
    job_description_embedding = model.encode(job_description, convert_to_tensor=True)

    sections = extract_sections(cv_data)
    cv_embedding = model.encode(sections, convert_to_tensor=True)

    # Compute section similarity
    scores = compute_section_similarity(cv_embedding, job_description_embedding)
    section_labels = ["Skills", "Work Experience", "Education", "Certifications", "Summary"]

    # Display section relevancy
    result_text.delete(1.0, tk.END)  # Clear previous result
    result_text.insert(tk.END, f"Job Description: {job_description}\n")
    result_text.insert(tk.END, "\nRelevancy Ranking:\n")

    ranked_sections = sorted(zip(section_labels, scores), key=lambda x: x[1], reverse=True)
    for section, score in ranked_sections:
        result_text.insert(tk.END, f"{section}: {score:.4f}\n")

    # Compute and display overall score
    overall_score = np.mean(scores)
    result_text.insert(tk.END, f"\nOverall Relevancy Score: {overall_score:.4f}")

# Tkinter GUI setup
root = tk.Tk()
root.title("CV Relevancy Checker")

# Labels and Entry Widgets
tk.Label(root, text="Job Description").grid(row=0, column=0, padx=10, pady=5)
job_description_entry = tk.Entry(root, width=50)
job_description_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Skills").grid(row=1, column=0, padx=10, pady=5)
skills_entry = tk.Entry(root, width=50)
skills_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Work Experience").grid(row=2, column=0, padx=10, pady=5)
work_experience_entry = tk.Entry(root, width=50)
work_experience_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Education").grid(row=3, column=0, padx=10, pady=5)
education_entry = tk.Entry(root, width=50)
education_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Certifications").grid(row=4, column=0, padx=10, pady=5)
certifications_entry = tk.Entry(root, width=50)
certifications_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Summary").grid(row=5, column=0, padx=10, pady=5)
summary_entry = tk.Entry(root, width=50)
summary_entry.grid(row=5, column=1, padx=10, pady=5)

# Submit Button
submit_button = tk.Button(root, text="Check Relevancy", command=process_cv)
submit_button.grid(row=6, column=0, columnspan=2, pady=10)

# Textbox to display results
result_text = tk.Text(root, height=10, width=60)
result_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

# Run the GUI
root.mainloop()
