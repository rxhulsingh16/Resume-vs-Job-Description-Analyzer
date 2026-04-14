
# Product Requirements Document (PRD)

## Resume vs Job Description Analyzer

### 1. Purpose
The Resume vs Job Description Analyzer is an interactive web application designed to help users compare resumes with job descriptions using machine learning. It enables both recruiters and job seekers to evaluate how well a resume matches a job description, automate batch analysis, and experiment with different ML models for best results.

### 2. Target Users
- **Recruiters**: To quickly screen and match resumes to job descriptions.
- **Job Seekers**: To optimize their resumes for specific job postings.
- **Data Scientists/ML Enthusiasts**: To experiment with NLP and classification models on resume/job data.

### 3. Main Features
#### a. File Upload & Data Preview
- Upload a CSV file containing resumes, job descriptions, and labels (1 for match, 0 for no match).
- Preview the uploaded data and check column names.

#### b. Exploratory Data Analysis (EDA)
- View dataset shape and label distribution.
- Visualize label counts with bar plots.

#### c. Data Processing
- Clean text data (remove non-alphabetic characters, lowercase).
- Combine resume and job description for feature extraction.
- Apply TF-IDF vectorization (with bigrams, stopword removal, max 5000 features).

#### d. Model Training & Evaluation
- Choose from KNN, SVM, MLP, or Random Forest classifiers.
- Train the selected model on the processed data.
- Evaluate with accuracy, confusion matrix, classification report, and 5-fold cross-validation.

#### e. Live Resume Matching
- Enter a resume and job description in the sidebar for instant similarity scoring (cosine similarity using TF-IDF).

### 4. Workflow
1. **Start the app**: Launch with `streamlit run app.py`.
2. **Upload your CSV**: The file must have columns: `resume`, `job_desc`, `label`.
3. **Explore Data**: Use the Data and EDA tabs to understand your dataset.
4. **Process Data**: Clean and vectorize text in the Processing tab.
5. **Train Model**: Select a model, train, and evaluate in the Model tab.
6. **Live Demo**: Use the sidebar to test resume/job description pairs instantly.

### 5. Example CSV Format
```
resume,job_desc,label
"Experienced data scientist...","Looking for data scientist...",1
"Entry-level developer...","Senior developer role...",0
```

### 6. Technology Stack
- **Frontend/UI**: Streamlit
- **Data Processing**: pandas, numpy, re
- **ML/NLP**: scikit-learn (TF-IDF, classifiers)
- **Visualization**: matplotlib, seaborn

### 7. Installation & Running
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Start the app:
	```bash
	streamlit run app.py
	```

### 8. Usage Tips
- Use the sidebar for live matching without uploading a file.
- For best results, ensure your CSV is clean and properly labeled.
- Try different models to see which performs best on your data.

### 9. Requirements
See `requirements.txt` for all dependencies.

### 10. License
MIT License

---

## Quick Reference

An interactive Streamlit web app for analyzing and matching resumes to job descriptions using machine learning. Supports both live text input and batch CSV upload for model training, evaluation, and similarity scoring.

### Features
- Upload a CSV file with columns: `resume`, `job_desc`, `label`
- Data preview, EDA (exploratory data analysis), and data cleaning
- TF-IDF vectorization and multiple ML models (KNN, SVM, MLP, Random Forest)
- Model training, evaluation (accuracy, confusion matrix, classification report, cross-validation)
- Live resume vs job description similarity scoring

### How to Run
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Start the app:
	```bash
	streamlit run app.py
	```

### Usage
- Use the sidebar to enter a resume and job description for live matching.
- Upload a CSV file to train and evaluate models on your dataset.
- Explore data, processing steps, and model results in the main tabs.

### Example CSV Format
```csv
resume,job_desc,label
"Experienced data scientist...","Looking for data scientist...",1
"Entry-level developer...","Senior developer role...",0
```

### Requirements
See `requirements.txt` for all dependencies.

### License
MIT License
