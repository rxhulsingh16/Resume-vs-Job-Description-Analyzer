import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Resume vs Job Description Analyzer")

# Live Resume Matching
st.sidebar.header("Live Resume Matching")
user_resume = st.sidebar.text_area("Enter Resume")
user_job = st.sidebar.text_area("Enter Job Description")

# File Upload
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data", "EDA", "Processing", "Model"
    ])

    # Data
    with tab1:
        st.subheader("Dataset Preview", anchor=None)
        st.write(df.head())
        df.columns = df.columns.str.lower()
        df.rename(columns={'resume_str': 'resume'}, inplace=True)
        df = df[['resume', 'job_desc', 'label']]
        st.write("Columns:", df.columns.tolist())

    required_cols = ['resume', 'job_desc', 'label']

    if all(col in df.columns for col in required_cols):

        # EDA
        with tab2:
            st.subheader("EDA", anchor=None)
            st.write("Shape:", df.shape)
            st.write("Label Distribution:")
            st.write(df['label'].value_counts())
            plt.figure()
            sns.countplot(x='label', data=df)
            st.pyplot(plt)

        # Processing
        with tab3:
            st.subheader("Data Processing", anchor=None)
            # clean_text
            def clean_text(text):
                text = re.sub(r'[^a-zA-Z ]', '', str(text))
                return text.lower()
            df['resume'] = df['resume'].apply(clean_text)
            df['job_desc'] = df['job_desc'].apply(clean_text)
            df['text'] = df['resume'] + " " + df['job_desc']
            st.success("Data cleaned successfully!")
            # TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1,2)
            )
            X = vectorizer.fit_transform(df['text'])
            y = df['label'].astype(int)
            st.success("TF-IDF Applied!")
            # train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Model
        with tab4:
            st.subheader("Model Training & Evaluation", anchor=None)
            # Model Selection
            model_name = st.selectbox(
                "Choose Model", 
                ["KNN", "SVM", "MLP", "Random Forest"]
            )
            # Model Init
            if model_name == "KNN":
                model = KNeighborsClassifier()
            elif model_name == "SVM":
                model = SVC(kernel='linear', class_weight='balanced')
            elif model_name == "MLP":
                model = MLPClassifier(max_iter=300)
            else:
                model = RandomForestClassifier()
            # Model Fit
            model.fit(X_train, y_train)
            st.success(f"{model_name} Model Trained Successfully!")
            # Prediction
            y_pred = model.predict(X_test)
            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", acc)
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            st.pyplot(plt)
            # Classification Report
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            # K-Fold
            scores = cross_val_score(model, X, y, cv=5)
            st.write("K-Fold Accuracy:", scores.mean())
            st.success("Model Evaluation Completed!")
            st.subheader("Model Comparison (Sample)")
            models = ["KNN", "SVM", "MLP", "Random Forest"]
            accuracies = [0.5, 0.6, 0.55, 0.7]
            st.bar_chart(accuracies)
        # Live Match Score
        if user_resume and user_job:
            st.subheader("Resume Matching Score", anchor=None)
            input_text = [clean_text(user_resume), clean_text(user_job)]
            input_vector = vectorizer.transform(input_text)
            similarity = cosine_similarity(input_vector[0], input_vector[1])
            match_score = similarity[0][0] * 100
            st.write(f"Match Score: {match_score:.2f}%")

    else:
        st.error("Dataset must contain columns: resume, job_desc, label")