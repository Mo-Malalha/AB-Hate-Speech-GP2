import streamlit as st
import requests
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import matplotlib.pyplot as plt

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First Model - Sexism Classification
def query_sexism(payload):
    API_URL = "https://api-inference.huggingface.co/models/rungalileo/tf_roberta_sexist_classifier"
    API_TOKEN = "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"  # Replace this with your actual API token
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Second Model - Racist and Offensive Classification
@st.cache_resource
def load_model():
    model = joblib.load('Bert2Labels.joblib')
    return model

def analyze_text(user_input):
    results = {}

    # First model - Sexism classification
    output_sexism = query_sexism({"inputs": user_input})
    if isinstance(output_sexism, list) and len(output_sexism) > 0:
        first_prediction = output_sexism[0]
        sexism_label = first_prediction[0].get('label')
        sexism_score = first_prediction[0].get('score')
        if sexism_label == '1' and sexism_score >= 0.65:
            results["Sexism Classification"] = "Sexism"
            results["Racist or Offensive Classification"] = "None"
            results["Class"] = 1
            results["Probabilities"] = sexism_score
        else:
            results["Sexism Classification"] = "None"
            # Second model - Racist and Offensive classification
            model = load_model()
            test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = test_sample['input_ids'].to(device)
            attention_mask = test_sample['attention_mask'].to(device)
            with torch.no_grad():
                output_racist_offensive = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output_racist_offensive.logits.detach().cpu().numpy()
            probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
            labels = ['Racism', 'Offensive']
            classification = labels[np.argmax(probabilities)]
            results["Racist or Offensive Classification"] = classification
            results["Class"] = 2 if classification == "Racism" else 3
            results["Probabilities"] = probabilities[np.argmax(probabilities)]
    return results

# Function to display and handle the comment selection
def display_random_comments(df):
    if 'tweet' not in df.columns:
        st.error("The 'tweet' column is not found in the dataset. Please check the CSV file.")
        return None

    if 'random_comments' not in st.session_state:
        st.session_state.random_comments = df['tweet'].sample(5).tolist()

    with st.expander("Random Comments"):
        selected_comment = st.radio("Select a comment to analyze:", st.session_state.random_comments)
        if selected_comment:
            st.session_state.selected_comment = selected_comment
    
    return selected_comment

# Define app function
def app():
    # Streamlit interface
    st.title("Text Classification App")

    # Option to upload a CSV or TXT file
    uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV file:")
            st.write(df)

            if st.button("Analyze CSV"):
                results = []
                for text in df['tweet']:
                    result = analyze_text(text)
                    results.append(result)
                df_results = pd.DataFrame(results)
                st.write("Analysis Results:")
                st.write(df_results)
                st.download_button(label="Download Results as CSV", data=df_results.to_csv(index=False),
                                   file_name="analysis_results.csv", mime="text/csv")

        elif uploaded_file.name.endswith('.txt'):
            text_lines = uploaded_file.read().decode("utf-8").splitlines()
            st.write("Uploaded TXT file:")
            st.write(text_lines)

            if st.button("Analyze TXT"):
                results = []
                for text in text_lines:
                    result = analyze_text(text)
                    results.append(result)
                df_results = pd.DataFrame(results)
                st.write("Analysis Results:")
                st.write(df_results)
                st.download_button(label="Download Results as CSV", data=df_results.to_csv(index=False),
                                   file_name="analysis_results.csv", mime="text/csv")

    # Add the random comments feature
    st.write("Select a random comment from the dataset for analysis:")
    try:
        df = pd.read_csv('C:/Users/win 10/Documents/psut/GP/gp2/NewBert/Datasets/ModifiedRO.csv', encoding='latin1')
        selected_comment = display_random_comments(df)
    except FileNotFoundError:
        st.error("Dataset file not found")

    st.write("Or enter text below for analysis:")
    user_input = st.text_area("Enter Text to Analyze", value=st.session_state.get('selected_comment', ''))
    button = st.button("Analyze")

    # Processing user input and model prediction
    if user_input and button:
        results = analyze_text(user_input)
        if "Racist or Offensive Classification" in results and results["Racist or Offensive Classification"] != "None":
            probabilities = results["Probabilities"]
            labels = ['Racism', 'Offensive']
            plt.figure(figsize=(6, 6))
            plt.bar(labels, [0 if label != results["Racist or Offensive Classification"] else probabilities for label in labels], color=['blue', 'red'])
            plt.xlabel('Labels')
            plt.ylabel('Probability')
            plt.title('Probability of the Classified Label')
            st.pyplot(plt)
        for key, value in results.items():
            if key != "Probabilities":
                st.write(f"{key}: {value}")

if __name__ == "__main__":
    app()