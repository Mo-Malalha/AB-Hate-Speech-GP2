import streamlit as st
import requests
import torch
import numpy as np
from transformers import AutoTokenizer
import joblib
import matplotlib.pyplot as plt

# Define tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First Model - Sexism Classification
def query_sexism(payload):
    API_URL = "https://api-inference.huggingface.co/models/rungalileo/tf_roberta_sexist_classifier"
    API_TOKEN = "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"  # Make sure to replace this with your actual API token
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Second Model - Racist and Offensive Classification
@st.cache_data()
def load_model():
    model = joblib.load('Bert2Labels.joblib')
    return model

# Define app function
def app():
    # Streamlit interface
    st.title("Text Classification App")
    user_input = st.text_area("Enter Text to Analyze")
    button = st.button("Analyze")

    # Processing user input and model prediction
    if user_input and button:
        # First model - Sexism classification
        output_sexism = query_sexism({"inputs": user_input})
        if isinstance(output_sexism, list) and len(output_sexism) > 0:
            first_prediction = output_sexism[0]
            sexism_label = first_prediction[0].get('label')
            sexism_score = first_prediction[0].get('score')
            
            
            if sexism_label == '1' and sexism_score >= 0.65:
                st.write("This text is classified as Sexist")
            else:
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
                plt.figure(figsize=(6, 6))
                plt.bar(labels, probabilities, color=['blue', 'red'])
                plt.xlabel('Labels')
                plt.ylabel('Probability')
                plt.title('Probabilities of Racist and Offensive Labels')
                st.pyplot(plt)
                correct_label = labels[np.argmax(probabilities)]
                st.write(f"The text is classified as {correct_label}.")

# If this file is executed directly, run the app
if __name__ == "__main__":
    app()
