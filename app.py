import torch
import streamlit as st
from transformers import BertTokenizer
import joblib
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading model and tokenizer, ensuring model is on the correct device
@st.cache(allow_output_mutation=True)
def get_model():
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = joblib.load('joblib_bert_model.joblib').to(device)  # Ensure model is moved to GPU
    return tokenizer, model

tokenizer, model = get_model()

# Streamlit interface
user_input = st.text_area("Enter Text to Analyze")
button = st.button("Analyze")

# Label dictionary
d = {
    1: 'Sexist',
    0: 'None',
    2: 'Racist',
    3: 'Offensive'
}

# Processing user input and model prediction
if user_input and button:
    # Encode the input on the correct device
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = test_sample['input_ids'].to(device)
    attention_mask = test_sample['attention_mask'].to(device)
    
    # Perform the prediction
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Move logits to CPU for numpy compatibility if necessary
    logits = output.logits.detach().cpu().numpy()
    st.write("Logits: ", logits)
    y_pred = np.argmax(logits, axis=1)
    st.write("Prediction: ", d[y_pred[0]])
