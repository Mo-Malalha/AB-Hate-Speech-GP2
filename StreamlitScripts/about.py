import streamlit as st

def app():
    st.markdown("# About Hate Speech Detection and Category Classification App")
    
    st.markdown("## Purpose")
    st.write(
        "The **Hate Speech Detection and Category Classification App** is designed to help users identify and categorize hate speech in text. "
        "The app uses advanced machine learning models to classify text into the following categories: Racism, Sexism, Offensive, and None."
    )
    
    st.markdown("## Features")
    st.write(
        "- **Upload CSV or TXT Files:** Upload your text data files and get an analysis report.\n"
        "- **Random Comment Analysis:** Select a random comment from the dataset for quick analysis.\n"
        "- **Real-time Text Analysis:** Enter text manually and get instant classification results.\n"
        "- **Download Results:** Download the analysis results as a CSV file for further use."
    )
    
    st.markdown("## Technology Stack")
    st.write(
        "This app leverages the following technologies:\n"
        "- **Python:** For backend logic and data processing.\n"
        "- **Streamlit:** For the interactive web interface.\n"
        "- **Transformers:** For leveraging pre-trained models in text classification.\n"
        "- **Hugging Face API:** For accessing advanced machine learning models."
    )
    
    st.markdown("## Team")
    st.write(
        "The development team consists of:\n"
        "- **Team Members: Zaid Abudllah,Mohammad Al Malalha , Hamzeh Bseiso ."
    )
    
    st.markdown("## Acknowledgements")
    st.write(
        "We would like to thank the open-source community and the developers of the machine learning models and libraries we used. "
        "Special thanks to [Hugging Face](https://huggingface.co/) for providing the pre-trained models and APIs that made this project possible."
    )
    
    st.markdown("## Contact")
    st.write(
        "For more information, feedback, or collaboration, please contact us at:\n"
        "- Email: zaidabdullah13@gmail.com ,hamzeh.bs12@gmail.com ,moh.mal22@gmail.com"
    )
    
if __name__ == "__main__":
    app()
