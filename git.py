import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import inflect
import logging
import re
import nltk  # Import nltk here
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords if not found
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Set up logging for error tracking
logging.basicConfig(level=logging.ERROR)

# Initialize the inflect engine and SBERT model
p = inflect.engine()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient model with good performance for similarity

# Function to preprocess text
def preprocess_text(text):
    # Normalize numeric representations
    text = ' '.join([p.number_to_words(word) if word.isdigit() else word for word in text.split()])
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase and remove non-alphabet characters
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    # Remove stopwords and apply lemmatization
    processed_text = " ".join(
        [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    )
    return processed_text

# Function to categorize semantic deviation based on similarity score
def categorize_semantic_deviation(similarity_score):
    if similarity_score >= 0.70:
        return "Matched"
    elif similarity_score >= 0.50:
        return "Need Review"
    else:
        return "Not Matched"

# Function to create a downloadable Excel file
def create_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Similarity Results')
    output.seek(0)
    return output

# Batch processing for large datasets
def batch_process(df, batch_size=100):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

# Function to calculate similarity using SBERT
def calculate_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    similarity_percentage = similarity_score * 100  # Convert to percentage
    return similarity_score, similarity_percentage

# Main function to define app layout
def main():
    st.title("Semantic Text Similarity (STS) Test with SBERT")

    # Use Tabs for smoother navigation
    tab1, tab2, tab3 = st.tabs(["Home", "Upload Data", "Manual Input"])

    with tab1:
        st.markdown("""
        <h2 style='font-size:28px;'>Semantic Similarity (Using SBERT)</h2>
        <p style='font-size:16px;'>Measures how similar two sentences are in meaning using SBERT.</p>
        <ul style='font-size:16px;'>
            <li><strong>Matched:</strong> 70% to 100%</li>
            <li><strong>Need Review:</strong> 50% to 69.9%</li>
            <li><strong>Not Matched:</strong>0% to 49.99% </li>
        </ul>
        """, unsafe_allow_html=True)

    with tab2:
        uploaded_file = st.file_uploader("Upload an Excel file with two columns", type=["xlsx"])

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                if df.shape[1] < 2:
                    st.error("The uploaded file must contain at least two columns.")
                    return

                st.write("Uploaded Data:")
                st.dataframe(df)

                sentence1_col = df.columns[0]
                sentence2_col = df.columns[1]

                if st.button("Calculate Similarity", key="upload_button"):
                    results = []
                    progress_bar = st.progress(0)

                    # Process rows in batches
                    total_batches = len(df) / 100
                    for i, batch in enumerate(batch_process(df)):
                        for _, row in batch.iterrows():
                            sentence1 = preprocess_text(row[sentence1_col])
                            sentence2 = preprocess_text(row[sentence2_col])

                            if pd.isna(sentence1) or pd.isna(sentence2):
                                similarity_percentage = 0
                            else:
                                _, similarity_percentage = calculate_similarity(sentence1, sentence2)

                            results.append({
                                "Sentence 1": row[sentence1_col],
                                "Sentence 2": row[sentence2_col],
                                "Similarity Percentage": round(similarity_percentage, 2),
                                "Semantic Deviation": categorize_semantic_deviation(similarity_percentage / 100)
                            })

                        progress_bar.progress(min((i + 1) / total_batches, 1.0))

                    results_df = pd.DataFrame(results)
                    st.write("Similarity Results:")
                    st.dataframe(results_df)

                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Results as Excel",
                        data=excel_data,
                        file_name="similarity_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                logging.error(f"Error processing file: {e}")
                st.error(f"Error processing file: {e}")

    with tab3:
        sentence1 = st.text_area("Enter the first sentence:")
        sentence2 = st.text_area("Enter the second sentence:")

        if st.button("Calculate Similarity", key="manual_button"):
            if sentence1 and sentence2:
                try:
                    sentence1_processed = preprocess_text(sentence1)
                    sentence2_processed = preprocess_text(sentence2)

                    similarity_score, similarity_percentage = calculate_similarity(sentence1_processed, sentence2_processed)

                    st.write(f"**Similarity Score:** {similarity_score:.4f}")
                    st.write(f"**Similarity Percentage:** {similarity_percentage:.2f}%")
                    st.write(f"**Semantic Deviation:** {categorize_semantic_deviation(similarity_score)}")

                    result_data = [{
                        "Sentence 1": sentence1,
                        "Sentence 2": sentence2,
                        "Similarity Score": round(similarity_score, 4),
                        "Similarity Percentage": round(similarity_percentage, 2),
                        "Semantic Deviation": categorize_semantic_deviation(similarity_score)
                    }]
                    results_df = pd.DataFrame(result_data)

                    excel_data = create_download_link(results_df)
                    st.download_button(
                        label="Download Result as Excel",
                        data=excel_data,
                        file_name="manual_similarity_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    logging.error(f"Error calculating similarity: {e}")
                    st.error(f"Error calculating similarity: {e}")
            else:
                st.error("Please enter both sentences.")

    st.markdown("---")
    st.write("### About this App")
    st.write("This app uses SBERT to calculate the semantic similarity between two sentences.")

if __name__ == "__main__":
    main()
