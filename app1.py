# -- coding: utf-8 --
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from transformers import MarianMTModel, MarianTokenizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import spacy

# Page configuration
st.set_page_config(page_title="Sustainability Analysis", layout="wide")

# Load models with caching
@st.cache_resource
def load_translation_model():
    return MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en"), \
           MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# File upload section
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Read and preprocess data
    data = pd.read_excel(uploaded_file, sheet_name='Form responses 1')
    data = data.drop(data.columns[0], axis=1)
    data.columns = ['voluntary', 'age_range', 'gender', 'country', 'campus', 
                   'department', 'sustain_percept', 'sustain_reason','sustain_def', 'sustain_app']
    
    # Data cleaning
    def clean_entry(entry):
        return entry.split('/')[0].strip()
    
    for col in ['voluntary', 'sustain_percept']:
        data[col] = data[col].apply(clean_entry)
    
    # Translation section
    st.header("Text Translation")
    if st.checkbox("Show Translation Progress"):
        tokenizer, model = load_translation_model()
        
        with st.spinner('Translating content...'):
            for col in ['sustain_reason', 'sustain_def', 'sustain_app']:
                data[f'{col}_translated'] = data[col].apply(
                    lambda x: tokenizer.decode(
                        model.generate(**tokenizer(x, return_tensors="pt", truncation=True))[0],
                        skip_special_tokens=True
                    ) if pd.notnull(x) else x
                )
        st.success("Translation completed!")

    # Main display
    st.subheader("Processed Data Preview")
    st.dataframe(data.head())

    # Analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Topic Modeling", "Word Cloud", "Clustering"])

    with tab1:
        st.header("Sentiment Analysis")
        data['sentiment_score'] = data['sustain_reason_translated'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        data['sentiment_category'] = data['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
        )
        st.write(data[['sustain_reason_translated', 'sentiment_score', 'sentiment_category']])

    with tab2:
        st.header("Topic Modeling")
        vectorizer = CountVectorizer(stop_words='english')
        dtm = vectorizer.fit_transform(data['sustain_def_translated'])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        
        for idx, topic in enumerate(lda.components_):
            st.subheader(f"Topic {idx+1}")
            st.write(", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]))

    with tab3:
        st.header("Word Cloud")
        all_text = " ".join(data['sustain_def_translated'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with tab4:
        st.header("Text Clustering")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['sustain_def_translated'])
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['cluster'] = kmeans.fit_predict(tfidf_matrix)
        st.write(data[['sustain_def_translated', 'cluster']])

    # Download section
    st.sidebar.header("Download Results")
    if st.sidebar.button("Prepare Download"):
        output = data.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Processed Data",
            data=output,
            file_name='processed_data.csv',
            mime='text/csv'
        )

else:
    st.info("Please upload an Excel file to begin analysis")

