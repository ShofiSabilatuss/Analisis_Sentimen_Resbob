import streamlit as st
import pandas as pd
from sentiment_model import predict_sentiment

st.title("📊 Analisis Sentimen")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df)

    if st.button("Analisis"):
        df['sentimen'] = df['text'].apply(predict_sentiment)
        st.success("Selesai!")

        st.dataframe(df)
