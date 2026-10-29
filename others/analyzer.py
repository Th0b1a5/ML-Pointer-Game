import streamlit as st
import pandas as pd

st.title("ML performance analyzer")

files = st.file_uploader("Upload a CSV file", accept_multiple_files=True)

for file in files:
    st.header(file.name)
    log = pd.read_csv(file)
    st.line_chart(log)
    with st.expander(f"Raw data for {file.name.removesuffix(".csv")}"):
        st.dataframe(log)
