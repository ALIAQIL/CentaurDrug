import requests
import streamlit as st

st.set_page_config(page_title="CentaurDrug", layout="wide")

st.title("CentaurDrug")
st.subheader("AI-assisted lead optimization with ADMET verification")

smiles = st.text_input("Enter SMILES", "CCO")

if st.button("Evaluate molecule"):
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={"smiles": smiles},
        timeout=30,
    )

    if response.ok:
        st.json(response.json())
    else:
        st.error("API error")