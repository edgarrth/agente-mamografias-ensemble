import os, requests, streamlit as st
API=os.getenv("API_URL","http://fastapi:8000")
st.set_page_config(page_title="Mammography AI Agent",layout="wide")
st.title("Mammography AI Agent — Research Prototype")
st.warning("Research only. The output is not an autonomous medical diagnosis.")
if st.button("Refresh status"):
    try:
        st.json(requests.get(f"{API}/workspace/status",timeout=20).json())
    except Exception as e: st.error(str(e))
st.subheader("Dataset acquisition")
datasets=st.multiselect("Datasets",["cbis_ddsm","vindr"],default=["cbis_ddsm"])
col1,col2=st.columns(2)
if col1.button("Request download / show instructions"):
    try: st.json(requests.post(f"{API}/datasets/download",json={"datasets":datasets},timeout=30).json())
    except Exception as e: st.error(str(e))
if col2.button("Prepare selected"):
    try: st.json(requests.post(f"{API}/datasets/prepare",json={"datasets":datasets},timeout=600).json())
    except Exception as e: st.error(str(e))
st.info("Normal and experimental runs are intentionally exposed through reproducible CLI commands documented in README.md.")
