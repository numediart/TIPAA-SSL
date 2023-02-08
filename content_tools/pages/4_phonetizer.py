import streamlit as st
st.set_page_config(page_icon="✂️", page_title="Phonetization, syllabification in text and phonetics as well as text processing for numbers, acronyms, etc.")
import pandas as pd
from streamlit import download_button
from glob import glob
import io
from utils import check_password, prefill_content



progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

def report_function(i, title=""):
    status_text.text(title+" %i%% Complete" % i)
    progress_bar.progress(i)

st.title("Phonetization, syllabification ")

@st.experimental_memo
def process(json_str):
    # Your processing function goes here
    # data = json.loads(json_str)
    sentences=[el for el in json_str.split('\n') if el!='']
    # Do something with the data
    # result = data['key']
    df=prefill_content(sentences)
    
    return df

if check_password():

    st.markdown("""
    This tool allows you to extract necessary information 
    You just have to paste a list of words or sentences, and then click on process. The tool will process the content and provide you the linguistic content containing:
    - the phonetization of the phrases, cut in syllables,
    - the text with processing for numbers, acronyms, etc. also cut into syllables 
    - note if there are inconsistencies in the number of syllables in phonetics and in texts (noting that in a list of word index), you can filter that in google sheet for manually fixing them if there are some
    """)

    json_str = st.text_area('Paste your words/sentences here:')

    if st.button('Process'):
        linguistic_data = process(json_str)
        
        # Display the DataFrame
        st.dataframe(linguistic_data)

        # text buffer
        ling_buf = io.BytesIO()
        # saving a data frame to a buffer (same as with a regular file):
        linguistic_data.to_csv(ling_buf)
        d=st.download_button('Download linguistic data', ling_buf, file_name="linguistic_data.csv")#, on_click=on_click_download_zip)  # Defaults to 'application/octet-stream'