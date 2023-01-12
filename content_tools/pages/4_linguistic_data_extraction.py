import streamlit as st
st.set_page_config(page_icon="✂️", page_title="Linguistic Data Extraction")
import pandas as pd
from streamlit import download_button
from process_data_entry_interface_output import nested_dict_to_df, df_to_nested_dict, process_content
from glob import glob
import io

from utils import check_password



progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

def report_function(i, title=""):
    status_text.text(title+" %i%% Complete" % i)
    progress_bar.progress(i)

st.title("Linguistic data extraction")

import json

@st.experimental_memo
def process(json_str):
    # Your processing function goes here
    data = json.loads(json_str)
    # Do something with the data
    # result = data['key']
    df=nested_dict_to_df(data)
    all_sentences_df, linguistic_data=process_content(df)
    
    return df, all_sentences_df, linguistic_data

if check_password():
    json_str = st.text_area('Paste your JSON variable here:')

    if st.button('Process'):
        df, all_sentences_df, linguistic_data = process(json_str)
        
        # Display the DataFrame
        st.dataframe(linguistic_data)

        # text buffer
        ling_buf = io.BytesIO()
        # saving a data frame to a buffer (same as with a regular file):
        linguistic_data.to_csv(ling_buf)
        d=st.download_button('Download linguistic data', ling_buf, file_name="linguistic_data.csv")#, on_click=on_click_download_zip)  # Defaults to 'application/octet-stream'


        # text buffer
        all_sentences_df_buf = io.BytesIO()
        # saving a data frame to a buffer (same as with a regular file):
        all_sentences_df.to_csv(all_sentences_df_buf)
        d=st.download_button('Download all phrases data', all_sentences_df_buf, file_name="all_phrases_df.csv")#, on_click=on_click_download_zip)  # Defaults to 'application/octet-stream'


        # text buffer
        df_buf = io.BytesIO()
        # saving a data frame to a buffer (same as with a regular file):
        df.to_csv(df_buf)
        d=st.download_button('Download master sheet data', df_buf, file_name="master_sheet.csv")#, on_click=on_click_download_zip)  # Defaults to 'application/octet-stream'

