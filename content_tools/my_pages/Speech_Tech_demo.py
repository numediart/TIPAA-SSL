import streamlit as st

st.set_page_config(page_icon="✂️", page_title="Speech Tech Demo")
import pandas as pd
from streamlit import download_button
from glob import glob
import io
from utils import (
    check_password,
    prefill_content,
    prefill_for_sentence,
    multiple_aspect_from_formatted_phonetics_audio,
)

import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components

from st_custom_components import st_audiorec


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()


def report_function(i, title=""):
    status_text.text(title + " %i%% Complete" % i)
    progress_bar.progress(i)


st.title("Speech Tech Demo")


# @st.experimental_memo
@st.cache_data
def process(json_str):
    # Your processing function goes here
    # data = json.loads(json_str)
    sentences = [el for el in json_str.split('\n') if el != '']
    # Do something with the data
    # result = data['key']
    df, df_errors = prefill_content(sentences)

    return df


if check_password():
    st.markdown(
        """
    Enter a sentence you want to practice and have your speech analyzed, then record yourself. Click on "Run Analysis" to analyze and see the results.
    """
    )

    sentence = st.text_input(f'Sentence')

    if len(sentence) > 0:
        formatted_phonetics = prefill_for_sentence(sentence)['phonetics']

        formatted_phonetics_mod = st.text_input(f'Phonetics', value=formatted_phonetics)
        # st.text(formatted_phonetics)

        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            # display audio data as received on the backend
            # st.audio(wav_audio_data, format='audio/wav')

            if st.button('Run Analysis'):
                detection_df = multiple_aspect_from_formatted_phonetics_audio(
                    wav_audio_data, phonetics=formatted_phonetics_mod, mode='bytes'
                )
                st.dataframe(detection_df)
