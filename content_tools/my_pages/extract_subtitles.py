import streamlit as st
st.set_page_config(page_icon="✂️", page_title="Speech recognition for subtitles")
import pandas as pd
from audio_segmentation import SAMPLERATE, build_all_words_df, group_words, create_srt
import librosa
import io
from utils import check_password, get_model, read_audio_file
model_name="hf_models/facebook/wav2vec2-base-960h"




progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

def report_function(i, title=""):
    status_text.text(title+" %i%% Complete" % i)
    progress_bar.progress(i)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/scissors_2702-fe0f.png",
    width=100,
)

st.title("Speech recognition for subtitles")

if check_password():
    # do this after paswword check
    model=get_model(model_name)

    c29, c30, c31 = st.columns([1, 6, 1])
    with c30:

        uploaded_file = st.file_uploader(
            "",
            key="extract_subtitles_file",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
        )

        # @st.experimental_memo
        @st.cache_data
        def extract_subtitles(uploaded_file):
            if uploaded_file is not None:
                file_container = st.expander("Check your uploaded .csv")                
            else:
                st.info(
                    f"""
                        👆 Upload an audio or video file first. 
                        """
                )

                st.stop()

            st.success(
                f"""
                    Upload successfull !
                    """
            )
        
            # audio,fs=librosa.load(uploaded_file, sr=SAMPLERATE)
            audio,fs=read_audio_file(uploaded_file, fs=SAMPLERATE)
            all_df_words=build_all_words_df(audio, model=model)
            phrases_df=group_words(all_df_words)

            f = io.BytesIO()
            create_srt(phrases_df, file=f)

            st.subheader("Sample of the data will appear below 👇 ")
            st.dataframe(phrases_df)

            return f
        
        f=extract_subtitles(uploaded_file)
        d=st.download_button('Download subtitles srt file', f, file_name="subtitles.srt")#, on_click=on_click_download_zip)  # Defaults to 'application/octet-stream'
