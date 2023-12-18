import streamlit as st
from st_audiorec import st_audiorec

st.set_page_config(page_icon="✂️", page_title="Speech Tech Demo")

from utils import (
    check_password,
    multiple_aspect_from_formatted_phonetics_audio,
    prefill_content,
    prefill_for_sentence,
)

from DL_speech_tech import MAX_PER_FOR_ACCEPTANCE

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()


def report_function(i, title=""):
    status_text.text(title + " %i%% Complete" % i)
    progress_bar.progress(i)


st.title("Speech Tech Demo")


# @st.experimental_memo
@st.cache_data
def process(json_str):
    sentences = [el for el in json_str.split('\n') if el != '']
    df, df_errors = prefill_content(sentences)

    return df


# if check_password():
if True:
    st.markdown(
        """
    Enter a sentence you want to practice and have your speech analyzed,
    then record yourself. Click on "Run Analysis" to analyze and see the results.
    """
    )

    sentence = st.text_input("Sentence")

    if len(sentence) > 0:
        formatted_phonetics = prefill_for_sentence(sentence)["phonetics"]

        formatted_phonetics_mod = st.text_input(
            "Adjust phonetics if needed:", value=formatted_phonetics
        )
        # st.text(formatted_phonetics)

        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            # display audio data as received on the backend
            # st.audio(wav_audio_data, format='audio/wav')

            if st.button("Run Analysis"):
                (
                    audio_status,
                    detection_df,
                    segmented_df,
                    per,
                    dtw_cost,
                ) = multiple_aspect_from_formatted_phonetics_audio(
                    wav_audio_data, phonetics=formatted_phonetics_mod, mode="bytes"
                )
                st.markdown("## Results\n### Mutiple speech aspect detection")
                st.markdown("Audio check status: " + audio_status)
                st.dataframe(detection_df)
                st.markdown(
                    f"""
                    ### Post analysis
                     - Phone error rate (lower is better): {per:.2f}
                     - DTW cost (higher is better): {dtw_cost:.2f}"""
                )
                if per is None:
                    st.error("Recording should be rejected (PER: None)")
                elif per > MAX_PER_FOR_ACCEPTANCE:
                    st.error(
                        f"Recording should be rejected (PER: {per:.2f} > {MAX_PER_FOR_ACCEPTANCE:.2f})"
                    )
                else:
                    st.success(
                        f"Recording should be accepted (PER: {per:.2f} <= {MAX_PER_FOR_ACCEPTANCE:.2f})"
                    )
                st.dataframe(segmented_df)
