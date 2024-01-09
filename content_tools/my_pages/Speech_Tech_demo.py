import pandas as pd
import streamlit as st
from st_audiorec import st_audiorec

from DL_speech_tech import (
    PostAnalysisResult,
    multiple_aspect_from_formatted_phonetics_audio,
)
from src.text_processing import count_syllables, prefill_for_sentence
from src.wav2vec2_frame_prediction import AudioLoadResult, AudioMode, AudioStatus

st.set_page_config(page_icon="✂️", page_title="Speech Tech Demo")
st.title("Speech Tech Demo")


def display_results(
    audio_load: AudioLoadResult,
    detection_df: pd.DataFrame | None,
    post_analysis_results: PostAnalysisResult | None,
    max_per: float,
    min_pitch_ratio: float,
):
    st.markdown("## Results")

    if audio_load.status == AudioStatus.SUCCESS:
        st.markdown("### Mutiple speech aspect detection")
        st.dataframe(detection_df, hide_index=True)

        st.markdown("### Post analysis")
        if post_analysis_results is None:
            st.error("Recording should be rejected (pairwise alignment failed)")
        elif post_analysis_results.df_segmented is None:
            st.error("Recording should be rejected (DTW failed)")
        else:
            per = post_analysis_results.per_aligned
            st.markdown(
                f"""
                - Detected aligned phones:
                    `{" ".join(post_analysis_results.predicted_aligned_phones)}`
                - Expected aligned phones:
                    `{" ".join(post_analysis_results.expected_aligned_phones)}`
                - Phone error rate (lower = better matching):
                    {per:.2f}
                - DTW cost (higher = better matching):
                    {post_analysis_results.dtw_cost:.2f}
                - Phone count ratio:
                    {post_analysis_results.phone_count_ratio:.2f}
                    (ratio of number of detected phones over expected phones)
                - Silence ratio:
                    {audio_load.silent_sample_ratio:.2f}
                    (fraction of silent samples in the raw recording)
                - Pitch ratio:
                    {audio_load.pitch_sample_ratio:.2f}
                    (fraction of samples with detected voice pitch)
                - Syllable frequency: {audio_load.speech_rate:.2f} syllables/s
                """
            )
            if per > max_per:
                st.error("Recording should be rejected (PER too high)")
            elif (
                audio_load.pitch_sample_ratio
                and audio_load.pitch_sample_ratio < min_pitch_ratio
            ):
                st.error("Recording should be rejected (pitch ratio too low)")
            else:
                st.success("Recording should be accepted")

            st.markdown("**Max proba based detection**")
            st.dataframe(post_analysis_results.df_detection, hide_index=True)
            st.markdown("**DTW based detection**")
            st.dataframe(post_analysis_results.df_segmented, hide_index=True)
    elif audio_load.status in [AudioStatus.TOO_LONG, AudioStatus.TOO_SHORT]:
        st.error(
            "Recording should be rejected"
            f" (syllable frequency: {audio_load.speech_rate:.2f} syllables/s"
        )
    else:
        st.error(f"Recording should be rejected ({audio_load.status!s})")


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
        assert formatted_phonetics_mod is not None
        syll_count = count_syllables(formatted_phonetics_mod)

        st.markdown("Either record yourself:")

        wav_audio_data_rec = st_audiorec()

        audio_upload = st.file_uploader(
            "Or upload an audio file:", type=["wav", "mp3", "ogg"]
        )

        if wav_audio_data_rec is not None or audio_upload is not None:
            if wav_audio_data_rec is not None:
                wav_audio_data = wav_audio_data_rec
                audio_type = "audio/wav"
                data_used = "recording"
            elif audio_upload is not None:
                wav_audio_data = audio_upload.getvalue()
                file_name = audio_upload.name.lower()
                if file_name.endswith(".mp3"):
                    audio_type = "audio/mp3"
                elif file_name.endswith(".ogg"):
                    audio_type = "audio/ogg"
                else:
                    audio_type = "audio/wav"
                # display audio data as received
                st.audio(wav_audio_data, format=audio_type)
                data_used = "upload"
            else:
                raise ValueError("No audio data")

            min_speech_rate = 1.0
            max_speech_rate = 8.0
            silence_threshold = 30.0
            proba_thresh = 0.5
            max_per = 0.7 if syll_count > 1 else 0.8
            min_pitch_ratio = 0.1

            if st.checkbox("Tune parameters"):
                min_speech_rate = st.number_input(
                    "Minimum speech rate (syllables/s)",
                    value=min_speech_rate,
                    min_value=0.0,
                )
                max_speech_rate = st.number_input(
                    "Maximum speech rate (syllables/s)",
                    value=max_speech_rate,
                    min_value=0.0,
                )
                proba_thresh = st.number_input(
                    "Minimum probability threshold",
                    value=proba_thresh,
                    min_value=0.0,
                    max_value=1.0,
                )
                silence_threshold = st.number_input(
                    "Silence amplitude threshold (dB)",
                    value=silence_threshold,
                    min_value=20.0,
                )
                max_per = st.number_input(
                    "Maximum phone error rate (PER) for acceptance",
                    value=max_per,
                    min_value=0.0,
                    max_value=1.0,
                )
                min_pitch_ratio = st.number_input(
                    "Minimum pitch frame ratio for acceptance",
                    value=min_pitch_ratio,
                    min_value=0.0,
                    max_value=1.0,
                )

            if st.button(f"Run Analysis (using {data_used})"):
                (
                    audio_load,
                    detection_df,
                    post_analysis_results,
                ) = multiple_aspect_from_formatted_phonetics_audio(
                    wav_audio_data,
                    phonetics=formatted_phonetics_mod,
                    mode=AudioMode.BYTES,
                    proba_thresh=proba_thresh,
                    min_speech_rate=min_speech_rate,
                    max_speech_rate=max_speech_rate,
                    silence_threshold=silence_threshold,
                )

                display_results(
                    audio_load,
                    detection_df,
                    post_analysis_results,
                    max_per,
                    min_pitch_ratio,
                )
