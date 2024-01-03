import streamlit as st
from st_audiorec import st_audiorec

from src.wav2vec2_frame_prediction import AudioMode
from DL_speech_tech import multiple_aspect_from_formatted_phonetics_audio

st.set_page_config(page_icon="✂️", page_title="Speech Tech Demo")

from utils import (
    check_password,
    prefill_content,
    prefill_for_sentence,
)

from DL_speech_tech import MAX_PER_FOR_ACCEPTANCE

st.title("Speech Tech Demo")


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

            if st.button(f"Run Analysis (using {data_used})"):
                (
                    audio_load,
                    detection_df,
                    post_analysis_results,
                ) = multiple_aspect_from_formatted_phonetics_audio(
                    wav_audio_data,
                    phonetics=formatted_phonetics_mod,
                    mode=AudioMode.BYTES,
                )
                print(audio_load)

                st.markdown("## Results")
                st.markdown("Basic audio check status: " + str(audio_load.status))

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
                        - Detected phones: {" ".join(post_analysis_results.predicted_phones)}
                        - Detected aligned phones: {" ".join(post_analysis_results.predicted_aligned_phones)}
                        - Expected aligned phones: {" ".join(post_analysis_results.expected_aligned_phones)}
                        - Phone error rate (lower = better matching): {per:.2f}
                        - DTW cost (higher = better matching): {post_analysis_results.dtw_cost:.2f}
                        - Silent frame ratio: {post_analysis_results.silent_frame_ratio:.2f} (fraction of silent frames in the raw recording)
                        - Phone count ratio: {post_analysis_results.phone_count_ratio:.2f} (ratio of number of detected phones over expected phones)
                        - Pitch ratio: {audio_load.pitch_frame_ratio:.2f} (ratio of pitch in the raw recording over expected pitch)
                        """
                    )
                    if per > MAX_PER_FOR_ACCEPTANCE:
                        st.error(
                            f"Recording should be rejected (PER: {per:.2f} > {MAX_PER_FOR_ACCEPTANCE:.2f})"
                        )
                    else:
                        st.success(
                            f"Recording should be accepted (PER: {per:.2f} <= {MAX_PER_FOR_ACCEPTANCE:.2f})"
                        )
                    st.dataframe(post_analysis_results.df_segmented, hide_index=True)
