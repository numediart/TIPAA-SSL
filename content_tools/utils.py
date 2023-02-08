# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, psutil;print_memory_usage=lambda stage: print(stage + ": "+ str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))


import streamlit as st
import inspect
import textwrap


import sys
sys.path.append('./')
print_memory_usage('RAM - streamlit utils, before loading')
from src.text_processing import prefill_content
print_memory_usage('RAM - streamlit utils, after loading text_processing')
from src.label_data_processing import get_formatted_cmudict
print_memory_usage('RAM - streamlit utils, after loading label_data_processing')
from src.pronunciation_dictionaries import cmu_reducer
print_memory_usage('RAM - streamlit utils, after loading pronunciation_dictionaries')

formatted_cmudict_df=get_formatted_cmudict()
print_memory_usage('RAM - streamlit utils, after formatted_cmudict_df')


@st.experimental_singleton
def get_model(model_type):
    from transformers import Wav2Vec2ForCTC
    # Create a model of the specified type
    return Wav2Vec2ForCTC.from_pretrained(model_type)



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
