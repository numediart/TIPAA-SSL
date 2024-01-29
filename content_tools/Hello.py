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

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

from utils import check_password

# Just trigger the import once for all here so that it's not loaded when loading another page
# from audio_segmentation import *
# from process_data_entry_interface_output import *


import streamlit as st

# with st.echo("below"):
from st_pages import Page, add_page_title, show_pages


def run():
    show_pages(
        [
            Page("content_tools/Hello.py", "Content tools", "🏠"),
            Page(
                "content_tools/my_pages/linguistic_data_extraction.py",
                "Linguistic Data Extraction",
                ":books:",
            ),
            Page("content_tools/my_pages/phonetizer.py", "Phonetizer", "✏️"),
            Page("content_tools/my_pages/CMUDict_explorer.py", "CMU Dict Explorer", "📖"),
            Page(
                "content_tools/my_pages/Speech_Tech_demo.py",
                "Speech Tech Demo",
                ":microphone:",
            ),
        ]
    )

    # st.set_page_config(
    #     page_title="Content tools",
    #     page_icon="👋",
    # )

    st.write("# Welcome to content-tools! 👋")

    # st.sidebar.success("Select a tool above.")

    st.markdown(
        """
        This site contains tools and demo related to Speech & Language processing to be used internally
        **👈 Select a tool from the sidebar** to see some examples
    """
    )


if __name__ == "__main__":
    # if check_password():
    run()
