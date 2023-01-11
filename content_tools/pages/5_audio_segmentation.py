import streamlit as st
st.set_page_config(page_icon="✂️", page_title="Audio Segmentation")
import pandas as pd
from audio_segmentation import analyze_files_and_build_transcripts, extract_zip_to_dict, reconstruct_zip_from_dict
from glob import glob
import zipfile, os

from datetime import datetime
import uuid
import shutil

from utils import check_password, get_model

model_name="hf_models/facebook/wav2vec2-base-960h"


def unzip_file(file, outdir='streamlit_apps'):
    cwd=os.getcwd()
    if not os.path.exists(outdir): os.makedirs(outdir)
    os.chdir(outdir)
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
            item.extractall()  # extract it in the working directory
    else:
        print("This is not a zipfile")
    os.chdir(cwd)





progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

def report_function(i, title=""):
    status_text.text(title+" %i%% Complete" % i)
    progress_bar.progress(i)

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/scissors_2702-fe0f.png",
    width=100,
)

st.title("Audio Segmentation")

if check_password():

    model=get_model(model_name)

    c29, c30, c31 = st.columns([1, 6, 1])
    with c30:

        uploaded_file = st.file_uploader(
            "",
            key="audio_segmentation_file",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
        )

        @st.experimental_memo
        def launch_analysis(file_dict, results_dir):
            analyze_files_and_build_transcripts(model, file_dict, report_callback=report_function,results_dir=results_dir)

        # @st.cache
        # @st.experimental_memo
        def process(uploaded_file):
            if uploaded_file is not None:
                file_container = st.expander("Check your uploaded .csv")

                now=datetime.now()
                date_time = now.strftime("%m_%d_%Y_%H:%M:%S")

                maindir='content_tools/uploaded/'+date_time+'_'+str(uuid.uuid4())
                # outdir=maindir+'/unzipped/'

                file_dict=extract_zip_to_dict(uploaded_file)

                


                # unzip_file(uploaded_file, outdir=outdir)
                # xlsx_files=[el for el in glob(outdir+'/*') if el.endswith('.xlsx')]+[el for el in glob(outdir+'/*/*') if el.endswith('.xlsx')]
                # if len(xlsx_files)!=1: 
                #     print('There should be exactly 1 xlsx file, but there is/are '+str(len(xlsx_files)))
                # else:
                #     xlsx_file=xlsx_files[0]

            else:
                st.info(
                    f"""
                        👆 Upload a .zip file first. 
                        """
                )

                st.stop()

            st.success(
                f"""
                    Upload successfull !
                    """
            )

            # df = pd.read_excel(xlsx_file)
            # print(df)

            # st.subheader("Sample of the data will appear below 👇 ")
            # st.dataframe(df)

            # directory,_=os.path.split(xlsx_file)
            results_dir=maindir+'/results/'

            # dir,_=os.path.split(path_xlsx)
            if not os.path.exists(results_dir): os.makedirs(results_dir)

            # sheet_to_df_map = pd.read_excel(xlsx_file, sheet_name=None)

            launch_analysis(file_dict, results_dir)
            # analyze_files_and_build_transcripts(_model, xlsx_file, report_callback=report_function,results_dir=results_dir)


            zf = zipfile.ZipFile(maindir+"/segmentation_results.zip", "w")
            for dirname, subdirs, files in os.walk(results_dir):
                zf.write(dirname)
                for filename in files:
                    zf.write(os.path.join(dirname, filename))
            zf.close()

            shutil.rmtree(results_dir)
            # shutil.rmtree(outdir)
            return maindir+"/segmentation_results.zip"
        
        zip_result_path=process(uploaded_file)

        # def on_click_download_zip():
        #     st.write('Thanks for downloading!')
        #     st.write('Removing files...')
        #     print('Removing files...')
        #     shutil.rmtree(outdir)

        with open(zip_result_path, 'rb') as f:
            d=st.download_button('Download Zip', f, file_name='segmentation_results.zip', mime="application/zip")#, on_click=on_click_download_zip)  # Defaults to 'application/octet-stream'
