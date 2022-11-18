
import os

def download_hf_model(hf_path_base="hf_models", hf_rep='charsiu', hf_model='en_w2v2_fc_10ms'):
    hf_model_path='/'.join([hf_path_base,hf_rep,hf_model])
    if not os.path.exists(hf_model_path):
        hf_path='/'.join([hf_path_base,hf_rep])
        if not os.path.exists(hf_path):os.makedirs(hf_path)
        cmd='cd '+hf_path+' && git clone https://huggingface.co/'+'/'.join([hf_rep,hf_model])+' && cd '+hf_model+' && git lfs pull'
        os.system(cmd)


hf_path_base="hf_models"
hf_rep='charsiu'
download_hf_model(hf_path_base=hf_path_base, hf_rep=hf_rep, hf_model='en_w2v2_fc_10ms')
download_hf_model(hf_path_base=hf_path_base, hf_rep=hf_rep, hf_model='tokenizer_en_cmu')
download_hf_model(hf_path_base=hf_path_base, hf_rep=hf_rep, hf_model='tokenizer_zh_pinyin')
download_hf_model(hf_path_base=hf_path_base, hf_rep="speechbrain", hf_model="metricgan-plus-voicebank")

download_hf_model(hf_path_base=hf_path_base, hf_rep="facebook", hf_model="wav2vec2-xlsr-53-espeak-cv-ft")
download_hf_model(hf_path_base=hf_path_base, hf_rep="facebook", hf_model="wav2vec2-xlsr-53-espeak-cv-ft")