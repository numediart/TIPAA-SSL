import torch
import numpy as np
from glob import glob
import librosa
import sys
import random
import soundfile as sf

from src.pronunciation_dictionaries import cmu_1_char_to_gibberish
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from scipy.io.wavfile import read

device = 'cpu' # cuda or cpu

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    # display(HTML(df.to_html()))
    return df

def melgan_analysis_synthesis(s):
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    mel = vocoder(torch.from_numpy(s.astype(np.float32))[None])
    s=vocoder.inverse(mel)  # audio (torch.tensor) -> (batch_size, 80, timesteps)
    return s.numpy().flatten()

# from speechbrain.pretrained import SpectralMaskEnhancement
# enhance_model = SpectralMaskEnhancement.from_hparams(
#         source="speechbrain/metricgan-plus-voicebank",
#         savedir="pretrained_models/metricgan-plus-voicebank",
# )



def speech_enhancement(s):
    from speechbrain.pretrained import SpectralMaskEnhancement
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="hf_models/speechbrain/metricgan-plus-voicebank"
        # source="speechbrain/metricgan-plus-voicebank"
    )
    
    s=torch.from_numpy(s.astype(np.float32))[None]
    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(s, lengths=torch.tensor([1.]))
    return enhanced.numpy().flatten()

if False:
    # If we want to use one of these, download them with scripts/downloas_models.py instead as for enhance_model

    from speechbrain.pretrained import VAD
    # Model is downloaded from the speechbrain HuggingFace repo
    VAD_model = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir="pretrained_models/vad-crdnn-libriparty",
    )
    def run_VAD(s):
        prob = VAD_model.get_speech_prob_chunk(torch.from_numpy(s.astype(np.float32))[None], torch.tensor([1.]))
        return prob

    # keep that in case I need an asr but hide it not to unnecessarily download the model while not used
    from speechbrain.pretrained import EncoderDecoderASR
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech", 
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
    # https://colab.research.google.com/drive/1hX5ZI9S4jHIjahFCZnhwwQmFoGAi3tmu?usp=sharing#scrollTo=BZ_rbmnBCy-w
    # https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech
    def asr(s):
        # Add relative length tensor
        texts, ns = asr_model.transcribe_batch(torch.from_numpy(s.astype(np.float32))[None], torch.tensor([1.]))
        return texts, ns

    # maybe try this one as speech enhancement, but it works at 8kHz which is annoying
    from speechbrain.pretrained import SepformerSeparation as separator
    separator_model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='pretrained_models/sepformer-wham-enhancement')
    def speech_separation(s):
        # for custom file, change path
        # est_sources = separator_model.separate_file(path='speechbrain/sepformer-wham-enhancement/example_wham.wav') 
        est_sources = separator_model.separate_batch(torch.from_numpy(s.astype(np.float32))[None], torch.tensor([1.])) 

def torch_example_use():
    model_list=torch.hub.list('s3prl/s3prl')
    model_type="wav2vec2_hug_base_960"

    if model_type not in model_list: print('The model type is not in the list, maybe the name has changes, check torch.hub.list of the repo')
    model=torch.hub.load('s3prl/s3prl',model_type)#.to(device)
    audio_path="audio_recordings/"
    paths=glob(audio_path+'/*')
    wavs=[]
    for p in paths:
        try:
            s,fs=librosa.load(p, sr=16000)
            wavs.append(torch.from_numpy(s))
        except:
            print("Unexpected error:", sys.exc_info()[0])

    # wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)] # list of unpadded wavs `[wav1, wav2, ...]`, each wav is in `torch.FloatTensor`
    with torch.no_grad():
        reps = model(wavs) # list of unpadded representations `[rep1, rep2, ...]`, each erp is of the shape `(extracted_seqlen, feature_dim)`

    # from s3prl.downstream import asr
    # config=asr.__path__[0]+'/config.yaml'

    # # from s3prl.downstream import libri_phone

    # # out=libri_phone(reps, downstream_expert='test', expdir='test')
    # expert=asr(config)
    # torch.nn.Module(config)


def inference(model, processor, s, fs=16000):
    input_values = processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to(device)
    with torch.no_grad(): logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    result = processor.batch_decode(pred_ids)[0]
    return result


def pred_to_gibberish(pred):
    gibberish=''
    for c in pred:
        if c!=' ':
            gibberish+=cmu_1_char_to_gibberish[c]+'_'
        else:
            gibberish+=' '
    gibberish=gibberish.replace('_ ',' ')[:-1]
    
    return gibberish

if __name__=="__main__":

    path = 'dl_models/wav2vec2-base-libri-pr'
    processor = Wav2Vec2Processor.from_pretrained(path)
    model = Wav2Vec2ForCTC.from_pretrained(path+'/checkpoint-10700')
    # model.to(device)
    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav'
    s,fs=librosa.load(path, sr=16000)
    result=inference(model, processor, s,fs)
    print(result)

    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav'
    s,fs=librosa.load(path, sr=16000)
    inference(model, processor, s, fs)

    n_chunks=40
    chunk_len=int(len(s)/n_chunks)
    for i in range(n_chunks):
        print(inference(model, processor, s[i*chunk_len:(i+1)*chunk_len], fs))

    # phonemes:
    processor.batch_decode([np.arange(1000)])[0]
    processor.batch_decode([[9]])[0]
    processor.batch_decode([np.arange(20,30)])[0]
    # the long i  "i:" is indeed one phoneme!
    processor.batch_decode([[30]])[0]
    [processor.batch_decode([[i]])[0] for i in range(400)]

    
    df=pd.read_csv('data/exercise_data_export.csv')
    preds=[]
    for i,r in df.iterrows():
        s,fs=librosa.load('data/scaleway-audio-files/'+r.audio_file_url, sr=16000)
        result=inference(model, processor, s,fs)
        print(result)
        preds.append(result)
    
    df["preds"]=preds

    df['pred_gibberish']=df["preds"].apply(lambda r:pred_to_gibberish(r))
    df.pronounciation_guide

    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_pContrast[df_pContrast.target_phoneme=='T'][['pred_gibberish','pronounciation_guide']]
    df_pContrast[df_pContrast.target_phoneme=='D'][['pred_gibberish','pronounciation_guide']]
    df_pContrast[df_pContrast.target_phoneme=='IH0_D'][['pred_gibberish','pronounciation_guide']]


    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']
    for v in vowels:
        res=df_pContrast[df_pContrast.target_phoneme.str.contains(v)]
        print(v,res[['pred_gibberish','pronounciation_guide']])

