import torch
import numpy as np
from glob import glob
import librosa
import sys
import random
import soundfile as sf

from text_processing import cmu_1_char_to_gibberish
import pandas as pd
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


from speechbrain.pretrained import SpectralMaskEnhancement
enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )
def speech_enhancement(s):
    # Load and add fake batch dimension
    # noisy = enhance_model.load_audio(
    #     "speechbrain/metricgan-plus-voicebank/example.wav"
    # ).unsqueeze(0)
    s=torch.from_numpy(s.astype(np.float32))[None]
    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(s, lengths=torch.tensor([1.]))
    return enhanced.numpy().flatten()


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

    from s3prl.downstream import asr
    config=asr.__path__[0]+'/config.yaml'

    # from s3prl.downstream import libri_phone

    # out=libri_phone(reps, downstream_expert='test', expdir='test')

    expert=asr(config)

    torch.nn.Module(config)



if __name__=="__main__":
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, TrainingArguments, Trainer

    path = 'dl_models/wav2vec2-base-libri-pr'
    processor = Wav2Vec2Processor.from_pretrained(path)
    model = Wav2Vec2ForCTC.from_pretrained(path+'/checkpoint-10700')
    # model.to(device)

    def inference(s, fs=16000):
        
        # input_values = processor(dataset_dict["test"][0]["speech"], sampling_rate=dataset_dict["test"][0]["sampling_rate"], return_tensors="pt").input_values.to(device)
        input_values = processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        result = processor.batch_decode(pred_ids)[0]

        # convert ids to tokens
        # print("".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())))

        """The output should make it a bit clearer how CTC works in practice. The model is to some extent invariant to speaking rate since it has learned to either just repeat the same token in case the speech chunk to be classified still corresponds to the same token. This makes CTC a very powerful algorithm for speech recognition since the speech file's transcription is often very much independent of its length.

        I again advise the reader to take a look at [this](https://distill.pub/2017/ctc) very nice blog post to better understand CTC.
        """
        return result

    path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav'
    s,fs=librosa.load(path, sr=16000)
    result=inference(s,fs)
    print(result)

    gibberish=''
    for c in result:
        if c!=' ':
            gibberish+=cmu_1_char_to_gibberish[c]
        else:
            gibberish+=' '
    

