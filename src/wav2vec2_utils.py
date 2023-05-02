from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Model
# from datasets import load_dataset
import librosa
import torch


import math
import numpy as np
# from src.phonetization import MAILABS_data, MLS_data
from jiwer import wer
# from src.phonetization import stress, unstress
import os
import pandas as pd
from src.text_processing import remove_stress_annots
from itertools import groupby

# from src.layer_extraction import get_last_hidden_state

def get_last_hidden_state(s, fs, processor, model):
    input_values = processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to('cpu')
    with torch.no_grad(): 
        return model(input_values).hidden_states[-1]
def get_logits(s, fs, processor, model):
    input_values = processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to('cpu')
    with torch.no_grad(): 
        logits = model(input_values).logits
    return logits

def inference(s, fs, processor, model):
    input_values = processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to('cpu')
    with torch.no_grad(): 
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    # this is just a lookup operation, so no worries about computational costs
    timed_pred=[processor.batch_decode([[id]])[0] for id in pred_ids[0]]

    pred_filtered=list(filter(None, timed_pred))
    # this removes consecutive duplicates
    phones=[x[0] for x in groupby(pred_filtered)]

    return timed_pred, phones

def compute_predictions_and_PER(df_sample):
    """compute predictions phoneme error rate for the examples od a dataframe, and create a new column storing predictions and PER

    Args:
        df_sample ([type]): [description]

    Returns:
        [type]: [description]
    """
    PERs=[]
    preds=[]
    for i,r in tqdm(df_sample.iterrows()):
        truth=list(filter(None, r.phonetics.split(' ')))
        audio_path=r.path
        
        s,fs=librosa.load(audio_path, sr=16000)
        timed_pred, phones_pred=inference(s, fs)

        # word error rate and phoneme error rate definitions are the same. It's just that the list of elements is here a list of phones instead of a list of words
        per= wer(' '.join(truth), ' '.join(phones_pred))
        PERs.append(per)
        preds.append(' '.join(phones_pred))
    df_sample['predictions']=preds
    df_sample['PER']=PERs
    return df_sample

def determine_start_end_vectors(df_segmented, time_per_output=0.02):
    df_segmented["end_vector"]=(df_segmented["end"]*(1/time_per_output)).astype(int)
    df_segmented["start_vector"]=(df_segmented["start"]*(1/time_per_output)).astype(int)
    return df_segmented

def phone_vectors(s, fs, df_segmented, processor, model, time_per_output=0.02, phone_type='phones', extractor_function=get_last_hidden_state):
    df_segmented=determine_start_end_vectors(df_segmented, time_per_output=time_per_output)
    
    #new dataframe to save results
    new_df= pd.DataFrame()
    new_df['phoneme']=remove_stress_annots(df_segmented[phone_type])

    #compute logits or hidden states
    hidden_vectors=np.array(extractor_function(s, fs, processor, model))
    
    frames_list=[]
    average_vector_list=[]
    for _,row in df_segmented.iterrows():
        target_vectors=hidden_vectors[0,row['start_vector']:row['end_vector']]
        frames_list.append(target_vectors)
        average_vector=np.mean(target_vectors, axis=0)
        average_vector_list.append(average_vector)
    new_df["average_vector"]=average_vector_list
    new_df["vector_sequence"]=frames_list
    
    return new_df

def phone_concat_vectors(s, fs, df_segmented, processor, model, time_per_output=0.02, phone_type='phones', extractor_function=get_last_hidden_state):
    df_segmented=determine_start_end_vectors(df_segmented, time_per_output=time_per_output)
    #compute logits or hidden states
    hidden_vector=np.array(extractor_function(s, fs, processor, model))
    df_vector_list=[]
    for _,row in df_segmented.iterrows():
        target_vector=hidden_vector[0,row['start_vector']:row['end_vector']]
        N=target_vector.shape[0]
        
        #new dataframe to save results
        new_df= pd.DataFrame()
        new_df['phoneme']=remove_stress_annots([row[phone_type]]*N)
        new_df['vector']=list(target_vector)
        df_vector_list.append(new_df)
    
    df_vectors=pd.concat(df_vector_list)
    df_vectors.reset_index(drop=True)
    
    return df_vectors

# creates dataframe from an audio sample containing frames hidden vector and silence hidden vector
def phone_concat_vectors_with_silence(s, fs, df_segmented, processor, model, time_per_output=0.02, phone_type='phones', extractor_function=get_last_hidden_state):
    df_segmented=determine_start_end_vectors(df_segmented, time_per_output=time_per_output)
    #compute logits or hidden states
    hidden_vector=np.array(extractor_function(s, fs, processor, model))
    df_vector_list=[]

    # add silences to dataframe
    target_vector=hidden_vector[0,:df_segmented.iloc[0].start_vector-1]
    N=target_vector.shape[0]
    new_df= pd.DataFrame()
    new_df['phoneme']=['[SIL]']*N
    new_df['vector']=list(target_vector)
    df_vector_list.append(new_df)

    for _,row in df_segmented.iterrows():
        target_vector=hidden_vector[0,row['start_vector']:row['end_vector']]
        N=target_vector.shape[0]
        
        #new dataframe to save results
        new_df= pd.DataFrame()
        new_df['phoneme']=remove_stress_annots([row[phone_type]]*N)
        new_df['vector']=list(target_vector)
        df_vector_list.append(new_df)
    
    df_vectors=pd.concat(df_vector_list)
    df_vectors.reset_index(drop=True)
    
    return df_vectors
    
def instances_per_phoneme(df_t, processor, model, number_of_examples=100, time_per_output=0.02, phone_type='cmu_phone', extractor_function=get_last_hidden_state):
    average_vectors=[]
    if number_of_examples==None: number_of_examples=len(df_t)

    for i in tqdm(range(number_of_examples)):
        # Obtain the information about the start and end of the sentence
        df_segmented=pd.DataFrame.from_records(df_t.iloc[i].phone_df)
        s,fs=librosa.load(df_t.iloc[i].wav_path, sr=16000)

        new_df=phone_vectors(s, fs, df_segmented, processor, model, time_per_output=time_per_output, phone_type=phone_type, extractor_function=extractor_function)
        
        
        if 'language_code' in df_t.iloc[i].keys():new_df['language_code']=df_t.iloc[i]['language_code']
        if 'genre' in df_t.iloc[i].keys():new_df['genre']=df_t.iloc[i]['genre']
        if 'speaker' in df_t.iloc[i].keys():new_df['speaker']=df_t.iloc[i]['speaker']
        average_vectors.append(new_df)
    
    df_all_instances=pd.concat(average_vectors)
    return df_all_instances

def instances_per_frame(df_t, processor, model, number_of_examples=100, time_per_output=0.02, phone_type='cmu_phone', extractor_function=get_last_hidden_state):
    vectors=[]
    if number_of_examples==None: number_of_examples=len(df_t)

    for i in tqdm(range(number_of_examples)):
        # Obtain the information about the start and end of the sentence
        df_segmented=pd.DataFrame.from_records(df_t.iloc[i].phone_df)
        s,fs=librosa.load(df_t.iloc[i].wav_path, sr=16000)

        #new_df=phone_concat_vectors(s, fs, df_segmented, processor, model, time_per_output=time_per_output, phone_type=phone_type, extractor_function=extractor_function)
        new_df=phone_concat_vectors_with_silence(s, fs, df_segmented, processor, model, time_per_output=time_per_output, phone_type=phone_type, extractor_function=extractor_function)
        
        
        if 'language_code' in df_t.iloc[i].keys():new_df['language_code']=df_t.iloc[i]['language_code']
        if 'genre' in df_t.iloc[i].keys():new_df['genre']=df_t.iloc[i]['genre']
        if 'speaker' in df_t.iloc[i].keys():new_df['speaker']=df_t.iloc[i]['speaker']
        if 'filename' in df_t.iloc[i].keys():new_df['filename']=df_t.iloc[i]['filename']
        vectors.append(new_df)
    
    df_all_frames=pd.concat(vectors)
    return df_all_frames

def plot_reduction(df_all_instances, reduction_technique='umap', base_name='wav2vec', legend_label='phoneme'):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    if reduction_technique=='umap':
        reducer = umap.UMAP(n_components=2)
    elif reduction_technique=='tsne':
        reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    elif reduction_technique=='pca':
        reducer = PCA(n_components=2)
    
    np_vectors=np.array(df_all_instances.average_vector.tolist())
    embedding = reducer.fit_transform(np_vectors)

    plt.clf()


    df_subset=pd.DataFrame()
    df_subset['x'] = embedding[:,0]
    df_subset['y'] = embedding[:,1]
    df_subset[legend_label]=df_all_instances[legend_label].tolist()
    # plt.figure(figsize=(16,10))
    plot_result=sns.scatterplot(
        x="x", y="y",
        hue=legend_label,
        # palette=sns.color_palette("hls", 39),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.legend([],[], frameon=False)
    one_example_per_label=df_subset.drop_duplicates(subset=[legend_label])
    for i, txt in enumerate(one_example_per_label[legend_label]):
        plt.annotate(txt, (one_example_per_label['x'].iloc[i], one_example_per_label['y'].iloc[i]))
    plt.savefig(base_name+'_phones_'+reduction_technique+'.png')
    return plot_result



def use_tests():
    # load model and processor
    # I download first, do "from_pretrained(path)" instead to know easier where they are and access the vocabs, config etc.
    # Else models are in: "~/.cache/huggingface/transformers"
    processor = Wav2Vec2Processor.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    base_model=Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53", output_hidden_states=True)
    base_model_ft=Wav2Vec2Model.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", output_hidden_states=True)
    model = Wav2Vec2ForCTC.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", output_hidden_states=True)

    from src.libri_phonetization_data import libri_phonetics_data
    df_t, df=libri_phonetics_data(data_set='dev-clean')

    df_test=instances_per_phoneme(df_t, processor, model, number_of_examples=1, time_per_output=0.02, phone_type='cmu_phone', extractor_function=get_last_hidden_state)

    df_test=instances_per_frame(df_t, processor, model, number_of_examples=1, time_per_output=0.02, phone_type='cmu_phone', extractor_function=get_last_hidden_state)
    
    # Inference with one example

    example=df.iloc[0]
    path=example.wav_path
    print(example.phones)
    # Loas an audio file
    s,fs=librosa.load(path, sr=16000)
    df_segmented=pd.DataFrame.from_records(df_t.iloc[0].phone_df)
    df_test=phone_vectors(s, fs, df_segmented, processor, model, time_per_output=0.02, phone_type='phone', extractor_function=get_last_hidden_state)


    # # This is just to show that "feature_extractor" here is just a normalization with some ratio / filtering
    # # output has the same shape as s
    # s1=feature_extractor(torch.tensor(s), sampling_rate=fs, return_tensors="pt")
    # # the ratio is almost constant (there might be a filtering)
    # c=s1.input_values/s

    timed_pred, phones=inference(s, fs, processor, model)
    print(timed_pred)
    print(len(timed_pred))
    print(phones)

    logits=get_logits(s, fs, processor, model)
    print(logits.shape)

    last_hidden_state=get_last_hidden_state(s, fs, processor, model)
    print(last_hidden_state)
    print(last_hidden_state.shape)
    
    # this is the base model of the finetuned model and therefore outputs the same last hidden state
    last_hidden_state=get_last_hidden_state(s, fs, processor, model=base_model_ft)
    print(last_hidden_state)
    print(last_hidden_state.shape)
    last_hidden_state=get_last_hidden_state(s, fs, processor, model=base_model)
    print(last_hidden_state)
    print(last_hidden_state.shape)
    
    print(model._modules.keys())
    model._modules['wav2vec2']._modules.keys()
    model._modules['wav2vec2']._modules['feature_projection']


    # Inference from different datasets

    df_t, df=libri_phonetics_data(data_set='dev-clean')
    df['phonetics']=df['phones'] #just different names...
    df['path']=df['wav_path'] #just different names...

    df_sample=df[:10]

    df_sample=compute_predictions_and_PER(df_sample)
    
    # -----------------------------
    from src.wav2vec2_utils import instances_per_phoneme, plot_reduction
    from src.libri_phonetization_data import libri_phonetics_data
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Model
    
    df_t, df=libri_phonetics_data(data_set='dev-clean')

    processor = Wav2Vec2Processor.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", output_hidden_states=True)
    df_all_instances=instances_per_phoneme(df_t, processor, model, number_of_examples=None, time_per_output=0.02,  phone_type='cmu_phone')

    df_all_instances=instances_per_phoneme(df_t, processor, model, number_of_examples=10, time_per_output=0.02,  phone_type='cmu_phone', extractor_function=get_logits)
    df_all_instances=instances_per_phoneme(df_t, processor, model, number_of_examples=None, time_per_output=0.02,  phone_type='cmu_phone', extractor_function=get_logits)
    plot_reduction(df_all_instances, reduction_technique='umap', base_name='wav2vec_logits')

    # define vectorized sigmoid
    sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    df_all_instances.apply(lambda r: sigmoid(r.average_vector), axis=1)

    
    
