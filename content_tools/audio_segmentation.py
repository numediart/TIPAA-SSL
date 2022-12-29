import pandas as pd
import os
import librosa
import re

import torch
import numpy as np
import ctc_segmentation
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

import soundfile as sf

from tqdm import tqdm

from Bio import pairwise2
from time import time

SAMPLERATE = 16000
WRITING_SAMPLERATE=44100

from transformers import Wav2Vec2ProcessorWithLM




# load model, processor and tokenizer
# model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
# model_name="facebook/wav2vec2-large-960h-lv60-self"
model_name="facebook/wav2vec2-base-960h"

processor = Wav2Vec2Processor.from_pretrained(model_name)
# processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name)


# https://www.biostars.org/p/246408/
# 
def words_to_unicode_chars(LISTA, LISTB):
    charcode=ord(u"一")-1
    LATtoHAN={}
    HANtoLAT={}
    LISTA_=[]
    LISTB_=[]
    for x in LISTA:
        if x in LATtoHAN.keys():
            LISTA_.append(LATtoHAN[x])
        else:
            charcode+=1
            LATtoHAN[x]=chr(charcode)
            HANtoLAT[chr(charcode)]=x
            LISTA_.append(LATtoHAN[x])
    for x in LISTB:
        if x in LATtoHAN.keys():
            LISTB_.append(LATtoHAN[x])
        else:
            charcode+=1
            LATtoHAN[x]=chr(charcode)
            HANtoLAT[chr(charcode)]=x
            LISTB_.append(LATtoHAN[x])

    LISTA__="".join(LISTA_)
    LISTB__="".join(LISTB_)

    return LISTA__, LISTB__, LATtoHAN, HANtoLAT

# def unicode_chars_to_words(seqA, seqB, LATtoHAN, HANtoLAT):
#     RESA=[]
#     RESB=[]
#     for w in alignments[0][0]:
#         if (w == "-"):
#             RESA.append("-")
#         else:
#             RESA.append(HANtoLAT[w])
#     for w in alignments[0][1]:
#         if (w == "-"):
#             RESB.append("-")
#         else:
#             RESB.append(HANtoLAT[w])
#     return RESA, RESB

# LISTA__, LISTB__, LATtoHAN, HANtoLAT = words_to_unicode_chars(LISTA, LISTB)
# alignments=pairwise2.align.globalxx(LISTA__,LISTB__)
# RESA, RESB=unicode_chars_to_words(alignments[0].seqA, alignments[0].seqB, LATtoHAN, HANtoLAT)



def get_word_timestamps(
    audio : np.ndarray,
    model : Wav2Vec2ForCTC,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer,
    samplerate : int = SAMPLERATE
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
    

    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)

    # This is to use the LM processor
    # pred_transcript = processor.decode(logits.numpy()).text

    
    # Split the transcription into words
    words = pred_transcript.split(" ")
    
    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    return [{"text" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(words, segments)]


def build_all_words_df(audio, model, report_callback=None):
    n_samples_per_batch=20*SAMPLERATE
    last_time=0
    start=0
    end=n_samples_per_batch
    dfs=[]

    print("duration:",len(audio)/SAMPLERATE)
    
    if report_callback: report_callback(0, 'Analyzing '+report_callback.file)

    while end<len(audio):
        df_words=pd.DataFrame.from_records(get_word_timestamps(audio[start:end], model=model))
        # as these times are relative to the audio I give, I have to add the "last_time" to it corresponding to start index
        df_words[['start','end']]+=last_time

        dfs.append(df_words)
        # to start from a word that was not cut in two, and have an overlap, take the ante previous of the last one
        last_time=df_words.end.iloc[-3]
        start=int(last_time*SAMPLERATE)
        end=start+n_samples_per_batch

        # print(df_words)
        print("last time:",last_time)
        print("duration:",len(audio)/SAMPLERATE)
        if report_callback: report_callback(int(last_time/(len(audio)/SAMPLERATE)*100), 'Analyzing '+report_callback.file)

    last_time=df_words.end.iloc[-3]
    start=int(last_time*SAMPLERATE)
    df_words=pd.DataFrame.from_records(get_word_timestamps(audio[start:end], model=model))
    df_words[['start','end']]+=last_time
    dfs.append(df_words)

    if report_callback: report_callback(100, 'Analyzing '+report_callback.file)


    all_df_words=pd.concat(dfs)

    all_df_words['reverse_end_cummin']=all_df_words[::-1].end.cummin()[::-1]
    all_df_words=all_df_words[all_df_words.end==all_df_words.reverse_end_cummin]
    
    # check if start is bigger than any end before, if not remove. Put a zero instead of the nan introduced because of the shift to neutralize the effect of the first row
    # drop_overlapping_rows = lambda df : df.loc[(df['start'] >= df.end.cummax().shift().fillna(0)) , :]
    # all_df_words=drop_overlapping_rows(all_df_words)

    return all_df_words


def get_percentile(vector, n):
    if n < 1 or n > 100:
        print("Invalid percentile")
    else:
        vector.sort()
        index = (n/100) * len(vector)
        if index % 1 == 0:
            print(vector[int(index) - 1])
        else:
            print((vector[int(index) - 1] + vector[int(index)])/2)


def group_words(all_df_words):
    all_df_words=all_df_words.reset_index(drop=True)
    gaps=all_df_words.start-all_df_words.end.shift()
    # gaps.dropna().median()
    # np.histogram(gaps.dropna())
    gaps[gaps>0.02].index

    last_start_index=0
    phrases=[]
    for i in gaps[gaps>0.02].index:
        phrase=all_df_words[last_start_index:i]
        phrases.append(phrase)
        last_start_index=i
    phrases.append(all_df_words[last_start_index:])
    
    
    # audio_offset_start=0.1
    # audio_offset_end=0.5

    phrase_cutting_data=[]
    print(len(phrases))
    for phrase_df in tqdm(phrases):
        if len(phrase_df)>0:
            start=phrase_df.start.iloc[0]
            end=phrase_df.end.iloc[-1]
            phrase_cutting_data.append({'text':' '.join(phrase_df.text.str.lower().tolist()), 'start':start, 'end':end})

    phrase_cutting_data_df=pd.DataFrame.from_records(phrase_cutting_data)


    return phrase_cutting_data_df


def count_characters(string, character):
  count = 0
  for c in string:
    if c == character:
      count = count + 1
  return count


def extract_timings(df, audio_path, model, report_callback=None):
    
    # Run ASR to get words
    print('Loading audio')
    t=time()
    audio,fs=librosa.load(audio_path, sr=SAMPLERATE)
    print('finished loading, took ',time()-t,' seconds')
    if report_callback: report_callback(100, 'Finished loading, took '+str(round((time()-t),2))+' seconds')
    print('Speech recognition in audio')
    t=time()
    all_df_words=build_all_words_df(audio, model=model, report_callback=report_callback)
    print('finished analyzing, took ',time()-t,' seconds')
    # to release some memory for now
    del audio



    # the ASR works in uppper case
    chars_to_ignore_regex = '[\,\?\.\!\¡\;\:\"\*\{\}]'
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english

    transcripts=df.iloc[:,2].str.lower().tolist()
    transcripts=[re.sub(chars_to_ignore_regex, '',el) for el in transcripts]

    # A = predicted
    # B = transcript

    LISTA=all_df_words.text.str.lower().tolist()
    LISTB=' '.join(transcripts).split(' ')

    print('Alignment on all characters...')
    joined_LISTA=' '.join(LISTA).replace('-','_')
    joined_LISTB=' '.join(LISTB).replace('-','_')
    t=time()
    alignments_chars=pairwise2.align.globalxx(joined_LISTA,joined_LISTB)
    print('finished, took ',time()-t,' seconds')

    print('retieving word parts in transcripts from predicted words')
    t=time()
    start_char=0
    retrieve_B=[]
    for el in alignments_chars[0].seqA.split(' '):
        len(el+' ')
        retrieve_B.append(alignments_chars[0].seqB[start_char:start_char+len(el+' ')])
        start_char+=len(el+' ')
    print('finished, took ',time()-t,' seconds')

    all_df_words['text_predicted_dashed']=alignments_chars[0].seqA.split(' ')
    all_df_words['text_transcript_dashed']=retrieve_B
    all_df_words['text_transcript_undashed']=all_df_words.text_transcript_dashed.str.replace('-','').replace('_','-')

    assert ''.join(all_df_words.text_transcript_dashed.tolist()).replace('-','').replace('_','-')==' '.join(transcripts)

    # build list of original IDs thanks to number of words in each sentence
    df['norm_text']=transcripts
    df['char_cumsum']=(df.norm_text+' ').str.len().cumsum()
    all_df_words['char_cumsum']=all_df_words['text_transcript_undashed'].str.len().cumsum()
    last_cumsum=0
    phrases_dfs={}
    for i,r in df.iterrows():
        phrases_dfs[r.id]=all_df_words[(all_df_words.char_cumsum>last_cumsum)&(all_df_words.char_cumsum<=r.char_cumsum)]
        last_cumsum=r.char_cumsum
    
    del all_df_words
    return phrases_dfs

def cutting(df, audio_path, phrases_dfs, out_folder):
    print('Reloading audio for cutting')
    t=time()
    audio,fs=librosa.load(audio_path, sr=WRITING_SAMPLERATE)
    print('finished loading, took ',time()-t,' seconds')

    
    phrase_cutting_data=[]
    not_detected=[]
    # folder_1="data/"+out_folder+'/'
    if not os.path.exists(out_folder):os.makedirs(out_folder)

    
    audio_offset_start=0.25
    audio_offset_end=0.5

    print(len(phrases_dfs))
    for k in tqdm(phrases_dfs): 
        phrase_df=phrases_dfs[k]
        phrase_df=phrase_df.replace('-',np.nan).dropna()
        if len(phrase_df)>0:
            start=phrase_df.start.iloc[0]
            end=phrase_df.end.iloc[-1]
            try:
                text=df[df.id==k].norm_text.iloc[0].replace(' ','_')
            except:import pdb;pdb.set_trace()

            sf.write(out_folder+'/'+k+'-'+text+'.wav',audio[int((start+audio_offset_start)*WRITING_SAMPLERATE):int((end+audio_offset_end)*WRITING_SAMPLERATE)], WRITING_SAMPLERATE)

            dashed_text=''.join(phrase_df.text_transcript_dashed.tolist())
            text_predicted_dashed=' '.join(phrase_df.text_predicted_dashed.tolist())
            phrase_cutting_data.append({'id':k, 'text':df[df.id==k].norm_text.iloc[0], 'start':start, 'end':end, 'mismatch_rate':count_characters(dashed_text, '-')/len(dashed_text), 'mismatch_text':dashed_text, 'text_predicted_dashed':text_predicted_dashed})
        else:
            not_detected.append(k)
    
    phrase_cutting_data=pd.DataFrame.from_records(phrase_cutting_data)

    return phrase_cutting_data, not_detected


import io
def write_string_to_file(string, file):
    """Write a string to a file with a path or a file-like object.
    
    Args:
        string: the string to write to the file
        file: a file path or a file-like object
    """
    if isinstance(file, str):
        # file is a path, open the file in write mode to get a file-like object
        file_obj = open(file, 'w')
    else:
        # file is a file-like object
        file_obj = file
        
    if isinstance(file_obj, io.BytesIO):
        # file is a io.BytesIO object, convert the string to bytes before writing
        data = string.encode()
    else:
        # file is a file-like object that works with strings, write the string as is
        data = string
        
    file_obj.write(data)
    
    # close the file if it was opened by the function
    if isinstance(file, str):
        file_obj.close()

def create_srt(df, file="output.srt"):
    """Write text with timings to a  srt file with a path or a file-like object.
    
    Args:
        df: the dataframe with text, start, end columns
        file: a file path or a file-like object
    """

    lines=[]
    for index, row in df.iterrows():
        start = row["start"]
        end = row["end"]
        text = row["text"]
        #convert start and end to HH:MM:SS,MS format
        start_h = int(start/3600)
        start_m = int((start%3600)/60)
        start_s = int((start%3600)%60)
        start_ms = int((start%1)*1000)
        end_h = int(end/3600)
        end_m = int((end%3600)/60)
        end_s = int((end%3600)%60)
        end_ms = int((end%1)*1000)

        #write to file
        lines.append(str(index+1)+"\n")
        lines.append(str(start_h).zfill(2)+":"+str(start_m).zfill(2)+":"+str(start_s).zfill(2)+","+str(start_ms).zfill(3)+" --> "+str(end_h).zfill(2)+":"+str(end_m).zfill(2)+":"+str(end_s).zfill(2)+","+str(end_ms).zfill(3)+"\n")
        lines.append(text+"\n\n")
    
    data=''.join(lines)
    write_string_to_file(data, file)


def analyze_files_and_build_transcripts(model, file_dict, report_callback=None, results_dir='./results/'):


    xlsx_files=[el for el in file_dict if el.endswith('.xlsx')]
    if len(xlsx_files)!=1: 
        print('There should be exactly 1 xlsx file, but there is/are '+str(len(xlsx_files)))
    else:
        xlsx_file=xlsx_files[0]
        
    dir,_=os.path.split(xlsx_file)
    sheet_df_map = pd.read_excel(file_dict[xlsx_file], sheet_name=None)

    first_pass_dfs={}
    for k in sheet_df_map:
        if report_callback:
            report_callback.file=k
            report_callback(0, 'Loading '+k)
        df=sheet_df_map[k]

        audio_path=dir+'/'+k+'.wav'

        timings = extract_timings(df, file_dict[audio_path], model=model, report_callback=report_callback)
        # report_callback(100, 'Loading '+k)
        if report_callback: report_callback(0, 'Cutting '+k)
        # Reset the position of the file pointer to the beginning of the file
        file_dict[audio_path].seek(0)
        phrase_cutting_data_df, not_detected= cutting(df, file_dict[audio_path], timings, results_dir)
        if report_callback: report_callback(100, 'Cutting '+k)
        first_pass_dfs[k]=phrase_cutting_data_df
        srt_path=results_dir+'/'+k+'.srt'
        try:
            create_srt(phrase_cutting_data_df, srt_path)
        except:
            print(phrase_cutting_data_df)
            print(srt_path)
            raise('failed in create_srt()')
            # import pdb;pdb.set_trace()

    all_first_pass_df=pd.concat(first_pass_dfs.values())
    all_first_pass_df['audio_file']=sum([len(first_pass_dfs[k])*[k] for k in first_pass_dfs],[])

    # all_second_pass_df=pd.concat(second_pass_dfs.values())
    # all_second_pass_df['audio_file']=sum([len(second_pass_dfs[k])*[k] for k in first_pass_dfs],[])
    # all_third_pass_df=pd.concat(third_pass_dfs.values())
    # all_third_pass_df['audio_file']=sum([len(third_pass_dfs[k])*[k] for k in first_pass_dfs],[])

    all_first_pass_df.round(2).to_csv(results_dir+'/timed_transcriptions.csv',index=False)
    all_first_pass_df.round(2).to_excel(results_dir+'/timed_transcriptions.xlsx',index=False)

    # all_second_pass_df.round(2).to_csv('data/andrew_second_pass.csv',index=False)
    # all_second_pass_df.round(2).to_excel('data/andrew_second_pass.xlsx',index=False)
    # all_third_pass_df.round(2).to_csv('data/andrew_third_pass.csv',index=False)
    # all_third_pass_df.round(2).to_excel('data/andrew_third_pass.xlsx',index=False)

    # all_ids_processed=set(all_first_pass_df.id).union(set(all_second_pass_df.id))

    all_ids=set().union(*[set(sheet_df_map[k].id) for k in sheet_df_map])

    # not_processed_ids=all_ids-all_ids_processed



if __name__=="__main__":
    model=Wav2Vec2ForCTC.from_pretrained(model_name)
    # ------------- without trancscript
    filename="Y2Mate.is - C2W - Prof. Philippe Dubois, RectorPresident of UMONS-_1_dGs28JMQ-720p-1659731395421.mp4"
    audio_path="data/"+filename
    audio,fs=librosa.load(audio_path, sr=SAMPLERATE)
    all_df_words=build_all_words_df(audio, model=model)
    phrases_df=group_words(all_df_words)
    create_srt(phrases_df, file="data/"+filename+".srt")

    # ------------- one big file of Jessie
    filename='Jessie_Legal_English_Task_1'
    df=pd.read_csv('data/'+filename+'.csv')
    # the dropna is because in the original spreadsheet some rows were only a title, so no value for text
    df=df[~df.iloc[:,2].isnull()]

    audio_path='data/Flowchase - Legal English - Task 1.wav'

    phrase_cutting_data_df = extract_timings(df, audio_path, model=model)
    cutting(df, audio_path, phrase_cutting_data_df, 'jessie')

    create_srt(phrase_cutting_data_df)

    set(df.id)-set(phrase_cutting_data_df.id)

    # df[df.id.isin(not_detected)]

    # --------------------
    