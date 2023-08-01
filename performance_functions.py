from cProfile import label
from src.libri_phonetization_data import build_librispeech_words_df
from src.libri_phonetization_data import phonetics_for_row, select_libri
from tqdm import tqdm
import pandas as pd
import ast
from collections import Counter

from src.label_data_processing import build_user_data_df
# exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')

from DL_speech_tech import schwa_sound_from_formatted_phonetics_audio, phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, start_end_contrast_from_formatted_phonetics_audio, default_model#, default_model_charsiu

from src.audio_processing import read_audio_file

from src.audio_processing import prepare_audio_file

from src.label_data_processing import get_data_stressed_content, actor_recordings, synth_words_data
from src.text_processing import *
from src.text_processing import word_stress_from_cmu
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants, cmu_phones, cmu_alphabet, ipa_alphabet

from src.libri_phonetization_data import *

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
# disable pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

def formatted_audiobook_data(selection, libri_words_df, target_phones=None):
    # retrieve phonetics by word thanks to 'phonetics_fot_row'
    selection['split_phonetics']=selection.apply(lambda r: [p.split(' ') for p in phonetics_for_row(r, libri_words_df)], axis=1)
    # Generate formatted_phonetics by syllabifying each word with sonoripy, then joining phonemes, syllables and words with separators
    selection['cmu_phonetics']=selection.apply(lambda r: ' '.join(['|'.join(['_'.join(syl) for syl in SonoriPy(w)[0]]) for w in r.split_phonetics]), axis=1)
    
    # I figured out there are empty elements resulting in double space in formatted phonetics. This is problematic. 
    # As a fix, I insert a "AH0" to have something very neutral close to silence
    selection['cmu_phonetics']=selection.apply(lambda r: r['cmu_phonetics'].replace('  ', ' AH0 '), axis=1)

    selection['target_word_indexes']=selection['word_idx']
    selection['fpath']=selection['wav_path']

    # to extract a target_syllable_index, I use a function that find the first element containing a string
    def find_first(lst, predicate): return next((i for i,j in enumerate(lst) if predicate(j)), float('nan'))
    find_first_containing_target=lambda l: find_first(l, lambda x: target_phones in x.split('_'))

    if target_phones is not None:
        selection['target_syllable_indexes']=selection.apply(lambda r: find_first_containing_target(r['cmu_phonetics'].split(' ')[r['word_idx']].split('|')), axis=1)
        # find_first_containing_target(phonetics.split(' ')[target_word_idx].split('|'))
    else:
        # this is for termination contrasts
        selection['target_syllable_indexes']=-1
    return selection

def count_values(phonetic_detections):
    d=Counter(phonetic_detections)
    d = pd.DataFrame.from_dict(d, orient='index')
    if len(d)>0:
        d=d.sort_values(by=0, ascending=False)
        d=d / d.sum()*100
    return d

from src.code_utils import internal_error

def compute_predictions(selection, target_phones='AO1', tech_function=phonemeContrast_from_formatted_phonetics_audio, 
                        basis=None, alternatives=cmu_vowels, 
                        model=default_model,
                        **kwargs):
    phonetic_detections=[]
    records=[]
    errors_data=[]
    print('number of examples:', len(selection))
    if len(selection)>0:
        for i,r in tqdm(selection.iterrows()):
            if type(r.target_word_indexes)==str:
                target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
                target_syllable_idx=ast.literal_eval(r.target_syllable_indexes)[0]
            else:
                target_word_idx=r.target_word_indexes
                target_syllable_idx=r.target_syllable_indexes
            try:
                s,fs=read_audio_file(r.fpath, fs=16000)
            except Exception as e: 
                print('error in reading audio in compute_predictions')
                print('row information')
                print(r)
                print(e)
            try:
                # this is only for start_end_contrasts, I am putting a default if does not exist
                contrast = r.contrast if 'contrast' in r else 'end'
                res=tech_function(s,phonetics=r.cmu_phonetics,target_word_idx=target_word_idx,target_syllable_idx=target_syllable_idx,target_occurence_idx=0,target_phones=target_phones,basis=basis,contrast=contrast,alternatives=alternatives,mode='numpy',model=model,**kwargs)
                phonetic_detections.append(res['phonetic_detection'])
                records.append(res)
            except Exception as e: 
                print('error in the tech_function in compute_predictions')
                print('row information')
                print(r)
                print(e)
                print(internal_error())
                error_data=internal_error()
                # import pdb;pdb.set_trace()
                errors_data.append(error_data)
                
        result_df=pd.DataFrame.from_records(records)
    else:
        print('Selection to compute_prediction is empty')
        result_df=pd.DataFrame()
    
    print("errors that occured:")
    print(errors_data)
    return result_df




def stress_GE_performance_test(level='sentence'):
    df=get_data_stressed_content()
    stress_intensities=[]
    stress_binaries=[]

    df['phonetics']=df.text.apply(lambda r: prefill_for_sentence(r)['phonetics'])
    df=df[~df['phonetics'].str.contains('{')]

    print('n rows:',len(df))
    for i,row in tqdm(df.iterrows()):
        s,fs=read_audio_file(row.audio_path, fs=16000)
        
        n_words_by_chunk=chunk_text(text=row.text)
        res=stress_from_formatted_phonetics(s,phonetics=row.phonetics, n_words_by_chunk=n_words_by_chunk, level=level, mode="numpy")
        print(res)
        stress_intensities.append(res['stress_intensities'])
        stress_binaries.append(res['stress_binaries'])
    df['stress_intensities']=stress_intensities
    df['stress_binaries']=stress_binaries

    if level=='sentence':
        # Here we filter out sentences with more than a stressed word
        sum_df=df.bins.apply(lambda r:sum(r))
        df=df[sum_df==1]
        
        # get the index of the chunk containing the stress, which is the "chunk of interest"
        n_word_by_chunk=df.text.apply(lambda r:chunk_text(r))
        cumsum=n_word_by_chunk.apply(lambda r:np.cumsum([0]+r))
        idx_1=df.bins.apply(lambda r:r.index(1))

        def get_range(cumsum,idx_1):
            range_df=pd.concat([cumsum,idx_1], axis=1)
            ranges=[]
            for i,r in range_df.iterrows():
                idx=r.bins
                ns=r.text
                for i,n in enumerate(ns[::-1]):
                    if idx>=n: 
                        range_of_i=[r.text[len(ns)-i-1],r.text[len(ns)-i]]
                        ranges.append(range_of_i)
                        break
            range_df['ranges']=ranges
            return range_df
        
        # get the range in terms of words for the "chunk of interest"
        range_df=get_range(cumsum,idx_1)
        df_all=pd.concat([df, range_df[['ranges']]], axis=1)

        # COI = Chunk Of Interest
        df_all['COI_stress_intensities']=df_all.apply(lambda r: r['stress_intensities'][r.ranges[0]:r.ranges[1]], axis=1)
        df_all['COI_bins']=df_all.apply(lambda r: r['bins'][r.ranges[0]:r.ranges[1]], axis=1)
        df_all['COI_stress_binaries']=df_all.apply(lambda r: r['stress_binaries'][r.ranges[0]:r.ranges[1]], axis=1)

        all_bins=sum(df_all.COI_bins.tolist(),[])
        rate_of_1=sum(all_bins)/len(all_bins)
        print('rate_of_1',rate_of_1)

        all_bins_pred=sum(df_all.COI_stress_binaries.tolist(),[])
        rate_of_1_pred=sum(all_bins_pred)/len(all_bins_pred)
        print('rate_of_1_pred',rate_of_1_pred)

        failures=df_all[df_all.COI_stress_intensities.apply(lambda r: len(r))==0]
        rate_of_failures=len(failures)/len(df_all)

        df_all=df_all[df_all.COI_stress_intensities.apply(lambda r: len(r))!=0]

        # check if the 1 in COI_bins was predicted as 1 inside COI_stress_binaries. Let's call it the "Binary of Interest" or BOI
        BOI=df_all.apply(lambda r:r['COI_stress_binaries'][r['COI_bins'].index(1)], axis=1)
        rate_BOI=BOI.sum()/len(BOI)
        print('rate_BOI',rate_BOI)
        return df_all
    elif level=='word':
        split_phonetics=lambda phonetics: [p.replace('|','_').split('_') for p in phonetics.split(' ')]


        df.index=range(len(df))
        word_stress_binaries=df.phonetics.apply(lambda r: [word_stress_from_cmu(p) for p in split_phonetics(r)])

        # I realized some words have all "syllables" stressed (even though they are > 1 syl). In fact, it corresponds to acronyms
        # here I remove them
        
        # To see them:
        # word_stress_binaries[word_stress_binaries.apply(lambda r: sum([(np.prod(el)==1)&(len(el)>1) for el in r]))>0]

        # only keep whn it's not the case
        word_stress_binaries=word_stress_binaries[word_stress_binaries.apply(lambda r: sum([(np.prod(el)==1)&(len(el)>1) for el in r]))==0]
        df=df.loc[word_stress_binaries.index,:]

        # word_stress_binaries[word_stress_binaries.apply(lambda r: [1,1] in r)]
        # word_stress_binaries[word_stress_binaries.apply(lambda r: [1,1,1,1] in r)]

        def select_by_len(stress_binaries):
            all_word_bins=sum(stress_binaries.tolist(),[])
            max_len=max([len(el) for el in all_word_bins])
            min_len=min([len(el) for el in all_word_bins])
            print('min_len',min_len)
            print('max_len',max_len)
            all_word_bins_by_len=[]
            for l in range(max_len):
                all_word_bins_by_len.append([el for el in all_word_bins if len(el)==l+1])
            return all_word_bins_by_len
        
        # filter out examples with different number of words
        # see them:
        # (word_stress_binaries.apply(lambda r: len(r))!=df['stress_binaries'].apply(lambda r: len(r))).sum()
        word_stress_binaries=word_stress_binaries[word_stress_binaries.apply(lambda r: len(r))==df['stress_binaries'].apply(lambda r: len(r))]
        df=df.loc[word_stress_binaries.index]

        assert len(sum(word_stress_binaries.tolist(),[])) == len(sum(df['stress_binaries'].tolist(),[]))

        # # When I don't exclude compound words (containing "{" and "}"), this will show the problematic of empty and non-consistent number of words
        # word_stress_binaries[word_stress_binaries.apply(lambda r: [] in  r)]
        # df.text[word_stress_binaries.apply(lambda r: [] in  r)]
        # df.phonetics[word_stress_binaries.apply(lambda r: [] in  r)]

        GT_by_len=select_by_len(word_stress_binaries)
        preds_by_len=select_by_len(df['stress_binaries'])

        # these should be the same
        print([len(el) for el in GT_by_len])
        print([len(el) for el in preds_by_len])

        for l, (GT,pred) in enumerate(zip(GT_by_len, preds_by_len)):
            error_rate=sum(sum(np.abs(np.array(GT)-np.array(pred))))/np.prod(np.array(pred).shape)
            print('words of len '+str(l+1)+' error rate:'+str(error_rate))

def pContrast_for_user_data( target_phones='AO1', n=100, model=default_model):
    user_data=build_user_data_df()
    selection=user_data[user_data.target_phoneme==target_phones]
    selection['audio_file_url']=selection['fpath']
    selections=[]
    n_user=50
    for u in selection.user_id.unique()[:n_user]:
        for ex in selection.exercise_id.unique():
            selections.append(selection[selection.user_id==u][selection.exercise_id==ex])
    df=pd.concat(selections)

    df=df.sample(frac=1, random_state=1234)[:n]

    df['split_phonetics']=df.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)

    result_df=compute_predictions(df, target_phones=target_phones, model=model)
    phonetic_detections=result_df.phonetic_detection

    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return result_df, d


def final_ed_for_user_data( target_phones='D', n=100, model=default_model):
    user_data=build_user_data_df()
    selection=user_data[user_data.target_phoneme==target_phones]
    
    selection['audio_file_url']=selection['fpath']
    selections=[]
    n_user=50
    for u in selection.user_id.unique()[:n_user]:
        for ex in selection.exercise_id.unique():
            selections.append(selection[selection.user_id==u][selection.exercise_id==ex])
    df=pd.concat(selections)

    df=df.sample(frac=1, random_state=1234)[:n]

    df['split_phonetics']=df.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)

    result_df=compute_predictions(df, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio, model=model)
    phonetic_detections=result_df.phonetic_detection

    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return result_df, d

def final_ed_for_actor_recordings(target_phones='D', model=default_model):
    df=actor_recordings()

    df['fpath']=df['audio_file_url']

    selection=df[df.target_phoneme==target_phones]
    result_df=compute_predictions(selection, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio, model=model)
    phonetic_detections=result_df.phonetic_detection

    success_rate=len(result_df[result_df.gibberish_truth==result_df.gibberish_detected])/len(result_df)
    print('errors:',result_df[result_df.gibberish_truth!=result_df.gibberish_detected])
    print(success_rate)
    
    d=count_values(phonetic_detections)
    d.columns=[target_phones]
    return result_df, d

def pContrast_for_actor_recordings(target_phones='AO1', speakers=None, model=default_model):
    df=actor_recordings()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    
    if speakers is not None:
        df_pContrast=df_pContrast[df_pContrast.audio_file_url.apply(lambda r: r.split('/')[-2]).isin(speakers)]
    
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]
    if len(selection)>0:
        
        try:
            selection['split_phonetics']=selection.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)
        except:
            import pdb;pdb.set_trace()
        selection['fpath']=selection.audio_file_url
        # selection['audio_file_idx']=selection.fk_audio_recording_id

        result_df=compute_predictions(selection, target_phones=target_phones, model=model)
        phonetic_detections=result_df.phonetic_detection

        d=count_values(phonetic_detections)
        d.columns=[target_phones]
    else:
        result_df=None
        d=None

    return result_df, d



def pContrast_from_audiobook_data(data_set='test-other', target_phones='AO1', n=None, alternatives=cmu_vowels, model=default_model):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    if target_phones!='G':
        # Select words containing a phone. For 'T', to avoid capturing also 'TH', it has to contain 'T '  or end by ' T'
        # same for D/DH, J/JH etc.
        selection=libri_words_df[libri_words_df.phones.str.endswith(' '+target_phones)|libri_words_df.phones.str.contains(target_phones+' ')]
    else:
        # same idea, but only for G/NG
        selection=libri_words_df[libri_words_df.phones.str.startswith(target_phones+' ')|libri_words_df.phones.str.contains(' '+target_phones)]

    selection=formatted_audiobook_data(selection, libri_words_df, target_phones=target_phones)

    selection['cmu_phonetics']=selection.apply(lambda r: r.cmu_phonetics.replace('CH','T_SH').replace('JH','D_ZH'), axis=1)
    df=selection
    df_words=df.apply(lambda r: r.cmu_phonetics.split(' ')[r.word_idx].replace('|','_'), axis=1)
    df_target=df[(df_words.str.endswith('_'+target_phones)|df_words.str.startswith(target_phones+'_')|df_words.str.contains('_'+target_phones+'_'))]

    result_df=compute_predictions(df_target, target_phones=target_phones, alternatives=alternatives, model=model)
    phonetic_detections=result_df.phonetic_detection
    
    d=count_values(phonetic_detections)
    d.columns=[target_phones]
    return result_df, d

def start_end_phoneme_from_audiobook_data(phoneme='HH', basis='HH', contrast="start",data_set='dev-clean', n=100, model=default_model):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    if contrast=="start":
        idx=0
    elif contrast=="end":
        idx=-1

    words_in_p=[el for el in cmudict_dict.keys() if cmudict_dict[el][0][idx]==phoneme]
    words_in_p=set(words_in_p)
    libri_words_list=libri_words_df.word.unique()
    libri_words_in_p=[el for el in libri_words_list if el in words_in_p]

    selections=[]
    print(len(libri_words_in_p))
    for word in tqdm(libri_words_in_p):
        selection=libri_words_df[libri_words_df.word==word]
        if len(selection)>0:
            selection=formatted_audiobook_data(selection, libri_words_df)
            selections.append(selection)
    selections=pd.concat(selections)

    freq_words=count_values(selections.word.tolist()).index[:9].tolist()
    selections=selections[~selections.word.isin(freq_words)]
    
    if basis!=phoneme:
        target_phones=''
    else:
        target_phones=phoneme

    result_df=compute_predictions(selections, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio, basis=basis, contrast=contrast, model=model)
    phonetic_detections=result_df.phonetic_detection
    
    selections=selections.reset_index(drop=True)
    result_df['cmu_phonetics']=selections['cmu_phonetics']
    result_df['fpath']=selections['fpath']
    d=count_values(phonetic_detections)
    d.columns=[phoneme]
    return phonetic_detections,result_df,d

def final_ed_from_audiobook_data(data_set='test-other', target_phones='D', n=None, model=default_model):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    selection=select_libri(select_libri(libri_words_df,'D', option='endswith'), 'ed', column='word', option='endswith')
    selection_id=select_libri(selection, 'IH0 D', option="endswith")
    selection_d=selection[~selection.index.isin(selection_id.index.tolist())]
    selection_t=select_libri(select_libri(libri_words_df,'T', option='endswith'), 'ed', column='word', option='endswith')

    selections={}
    selections['D']=selection_d
    selections['IH0_D']=selection_id
    selections['T']=selection_t
    
    selection=formatted_audiobook_data(selections[target_phones], libri_words_df)

    result_df=compute_predictions(selection, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio, model=model)
    phonetic_detections=result_df.phonetic_detection

    success_rate=len(result_df[result_df.gibberish_truth==result_df.gibberish_detected])/len(result_df)

    errors_df=result_df[result_df.gibberish_truth!=result_df.gibberish_detected]
    print('errors:',errors_df)

    if len(errors_df)>0: print("last error:"); errors_df.iloc[-1]
    
    print(success_rate)
    selection=selection.reset_index(drop=True)
    result_df['cmu_phonetics']=selection['cmu_phonetics']
    result_df['fpath']=selection['fpath']

    d=count_values(phonetic_detections)
    d.columns=[target_phones]
    return result_df, d

def final_s_from_audiobook_data(data_set='dev-clean', n=None, model=default_model):

    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    # words that finish in "s" with phoneme "S" that also exist without an "s" and with last phoneme then not being "S"
    words_in_s=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='S' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='S']
    words_in_s=set(words_in_s)

    words_in_z=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='Z' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='Z']
    words_in_z=set(words_in_z)

    libri_words_list=libri_words_df.word.unique()
    libri_words_in_s=[el for el in libri_words_list if el in words_in_s]
    libri_words_in_z=[el for el in libri_words_list if el in words_in_z]
    
    # word=libri_words_in_s[3]

    selections_no_s=[]
    selections_s=[]
    print(len(libri_words_in_s))
    for word in tqdm(libri_words_in_s):
        selection, selection_s = selection_with_and_without_s(libri_words_df, word[:-1])

        if len(selection)>0 and len(selection_s)>0:
            selection=formatted_audiobook_data(selection, libri_words_df)
            selection_s=formatted_audiobook_data(selection_s, libri_words_df)

            selections_no_s.append(selection)
            selections_s.append(selection_s)
    
    selections_no_s=pd.concat(selections_no_s)
    selections_s=pd.concat(selections_s)
    
    
    selections_no_z=[]
    selections_z=[]
    for word in tqdm(libri_words_in_z):
        selection, selection_z = selection_with_and_without_s(libri_words_df, word[:-1])

        if len(selection)>0 and len(selection_z)>0:
            selection=formatted_audiobook_data(selection, libri_words_df)
            selection_z=formatted_audiobook_data(selection_z, libri_words_df)

            selections_no_z.append(selection)
            selections_z.append(selection_z)

    selections_no_z=pd.concat(selections_no_z)
    selections_z=pd.concat(selections_z)

    freq_words_no_s=count_values(selections_no_s.word.tolist()).index[:2].tolist()
    freq_words_s=count_values(selections_s.word.tolist()).index[:2].tolist()
    
    freq_words_no_z=count_values(selections_no_z.word.tolist()).index[:2].tolist()
    freq_words_z=count_values(selections_z.word.tolist()).index[:2].tolist()

    selections_no_s=selections_no_s[~selections_no_s.word.isin(freq_words_no_s)]
    selections_s=selections_s[~selections_s.word.isin(freq_words_s)]
    
    selections_no_z=selections_no_z[~selections_no_z.word.isin(freq_words_no_z)]
    selections_z=selections_z[~selections_z.word.isin(freq_words_z)]

    # filter our first ones like "it" or "its" that are too frequent

    # selections_no_s.apply(lambda r: r.cmu_phonetics.split(' '), axis=1)

    # words_added_s=selections_no_s.apply(lambda r: r.cmu_phonetics.split(' ')[r.word_idx]+'_S', axis=1)

    # for fake mistakes, add s termination in formatted_phonetics, even though it's not in audio
    selections_no_s['cmu_phonetics']=selections_no_s.apply(lambda r: ' '.join( r.cmu_phonetics.split(' ')[:r.word_idx] + [r.cmu_phonetics.split(' ')[r.word_idx]+'_S'] + r.cmu_phonetics.split(' ')[r.word_idx+1:]), axis=1)
    selections_no_z['cmu_phonetics']=selections_no_z.apply(lambda r: ' '.join( r.cmu_phonetics.split(' ')[:r.word_idx] + [r.cmu_phonetics.split(' ')[r.word_idx]+'_Z'] + r.cmu_phonetics.split(' ')[r.word_idx+1:]), axis=1)

    selections_no_s.iloc[0]

    result_df_no_s=compute_predictions(selections_no_s[:10], target_phones='S', tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='IH0_S', model=model)
    phonetic_detections_no_s=result_df_no_s.phonetic_detection
    result_df_s=compute_predictions(selections_s, target_phones='S', tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='IH0_S', model=model)
    phonetic_detections_s=result_df_s.phonetic_detection

    
    result_df_no_z=compute_predictions(selections_no_z[:10], target_phones='Z', tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='IH0_Z', model=model)
    phonetic_detections_no_z=result_df_no_z.phonetic_detection
    result_df_z=compute_predictions(selections_z, target_phones='Z', tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='IH0_Z', model=model)
    phonetic_detections_z=result_df_z.phonetic_detection

    # success_rate=len(result_df[result_df.gibberish_truth==result_df.gibberish_detected])/len(result_df)
    # print('errors:',result_df[result_df.gibberish_truth!=result_df.gibberish_detected])
    # result_df[result_df.gibberish_truth!=result_df.gibberish_detected].iloc[-1]
    # print(success_rate)

    selections=selections.reset_index(drop=True)
    # result_df['cmu_phonetics']=selections['cmu_phonetics']
    # result_df['fpath']=selections['fpath']

    # d=count_values(phonetic_detections)
    # d_s=count_values(phonetic_detections_s)

    # return phonetic_detections, phonetic_detections_s, d, d_s



def final_ed_fake_mistakes(n=100):
    df=synth_words_data()
    df['phonetics']=df.apply(lambda r: r.phonetics.replace('CH','T_SH').replace('JH','D_ZH'), axis=1)


    df['target_word_indexes']=0
    df['fpath']=df['path']
    df_target=df[(df.phonetics.str.endswith('_'+"IH0_D"))&~(df.text.str.endswith('ed'))]

    # just get the index of the first occurence of the target
    df_target['target_syllable_indexes']=-1

    selection=df_target.sample(frac=1, random_state=0)[:n]
    selection['cmu_phonetics']=selection['phonetics']

    r=selection.iloc[0]
    s,fs=read_audio_file(r.path, fs=16000)

    
    from src.charsiu_utils import charsiu_phone_forced_aligner
    default_model_charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=r.phonetics,target_word_idx=0,target_syllable_idx=-1,target_occurence_idx=0,target_phones='IH0_D',basis='D_D',contrast='end',mode='numpy',model=default_model_charsiu)


    result_df=compute_predictions(selection, target_phones='IH0_D', basis='IH0_T', tech_function=start_end_contrast_from_formatted_phonetics_audio, model=default_model_charsiu)
    # result_df=compute_predictions(selection, target_phones='T', basis='T_T', tech_function=start_end_contrast_from_formatted_phonetics_audio, model=model)

    phonetic_detections=result_df.phonetic_detection
    
    d=count_values(phonetic_detections)
    # d.columns=[target_phones]

    return result_df, d


def schwa_sound_on_synth_words(target_phones='AH0', n=None, accent=None, model=default_model):
    df=synth_words_data(accent=accent)

    df_target=df[(df.phonetics.str.endswith('_'+target_phones)|df.phonetics.str.startswith(target_phones+'_')|df.phonetics.str.contains('_'+target_phones+'_'))]
    # just get the index of the first occurence of the target    
    df_target['target_syllable_indexes']=df_target['syl_p'].apply(lambda r: [1 if target_phones in el else 0 for el in r].index(1))
    selection=df_target.sample(frac=1, random_state=0)[:n]
    selection['cmu_phonetics']=selection['phonetics']

    result_df=compute_predictions(selection, target_phones=target_phones, tech_function=schwa_sound_from_formatted_phonetics_audio, model=model)
    phonetic_detections=result_df.phonetic_detection
    
    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return result_df, d

def final_ed_on_synth_words(target_phones='IH0_D', n=None, accent=None, model=default_model):
    df=synth_words_data(accent=accent)

    df_target=df[(df.phonetics.str.endswith('_'+target_phones))&(df.text.str.endswith('ed'))]
    # just get the index of the first occurence of the target
    df_target['target_syllable_indexes']=-1

    selection=df_target.sample(frac=1, random_state=0)[:n]
    selection['cmu_phonetics']=selection['phonetics']

    result_df=compute_predictions(selection, target_phones=target_phones, basis='T_T', tech_function=start_end_contrast_from_formatted_phonetics_audio, model=model)
    # result_df=compute_predictions(selection, target_phones='T', basis='T_T', tech_function=start_end_contrast_from_formatted_phonetics_audio, model=model)

    phonetic_detections=result_df.phonetic_detection
    
    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return result_df, d


def pContrast_on_synth_words(target_phones='AO1', n=None, alternatives=cmu_vowels, accent=None, model=default_model):

    # ex for fr_FR, as in "rue" or "lu":

    # df=synth_words_data(path="data/synth_audio/mfa_words/standard/prosody/fr_FR", phonetic_dict=mfa_dicts['fr_FR'], mode='MFA_IPA')
    # target_phones='y'
    # from src.pronunciation_dictionaries import ipa_vowels
    # alternatives=ipa_vowels
    
    df=synth_words_data(accent=accent)

    df_target=df[(df.phonetics.str.endswith('_'+target_phones)|df.phonetics.str.startswith(target_phones+'_')|df.phonetics.str.contains('_'+target_phones+'_'))]
    # just get the index of the first occurence of the target    
    df_target['target_syllable_indexes']=df_target['syl_p'].apply(lambda r: [1 if target_phones in el else 0 for el in r].index(1))
    selection=df_target.sample(frac=1, random_state=0)[:n]

    # selection.progress_apply(lambda r: prefill_for_sentence(r.text, mode='MFA_IPA')['cmu_phonetics'], axis=1)
    # selection.text.progress_apply(lambda r: mfa_dicts['en_US'][r][0] if r in mfa_dicts['en_US'] else float('nan')).dropna()
    
    selection['cmu_phonetics']=selection['phonetics']
    result_df=compute_predictions(selection, target_phones=target_phones, alternatives=alternatives, model=model)

    if len(result_df)>0:
        phonetic_detections=result_df.phonetic_detection
        d=count_values(phonetic_detections)
        d.columns=[target_phones]
    else:d=pd.DataFrame([np.nan])

    return result_df, d


# to be removed
if False:
    def syl_contrast_on_synth_words(syl_target='P_EH1', n=None, accent=None, model=default_model):
        from DL_speech_tech import syllable_contrast_from_formatted_phonetics_audio

        # ex for fr_FR, as in "rue" or "lu":

        # df=synth_words_data(path="data/synth_audio/mfa_words/standard/prosody/fr_FR", phonetic_dict=mfa_dicts['fr_FR'], mode='MFA_IPA')
        # target_phones='y'
        # from src.pronunciation_dictionaries import ipa_vowels
        # alternatives=ipa_vowels
        
        df=synth_words_data(accent=accent)
        df=df[df.apply(lambda r: syl_target.split('_') in r.syl_p, axis=1)]

        selection=df[df.apply(lambda r: syl_target.split('_') in r.syl_p, axis=1)].sample(frac=1, random_state=0)[:n]

        selection['target_syllable_indexes']=selection.syl_p.apply(lambda r: r.index(syl_target.split('_')))
        selection['cmu_phonetics']=selection['phonetics']
        result_df=compute_predictions(selection, model=model, tech_function=syllable_contrast_from_formatted_phonetics_audio)
        phonetic_detections=result_df.phonetic_detection.str.join('_')
        
        d=count_values(phonetic_detections)
        d.columns=[syl_target]

        return result_df, d



def use_tests():

    
    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    default_model_ipa = Wav2Vec2ForFramePrediction(ipa_alphabet)
    default_model_ipa.load(name='model_mailabs_pca_0.95_knn_10_cos_w_UK_US_FR_ES')


    o_list=['AA1', 'AO1', 'OW1']
    i_list=['IH1', 'IY1']
    ed_list=["IH0_D", "D", "T"]

    # from performance_functions import *
    ds_baseline=[]
    for p in o_list:
        results_df, d = pContrast_on_synth_words(target_phones=p, n=100)
        ds_baseline.append(d)
        

    from src.charsiu_utils import charsiu_phone_forced_aligner
    prod_model = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    ds_prod=[]
    for p in o_list:
        results_df, d = pContrast_on_synth_words(target_phones=p, n=100, model=prod_model)
        ds_prod.append(d)
    
    ds_prod_i=[]
    for p in i_list:
        results_df, d = pContrast_on_synth_words(target_phones=p, n=100, model=prod_model)
        ds_prod_i.append(d)
    
    ds_actor_prod=[]
    for p in o_list:
        results_df, d = pContrast_for_actor_recordings(target_phones=p, model=prod_model)
        ds_actor_prod.append(d)
    
    ds_actor_F1_prod=[]
    for p in o_list:
        results_df, d = pContrast_for_actor_recordings(target_phones=p, speakers=['F1'], model=prod_model)
        ds_actor_F1_prod.append(d)
    
    
    
    from src.charsiu_utils import charsiu_phone_forced_aligner
    default_model_charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    final_ed_for_actor_recordings(target_phones='T', model=default_model_charsiu)
    final_ed_for_actor_recordings(target_phones='T', model=default_model)

    final_ed_for_actor_recordings(target_phones='D', model=default_model_charsiu)
    final_ed_for_actor_recordings(target_phones='D', model=default_model)

    final_ed_for_user_data( target_phones='D', n=100, model=default_model_charsiu)
    final_ed_for_user_data( target_phones='D', n=100, model=default_model)

    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    model_cv_lda = Wav2Vec2ForFramePrediction(cmu_alphabet)
    model_cv_lda.load(name='model_commonvoice_pca_99_lda')

    model_cv_knn = Wav2Vec2ForFramePrediction(cmu_alphabet)
    model_cv_knn.load(name='model_commonvoice_pca_99_knn_uk_us_ca_n_1000')
    

    ds_actor_prod=[]
    for p in ed_list:
        results_df, d = final_ed_for_actor_recordings(target_phones=p, model=default_model_charsiu)
        ds_actor_prod.append(d)
    [el.T for el in ds_actor_prod]
    
    ds_actor_pipeline=[]
    for p in ed_list:
        results_df, d = final_ed_for_actor_recordings(target_phones=p, model=default_model)
        ds_actor_pipeline.append(d)
    [el.T for el in ds_actor_pipeline]

    
    ds_actor_pipeline_cv=[]
    for p in ed_list:
        results_df, d = final_ed_for_actor_recordings(target_phones=p, model=model_cv_knn)
        ds_actor_pipeline_cv.append(d)
    [el.T for el in ds_actor_pipeline_cv]

    # ----------------- test on users -----------------------
    
    ds_user_prod_ed=[]
    for p in ed_list:
        results_df, d = final_ed_for_user_data(target_phones=p, model=default_model_charsiu, n=100)
        ds_user_prod_ed.append(d)
    [el.T for el in ds_user_prod_ed]
    
    ds_user_pipeline_ed=[]
    for p in ed_list:
        results_df, d = final_ed_for_user_data(target_phones=p, model=default_model, n=100)
        ds_user_pipeline_ed.append(d)
    [el.T for el in ds_user_pipeline_ed]

    
    ds_user_pipeline_cv_ed=[]
    for p in ed_list:
        results_df, d = final_ed_for_user_data(target_phones=p, model=model_cv_knn, n=100)
        ds_user_pipeline_cv_ed.append(d)
    [el.T for el in ds_user_pipeline_cv_ed]

    
    [el.T for el in ds_actor_prod]
    [el.T for el in ds_actor_pipeline]
    [el.T for el in ds_actor_pipeline_cv]
    [el.T for el in ds_user_prod_ed]
    [el.T for el in ds_user_pipeline_ed]
    [el.T for el in ds_user_pipeline_cv_ed]

    
    ds_user_prod_o=[]
    for p in o_list:
        results_df, d = pContrast_for_user_data(target_phones=p, model=default_model_charsiu, n=100)
        ds_user_prod_o.append(d)
    
    
    ds_user_pipeline_o=[]
    for p in o_list:
        results_df, d = pContrast_for_user_data(target_phones=p, model=default_model, n=100)
        ds_user_pipeline_o.append(d)

    ds_user_pipeline_cv_o=[]
    for p in o_list:
        results_df, d = pContrast_for_user_data(target_phones=p, model=model_cv_knn, n=100)
        ds_user_pipeline_cv_o.append(d)
    
    [el.T for el in ds_user_prod_o]
    [el.T for el in ds_user_pipeline_o]
    [el.T for el in ds_user_pipeline_cv_o]
    
    # ----------------------

    ds=[]
    for p in o_list:
        results_df, d = pContrast_on_synth_words(target_phones=p, n=100)
        ds.append(d)

    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    from sklearn.neighbors import KNeighborsClassifier
    model = Wav2Vec2ForFramePrediction(cmu_alphabet, frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.load(name='model_mailabs_pca_0.95_knn_10_w')
    # results_df2, d2 = pContrast_on_synth_words(n=100, model=model)
    ds2=[]
    for p in o_list:
        results_df, d = pContrast_on_synth_words(target_phones=p, n=100, model=model)
        ds2.append(d)
    
    
    ds2_i=[]
    for p in i_list:
        results_df, d = pContrast_on_synth_words(target_phones=p, n=100, model=model)
        ds2_i.append(d)
    
    ds2_actor=[]
    for p in o_list:
        results_df, d = pContrast_for_actor_recordings(target_phones=p, model=model)
        ds2_actor.append(d)
    
    ds2_actor_F1=[]
    for p in o_list:
        results_df, d = pContrast_for_actor_recordings(target_phones=p, speakers=['F1'], model=model)
        ds2_actor_F1.append(d)
    
    ds2_actor_no_F1=[]
    for p in o_list:
        results_df, d = pContrast_for_actor_recordings(target_phones=p, speakers=['M1', 'F4', 'tts', 'F2', 'M4', 'flwc-phrase-audios', 'zoe'], model=model)
        ds2_actor_no_F1.append(d)
    
    ds2_actor_per_speaker={}
    speakers=['F1', 'M1', 'F4', 'tts', 'F2', 'M4', 'flwc-phrase-audios', 'zoe']
    for s in speakers:
        ds2_actor_per_speaker[s]=[]
        for p in o_list:
            results_df, d = pContrast_for_actor_recordings(target_phones=p, speakers=[s], model=model)
            ds2_actor_per_speaker[s].append(d)
    
    
    ds2_user=[]
    for p in o_list:
        results_df, d = pContrast_for_user_data(target_phones=p, model=model, n=100)
        ds2_user.append(d)


    # sum(results_df.phonetic_detection==results_df2.phonetic_detection)/len(results_df)


    
    from src.load_data import load_libri_dataset, load_libri_dataset_audio_timings
    # phoneme predictions on a train dataset with forced alignment
    # comment for cmu or ipa
    df_t_train, df_t_test = load_libri_dataset()
    data = load_libri_dataset_audio_timings(df_t_test)
    # data = load_test_dataset(df_t_test)
    
    from tqdm import tqdm
    preds=[model.predict_with_timings(r.s, r.phones) for i,r in tqdm(data.iterrows())]
    preds2=[default_model.predict_with_timings(r.s, r.phones) for i,r in tqdm(data.iterrows())]
    
    preds_df=pd.concat(preds)
    preds_df2=pd.concat(preds2)
    sum(preds_df.pred_phones_audio==preds_df2.pred_phones_audio)/len(preds_df)
    