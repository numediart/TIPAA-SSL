from speech_tech import *
from utils.text_processing import *
from utils.audio_processing import prepare_audio_file
from glob import glob
import os
import pandas as pd
from utils.libri_phonetization_data import *
from tqdm import tqdm
import pickle
import seaborn as sns
from utils.label_data_processing import target_to_alternatives, get_sentenceStress_annotation, get_data_new_content, get_data, make_all_phones_annotation_files

def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def rename_underscore_to_dash(files=glob("/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/*/*/*")):
    """Replace all "-" with "_" in filenames because I had inconsistent namings. This could be generalized if needed 
    (put symbol1='-' and symbol2='_' as parameters)

    Args:
        files (list, optional): list of paths (using glob is convenient). Defaults to glob("/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/*/*/*").
    """
    for f in files:
        base, filename=os.path.split(f)
        if '_' in filename:
            new='-'.join(filename.split('_'))
            os.system('mv "'+f+'" "'+os.path.join(base,new)+'"')
            

def compute_errors(preds, GTs):
    diffs=[]
    total_n=0
    stress_error=0
    mismatches={}
    n_example_error=0
    example_errors=[]
    for i,pred in enumerate(preds):
        if len(pred)==len(GTs[i]):
            diff=abs(pred-np.array(GTs[i]))
            diffs.append(diff)
            total_n+=len(pred)
            stress_error+=diff.sum()
            if diff.sum()>0: n_example_error+=1; example_errors.append(i)
        else:
            mismatches[i]=(pred,GTs[i])
    mismatch_rate=len(mismatches.keys())/len(preds)
    bin_error_rate=stress_error/total_n
    example_error_rate=n_example_error/(len(preds)-len(mismatches.keys()))
    print('n of examples:', len(preds))
    print('mismatch_rate:',mismatch_rate)
    print('bin error rate:', bin_error_rate)
    print('example error rate:', example_error_rate)

    return example_errors

def stress_performance_test(level='sentence', audio_path="data/audio-with-analysis-ids/audio/"):
    d=get_data()
    if level=="sentence":
        a,textDict=get_sentenceStress_annotation()
        focusType='sentencestress'
        d=d[d.focusType==focusType]
    elif level=="word":
        focus='wordstress'
        d=d[d.focusType==focus]
        # a=d.text.apply(lambda r:word_stress_from_text(r) )
        a={}
        for i,row in tqdm(d.iterrows()):
            a[row.analysisId]=word_stress_from_text(row.text) 
    else:
        print("level should be 'word' or 'sentence'")
        raise "level should be 'word' or 'sentence'"

    
    preds, statuss, GTs, errors, p_errors = [],[],[],[],[]
    d.text=d.apply(lambda r:remove_special_characters(r['text']), axis=1)
    analysed_ids=[]
    for id,bin in a.items():
        if level=='sentence':
            row=d[d.text==remove_special_characters(textDict[id])]
        elif level=='word':
            row=d[d.analysisId==id]
        if len(row)>0:
            path=os.path.join(audio_path, row.primaryKey.values[0]+'.wav')
            _, rID=prepare_audio_file(path)
            make_all_phones_annotation_files(rID,remove_special_characters(row.text.values[0]))

            # phonetics=phonetics_from_sentence(row.text.values[0])
            formatted_phonetics=prefill_for_sentence(row.text.values[0])['cmu_phonetics']
            # sentence=textDict[id]
            try:
                if level=='sentence':
                    sentence=textDict[id]
                    res=stress_from_formatted_phonetics(rID,phonetics=formatted_phonetics, text=sentence, level="sentence")
                    # res=sentenceStress(rID,rID)
                    pred=res['stress_binaries']
                elif level=='word':
                    res=stress_from_formatted_phonetics(rID,phonetics=formatted_phonetics, level="word")
                    # res=wordStress(rID,rID)
                    # I merge word results to word word with compute_errors
                    pred=merge_list(res['stress_binaries'])
                else:
                    print("level should be 'word' or 'sentence'")
                    raise "level should be 'word' or 'sentence'"
                
                status=res['status']
                statuss.append(status)
                preds.append(pred)
                GTs.append(bin)
                analysed_ids.append(id)
            except  Exception as e:
                # import pdb;pdb.set_trace()
                print('Error with:')
                print(row)
                print(e)
                errors.append(row)
                p_errors.append(path)
    
    print('preds:',preds)
    print('GTs:',GTs)
    all_zero_baseline=[np.zeros(len(el)) for el in GTs]
    print(errors)
    print("all zero baseline")
    compute_errors(all_zero_baseline, GTs)

    print("algo performance")
    compute_errors(preds, GTs)
    return errors



def stress_GE_performance_test(level='sentence'):
    df=get_data_new_content()
    stress_intensities=[]
    stress_binaries=[]
    for i,row in df.iterrows():
        formatted_phonetics=prefill_for_sentence(row.text)['cmu_phonetics']
        _, rID=prepare_audio_file(row.audio_path, speech_correction=False)
        res=stress_from_formatted_phonetics(rID,phonetics=formatted_phonetics, text=row.text, level=level)
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
        # df_all.apply(lambda r: r['stress_intensities'], axis=1)

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
        phonetics=prefill_for_sentence(row.text)['cmu_phonetics']
        split_phonetics=lambda phonetics: [p.replace('|','_').split('_') for p in phonetics.split(' ')]


        df.index=range(len(df))
        word_stress_binaries=df.text.apply(lambda r: [word_stress_from_cmu(p) for p in split_phonetics(prefill_for_sentence(r)['cmu_phonetics'])])

        # I realized some words have all "syllables" stresses (even though they are > 1 syl). In fact, it corresponds to acronyms
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

        GT_by_len=select_by_len(word_stress_binaries)
        preds_by_len=select_by_len(df['stress_binaries'])

        # these should be the same
        print([len(el) for el in GT_by_len])
        print([len(el) for el in preds_by_len])

        for l, (GT,pred) in enumerate(zip(GT_by_len, preds_by_len)):
            error_rate=sum(sum(np.abs(np.array(GT)-np.array(pred))))/np.prod(np.array(pred).shape)
            print('words of len '+str(l+1)+' error rate:'+str(error_rate))


def compute_prediction_results(selection, libri_words_df, target_phones='IH0 D', 
                            alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']):
    """Calls phonemeContrast for all rows of a selection and gather the predictions in the dataframe and returns it.

    Args:
        selection ([type]): [description]
        libri_words_df ([type]): [description]
        target_phones (str, optional): [description]. Defaults to 'IH0 D'.
        alternatives (list, optional): [description]. Defaults to ['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'].

    Returns:
        dataframe: results_df containing info from input + column for detected_transcriptions and status (to see if there was an error in the process)
    """
    detected_transcriptions=[]
    statuss=[]
    result_records=[]
    for i,row in tqdm(selection.iterrows()):
        sentence=get_sentence(row.path)
        phonetics=phonetics_for_row(row, libri_words_df)
        phonetics=[p.split(' ') for p in phonetics]

        status_audio, rID=prepare_audio_file(row.wav_path)        
        make_pContrast_annotation_files_from_phonetics(rID,phonetics=phonetics, word_idx=row.word_idx, target_phones=target_phones, alternatives=alternatives)
        status,results=phonemeContrast(rID, rID)
        
        statuss.append(status)
        if results!=[]:
            detected_transcription=results[1]
            confidence_scores=results[0].iloc[:,3].tolist()
            phonetic_detection=results[0][results[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
            
            phonetic_detection_timings=results[0][results[0].iloc[:,2].str.contains('_')].iloc[:,0:2].to_numpy()
            d={'detected_phone':phonetic_detection,'phonetic_detection_timings':phonetic_detection_timings,'phones':row.phones,'detected_transcription':' '.join(detected_transcription), 'confidence_scores':confidence_scores, 'status':status, 'path':row.path, 'wav_path':row.wav_path, 'sentence':sentence, 'phonetics':phonetics, 'word_idx':row.word_idx, 'file_idx':row.file_idx}
            detected_transcriptions.append(detected_transcription)
        else:
            detected_transcriptions.append([])
            d={'detected_phone':'','phones':row.phones, 'detected_transcription':'', 'confidence_scores':[], 'status':status, 'path':row.path, 'wav_path':row.wav_path, 'sentence':sentence, 'phonetics':phonetics, 'word_idx':row.word_idx, 'file_idx':row.file_idx}
        result_records.append(d)
    results_df=pd.DataFrame.from_records(result_records)
    return results_df

def compute_score(results_df, correct_terminations=['T IH0 D', 'T IH1 D', 'T IH2 D']):
    n_corr=0
    for el in correct_terminations: 
        n_corr+=len(results_df[results_df.detected_transcription.str.endswith(el)])
    rate=n_corr/len(results_df)
    print('rate :', rate)
    return rate

def get_rest(results_df, correct_terminations=['T IH0 D', 'T IH1 D', 'T IH2 D']):
    rest=results_df
    for el in correct_terminations:
        rest=rest[~rest.detected_transcription.str.endswith(el)]
    return rest

def performance_test(selection, results_df, correct_terminations):
    # print(rate)
    rate=compute_score(results_df, correct_terminations=correct_terminations)
    rest=get_rest(results_df, correct_terminations=correct_terminations)
    selection.index=results_df.index
    # selection.loc[rest.index]
    rest=pd.concat([selection.loc[rest.index][['word','phones']],rest], axis=1)
    print(rest)
    return rate, rest

def get_phone_termination_dict():
    """Get rules of terminations for -ed module (maybe to be generalized)

    Returns:
        dicts: associates phoneme to the correct termination and accepted alternatives
    """
    consonants=[p for p in cmudict.phones() if p[-1][0]!='vowel']
    consonants_not_D_T=[p for p in consonants if p[0] not in ['D','T']]
    liquid=[p for p in consonants if p[-1][0]=='liquid']
    semivowel=[p for p in consonants if p[-1][0]=='semivowel']
    nasal=[p for p in consonants if p[-1][0]=='nasal']

    others=[p for p in consonants if (p[-1][0]!='nasal' and  p[-1][0]!='liquid' and  p[-1][0]!='semivowel')]

    vuv_dict={'B':'v',
            'CH':'uv',
            'D':'v',
            'DH':'v',
            'F':'uv',
            'G':'v',
            'HH':'uv',
            'JH':'v',
            'K':'uv',
            'P':'uv',
            'S':'uv',
            'SH':'uv',
            'T':'uv',
            'TH':'uv',
            'V':'v',
            'Z':'v',
            'ZH':'v'}
    
    for el in liquid: vuv_dict[el[0]]='v'
    for el in nasal: vuv_dict[el[0]]='v'
    for el in semivowel: vuv_dict[el[0]]='v'

    vuv_termination_dict={'v':'D', 'uv':'T'}

    phone_termination_dict={}
    for p,vuv in vuv_dict.items(): phone_termination_dict[p]=vuv_termination_dict[vuv]
    phone_termination_dict['T']= 'IH0 D'
    phone_termination_dict['D']= 'IH0 D'
    
    correct_alternatives={}
    correct_alternatives['IH0 D']=['IH0 D', 'IH1 D', 'IH2 D']
    correct_alternatives['D']=['D', 'D AH0']
    correct_alternatives['T']=['T', 'T AH0']

    return phone_termination_dict, correct_alternatives




def performance_from_results(results_df):
    """This looks at "detected_transcription" and "phones" columns.
    It checks when there is a match after removing stress characters (0,1,2) in cmu vowels.

    Args:
        results_df ([dataframe]): output of compute_prediction_results function

    Returns:
        [type]: [description]
    """
    match=[]
    for i,row in results_df.iterrows():
        # if row.phones=='': import pdb;pdb.set_trace()
        if row.detected_transcription!='': #import pdb;pdb.set_trace()
            match.append(remove_stress_annots(row.phones.split(' '))==remove_stress_annots(row.detected_transcription.split(' ')))
        else:
            match.append(False)
    success_rate=sum(match)/len(match)
    all_results={'results_df':results_df, 'failure':results_df[[not el for el in match]],  'performance':success_rate}
    return all_results

def pContrast_from_audiobook_data(data_set='dev-clean', target_phones='AO1', alternatives=['AO0', 'OW0','AO1', 'OW1','AO2', 'OW2'], n=None):
    """This function uses libri_words_df containing information of sentences and words in 
    librispeech dataset with phonetic transcriptions and timings.
    It selects sentences containing a word with target_phones in it, 
    compute prediction on that (make annotation files, call htk, call phonemeContrast module)
    Then it computes a success rate by checking when the detected transcription is the same as the ground truth 
    discarding cmu vowel stresses.

    Args:
        data_set (str, optional): [description]. Defaults to 'dev-clean'.
        target_phones (str, optional): [description]. Defaults to 'AO1'.
        alternatives (list, optional): [description]. Defaults to ['AO0', 'OW0','AO1', 'OW1','AO2', 'OW2'].
        n ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)

    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    selection=libri_words_df[libri_words_df.phones.str.contains(target_phones)]

    if len(selection)>0:
        results_df=compute_prediction_results(selection, libri_words_df, target_phones=target_phones, alternatives=alternatives)
        all_results=performance_from_results(results_df)
    else: all_results=[]
    if n is None:
        pickle.dump(all_results, open('performance_results/pContrast_performance_'+data_set+'_'+target_phones+'.p', 'wb'))
    else:
        pickle.dump(all_results, open('performance_results/pContrast_performance_'+data_set+'_'+target_phones+'_from_'+str(n)+'egs.p', 'wb'))
    return all_results


def pContrast_for_actor_recordings(target_phones='AO1'):
    df=pd.read_csv('data/exercise_data_export.csv')
    df['audio_file_url']='data/scaleway-audio-files/'+df['audio_file_url']
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]    
    selection['split_phonetics']=selection.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)
    selection['fpath']=selection.audio_file_url
    selection['audio_file_idx']=selection.fk_audio_recording_id

    phones=cmudict.phones()
    cmu_vowels=[p[0]+'1' for p in phones if p[1][0]=='vowel']

    alternatives=cmu_vowels
    statuss=[]
    result_records=[]
    detected_transcriptions=[]
    for i,row in selection.iterrows():    
        target_word_idx=ast.literal_eval(row.target_word_indexes)[0]
        target_syllable_idx=ast.literal_eval(row.target_syllable_indexes)[0]
        status_audio, rID=prepare_audio_file(row.fpath)        
        make_pContrast_annotation_files_from_phonetics(rID,phonetics=row.split_phonetics, word_idx=target_word_idx, target_phones=target_phones, alternatives=alternatives)
        status,results=phonemeContrast(rID, rID)

        statuss.append(status)
        if results!=[]:
            detected_transcription=results[1]
            confidence_scores=results[0].iloc[:,3].tolist()
            phonetic_detection=results[0][results[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
            phonetic_detection_timings=results[0][results[0].iloc[:,2].str.contains('_')].iloc[:,0:2].to_numpy()
            d={'detected_phone':phonetic_detection,'phonetic_detection_timings':phonetic_detection_timings,'phones':row.cmu_phonetics,'detected_transcription':' '.join(detected_transcription), 'confidence_scores':confidence_scores, 'status':status,'wav_path':row.fpath}
            detected_transcriptions.append(detected_transcription)
        else:
            detected_transcriptions.append([])
            d={'detected_phone':'','phones':row.cmu_phonetics, 'detected_transcription':'', 'confidence_scores':[], 'status':status, 'wav_path':row.fpath}
        result_records.append(d)
    results_df=pd.DataFrame.from_records(result_records)

    phonetic_detections=results_df.detected_phone.apply(lambda r: r[0] if len(r)>0 else 'err').tolist()
    d=Counter(phonetic_detections)
    # print(d)
    d = pd.DataFrame.from_dict(d, orient='index')
    d.columns=[target_phones]
    d=d.sort_values(by=target_phones, ascending=False)
    d=d / d.sum()*100

    return phonetic_detections, d


def unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='IY1', target_graphemes='ea', alternatives=['IY1', 'EH1'], n=None):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)

    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    selection=libri_words_df[libri_words_df.phones.str.contains(target_phones) & libri_words_df.word.str.contains(target_graphemes)]

    if len(selection)>0:
        results_df=compute_prediction_results(selection, libri_words_df, target_phones=target_phones, alternatives=alternatives)
        all_results=performance_from_results(results_df)
    if n is None:
        pickle.dump(all_results, open('performance_results/unpredictable_vowels_performance_'+data_set+'_'+target_graphemes+'_'+target_phones+'.p', 'wb'))
    else:
        pickle.dump(all_results, open('performance_results/unpredictable_vowels_performance_'+data_set+'_'+target_graphemes+'_'+target_phones+'_from_'+str(n)+'egs.p', 'wb'))
    return all_results


def edAnalysis_from_audiobook_data(data_set='dev-clean', n=None):
    """This function uses libri_words_df containing information of sentences and words in 
    librispeech dataset with phonetic transcriptions and timings.

    It selects sentences containing a word with each pair (pretermination,termination) of words in "-ed" 
    It computes prediction on that (make annotation files, call htk, call phonemeContrast module)
    Then it computes, for every pair (pretermination,termination), a success rate by checking when the 
    detected transcription ENDS WITH the same termination as the ground truth 

    Args:
        data_set (str, optional): [description]. Defaults to 'dev-clean'.
        n ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    # if n is not None: libri_words_df=libri_words_df.iloc[:n,:]

    phone_termination_dict, correct_alternatives=get_phone_termination_dict()

    results_dfs={}
    rests_dfs={}
    performances={}

    for pretermination, termination in tqdm(phone_termination_dict.items()):
        full_termination=' '+' '.join([pretermination,termination])
        correct_terminations=[' '+' '.join([pretermination,el]) for el in correct_alternatives[termination]]
        selection=libri_words_df[libri_words_df.phones.str.endswith(full_termination)]
        # make sure the text ends with -ed. It get rids of words such as "and"
        selection=selection[selection.word.str.endswith('ed')]
        

        if len(selection)>0:
            results_df=compute_prediction_results(selection, libri_words_df, target_phones=termination)
            rate, rest=performance_test(selection, results_df, correct_terminations)

            results_dfs[pretermination]=results_df
            rests_dfs[pretermination]=rest
            performances[pretermination]=rate
    
    all_results={'results_dfs':results_dfs, 'rests_dfs':rests_dfs,  'performances':performances}
    if n is None:
        pickle.dump(all_results, open('performance_results/ed_performance_'+data_set+'.p', 'wb'))
    else:
        pickle.dump(all_results, open('performance_results/ed_performance_'+data_set+'_from_'+str(n)+'egs.p', 'wb'))
    return all_results

def edAnalysis_experiment(data_set='dev-clean'):
    with open('performance_results/ed_performance_'+data_set+'.p', 'rb') as f: ed_perf=pickle.load(f)
    ed_perf['results_dfs'].keys()
    ed_perf['results_dfs']['N']
    
    # Select rows with 'T' as first element and get their timings
    mask=ed_perf['results_dfs']['N'].detected_phone.apply(lambda x:x[0]=='D')
    df=ed_perf['results_dfs']['N'][mask]
    timings=ed_perf['results_dfs']['N'][mask].phonetic_detection_timings.apply(lambda x:x[0])

    voicing=[]
    for i,r in df.iterrows():
        s,fs=load_audio(r.wav_path)
        start=int(timings[i][0]*fs)
        end=int(timings[i][1]*fs)
        p_signal=s[start:end]
        f0Samples=getf0Samples(p_signal,fs)
        voicing.append(len(f0Samples[~np.isnan(f0Samples)])/len(f0Samples))


def final_s_from_audiobook_data(data_set='dev-clean', n=None):
    
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    # words that finish in "s" with phoneme "S" that also exist without an "s" and with last phoneme then not being "S"
    words_in_s=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='S' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='S']
    words_in_s=set(words_in_s)

    selections=[]
    selections_s=[]
    results_dfs=[]
    results_dfs_s=[]
    # rests_dfs=[]
    # rests_dfs_s=[]

    libri_words_list=libri_words_df.word.unique()
    for word in libri_words_list:
        if word+'s' in words_in_s:
            selection, selection_s = selection_with_and_without_s(libri_words_df, word)
            if len(selection)>0 and len(selection_s)>0:
                target=selection.iloc[0].phones.split(' ')[-1]
                results_df=compute_prediction_results(selection, libri_words_df, target_phones=target, 
                                alternatives=[target, target + ' S'])
                results_df_s=compute_prediction_results(selection_s, libri_words_df, target_phones=target+ ' S', 
                                alternatives=[target, target + ' S'])
                selections.append(selection)
                selections_s.append(selection_s)
                results_dfs.append(results_df)
                results_dfs_s.append(results_df_s)
    
    sdf=pd.concat(selections)
    rdf=pd.concat(results_dfs)

    all_results=performance_from_results(rdf)

    sdf_s=pd.concat(selections_s)
    rdf_s=pd.concat(results_dfs_s)

    all_results_s=performance_from_results(rdf_s)


    if n is None:
        pickle.dump(all_results, open('performance_results/no_final_s_performance_'+data_set+'.p', 'wb'))
        pickle.dump(all_results_s, open('performance_results/final_s_performance_'+data_set+'.p', 'wb'))
    else:
        pickle.dump(all_results, open('performance_results/no_final_s_from_audiobook_data_performance_'+data_set+'_from_'+str(n)+'egs.p', 'wb'))
        pickle.dump(all_results_s, open('performance_results/final_s_from_audiobook_data_performance_'+data_set+'_from_'+str(n)+'egs.p', 'wb'))
    
    # all_results_s['failure'][all_results_s['failure'].phones.str.endswith(' T S')]
    # all_results['failure'][all_results['failure'].phones.str.endswith(' T')]

    return all_results, all_results_s
            

def confusion_analysis_of_pContrast(phone_set, n=100, data_set='dev-clean'):
    confusion_records=[]
    for p in phone_set:
        r=pContrast_from_audiobook_data(data_set=data_set, target_phones=p, alternatives=phone_set, n=n)
        dict_count_prediction={}
        for phone in phone_set: dict_count_prediction[phone]=0
        if r!=[]:
            for i,row in r['results_df'].iterrows(): 
                # row.detected_phone
                for el in row.detected_phone: dict_count_prediction[el]+=1
        confusion_records.append(dict_count_prediction)
    confusion_df=pd.DataFrame.from_records(confusion_records)
    confusion_df.index=phone_set
    return confusion_df

def vowels_consonants_confusions_from_audiobook_data(n=100, data_set='dev-clean'):
    import cmudict
    phones=cmudict.phones()

    cmu_vowels=[p[0]+'1' for p in phones if p[1][0]=='vowel']
    cmu_consonants=[p[0] for p in phones if p[1][0]!='vowel']
    # results={}

    vowel_confusion_df=confusion_analysis_of_pContrast(cmu_vowels, n=n, data_set=data_set)
    vowel_confusion_df_norm=(vowel_confusion_df.div(vowel_confusion_df.sum(axis=1), axis=0)*100).round(1)
    vowel_confusion_df.to_csv('performance_results/vowel_confusion_df_n_'+str(n)+'_'+data_set+'.csv')
    vowel_confusion_df_norm.to_csv('performance_results/vowel_confusion_df_norm_n_'+str(n)+'_'+data_set+'.csv')
    
    consonant_confusion_df=confusion_analysis_of_pContrast(cmu_consonants, n=n, data_set=data_set)
    consonant_confusion_df_norm=(consonant_confusion_df.div(consonant_confusion_df.sum(axis=1), axis=0)*100).round(1)
    consonant_confusion_df.to_csv('performance_results/consonant_confusion_df_n_'+str(n)+'_'+data_set+'.csv')
    consonant_confusion_df_norm.to_csv('performance_results/consonant_confusion_df_norm_n_'+str(n)+'_'+data_set+'.csv')

    plt.clf()
    sns.heatmap(vowel_confusion_df_norm, annot=True, cmap='YlGnBu')
    plt.savefig('performance_results/vowel_contrast_confusion_n_'+str(n)+'_'+data_set+'.png')
    plt.clf()
    sns.heatmap(consonant_confusion_df_norm, annot=True, cmap='YlGnBu')
    plt.savefig('performance_results/consonant_contrast_confusion_n_'+str(n)+'_'+data_set+'.png')


def vowels_confusions_actor_recordings():
    # for actors recordings, take only the true targets
    vowels=['IY1','IH1','AO1','AA1','OW1']
    results=[]
    for v in vowels: 
        print("vowel:",v)
        # predictions[v]=pContrast_from_audiobook_data(data_set='dev-clean', target_phones=v, n=n)
        # _, predictions[v]=pContrast_for_actor_recordings(target_phones=v)
        _, rates=pContrast_for_actor_recordings(target_phones=v)
        # _, predictions[v].to_csv('performance_results/vowel_accuracies_'+v+'.csv')
        results.append(rates)

    return results

if __name__ == "__main__":

    stress_performance_test()

    df=stress_ranking_test()

    # remove examples with empty detection (nothing recognized)
    df=df[df.apply(lambda r:len(r.stress_intensities), axis=1)>0]

    ratio=df.apply(lambda r: r['stress_binaries'][r['bins'].index(1)], axis=1).sum()/len(df)

    # wrong examples:
    df[df.apply(lambda r: r['stress_binaries'][r['bins'].index(1)], axis=1)==0][['text','bins','stress_binaries','stress_intensities']]

    ratio=len(df[df.stress_intensities.apply(lambda r:[int(el/100) for el in r])==df.bins])/len(df)

    ratio=len(df[df.COI_stress_intensities.apply(lambda r:[int(el/100) for el in r])==df.COI_bins])/len(df)
    ratio=len(df[df.COI_stress_intensities.apply(lambda r:[np.argmax(r)] if len(r)<2 else np.argpartition(r, -2)[-2:])==df.COI_bins.apply(lambda r:r.index(1))])/len(df)

    # select examples for which the 1 is among the two largest intensities and with predicted intensity>thresh
    n_max=2
    threshold=60
    select=df[df.apply(lambda r:(
        r.COI_bins.index(1) in 
        ([np.argmax(r.COI_stress_intensities)] 
        if len(r.COI_stress_intensities)<n_max+1 else np.argpartition(r.COI_stress_intensities, -n_max)[-n_max:])
        )
        & (r.COI_stress_intensities[r.COI_bins.index(1)]>threshold)
    , axis=1)]
    ratio=len(select)/len(df)

    # df[df.apply(lambda r:r.COI_stress_intensities[r.COI_bins.index(1)]>60, axis=1)]
    # n_max_stress_intensities=df.COI_stress_intensities.apply(lambda r:[np.max(r)] if len(r)<n_max else [r[el] for el in np.argpartition(r, -n_max)[-n_max:]])
    # n_max_stress_intensities[n_max_stress_intensities.apply(lambda r:min(r))<10]

    df.stress_intensities.apply(lambda r:[int(el/100) for el in r])

    df.bins.apply(lambda r:r.index(1))
    sum_df=df.bins.apply(lambda r:sum(r))
    # sum_df[sum_df==0]
    # df[sum_df==0]

    df_1=df[sum_df==1]

    if False:
        # pContrast_from_audiobook_data(contrasted_phonemes=['DH', 'TH'], alternatives=['DH', 'TH'], module="thContrast")
        # show_summary(contrasted_phonemes=['DH', 'TH'], module="thContrast")
        # show_summary(contrasted_phonemes=['IY1', 'IH1'], module="iContrast")
        # show_summary(contrasted_phonemes=['AO1', 'OW1'], module="oContrast")
        # show_summary()

        # pContrast_from_audiobook_data()
        # edAnalysis_from_audiobook_data(data_set='test-other')
        pContrast_from_audiobook_data(target_phones='AO1', alternatives=['AO0', 'OW0','AO1', 'OW1','AO2', 'OW2'], n=100)
        pContrast_from_audiobook_data(target_phones='AO1', alternatives=['AO1', 'OW1'], n=100)
        pContrast_from_audiobook_data(target_phones='IY1', alternatives=['IH0', 'IY0','IH1', 'IY1','IH2', 'IY2'], n=100)
        pContrast_from_audiobook_data(target_phones='IH1', alternatives=['IH1', 'IY1'], n=100)
        pContrast_from_audiobook_data(target_phones='IH0', alternatives=['IH0', 'IY0'], n=100)

        
        pContrast_from_audiobook_data(target_phones='IH1', alternatives=['IH1', 'IY1', 'AY1'], n=20)

        # target='IH1'
        for target in target_to_alternatives.keys():
            pContrast_from_audiobook_data(target_phones=target, alternatives=target_to_alternatives[target], n=100)

        alternatives=['F T',
            'F IH2 D',
            'F EH2 D',
            'F',
            'F D']
        pContrast_from_audiobook_data(target_phones="F T", alternatives=alternatives, n=1000)

        unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='IY1', target_graphemes='ea', alternatives=['IY1', 'EH1', 'EY1'], n=100)
        unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='EH1', target_graphemes='ea', alternatives=['IY1', 'EH1', 'EY1'], n=100)
        unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='EY1', target_graphemes='ea', alternatives=['IY1', 'EH1', 'EY1'], n=1000)
        
        unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='IY1', target_graphemes='ie', alternatives=['IY1', 'AY1'], n=1000)
        unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='AY1', target_graphemes='ie', alternatives=['IY1', 'AY1', 'Y EH1'], n=1000)


        results=pContrast_from_audiobook_data(target_phones='DH', alternatives=['DH', 'TH'], n=100)
        row=results['failure'].iloc[1]
        results['results_df'].status.unique()

        # compute_prediction_for_row(row,row.phonetics,target_phones='DH', alternatives=['DH', 'TH'])

        pContrast_from_audiobook_data(target_phones='DH', alternatives=['DH', 'Z'], n=100)
        results=pContrast_from_audiobook_data(target_phones='TH', alternatives=['DH', 'TH'], n=100)
        # results=pContrast_from_audiobook_data(contrasted_phonemes=contrasted_phonemes, alternatives=alternatives, module="thContrast", n=100)
        
        
        row=results['failure'].iloc[-1]

        # phonetics_from_sentence(results['results_dfs']['TH'].iloc[0].sentence)
        target_phones='TH'

        # p=set_params(basename='test', module="thContrast")
        # prepare_audio_file(p)
        status_audio, rID=prepare_audio_file(row.wav_path)
        # p['rand_fileName']=rID
        # make_dir(os.path.split(p['inputPhoneticTranscription'])[0])
        # make_dir(os.path.split(p['inputGrammar'])[0])
        # alternatives=['DH', 'TH']
        # make_generic_dct_from_phonetics(phonetics=row.phonetics, word_idx=row.word_idx, target_phones=target_phones, alternatives=alternatives, path=p['inputPhoneticTranscription'])
        # make_grammar_from_dct(path_dct=p['inputPhoneticTranscription'],path_grammar=p['inputGrammar'])
        # status, results= phonemeContrast(p['rand_fileName'])
