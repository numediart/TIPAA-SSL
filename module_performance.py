from speech_tech import *
from label_data_processing import *
from text_processing import *
from glob import glob
import os
import pandas as pd
from libri_phonetization_data import *
from tqdm import tqdm
import pickle
# Performance tests
import seaborn as sns

from speech_tech import target_to_alternatives, graphemes_to_alternatives

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

module_name_to_focus_type={'wordStress':'wordstress',
    'sentenceStress':'sentencestress',
    'iContrast':'shortIlongI'}
def get_preds_and_GTs_from_data(d, a, module, audio_path="../audio-with-analysis-ids/audio/"):
    """This function works for wordStress, sentenceStress. It does not work for iContrast, because an annotation file (an thus a sentenceID) can correspond to several texts.
     It uses the data.json file containing information 
    from dynamoDB: (audio, text, sentenceID, module), and get the corresponding dct files to run prediction of a module on it.

    Args:
        d (dataframe): information  from dynamoDB audio, text, sentenceID, module
        a (dict): annotations (ground truth)
        module (string): module name
        audio_path (str, optional): [description]. Defaults to "../audio-with-analysis-ids/audio/".

    Returns:
        lists: [description]
    """
    preds, statuss, GTs, errors, p_errors = [],[],[],[],[]

    for i,row in tqdm(d.iterrows()):
        id=row.analysisId
        if id in a:
            ground_truth=a[id]
            path=os.path.join(audio_path, row.primaryKey+'.wav')
            p=set_params(sentenceID=id, waveFileAddress=path, module=module)
            # prepare_audio_file(p)
            status_audio, rID=prepare_audio_file(path)
            p['rand_fileName']=rID
            try:
                # this calls the function with the name of the module
                res=globals()[module](p)
                
                pred=res['stress_binaries']
                status=res['status']
                statuss.append(status)
                preds.append(pred)
                GTs.append(ground_truth)
            except:
                print('Error with:')
                print(row)
                errors.append(row)
                p_errors.append(p)
    
    print(preds)
    print(GTs)
    print(errors)

    return preds, statuss, GTs, errors

def wordStress_performance_test():
    d=get_data()
    #d[d.analysisId==232][d.focusType=='wordstress']
    # a=get_wordStress_annotation()
    module='wordStress'
    focus='wordstress'
    # audio_path="../audio-with-analysis-ids/audio/"
    d=d[d.focusType==focus]
    a={}
    for i,row in tqdm(d.iterrows()):
        a[row.analysisId]=word_stress_from_text(row.text) 
        print(row)
        print(a[row.analysisId])
    preds, statuss, GTs, errors=get_preds_and_GTs_from_data(d, a, module)
    preds=[np.concatenate(el) for el in preds]
    example_errors=compute_errors(preds,GTs)

    # row=d[d.analysisId==422].iloc[0]
    # audio_path="../audio-with-analysis-ids/audio/"
    # path=os.path.join(audio_path, row.primaryKey+'.wav')
    # p=set_params(sentenceID=row.analysisId, waveFileAddress=path, module=module)
    # status, textgridData, s = get_annotated_signal(p)
    # wordStress(p)

    # os.system('cat '+p['inputPhoneticTranscription'])
    # os.system('cat ./lexicon/wordStress/dct_old/'+os.path.split(p['inputPhoneticTranscription'])[-1])
    # os.system('cat '+p['inputGrammar'])

def sentenceStress_performance_test():
    d=get_data()
    #d[d.analysisId==232][d.focusType=='wordstress']
    a,textDict=get_sentenceStress_annotation()
    module='sentenceStress'
    focusType='sentencestress'
    d=d[d.focusType==focusType]
    audio_path="../audio-with-analysis-ids/audio/"
    preds, statuss, GTs, errors, p_errors = [],[],[],[],[]

    for i,r in d.iterrows():
        d=d.replace(d.loc[i].text,remove_special_characters(r.text))

    analysed_ids=[]
    for id,bin in a.items():
        row=d[d.text==remove_special_characters(textDict[id])]
        if len(row)>0:
            ground_truth=a[id]
            path=os.path.join(audio_path, row.primaryKey.values[0]+'.wav')
            p=set_params(sentenceID=id, waveFileAddress=path, module='sentenceStress')
            # prepare_audio_file(p)
            status_audio, rID=prepare_audio_file(path)
            p['rand_fileName']=rID
            a,textDict=get_sentenceStress_annotation()
            p=make_all_phones_annotation_files(p,remove_special_characters(textDict[p['sentenceID']]))
            try:
                res=sentenceStress(p)
                pred=res['stress_binaries']
                status=res['status']
                statuss.append(status)
                preds.append(pred)
                GTs.append(ground_truth)
                analysed_ids.append(id)
            except:
                print('Error with:')
                print(row)
                errors.append(row)
                p_errors.append(p)
                # import pdb;pdb.set_trace()
    
    print(preds)
    print(GTs)
    # print(GTs_from_phonetics)
    all_zero_baseline=[np.zeros(len(el)) for el in GTs]
    print(errors)
    print("all zero baseline")
    compute_errors(all_zero_baseline, GTs)

    print("algo performance")
    compute_errors(preds, GTs)



def iContrast_automatic_annot_performance_test(path='../audio-with-analysis-ids/iContrast_data.csv', audio_path="../audio-with-analysis-ids/audio/"):
    df=pd.read_csv(path)
    # only those with analysisID in 3 digits have a manual annotation
    df=df[df.analysisId>100]
    a,w_id,s_id=get_iContrast_annotations()
    module='iContrast'
    focus=module_name_to_focus_type[module]
    
    # preds, statuss, GTs, errors=get_preds_and_GTs_from_data(d, a, module)
    alternatives=['IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2']
    all_results=[]
    statuss=[]
    short_long_prediction={}
    for i,row in tqdm(df.iterrows()):
        id=row.analysisId
        ground_truth=a[row.primaryKey]
        word_id=w_id[id]
        syl_id=s_id[id]
        # row=d[d.analysisId==id]
        text=row.text
        phonetics=phonetics_from_sentence(text)
        vowels_in_word=[el for el in phonetics[word_id] if el[-1] in str([0,1,2])]
        target_phones=vowels_in_word[syl_id]
        # target_phones=

        path=os.path.join(audio_path, row.primaryKey+'.wav')
        status,results=phonemeContrast_from_text_audio(text, path, word_id, target_phones, alternatives)
        all_results.append(results)
        statuss.append(status)

        if len(results)>0:
            # this line get in textgridData, among the phones, the one with '_' because it is the one that have several alternatives
            try:
                phone=results[0][results[0].iloc[:,2].str.startswith('p')&results[0].iloc[:,2].str.contains('_')].detected_transcription.values[0]
            except:
                pdb.set_trace()
            if phone[:2]=='IH':
                short_long_prediction[row.primaryKey]=0
            else:
                short_long_prediction[row.primaryKey]=1
        # else:
        #     short_long_prediction[id]=np.nan
    
    mismatch_rate=(len(a)-len(short_long_prediction))/len(short_long_prediction)

    diff=[abs(a[id]-short_long_prediction[id]) for id in short_long_prediction]
    # [k,v ]
    example_error_rate=sum(diff)/(len(diff))
    print('n of examples:', len(a))
    print('mismatch_rate:',mismatch_rate)
    # print('bin error rate:', bin_error_rate)
    print('example error rate:', example_error_rate)



def edAnalysis_performance_test():
    files=glob("/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/*/*/*wav")
    xls='/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/20 words - research/20 problems-research.xlsx'
    a=pd.read_excel(xls)
    path='audio_recordings/edAnalysis'
    ed_sentenceID=pd.read_csv(os.path.join('./lexicon/edAnalysis/dct','ed_sentenceID.csv'))
    results, analyzed_files=[],[]
    # tgs=[]
    for i,r in ed_sentenceID.iterrows():
        text=r[0].split('.')[0]
        text_dashed='-'.join(text.split(' '))
        print(text_dashed)
        for f in files:
            if text_dashed in os.path.split(f)[-1]:
                sentence_id_from_text=ed_sentenceID[ed_sentenceID['Unnamed: 0'].str.contains(text)]['Unnamed: 1'].values[0]
                p=set_params(sentenceID=sentence_id_from_text, waveFileAddress=f, module="edAnalysis")
                # prepare_audio_file(p)
                status_audio, rID=prepare_audio_file(f)
                p['rand_fileName']=rID
                # p['inputPhoneticTranscription']='./lexicon/edAnalysis/dct/'+text+'.dct'
                # p['inputGrammar']='./lexicon/edAnalysis/grammar/'+text+'.txt'
                # status, textgridData, s = get_annotated_signal(p)
                results.append(edAnalysis(p)) 
                # tgs.append(textgridData)
                analyzed_files.append(f)

    df_results=pd.DataFrame()
    df_results['wav']=[os.path.split(f)[-1] for f in analyzed_files]
    df_results['prediction']=pd.DataFrame(results).iloc[:,1]
    df_results[df_results.wav.str.contains('F1')]
    df_results[df_results.wav.str.contains('F2')]
    df_results[df_results.wav.str.contains('M1')]
    df_results[df_results.wav.str.contains('M2')]

def compute_voicings(selection, libri_words_df, target_phones='IH0 D'):
    prosodies=[]
    for i,row in tqdm(selection.iterrows()):
        sentence=get_sentence(row.path)
        # retrieve phonetics by word that ins not available directly from textgrids, but can be extracted from the dataframe
        # as I already extracted phonemes for each word using overlapping in timings
        phonetics=[]
        for w in sentence.split(' '):
            phonetics.append(libri_words_df[libri_words_df.word==w].iloc[0,:].phones)

        # set params and make label files for phonetics and grammar
        p=set_params(waveFileAddress=row.wav_path, sentenceID=None, basename='test', module='edAnalysis')
        # prepare_audio_file(p)
        status_audio, rID=prepare_audio_file(row.wav_path)
        p['rand_fileName']=rID
        make_generic_dct_from_phonetics(phonetics=phonetics, word_id=row.word_idx, target_phones=target_phones, path=p['inputPhoneticTranscription'])
        make_grammar_from_dct(path_dct=p['inputPhoneticTranscription'],path_grammar=p['inputGrammar'])
        prosodies.append(prosody_by_phone(p))


def compute_prediction_results(selection, libri_words_df, target_phones='IH0 D', 
                            alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']):
    """Calls phonemeContrast_from_phonetics_audio for all rows of a selection and gather the predictions in the dataframe and returns it.

    Args:
        selection ([type]): [description]
        libri_words_df ([type]): [description]
        target_phones (str, optional): [description]. Defaults to 'IH0 D'.
        alternatives (list, optional): [description]. Defaults to ['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'].

    Returns:
        [type]: [description]
    """
    detected_transcriptions=[]
    statuss=[]
    result_records=[]
    for i,row in tqdm(selection.iterrows()):
        sentence=get_sentence(row.path)
        phonetics=phonetics_for_row(row, libri_words_df)
        phonetics=[p.split(' ') for p in phonetics]
        p=set_params(waveFileAddress=row.wav_path)
        # prepare_audio_file(p)
        status_audio, rID=prepare_audio_file(row.wav_path)
        p['rand_fileName']=rID
        status, results = phonemeContrast_from_phonetics_audio(phonetics=phonetics, p=p, word_id=row.word_idx, target_phones=target_phones, alternatives=alternatives)
        statuss.append(status)
        if results!=[]:
            detected_transcription=results[1]
            phonetic_detection=results[0][results[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
            d={'detected_phone':phonetic_detection,'phones':row.phones,'detected_transcription':' '.join(detected_transcription), 'status':status, 'path':row.path, 'wav_path':row.wav_path, 'sentence':sentence, 'phonetics':phonetics, 'word_idx':row.word_idx, 'file_idx':row.file_idx}
            detected_transcriptions.append(detected_transcription)
        else:
            detected_transcriptions.append([])
            d={'detected_phone':'','phones':row.phones, 'detected_transcription':'', 'status':status, 'path':row.path, 'wav_path':row.wav_path, 'sentence':sentence, 'phonetics':phonetics, 'word_idx':row.word_idx, 'file_idx':row.file_idx}
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


def unpredictable_vowels_from_audiobook_data(data_set='dev-clean', target_phones='IY1', target_graphemes='ea', alternatives=['IY1', 'EH1'], n=None):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)

    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    selection=libri_words_df[libri_words_df.phones.str.contains(target_phones) & libri_words_df.word.str.contains(target_graphemes)]

    if len(selection)>0:
        results_df=compute_prediction_results(selection, libri_words_df, target_phones=target_phones, alternatives=alternatives)
        all_results=performance_from_results(results_df)
    if n is None:
        pickle.dump(all_results, open('performance_results/unpredictaple_vowels_performance_'+data_set+'_'+target_graphemes+'_'+target_phones+'.p', 'wb'))
    else:
        pickle.dump(all_results, open('performance_results/unpredictaple_vowels_performance_'+data_set+'_'+target_graphemes+'_'+target_phones+'_from_'+str(n)+'egs.p', 'wb'))
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



def confusion_analysis_of_pContrast(phone_set, n=100):
    confusion_records=[]
    for p in phone_set:
        r=pContrast_from_audiobook_data(data_set='dev-clean', target_phones=p, alternatives=phone_set, n=n)
        # results[p]=r

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


def vowels_consonants_confusions_from_audiobook_data(n=100):
    import cmudict
    phones=cmudict.phones()

    cmu_vowels=[p[0]+'1' for p in phones if p[1][0]=='vowel']
    cmu_consonants=[p[0] for p in phones if p[1][0]!='vowel']
    # results={}

    vowel_confusion_df=confusion_analysis_of_pContrast(cmu_vowels, n=n)
    vowel_confusion_df_norm=(vowel_confusion_df.div(vowel_confusion_df.sum(axis=1), axis=0)*100).round(1)
    vowel_confusion_df.to_csv('performance_results/vowel_confusion_df.csv')
    vowel_confusion_df_norm.to_csv('performance_results/vowel_confusion_df_norm.csv')
    
    consonant_confusion_df=confusion_analysis_of_pContrast(cmu_consonants, n=n)
    consonant_confusion_df_norm=(consonant_confusion_df.div(consonant_confusion_df.sum(axis=1), axis=0)*100).round(1)
    consonant_confusion_df.to_csv('performance_results/consonant_confusion_df.csv')
    consonant_confusion_df_norm.to_csv('performance_results/consonant_confusion_df_norm.csv')

    
    plt.clf()
    sns.heatmap(vowel_confusion_df_norm, annot=True, cmap='YlGnBu')
    plt.savefig('performance_results/vowel_contrast_confusion.png')
    plt.clf()
    sns.heatmap(consonant_confusion_df_norm, annot=True, cmap='YlGnBu')
    plt.savefig('performance_results/consonant_contrast_confusion.png')

if False:
    # I checked that automatic annot version worked as well, and deprecated iContrast(p) module
    def iContrast_performance_test(path='../audio-with-analysis-ids/iContrast_data.csv', audio_path="../audio-with-analysis-ids/audio/"):
        # d=get_data()
        #d[d.analysisId==232][d.focusType=='wordstress']
        # a=get_wordStress_annotation()
        module='iContrast'
        # focus=module_name_to_focus_type[module]
        # # audio_path="../audio-with-analysis-ids/audio/"
        # d=d[d.focusType==focus]

        df=pd.read_csv(path)
        # only those with analysisID in 3 digits have a manual annotation
        df=df[df.analysisId>100]
        a,w_id,s_id=get_iContrast_annotations()
        
        # preds, statuss, GTs, errors=get_preds_and_GTs_from_data(d, a, module)

        GTs,preds=[],[]
        for i,row in tqdm(df.iterrows()):
            ground_truth=a[row.primaryKey]
            path=os.path.join(audio_path, row.primaryKey+'.wav')
            p=set_params(sentenceID=row.analysisId, waveFileAddress=path, module=module)
            status,result=iContrast(p)
            GTs.append(ground_truth)
            preds.append(result)

        diff=[abs(el[0]-el[1]) for el in zip(preds,GTs)]# if el[0]!=[]]
        # sum(diff)
        
        mismatch_rate=(len(preds)-len(diff))/len(preds)
        # bin_error_rate=stress_error/total_n
        example_error_rate=sum(diff)/(len(diff))
        print('n of examples:', len(preds))
        print('mismatch_rate:',mismatch_rate)
        # print('bin error rate:', bin_error_rate)
        print('example error rate:', example_error_rate)

    # This is obsolete and probably not working anymore
    def show_summary(data_set='dev-clean', target='AO1'):
        # performance=pickle.load(open('pContrast_performance_'+module+'_'+data_set+'_'+'_'.join(contrasted_phonemes)+'.p','rb'))
        performance=pickle.load(open('ed_performance_'+data_set+'.p','rb'))
        performances=performance['performances']
        results_dfs=performance['results_dfs']
        failure_dfs=performance['failure_dfs']
        lens={}
        for k,el in results_dfs.items(): lens[k]=len(el)

        phone_termination_dict, correct_alternatives=get_phone_termination_dict()

        # summary=pd.DataFrame.from_records([phone_termination_dict, lens, performances]).T
        # summary.columns=['termination', 'n of examples', 'performances']

        summary=pd.DataFrame.from_records([lens, performances]).T
        summary.columns=['n of examples', 'performances']

        # print(summary)

        # bad performances are mostly with D terminations
        # print(summary[summary.performances<0.78])
        # print(summary[summary.performances>0.78])

        print(summary.sort_values('performances').dropna())

        weighted_score=0
        tot=0
        for k in lens.keys():
            tot+=lens[k]
            weighted_score+=performances[k]*lens[k]
        weighted_score/=tot

        print('weighted_score:', weighted_score)
        
        # save examples of wrong detection
        
        # file_selection_path='file_selection'

        # # phonemes=rests_dfs.keys()
        # phonemes=['DH','JH','Z','V','L']

        # make_dir(file_selection_path)
        # for k in phonemes:
        #     if k in rests_dfs.keys():
        #         rests_df=rests_dfs[k]
        #         folder_copy_path=os.path.join(file_selection_path,k)
        #         make_dir(folder_copy_path)
        #         for i,r in tqdm(rests_df.iterrows()):
        #             # copy_path=os.path.join(folder_copy_path,os.path.split(r.wav_path)[-1])
        #             copy_path=os.path.join(folder_copy_path,r.detected_transcription+'.flac')

        #             if os.path.exists(r.wav_path) and not os.path.exists(copy_path):
        #                 shutil.copy(r.wav_path, copy_path)
                
    # Now sentenceStress_performance_test() uses automatic annotations
    def sentenceStress_automatic_annot_performance_test():
        d=get_data()
        #d[d.analysisId==232][d.focusType=='wordstress']
        a,textDict=get_sentenceStress_annotation()
        module='sentenceStress'
        focusType='sentencestress'
        d=d[d.focusType==focusType]
        audio_path="../audio-with-analysis-ids/audio/"
        preds, statuss, GTs, errors, p_errors = [],[],[],[],[]

        for i,r in d.iterrows():
            d=d.replace(d.loc[i].text,remove_special_characters(r.text))
        # d=d[d.text.isin(set([remove_special_characters(el) for el in textDict.values()]))]

        analysed_ids=[]
        for id,bin in a.items():
            # print(bin)
            row=d[d.text==remove_special_characters(textDict[id])]
            # if len(row)==0:print(id)
            if len(row)>0:
                # print(remove_special_characters(textDict[id]))
                # row=d[d.analysisId==id]  #[d.focusType=='wordstress']
                ground_truth=a[id]
                path=os.path.join(audio_path, row.primaryKey.values[0]+'.wav')
                # p=set_params(sentenceID=id, waveFileAddress=path, module='sentenceStress')

                try:
                    # status, scores_by_word=vowel_stresses(p)
                    status, scores_by_word=vowel_stresses_from_text_audio(row.text.values[0], path)
                    max_scores_by_word=[max(el) for el in scores_by_word]
                    binResult=np.zeros(len(max_scores_by_word))
                    binResult[np.argmax(max_scores_by_word)]=1

                    # status, pred=sentenceStress(p)
                    statuss.append(status)
                    preds.append(binResult)
                    GTs.append(ground_truth)
                    analysed_ids.append(id)
                except:
                    print('Error with:')
                    print(row)
                    errors.append(row)
                    # p_errors.append(p)
                    # import pdb;pdb.set_trace()
        
        print(preds)
        print(GTs)
        # print(GTs_from_phonetics)
        all_zero_baseline=[np.zeros(len(el)) for el in GTs]
        print(errors)
        print("all zero baseline")
        compute_errors(all_zero_baseline, GTs)

        print("algo performance")
        compute_errors(preds, GTs)

    def compute_prediction_for_row(row, phonetics, target_phones='IH0 D', 
                                alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']):
        """Compute the prediction of phonemeContrast module with the information of one row libri_words_df

        Args:
            row (dataframe row): [description]
            phonetics (list): list of phonemized words, e.g. phonetics=['K AE1 L IH0 K OW0', 'HH EH1 Z IH0 T EY2 T IH0 D']
            target_phones (str, optional): [description]. Defaults to 'IH0 D'.
            alternatives (list, optional): [description]. Defaults to ['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'].

        Returns:
            [type]: [description]
        """
        # set params and make label files for phonetics and grammar
        p=set_params(waveFileAddress=row.wav_path)
        p['inputPhoneticTranscription']='inputs/'+p['rand_fileName']+'.dct'
        p['inputGrammar']='inputs/'+p['rand_fileName']+'.txt'
        make_generic_dct_from_phonetics(phonetics=phonetics, word_id=row.word_idx, target_phones=target_phones, alternatives=alternatives, path=p['inputPhoneticTranscription'])
        make_grammar_from_dct(path_dct=p['inputPhoneticTranscription'],path_grammar=p['inputGrammar'])
        # run phonemeContrast module
        status, results= phonemeContrast(p)

        # status, results=phonemeContrast_from_phonetics_audio(phonetics=phonetics, audio_path=row.wav_path, target_phones=target_phones, alternatives=alternatives)

        return status, results


if __name__ == "__main__":
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
    pContrast_from_audiobook_data(target_phones='IH1', alternatives=['IH1', 'IY1'], n=20)
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

    # compute_prediction_for_row(row,row.phonetics,target_phones='DH', alternatives=['DH', 'TH'])

    pContrast_from_audiobook_data(target_phones='DH', alternatives=['DH', 'Z'], n=100)
    results=pContrast_from_audiobook_data(target_phones='TH', alternatives=['DH', 'TH'], n=100)
    # results=pContrast_from_audiobook_data(contrasted_phonemes=contrasted_phonemes, alternatives=alternatives, module="thContrast", n=100)
    
    
    row=results['failure'].iloc[-1]

    # phonetics_from_sentence(results['results_dfs']['TH'].iloc[0].sentence)
    target_phones='TH'
    # make_generic_dct_from_phonetics(phonetics=row.phonetics, word_id=row.word_id, target_phones=target_phones, alternatives=alternatives, path='test.dct')
    # make_grammar_from_dct()

    p=set_params(waveFileAddress=row.wav_path, sentenceID=None, basename='test', module="thContrast")
    # prepare_audio_file(p)
    status_audio, rID=prepare_audio_file(row.wav_path)
    p['rand_fileName']=rID
    make_dir(os.path.split(p['inputPhoneticTranscription'])[0])
    make_dir(os.path.split(p['inputGrammar'])[0])
    alternatives=['DH', 'TH']
    make_generic_dct_from_phonetics(phonetics=row.phonetics, word_id=row.word_id, target_phones=target_phones, alternatives=alternatives, path=p['inputPhoneticTranscription'])
    make_grammar_from_dct(path_dct=p['inputPhoneticTranscription'],path_grammar=p['inputGrammar'])
    status, results= phonemeContrast(p)
