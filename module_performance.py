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


def rename_underscore_to_dash(files=glob("/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/*/*/*")):
    """Replace all "-" with "_" in filenames because I had incinnsistent namings. This could be generalized if needed 
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
    for i,pred in enumerate(preds):
        if len(pred)==len(GTs[i]):
            diff=abs(pred-GTs[i])
            diffs.append(diff)
            total_n+=len(pred)
            stress_error+=diff.sum()
            if diff.sum()>0: n_example_error+=1
        else:
            mismatches[i]=(pred,GTs[i])
    mismatch_rate=len(mismatches.keys())/len(preds)
    bin_error_rate=stress_error/total_n
    example_error_rate=n_example_error/(len(preds)-len(mismatches.keys()))
    print('n of examples:', len(preds))
    print('mismatch_rate:',mismatch_rate)
    print('bin error rate:', bin_error_rate)
    print('example error rate:', example_error_rate)

module_name_to_focus_type={'wordStress':'wordstress',
    'sentenceStress':'sentencestress',
    'iContrast':'shortIlongI'}
def get_preds_and_GTs_from_data(d, a, module, audio_path="../audio-with-analysis-ids/audio/"):
    """This function works for wordStress? sentenceStress and iContrast. It uses the data.json file containing information 
    from dynamoDB: (audio, text, sentenceID, module), and get the corresponding dct files to run prediction of a module on it.

    Args:
        d (dataframe): information  from dynamoDB audio, text, sentenceID, module
        a (dict): annotations (ground truth)
        module (string): module name
        audio_path (str, optional): [description]. Defaults to "../audio-with-analysis-ids/audio/".

    Returns:
        lists: [description]
    """
    focus=module_name_to_focus_type[module]
    preds, statuss, GTs, errors, p_errors = [],[],[],[],[]
    # pdb.set_trace()

    for i,row in tqdm(d.iterrows()):
        # print(bin)
        # row=d[d.analysisId==id]  #[d.focusType=='wordstress']
        id=row.analysisId
        # if len(row)>0 and id in a:
        if id in a:
            ground_truth=a[id]
            path=os.path.join(audio_path, row.primaryKey+'.wav')
            p=set_params(sentenceID=id, waveFileAddress=path, module=module)
            # pdb.set_trace()
            status, textgridData, s = get_annotated_signal(p)
            try:
                # this calls the function with the name of the module
                status, pred=globals()[module](p)
                statuss.append(status)
                preds.append(pred)
                GTs.append(ground_truth)
                # GTs_from_phonetics.append(word_stress_from_text(row.text))
            except:
                print('Error with:')
                print(row)
                errors.append(row)
                p_errors.append(p)
                # import pdb;pdb.set_trace()
    
    print(preds)
    print(GTs)
    # print(GTs_from_phonetics)
    print(errors)

    # pdb.set_trace()
    # compute_errors(preds,GTs)

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
    compute_errors(preds,GTs)

    clean_temp_files()

def sentenceStress_performance_test():
    d=get_data()
    #d[d.analysisId==232][d.focusType=='wordstress']
    a,textDict=get_sentenceStress_annotation()
    module='sentenceStress'
    focusType='sentencestress'
    d=d[d.focusType==focusType]
    audio_path="../audio-with-analysis-ids/audio/"
    preds, statuss, GTs, errors, p_errors = [],[],[],[],[]
    GTs_from_phonetics=[]

    for i,r in d.iterrows():
        d=d.replace(d.loc[i].text,remove_special_characters(r.text))
    # d=d[d.text.isin(set([remove_special_characters(el) for el in textDict.values()]))]

    for id,bin in a.items():
        # print(bin)
        row=d[d.text==remove_special_characters(textDict[id])]
        if len(row)>0:
            # print(remove_special_characters(textDict[id]))
            # row=d[d.analysisId==id]  #[d.focusType=='wordstress']
            ground_truth=a[id]
            path=os.path.join(audio_path, row.primaryKey.values[0]+'.wav')
            p=set_params(sentenceID=id, waveFileAddress=path, module='sentenceStress')
            try:
                status, pred=sentenceStress(p)
                statuss.append(status)
                preds.append(pred)
                GTs.append(ground_truth)
                # GTs_from_phonetics.append(word_stress_from_text(row.text.values[0]))
            except:
                print('Error with:')
                print(row)
                errors.append(row)
                p_errors.append(p)
                # import pdb;pdb.set_trace()
    
    print(preds)
    print(GTs)
    # print(GTs_from_phonetics)
    print(errors)
    compute_errors(preds, GTs)
    clean_temp_files()

def iContrast_performance_test():
    d=get_data()
    #d[d.analysisId==232][d.focusType=='wordstress']
    # a=get_wordStress_annotation()
    module='iContrast'
    focus=module_name_to_focus_type[module]
    # audio_path="../audio-with-analysis-ids/audio/"
    d=d[d.focusType==focus]
    a=get_iContrast_annotations()
    
    preds, statuss, GTs, errors=get_preds_and_GTs_from_data(d, a, module)

    diff=[abs(el[0]-el[1]) for el in zip(preds,GTs) if el[0]!=[]]
    sum(diff)
    
    mismatch_rate=(len(preds)-len(diff))/len(preds)
    # bin_error_rate=stress_error/total_n
    example_error_rate=sum(diff)/(len(diff))
    print('n of examples:', len(preds))
    print('mismatch_rate:',mismatch_rate)
    # print('bin error rate:', bin_error_rate)
    print('example error rate:', example_error_rate)

    clean_temp_files()

def edAnalysis_performance_test():
    files=glob("/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/*/*/*wav")
    xls='/mnt/c/Users/noe_t/Downloads/20 words - research-20210401T100243Z-001/20 words - research/20 problems-research.xlsx'
    a=pd.read_excel(xls)
    path='audio_recordings/edAnalysis'
    ed_sentenceID=pd.read_csv(os.path.join('./lexicon/edAnalysis/dct','ed_sentenceID.csv'))
    results, tgs, analyzed_files=[],[],[]
    for i,r in ed_sentenceID.iterrows():
        text=r[0].split('.')[0]
        text_dashed='-'.join(text.split(' '))
        print(text_dashed)
        for f in files:
            if text_dashed in os.path.split(f)[-1]:
                sentence_id_from_text=ed_sentenceID[ed_sentenceID['Unnamed: 0'].str.contains(text)]['Unnamed: 1'].values[0]
                p=set_params(sentenceID=sentence_id_from_text, waveFileAddress=f, module="edAnalysis")
                # p['inputPhoneticTranscription']='./lexicon/edAnalysis/dct/'+text+'.dct'
                # p['inputGrammar']='./lexicon/edAnalysis/grammar/'+text+'.txt'
                status, textgridData, s = get_annotated_signal(p)
                results.append(edAnalysis(p)) 
                tgs.append(textgridData)
                analyzed_files.append(f)

    df_results=pd.DataFrame()
    df_results['wav']=[os.path.split(f)[-1] for f in analyzed_files]
    df_results['prediction']=pd.DataFrame(results).iloc[:,1]
    df_results[df_results.wav.str.contains('F1')]
    df_results[df_results.wav.str.contains('F2')]
    df_results[df_results.wav.str.contains('M1')]
    df_results[df_results.wav.str.contains('M2')]

    clean_temp_files()



def compute_prediction_results(selection, libri_words_df, termination='IH0 D'):
    detected_transcriptions=[]
    statuss=[]
    match=[]
    result_records=[]
    for i,row in tqdm(selection.iterrows()):
        sentence=get_sentence(row.path)
        # retrieve phonetics by word that ins not available directly from textgrids, but can be extracted from the dataframe
        # as I already extracted phonemes for each word using overlapping in timings
        phonetics=[]
        for w in sentence.split(' '):
            phonetics.append(libri_words_df[libri_words_df.word==w].iloc[0,:].phones)

        # set params and make label files for phonetics and grammar
        p=set_params(waveFileAddress=row.wav_path, sentenceID=None, basename='test', module='edAnalysis')
        make_generic_dct_from_phonetics(phonetics=phonetics, word_id=row.word_idx, termination=termination, path=p['inputPhoneticTranscription'])
        make_grammar_from_dct(path_dct=p['inputPhoneticTranscription'],path_grammar=p['inputGrammar'])

        # run phonemeContrast module
        status, results= phonemeConstrast(p)
        # status, textgridData, s=get_annotated_signal(p)
        statuss.append(status)
        if results!=[]:
            textgridData=results[0]
            detected_transcription=results[1]
            # phonetics=pd.read_csv(p['inputPhoneticTranscription'], header=None)
            match.append(row.phones==' '.join(detected_transcription))
            d={'phones':row.phones,'detected_transcription':' '.join(detected_transcription), 'status':status, 'path':row.path, 'wav_path':row.wav_path}
            detected_transcriptions.append(detected_transcription)
        else:
            detected_transcriptions.append([])
            match.append(False)
            d={'phones':row.phones, 'detected_transcription':'', 'status':status, 'path':row.path, 'wav_path':row.wav_path}
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

def performance_test(selection, results_df, correct_terminations, termination):
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

def edAnalysis_from_audiobook_data(data_set='dev-clean'):
    libri_words_df=build_librispeech_words_df(data_set=data_set)
    phone_termination_dict, correct_alternatives=get_phone_termination_dict()

    results_dfs={}
    rests_dfs={}
    performances={}

    for pretermination, termination in tqdm(phone_termination_dict.items()):
        full_termination=' '+' '.join([pretermination,termination])
        correct_terminations=[' '+' '.join([pretermination,el]) for el in correct_alternatives[termination]]
        selection=libri_words_df[libri_words_df.phones.str.endswith(full_termination)]

        if len(selection)>0:
            results_df=compute_prediction_results(selection, libri_words_df, termination=termination)
            rate, rest=performance_test(selection, results_df, correct_terminations, termination)

            results_dfs[pretermination]=results_df
            rests_dfs[pretermination]=rest
            performances[pretermination]=rate
    
    pickle.dump({'results_dfs':results_dfs, 'rests_dfs':rests_dfs,  'performances':performances}, open('ed_performance_'+data_set+'.p', 'wb'))

def show_summary(data_set='dev-clean'):
    ed_performance=pickle.load(open('ed_performance_'+data_set+'.p','rb'))
    performances=ed_performance['performances']
    results_dfs=ed_performance['results_dfs']
    rests_dfs=ed_performance['rests_dfs']
    lens={}
    for k,el in results_dfs.items(): lens[k]=len(el)

    phone_termination_dict, correct_alternatives=get_phone_termination_dict()
    summary=pd.DataFrame.from_records([phone_termination_dict, lens, performances]).T
    summary.columns=['termination', 'n of examples', 'performances']

    print(summary)

    # bad performances are mostly with D terminations
    print(summary[summary.performances<0.78])
    print(summary[summary.performances>0.78])

    print(summary.sort_values('performances').dropna())

    weighted_score=0
    tot=0
    for k in lens.keys():
        tot+=lens[k]
        weighted_score+=performances[k]*lens[k]
    weighted_score/=tot

    print('weighted_score:', weighted_score)
    
    # save examples of wrong detection
    def make_dir(path):
        if not os.path.exists(path): os.makedirs(path)
    
    file_selection_path='file_selection'

    # phonemes=rests_dfs.keys()
    phonemes=['DH','JH','Z','V','L']

    make_dir(file_selection_path)
    for k in phonemes:
        rests_df=rests_dfs[k]
        folder_copy_path=os.path.join(file_selection_path,k)
        make_dir(folder_copy_path)
        for i,r in tqdm(rests_df.iterrows()):
            # copy_path=os.path.join(folder_copy_path,os.path.split(r.wav_path)[-1])
            copy_path=os.path.join(folder_copy_path,r.detected_transcription+'.flac')

            if os.path.exists(r.wav_path) and not os.path.exists(copy_path):
                shutil.copy(r.wav_path, copy_path)
            


