from speech_tech import *

# Performance tests

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
    """This function works for wordStress and sentenceStress. It uses the data.json file containing information 
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
            pdb.set_trace()
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
    a={}
    for i,row in tqdm(d.iterrows()):
        # p=set_params(sentenceID=row.analysisId, module="iContrast")
        
        # # get the line showing phoneme sequence
        # phonetics=pd.read_csv(p['inputPhoneticTranscription'], header=None)

        # dict_phones={}
        # for i,r in phonetics.iterrows():
        #     #lines starting by d correspond to phonemes
        #     if r[0][0]=='p':
        #         line=r.values[0].split(' ')
        #         phone=line[-1]
        #         info=line[1][1:-1]
        #         print(phone)
        #         print(info)
        #         dict_phones[info]=phone
        #     if r[0][0]=='w':
        #         line=r.values[0].split(' ')
        #         phone=line[-1]
        #         info=line[1][1:-1]
        #         print(phone)
        #         print(info)
        #         dict_phones[info]=phone

        # dummy labels for now
        a[row.analysisId]=0
        phonetics_from_sentence(row.text)
        # a[row.analysisId]=word_stress_from_text(row.text) 
        print(row)
        print(a[row.analysisId])
    
    preds, statuss, GTs, errors=get_preds_and_GTs_from_data(d, a, module)

    diff=[abs(el[0]-el[1]) for el in zip(preds,GTs) if el[0]!=[]]
    sum(diff)

    
    mismatch_rate=(len(preds)-len(diff))/len(preds)
    bin_error_rate=stress_error/total_n
    example_error_rate=n_example_error/(len(preds)-len(mismatches.keys()))
    print('n of examples:', len(preds))
    print('mismatch_rate:',mismatch_rate)
    print('bin error rate:', bin_error_rate)
    print('example error rate:', example_error_rate)

    clean_temp_files()

