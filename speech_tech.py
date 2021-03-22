import uuid
from scipy.io.wavfile import read, write
import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cmudict

from audio_processing import *
from htk_utils import *

def clean_temp_files():
    """Clean the files generated for and by the HTK model (as it uses input and output files)
    """
    os.system('rm inputs/*')
    os.system('rm results/*')

def set_params(
    # waveFileAddress='/root/flowchase/sent.wav',
    waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav',
    sentenceID=1,
    keep_logs = 0, # 1 if we want to keep logs of all processed files (e.g., for troubleshooting), 0 otherwise
    fs_target = 16000, # the target sampling frequency
    binResult = [],
    status = -1,
    result = [],
    modelName = 'libri',
    module='sentenceStress'):

    inputPhoneticTranscription_base = './lexicon/'+module+'/dct/phrase_'
    inputGrammar_base = './lexicon/'+module+'/grammar/phrase_'
    
    inputGrammar = '%s%d.txt' % (inputGrammar_base, sentenceID)
    inputPhoneticTranscription = '%s%d.dct' % (inputPhoneticTranscription_base, sentenceID)
    params={}
    params['waveFileAddress']=waveFileAddress
    params['sentenceID']=sentenceID
    params['keep_logs']=keep_logs
    params['fs_target']=fs_target
    params['inputPhoneticTranscription_base']=inputPhoneticTranscription_base
    params['inputGrammar_base']=inputGrammar_base
    params['binResult']=binResult
    params['status']=status
    params['result']=result
    params['inputPhoneticTranscription']=inputPhoneticTranscription
    params['inputGrammar']=inputGrammar
    params['modelName']=modelName

    return params

def createJSON(binResult, status, rand_fileName):
    """prepare json output message

    Args:
        binResult (): Result to send
        status (int): 0 if nothing went wrong, other values (negative) if not
        rand_fileName (string): random filename for temp files

    Returns:
        string: JSON string
    """
    
    if status != 0:
        res = '{"status": "%d"}'% status
        '%s%d.dct' % (inputPhoneticTranscription_base, sentenceID)
    else:
        oneString = '%d,' % binResult # string with all values separated by a comma
        res = '{"status": "%d", "sequence of words and pauses (marked as zeros)": "%s"}'% (status, oneString[:-1]) # (end-1): strip final comma
    
    return res
    
def load_audio(waveFileAddress, fs=16000):
    """Load audio, remove DC and normalize waveform

    Args:
        waveFileAddress (string): wav file address

    Returns:
        numpy array, int: waveform signal and frequency of sampling
    """
    # fs, s = read(waveFileAddress)
    s,fs=librosa.load(waveFileAddress, sr=fs)

    #trim silences
    # s, index = librosa.effects.trim(s, top_db=20)
    # remove DC
    s = s - s[int(0.15*len(s)):int(0.85*len(s))].mean() # we exclude 15% at each side that might contain buffer initialization/release noises
    # normalization
    s = 0.90*s/max(abs(s))
    return s, fs

# Modules

def get_annotated_signal(p=set_params()):
    if os.path.exists(p['waveFileAddress']):
        s,fs = load_audio(p['waveFileAddress'], fs=p['fs_target'])
    else:
        return "error: "+p['waveFileAddress']+" could not be loaded", None, None
    try:
        textgridData, cmdout2=get_textgrid_data(s,fs,p)
    except Exception as e: 
        print(e)
        return "error: could not get textgridData (check htk errors)", None, None
    return 0, textgridData, s

def chunking(
    p=set_params(module='chunking')
    ):
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return statut, []

    minSilDur = 0.090 # we ask for at least 90ms of silence
    silence_durations=textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,1]-textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,0]
    idx_to_filter=silence_durations[silence_durations>minSilDur].index

def sentenceStress(
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress')
    ):

    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return statut, []

    # find the number of words and phonemes per word in the input phrase
    # the '_' delimits the information of the word and number of phonemes, e.g. for "w2_7", there are 7 phonemes
    # if there is no '_', it is e.g. "o4". We put 1 in that case and else, the number of phonemes
    l=[el.split('_') for el in textgridData.iloc[:,2].tolist()]
    phonemesPerWord=[1 if len(el)==1 else int(el[-1]) for el in l]

    # each word start and end position expressed in samples
    startPositions_samples = (round(p['fs_target']*textgridData.iloc[:,0])+1).astype(int)
    stopPositions_samples = round(p['fs_target']*textgridData.iloc[:,1]).astype(int)

    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples.iloc[-1]<len(s), "The end of the last phoneme should be inside the signal"

    # TODO: verification of alignment

    # f0Samples=getIntonation(s, p['fs_target'])
    f0Samples=getIntonation(s.astype(np.float64), p['fs_target'])
    intensity=getIntensity(s, p['fs_target'])

    # plt.plot(f0Samples)
    # plt.show()
    
    # plt.plot(intensity)
    # plt.show()

    Dur=(textgridData.iloc[:,1]-textgridData.iloc[:,0])/(np.array(phonemesPerWord)+1)  # +1 assuming stressed phonemes = 2*other phonemes
    Dur=np.array(Dur.tolist())
    # plt.plot(Dur)
    # plt.show()

    startPositions_samples=startPositions_samples.tolist()
    stopPositions_samples=stopPositions_samples.tolist()
    
    Imax=np.zeros(len(textgridData))
    Fmax=np.zeros(len(textgridData))
    for i in range(len(textgridData)):
        if phonemesPerWord[i] == 1:
            Dur[i] = 0.6 * Dur[i]
        elif phonemesPerWord[i] == 2:
            Dur[i] = 0.8 * Dur[i]
        temp_sort = sorted(intensity[startPositions_samples[i]:stopPositions_samples[i]], reverse = True)
        Imax[i] = np.median(temp_sort[0:np.round(0.05*len(temp_sort)).astype(int)])
        temp_sort = sorted(f0Samples[startPositions_samples[i]:stopPositions_samples[i]], reverse = True)
        Fmax[i] = np.median(temp_sort[0:np.round(0.05*len(temp_sort)).astype(int)])
    
    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zFmax = normalize(Fmax)
    zDur = normalize(Dur)

    # plt.plot(zImax)
    # plt.show()
    
    # plt.plot(zFmax)
    # plt.show()

    # combine the features into a final result
    # weighted_score = (0.6*zImax + 0.4*zFmax + 0.2*zDur)/1.2;
    weighted_score = (0.6*zImax + 0.4*zFmax)/1.0 # needs fine-tuning once enough user data are available - in the long term consider additional futures and train a classifier with annotated user data
    rateThreshold = 1.01
    nWords=len(textgridData)
    binResult = np.zeros(nWords)
    
    plt.plot(weighted_score)
    plt.savefig('sentence_curve.png')

    if nWords == 1:
        binResult[0] = 1
    elif nWords == 2:
        sWS_id=np.argsort(weighted_score)[::-1]
        sWS_val=weighted_score[sWS_id]
        # TODO : I have to check if this make any sense. 
        # in the case with only two words, we ckeck if the higher is at least 1% higher than the other and put 1 there... (why this 1% ?)
        if sWS_val[0] > rateThreshold*sWS_val[1]:
            binResult[sWS_id[0]] = 1
    else:
        #TODO : this mean score has to be adapted because he use a normalization that led to values in a small range such
        # meanScore = 0.90*max(weighted_score)
        # meanScore= weighted_score.mean()
        meanScore=0.2
        if (weighted_score[0] > rateThreshold*weighted_score[1]) and (weighted_score[0] > meanScore):
            binResult[0] = 1
        elif (weighted_score[1] > rateThreshold*max(weighted_score[[0, 2]])) and (weighted_score[1] > meanScore):
            binResult[1] = 1
        elif (weighted_score[-1] > rateThreshold*weighted_score[-2]) and (weighted_score[-1] > meanScore):
            binResult[-1] = 1
        if nWords > 3:
            for i in range(2,nWords-1):
                if (weighted_score[i] > rateThreshold*max(weighted_score[[i-1, i+1]])) and (weighted_score[i] > meanScore):
                    binResult[i] = 1
    
    return "success", binResult
        
def wordStress(
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/WS_111_toothpaste.wav', module='wordStress')
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='wordStress')
    ):
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return statut, []

    # TODO: verification of alignment
    # TODO: check if all phonemes were found in the speech signal
    nEntries = len(textgridData)

    # get the line showing phoneme sequence
    phonetics=pd.read_csv(p['inputPhoneticTranscription'], header=None)

    dict_phones={}
    for i,r in phonetics.iterrows():
        #lines startint by d correspond to phonemes
        if r[0][0]=='d':
            line=r.values[0].split(' ')
            phone=line[0]
            info=line[1][1:-1]
            # print(phone)
            # print(info)
            dict_phones[info]=phone

    phone_seq=[dict_phones[r[2]] for i,r in textgridData.iterrows()]
    phone_set=set(phone_seq)
    count_total = len(phone_seq)
    count_unique = len(phone_set)

    # TODO: use these for verification 


    #extract the number of words by taking the index of the last one
    nWords=int(textgridData[2].iloc[-1].split('_')[0][1:])

    # for each entry, the second column is something like "w1_v1_1" or "w1_g1_1". the v is for vowel.
    # I tranform that to a table
    phone_df=pd.DataFrame([r[2].split('_') for i,r in textgridData.iterrows()])

    nVowelsPerWord=np.zeros(nWords, dtype=int)
    # for each entry of textgridData, we look at the word index, and if the phone is a vowel, we increment its number of vowels
    for i,r in phone_df.iterrows():
        word_index=int(r[0][1:])-1
        phone_type=r[1][0]
        # print(word_index)
        # print(phone_type)
        if phone_type=='v':
            nVowelsPerWord[word_index]+=1

    is_vowel=[r[2].split('_')[1][0]=='v' for i,r in textgridData.iterrows()]
    indxVowels = [i for i, val in enumerate(is_vowel) if val] 
    nVowels=len(indxVowels)

    # TODO: I do not know what this is but this is set to 1 when vowel ELSE corresponds to the integer at the end of "w1_v1_1"
    phoPerEntry=[int(r[2].split('_')[-1]) for i,r in textgridData.iterrows()]
    
    # TODO: check words duration
    # TODO: check acoustic scores

    

    # % check words duration
    # potErrors = 0;
    # grosErrors = 0;
    # for i = 1:nEntries
    #     if (textgridData{2}(i) - textgridData{1}(i) < 0.040*phoPerEntry(i)) || (textgridData{2}(i) - textgridData{1}(i) > 0.230*phoPerEntry(i))
    #         potErrors = potErrors + 1;
    #     end
    #     if textgridData{2}(i) - textgridData{1}(i) > 0.450*phoPerEntry(i)
    #         grosErrors = grosErrors + 1;
    #     end
    # end
    # if doVerification == 1 && (potErrors > 0.5*nEntries || grosErrors > 0)
    #     status = -500;
    #     result = createJSON(binResult, status, rand_fileName);
    #     return;
    # end

    # % check acoustic scores
    # if doVerification == 1 && mean(textgridData{4}) < 8.5
    #     status = -600;
    #     result = createJSON(binResult, status, rand_fileName);
    #     return;
    # end


    f0Samples=getIntonation(s, p['fs_target'])
    intensity=getIntensity(s, p['fs_target'])

    # extract features

    # each word start and end position expressed in samples
    startPositions_samples = (round(p['fs_target']*textgridData.iloc[:,0])+1).astype(int).tolist()
    stopPositions_samples = round(p['fs_target']*textgridData.iloc[:,1]).astype(int).tolist()
    
    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples[-1]<len(s), "The end of the last phoneme should be inside the signal"

    Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
    sylType=np.zeros(nVowels)
    for i in range(nVowels):
        range_vowel=range(startPositions_samples[indxVowels[i]], stopPositions_samples[indxVowels[i]])
        Ivowel=intensity[range_vowel]
        Fvowel=f0Samples[range_vowel]
        
        Imax.append(max(Ivowel))
        Imean.append(np.mean(Ivowel))
        Fmax.append(max(Fvowel))
        Fmean.append(np.mean(Fvowel))

        Dur.append(textgridData[1].iloc[indxVowels[i]]-textgridData[0].iloc[indxVowels[i]])

        textgridData[2].iloc[indxVowels[i]]

        syltype_phone=int(phone_df[2].iloc[indxVowels[i]])
        if syltype_phone == 2:  # the sylType is 0 for unstressed, 0.5 for secondary stressed syllables and 1 for primary stressed syllables
            sylType[i] = 0.5
        else:
            sylType[i]=syltype_phone
    
    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zImean = normalize(Imean)
    zFmax = normalize(Fmax)
    zFmean = normalize(Fmean)
    zDur = normalize(Dur)

    # combine the features
    weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur + 0.4*sylType)/3.6  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data

    plt.plot(weighted_score)
    plt.savefig('word_curve.png')
    # chose prominent vowel per word
    binResult=np.zeros(len(weighted_score))
    for i in range(nWords):
        # make the indices range of syllables (or vowels) in the sentence
        if i == 0:
            syl_id = np.arange(nVowelsPerWord[i])
        else:
            syl_id = np.arange(nVowelsPerWord[i])+sum(nVowelsPerWord[:i])
        maxscore = max(weighted_score[syl_id]) # the max score over a word
        for j in range(len(syl_id)):
            if weighted_score[syl_id[j]] == maxscore:
                binResult[syl_id[j]] = 1
            elif weighted_score[syl_id[j]] >= 0.98*maxscore: # syllables close to the max score (i.e., 98% of its value) are also considered as stressed
                binResult[syl_id[j]] = 1
            else:
                binResult[syl_id[j]] = 0
    
    return "success", binResult

def number_and_indices(textgridData, char='w'):
    #take index when the first character is char
    idxs=[i if el[0]==char else np.nan for i,el in enumerate(textgridData.iloc[:,2].tolist())]
    # remove nans (x=nan is the only value such that x!=x)
    idxs = [x for x in idxs if x==x]
    n=len(idxs)
    return n, idxs

def iConstrast(
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")
    ):
    
    #p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_sleep.wav', module="iContrast")
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return statut, []

    # find the number of words and phonemes per word in the input phrase
    # the '_' delimits the information of the word and number of phonemes, e.g. for "w2_7", there are 7 phonemes
    # if there is no '_', it is e.g. "o4". We put 1 in that case and else, the number of phonemes
    # l=[el.split('_') for el in textgridData.iloc[:,2].tolist()]
    # phonemesPerWord=[1 if len(el)==1 else int(el[-1]) for el in l]

    # TODO: this is for the verification and it is not finished
    nWords, indxWords = number_and_indices(textgridData, 'w')
    nPho, indxPho = number_and_indices(textgridData, 'p')
    nSil, indxSil = number_and_indices(textgridData, 's')

    # TODO: weird, he does a loop for every phoneme, check if it is a short or long, and repeat, but does not record results.
    # only the last result will be kept (maybe it works because there is only one vowel that needs to be checked)
    for i,el in enumerate(textgridData.iloc[:,2].tolist()):
        print(el[-2:])
        if el[-2:]=='*s':
            binResult=0
        elif el[-2:]=='*l':
            binResult=1
            if textgridData.iloc[i, 1]-textgridData.iloc[i, 0]<0.07:
                binResult=0
    
    return "success", binResult

def oConstrast():#p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")):
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Laaw_MP3.mp3', module="oContrast")
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Laaw_WAV.wav', module="oContrast")
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Law_WAV.wav', module="oContrast")
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Low_WAV.wav', module="oContrast")
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return statut, []

def edAnalysis(
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/ed_acceptEED.wav', module="edAnalysis")
    ):
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return statut, []
    
    # TODO: this is for the verification and it is not finished
    nWords, indxWords = number_and_indices(textgridData, 'w')
    nPho, indxPho = number_and_indices(textgridData, 'p')
    nSil, indxSil = number_and_indices(textgridData, 's')

    binResult = 0
    # check pronunciation
    for i in range(nPho):
        if '*cor' in textgridData[2][indxPho[i]]:
            status = 0
            binResult = 2
        elif '*err' in textgridData[2][indxPho[i]]:
            status = 0
            binResult = 1
    
    return "success", binResult


# Data processing
def get_data(path_to_json='../audio-with-analysis-ids/data.json'):
    """Get a dataframe containing info of audio recordings with sentence ids

    Args:
        path_to_json (str, optional): [description]. Defaults to '../audio-with-analysis-ids/data.json'.

    Returns:
        DataFrame: [description]
    """
    data=pd.read_json(path_to_json)
    return data

def get_wordStress_annotation(path='../audio-with-analysis-ids/wordStress_annotations.csv'):
    d=get_data()
    d=d[d.focusType=="wordstress"]

    df=pd.read_csv(path)

    wordStress_annotations={}
    for i,r in d.iterrows():
        analysisId=r.analysisId
        first_digit=int(analysisId/100)

        if first_digit>0:
            col=df[str(first_digit)]

            text=d[d.analysisId==analysisId].text.values[0]

            col_idx=df.columns.tolist().index(str(first_digit))
            match=col[col==text]

            if len(match)>0:
                row_idx=match.index[0]

                bin_annotation=df.iloc[row_idx,col_idx+3]
                bin_annotation_list=[int(el) for el in bin_annotation.split('-')]

                wordStress_annotations[analysisId]=bin_annotation_list
            else:
                print('Text not found')
                print(r)
        else:
            print('index with less than 3 digits')
            print(r)
    
    return wordStress_annotations

def get_cmudict_info(word='university'):
    """get the first possible phonetisation of a word from cmudict

    Args:
        word (str, optional): input. Defaults to 'university'.
    Returns:
        list: phonemes and a number for each vowle indicating stress: 0=no stress, 1=primary stress, 2=secondary stress
    """
    return cmudict.dict()[word][0]

import re

def remove_special_characters(sentence="Where's the best place to have coffee ?"):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    sentence = re.sub(chars_to_ignore_regex, '', sentence).lower()
    return sentence

def phonetics_from_sentence(sentence="Where's the best place to have coffee ?"):
    sentence=remove_special_characters(sentence)
    words=sentence.split(' ')
    # drop empty strings
    words = list(filter(None, words))
    words_phones=[]
    for w in words:
        phones=get_cmudict_info(w)
        words_phones.append(phones)
    return words_phones

def word_stress_from_cmu(phonetics=['K', 'AA1', 'F', 'IY0']):
    binResult=[]
    for p in phonetics:
        try:
            stress_info=int(p[-1])
            if stress_info==1: #primary stress
                binResult.append(1)
            else:
                binResult.append(0)
        except:
            # it's not a vowel if last character is not a number
            pass
    return binResult

def word_stress_from_text(sentence="Where's the best place to have coffee ?"):
    phonetics=phonetics_from_sentence(sentence) #return a list of phoneme list (by word)
    result = []
    for el in phonetics:
        result+=el
    binResult=word_stress_from_cmu(result)
    return binResult

if __name__ == "__main__":
    # execute only if run as a script

    # Test performance of wordStress module

    d=get_data()
    #d[d.analysisId==232][d.focusType=='wordstress']

    a=get_wordStress_annotation()

    d=d[d.focusType=='wordstress']
    audio_path="../audio-with-analysis-ids/audio/"

    preds=[]
    GTs=[]
    GTs_from_phonetics=[]
    errors=[]
    p_errors=[]
    for id,bin in a.items():
        # print(bin)
        row=d[d.analysisId==id]  #[d.focusType=='wordstress']
        ground_truth=a[id]
        path=os.path.join(audio_path, row.primaryKey.values[0]+'.wav')
        p=set_params(sentenceID=id, waveFileAddress=path, module='wordStress')
        try:
            _, pred=wordStress(p)
            # print('Ground truth:')
            # print(ground_truth)
            # print('prediction:')
            # print(pred)
            preds.append(pred)
            GTs.append(ground_truth)
            GTs_from_phonetics.append(word_stress_from_text(row.text.values[0]))
        except:
            print('Error with:')
            print(row)
            errors.append(row)
            p_errors.append(p)
            # import pdb;pdb.set_trace()
    
    print(preds)
    print(GTs)
    print(GTs_from_phonetics)
    print(errors)
    stress_error_rate=stress_error/total_n_vowel
    print('stress error rate:', stress_error_rate)
    # errors=pd.concat(errors)

    diffs=[]
    total_n_vowel=0
    stress_error=0
    for i,pred in enumerate(preds):
        diff=abs(pred-GTs_from_phonetics[i])
        diffs.append(diff)
        total_n_vowel+=len(pred)
        stress_error+=diff.sum()
    stress_error_rate=stress_error/total_n_vowel

    clean_temp_files()