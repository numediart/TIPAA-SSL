import uuid
from scipy.io.wavfile import read, write
import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cmudict
from time import time

from tqdm import tqdm

from audio_processing import load_audio, getIntonation, getIntensity, normalize, getf0Samples
from htk_utils import process_grammar, htk_recognition, get_textgrid_data

def clean_temp_files():
    """Clean the files generated for and by the HTK model (as it uses input and output files)
    TODO: This is probably dangerous when we use it in parallel, multithreading... 
    In the future, just delete the specific files after processing, by putting name in parameters (inputs and results files)
    """
    os.system('rm inputs/*')
    os.system('rm results/*')

def set_params(
    # waveFileAddress='/root/flowchase/sent.wav',
    waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav',
    sentenceID=1,
    basename='phrase_',
    fs_target = 16000, # the target sampling frequency
    modelName = 'libri',
    module='sentenceStress'):
    """Set parameters for an analysis task: wav, dct and grammar files as well as module to use

    Args:
        waveFileAddress (str, optional): [description]. Defaults to 'audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav'.
        sentenceID (int, optional): [description]. Defaults to 1.
        fs_target (int, optional): [description]. Defaults to 16000.
        module (str, optional): [description]. Defaults to 'sentenceStress'.

    Returns:
        dict
    """
    inputPhoneticTranscription_base = './lexicon/'+module+'/dct/'+basename
    inputGrammar_base = './lexicon/'+module+'/grammar/'+basename

    
    
    if not (sentenceID is None):
        inputGrammar = '%s%d.txt' % (inputGrammar_base, sentenceID)
        inputPhoneticTranscription = '%s%d.dct' % (inputPhoneticTranscription_base, sentenceID)
    else:
        inputGrammar = '%s.txt' % (inputGrammar_base)
        inputPhoneticTranscription = '%s.dct' % (inputPhoneticTranscription_base)
    params={}
    params['waveFileAddress']=waveFileAddress
    params['fs_target']=fs_target
    params['inputPhoneticTranscription']=inputPhoneticTranscription
    params['inputGrammar']=inputGrammar
    params['modelName']=modelName
    params['rand_fileName']=str(uuid.uuid4())

    return params

def get_annotated_signal(p=set_params()):
    """Load audio file and annotation files corresponding to parameters, 
    and calls "textgridData" to obtain htk predictions of phonemes and
    their timings

    Args:
        p (dictionary, optional): global parameters (wav, dct, grammar file paths, ). Defaults to set_params().

    Returns:
        status, textgridData, s (int, DataFrame, np array): textgridData contains phonetic predictions 
        from htk model with their timings and log probability
    """
    if os.path.exists(p['waveFileAddress']):
        s,fs = load_audio(p['waveFileAddress'], fs=p['fs_target'])
    else:
        return "error: "+p['waveFileAddress']+" could not be loaded", None, None
    try:
        textgridData, cmdout2=get_textgrid_data(s,fs,p)
    except Exception as e: 
        print(e)
        return "error: could not get textgridData (check htk errors)", None, None
    
    # if textgridData.iloc[:,3].mean()<8.5:
    if textgridData.iloc[:,3].mean()<5:
        return "success: low posterior probability, the model is not confident with the recognition", None, None
    
    # each row is True if out of vocabulary, False if it is a detected phoneme or word
    is_out_of_vocabulary=textgridData.iloc[:,2].str[:1].str.contains('o')

    if is_out_of_vocabulary.product():
        return "success: all of the elements were out of vocabulary", None, None
    # if is_out_of_vocabulary.sum():
    #     return "error: at least one element was out of vocabulary", None, None

    return 0, textgridData, s

def verification_n_of_phoneme(textgridData, p):

    # TODO: check if all phonemes were found in the speech signal
    # TODO: this assumes every phoneme of the exercise are different... As I implemented 
    # the detection of out of vocabulary elements, maybe I can remove this

    nEntries = len(textgridData)

    # get the line showing phoneme sequence
    phonetics=pd.read_csv(p['inputPhoneticTranscription'], header=None)

    dict_phones={}
    for i,r in phonetics.iterrows():
        #lines starting by d correspond to phonemes
        if r[0][0]=='d':
            line=r.values[0].split(' ')
            phone=line[0]
            info=line[1][1:-1]
            # print(phone)
            # print(info)
            dict_phones[info]=phone
        if r[0][0]=='o':
            dict_phones[r[0].split(' ')[0]]=r[0].split(' ')[0]

    phone_seq=[dict_phones[r[2]] for i,r in textgridData.iterrows()]
    phone_set=set(phone_seq)
    count_total = len(phone_seq)
    count_unique = len(phone_set)
    return nEntries==0 or count_total!=nEntries or count_unique!=nEntries

def chunking(
    p=set_params(module='chunking')
    ):
    """The goal of this module is to find pauses in a longer sequence such as a read paragraph.
    TODO: get_textgrid_data function filter out silences, so it won't work. Start from htk_recognition function in htk_utils

    Args:
        p ([type], optional): [description]. Defaults to set_params(module='chunking').

    Returns:
        [type]: [description]
    """
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return status, []

    minSilDur = 0.090 # we ask for at least 90ms of silence
    silence_durations=textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,1]-textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,0]
    idx_to_filter=silence_durations[silence_durations>minSilDur].index

def sentenceStress(
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress')
    # p=set_params(sentenceID=1, waveFileAddress='audio_recordings/WS_111_toothpaste.wav', module='sentenceStress')
    ):
    """Use textgridData to have the timings of words and compute prosody features (intesity, pitch) to compute 
    a value by word representing a stress intensity

    Args:
        p (dict, optional): global parameters. Defaults to set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress').

    Returns:
        string, list of binaries: status, stress results by word (0=no stress,  1=stress)
    """
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return status, []

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
        #TODO : this mean score has to be adapted because he uses a normalization that led to values in a small range 
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
    p=set_params(sentenceID=112, waveFileAddress='audio_recordings/WS_111_toothpaste.wav', module='wordStress')
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='wordStress')
    ):
    """Use textgridData to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
    a value by vowel representing a stress intensity

    Args:
        p (dict, optional): global parameters. Defaults to set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress').

    Returns:
        string, list of binaries: status, stress results by vowel (0=no stress,  1=stress)
    """
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return status, []

    # TODO: use these for verification 
    # if verification_n_of_phoneme(textgridData, p):
    #     return "error: inconsistent number of detected phonemes", []

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

    # number of phoneme per entry in dct file this is set to 1 when vowel ELSE corresponds to the integer at the end of "w1_v1_1"
    phoPerEntry=[int(r[2].split('_')[-1]) for i,r in textgridData.iterrows()]
    
    # TODO: check words duration

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

        # textgridData[2].iloc[indxVowels[i]]

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
    """get total number and indices of entries starting with char 

    Args:
        textgridData (DataFrame): phonetic prediction of htk model with timings and log probabilities
        char (str, optional): Defaults to 'w'.

    Returns:
        int, list of int: number and list of indices
    """
    #take index when the first character is char
    idxs=[i if el[0]==char else np.nan for i,el in enumerate(textgridData.iloc[:,2].tolist())]
    # remove nans (x=nan is the only value such that x!=x)
    idxs = [x for x in idxs if x==x]
    n=len(idxs)
    return n, idxs

def iContrast(
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")
    ):
    """Use textgridData to have the timings of vowels and check if the vowel detected is a short or long vowel.

    Args:
        p ([type], optional): [description]. Defaults to set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast").

    Returns:
        int: 0 if short, 1 if long
    """
    #p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_sleep.wav', module="iContrast")
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return status, []

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
        # print(el[-2:])
        if el[-2:]=='*s':
            binResult=0
        elif el[-2:]=='*l':
            binResult=1
            if textgridData.iloc[i, 1]-textgridData.iloc[i, 0]<0.07:
                binResult=0
    
    return "success", binResult


def prosody_by_phone(p):
    status, textgridData, s = get_annotated_signal(p)

    f0Samples=getIntonation(s, p['fs_target'])
    vuv=np.nan_to_num(getf0Samples(s, p['fs_target']), nan=0).astype(bool)

    intensity=getIntensity(s, p['fs_target'])
    detected_phonemes=textgridData[textgridData.iloc[:,2].str[0]=='p']
    Imax,Imean,Fmax,Fmean,Dur,voicing=[],[],[],[],[],[]
    for i,r in detected_phonemes.iterrows():
        start=round(p['fs_target']*r[0])
        end=round(p['fs_target']*r[1])
        F_phone=f0Samples[start:end]
        I_phone=intensity[start:end]
        vuv_phone=vuv[start:end]
        Fmax.append(max(F_phone))
        Imax.append(max(I_phone))
        Fmean.append(np.mean(F_phone))
        Imean.append(np.mean(I_phone))
        Dur.append(end-start)
        voicing.append(sum(vuv_phone)/len(vuv_phone))
    d={}
    d['Imax']=Imax
    d['Imean']=Imean
    d['Fmax']=Fmax
    d['Fmean']=Fmean
    d['Dur']=Dur
    d['voicing']=voicing
    return "success", [textgridData, d]


def phonemeContrast(#p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")):
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Laaw_MP3.mp3', module="oContrast")
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Laaw_WAV.wav', module="oContrast")
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Law_WAV.wav', module="oContrast")
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Low_WAV.wav', module="oContrast")
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")
    ):
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return status, []
    
    detected_phonemes=textgridData[textgridData.iloc[:,2].str[0]=='p']    
    phonetics=pd.read_csv(p['inputPhoneticTranscription'], header=None, sep='(\] |\[)', engine='python')

    detected_transcription=[]
    for r in detected_phonemes.iloc[:,2]:
        detected_transcription.append(phonetics[phonetics.iloc[:,2]==r][4].values[0])
    
    return "success", [textgridData, detected_transcription]

    # phonetics[phonetics[0].str[0]=='p'].apply(lambda r:r.str.split(' '))
    # pd.DataFrame(phonetics[phonetics[0].str[0]=='p'].apply(lambda r:r.str.split(' ')).iloc[:,0].tolist())
    # for i,r in phonetics[phonetics[0].str[0]=='p'].iterrows():
    # dict_phones={}
    # for i,r in phonetics.iterrows():
    #     #lines starting by d correspond to phonemes
    #     if r[0][0]=='p':
    #         line=r.values[0].split(' ')
    #         phone=line[0]
    #         info=line[1][1:-1]
    #         # print(phone)
    #         # print(info)
    #         dict_phones[info]=phone
    #     elif r[0][0]=='o' or r[0][0]=='s':
    #         dict_phones[r[0].split(' ')[0]]=r[0].split(' ')[0]
    #     else:
    #         dict_phones[r[0].split(' ')[1][1:-1]]=r[0].split(' ')[-1]

    # for i,r in textgridData.iterrows():
    #     print(r[2])
    #     print(dict_phones[r[2]])

def edAnalysis(
    # p=set_params(sentenceID=1, waveFileAddress='audio_recordings/edAnalysis/1_err.wav', module="edAnalysis")
    # p=set_params(sentenceID=1, waveFileAddress='audio_recordings/edAnalysis/1_corr.wav', module="edAnalysis")
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/ed_accepted.wav', module="edAnalysis")
    # p=set_params(sentenceID=2, waveFileAddress='audio_recordings/turned_around.mp3', module="edAnalysis")
    # p=set_params(sentenceID=2, waveFileAddress='audio_recordings/turnEED_around.mp3', module="edAnalysis")
    ):
    status, textgridData, s = get_annotated_signal(p)
    if textgridData is None:
        return status, []
    
    # print(textgridData)
    
    # TODO: this is for the verification and it is not finished
    nWords, indxWords = number_and_indices(textgridData, 'w')
    nPho, indxPho = number_and_indices(textgridData, 'p')
    nSil, indxSil = number_and_indices(textgridData, 's')
    nOutOfVoc, indxOutOfVoc = number_and_indices(textgridData, 'o')

    binResult = 0
    # check pronunciation
    for i in range(nPho):
        if '*cor' in textgridData[2][indxPho[i]]:
            status = 0
            binResult = 2
        elif '*err' in textgridData[2][indxPho[i]]:
            status = 0
            binResult = 1
        
    return "success", [binResult]

def timing_test(module='wordStress', n=100, p=None):
    times=[]
    for i in range(n):
        start=time()
        # wordStress()
        if p is not None:
            globals()[module](p)
        else:
            globals()[module]()
        times.append(time()-start)
    print(np.mean(times))
    return np.mean(times)
    
if __name__ == "__main__":
    # execute only if run as a script
    # Test performance of wordStress module

    start=time()
    wordStress()
    print(time()-start)

    modules=['edAnalysis','phonemeContrast','iContrast','wordStress','sentenceStress']
    timings=[timing_test(el) for el in modules]
    print(np.round(timings,2))

    get_annotated_signal_timing=timing_test('get_annotated_signal')

    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress')


    get_annotated_signal_timing2=timing_test('get_annotated_signal', p=p)
    iContrast_timing=timing_test('iContrast')
    wordStress_timing=timing_test('wordStress')

    start=time();get_annotated_signal(p);print(time()-start)
    start=time();s,fs = load_audio(p['waveFileAddress'], fs=p['fs_target']);print(time()-start) #0.08
    start=time();s,fs=librosa.load(p['waveFileAddress'], sr=p['fs_target']);print(time()-start) #0.08

    from scipy.io.wavfile import read, write
    import soundfile as sf

    # start=time();fs,s = read(p['waveFileAddress']);print(time()-start)
    start=time();s,fs = sf.read(p['waveFileAddress']);print(time()-start) # 0.006
    start=time();s = s.T;print(time()-start)
    start=time();s = librosa.resample(s, fs, p['fs_target']);print(time()-start)  # 0.05

    start=time();get_textgrid_data(s,fs,p);print(time()-start)

    # inside get_textgrid_data
    rand_fileName = p['rand_fileName']
    start=time();write('./inputs/'+ rand_fileName+ '.wav', fs, (s*32767).astype(np.int16));print(time()-start)
    start=time();process_grammar(p['inputGrammar'], rand_fileName);print(time()-start)
    start=time();textgridData, out2=htk_recognition(p['modelName'], rand_fileName, p['inputPhoneticTranscription']);print(time()-start)
    start=time();textgridData=textgridData[(textgridData.iloc[:,2]!='sil')&(textgridData.iloc[:,2]!='sp')];print(time()-start)

    # Inside htk_recognition
    cmd2 = 'HVite -A -T 1 -a -C ./model/' +p['modelName']+ '/Align.cfg -H ./model/' +p['modelName']+ '/hmm-mono -H \
    ./model/generalSpeech/hmm-gs_1 -H ./model/generalSpeech/hmm-gss_2 -H ./model/generalSpeech/hmm-gss_3 -H ./model/generalSpeech/hmm-gss_4 \
    -H ./model/generalSpeech/hmm-gss_5 -w ./inputs/'+ p['rand_fileName']+ '.net -l ./results -o N ' +p['inputPhoneticTranscription']+ ' ./model/' +p['modelName']+ '/monophones \
    ./inputs/' +p['rand_fileName']+ '.wav'
    start=time();out2 = os.popen(cmd2).read();print(time()-start)

    # out2=os.system(cmd2)
    # print('out2', out2)

    # read rec file (i.e., the alignment outcome)
    textgridData = pd.read_csv('./results/'+ rand_fileName +'.rec', sep=' ', header=None)