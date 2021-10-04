from scipy.io.wavfile import write, read
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from audio_processing import load_audio, getIntonation, getIntensity, normalize, getf0Samples
from htk_utils import get_textgrid_data, clean_htk_files

from label_data_processing import make_all_phones_annotation_files, make_all_phones_annotation_files_from_phonetics, make_pContrast_annotation_files_from_phonetics, make_pContrast_annotation_files
from text_processing import phonetics_from_sentence
import uuid
import time
from glob import glob
inputs=glob('inputs/*')
for f in inputs: os.remove(f)

cached_filenames={}

def prepare_audio_file(audio_file, fs=16000):
    rID=str(uuid.uuid4())
    if os.path.exists(audio_file):
        s,fs = load_audio(audio_file, fs=fs)
    else:
        return "error: "+audio_file+" could not be loaded", None
    write('./inputs/'+ rID+ '.wav', fs, (s*32767).astype(np.int16))

    return "success", rID

def get_annotated_signal(rand_fileName, wav_name):
    """Load audio file and annotation files corresponding to parameters, 
    and calls "textgridData" to obtain htk predictions of phonemes and
    their timings

    Args:
        p (dictionary, optional): global parameters (wav, dct, grammar file paths, ). Defaults to set_params().
    Returns:
        status, textgridData, s (int, DataFrame, np array): textgridData contains phonetic predictions 
        from htk model with their timings and log probability
    """
    # if wav_name is None:
    #     wav_name=rand_fileName
    # rand_fileName, inputPhoneticTranscription, inputGrammar = p['rand_fileName'], p['inputPhoneticTranscription'], p['inputGrammar']
    try:
        fs,s=read('./inputs/'+ wav_name+ '.wav')
    except FileNotFoundError:
        return "error: audio file not found", None, None, None
    s=s/32767
    try:
        # import pdb;pdb.set_trace()
        textgridData, out=get_textgrid_data(rand_fileName, wav_name)
        if out.stderr.decode('utf-8')!='':
            return "error: could not get textgridData, htk error is:"+out.stderr.decode('utf-8'), None, None, None
        if len(textgridData)==0:
            return "success: part or all the phrase was not recognized in expected phonemes", None, None, None
    except Exception as e:
        print("get_annotated_signal Exception:",e)
        clean_htk_files(wav_name)
        return "error: could not get textgridData, check htk error", None, None, None
    
    # if textgridData.iloc[:,3].mean()<8.5:
    # if textgridData.iloc[:,3].mean()<5:
    #     return "success: low posterior probability, the pronunciation seems too far from expected phonetics", None, None, None
    
    # each row is True if out of vocabulary, False if it is a detected phoneme or word
    is_out_of_vocabulary=textgridData.iloc[:,2].str[:1].str.contains('o')

    if is_out_of_vocabulary.product():
        return "success: all of the elements were out of vocabulary", None, None, None
    # if is_out_of_vocabulary.sum():
    #     return "error: at least one element was out of vocabulary", None, None

    return "success", textgridData, s, fs

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
            dict_phones[info]=phone
        if r[0][0]=='o':
            dict_phones[r[0].split(' ')[0]]=r[0].split(' ')[0]

    phone_seq=[dict_phones[r[2]] for i,r in textgridData.iterrows()]
    phone_set=set(phone_seq)
    count_total = len(phone_seq)
    count_unique = len(phone_set)
    return nEntries==0 or count_total!=nEntries or count_unique!=nEntries



def chunking(
    rID
    ):
    """The goal of this module is to find pauses in a longer sequence such as a read paragraph.
    TODO: get_textgrid_data function filter out silences, so it won't work. Start from htk_recognition function in htk_utils

    Args:
        p ([type], optional): [description]. Defaults to set_params().

    Returns:
        [type]: [description]
    """
    status, textgridData, s, fs = get_annotated_signal(rID)
    if textgridData is None:
        return status, []

    minSilDur = 0.090 # we ask for at least 90ms of silence
    silence_durations=textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,1]-textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,0]
    idx_to_filter=silence_durations[silence_durations>minSilDur].index


def vowels(textgridData):
    """extract vowels among phonemes in textgridData
    Args:
        textgridData ([type]): [description]
    Returns:
        indxVowels, nVowelsPerWord: index of vowels and number of vowels
    """
    # for each entry, the second column is something like "w1_v1_1" or "w1_c1_1". the v is for vowel.
    # I tranform that to a table
    phone_df=pd.DataFrame([r[2].split('_') for i,r in textgridData.iterrows()])
    vowels_df=phone_df[phone_df.apply(lambda r:r[1][0]=='v', axis=1)]
    indxVowels=vowels_df.index.tolist()

    #extract the number of words by taking the index of the last one
    # nWords=int(textgridData[2].iloc[-1].split('_')[0][1:])+1
    nWords=len(phone_df.iloc[:,0].unique())
    nVowelsPerWord=[]
    for w in phone_df.iloc[:,0].unique():
        nVowelsPerWord.append(len(vowels_df[vowels_df.iloc[:,0]==w]))
    return indxVowels, nVowelsPerWord

def check_words_duration():
    pass
    # number of phoneme per entry in dct file this is set to 1 when vowel ELSE corresponds to the integer at the end of "w1_v1_1"
    # phoPerEntry=[int(r[2].split('_')[-1]) for i,r in textgridData.iterrows()]
    
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

def compute_stress_score(textgridData, s, fs, indxVowels):
    """Use textgridData to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
    a value by vowel representing a stress intensity

    Args:
        textgridData ([type]): [description]
        s (np array): audio signal
        fs (int): frequency of sampling
        indxVowels ([type]): indices of vowels in textgridData
        nVowelsPerWord ([type]): number of vowels per word

    Returns:
        weighted_score [type]: stress intensity score
    """
    f0Samples=getIntonation(s, fs)
    intensity=getIntensity(s, fs)

    # extract features
    # each word start and end position expressed in samples
    startPositions_samples = (round(fs*textgridData.iloc[:,0])+1).astype(int).tolist()
    stopPositions_samples = round(fs*textgridData.iloc[:,1]).astype(int).tolist()
    
    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples[-1]<len(s), "The end of the last phoneme should be inside the signal"

    Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
    nVowels=len(indxVowels)
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
        
        phone_df=pd.DataFrame([r[2].split('_') for i,r in textgridData.iterrows()])
        # here we use the prediction of HMM model as an indication, as it has to classify 0, 1 or 2
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
    # weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur)/3.2  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data
    return weighted_score

def vowel_stresses(
        rand_fileName, wav_name
        ):
    """vowels_stresses() computes prosody features (intesity, pitch, ...) to compute 
    a value by vowel, located thanks to textgridData, representing a stress intensity.
    It also plots a curve representing the stress evolution.

    Args:
        p ([type], optional): [description]. Defaults to set_params().

    Returns:
        string, list of float list: status, stress intensities by word
    """
    
    status, textgridData, s, fs = get_annotated_signal(rand_fileName, wav_name)
    if textgridData is None:
        return status, []

    # TODO: use these for verification 
    # if verification_n_of_phoneme(textgridData, p):
    #     return "error: inconsistent number of detected phonemes", []

    indxVowels, nVowelsPerWord=vowels(textgridData)
    weighted_score=compute_stress_score(textgridData, s,  fs, indxVowels)
    # print(weighted_score)
    # fig=plt.figure()
    # plt.plot(weighted_score)
    # plt.savefig('word_curve.png')
    # chose prominent vowel per word

    assert sum(nVowelsPerWord)==len(weighted_score)

    weighted_score_by_word=[]
    syl_start=0
    for w_i,n_v in enumerate(nVowelsPerWord):
        weighted_score_by_word.append(weighted_score[syl_start:syl_start+n_v].tolist())
        syl_start+=n_v
    return "success", weighted_score_by_word

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

def prosody_by_phone(rand_fileName, wav_name):
    
    # if wav_name is None:
    #     wav_name=rand_fileName
    status, textgridData, s, fs = get_annotated_signal(rand_fileName, wav_name)

    f0Samples=getIntonation(s, fs)
    vuv=np.nan_to_num(getf0Samples(s, fs), nan=0).astype(bool)

    intensity=getIntensity(s, fs)
    # detected_phonemes=textgridData[textgridData.iloc[:,2].str[0]=='p']
    detected_phonemes=textgridData[textgridData.iloc[:,2].str[0]=='w']
    Imax,Imean,Fmax,Fmean,Dur,voicing=[],[],[],[],[],[]
    for i,r in detected_phonemes.iterrows():
        start=round(fs*r[0])
        end=round(fs*r[1])
        F_phone=f0Samples[start:end]
        I_phone=intensity[start:end]
        vuv_phone=vuv[start:end]
        Fmax.append(max(F_phone))
        Imax.append(max(I_phone))
        Fmean.append(np.mean(F_phone))
        Imean.append(np.mean(I_phone))
        Dur.append(r[1]-r[0])
        voicing.append(sum(vuv_phone)/len(vuv_phone))
    d={}
    d['Imax']=Imax
    d['Imean']=Imean
    d['Fmax']=Fmax
    d['Fmean']=Fmean
    d['Dur']=Dur
    d['voicing']=voicing

    return "success", [textgridData, d]


def phonemeContrast(
    rand_fileName, wav_name
    ):
    """This functions uses textgridData that now has information of all detected phonetic transcriptions.
    It returns textgridData and the phonetics of the studied word.
    This is to be used with automatically generated dct and grammar files, because it uses the conventions chosen for their generation.

    Returns:
        string, [DataFrame, list]: status, [textgridData, detected_transcription as a list of phonemes]
    """
    
    # if wav_name is None:
    #     wav_name=rand_fileName
    status, textgridData, s, fs = get_annotated_signal(rand_fileName, wav_name)
    if textgridData is None:
        return status, []
    
    detected_transcription=textgridData[textgridData.iloc[:,2].str[0]=='p']['detected_transcription'].tolist()
    return "success", [textgridData, detected_transcription]

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

def phonemeContrast_from_phonetics_audio(
                    rID,
                    phonetics=[['T', 'ER1', 'N', 'D'], ['ER0', 'AW1', 'N', 'D']], 
                    word_idx=0, 
                    target_phones='D', 
                    alternatives=['T', 'D', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']):
    """This uses phonetics, and target phones as well as alternatives to build annotation files for htk.
    Then it calls phonemeContrast module with this information. It also prints where are the differences in the phonetic entries
    between ground truth and predictions (might be returned in the future)

    Args:
        phonetics (list, optional): [description]. Defaults to [['T', 'ER1', 'N', 'D'], ['ER0', 'AW1', 'N', 'D']].
        p ([type], optional): [description]. Defaults to set_params().
        word_idx (int, optional): [description]. Defaults to 0.
        target_phones (str, optional): [description]. Defaults to 'D'.
        alternatives (list, optional): [description]. Defaults to ['T', 'D', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'].

    Returns:
        [type]: [description]
    """

    p_idx='pC - '+' '.join(['_'.join(w) for w in phonetics])+str(word_idx)+target_phones+'_'.join(alternatives)

    if p_idx not in cached_filenames:
        time_str=time.asctime( time.localtime(time.time()) )
        uid=str(uuid.uuid4())+'_'+time_str.replace(' ','_')
        cached_filenames[p_idx]=uid

        make_pContrast_annotation_files_from_phonetics(uid,phonetics=phonetics, word_idx=word_idx, target_phones=target_phones, alternatives=alternatives)

        # with open(path_cached_filenames, 'w') as fp:
        #     json.dump(cached_filenames, fp)

    status,result=phonemeContrast(cached_filenames[p_idx], rID)
    return status, result



def vowel_stresses_from_phonetics_audio(
    rID,
    phonetics=[['AY1'], ['W', 'UH1', 'D'], ['L', 'AH1', 'V'], ['T', 'UW1'], ['G', 'OW1'], ['T', 'UW1'], ['AY1', 'ER0', 'L', 'AH0', 'N', 'D']]
    ):
    p_idx='all_phones - '+' '.join(['_'.join(w) for w in phonetics])

    if p_idx not in cached_filenames:
        time_str=time.asctime( time.localtime(time.time()) )
        uid=str(uuid.uuid4())+'_'+time_str.replace(' ','_')
        cached_filenames[p_idx]=uid

        make_all_phones_annotation_files_from_phonetics(uid,phonetics)
        # with open(path_cached_filenames, 'w') as fp:
        #     json.dump(cached_filenames, fp)

    status,result=vowel_stresses(cached_filenames[p_idx], rID)
    return status, result

def sentenceStress_from_phonetics_audio(
    rID,
    phonetics=[['AY1'], ['W', 'UH1', 'D'], ['L', 'AH1', 'V'], ['T', 'UW1'], ['G', 'OW1'], ['T', 'UW1'], ['AY1', 'ER0', 'L', 'AH0', 'N', 'D']]
    # p=set_params()
    ):
    status, weighted_score_by_word=vowel_stresses_from_phonetics_audio(rID,phonetics)
    
    max_scores_by_word=[max(el) for el in weighted_score_by_word]
    binResult=np.zeros(len(max_scores_by_word)).astype(int)
    binResult[np.argmax(max_scores_by_word)]=1

    return {"status": "success", "stress_intensities": [int(el*100) for el in max_scores_by_word], "stress_binaries": binResult.tolist()}



def max_by_line(a):
    a_max=[]
    for el in a:
        a_max.append((el == np.max(el)).astype(int))
    return a_max

def merge_list(l):
    merged=[]
    for el in l: merged+=el
    return merged

def wordStress_from_phonetics_audio(
    rID,
    phonetics=[['AY1'], ['W', 'UH1', 'D'], ['L', 'AH1', 'V'], ['T', 'UW1'], ['G', 'OW1'], ['T', 'UW1'], ['AY1', 'ER0', 'L', 'AH0', 'N', 'D']]
    ):
    status, weighted_score_by_word=vowel_stresses_from_phonetics_audio(rID, phonetics)
    
    if weighted_score_by_word == []:
        return {"status": status, "stress_intensities": [], "stress_binaries": []}


    bin_score_by_word=max_by_line(weighted_score_by_word)
    # binResult=np.concatenate(bin_score_by_word)

    weighted_score_by_word=[[int(x*100) for x  in sublist] for sublist in weighted_score_by_word]
    return {"status": "success", "stress_intensities": weighted_score_by_word, "stress_binaries": [el.tolist() for el in bin_score_by_word]}

def stress_from_formatted_phonetics(rID,phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D", text="I would love to go to Ireland!", level="word", chunking_chars=[',',';','.','!','?', ':']): #'[\,\?\.\!\;\:\"\*]'
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    lens=[len(el) for el in split_phonetics]
    merged_phonetics=merge_list(split_phonetics)

    status, weighted_score_by_syllable=vowel_stresses_from_phonetics_audio(rID,merged_phonetics)
    
    if weighted_score_by_syllable == []:
        return {"status": status, "stress_intensities": [], "stress_binaries": []}
    weighted_score_by_word_by_syllable_by_vowel=[]
    i=0
    for l in lens:
        weighted_score_by_word_by_syllable_by_vowel.append(weighted_score_by_syllable[i:i+l])
        i+=l

    if level=="word":
        weighted_score_by_word_by_syllable_int=[[max([int(vowel*100) for vowel  in syl])  for syl  in word] for word in weighted_score_by_word_by_syllable_by_vowel]
        bin_score_by_word_by_syllable=max_by_line(weighted_score_by_word_by_syllable_int)
        bin_score_by_word_by_syllable=[el.tolist() for el in bin_score_by_word_by_syllable]
        return {"status": "success", "stress_intensities": weighted_score_by_word_by_syllable_int, "stress_binaries": bin_score_by_word_by_syllable}
    elif level=="sentence":
        def remove_downwards_trend(y):
            if len(y)>1:
                # Remove downwards trend
                x=range(len(y))
                model = np.polyfit(x, y, 1)
                a=model[0]
                b=model[1]
                y=y-(a*x+b)

                # normalize between 0 and 100
                y=y-min(y)
                y=y/max(y)*100
            else:
                y=np.array(y)
            return y.astype(int).tolist()
        
        def chunk_text(text):
            for c in chunking_chars:
                text=text.replace(c, chunking_chars[0])
            chunks=text.split(chunking_chars[0])
            chunks = list(filter(None, chunks)) # remove empty string
            # split each chunk in words, remove empty strings, get length (to know the n of words in each chunk)
            n_words_by_chunk=[len(list(filter(None, el.split(' ')))) for el in chunks]
            assert sum(n_words_by_chunk)==len(text.split(' ')), "Checking number of words is the same after chunking"
            return n_words_by_chunk
        
        n_words_by_chunk=chunk_text(text)
        max_score_by_word=[int(max(l)[0]*100) for l in weighted_score_by_word_by_syllable_by_vowel]

        scores_grouped_by_chunk=[]
        i=0
        for l in n_words_by_chunk:
            scores_grouped_by_chunk.append(max_score_by_word[i:i+l])
            i+=l
        
        # Remove downwards trends: it seems to have a positive impact on the performance. But it would be good to test
        # with more examples
        scores_grouped_by_chunk=[remove_downwards_trend(el) for el in scores_grouped_by_chunk]
        def intensity_to_bin(score_by_word):
            bin_score_by_word=np.zeros(len(score_by_word)).astype(int).tolist()
            imax=np.argmax(score_by_word)
            bin_score_by_word[imax]=1
            return bin_score_by_word

        bins_by_chunk=[]
        for chunk in scores_grouped_by_chunk:
            bin=intensity_to_bin(chunk)
            bins_by_chunk.append(bin)
        
        bin_score_by_word=[]
        for el in bins_by_chunk: bin_score_by_word+=el
        # import pdb;pdb.set_trace()
        return {"status": "success", "stress_intensities": merge_list(scores_grouped_by_chunk), "stress_binaries": bin_score_by_word}
    else:
        print("No such level in stress_from_formatted_phonetics. It has to be either 'word' or 'sentence'.")

def phonemeContrast_from_formatted_phonetics_audio(
                    rID,
                    phonetics='T_ER1_N_D ER0|AW1_N_D', 
                    # p=set_params(), 
                    word_idx=0, 
                    target_phones='D', 
                    # alternatives=['T', 'D', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']
                    alternatives='T D IH0_D IH1_D IH2_D EH2_D AH0_D'
                    ):
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    split_alternatives=[alt.replace('_',' ') for alt in alternatives.split(' ')]
    target_phones=target_phones.replace('_', ' ')
    merged_phonetics=merge_list(split_phonetics)

    print(target_phones)
    print(alternatives)
    status, result=phonemeContrast_from_phonetics_audio(rID,
                    phonetics=merged_phonetics, 
                    # p=p, 
                    word_idx=word_idx, 
                    target_phones=target_phones, 
                    alternatives=split_alternatives)
    if result!=[]:
        result[0].detected_transcription=result[0].detected_transcription.str.replace(' ','_')
    
    print('phonemeContrast_from_formatted_phonetics_audio result:', result)
    return status, result

# obsolete functions backup
if False:
    
    def wordStress(
        rand_fileName, wav_name
        # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav')
        ):
        """calls vowels_stresses() that compute prosody features (intesity, pitch, ...) to compute 
        a value by vowel representing a stress intensity, and take the max by word and build a binary vector with ones on maximums

        Args:
            p (dict, optional): global parameters. Defaults to set_params().

        Returns:
            string, list of binaries: status, stress results by vowel (0=no stress,  1=stress)
        """
        
        # if wav_name is None:
        #     wav_name=rand_fileName
        status, weighted_score_by_word=vowel_stresses(rand_fileName, wav_name)

        if weighted_score_by_word == []:
            return {"status": status, "stress_intensities": [], "stress_binaries": []}


        def max_by_line(a):
            a_max=[]
            for el in a:
                a_max.append((el == np.max(el)).astype(int))
            return a_max

        bin_score_by_word=max_by_line(weighted_score_by_word)
        # binResult=np.concatenate(bin_score_by_word)

        weighted_score_by_word=[[int(x*100) for x  in sublist] for sublist in weighted_score_by_word]

        return {"status": "success", "stress_intensities": weighted_score_by_word, "stress_binaries": [el.tolist() for el in bin_score_by_word]}

    def sentenceStress(
        rand_fileName, wav_name
        ):
        """calls vowels_stresses() that compute prosody features (intesity, pitch, ...) to compute 
        a value by vowel representing a stress intensity, and 
        -take the max by word 
        -take the max of these max to have the most stressed word
        and build a binary vector with a one on this word index

        Args:
            p ([type], optional): [description]. Defaults to set_params().

        Returns:
            string, list of binaries: status, stress results by word (0=no stress,  1=stress)
        """
        
        status, weighted_score_by_word=vowel_stresses(rand_fileName, wav_name)

        
        if weighted_score_by_word == []:
            return {"status": status, "stress_intensities": [], "stress_binaries": []}



        
        max_scores_by_word=[max(el) for el in weighted_score_by_word]
        stress_intensities=[int(el*100) for el in max_scores_by_word]

        # Try to remove downwards trend
        x=range(len(stress_intensities))
        model = np.polyfit(x, y, 1)
        a=model[0]
        b=model[1]
        stress_intensities=stress_intensities-(a*x+b)

        # max_scores_by_word=[np.median(el) for el in weighted_score_by_word]
        binResult=np.zeros(len(stress_intensities)).astype(int)
        binResult[np.argmax(stress_intensities)]=1
        print("------------")
        print(stress_intensities)

        return {"status": "success", "stress_intensities": stress_intensities, "stress_binaries": binResult.tolist()}

        # return "success", [max_scores_by_word, binResult]


    # these are not valid anymore, I removed audio information from the rID
    def vowel_stresses_from_text_audio(text='I would love to go to Ireland!', audio_path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav'):
        """This uses text to get phonetics, to build annotation files for htk.
        Then it calls vowel_stresses module with this information.
        Returns:
            string, list of float list: status, stress intensities by word
        """
        p=set_params()
        make_all_phones_annotation_files(p,text)
        status,result=vowel_stresses(p['rand_fileName'])
        return status, result
    def phonemeContrast_from_text_audio(
                        text='turned around', 
                        word_idx=0, 
                        target_phones='D', 
                        alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']):
        """This uses text to get phonetics, and target phones as well as alternatives to build annotation files for htk.
        Then it calls phonemeContrast module with this information. It also prints where are the differences in the phonetic entries
        between ground truth and predictions (might be returned in the future)

        Returns:
            string, [DataFrame, list]: status, [textgridData, detected_transcription as a list of phonemes]
        """
        p=set_params()
        make_pContrast_annotation_files(p,text=text, word_idx=word_idx, target_phones=target_phones, alternatives=alternatives)
        status,result=phonemeContrast(p['rand_fileName'])
        phonetic_GT=phonetics_from_sentence(text)[int(word_idx)]
        print('difference between ground truth and prediction:', [int(el[0]!=el[1]) for el in zip(result[-1], phonetic_GT)])
        return status, result


    # generalized and replaced by phoneme contrast
    def edAnalysis(
        p=set_params(sentenceID=1, module="edAnalysis")
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


    # generalized and replaced by phoneme contrast
    def iContrast(
        p=set_params(module="iContrast")
        ):
        """Use textgridData to have the timings of vowels 
        (and check if the vowel detected is a short or long vowel. -> I removed that part with no noticeable change in performance)

        Args:
            p ([type], optional): [description]. Defaults to set_params(module="iContrast").

        Returns:
            int: 0 if short, 1 if long
        """
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
                # if textgridData.iloc[i, 1]-textgridData.iloc[i, 0]<0.07:
                #     binResult=0
        return "success", binResult

    
    def sentenceStress_old(
        p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress')
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
        
        # this
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
        # weighted_score = (0.6*zImax + 0.4*zFmax)/1.0 # needs fine-tuning once enough user data are available - in the long term consider additional features and train a classifier with annotated user data
        # weighted_score = zImax*zFmax # needs fine-tuning once enough user data are available - in the long term consider additional features and train a classifier with annotated user data
        weighted_score=zImax

        weighted_score=normalize(weighted_score)

        rateThreshold = 1.01
        nWords=len(textgridData)
        binResult = np.zeros(nWords)
        
        fig=plt.figure()
        plt.plot(weighted_score)
        # plt.plot(zFmax)
        plt.savefig('sentence_curve.png')

        # The original method from georgious does something with the evolution of the weighted_score
        # and then does a threshold. If the threshold is very high (0.98), with my normalization, 
        # it is almost the same (exactly the same for the examples I have) as just taking the max.

        # The second is thus a lot more simple: put one at the max of weighted_score
        if False:
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
                meanScore=0.98

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
        else:
            binResult[np.argmax(weighted_score)]=1

        return "success", binResult



if __name__ == "__main__":
    # execute only if run as a script

    # # examples:
    # status, result=phonemeContrast_from_text_audio()
    # status, result=phonemeContrast_from_text_audio(text='Did he fall asleep?',
    #                 audio_path='../audio-with-analysis-ids/audio/5deb3ea1-1c0f-4a2b-a2ca-36e0e856c11d.wav', 
    #                 word_idx=0, 
    #                 target_phones='IH1', 
    #                 alternatives=['IH0', 'IH2', 'IY0', 'IY1', 'IY2'])

    phonetics=phonetics_from_sentence('Did he fall asleep?')

    # Test performance of wordStress module
    start=time()
    # wordStress()
    print(time()-start)

    modules=['edAnalysis','phonemeContrast','iContrast','wordStress','sentenceStress']
    timings=[timing_test(el) for el in modules]
    print(np.round(timings,2))

    get_annotated_signal_timing=timing_test('get_annotated_signal')

    # p=set_params(sentenceID=111, module="iContrast")
    # p=set_params(sentenceID=1, module='sentenceStress')

    # get_annotated_signal_timing2=timing_test('get_annotated_signal', p=p)
    iContrast_timing=timing_test('iContrast')
    wordStress_timing=timing_test('wordStress')

    # start=time();get_annotated_signal(p);print(time()-start)
    # start=time();s,fs = load_audio(p['waveFileAddress'], fs=p['fs_target']);print(time()-start) #0.08
    # start=time();s,fs=librosa.load(p['waveFileAddress'], sr=p['fs_target']);print(time()-start) #0.08

    from scipy.io.wavfile import write

    # start=time();fs,s = read(p['waveFileAddress']);print(time()-start)
    # start=time();s,fs = sf.read(p['waveFileAddress']);print(time()-start) # 0.006
    # start=time();s = s.T;print(time()-start)
    # start=time();s = librosa.resample(s, fs, p['fs_target']);print(time()-start)  # 0.05

    # start=time();get_textgrid_data(s,fs,p);print(time()-start)

    # # inside get_textgrid_data
    # rand_fileName = p['rand_fileName']
    # start=time();write('./inputs/'+ rand_fileName+ '.wav', fs, (s*32767).astype(np.int16));print(time()-start)
    # start=time();process_grammar(p['inputGrammar'], rand_fileName);print(time()-start)
    # start=time();textgridData, out2=htk_recognition(p['modelName'], rand_fileName, p['inputPhoneticTranscription']);print(time()-start)
    # start=time();textgridData=textgridData[(textgridData.iloc[:,2]!='sil')&(textgridData.iloc[:,2]!='sp')];print(time()-start)

    # Inside htk_recognition
    # cmd2 = 'HVite -A -T 1 -a -C ./model/' +p['modelName']+ '/Align.cfg -H ./model/' +p['modelName']+ '/hmm-mono -H \
    # ./model/generalSpeech/hmm-gs_1 -H ./model/generalSpeech/hmm-gss_2 -H ./model/generalSpeech/hmm-gss_3 -H ./model/generalSpeech/hmm-gss_4 \
    # -H ./model/generalSpeech/hmm-gss_5 -w ./inputs/'+ p['rand_fileName']+ '.net -l ./results -o N ' +p['inputPhoneticTranscription']+ ' ./model/' +p['modelName']+ '/monophones \
    # ./inputs/' +p['rand_fileName']+ '.wav'
    # start=time();out2 = os.popen(cmd2).read();print(time()-start)

    # read rec file (i.e., the alignment outcome)
    # textgridData = pd.read_csv('./results/'+ rand_fileName +'.rec', sep=' ', header=None)

    alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']
    # alternatives=['T', 'D', 'IH0 D', 'EH1 D', 'AH0 D']
    # p=set_params(sentenceID=111, waveFileAddress='audio_recordings/ed_acceptEED.wav', module="edAnalysis")
    # make_pContrast_annotation_files(p, text="accepted",word_idx=0, target_phones='IH0 D',
    # alternatives=alternatives)
    # status, results= phonemeContrast(p)

    # make_pContrast_annotation_files(p, text="law",word_idx=0, target_phones='AO1', alternatives=['AO1','OW1','AA1', 'AH1 W'])
