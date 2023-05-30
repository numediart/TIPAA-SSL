import os, psutil;print_memory_usage=lambda stage: print(stage + ": "+ str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

print_memory_usage("RAM - start of DL_speech_tech")
from scipy.io.wavfile import  read
import numpy as np
import pandas as pd
import soundfile as sf
from syllabipy.sonoripy import SonoriPy
import base64
from linetimer import CodeTimer

print_memory_usage("RAM - DL_speech_tech after external dependencies")
from src.audio_processing import getIntonation, getIntensity, normalize, read_audio_string, read_audio_bytes

print_memory_usage("RAM - DL_speech_tech after src.audio_processing")

from src.text_processing import unstress, split_phonetics, remove_stress_annots, drop_consecutive_duplicates, drop_consecutive_duplicate_elements, phonetics_indexed_df_from_formatted_phonetics
print_memory_usage("RAM - DL_speech_tech after src.text_processing")
from src.pronunciation_dictionaries import cmu_vowels, cmu_stressed_vowels, cmu_consonants, cmu_to_gibberish, cmu_diphtongs
from src.pronunciation_dictionaries import ipa_vowels, ipa_consonants, ipa_to_gibberish
print_memory_usage("RAM - DL_speech_tech after src.pronunciation_dictionaries")



# initialize model
# from src.charsiu_utils import charsiu_phone_forced_aligner
# default_model_charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
# print_memory_usage("RAM - DL_speech_tech after default_model_charsiu")


phoneme_GT_proba_threshold_dict={}
default_thresh=0.2
for k in cmu_stressed_vowels: phoneme_GT_proba_threshold_dict[k]=default_thresh
for k in cmu_consonants: phoneme_GT_proba_threshold_dict[k]=default_thresh

# phoneme_GT_proba_threshold_dict['AO0']=0.1
# phoneme_GT_proba_threshold_dict['AO1']=0.1
# phoneme_GT_proba_threshold_dict['AO2']=0.1

# model=pickle.load(open('model_mailabs_umap_2_gmm_300.pkl','rb'))

from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction, extract_word
print_memory_usage("RAM - DL_speech_tech after wav2vec2_frame_prediction")

# default_model = Wav2Vec2ForFramePrediction('cmu')

default_model = Wav2Vec2ForFramePrediction('cmu',w2v2_model_format="onnx")
# default_model.load(name='model_mailabs_pca_0.95_knn_10_w')
# default_model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w')
default_model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w_no_CH_JH')


# default_model.load(name='model_mailabs_pca_99_logistic_regression')
# default_model.load(name='model_mailabs_pca_99_knn_5_cos_w')

# default_model=default_model_charsiu

print_memory_usage("RAM - DL_speech_tech after default_model")


target_accepted_alternatives={
    'AA': ['AA', 'AO'],
    'AO': ['AA', 'AO'],
    # 'OW': ['AA', 'OW'],
    'D': ['D', 'T'],
    'Z': ['Z', 'S'],
    'S': ['Z', 'S'],
    # 'T': ['D', 'T'],
    # 'IH': ['IH', 'AH', 'EH']
}

terminations_accepted_alternatives={
    'IH_Z':['AH_Z','IH_Z', 'ER_Z', 'IY_Z'],
    'IH_D':['AH_D','IH_D', 'ER_D', 'IY_D'],
    }


# For final -ed and final -s, we use a termination contrast with a basis that can accept enough phonemes, I take the longest target "IH_D" or "IH_Z"
# because of this, if the target is D/T or S/Z, it is frequent to have a border effect and that the real phoneme before or after is included in the result
# I want to accept these as correct

# for "D", I want to accept anything finishing with "D" except those corresponding to "IH_D"
# but if I accept all vowels, it means I wouldn't give feedback for a mistake like "S_N_OW_EH_D" for "snowed"
# therefore, I accept only consonants and diphtongs. 
# It's also more likely to have this border effect with consonant because the target is a consonant (verified experimentally looking at confusions)

# However, for the phoneme after, I want to accept both vowels and consonants

all_Z=[p+"_Z" for p in list(cmu_consonants)]+["Z_"+p for p in list(cmu_consonants)+list(cmu_vowels)]
all_S=[p+"_S" for p in list(cmu_consonants)]+["S_"+p for p in list(cmu_consonants)+list(cmu_vowels)]
terminations_accepted_alternatives["Z"]=[el for el in all_Z if el not in terminations_accepted_alternatives['IH_Z']]
terminations_accepted_alternatives["S"]=[el for el in all_S if el not in terminations_accepted_alternatives['IH_Z']]


all_D=[p+"_D" for p in list(cmu_consonants)+list(cmu_diphtongs)]+["D_"+p for p in list(cmu_consonants)+list(cmu_vowels)]
all_T=[p+"_T" for p in list(cmu_consonants)]+["T_"+p for p in list(cmu_consonants)+list(cmu_vowels)]
terminations_accepted_alternatives["D"]=[el for el in all_D if el not in terminations_accepted_alternatives['IH_D']]
terminations_accepted_alternatives["T"]=[el for el in all_T if el not in terminations_accepted_alternatives['IH_D']]

# As in target_accepted_alternatives, we want to accept ['D','T'] for 'D', we should do the same when there is a superfluous phoneme, therefore:
terminations_accepted_alternatives["D"]+=terminations_accepted_alternatives["T"]

def audio_load_and_check(audio, phonetics, max_speech_rate=8, mode='file', fs=16000):
    """Load audio with modes: from a "file", from "base64" encoding, from "bytes", or directly a "numpy" array
    Then check duration to see if it's plausible
    
    I first detect if the audio is too short to have a realistic speech rate
        https://www.science.org/doi/10.1126/sciadv.aaw2594
    https://www.reddit.com/r/languagelearning/comments/f5o1om/distribution_of_syllable_rate_sr_in_syllables_per/
    Speech rate is always between 5 and 8 syl/second
    """
    n_syllables_tot=sum([len(el.split('|')) for el in phonetics.split(' ')])

    # TODO: change this by the use of src.audio_processing.read_audio_file
    if mode=='file':
        try:
            f=sf.SoundFile('./inputs/'+ audio+ '.wav')
        except FileNotFoundError:
            return "error: audio file not found", None
        if f.frames==0:  return "success: audio is empty (has zero sample)", None
        duration=f.frames / f.samplerate
        speech_rate=n_syllables_tot/duration
        if speech_rate>max_speech_rate: 
            return "success: audio is too short compared to the expected number of syllables", None
        try:
            fs,s=read('./inputs/'+ audio+ '.wav')
            s=s/32767
        except FileNotFoundError:
            return "error: audio file not found", None
        
    elif (mode=='base64' or mode=='bytes' or mode=="numpy"):
        if mode=='base64': s, fs= read_audio_string(audio, fs=fs)
        elif mode=='bytes': s, fs= read_audio_bytes(audio, fs=fs)
        elif mode=="numpy": s=audio

        if len(s)==0:  return "success: audio is empty (has zero sample)", None
        duration=len(s) / fs
        speech_rate=n_syllables_tot/duration
        if speech_rate>max_speech_rate: 
            return "success: audio is too short compared to the expected number of syllables", None
    else:
        return "error: mode for audio_load_and_check() must be file, base64, bytes or numpy", None
    
    if np.abs(s).sum()==0: return "success: no voiced sound detected (only 0's in waveform)", None
    
    f0Samples=getIntonation(s, fs)
    if sum([el!=el for el in f0Samples])==len(f0Samples):
        return "success: no voiced sound detected (no pitch detected)", None
    return "success", s

def remove_downwards_trend(y):
    if len(y)>2:
        # Remove downwards trend
        x=range(len(y))
        linear_f = np.polyfit(x, y, 1)
        a=linear_f[0]
        b=linear_f[1]
        y=y-(a*x+b)

        # normalize between 0 and 100
        y=y-min(y)
        y=y/max(y)*100
    else:
        y=np.array(y)
    return y.astype(int).tolist()


import math
roundup=lambda n: math.ceil(n)

def intensity_to_bin(score_by_word, n_max=2):
    bin_score_by_word=np.zeros(len(score_by_word)).astype(int).tolist()
    
    if len(score_by_word)==1:
        imax=np.argmax(score_by_word)
        bin_score_by_word[imax]=1
    else:
        imaxes=[np.argmax(score_by_word)] if len(score_by_word)<n_max else np.argpartition(score_by_word, -n_max)[-n_max:]
        for imax in imaxes:
            if score_by_word[imax]>60:
                bin_score_by_word[imax]=1
    return bin_score_by_word

def predict_phone(forced_aligner, df_word, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx=0, phoneme_set=cmu_vowels, GT_proba_threshold=0.2):
    """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'
    """
    phoneme_set=[p for p in remove_stress_annots(phoneme_set)]+["[SIL]"]
    # split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
    # df_word=self.predict_word(audio, split_phonetics, target_word_idx)

    if len(df_word)>0:
        word=phonetics.split(' ')[target_word_idx]
        syllables=[syl.split('_') for syl in word.split('|')]
        syllable=syllables[target_syllable_idx]
        syl=remove_stress_annots(syllable)
        idxs_of_target_occurences=[i for i,p in enumerate(syl) if unstress(target_phones) ==p]

        if target_occurence_idx<len(idxs_of_target_occurences):
            p_idx_local=idxs_of_target_occurences[target_occurence_idx]
        else:
            # self.status="error: target_occurence_idx is out of bounds"
            phonetic_detection=float('nan')
            syl=float('nan')
            return phonetic_detection, syl

        len_previous_syllables=sum([len(el) for el in syllables[:target_syllable_idx]])
        p_idx_global=len_previous_syllables+p_idx_local

        phonetic_detection = df_word.iloc[p_idx_global].pred_phones_audio

        phoneme_set_ids=forced_aligner.labelize_phonemes(phoneme_set)
        proba_means=df_word.iloc[p_idx_global].proba_means

        # if GT_proba is beyond the threshold, we take it as prediction
        if df_word.iloc[p_idx_global].GT_proba>GT_proba_threshold:
            phonetic_detection=target_phones
        else:
            # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
            filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
            # if everything is 0 in the filtered proba, then we keep the phonetic detection that was in the non-filtered proba. E.g., for the word "new", imagine we target the "Y" of "N_Y_UW", but it is pronounced the british way "N_UW"
            # Then, there could be zero probability in consonants, and therefore, we keep the vowel that should be close to "UW"
            if sum(filtered_proba_means)!=0:
                phonetic_detection=forced_aligner.id_to_p[np.argmax(filtered_proba_means)]
        syl[p_idx_local]=phonetic_detection
        
    else:
        phonetic_detection=float('nan')
        syl=float('nan')
    return phonetic_detection, syl

def compute_stress_score(df_segmented, audio, fs=16000):
    """Use df_segmented to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
    a value by vowel representing a stress intensity

    Args:
        phonetics (str): formatted phonetics
        audio (np array): audio signal
    Returns:
        weighted_score [type]: stress intensity score
    """
    # select vowels
    filtered_df=df_segmented[df_segmented.phones.isin(cmu_vowels)]

    f0Samples=getIntonation(audio, fs)
    intensity=getIntensity(audio, fs)

    # extract features
    # each word start and end position expressed in samples
    startPositions_samples = (round(fs*filtered_df.loc[:,'start'])+1).astype(int).tolist()
    stopPositions_samples = round(fs*filtered_df.loc[:,'end']).astype(int).tolist()

    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples[-1]<len(audio), "The end of the last phoneme should be inside the signal"

    Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
    for i in range(len(filtered_df)):
        range_vowel=range(startPositions_samples[i], stopPositions_samples[i])
        Ivowel=intensity[range_vowel]
        Fvowel=f0Samples[range_vowel]
        
        Imax.append(max(Ivowel))
        Imean.append(np.mean(Ivowel))
        Fmax.append(max(Fvowel))
        Fmean.append(np.mean(Fvowel))

        Dur.append(filtered_df['end'].iloc[i]-filtered_df['start'].iloc[i])
        
    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zImean = normalize(Imean)
    zFmax = normalize(Fmax)
    zFmean = normalize(Fmean)
    zDur = normalize(Dur)

    # combine the features
    weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur)/3.2  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data

    return weighted_score

###################   Pronunciation aspect functions  ################

def audio_to_phone_prob_matrix(audio, phonetics, max_speech_rate=8, mode="numpy", model=default_model):
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    with CodeTimer('load audio'): audio_status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    if audio_status=="success": 
        with CodeTimer('phone_prob_matrix prediction'): phone_prob_matrix = model.predict_phone_prob_matrix(s, model.fs)
        return audio_status, s, phone_prob_matrix
    else:
        return audio_status, s, None

def phone_prob_matrix_segmentation(phone_prob_matrix, phonetics, model=default_model):
    phoneme_list=phonetics.replace(' ','_').replace('|','_').split('_')
    with CodeTimer('DTW'): 
        df_segmented=model.forced_aligner.probas_to_df_segmented(phone_prob_matrix, phoneme_list, fs=model.fs, time_per_output=model.time_per_output)
        model.pred_phones_audio = list(df_segmented.pred_phones_audio.values)
    return df_segmented

def stress_from_formatted_phonetics(audio,phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D", 
                                    n_words_by_chunk=[7],
                                    level="sentence", 
                                    max_speech_rate=8, mode='file',
                                    model=default_model,
                                    vowels=cmu_vowels
                                    ):
    audio_status, s, phone_prob_matrix = audio_to_phone_prob_matrix(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode, model=default_model)
    if audio_status!="success": 
        return {"status": audio_status, "stress_intensities": [], "stress_binaries": []}
    df_segmented=phone_prob_matrix_segmentation(phone_prob_matrix, phonetics, model=model)

    print("model status", model.status)
    if model.status!="success":  return {"status": model.status, "stress_intensities": [], "stress_binaries": []}

    with CodeTimer('stress extraction'): ws=compute_stress_score(df_segmented, s, fs=model.fs)
    
    # TODO: I think I should check for voiceness, but I don't know if I should do it for all vowels
    if sum([el!=el for el in ws])==len(ws):
        status = "success: no voiced sound detected inside supposed vowels (no pitch detected)"
        return {"status": status, "stress_intensities": [], "stress_binaries": []}

    phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics)
    is_vowel=phonetics_indexed_df.apply(lambda r: unstress(r.phones) in vowels, axis=1)
    vowels_indexed_df=phonetics_indexed_df[is_vowel]
    # print(vowels_indexed_df)

    # assert len(vowels_indexed_df) == len(ws), "n of vowels should be the same as length of vowel stresses"

    if len(vowels_indexed_df) != len(ws):
        status = "error: n of vowels should be the same as length of vowel stresses"
        print("n of vowels should be the same as length of vowel stresses")
        return {"status": status, "stress_intensities": [], "stress_binaries": []}

    try:
        vowels_indexed_df.loc[:,'stress_scores']=(100*ws).astype(int)
    except:
        status = "error: n of vowels should be the same as length of vowel stresses"
        print("n of vowels should be the same as length of vowel stresses")
        return {"status": status, "stress_intensities": [], "stress_binaries": []}
        # import pdb;pdb.set_trace()

    word_bins=[]
    word_intensities=[]
    for w_idx in vowels_indexed_df.word_idx.unique():
        word=vowels_indexed_df[vowels_indexed_df.word_idx==w_idx]
        bins=[0]*len(word)
        bins[word.stress_scores.argmax()]=1
        word_bins.append(bins)
        word_intensities.append(word.stress_scores.tolist())

    if level=="word":
        return {"status": "success", "stress_intensities": word_intensities, "stress_binaries": word_bins}
    elif level=="sentence":
        max_word_intensities=[max(w) for w in word_intensities]
        scores_grouped_by_chunk=[]
        cumsum=0
        for n in n_words_by_chunk:
            scores_grouped_by_chunk.append(max_word_intensities[cumsum:cumsum+n])
            cumsum+=n
        
        # I tried this on General English data, and in the end, it does not seem to improve
        # scores_grouped_by_chunk=[remove_downwards_trend(el) for el in scores_grouped_by_chunk]

        bins_by_chunk=[]
        for chunk in scores_grouped_by_chunk:
            bin=intensity_to_bin(chunk, n_max=roundup(len(chunk)/3))
            bins_by_chunk.append(bin)
        return {"status": "success", "stress_intensities": sum(scores_grouped_by_chunk,[]), "stress_binaries": sum(bins_by_chunk,[])}
    else:
        return {"status": "error: "+level+"is not a valid level in stress_from_formatted_phonetics. It has to be either 'word' or 'sentence'.", "stress_intensities": [], "stress_binaries": []}

def phonemeContrast_from_df_segmented(df_segmented,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=0, 
                            target_occurence_idx=0, # will be 0  all the time for vowels, and most of the time for consonants
                            target_phones='ER1',
                            alternatives=cmu_vowels,
                            model=default_model,
                            to_gibberish=cmu_to_gibberish,
                            **kwargs
                    ):

    g_t=[to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]

    phones_by_words=[el.split('_') for el in phonetics.replace('|','_').split(' ')]
    df_word=extract_word(df_segmented, phones_by_words, target_word_idx)
    phonetic_detection, detected_syllable=predict_phone(model.forced_aligner, df_word, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx, phoneme_set=alternatives) #, GT_proba_threshold=0.2)


    # if it's nan
    if detected_syllable!=detected_syllable: 
        return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

    if 'SIL' in phonetic_detection: 
        return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

    g_d=[to_gibberish[unstress(p)] for p in detected_syllable]
    
    # post-correction: for the target, when we are in a case of accepted alternative in prediction, we replace it with the GT
    if unstress(target_phones) in target_accepted_alternatives:
        if unstress(phonetic_detection) in target_accepted_alternatives[unstress(target_phones)]:
            phonetic_detection=unstress(target_phones)
            g_d=g_t
    
    # print(phonetic_detection)
    
    # phonetic detection needs to be the stressed version for backwards compatibility (when correct, when it's not, we don't care)
    # print("phonetic_detection",phonetic_detection)
    # print("target_phones",target_phones)
    if phonetic_detection==unstress(target_phones): 
        phonetic_detection=target_phones
        g_d=g_t
    return {"status": "success", "phonetic_detection": phonetic_detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}

def phonemeContrast_from_formatted_phonetics_audio(audio,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=0, 
                            target_occurence_idx=0, # will be 0  all the time for vowels, and most of the time for consonants
                            target_phones='ER1',
                            alternatives=cmu_vowels,
                            max_speech_rate=8, mode='file', 
                            model=default_model,
                            to_gibberish=cmu_to_gibberish,
                            **kwargs
                    ):
    phonetics=phonetics.replace('CH', 'T_SH').replace('JH','D_ZH')
    g_t=[to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
    audio_status, _, phone_prob_matrix = audio_to_phone_prob_matrix(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode, model=default_model)
    if audio_status!="success": 
        return {"status": audio_status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}
    df_segmented=phone_prob_matrix_segmentation(phone_prob_matrix, phonetics, model=model)

    # if the probabilities were too low, 
    # maybe change this to empty if we want the other feedback "are you saying the right words", 
    # or change null to "non-speech", nothing, nonsense or the pred_phones_audio
    if model.status!="success": 
        # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
        g_d=[to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in model.pred_phones_audio]
        if g_d==[]:
            return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  'nothing'}
        else:
            if len(g_d)>10:
                return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  'nonsense'}
            else:
                return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  '_'.join(g_d)}
        
    return phonemeContrast_from_df_segmented(df_segmented,phonetics=phonetics,target_word_idx=target_word_idx,target_syllable_idx=target_syllable_idx,target_occurence_idx=target_occurence_idx,target_phones=target_phones,alternatives=alternatives,model=model,to_gibberish=to_gibberish)



target_to_basis={
    "HH":"HH",
    '':'HH',
    'D':'IH0_D',
    'IH0_D':'IH0_D',
    'T':'IH0_T',
    'S':'IH0_S',
    'Z':'IH0_Z',
    'IH0_Z':'IH0_Z',
}


def phonetic_reference_processing(phonetics, target_word_idx, target_syllable_idx, target_phones, basis, position, contrast_idx):
    ### Phonetics processing to build a reference with the basis replacing target_phones
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')

    phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics.split(' ')[target_word_idx])

    syl_GT=remove_stress_annots(phonetics.split(' ')[target_word_idx].split('|')[target_syllable_idx].split('_'))
    split_phonetics_words=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
    split_phonetics_syls=[[syl.split('_') for syl in p.split('|')] for p in phonetics.split(' ')]

    # I cannot only split with "_" because splitting ''  with "_" generate a list with 1 element: [''],  and its len is 1.
    # So I filter these
    n_p_target=len([el for el in target_phones.split('_') if el!=''])
    n_ter_basis=len([el for el in basis.split('_') if el!=''])
    
    syl_idxs=phonetics_indexed_df.syl_idx.tolist()
    
    if position=="end":
        #in the target syllable, replace the barget with the basis
        split_phonetics_syls[target_word_idx][target_syllable_idx]=split_phonetics_syls[target_word_idx][target_syllable_idx][:-n_p_target]+basis.split('_')
        # in the split phonetics at word level, just replace the word with the new version by merging the syllables
        split_phonetics_words[target_word_idx]=sum(split_phonetics_syls[target_word_idx],[])
        GT=syl_GT[-n_p_target-1:]
        syl_idxs=syl_idxs[:-n_p_target]+[syl_idxs[contrast_idx]]*n_ter_basis
    elif position=="start":
        #in the target syllable, replace the barget with the basis
        split_phonetics_syls[target_word_idx][target_syllable_idx]=[el for el in basis.split('_') if el!='']+split_phonetics_syls[target_word_idx][target_syllable_idx][n_p_target:]

        # split_phonetics_words[target_word_idx]=basis.split('_')+target_word[n_p_target:]
        split_phonetics_words[target_word_idx]=sum(split_phonetics_syls[target_word_idx],[])
        
        GT=syl_GT[:n_p_target+1]
        syl_idxs=[syl_idxs[contrast_idx]]*n_ter_basis+syl_idxs[n_p_target:]
    
    return split_phonetics_words, syl_GT, GT, syl_idxs


def start_end_contrast_from_formatted_phonetics_audio(audio,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=None, 
                            target_phones='D',
                            # basis='[UNK]_D',
                            basis=None,
                            max_speech_rate=8, mode='file', 
                            position="end", 
                            model=default_model,
                            vowels=cmu_vowels,
                            consonants=cmu_consonants,
                            to_gibberish=cmu_to_gibberish,
                            **kwargs
                    ):
    audio_status, _, phone_prob_matrix = audio_to_phone_prob_matrix(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode, model=default_model)
    if audio_status!="success":
        return {"status": audio_status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}
    df_segmented=phone_prob_matrix_segmentation(phone_prob_matrix, phonetics, model=model)

    if position=="end": contrast_idx=-1
    elif position=="start": contrast_idx=0
    else: print("contrast should be start or end")
    
    # by default we apply the logic at the word level. End of word means last syllable, start of word means first syllable
    if target_syllable_idx is None:
        if position=="end": target_syllable_idx=-1
        elif position=="start": target_syllable_idx=0
        else:
            print("position should be start or end")

    if basis is None: basis=target_to_basis[target_phones]
    split_phonetics_by_words, syl_GT, GT, syl_idxs=phonetic_reference_processing(phonetics, target_word_idx, target_syllable_idx, target_phones, basis, position, contrast_idx)
    df_word=extract_word(df_segmented, split_phonetics_by_words, target_word_idx)
    # g_t=[to_gibberish[unstress(p)] for p in GT]

    g_t=[to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
    if model.status!="success": 
        # g_d=[to_gibberish[unstress(p)] for p in model.pred_phones_audio]
        g_d=[to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in model.pred_phones_audio]
        
        if g_d==[]:
            return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  'nothing'}
        else:
            if len(g_d)>10:
                return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  'nonsense'}
            else:
                return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  '_'.join(g_d)}
    
    print(df_word)


    # selection of the syllable, then divide it into a root and a termination (or start and root)
    df_word['syl_idx']=syl_idxs

    # As we use index based on numbers in dataframe, cyclical indexing is not completely supported. I just add the case here for the use of -1
    if target_syllable_idx==-1: target_syllable_idx=syl_idxs[-1]
    df_syl=df_word[df_word.syl_idx==target_syllable_idx]
    df_syl=df_syl[df_syl.pred_phones_audio!='[SIL]']

    # filter out phonemes too short inside the termination, 
    # if forced alignment lead to assigning very few frames for a phoneme, we assume it means it does not really exists
    # the minimal duration end-start difference is 0.02 (equals 1 frame), hence this filter:
    df_syl=df_syl[(df_syl.end-df_syl.start)>model.time_per_output]

    n_ter_basis=len([el for el in basis.split('_') if el!=''])
    #  In this case (e.g. starting h-) we want to use the fact that it is possible to reduce the set of possibilities to vowels or consonants as there is only 1 phoneme. 
    # It's the same method as in vowel/consonant contrast
    if n_ter_basis==1 and basis!='':
        p_idx=contrast_idx
        if unstress(basis) in consonants:
            phoneme_set=consonants
        elif unstress(basis) in vowels:
            phoneme_set=vowels
        
        phoneme_set_ids=[model.p_to_id[el] for el in remove_stress_annots(phoneme_set)]+["[SIL]"]

        if len(df_syl)>0:
            proba_means=df_syl.iloc[p_idx].proba_means

            # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
            filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
            idx_mean_max=np.argmax(filtered_proba_means)
            phonetic_detection=model.id_to_p[int(idx_mean_max)]

            # As we use index based on numbers in dataframe, cyclical indexing can be supported by taking the the last df index from the list
            df_syl.loc[df_syl.index[p_idx],"pred_phones_audio"]=phonetic_detection
    
    print(df_syl)

    # syl_detected=drop_consecutive_duplicates(df_syl[['pred_phones_audio']]).pred_phones_audio.tolist()
    syl_detected=df_syl[['pred_phones_audio']].pred_phones_audio.tolist()

    if position=="end":
        ter=syl_detected[-n_ter_basis-1:]
    elif position=="start":
        # detected start
        ter=syl_detected[:n_ter_basis+1]
    
    ter=drop_consecutive_duplicate_elements(ter)
    
    print('ter', ter)
    if len(ter)==len(GT):
        ter_post=[]
        for i,p in enumerate(ter):
            # post-correction: for the target, when we are in a case of accepted alternative in prediction, we replace it with the GT
            if unstress(GT[i]) in target_accepted_alternatives:
                if unstress(p) in target_accepted_alternatives[unstress(GT[i])]:
                    ter_post.append(unstress(GT[i]))
                else:
                    ter_post.append(unstress(p))
            else:
                ter_post.append(unstress(p))
    else: ter_post=ter

    print('ter_post', ter_post)

    # there might be consecutive duplicates when we concatenate root and ter_post
    # g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in root+ter_post])
    
    # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
    g_d=[to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in ter_post]
    
    if position=="end":
        detection=ter_post[1:]
    elif position=="start":
        detection=ter_post[:-1]

    target_phones_list=[el for el in target_phones.split('_') if el!='']
    
    # for words like "ordered", at this step detection would be ['ER', 'D'], and won't be in terminations_accepted_alternatives of "D"
    # because it is an accepted alternative of "IH_D"
    if len(detection)>0:
        if position=="end":
            if detection==syl_GT[-len(detection):]: detection=target_phones_list
        elif position=="start":
            if detection==syl_GT[:-len(detection)]: detection=target_phones_list

    detection='_'.join(detection)

    print("detection before post corr of whole termination", detection)
    print('target_phones', target_phones)
    
    # phonetic detection needs to be the stressed version for backwards compatibility 
    # however, here I convert to stressed version only when correct
    if target_phones!='': # don't try to split in case of case of final -s "nothing" target
        # if dectection is the same as target_phones (without the stress marks because charsiu don't put that), change back to target phones
        if detection.split('_')==remove_stress_annots(target_phones_list): detection=target_phones

        # post-correction for the whole termination not to differentiate between IH_D and AH_D     or    IH_Z and AH_Z
        unstressed_target='_'.join(remove_stress_annots(target_phones_list))
        if unstressed_target in terminations_accepted_alternatives:
            if detection in terminations_accepted_alternatives[unstressed_target]:
                detection=target_phones

    print('detection', detection)
    return {"status": "success", "phonetic_detection": detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}


def phonetic_content_analysis(s, phonetics, model=default_model,
                            vowels=cmu_vowels,
                            consonants=cmu_consonants):
    phonetic_content=model.analyze_phonetic_content(s, phonetics)
    if len(phonetic_content)==0: return phonetic_content
    
    def syl_analysis(syl_df):
        # inside a syllable or word, there cannot be several times the same phoneme consecutively
        collapsed_syl=drop_consecutive_duplicates(syl_df[['pred_phones_audio']])

        # one syllable in ground truth can correspond in several syllables in prediction, e.g. moved -> movED
        # or also in "0 syllable" if there is no vowel. If that's the case,  I have to consider it is 1 syllable
        pred_syls=SonoriPy(collapsed_syl.pred_phones_audio.tolist())[0]
        if pred_syls==[]: pred_syls=[collapsed_syl.pred_phones_audio.tolist()]

        # extract syllable indices for predicted syls
        pred_syls_indxs=sum([[i]*n for i,n in enumerate([len(syl) for syl in pred_syls])], [])

        collapsed_syl.loc[:,'pred_syls_indxs_inside_GT_syl']=pred_syls_indxs

        # I align the collapsed syllable to the timed one. This leads to NaNs that have to be filled
        syl_df.loc[:,'pred_syls_indxs_inside_GT_syl']=collapsed_syl.loc[:,'pred_syls_indxs_inside_GT_syl'].astype(int)
        syl_df=syl_df.fillna(method="ffill")

        # in each syllable in prediction, I only keep one vowel, by majority vote, i.e. I drop all vowels except max frames in each syl
        # for i in syl_df.pred_syls_indxs_inside_GT_syl.unique():
        #     s=syl_df.loc[syl_df.pred_syls_indxs_inside_GT_syl==i]
        #     v=s[s.pred_phones_audio.isin(vowels)]
        #     if len(v)>0:
        #         m=v.n_frames.idxmax()
        #         l=[el for el in v.index.tolist() if el != m]
        #         syl_df=syl_df.drop(l)
        
        return syl_df

    # reduction: we go in each syllable
    dfs=[]
    for w_idx in range(phonetic_content.word_idx.values[-1]+1):
        w_df=phonetic_content[phonetic_content.word_idx==w_idx]
        for s_idx in range(w_df.syl_idx.values[-1]+1):
            s_df=w_df[phonetic_content.syl_idx==s_idx]
            syl_df=syl_analysis(s_df)
            dfs.append(syl_df)
    phonetic_content=pd.concat(dfs)

    # post-correction : for each row when we are in a case of accepted alternative in prediction, we replace it with the GT
    for i,r in phonetic_content.iterrows():
        if r.phones in target_accepted_alternatives:
            if r.pred_phones_audio in target_accepted_alternatives[r.phones]:
                phonetic_content.loc[i, 'pred_phones_audio']=r.phones

    # phonetic_content=phonetic_content[phonetic_content.n_frames>1]
    phonetic_content=phonetic_content[phonetic_content.pred_phones_audio!='[SIL]']
    phonetic_content=phonetic_content.loc[drop_consecutive_duplicates(phonetic_content[['phones','pred_phones_audio']]).index,:]
    return phonetic_content

def syllable_contrast_from_formatted_phonetics_audio(audio,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=0,
                            max_speech_rate=8, mode='file',
                            to_gibberish=cmu_to_gibberish,
                            model=default_model,
                            **kwargs
                    ):
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    audio_status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    g_t=[to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]

    if audio_status!="success":
        return {"status": audio_status,  "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

    phonetic_content=phonetic_content_analysis(s, phonetics)
    if model.status!="success": return {"status": model.status, "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

    if len(phonetic_content)==0: return {"status": "phonetic_content is empty", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

    syllable_content=phonetic_content[phonetic_content.word_idx==target_word_idx][phonetic_content.syl_idx==target_syllable_idx]
    detected_syllable=syllable_content.pred_phones_audio.tolist()
    g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in detected_syllable])

    g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in detected_syllable])
    return {"status": "success", "phonetic_detection":detected_syllable,  "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    
def analyze_start_end_for_synth_word(word, words_selected_df, target_word_idx=0,target_syllable_idx=-1,target_phones='Z',basis=None,position='end',model=default_model):
    from src.audio_processing import read_audio_file
    from src.text_processing import prefill_for_sentence
    
    phonetics=prefill_for_sentence(word)['phonetics']

    results=[]
    for i,r in words_selected_df.iterrows():
        audio,fs=read_audio_file(r.path, fs=16000)
        # default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
        res=start_end_contrast_from_formatted_phonetics_audio(audio,phonetics=phonetics,target_word_idx=target_word_idx, target_syllable_idx=target_syllable_idx,target_phones=target_phones,basis=basis,position=position,mode='numpy',model=model)
        results.append(res)
    
    return results


def use_tests():
    # from DL_speech_tech import *
    from src.label_data_processing import actor_recordings, final_s_artificial_data, synth_words_data
    df_words=synth_words_data().dropna()

    word="perhaps"
    words_selected_df=df_words[df_words.text==word]
    analyze_start_end_for_synth_word(word, words_selected_df, target_word_idx=0,target_syllable_idx=1,target_phones='HH',position='start',model=default_model)

    from src.audio_processing import read_audio_file
    from src.text_processing import prefill_for_sentence
    from src.label_data_processing import actor_recordings, synth_words_data

    word="hundred"
    phonetics=prefill_for_sentence(word)['phonetics']
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_"+word+".mp3"
    audio,fs=read_audio_file(path, fs=16000)
    # default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    res=start_end_contrast_from_formatted_phonetics_audio(audio,phonetics=phonetics,target_word_idx=0,target_syllable_idx=0,target_phones='N',basis="N",position='start',mode='numpy',model=default_model)


    word="orders"
    phonetics=prefill_for_sentence(word)['phonetics']
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_"+word+".mp3"
    audio,fs=read_audio_file(path, fs=16000)
    # default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    res=start_end_contrast_from_formatted_phonetics_audio(audio,phonetics=phonetics,target_word_idx=0,target_phones='Z',position='end',mode='numpy',model=default_model)
    # res=start_end_contrast_from_formatted_phonetics_audio(audio,phonetics=phonetics,target_word_idx=0,target_phones='Z',position='end',mode='numpy',model=default_model_charsiu)

    
    word="history"
    phonetics=prefill_for_sentence(word)['phonetics']
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_"+word+".mp3"
    s,fs=read_audio_file(path, fs=16000)
    # default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=phonetics,target_word_idx=0,target_phones='Z',position='end',mode='numpy',model=default_model)


    path='data/synth_audio/cmu_words/standard/prosody/Brian/M_UK_ekk.mp3'
    formatted_phonetics=prefill_for_sentence('ekk')['phonetics']
    s,fs=read_audio_file(path, fs=16000)
    phonemeContrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, 
                                                        target_word_idx=0, 
                                                        target_syllable_idx=1, 
                                                        target_phones='EY1',
                                                        alternatives=cmu_vowels, mode='numpy')

    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    # path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'
    # path='data/audio_recordings/turned_around.mp3'
    s,fs=read_audio_file(path, fs=16000)
    formatted_phonetics=prefill_for_sentence('I would love to go to ireland')['phonetics']
    stress_from_formatted_phonetics(s,phonetics=formatted_phonetics, 
                                    level="sentence", 
                                    n_words_by_chunk=[7],
                                    max_speech_rate=8, mode='numpy'
                                    )

    
    # path='data/synth_audio/cmu_words/standard/prosody/Brian/M_UK_international.mp3'
    path="data/synth_audio/cmu_words/test_Lea_french_accent/F_FR_annotation.mp3"
    s,fs=read_audio_file(path, fs=16000)
    formatted_phonetics=prefill_for_sentence('annotation')['phonetics']
    stress_from_formatted_phonetics(s,phonetics=formatted_phonetics, 
                                    level="word", 
                                    # n_words_by_chunk=[7],
                                    max_speech_rate=8, mode='numpy'
                                    )
    
    path='data/synth_audio/cmu_words/standard/prosody/Brian/M_UK_france.mp3'
    s,fs=read_audio_file(path, fs=16000)
    formatted_phonetics=prefill_for_sentence('france')['phonetics']
    default_model.predict_with_timings(s, sum(sum(split_phonetics(formatted_phonetics),[]),[]))
    stress_from_formatted_phonetics(s,phonetics=formatted_phonetics,level="word", mode='numpy')
    
    df=synth_words_data().dropna()
    df['target_word_indexes']=0
    df['target_syllable_indexes']=0
    df['contrast']='start'
    df['fpath']=df['path']
    clusters=["TH_R", "P_R", "S_P_L", "S_K_R"]
    cluster=clusters[-1]
    
    df.phonetics.str.startswith(cluster)
    df_c=df[df.phonetics.str.startswith(cluster)]

    row=df_c.iloc[0]
    s,fs=read_audio_file(row.path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=row.phonetics, target_word_idx=0, target_phones=cluster,basis=cluster,position='start',mode='numpy')
    
    
    word="listens"
    phonetics=prefill_for_sentence(word)['phonetics']
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_"+word+".mp3"
    s,fs=read_audio_file(path, fs=16000)
    # default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=phonetics,target_word_idx=0,target_phones='Z',position='end',mode='numpy',model=default_model)

    
    cmu_phonetics="L_IH1|S_AH0_N"
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_listen.mp3"
    s,fs=read_audio_file(path, fs=16000)
    default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=cmu_phonetics,target_word_idx=0,target_phones='Z',position='end',mode='numpy',model=default_model)

    
    cmu_phonetics="L_IH1|S_AH0_N_D"
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_listen.mp3"
    s,fs=read_audio_file(path, fs=16000)
    default_model.predict_with_timings(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    # default_model_charsiu.align_phones(s, sum(sum(split_phonetics(cmu_phonetics),[]),[]))
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=cmu_phonetics,target_word_idx=0,target_phones='D',position='end',mode='numpy',model=default_model)
    # res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=cmu_phonetics,target_word_idx=0,target_phones='D',position='end',mode='numpy',model=default_model_charsiu)

    
    
    df=synth_words_data()
    df['target_word_indexes']=0
    df['target_syllable_indexes']=-1
    df['fpath']=df['path']
    df=df.dropna()

    df_z=df[(~(df.phonetics.str.endswith('AH0_Z')|df.phonetics.str.endswith('IH0_Z'))&~df.phonetics.str.endswith('_S'))&df.text.str.endswith('s')]
    r=df_z.sample(frac=1, random_state=0)[:100].iloc[-4]
    s,fs=read_audio_file(r.path, fs=16000)
    # res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=r.phonetics,target_word_idx=0,target_syllable_idx=-1,target_occurence_idx=0,target_phones='Z',basis='Z_Z',position='end',mode='numpy',model=default_model_charsiu)

    # res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=r.phonetics,target_word_idx=0,target_syllable_idx=-1,target_occurence_idx=0,target_phones='Z',basis='Z_Z',position='end',mode='numpy',model=default_model_charsiu)

    

    
    formatted_phonetics=prefill_for_sentence('turned around', mode='CMU')['phonetics']
    s,fs=read_audio_file('data/audio_recordings/turnEED_around.mp3', fs=16000)
    # s,fs=read_audio_file('data/audio_recordings/turned_around.mp3', fs=16000)
    default_model.predict_phone_prob_matrix(s,fs).shape
    default_model.predict_with_timings(s, sum(sum(split_phonetics(formatted_phonetics),[]),[]))
    default_model.analyze_phonetic_content(s, formatted_phonetics)
    phonetic_content=phonetic_content_analysis(s, formatted_phonetics)
    phonetic_content[['phones', 'start_idx', 'end_idx', 'pred_phones_audio','proba_means', 'GT_proba', 'start', 'end', 'n_times']]
    
    syllable_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, 
                            target_word_idx=1, 
                            target_syllable_idx=0,
                            max_speech_rate=8, mode='numpy',
                            to_gibberish=cmu_to_gibberish,
                            model=default_model
                    )
    

    prefill_for_sentence('expected', mode='MFA_IPA')['cmu_phonetics']

    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    model_ipa = Wav2Vec2ForFramePrediction('ipa')
    model_ipa.load(name='model_mailabs_pca_0.95_knn_10_cos_w_UK_US_FR_ES')

    formatted_phonetics=prefill_for_sentence('turned around', mode='MFA_IPA')['cmu_phonetics']
    
    # s,fs=librosa.load('data/audio_recordings/turnEED_around.mp3', sr=16000)
    s,fs=read_audio_file('data/audio_recordings/turned_around.mp3', fs=16000)
    model_ipa.predict_phone_prob_matrix(s,fs).shape
    model_ipa.predict_with_timings(s, sum(sum(split_phonetics(formatted_phonetics),[]),[]))

    s,fs=read_audio_file('data/audio_recordings/turned_around.mp3', fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, target_word_idx=0, target_phones='d',basis='ɪ_d',
                            model=model_ipa,
                            vowels=ipa_vowels,
                            consonants=ipa_consonants,
                            to_gibberish=ipa_to_gibberish,
                            mode='numpy')
    
    
    
    s,fs=read_audio_file('data/audio_recordings/turnEED_around.mp3', fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, target_word_idx=0, target_phones='D',model=default_model,mode='numpy')
    
    
    path='data/synth_audio/cmu_words/standard/prosody/Amy/F_UK_hate.mp3'
    formatted_phonetics=prefill_for_sentence('hate')['phonetics']
    s,fs=read_audio_file(path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, 
                            target_word_idx=0, 
                            target_phones='HH',
                            basis='HH', position="start", mode='numpy'
                    )
    
    path='data/synth_audio/cmu_words/standard/prosody/Amy/F_UK_ate.mp3'
    formatted_phonetics=prefill_for_sentence('ate')['phonetics']
    s,fs=read_audio_file(path, fs=16000)
    default_model.predict_with_timings(s, sum(sum(split_phonetics(formatted_phonetics),[]),[]))
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, 
                            target_word_idx=0, 
                            target_phones='',
                            basis='HH', position="start", mode='numpy'
                    )
    
    df=actor_recordings()
    target_phones='T'
    selection=df[df.target_phoneme==target_phones]
    row=selection.iloc[0]
    s,fs=read_audio_file(path, fs=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=row.cmu_phonetics, 
                            target_word_idx=0, 
                            target_phones=target_phones,
                            basis='IH0_D', mode='numpy'
                    )