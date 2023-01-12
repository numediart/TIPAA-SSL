from scipy.io.wavfile import  read
import numpy as np
import pandas as pd
from src.audio_processing import getIntonation, read_audio_string, read_audio_bytes
import soundfile as sf
from src.text_processing import unstress, split_phonetics, remove_stress_annots, drop_consecutive_duplicates, drop_consecutive_duplicate_elements, phonetics_indexed_df_from_formatted_phonetics
from src.pronunciation_dictionaries import cmu_vowels, cmu_stressed_vowels, cmu_consonants, cmu_to_gibberish
from src.pronunciation_dictionaries import ipa_vowels, ipa_consonants, ipa_to_gibberish
from syllabipy.sonoripy import SonoriPy
import base64
from linetimer import CodeTimer

# initialize model
from src.charsiu_utils import charsiu_phone_forced_aligner
default_model_charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')


phoneme_GT_proba_threshold_dict={}
default_thresh=0.2
for k in cmu_stressed_vowels: phoneme_GT_proba_threshold_dict[k]=default_thresh
for k in cmu_consonants: phoneme_GT_proba_threshold_dict[k]=default_thresh

# phoneme_GT_proba_threshold_dict['AO0']=0.1
# phoneme_GT_proba_threshold_dict['AO1']=0.1
# phoneme_GT_proba_threshold_dict['AO2']=0.1

# model=pickle.load(open('model_mailabs_umap_2_gmm_300.pkl','rb'))

from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
default_model = Wav2Vec2ForFramePrediction('cmu')
# default_model.load(name='model_mailabs_pca_0.95_knn_10_w')
# default_model.load(name='model_mailabs_pca_99_logistic_regression')
default_model.load(name='model_mailabs_pca_99_knn_5_cos_w')


target_accepted_alternatives={
    # 'AA': ['AA', 'AO'],
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
# but if I accept all vowels, it mens I wouldn't give feedback for a mistake like "S_T_AA_R_T_EY_D" for "started"
# therefore, I accept only consonants. It's also more likely to have this border effect with consonant because the target is a consonant (verified experimentally looking at confusions)

# all_Z=[p+"_Z" for p in (list(cmu_vowels) + list(cmu_consonants))]
# all_S=[p+"_S" for p in (list(cmu_vowels) + list(cmu_consonants))]
all_Z=[p+"_Z" for p in list(cmu_consonants)]+["Z_"+p for p in list(cmu_consonants)]
all_S=[p+"_S" for p in list(cmu_consonants)]+["S_"+p for p in list(cmu_consonants)]
terminations_accepted_alternatives["Z"]=[el for el in all_Z if el not in terminations_accepted_alternatives['IH_Z']]
terminations_accepted_alternatives["S"]=[el for el in all_S if el not in terminations_accepted_alternatives['IH_Z']]


# all_Z=[p+"_Z" for p in (list(cmu_vowels) + list(cmu_consonants))]
# all_S=[p+"_S" for p in (list(cmu_vowels) + list(cmu_consonants))]
all_D=[p+"_D" for p in list(cmu_consonants)]+["D_"+p for p in list(cmu_consonants)]
all_T=[p+"_T" for p in list(cmu_consonants)]+["T_"+p for p in list(cmu_consonants)]
terminations_accepted_alternatives["D"]=[el for el in all_D if el not in terminations_accepted_alternatives['IH_D']]
terminations_accepted_alternatives["T"]=[el for el in all_T if el not in terminations_accepted_alternatives['IH_D']]

def audio_load_and_check(audio, phonetics, max_speech_rate=8, mode='file', fs=16000):
    """Load audio with 2 modes: from a "file", from "base64" encoding, from "bytes", or directly a "numpy" array
    Then check duration to see if it's plausible
    
    I first detect if the audio is too short to have a realistic speech rate
        https://www.science.org/doi/10.1126/sciadv.aaw2594
    https://www.reddit.com/r/languagelearning/comments/f5o1om/distribution_of_syllable_rate_sr_in_syllables_per/
    Speech rate is always between 5 and 8 syl/second
    """
    n_syllables_tot=sum([len(el.split('|')) for el in phonetics.split(' ')])

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
        return "error: mode for audio_load_and_check() must be file or base64", None
    
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



###################   Pronunciation aspect functions  ################


def stress_from_formatted_phonetics(audio,phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D", 
                                    # text="I would love to go to Ireland!", 
                                    n_words_by_chunk=[7],
                                    level="sentence", 
                                    # chunking_chars=[',',';','.','!','?', ':', '/'],
                                    max_speech_rate=8, mode='file',
                                    model=default_model_charsiu,
                                    vowels=cmu_vowels
                                    ): #'[\,\?\.\!\;\:\"\*]'
    
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    with CodeTimer('load audio'):    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    if status=="success":
        # print(phonetics)

        with CodeTimer('inference + stress'): ws=model.compute_stress_score(s,phonetics)
        if model.status!="success": 
            return {"status": model.status, "stress_intensities": [], "stress_binaries": []}

        # TODO: I think I should check for voiceness, but I don't know if I should do it for all vowels
        if sum([el!=el for el in ws])==len(ws):
            status = "success: no voiced sound detected inside supposed vowels (no pitch detected)"
            return {"status": status, "stress_intensities": [], "stress_binaries": []}

        phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics)
        is_vowel=phonetics_indexed_df.apply(lambda r: unstress(r.phones) in vowels, axis=1)
        vowels_indexed_df=phonetics_indexed_df[is_vowel]
        # print(vowels_indexed_df)
        vowels_indexed_df.loc[:,'stress_scores']=(100*ws).astype(int)
    else:
        return {"status": status, "stress_intensities": [], "stress_binaries": []}

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
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    g_t=[to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
    if status=="success":
        # try:
        phonetic_detection, detected_syllable=model.predict_phone(s, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx, phoneme_set=alternatives)#, GT_proba_threshold=phoneme_GT_proba_threshold_dict[target_phones])
        # except:
        #     import pdb;pdb.set_trace()

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
    
        # if it's nan
        if detected_syllable!=detected_syllable: 
            return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

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
    else:
        return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

def start_end_contrast_from_formatted_phonetics_audio(audio,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_phones='D',
                            # basis='[UNK]_D',
                            basis='IH0_D',
                            max_speech_rate=8, mode='file', contrast="end", 
                            model=default_model,
                            vowels=cmu_vowels,
                            consonants=cmu_consonants,
                            to_gibberish=cmu_to_gibberish,
                            **kwargs
                    ):
    
    if contrast=="end":
        contrast_idx=-1
    elif contrast=="start":
        contrast_idx=0
    else:
        print("contrast should be start or end")

    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)

    phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics.split(' ')[target_word_idx])
    idx_syl_ter=phonetics_indexed_df.syl_idx.iloc[contrast_idx]
    df_syl_GT=phonetics_indexed_df[phonetics_indexed_df.syl_idx==idx_syl_ter]
    syl_GT=remove_stress_annots(df_syl_GT.phones.tolist())

    n_p_target=len(target_phones.split('_'))
    n_ter_basis=len(basis.split('_'))

    if contrast=="end":
        GT=syl_GT[-n_p_target-1:]
    elif contrast=="start":
        GT=syl_GT[:n_p_target+1]

    g_t=[to_gibberish[unstress(p)] for p in GT]
    if status=="success":
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        target_word=split_phonetics[target_word_idx]

        if contrast=="end":
            split_phonetics[target_word_idx]=target_word[:-n_p_target]+basis.split('_')
        elif contrast=="start":
            split_phonetics[target_word_idx]=basis.split('_')+target_word[n_p_target:]
        
        # same change for indexes
        syl_idxs=phonetics_indexed_df.syl_idx.tolist()
        if contrast=="end":
            syl_idxs=syl_idxs[:-n_p_target]+[syl_idxs[contrast_idx]]*len(basis.split('_'))
        elif contrast=="start":
            syl_idxs=[syl_idxs[contrast_idx]]*len(basis.split('_'))+syl_idxs[n_p_target:]

        df_word=model.predict_word(s, split_phonetics, target_word_idx)

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


        #  In this case (e.g. starting -h) we want to use the fact that it is possible to reduce the set of possibilities to vowels or consonants as there is only 1 phoneme. It's the same method as in vowel/consonant contrast
        if n_ter_basis==1 and basis!='':
            p_idx_global=contrast_idx
            if unstress(basis) in consonants:
                phoneme_set=consonants
            elif unstress(basis) in vowels:
                phoneme_set=vowels
            
            phoneme_set_ids=[model.p_to_id[el] for el in remove_stress_annots(phoneme_set)]
            proba_means=df_word.iloc[p_idx_global].proba_means

            # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
            filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
            idx_mean_max=np.argmax(filtered_proba_means)

            # phonetic_detection=model.charsiu_processor.mapping_id2phone(int(idx_mean_max))
            phonetic_detection=model.id_to_p[int(idx_mean_max)]

            ter=[phonetic_detection]

        else:
            # selection of the syllable, then divide it into a root and a termination (or start and root)
            df_word['syl_idx']=syl_idxs
            df_syl=df_word[df_word.syl_idx==syl_idxs[contrast_idx]]
            print(df_syl)
            df_syl=df_syl[df_syl.pred_phones_audio!='[SIL]']
            syl_detected=drop_consecutive_duplicates(df_syl[['pred_phones_audio']]).pred_phones_audio.tolist()

            if contrast=="end":
                # root based on GT
                root=df_syl.phones.tolist()[:-n_ter_basis]
                # detected termination
                ter=syl_detected[len(root)-1:]
            elif contrast=="start":
                # detected start
                ter=syl_detected[:n_ter_basis+1]
                # root=df_syl.phones.tolist()[:-n_ter_basis]
        
        print(ter)
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

        # there might be consecutive duplicates when we concatenate root and ter_post
        # g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in root+ter_post])
        
        # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
        g_d=[to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in ter_post]
        
        if contrast=="end":
            detection=ter_post[1:]
        elif contrast=="start":
            detection=ter_post[:-1]

        detection='_'.join(detection)

        # detection=ter_post
        
        # phonetic detection needs to be the stressed version for backwards compatibility 
        # (however, here I convert to stressed version only when correct, it might work, but could cause problems?)
        if target_phones!='': # don't try to split in case of case of final -s "nothing" target
            # if dectection is the same as target_phones (without the stress marks because charsiu don't put that), change back to target phones
            if detection.split('_')==remove_stress_annots(target_phones.split('_')): detection=target_phones
        
            # post-correction for the whole termination not to differentiate between IH_D and AH_D     or    IH_Z and AH_Z
            unstressed_target='_'.join(remove_stress_annots(target_phones.split('_')))
            if unstressed_target in terminations_accepted_alternatives:
                if detection in terminations_accepted_alternatives[unstressed_target]:
                    detection=target_phones
        
        # WARNING: that logic would accept IH_Z as Z which is not wished
        # if the end (or start) of detection is the same as target_phones, I want to consider it correct if the phoneme corresponds to what's in the root
        # e.g. the words id "cops", target in "S" and detection is "P_S" --> should be correct

        # # I want to do this on the stress version with post-correction from accepted alternatives
        # if len(detection.split('_'))>n_p_target:
        #     if contrast=="end":
        #         if '_'.join(detection.split('_')[-n_p_target:])==target_phones:
        #             detection=target_phones
        #     if contrast=="start":
        #         if '_'.join(detection.split('_')[:n_p_target])==target_phones:
        #             detection=target_phones

        print('detection', detection)
        return {"status": "success", "phonetic_detection": detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

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
    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    g_t=[to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]

    if status=="success":
        phonetic_content=phonetic_content_analysis(s, phonetics)
        if model.status!="success": return {"status": model.status, "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

        if len(phonetic_content)==0: return {"status": "phonetic_content is empty", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

        syllable_content=phonetic_content[phonetic_content.word_idx==target_word_idx][phonetic_content.syl_idx==target_syllable_idx]
        detected_syllable=syllable_content.pred_phones_audio.tolist()
        g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in detected_syllable])

        g_d=drop_consecutive_duplicate_elements([to_gibberish[unstress(p)] for p in detected_syllable])
        return {"status": "success", "phonetic_detection":detected_syllable,  "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status,  "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}
    

if __name__=="__main__":
    from DL_speech_tech import *

    from src.audio_processing import prepare_audio_file
    from src.text_processing import *
    from src.label_data_processing import *
    import librosa

    path='scripts/synth_audio/cmu_words/standard/prosody/Brian/M_UK_ekk.mp3'
    formatted_phonetics=prefill_for_sentence('ekk')['cmu_phonetics']
    s,fs=librosa.load(path, sr=16000)
    phonemeContrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, 
                                                        target_word_idx=0, 
                                                        target_syllable_idx=1, 
                                                        target_phones='EY1',
                                                        alternatives=cmu_vowels, mode='numpy')

    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    # path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'
    # path='data/audio_recordings/turned_around.mp3'
    encode_string = base64.b64encode(open(path, "rb").read())
    formatted_phonetics=prefill_for_sentence('I would love to go to ireland')['cmu_phonetics']
    stress_from_formatted_phonetics(encode_string,phonetics=formatted_phonetics, 
                                    level="sentence", 
                                    n_words_by_chunk=[7],
                                    max_speech_rate=8, mode='base64'
                                    )

    
    df=synth_words_data()
    df['target_word_indexes']=0
    df['target_syllable_indexes']=0
    df['contrast']='start'
    df['fpath']=df['path']
    clusters=["TH_R", "P_R", "S_P_L", "S_K_R"]
    cluster=clusters[-1]
    
    df.phonetics.str.startswith(cluster)
    df_c=df[df.phonetics.str.startswith(cluster)]

    row=df_c.iloc[0]
    s,fs=librosa.load(row.path, sr=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=row.phonetics, target_word_idx=0, target_phones=cluster,basis=cluster,
                            contrast='start',
                            mode='numpy')
    

    cmu_phonetics="T_AO1_S_T"
    path="scripts/synth_audio/cmu_words/standard/prosody/Joanna/F_US_tossed.mp3"
    s,fs=librosa.load(path, sr=16000)
    res=start_end_contrast_from_formatted_phonetics_audio(s,phonetics=cmu_phonetics,target_word_idx=0,target_syllable_idx=0,target_occurence_idx=0,target_phones='T',basis='IH0_D',contrast='end',mode='numpy',model=default_model_charsiu)

    


    

    
    formatted_phonetics=prefill_for_sentence('turned around', mode='CMU')['cmu_phonetics']
    import librosa
    # s,fs=librosa.load('data/audio_recordings/turnEED_around.mp3', sr=16000)
    s,fs=librosa.load('data/audio_recordings/turned_around.mp3', sr=16000)
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
    s,fs=librosa.load('data/audio_recordings/turned_around.mp3', sr=16000)
    model_ipa.predict_phone_prob_matrix(s,fs).shape
    model_ipa.predict_with_timings(s, sum(sum(split_phonetics(formatted_phonetics),[]),[]))



    # _, rID=prepare_audio_file('data/audio_recordings/turnEED_around.mp3')
    # _, rID=prepare_audio_file('data/audio_recordings/turned_around.mp3')
    s,fs=librosa.load('data/audio_recordings/turned_around.mp3', sr=16000)
    start_end_contrast_from_formatted_phonetics_audio(s,phonetics=formatted_phonetics, target_word_idx=0, target_phones='d',basis='ɪ_d',
                            model=model_ipa,
                            vowels=ipa_vowels,
                            consonants=ipa_consonants,
                            to_gibberish=ipa_to_gibberish,
                            mode='numpy')
    
    
    
    
    # phonemeContrast_from_formatted_phonetics_audio(rID,phonetics=formatted_phonetics, 
    #                         target_word_idx=0, 
    #                         target_syllable_idx=0, 
    #                         target_phones='ɝ',
    #                         alternatives=ipa_vowels,
    #                         model=model_ipa,
    #                         to_gibberish=ipa_to_gibberish)
    
    formatted_phonetics=prefill_for_sentence('turned around')['cmu_phonetics']
    _, rID=prepare_audio_file('data/audio_recordings/turnEED_around.mp3')
    # _, rID=prepare_audio_file('data/audio_recordings/turned_around.mp3')
    start_end_contrast_from_formatted_phonetics_audio(rID,phonetics=formatted_phonetics, 
                            target_word_idx=0, 
                            target_phones='D',
                            basis='IH0_D',
                    )
    
    path='scripts/synth_audio/cmu_words/standard/prosody/Amy/F_UK_hate.mp3'
    formatted_phonetics=prefill_for_sentence('hate')['cmu_phonetics']
    _, rID=prepare_audio_file(path)
    start_end_contrast_from_formatted_phonetics_audio(rID,phonetics=formatted_phonetics, 
                            target_word_idx=0, 
                            target_phones='HH',
                            basis='HH', contrast="start"
                    )
    
    path='scripts/synth_audio/cmu_words/standard/prosody/Amy/F_UK_ate.mp3'
    formatted_phonetics=prefill_for_sentence('ate')['cmu_phonetics']
    _, rID=prepare_audio_file(path)
    start_end_contrast_from_formatted_phonetics_audio(rID,phonetics=formatted_phonetics, 
                            target_word_idx=0, 
                            target_phones='',
                            basis='HH', contrast="start"
                    )
    
    df=actor_recordings()
    target_phones='T'
    selection=df[df.target_phoneme==target_phones]
    row=selection.iloc[0]
    _, rID=prepare_audio_file(row.audio_file_url)
    start_end_contrast_from_formatted_phonetics_audio(rID,phonetics=row.cmu_phonetics, 
                            target_word_idx=0, 
                            target_phones=target_phones,
                            basis='IH0_D',
                    )