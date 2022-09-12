from scipy.io.wavfile import  read
import numpy as np
import pandas as pd
from utils.audio_processing import getIntonation, getIntensity, normalize
import soundfile as sf
import io
from utils.text_processing import unstress, split_phonetics, remove_stress_annots, drop_consecutive_duplicates, drop_consecutive_duplicate_elements, chunk_text, phonetics_indexed_df_from_formatted_phonetics
from utils.pronunciation_dictionaries import cmu_vowels, cmu_consonants, cmu_to_gibberish
from syllabipy.sonoripy import SonoriPy
from utils.charsiu_utils import charsiu_phone_forced_aligner
import base64
import librosa

# initialize model
model = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')

target_accepted_alternatives={
    'AA': ['AA', 'AO'],
    'AO': ['AA', 'AO'],
    'D': ['D', 'T'],
    'Z': ['Z', 'S'],
    # 'T': ['D', 'T'],
    # 'IH': ['IH', 'AH', 'EH']
}

terminations_accepted_alternatives={
    'IH_Z':['AH_Z','IH_Z'],
    'IH_D':['AH_D','IH_D']
    }

def audio_load_and_check(audio, phonetics, max_speech_rate=8, mode='file', fs=16000):
    """Load audio with 2 modes: from a "file" or from "base64" encoding
    Then check duration to see if it's plausible
    """
    # I first detect if the audio is too short to have a realistic speech rate
    #     https://www.science.org/doi/10.1126/sciadv.aaw2594
    # https://www.reddit.com/r/languagelearning/comments/f5o1om/distribution_of_syllable_rate_sr_in_syllables_per/
    # Speech rate is always between 5 and 8 syl/second
    n_syllables_tot=sum([len(el.split('|')) for el in phonetics.split(' ')])

    if mode=='file':
        try:
            f=sf.SoundFile('./inputs/'+ audio+ '.wav')
        except FileNotFoundError:
            return "error: audio file not found", None
        duration=f.frames / f.samplerate
        speech_rate=n_syllables_tot/duration
        if speech_rate>max_speech_rate: 
            return "success: audio is too short compared to the expected number of syllables", None

        try:
            fs,s=read('./inputs/'+ audio+ '.wav')
            s=s/32767
        except FileNotFoundError:
            return "error: audio file not found", None
    elif mode=='base64':
        decode_string = base64.b64decode(audio)
        # s,fs=sf.read(io.BytesIO(decode_string))
        s,fs=librosa.load(io.BytesIO(decode_string), sr=fs)
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

def stress_from_formatted_phonetics(audio,phonetics="AY1 W_UH1_D L_AH1_V T_UW1 G_OW1 T_UW1 AY1|ER0|L_AH0_N_D", 
                                    # text="I would love to go to Ireland!", 
                                    n_words_by_chunk=[7],
                                    level="sentence", 
                                    # chunking_chars=[',',';','.','!','?', ':', '/'],
                                    max_speech_rate=8, mode='file'
                                    ): #'[\,\?\.\!\;\:\"\*]'
    
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    if status=="success":
        # print(phonetics)
        ws=model.compute_stress_score(s,phonetics)
        if model.status!="success": 
            return {"status": model.status, "stress_intensities": [], "stress_binaries": []}

        # TODO: I think I should check for voiceness, but I don't know if I should do it for all vowels
        if sum([el!=el for el in ws])==len(ws):
            status = "success: no voiced sound detected inside supposed vowels (no pitch detected)"
            return {"status": status, "stress_intensities": [], "stress_binaries": []}

        phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics)
        is_vowel=phonetics_indexed_df.apply(lambda r: unstress(r.phones) in cmu_vowels, axis=1)
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
                            max_speech_rate=8, mode='file', **kwargs
                    ):
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    g_t=[cmu_to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
    if status=="success":
        # try:
        phonetic_detection, detected_syllable=model.predict_phone(s, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx, phoneme_set=alternatives)
        # except:
        #     import pdb;pdb.set_trace()

        # if the probabilities were too low, 
        # maybe change this to empty if we want the other feedback "are you saying the right words", 
        # or change null to "non-speech", nothing, nonsense or the pred_phones_audio
        if model.status!="success": 
            # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
            g_d=[cmu_to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in model.pred_phones_audio]
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

        g_d=[cmu_to_gibberish[unstress(p)] for p in detected_syllable]
        
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
                            max_speech_rate=8, mode='file', contrast="end", **kwargs
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

    g_t=[cmu_to_gibberish[unstress(p)] for p in GT]
    # g_t=[cmu_to_gibberish[unstress(p)] for p in syl_GT]
    # g_t=[cmu_to_gibberish[unstress(p)] for p in target_phones.split('_')]
    # g_t=[cmu_to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]
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
        # print(df_word)
        # print(phonetics_indexed_df)

        if model.status!="success": 
            g_d=[cmu_to_gibberish[unstress(p)] for p in model.pred_phones_audio]
            if g_d==[]:
                return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  'nothing'}
            else:
                if len(g_d)>10:
                    return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  'nonsense'}
                else:
                    return {"status": model.status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  '_'.join(g_d)}

        if n_ter_basis==1 and basis!='':
        # if False:
            #  In this case we want to use the fact that it is possible to reduce the set of possibilities to vowels or consonants as there is only 1 phoneme. It's the same method as in vowel/consonant contrast
            p_idx_global=contrast_idx
            if unstress(basis) in cmu_consonants:
                phoneme_set=cmu_consonants
            elif unstress(basis) in cmu_vowels:
                phoneme_set=cmu_vowels
                
            phoneme_set=[[p] for p in  remove_stress_annots(phoneme_set)]
            phoneme_set_ids=model.charsiu_processor.get_phone_ids(phoneme_set)[1:-1]
            proba_means=df_word.iloc[p_idx_global].proba_means
            # if GT_proba is beyond the threshold, we take it as prediction
            # if df_word.iloc[p_idx_global].GT_proba>GT_proba_threshold:
            #     phonetic_detection=target_phones
            # else:

            # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
            filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
            idx_mean_max=np.argmax(filtered_proba_means)
            phonetic_detection=model.charsiu_processor.mapping_id2phone(int(idx_mean_max))

            ter=[phonetic_detection]

        else:
            df_word['syl_idx']=syl_idxs
            df_syl=df_word[df_word.syl_idx==syl_idxs[contrast_idx]]

            df_syl=df_syl[df_syl.pred_phones_audio!='[SIL]']
            syl_detected=drop_consecutive_duplicates(df_syl[['pred_phones_audio']]).pred_phones_audio.tolist()

            if contrast=="end":
                # root based on GT
                root=df_syl.cmu_phones.tolist()[:-n_ter_basis]
                # detected termination
                ter=syl_detected[len(root)-1:]
            elif contrast=="start":
                # root based on GT
                # root=df_syl.cmu_phones.tolist()[n_ter_basis:]
                # detected termination
                ter=syl_detected[:n_ter_basis+1]
        
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
        # g_d=drop_consecutive_duplicate_elements([cmu_to_gibberish[unstress(p)] for p in root+ter_post])
        
        # convert to gibberish, but translate UNK token to 'uh', the schwa because we don't know what it is
        g_d=[cmu_to_gibberish[unstress(p)] if not 'UNK' in p else 'uh' for p in ter_post]
        
        # phonetic detection needs to be the stressed version for backwards compatibility 
        # (however, here I convert to stressed version only when correct, it might work, but could cause problems?)
        if contrast=="end":
            detection=ter_post[1:]
        elif contrast=="start":
            detection=ter_post[:-1]


        detection='_'.join(detection)

        # detection=ter_post
        if target_phones!='': # don't try to split in case of case of final -s "nothing" target
            # if dectection is the same as target_phones (without the stress marks because charsiu don't put that), change back to target phones
            if detection.split('_')==remove_stress_annots(target_phones.split('_')): detection=target_phones
        
            # post-correction for the whole termination not to differentiate between IH_D and AH_D     or    IH_Z and AH_Z
            unstressed_target='_'.join(remove_stress_annots(target_phones.split('_')))
            if unstressed_target in terminations_accepted_alternatives:
                if detection in terminations_accepted_alternatives[unstressed_target]:
                    detection=target_phones

        return {"status": "success", "phonetic_detection": detection, "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status, "phonetic_detection": "null", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

def phonetic_content_analysis(s, phonetics):
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
        for i in syl_df.pred_syls_indxs_inside_GT_syl.unique():
            s=syl_df.loc[syl_df.pred_syls_indxs_inside_GT_syl==i]
            v=s[s.pred_phones_audio.isin(cmu_vowels)]
            if len(v)>0:
                m=v.n_frames.idxmax()
                l=[el for el in v.index.tolist() if el != m]
                syl_df=syl_df.drop(l)
        
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
        if r.pred_phones in target_accepted_alternatives:
            if r.pred_phones_audio in target_accepted_alternatives[r.pred_phones]:
                phonetic_content.loc[i, 'pred_phones_audio']=r.pred_phones

    
    # phonetic_content=phonetic_content[phonetic_content.n_frames>1]
    phonetic_content=phonetic_content[phonetic_content.pred_phones_audio!='[SIL]']

    # phonetic_content.loc[phonetic_content.pred_phones_audio=='[SIL]','pred_phones_audio']=''

    phonetic_content=phonetic_content.loc[drop_consecutive_duplicates(phonetic_content[['pred_phones','pred_phones_audio']]).index,:]
    return phonetic_content

def syllable_contrast_from_formatted_phonetics_audio(audio,phonetics='T_ER1_N_D ER0|AW1_N_D', 
                            target_word_idx=0, 
                            target_syllable_idx=0,
                            max_speech_rate=8, mode='file'
                    ):
    phonetics=phonetics.replace('-',' ').replace('{','').replace('}','')
    status, s = audio_load_and_check(audio, phonetics, max_speech_rate=max_speech_rate, mode=mode)
    g_t=[cmu_to_gibberish[unstress(p)] for p in split_phonetics(phonetics)[target_word_idx][target_syllable_idx]]

    if status=="success":
        phonetic_content=phonetic_content_analysis(s, phonetics)
        if model.status!="success": return {"status": model.status, "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

        if len(phonetic_content)==0: return {"status": "phonetic_content is empty", "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}

        syllable_content=phonetic_content[phonetic_content.word_idx==target_word_idx][phonetic_content.syl_idx==target_syllable_idx]
        detected_syllable=syllable_content.pred_phones_audio.tolist()
        g_d=drop_consecutive_duplicate_elements([cmu_to_gibberish[unstress(p)] for p in detected_syllable])

        return {"status": "success", "gibberish_truth": '_'.join(g_t), "gibberish_detected": '_'.join(g_d)}
    else:
        return {"status": status,  "gibberish_truth":  '_'.join(g_t), "gibberish_detected":  "null"}
    

if __name__=="__main__":
    from DL_speech_tech import *

    from utils.audio_processing import prepare_audio_file
    from utils.text_processing import *
    from utils.label_data_processing import *

    path='scripts/synth_audio/cmu_words/standard/prosody/Brian/M_UK_ekk.mp3'
    # encode_string = base64.b64encode(open(path, "rb").read())
    formatted_phonetics=prefill_for_sentence('ekk')['cmu_phonetics']
    _, rID=prepare_audio_file(path)
    phonemeContrast_from_formatted_phonetics_audio(rID,phonetics=formatted_phonetics, 
                                                        target_word_idx=0, 
                                                        target_syllable_idx=1, 
                                                        target_phones='EY1',
                                                        alternatives=cmu_vowels, mode='file')

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