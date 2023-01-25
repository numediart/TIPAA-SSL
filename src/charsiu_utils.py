import librosa
import pandas as pd
import numpy as np

from charsiu.src.Charsiu import charsiu_forced_aligner
import torch
import numpy as np
from charsiu.src.utils import seq2duration,forced_align
from src.text_processing import group_consecutive_duplicates, remove_stress_annots, phonetics_indexed_df_from_formatted_phonetics, unstress, drop_consecutive_duplicate_elements, drop_consecutive_duplicates
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants
from src.audio_processing import getIntonation, getIntensity, normalize

# https://stackoverflow.com/questions/51269456/pandas-delete-consecutive-duplicates-but-keep-the-first-and-last-value
keep_first_last=lambda s: s[~((s == s.shift(1)) & (s == s.shift(-1)))]

# processing functions of df_segmented, which is the output of prediction and forced alignment

def extract_word(df_segmented, phonetics, target_word_idx):
    """phonetics must be a list of list of phonemes, e.g.: phonetics=[['AY1'],['EH1', 'N', 'D', 'IH0', 'D']]
    """
    # merge lists
    phones=sum(phonetics,[])

    df_segmented=df_segmented[df_segmented.phones != '[SIL]']
    # if phones contains twice the same phone, e.g. "PhiliP Paints well", both P will be collapsed when computing the timings from DTW.
    # this results in a df_segmented shorter than "phones" list. I thus have to duplicate the corresponding row when it happens
    if len(phones)>len(df_segmented):
        # here make sure the index is a range. I will insert using .loc at i+0.5, then reset index every time
        # https://stackoverflow.com/questions/15888648/is-it-possible-to-insert-a-row-at-an-arbitrary-position-in-a-dataframe-using-pan?rq=1
        df_segmented=df_segmented.reset_index(drop=True)
        for i in range(len(phones)-1):
            if phones[i]==phones[i+1]:
                df_segmented.loc[i+0.5]=df_segmented.loc[i]
                df_segmented=df_segmented.reset_index(drop=True)
    
    start_idx=sum([len(p) for p in phonetics][:target_word_idx])
    end_idx=sum([len(p) for p in phonetics][:target_word_idx+1])
    df_word=df_segmented[start_idx:end_idx]
    return df_word

# get the blocks of consecutive identical rows in cols
get_blocks = lambda a,cols: a.loc[(a[cols].shift() == a[cols]).any(axis=1)|(a[cols].shift(-1) == a[cols]).any(axis=1)]
class charsiu_phone_forced_aligner(charsiu_forced_aligner):
    def __init__(self, aligner, sil_threshold=4, **kwargs):
        super().__init__(aligner, sil_threshold, **kwargs)
        # this is a state variable that impact the worflow. I check for success after calling align_phone, if it's not, I return None an this status variable will also be checked in DL_speech_tech
        self.status="success"
        self.phonetic_content=None
        self.pred_phones_audio=""

        self.p_to_id=self.charsiu_processor.processor.tokenizer.encoder
        self.id_to_p=self.charsiu_processor.processor.tokenizer.decoder
    
    def predict_prob_matrix_and_phones(self, audio):
        
        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():  out = self.aligner(audio)
        cost = torch.softmax(out.logits,dim=-1).detach().cpu().numpy().squeeze()

        
        pred_ids_audio = torch.argmax(out.logits.squeeze(),dim=-1)
        pred_ids_audio = pred_ids_audio.detach().cpu().numpy()
        pred_phones_audio = [self.charsiu_processor.mapping_id2phone(int(i)) for i in pred_ids_audio]

        return cost, pred_phones_audio

    def align_phones(self, audio, phones, GT_alignment_proba_threshold=0.17):
        '''
        Perform forced alignment

        Parameters
        ----------
        audio : np.ndarray [shape=(n,)]
            time series of speech signal
        phones : phones must be a list of phonemes, e.g.: phones=['EH1', 'N', 'D', 'IH0', 'D']

        Returns
        -------
        A tuple of aligned phones in the form (start_time, end_time, phone)

        '''
        phones=[[p] for p in  remove_stress_annots(phones)]
        phone_ids = self.charsiu_processor.get_phone_ids(phones)

        cost, pred_phones_audio=self.predict_prob_matrix_and_phones(audio)


        pred_probas=np.max(cost, axis=1)

        sil_mask = self._get_sil_mask(cost)
        nonsil_idx = np.argwhere(sil_mask!=self.charsiu_processor.sil_idx).squeeze()

        if len(nonsil_idx)>0:
            try:
                aligned_phone_ids = forced_align(cost[nonsil_idx,:],phone_ids[1:-1])
            except:
                self.status="success: the phrase was not recognized in expected phonemes"
                return None, None, None
            aligned_phones = [self.charsiu_processor.mapping_id2phone(phone_ids[1:-1][i]) for i in aligned_phone_ids]
            pred_phones = self._merge_silence(aligned_phones,sil_mask)
            alignment_phones = seq2duration(pred_phones,resolution=self.resolution)

            # for each alignment, go inside the corresponding frames of per frame predictions from audio and 
            # do an average over time of the probabilities
            timestep=self.resolution
            # most_pred_phones_audio=[]
            max_proba_mean_phones_audio=[]
            proba_means=[]
            for a in alignment_phones:
                start=a[0]
                end=a[1]
                select_cost=cost[round(start/timestep):round(end/timestep)]
                proba_mean=select_cost.mean(axis=0)
                proba_means.append(proba_mean)
        else:
            sil='[SIL]'
            alignment_phones=[(0, len(pred_phones_audio)*self.resolution, sil)]
            # most_pred_phones_audio=[sil]
            # max_proba_mean_phones_audio=[sil]
            sil_vec=np.zeros(cost.shape[-1])
            sil_vec[0]=1
            proba_means=[sil_vec]
            pred_phones=[sil]*len(pred_phones_audio)

        proba_means_matrix=np.array(proba_means)
        idx_mean_maxs=np.argmax(proba_means_matrix, axis=1)
        max_proba_mean_phones_audio=[self.charsiu_processor.mapping_id2phone(int(i)) for i in idx_mean_maxs]        
        
        df_segmented=pd.DataFrame(alignment_phones)
        df_segmented.columns=['start','end','phones']
        df_segmented.loc[:,'pred_phones_audio']=max_proba_mean_phones_audio
        df_segmented.loc[:,'proba_means']=proba_means
        p_to_id=lambda p: self.charsiu_processor.mapping_phone2id(p)
        
        df_segmented['GT_proba']=df_segmented.apply(lambda r: r.proba_means[p_to_id(r.phones)], axis=1)
        df_segmented['pred_proba']=df_segmented.apply(lambda r: r.proba_means[p_to_id(r.pred_phones_audio)], axis=1)
        
        def divide_consecutive_duplicates(p_df, phone_list):
            grouped_phone_list=group_consecutive_duplicates(phone_list)
            n_times=[el[-1] for i,el in enumerate(grouped_phone_list)]

            p_df['n_times']=n_times
            p_df_full=p_df.loc[p_df.index.repeat(p_df.n_times)]
            p_df_full=p_df_full.reset_index()

            blocks=get_blocks(p_df_full, ['phones'])
            
            block_starts_ends=keep_first_last(blocks.phones)
            # keep firsts and lasts (thus only when there is two consecutive phonemes)
            starts=block_starts_ends.loc[block_starts_ends.shift(-1) == block_starts_ends].index.tolist()
            ends=block_starts_ends.loc[block_starts_ends.shift(+1) == block_starts_ends].index.tolist()

            for start_idx,end_idx in zip(starts,ends):
                select=p_df_full.iloc[start_idx:end_idx+1]
                start=select.start.iloc[0]
                end=select.end.iloc[-1]
                interval=(end-start)/len(select)

                steps=start+np.cumsum([interval]*(len(select)-1))

                p_df_full.iloc[start_idx+1:end_idx+1].start=steps
                p_df_full.iloc[start_idx:end_idx].end=steps
            p_df_full[['start','end']]=p_df_full[['start','end']].round(2)
            return p_df_full

        
        def collapse_consecutive_duplicates(df):
            df=df.reset_index(drop=True)
            blocks=[]

            groups=df.groupby([(df.phones != df.phones.shift()).cumsum()])
            for i, g in groups:
                #print('---');      print (g);         print (g.phones.tolist());r=g.iloc[0];   r.end=g.iloc[-1].end;   
                r=g.iloc[0]
                r.end=g.iloc[-1].end
                blocks.append(r.to_dict())
            return pd.DataFrame.from_records(blocks)
        
        
        # drop silence, collapse consecutive duplicates (some are superfluous, 
        # e.g. phonemes interrupted by a silence), 
        # then divide interval for consecutive duplicate phonemes in the ground truth            
        df_segmented2=df_segmented[df_segmented.phones != '[SIL]']
        # collapse_consecutive_duplicates(df_segmented)
        try:
            df_segmented2=collapse_consecutive_duplicates(df_segmented2)
        except: 
            # import pdb;pdb.set_trace()
            self.status= "error: error in align_phones(), when collapsing consecutive duplicates"
            return None, None, None

        try:
            phone_list=sum(phones,[])
            if len(df_segmented2)>0:
                df_segmented=divide_consecutive_duplicates(df_segmented2, phone_list)
        except: 
            # import pdb;pdb.set_trace()
            self.status= "error: error in align_phones(), when dividing collapsed consecutive duplicates"
            return None, None, None
        
        if df_segmented.phones.tolist()!=df_segmented.pred_phones_audio.tolist():
            if df_segmented[df_segmented.phones!='[SIL]'].GT_proba.mean() > GT_alignment_proba_threshold:
                self.status="success"
            else:
                self.status="success: the phrase was not recognized in expected phonemes"
                return None, None, None

        df=pd.DataFrame()
        df.loc[:,'pred_phones']=pred_phones
        df.loc[:,'pred_phones_audio']=pred_phones_audio
        
        GT_idxs=self.charsiu_processor.get_phone_ids([[el] for el in df.pred_phones.tolist()])[1:-1]
        GT_probas=np.array([cost[i,idx] for i,idx in enumerate(GT_idxs)])

        detailed_alignment_phones=drop_consecutive_duplicates(df)

        # to get the number of frames of each consecutive combination of pred_phones_audio and pred_phones
        # I look at differences of indices. For the last one, we need to make diff with len(df)
        n_frames=np.diff(detailed_alignment_phones.index).tolist()
        n_frames.append(len(df)-detailed_alignment_phones.index[-1])
        detailed_alignment_phones.loc[:,'n_frames']=n_frames

        # Compute average probas for each row
        idxs_for_ranges=detailed_alignment_phones.index.tolist()+[len(df)]
        pred_proba_means=[]
        GT_proba_means=[]
        for i in range(len(detailed_alignment_phones)):
            pred_proba_means.append(pred_probas[idxs_for_ranges[i]:idxs_for_ranges[i+1]].mean())
            GT_proba_means.append(GT_probas[idxs_for_ranges[i]:idxs_for_ranges[i+1]].mean())
        detailed_alignment_phones.loc[:,'pred_proba_means']=pred_proba_means
        detailed_alignment_phones.loc[:,'GT_proba_means']=GT_proba_means

        
        self.phonetic_content=detailed_alignment_phones
        self.pred_phones_audio=drop_consecutive_duplicate_elements(detailed_alignment_phones[detailed_alignment_phones.pred_phones_audio!='[SIL]'].pred_phones_audio.tolist())

        return alignment_phones, df_segmented, detailed_alignment_phones
        
    def analyze_phonetic_content(self, audio, phonetics):
        """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'
        """
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        phones=sum(split_phonetics,[])
        # seq_p=[[p] for p in  remove_stress_annots(phones)]
        alignment_phones, pred_phones_audio, detailed_alignment_phones = self.align_phones(audio=audio,phones=phones)
        if self.status!="success": return None

        detailed_alignment_phones=detailed_alignment_phones[detailed_alignment_phones.pred_phones != '[SIL]']

        if len(detailed_alignment_phones)==0: return detailed_alignment_phones
        
        # if I filter out silences contained in pred_phones_audio, it can sometimes remove phones from pred_phones, which is problematic for 
        # the following alignment
        #[detailed_alignment_phones.pred_phones_audio != '[SIL]']

        phonetics_indexed_df=phonetics_indexed_df_from_formatted_phonetics(phonetics)

        # here we align phonetics_indexed_df to the detailed_alignment_phones to be able to get an indexation on the "really pronounced phonetics"
        # from part of audio that corresponded to specific phones in ground truth (according to forced-alignment)
        orig_phones=remove_stress_annots(phones)
        pred_phones=detailed_alignment_phones.pred_phones.tolist()
        assert orig_phones[0] == pred_phones[0], "The first phone of alignment pred and ground truth should be the same"
        indx_in_phones=0
        pred_phones_original_indices=[]
        for i,p in enumerate(pred_phones):
            if p == orig_phones[indx_in_phones]:
                pred_phones_original_indices.append(indx_in_phones)
            else:
                indx_in_phones+=1
                # Given it was not equal to the previous element, after going to the next element of ground truth, it should be the same"
                # except if there was twice the same phoneme (because it was the end of last word and start of current word)
                if p == orig_phones[indx_in_phones]:
                    pred_phones_original_indices.append(indx_in_phones)
                else:
                    assert orig_phones[indx_in_phones]==orig_phones[indx_in_phones-1], "This should correspond to the case of two consecutive identical phonemes, because they are in two consecutive words"
                    indx_in_phones+=1
                    assert p == orig_phones[indx_in_phones], "This should correspond to the case of two consecutive identical phonemes, because they are in two consecutive words"
                    pred_phones_original_indices.append(indx_in_phones)

        # detailed_alignment_phones.loc[:,'p_idx']=phonetics_indexed_df.loc[pred_phones_original_indices,'p_idx'].tolist()
        detailed_alignment_phones.loc[:,'word_idx']=phonetics_indexed_df.loc[pred_phones_original_indices,'word_idx'].tolist()
        detailed_alignment_phones.loc[:,'syl_idx']=phonetics_indexed_df.loc[pred_phones_original_indices,'syl_idx'].tolist()
        # detailed_alignment_phones=detailed_alignment_phones[['word_idx','syl_idx','p_idx','pred_phones', 'GT_proba_means', 'pred_phones_audio', 'pred_proba_means', 'n_frames']]
        detailed_alignment_phones=detailed_alignment_phones[['word_idx','syl_idx','pred_phones', 'GT_proba_means', 'pred_phones_audio', 'pred_proba_means', 'n_frames']]

        return detailed_alignment_phones

    def predict_word(self, audio, phonetics, target_word_idx):
        """phonetics must be a list of list of phonemes, e.g.: phonetics=[['AY1'],['EH1', 'N', 'D', 'IH0', 'D']]
        """
        # merge lists
        phones=sum(phonetics,[])
        alignment_phones, df_segmented, phonetic_content = self.align_phones(audio=audio,phones=phones)
        
        if self.status!="success": return None

        df_word=extract_word(df_segmented, phonetics, target_word_idx)
        return df_word
    
    def predict_phone(self, audio, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx=0, phoneme_set=cmu_vowels, GT_proba_threshold=0.2):
        """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'
        """
        
        phoneme_set=[[p] for p in  remove_stress_annots(phoneme_set)]
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        df_word=self.predict_word(audio, split_phonetics, target_word_idx)
        
        if self.status!="success": return None, None

        if len(df_word)>0:
            word=phonetics.split(' ')[target_word_idx]
            syllables=[syl.split('_') for syl in word.split('|')]
            syllable=syllables[target_syllable_idx]
            syl=remove_stress_annots(syllable)

            idxs_of_target_occurences=[i for i,p in enumerate(syl) if unstress(target_phones) ==p]

            # find the phoneme index:
            # p_idx_local=syl.index(unstress(target_phones))

            if target_occurence_idx<len(idxs_of_target_occurences):
                p_idx_local=idxs_of_target_occurences[target_occurence_idx]
            else:
                self.status="error: target_occurence_idx is out of bounds"
                phonetic_detection=float('nan')
                syl=float('nan')
                return phonetic_detection, syl

            len_previous_syllables=sum([len(el) for el in syllables[:target_syllable_idx]])
            p_idx_global=len_previous_syllables+p_idx_local

            phoneme_set_ids=self.charsiu_processor.get_phone_ids(phoneme_set)[1:-1]
            proba_means=df_word.iloc[p_idx_global].proba_means

            # if GT_proba is beyond the threshold, we take it as prediction
            if df_word.iloc[p_idx_global].GT_proba>GT_proba_threshold:
                phonetic_detection=target_phones
            else:
                # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
                filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
                idx_mean_max=np.argmax(filtered_proba_means)
                phonetic_detection=self.charsiu_processor.mapping_id2phone(int(idx_mean_max))
            syl[p_idx_local]=phonetic_detection
            
        else:
            phonetic_detection=float('nan')
            syl=float('nan')
        return phonetic_detection, syl

    def compute_stress_score(self, audio, phonetics):
        """Use df_segmented to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
        a value by vowel representing a stress intensity

        Args:
            phonetics (str): formatted phonetics
            audio (np array): audio signal
        Returns:
            weighted_score [type]: stress intensity score
        """
        
        # phonetics=sum(phonetics,[])
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        split_phonetics=sum(split_phonetics,[])
        _, df_segmented, _ = self.align_phones(audio=audio,phones=split_phonetics)
        if self.status!="success": return None

        # select vowels
        filtered_df=df_segmented[df_segmented.phones.isin(cmu_vowels)]#.index.tolist()

        f0Samples=getIntonation(audio, self.sr)
        intensity=getIntensity(audio, self.sr)

        # extract features
        # each word start and end position expressed in samples
        startPositions_samples = (round(self.sr*filtered_df.loc[:,'start'])+1).astype(int).tolist()
        stopPositions_samples = round(self.sr*filtered_df.loc[:,'end']).astype(int).tolist()

        # to make sure we don t go beyond the end of the signal
        assert stopPositions_samples[-1]<len(audio), "The end of the last phoneme should be inside the signal"

        Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
        # nVowels=len(indxVowels)
        # sylType=np.zeros(len(filtered_df))
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




if __name__=="__main__":
    from src.charsiu_utils import *
    from transformers import Wav2Vec2Processor
    from src.libri_phonetization_data import libri_phonetics_data
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # from src.layer_extraction import get_last_hidden_state, get_logits, inference
    # from scripts.wav2vec2_utils import instances_per_phoneme, plot_reduction, phone_average_vectors
    
    # initialize model
    charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    df_t, df=libri_phonetics_data(data_set='dev-clean')
    df_phones=df_t.apply(lambda r: pd.DataFrame.from_records(r.phone_df).cmu_phone.tolist(), axis=1)

    # actor recordings
    df=pd.read_csv('data/exercise_data_export.csv')
    df.loc[:,'audio_file_url']='data/scaleway-audio-files/'+df['audio_file_url']
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    target_phones="T"
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]
    example=selection.iloc[10]
    # example=selection.loc[1131]
    # "ended"
    example=df.iloc[968]
    phonetics=example.cmu_phonetics

    # example=df.iloc[1539]
    # phonetics='AY1 K_AE1_N_T W_EY1|T_IH0_D F_AO1_R AW1_R S_AH1|M_ER0 R_OW1_D|T_R_IH2_P'

    path=example.audio_file_url
    s,fs=librosa.load(path, sr=16000)
    split_phonetics=sum([p.replace('|','_').split('_') for p in phonetics.split(' ')],[])
    # phones=[[p] for p in  remove_stress_annots(split_phonetics)]
    _, df_segmented, phonetic_content = charsiu.align_phones(audio=s,phones=split_phonetics)
    df_segmented[df_segmented.phones!='[SIL]'].GT_proba.median()
    df_segmented[df_segmented.phones!='[SIL]'].GT_proba.mean()

    df_word=charsiu.predict_word(s, [split_phonetics], 0) 
    p_idx_global=-1
    GT_proba_threshold=0.2
    phoneme_set=cmu_consonants
    
    phoneme_set=[[p] for p in  remove_stress_annots(phoneme_set)]
    proba_means=df_word.iloc[p_idx_global].proba_means

    phoneme_set_ids=charsiu.charsiu_processor.get_phone_ids(phoneme_set)[1:-1]
    # if GT_proba is beyond the threshold, we take it as prediction
    if df_word.iloc[p_idx_global].GT_proba>GT_proba_threshold:
        phonetic_detection=target_phones
    else:
        # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
        filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
        idx_mean_max=np.argmax(filtered_proba_means)
        phonetic_detection=charsiu.charsiu_processor.mapping_id2phone(int(idx_mean_max))



    phonetic_content.GT_proba_means.median()
    phonetic_content.GT_proba_means.mean()

    df_segmented.apply(lambda r: r.proba_means.sum(), axis=1)

    

    # charsiu.predict_phone(audio=s,phones=split_phonetics)
    charsiu.predict_phone(s, phonetics, 0, 1, 'D', target_occurence_idx=0, phoneme_set=cmu_vowels)
    charsiu.predict_phone(s, phonetics, 0, 1, 'D', target_occurence_idx=0, phoneme_set=cmu_consonants)
    charsiu.predict_phone(s, phonetics, 0, 1, 'D', target_occurence_idx=1, phoneme_set=cmu_consonants)

    # df_segmented=charsiu.force_and_predict(s,split_phonetics)
    from DL_speech_tech import phonetic_content_analysis
    phonetic_content=phonetic_content_analysis(s,phonetics)
    

    charsiu.predict_phone(s, phonetics, 0, 1, 'D')

    #  Inference with the example number N
    N=0
    example=df.iloc[N]
    path=example.wav_path
    print(example.phones)
    example['phones']=df_phones.iloc[N]
    # Loas an audio file
    s,fs=librosa.load(path, sr=16000)

    alignment = charsiu.align(audio=path,text=example.text)
    print(alignment)
    # _, df_segmented, _ = charsiu.align_phones(audio=s,phones=phones)
    # df_segmented=charsiu.force_and_predict(s,example.phones)
    df_segmented=df_segmented[df_segmented.phones != '[SIL]']

    # charsiu.compute_stress_score

    df_segmented[df_segmented.phones!=df_segmented.pred_phones_audio]

    # ws=compute_stress_score(df_segmented, s, fs)

    charsiu.compute_stress_score(s,example.phones)

    path_to_json='/data/audio-with-analysis-ids/data.json'
    d=pd.read_json(path_to_json)
    