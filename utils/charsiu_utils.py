import os
import librosa
import pandas as pd
import numpy as np

from charsiu.src.Charsiu import charsiu_forced_aligner, charsiu_attention_aligner, charsiu_predictive_aligner

import sys
import torch
import numpy as np
from charsiu.src.utils import seq2duration,forced_align
from utils.text_processing import remove_stress_annots, phonetics_indexed_df_from_formatted_phonetics, cmu_vowels, cmu_consonants, unstress
from utils.audio_processing import getIntonation, getIntensity, normalize
from collections import Counter
from utils.text_processing import cmu_vowels

drop_consecutive_duplicates= lambda df: df.loc[(df.shift()!=df).sum(axis=1).astype(bool)]


class charsiu_phone_forced_aligner(charsiu_forced_aligner):
    def __init__(self, aligner, sil_threshold=4, **kwargs):
        super().__init__(aligner, sil_threshold, **kwargs)
        self.status="success"
        self.phonetic_content=None
        self.pred_phones_audio=""
    def align_phones(self, audio, phones):
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

        audio = self.charsiu_processor.audio_preprocess(audio,sr=self.sr)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)
        # phones, words = self.charsiu_processor.get_phones_and_words(text)
        phone_ids = self.charsiu_processor.get_phone_ids(phones)

        with torch.no_grad():  out = self.aligner(audio)
        
        # hidden_states= out.hidden_states
        # last_hidden_state=hidden_states[-1]
        # logits=out.logits

        cost = torch.softmax(out.logits,dim=-1).detach().cpu().numpy().squeeze()

        pred_probas=np.max(cost, axis=1)

        pred_ids_audio = torch.argmax(out.logits.squeeze(),dim=-1)
        pred_ids_audio = pred_ids_audio.detach().cpu().numpy()
        pred_phones_audio = [self.charsiu_processor.mapping_id2phone(int(i)) for i in pred_ids_audio]
        # pred_phones = seq2duration(pred_phones,resolution=self.resolution)
        
        sil_mask = self._get_sil_mask(cost)
        nonsil_idx = np.argwhere(sil_mask!=self.charsiu_processor.sil_idx).squeeze()

        if len(nonsil_idx)>0:
            aligned_phone_ids = forced_align(cost[nonsil_idx,:],phone_ids[1:-1])
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
            alignment_phones=[(0, len(pred_ids_audio)*self.resolution, sil)]
            # most_pred_phones_audio=[sil]
            # max_proba_mean_phones_audio=[sil]
            sil_vec=np.zeros(cost.shape[-1])
            sil_vec[0]=1
            proba_means=[sil_vec]
            pred_phones=[sil]*len(pred_ids_audio)

        proba_means_matrix=np.array(proba_means)
        idx_mean_maxs=np.argmax(proba_means_matrix, axis=1)
        max_proba_mean_phones_audio=[self.charsiu_processor.mapping_id2phone(int(i)) for i in idx_mean_maxs]        
        
        df_segmented=pd.DataFrame(alignment_phones)
        df_segmented.columns=['start','end','cmu_phones']
        df_segmented.loc[:,'pred_phones_audio']=max_proba_mean_phones_audio
        df_segmented.loc[:,'proba_means']=proba_means
        p_to_id=lambda p: self.charsiu_processor.mapping_phone2id(p)
        
        df_segmented['GT_proba']=df_segmented.apply(lambda r: r.proba_means[p_to_id(r.cmu_phones)], axis=1)
        df_segmented['pred_proba']=df_segmented.apply(lambda r: r.proba_means[p_to_id(r.pred_phones_audio)], axis=1)
        
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

        if df_segmented[df_segmented.cmu_phones!='[SIL]'].GT_proba.mean() > 0.2:
            self.status="success"
        else:
            self.status="success: the phrase was not recognized in expected phonemes"
        
        self.phonetic_content=detailed_alignment_phones
        
        self.pred_phones_audio=drop_consecutive_duplicate_elements(phonetic_content[phonetic_content.pred_phones_audio!='[SIL]'].pred_phones_audio.tolist())

        return alignment_phones, df_segmented, detailed_alignment_phones
    
    def predict_word(self, audio, phonetics, target_word_idx):
        """phonetics must be a list of list of phonemes, e.g.: phonetics=[['AY1'],['EH1', 'N', 'D', 'IH0', 'D']]
        """
        # merge lists
        phones=sum(phonetics,[])
        alignment_phones, df_segmented, phonetic_content = self.align_phones(audio=audio,phones=phones)
        df_segmented=df_segmented[df_segmented.cmu_phones != '[SIL]']
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
    
    def predict_phone(self, audio, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx=0, phoneme_set=cmu_vowels, GT_proba_threshold=0.5):
        """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'
        """
        
        phoneme_set=[[p] for p in  remove_stress_annots(phoneme_set)]
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        df_word=self.predict_word(audio, split_phonetics, target_word_idx)

        if len(df_word)>0:
            word=phonetics.split(' ')[target_word_idx]
            syllables=[syl.split('_') for syl in word.split('|')]
            syllable=syllables[target_syllable_idx]
            syl=remove_stress_annots(syllable)

            idxs_of_target_occurences=[i for i,p in enumerate(syl) if target_phones ==p]

            # find the phoneme index:
            p_idx_local=syl.index(unstress(target_phones))
            # p_idx_local=idxs_of_target_occurences[target_occurence_idx]

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

    def analyze_phonetic_content(self, audio, phonetics):
        """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'
        """
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        phones=sum(split_phonetics,[])
        # seq_p=[[p] for p in  remove_stress_annots(phones)]
        alignment_phones, pred_phones_audio, detailed_alignment_phones = self.align_phones(audio=audio,phones=phones)

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
                    assert orig_phones[indx_in_phones]==orig_phones[indx_in_phones-1], "This should correspond to the case of two consecutiva identical phonemes, because they are in two consecutive words"
                    indx_in_phones+=1
                    assert p == orig_phones[indx_in_phones], "This should correspond to the case of two consecutiva identical phonemes, because they are in two consecutive words"
                    pred_phones_original_indices.append(indx_in_phones)

        # detailed_alignment_phones.loc[:,'p_idx']=phonetics_indexed_df.loc[pred_phones_original_indices,'p_idx'].tolist()
        detailed_alignment_phones.loc[:,'word_idx']=phonetics_indexed_df.loc[pred_phones_original_indices,'word_idx'].tolist()
        detailed_alignment_phones.loc[:,'syl_idx']=phonetics_indexed_df.loc[pred_phones_original_indices,'syl_idx'].tolist()
        # detailed_alignment_phones=detailed_alignment_phones[['word_idx','syl_idx','p_idx','pred_phones', 'GT_proba_means', 'pred_phones_audio', 'pred_proba_means', 'n_frames']]
        detailed_alignment_phones=detailed_alignment_phones[['word_idx','syl_idx','pred_phones', 'GT_proba_means', 'pred_phones_audio', 'pred_proba_means', 'n_frames']]

        return detailed_alignment_phones

    def compute_stress_score(self, audio, phonetics):
        """Use textgridData to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
        a value by vowel representing a stress intensity

        Args:
            textgridData ([type]): [description]
            s (np array): audio signal
            fs (int): frequency of sampling
        Returns:
            weighted_score [type]: stress intensity score
        """
        
        # phonetics=sum(phonetics,[])
        split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
        split_phonetics=sum(split_phonetics,[])
        _, textgridData, _ = self.align_phones(audio=audio,phones=split_phonetics)
        # textgridData=self.force_and_predict(audio,split_phonetics)
        textgridData=textgridData[textgridData.cmu_phones != '[SIL]']

        # I have to collapse if several consecutive vowels are the same. it can happen when the predictions are not the same.
        # I thus have to group the timings (first start until last end)

        #  here we delete consecutives but keep first and last, so there is a possibility of only two consecutves after that, and having overall start and end
        # https://stackoverflow.com/questions/51269456/pandas-delete-consecutive-duplicates-but-keep-the-first-and-last-value
        keep_first_last=lambda s: s[~((s == s.shift(1)) & (s == s.shift(-1)))]

        test=keep_first_last(textgridData.cmu_phones)

        # keep firsts and lasts (thus only when there is two consecutive phonemes)
        starts=test.loc[test.shift(-1) == test]
        ends=test.loc[test.shift(+1) == test]

        assert len(starts)==len(ends)

        # drop duplicates keeping first
        drop_duplicates=lambda a: a.loc[a.shift(+1) != a]
        filtered_df=textgridData.loc[drop_duplicates(textgridData.cmu_phones).index]

        # here in the filtered_df containg only the first occurence for equal consecutive examples, we replace the 'end' value with the line in the "ends"
        for rownum,(indx,val) in enumerate(starts.iteritems()): filtered_df.loc[indx,'end']=textgridData.loc[ends.index[rownum],'end']
        filtered_df=filtered_df[filtered_df.cmu_phones.isin(cmu_vowels)]#.index.tolist()

        f0Samples=getIntonation(audio, self.sr)
        intensity=getIntensity(audio, self.sr)

        # extract features
        # each word start and end position expressed in samples
        startPositions_samples = (round(self.sr*filtered_df.iloc[:,0])+1).astype(int).tolist()
        stopPositions_samples = round(self.sr*filtered_df.iloc[:,1]).astype(int).tolist()

        # to make sure we don t go beyond the end of the signal
        assert stopPositions_samples[-1]<len(audio), "The end of the last phoneme should be inside the signal"

        Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
        # nVowels=len(indxVowels)
        sylType=np.zeros(len(filtered_df))
        for i in range(len(filtered_df)):
            range_vowel=range(startPositions_samples[i], stopPositions_samples[i])
            Ivowel=intensity[range_vowel]
            Fvowel=f0Samples[range_vowel]
            
            Imax.append(max(Ivowel))
            Imean.append(np.mean(Ivowel))
            Fmax.append(max(Fvowel))
            Fmean.append(np.mean(Fvowel))

            Dur.append(filtered_df['end'].iloc[i]-filtered_df['start'].iloc[i])
            
            # phone_df=pd.DataFrame([r[2].split('_') for i,r in filtered_df.iterrows()])
            # # here we use the prediction of HMM model as an indication, as it has to classify 0, 1 or 2
            # syltype_phone=int(phone_df[2].iloc[i])
            # if syltype_phone == 2:  # the sylType is 0 for unstressed, 0.5 for secondary stressed syllables and 1 for primary stressed syllables
            #     sylType[i] = 0.5
            # else:
            #     sylType[i]=syltype_phone
        
        # normalization of features (projection to [0 1] range)
        zImax = normalize(Imax)
        zImean = normalize(Imean)
        zFmax = normalize(Fmax)
        zFmean = normalize(Fmean)
        zDur = normalize(Dur)

        # combine the features
        # weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur + 0.4*sylType)/3.6  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data
        weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur)/3.2  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data

        return weighted_score




if __name__=="__main__":
    from utils.charsiu_utils import *
    from transformers import Wav2Vec2Processor
    from utils.libri_phonetization_data import libri_phonetics_data
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # from src.layer_extraction import get_last_hidden_state, get_logits, inference
    # from scripts.wav2vec2_espeak import instances_per_phoneme, plot_reduction, phone_average_vectors
    
    # initialize model
    charsiu = charsiu_phone_forced_aligner(aligner='hf_models/charsiu/en_w2v2_fc_10ms', device='cpu')
    df_t, df=libri_phonetics_data(data_set='dev-clean')
    df_cmu_phones=df_t.apply(lambda r: pd.DataFrame.from_records(r.phone_df).cmu_phone.tolist(), axis=1)

    # actor recordings
    df=pd.read_csv('data/exercise_data_export.csv')
    df.loc[:,'audio_file_url']='data/scaleway-audio-files/'+df['audio_file_url']
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    target_phones="IH0_D"
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]
    # example=selection.iloc[10]
    example=selection.loc[1131]
    # "ended"
    example=df.iloc[968]
    phonetics=example.cmu_phonetics

    example=df.iloc[1539]
    phonetics='AY1 K_AE1_N_T W_EY1|T_IH0_D F_AO1_R AW1_R S_AH1|M_ER0 R_OW1_D|T_R_IH2_P'

    path=example.audio_file_url
    s,fs=librosa.load(path, sr=16000)
    split_phonetics=sum([p.replace('|','_').split('_') for p in phonetics.split(' ')],[])
    # phones=[[p] for p in  remove_stress_annots(split_phonetics)]
    _, df_segmented, phonetic_content = charsiu.align_phones(audio=s,phones=split_phonetics)
    df_segmented[df_segmented.cmu_phones!='[SIL]'].GT_proba.median()
    df_segmented[df_segmented.cmu_phones!='[SIL]'].GT_proba.mean()

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
    example['cmu_phones']=df_cmu_phones.iloc[N]
    # Loas an audio file
    s,fs=librosa.load(path, sr=16000)

    alignment = charsiu.align(audio=path,text=example.text)
    print(alignment)
    # _, df_segmented, _ = charsiu.align_phones(audio=s,phones=phones)
    # df_segmented=charsiu.force_and_predict(s,example.cmu_phones)
    df_segmented=df_segmented[df_segmented.cmu_phones != '[SIL]']

    # charsiu.compute_stress_score

    df_segmented[df_segmented.cmu_phones!=df_segmented.pred_phones_audio]

    # ws=compute_stress_score(df_segmented, s, fs)

    charsiu.compute_stress_score(s,example.cmu_phones)

    path_to_json='/data/audio-with-analysis-ids/data.json'
    d=pd.read_json(path_to_json)
    



    
    # model=charsiu.__dict__["aligner"]
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    # phone_outputs=phone_average_vectors(s, fs, df_segmented, processor, model, time_per_output=0.01, phone_type='cmu_phones', extractor_function=get_logits)
    # phone_outputs=phone_average_vectors(s, fs, df_segmented, processor, model, time_per_output=0.01, phone_type='cmu_phones', extractor_function=inference)

    # # define vectorized sigmoid
    # sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    # phone_outputs.apply(lambda r: sigmoid(r.average_vector), axis=1)

    # # get_last_hidden_state(s, fs=16000, model=model)
    # # get_hidden_states(s, fs=16000, model=model)
    # # get_logits(s, fs=16000, model=model)

    # df_all_instances=instances_per_phoneme(df_t, number_of_examples=len(df), time_per_output=0.01,  phone_type='cmu_phone', model=model)
    # plot_reduction(df_all_instances, reduction_technique='umap', base_name='wav2vec2_frame_classification' )

    # instances_per_phoneme(df_t, number_of_examples=10, time_per_output=0.01,  phone_type='cmu_phone', model=model)

    # df_all_instances=instances_per_phoneme(df_t, processor, model, number_of_examples=10, time_per_output=0.02,  phone_type='cmu_phone', extractor_function=get_logits)


    # -------------------------

    # intialize model
    # /!\  the get_hiddden_states and logits does not work with the following model. I don't know if it has something to do with attention mechanism
    # charsiu = charsiu_attention_aligner('charsiu/en_w2v2_fs_10ms')
    # alignment = charsiu.align(audio=audio_path,text=text)

    # initialize model
    charsiu_pred = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')
    alignment = charsiu_pred.align(audio=path)