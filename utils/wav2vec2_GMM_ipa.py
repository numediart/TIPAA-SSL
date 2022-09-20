import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import librosa
import json
import ast
# from src.config import DEVICE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
# from utils.phone_processing import remove_stress_annots
from utils.audio_processing import getIntonation, getIntensity, normalize
from utils.text_processing import unstress, prefill_for_sentence, remove_stress_annots
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import cmudict
from operator import itemgetter
import itertools
from utils.libri_phonetization_data import libri_phonetics_data
from jiwer import wer
import seaborn as sns
from sklearn.metrics import confusion_matrix

from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Tokenizer, Wav2Vec2ForCTC

import umap.umap_ as umap

from utils.load_data import load_cmu_dataset, load_ipa_dataset, build_df_all_frames, df_all_frames_to_X_y, load_test_dataset, load_cmu_test_dataset, load_shuffled_ipa_dataset,leave_one_speaker_out, load_cmu_dataset_MAILABS
# from utils.metrics import compute_PER, plot_cf_matrix
from utils.w2v_gmm_forced_aligner import w2v_gmm_forced_aligner

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        item = d[key]
        # Check if in the inverted dict the key exists
        if item not in inverse: 
            # If not create a new list
            inverse[item] = [key] 
        else: 
            inverse[item].append(key) 
    return inverse

global cmu_alphabet 
global cmu_vowels
cmu_alphabet = [el[0] for el in cmudict.phones()]
cmu_phones_info=cmudict.phones()
cmu_phones=[el[0] for el in cmu_phones_info]
cmu_vowels=[p[0] for p in cmu_phones_info if p[1][0]=='vowel']
cmu_consonants=[p[0] for p in cmu_phones_info if p[1][0]!='vowel']

global ipa_alphabet
with open('data/mfa_phones.json', 'r') as openfile: ipa_alphabet = json.load(openfile)

class Wav2Vec2ForFrameGMMAssignment:

    def __init__(self, nbr_clusters, target_dim, phone_type):
        self.nbr_clusters = nbr_clusters
        self.target_dim = target_dim
        self.status = 'success'
        self.pred_phones_audio = []
        self.fs = 16000
        self.GT_proba_threshold = 1
        
        if phone_type == 'cmu':
            self.alphabet = cmu_alphabet
            self.forced_aligner = w2v_gmm_forced_aligner('cmu') 
        elif phone_type == 'ipa':
            self.alphabet = ipa_alphabet
            self.forced_aligner = w2v_gmm_forced_aligner('ipa') 

        self.gmm = GaussianMixture(n_components=self.nbr_clusters)
        self.reducer = umap.UMAP(n_components=target_dim, random_state=42)
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft", output_hidden_states=True) 
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")    

    def get_last_hidden_state(self, s, fs):
        input_values = self.processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to('cpu')
        with torch.no_grad(): 
            return self.model(input_values).hidden_states[-1]

    def reduce_lhs_dimension(self, lhs):
        return self.reducer.transform(lhs[0])

    def fit(self, X, y=None, supervized_reducer=False, save=False):
        self.X_train = X
        self.y = y
        if supervized_reducer:
            lbl_enc = LabelEncoder()
            self.reducer = self.reducer.fit(self.X_train, y = lbl_enc.fit_transform(self.y))
        else:
            self.reducer = self.reducer.fit(self.X_train)
        self.X_train_reduced = self.reducer.transform(self.X_train)
        self.gmm = self.gmm.fit(self.X_train_reduced)
        self.find_component_phoneme()

        if save:
            pickle.dump(self, open("./data/models/model_{}_{}_{}.pkl","wb".format(save, self.nbr_clusters, self.target_dim)))

    def predict_sample(self, s, fs, target_phonemes):
        phone_prob_matrix = self.predict_phone_prob_matrix(s, fs)
        cost_nonsil, _, _ = self.forced_aligner.get_cost_non_sil(phone_prob_matrix)
        aligned_phones = self.forced_aligner.get_forced_alignment(cost_nonsil, target_phonemes)
        predicted_phones, probs_means = self.forced_aligner.predict(aligned_phones, cost_nonsil, target_phonemes)
        self.pred_phones_audio = predicted_phones
        return predicted_phones

    # X must be a list of lists like [s, fs, target_phonemes] (ex: [[s, fs, target_phonemes], [s, fs, target_phonemes]])
    def predict(self, X): 
        target_phonemes_list = [x[2] for x in X]
        phone_prob_matrix_list = [self.predict_phone_prob_matrix(x[0], x[1]) for x in X]
        cost_nonsil_list = [self.forced_aligner.get_cost_non_sil(ppb)[0] for ppb in phone_prob_matrix_list]
        aligned_phones_list = [self.forced_aligner.get_forced_alignment(cost_nonsil, target_phonemes) for cost_nonsil, target_phonemes in zip(cost_nonsil_list, target_phonemes_list)]
        predicted_phones_list = [self.forced_aligner.predict(aligned_phones, cost_nonsil, target_phonemes)[0] for aligned_phones,cost_nonsil,target_phonemes in zip(aligned_phones_list,cost_nonsil_list,target_phonemes_list)]
        self.pred_phones_audio = predicted_phones_list
        return predicted_phones_list

    def predict_with_timings(self, s, target_phonemes):
        phone_prob_matrix = self.predict_phone_prob_matrix(s, self.fs)
        cost_nonsil, silence_frames_idx, non_silence_frames_idx = self.forced_aligner.get_cost_non_sil(phone_prob_matrix)
        aligned_phones = self.forced_aligner.get_forced_alignment(cost_nonsil, target_phonemes)
        if silence_frames_idx:
            alignment_with_silence = self.forced_aligner.get_alignment_with_silence(aligned_phones, silence_frames_idx, non_silence_frames_idx)
        else:
            alignment_with_silence = aligned_phones
        predicted_phones, probs_means = self.forced_aligner.predict(aligned_phones, cost_nonsil, target_phonemes)
        df_segmented = self.forced_aligner.get_df_segmented(alignment_with_silence, predicted_phones, target_phonemes, probs_means, fs=self.fs, time_per_output=0.02)
        self.pred_phones_audio = list(df_segmented.pred_phones_audio.values)
        return df_segmented

    def predict_word(self, s, phonetics, target_word_idx):
        """phonetics must be a list of list of phonemes, e.g.: phonetics=[['AY1'],['EH1', 'N', 'D', 'IH0', 'D']]
        """
        phones=sum(phonetics,[])
        df_segmented = self.predict_with_timings(s, phones)
        
        start_idx=sum([len(p) for p in phonetics][:target_word_idx])
        end_idx=sum([len(p) for p in phonetics][:target_word_idx+1])
        df_word=df_segmented[start_idx:end_idx]
        return df_word

    def predict_phone(self, audio, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx=0, phoneme_set=cmu_vowels):
        """phonetics must be formatted phonetics as a string, e.g.: 'EH1_N|D_IH0_D'
        """
        phoneme_set=[[p] for p in remove_stress_annots(phoneme_set)]
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

            phonetic_detection = df_word.iloc[p_idx_global].pred_phones_audio

            # #phoneme_set_ids=self.charsiu_processor.get_phone_ids(phoneme_set)[1:-1]
            phoneme_set_ids=self.forced_aligner.labelize_phonemes(phoneme_set)
            proba_means=df_word.iloc[p_idx_global].probs_means

            # df_word['GT_proba'] = [df_word.probs_means[i][j] for i,j in zip(range(len(df_word)), self.forced_aligner.labelize_phonemes(df_word.phones))]


            # if GT_proba is beyond the threshold, we take it as prediction
            if df_word.iloc[p_idx_global].GT_proba>self.GT_proba_threshold:
                phonetic_detection=target_phones
            else:
                # put 0 when not in phoneme_set so that we take max propa only among phoneme_set
                filtered_proba_means=[0 if i not in phoneme_set_ids else el for i,el in enumerate(proba_means)]
                idx_mean_max=np.argmax(filtered_proba_means)
                phonetic_detection = self.forced_aligner.label_encoder.inverse_transform([np.argmax(filtered_proba_means)])[0]
                # phonetic_detection=self.charsiu_processor.mapping_id2phone(int(idx_mean_max))
            syl[p_idx_local]=phonetic_detection
            
        else:
            phonetic_detection=float('nan')
            syl=float('nan')
        return phonetic_detection, syl

    # creates a dictionnary which computes every phoneme associated to each cluster
    def create_cluster_dict(self):
        cluster_dict = {}
        for i in range(self.nbr_clusters):
            cluster_dict[i] = []

        attributions = self.gmm.predict(self.X_train_reduced)
        for i in range(len(attributions)):
            cluster_dict[attributions[i]].append(self.y[i])

        return cluster_dict

    # creates a dictionnary which counts the occurences of each phoneme in each cluster
    def count_phoneme_per_cluster(self, cluster_dict):
        phoneme_count_per_cluster = {}
        for cluster in cluster_dict.keys():
            phoneme_count_per_cluster[cluster] = [(phon, cluster_dict[cluster].count(phon)) for phon in set(cluster_dict[cluster])]
        
        return phoneme_count_per_cluster

    # creates a dictionnary which computes the most reccurent phoneme in each cluster
    def define_phoneme_per_cluster(self, phoneme_count_per_cluster):
        component_to_phoneme = {}
        for component, phonemes in phoneme_count_per_cluster.items():
            if len(phonemes):
                sublist = [phon[1] for phon in phonemes]
                component_to_phoneme[component] = phonemes[sublist.index(max(sublist))][0]
        
        return component_to_phoneme

    # creates a dictionnary which computes every Gaussian component associated with each phoneme 
    def find_component_phoneme(self):
        self.cluster_dict = self.create_cluster_dict()
        self.phoneme_count_per_cluster = self.count_phoneme_per_cluster(self.cluster_dict)
        self.component_to_phoneme = self.define_phoneme_per_cluster(self.phoneme_count_per_cluster)
        self.phoneme_to_components = invert_dict(self.component_to_phoneme)

    # from an audio, computes the output of GMM and for this output, computes the most probable phoneme for each frame
    def predict_audio_phonemes_gmm(self, s_list, fs_list):
        lhs_list = [self.get_last_hidden_state(s, fs) for s,fs in zip(s_list, fs_list)]
        reduced_lhs = [self.reduce_lhs_dimension(lhs) for lhs in lhs_list]
        labels_list = self.gmm.predict(reduced_lhs)
        audio_preds_list = [[self.component_to_phoneme[l] for l in labels] for labels in labels_list]
        return audio_preds_list

    # from an audio sample, computes the probability matrix of each frame corresponding to every phoneme
    def predict_phone_prob_matrix(self, s, fs):
        lhs = self.get_last_hidden_state(s, fs)
        reduced_lhs = self.reduce_lhs_dimension(lhs)
        probs = self.gmm.predict_proba(reduced_lhs)
        phone_prob_matrix = np.zeros((len(lhs[0]), len(self.alphabet)))

        for i in range(len(lhs[0])):
            for j in range(len(self.alphabet)):
                if self.alphabet[j] in self.phoneme_to_components.keys():
                    phone_prob_matrix[i][j] = sum(probs[i][self.phoneme_to_components[self.alphabet[j]]])

        return phone_prob_matrix

    def compute_stress_score(self, audio, phonetics):
        """Use textgridData to have the timings of vowels and compute prosody features (intesity, pitch, ...) to compute 
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
        #_, textgridData, _ = self.align_phones(audio=audio,phones=split_phonetics)
        textgridData = self.predict_with_timings(audio, split_phonetics)

        # # TODO: remove this block: the silence and consecutive things processing, as I already do that in align_phones now
        # # textgridData=self.force_and_predict(audio,split_phonetics)
        # textgridData=textgridData[textgridData.cmu_phones != '[SIL]']
        # # I have to collapse if several consecutive vowels are the same. it can happen when the predictions are not the same.
        # # I thus have to group the timings (first start until last end)
        # #  here we delete consecutives but keep first and last, so there is a possibility of only two consecutves after that, and having overall start and end
        # test=keep_first_last(textgridData.cmu_phones)
        # # keep firsts and lasts (thus only when there is two consecutive phonemes)
        # starts=test.loc[test.shift(-1) == test]
        # ends=test.loc[test.shift(+1) == test]
        # assert len(starts)==len(ends)
        # # drop duplicates keeping first
        # drop_duplicates=lambda a: a.loc[a.shift(+1) != a]
        # filtered_df=textgridData.loc[drop_duplicates(textgridData.cmu_phones).index]
        # here in the filtered_df containg only the first occurence for equal consecutive examples, we replace the 'end' value with the line in the "ends"
        # for rownum,(indx,val) in enumerate(starts.iteritems()): filtered_df.loc[indx,'end']=textgridData.loc[ends.index[rownum],'end']

        # select vowels
        filtered_df=textgridData[textgridData.phones.isin(cmu_vowels)]#.index.tolist()

        f0Samples=getIntonation(audio, self.fs)
        intensity=getIntensity(audio, self.fs)

        # extract features
        # each word start and end position expressed in samples
        startPositions_samples = (round(self.fs*filtered_df.loc[:,'start'])+1).astype(int).tolist()
        stopPositions_samples = round(self.fs*filtered_df.loc[:,'end']).astype(int).tolist()

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

    # plot frame dimension reduction on train data dimension reduction
    def plot(self, s, fs, plt_label_points=True, plt_cbar=False):
        plt.scatter([x[0] for x in self.X_train_reduced], [x[1] for x in self.X_train_reduced], s=2, c=self.gmm.predict(self.X_train_reduced), cmap='gnuplot') 
        lhs = self.get_last_hidden_state(s, fs)  
        reduced_lhs = self.reduce_lhs_dimension(lhs)  
        plt.scatter([x[0] for x in reduced_lhs], [x[1] for x in reduced_lhs],c='r', s=20,marker='D') 
        if plt_label_points:
            label_points = list(zip(self.X_train_reduced, self.y))
            ban_list=[]
            for elem in label_points:
                if elem[1] not in ban_list:
                    ban_list.append(elem[1])
                    plt.annotate(elem[1], (elem[0][0], elem[0][1]))
        if plt_cbar:
            cbar = plt.colorbar()
            cbar.set_ticks(np.arange(self.nbr_clusters))
            cbar.set_ticklabels(list(self.component_to_phoneme.values()))

if __name__ == '__main__':

    if not os.path.exists('./data/'): os.makedirs('./data/')
    pickle.dump(classe, open("./data/models/model_librispeech_300_18.pkl","wb"))
    
    classe = pd.read_pickle('./data/classe.pkl')

    # loading a train sample
    #df_t, df=libri_phonetics_data(data_set='dev-clean')
    #df_cmu_phones=df_t.apply(lambda r: pd.DataFrame.from_records(r.phone_df).cmu_phone.tolist(), axis=1)
    N=0
    example=df.iloc[N]
    path=example.wav_path
    example['cmu_phones']=df_cmu_phones.iloc[N]
    example['cmu_phones']=remove_stress_annots(example.cmu_phones)
    s,fs=librosa.load(path, sr=16000)

    # loading a test sample (unseen by the model)
    N=0
    example=df_test.iloc[N]
    path=example.wav_path
    example['cmu_phones']=df_cmu_phones_test.iloc[N]
    example['cmu_phones']=remove_stress_annots(example.cmu_phones)
    s,fs=librosa.load(path, sr=16000)

    #comment if you want cmu or ipa
    df_t_train, df_t_test = load_cmu_dataset()
    #df_t_train, df_t_test = load_ipa_dataset(lang_code='en_US')
    df_all_frames = build_df_all_frames(df_t_train, 'cmu_phone')
    X, y = df_all_frames_to_X_y(df_all_frames)

    # leave one speaker out
    all_speakers = ['fr_FR', 'es_ES', 'en_UK', 'en_US']
    speaker_lang_code = 'en_UK'
    others = list(set(all_speakers) - set([speaker_lang_code]))

    # df_t_train, df_t_test = leave_one_speaker_out(speaker_lang_code, all_speakers)
    df_t_train, df_t_test = load_cmu_dataset_MAILABS(speaker_lang_code, ['en_US', 'en_UK'])
    df_all_frames = build_df_all_frames(df_t_train, 'ipa_phone')
    X, y = df_all_frames_to_X_y(df_all_frames)

    classe = Wav2Vec2ForFrameGMMAssignment(300,18,'cmu')
    classe.fit(X, y)
    classe.find_component_phoneme()

    # phoneme predictions on a train dataset with forced alignment
    # comment for cmu or ipa
    data = load_cmu_test_dataset(df_t_test)
    # data = load_test_dataset(df_t_test)

    # x = [[el.s,el.fs,el.cmu_phones] for _,el in data.iterrows()]
    x = [[el.s,el.fs,el.cmu_phones] for _,el in data.iterrows()]
    preds = classe.predict(x)

    # phoneme predictions on a single audio sample with forced alignment
    preds = classe.predict_sample(s, fs, target_phonemes)

    # phoneme predictions on a train dataset with forced alignment
    # comment for cmu or ipa
    data = load_cmu_test_dataset(df_t_test)
    # data = load_test_dataset(df_t_test)
    s_list = data.s.tolist()
    fs_list = data.fs.tolist()
    forced_aligner = w2v_gmm_forced_aligner('cmu') 
    target_phonemes_list = data.cmu_phones.tolist()
    # target_phonemes_list = [remove_stress_annots(i) for i in target_phonemes_list]
    phone_prob_matrix_list = [classe.predict_phone_prob_matrix(s, fs) for s,fs in zip(s_list, fs_list)]
    cost_nonsil_list = [forced_aligner.get_cost_non_sil(ppb)[0] for ppb in phone_prob_matrix_list]
    aligned_phones_list = [forced_aligner.get_forced_alignment(cost_nonsil, target_phonemes) for cost_nonsil, target_phonemes in zip(cost_nonsil_list, target_phonemes_list)]
    predicted_phones_list = [forced_aligner.predict(aligned_phones, cost_nonsil, target_phonemes)[0] for aligned_phones,cost_nonsil,target_phonemes in zip(aligned_phones_list,cost_nonsil_list,target_phonemes_list)]

    flat_list_predictions = [item for sublist in predicted_phones_list for item in sublist]
    flat_list_target = [item for sublist in target_phonemes_list for item in sublist]

    # compute PER for every test sample
    # PERs = compute_PER(predicted_phones_list, target_phonemes_list)
    # mean_PER = np.mean(PERs)

    # ax = plot_cf_matrix(flat_list_target, flat_list_predictions, labels = cmu_alphabet) 
    # plt.show()

    # test predict_word
    N = 0
    language_code = 'en_US'
    sentence = df_t_test.iloc[N].text
    target_word_idx = 3

    if classe.alphabet == cmu_alphabet:
        mode = 'CMU'
    elif classe.alphabet == ipa_alphabet:
        mode = 'MFA_IPA'

    ph= prefill_for_sentence(sentence=sentence, lang=language_code, mode=mode)['cmu_phonetics']
    split_phonetics=[p.replace('|','_').split('_') for p in ph.split(' ')]
    phonetics_list = ph.split(' ')
    phonetics_list = [[i] for i in phonetics_list]
    phonetics = [i[0].replace('|', '_').split('_') for i in phonetics_list]

    df_word = classe.predict_word(s, split_phonetics, target_word_idx)

    target_word_idx = 9
    target_syllable_idx = 0
    target_phones = 'AH'
    pred_phone = classe.predict_phone(s, ph, target_word_idx, target_syllable_idx, target_phones)