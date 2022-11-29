import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import librosa
import json
import ast


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder
from src.audio_processing import getIntonation, getIntensity, normalize
from src.text_processing import unstress, prefill_for_sentence, remove_stress_annots
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from umap.umap_ import UMAP




import cmudict
import itertools
from src.libri_phonetization_data import libri_phonetics_data
from jiwer import wer
import seaborn as sns
from sklearn.metrics import confusion_matrix
from time import time
from linetimer import CodeTimer
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.load_data import load_libri_dataset, df_all_frames_to_X_y, load_cmu_test_dataset
# from src.metrics import compute_PER, plot_cf_matrix
from src.dtw_forced_aligner import dtw_forced_aligner

from src.pronunciation_dictionaries import cmu_alphabet, ipa_alphabet, cmu_phones_info, cmu_reducer

global cmu_vowels
# cmu_phones=[el[0] for el in cmu_phones_info]
cmu_vowels=[p[0] for p in cmu_phones_info if p[1][0]=='vowel']
cmu_consonants=[p[0] for p in cmu_phones_info if p[1][0]!='vowel']


class Wav2Vec2ForFramePrediction:

    # phone_type = 'cmu' or 'ipa'
    def __init__(self, phone_type, reducer=PCA(n_components=0.95, random_state=42), frame_classifier=KNeighborsClassifier(10)):#, phoneme_classifier=None):
        self.status = 'success'
        self.pred_phones_audio = []
        self.fs = 16000
        
        if phone_type == 'cmu':
            self.alphabet = cmu_alphabet
            self.forced_aligner = dtw_forced_aligner('cmu') 
        elif phone_type == 'ipa':
            self.alphabet = ipa_alphabet
            self.forced_aligner = dtw_forced_aligner('ipa') 
        
        self.id_to_p={i:p for i,p in enumerate(self.alphabet+['[SIL]'])}
        self.p_to_id={p:i for i,p in enumerate(self.alphabet+['[SIL]'])}
        
        # if reducer == "umap":
        #     # parameters advised for clustering: https://umap-learn.readthedocs.io/en/latest/clustering.html
        #     self.reducer = UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=42)
        # elif reducer == "parametric_umap":
        #     from umap.parametric_umap import ParametricUMAP
        #     self.reducer = ParametricUMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=42)
        # elif reducer == "pca":
        #     self.reducer = PCA(n_components=target_dim, random_state=42)

        self.reducer=reducer
        self.frame_classifier=frame_classifier

        # self.phoneme_classifier=phoneme_classifier
        # if self.phoneme_classifier is not None:
        #     self.phoneme_reducer = PCA(n_components=target_dim, random_state=42)


        # import Wav2Vec2 feature extractor
        self.model = Wav2Vec2Model.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", output_hidden_states=True) 
        self.processor = Wav2Vec2Processor.from_pretrained("hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft")    

    def save(self, out_path='models', name='model_mailabs_pca_0.95_knn_10_w'):
        path=os.path.join(out_path,name)
        if not os.path.exists(path): os.makedirs(path)
        pickle.dump(self.reducer, open( path+"/reducer.p", "wb" ))
        pickle.dump(self.frame_classifier, open( path+"/frame_classifier.p", "wb" ))

    def load(self, out_path='models', name='model_mailabs_pca_0.95_knn_10_w'):
        path=os.path.join(out_path,name)
        self.reducer=pd.read_pickle(path+"/reducer.p")
        self.frame_classifier=pd.read_pickle(path+"/frame_classifier.p")
        



    # get output from an audio sample in w2v2 feature extractor
    def get_last_hidden_state(self, s, fs):
        input_values = self.processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to('cpu')
        with torch.no_grad(): 
            return self.model(input_values).hidden_states[-1]

    # reduce dimension from w2v2 output
    def reduce_lhs_dimension(self, lhs):
        return self.reducer.transform(lhs[0])

    def fit(self, X, y):
        self.X_train = X
        self.y_train_labels = y
        
        self.y_train=[self.p_to_id[el] for el in y]

        
        print('fit frame reducer...')
        self.reducer.fit(self.X_train)
        print('reduce training data')
        self.X_train_reduced = self.reducer.transform(self.X_train)
        print('fit frame classifier...')
        self.frame_classifier.fit(self.X_train_reduced, self.y_train)

    # def fit_phoneme(self, X, y):
    #     if self.phoneme_classifier is not None:
    #         self.X_train = X
    #         self.y_train_labels = y
    #         self.y_train=[self.p_to_id[el] for el in y]

    #         print('fit phoneme reducer...')
    #         self.phoneme_reducer.fit(self.X_train)
    #         print('reduce training data')
    #         self.X_train_reduced = self.phoneme_reducer.transform(self.X_train)
    #         print('fit phoneme classifier...')
    #         self.phoneme_classifier.fit(self.X_train_reduced, self.y_train)
    #     else:
    #         print('phoneme_classifier is set to None, so nothing is done. Averaged frame predictions will be used to predict phonemes')
     

    # from an audio sample, computes the probability matrix of each frame corresponding to every phoneme
    def predict_phone_prob_matrix(self, s, fs):
        self.timestamps = []
        start = time()
        self.lhs = self.get_last_hidden_state(s, fs)
        self.timestamps.append(time()-start)
        
        start = time()
        reduced_lhs = self.reduce_lhs_dimension(self.lhs)
        self.timestamps.append(time()-start)

        start = time()
        phone_prob_matrix = self.frame_classifier.predict_proba(reduced_lhs)
        self.timestamps.append(time()-start)
        
        # we defined the silence as the last token, we remove it here. 
        # Silence will be deteted in the forced aligner by checking that the sum of the remaining probablities are not close to 1 (<0.2)
        phone_prob_matrix = phone_prob_matrix[:,:-1]
        
        print('times of get_last_hidden_state, reduce_lhs_dimension, classifier predict_proba')
        print(self.timestamps)

        return phone_prob_matrix

    # predict and get proba means per phoneme alignment and GT
    def predict_with_timings(self, s, target_phonemes):
        phone_prob_matrix = self.predict_phone_prob_matrix(s, self.fs)

        df_segmented=self.forced_aligner.probas_to_df_segmented(phone_prob_matrix, target_phonemes, fs=self.fs)

        avg_vectors=[]
        for i,r in df_segmented.iterrows():
            avg_vector=self.lhs.numpy()[0][r.start_idx:r.end_idx,:].mean(axis=0)
            avg_vectors.append(avg_vector)
        df_segmented['average_vectors']=avg_vectors

        # if self.phoneme_classifier is not None:
        #     reduced_vectors=self.phoneme_reducer.transform(avg_vectors)
        #     pred_idxs=self.phoneme_classifier.predict(reduced_vectors)
        #     self.pred_phones_audio=[self.id_to_p[el] for el in list(pred_idxs)]
        #     df_segmented['pred_phones_audio']=self.pred_phones_audio
        # else:
        self.pred_phones_audio = list(df_segmented.pred_phones_audio.values)

        return df_segmented

    # predict a specific word in a sample through its index in phonetics
    def predict_word(self, s, phonetics, target_word_idx):
        """phonetics must be a list of list of phonemes, e.g.: phonetics=[['AY1'],['EH1', 'N', 'D', 'IH0', 'D']]
        """
        phones=sum(phonetics,[])
        df_segmented = self.predict_with_timings(s, phones)
        
        start_idx=sum([len(p) for p in phonetics][:target_word_idx])
        end_idx=sum([len(p) for p in phonetics][:target_word_idx+1])
        df_word=df_segmented[start_idx:end_idx]
        return df_word

    # predict a specific word in a sample through its word index in phonetics and its syllable index in word
    def predict_phone(self, audio, phonetics, target_word_idx, target_syllable_idx, target_phones, target_occurence_idx=0, phoneme_set=cmu_vowels, GT_proba_threshold=0.2):
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
            if df_word.iloc[p_idx_global].GT_proba>GT_proba_threshold:
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
        with CodeTimer('whole phone prediction'): textgridData = self.predict_with_timings(audio, split_phonetics)

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


if __name__ == '__main__':

    from src.load_data import *
    from src.wav2vec2_frame_prediction import *
    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction

    # ----------------------------


    # leave one speaker out
    # all_speakers = ['fr_FR', 'es_ES', 'en_UK', 'en_US']
    # speaker_lang_code = 'en_UK'
    # others = list(set(all_speakers) - set([speaker_lang_code]))

    # df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', phone_set='MFA_IPA')
    # df_t_train=df_t_train.dropna()
    # df_all_frames = build_df_all_frames(df_t_train, 'phone')
    # df_all_frames.to_pickle('df_all_frames_MAILABS_train_ipa.pkl')

    # Basis model Wav2Vec2ForFramePrediction, in CMU phoneme set, PCA reduction at 95% variance, and a 10-NN classifier
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(0.95,'cmu', reducer="pca")
    model.fit(X, y)
    # model.fit_phoneme(X_train, y_train)
    pickle.dump(model,open('model_mailabs_pca_0.95_knn_10.pkl','wb'))

    # Basis model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w')

    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.load(name='model_mailabs_pca_0.95_knn_10_w')

    # model.fit_phoneme(X_train, y_train)
    # pickle.dump(model,open('model_mailabs_pca_0.95_knn_10_w.pkl','wb'))

    # Basis model, but with IPA
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction(0.95,'ipa', reducer="pca")
    model.fit(X, y)
    pickle.dump(model,open('model_mailabs_pca_0.95_knn_10_ipa.pkl','wb'))



    
    
    
    df_all_instances=pd.read_pickle('df_all_phonemes_instances_MAILABS_train.pkl')
    df_all_instances['sum']=df_all_instances.iloc[:,1].apply(lambda r: r.sum())
    df_all_instances=df_all_instances.dropna()
    import numpy as np
    X_train=np.array(list(df_all_instances.iloc[:,1].values))    
    # id_to_p={i:p for i,p in enumerate(cmu_alphabet+['[SIL]'])}
    # p_to_id={p:i for i,p in enumerate(cmu_alphabet+['[SIL]'])}
    # y_train=df_all_instances.phoneme.apply(lambda r: p_to_id[r]).values
    y_train=list(df_all_instances.phoneme.values)


    
    # phoneme predictions on a train dataset with forced alignment
    # comment for cmu or ipa
    df_t_train, df_t_test = load_libri_dataset()
    data = load_cmu_test_dataset(df_t_test)
    # data = load_test_dataset(df_t_test)

    # phoneme predictions on a single audio sample with forced alignment
    pred = model.predict_with_timings(data.s.iloc[0], data.cmu_phones.iloc[0])
    prob_matrix = model.predict_phone_prob_matrix(data.s.iloc[0], 16000)

    # model2.phoneme_classifier
    
    # # phoneme predictions on a single audio sample with forced alignment
    # pred2 = model2.predict_with_timings(data.s.iloc[0], data.cmu_phones.iloc[0])
    # prob_matrix2 = model2.predict_phone_prob_matrix(data.s.iloc[0], 16000)

    # from tqdm import tqdm
    # preds=[model.predict_with_timings(r.s, r.cmu_phones) for i,r in tqdm(data.iterrows())]
    # preds2=[model2.predict_with_timings(r.s, r.cmu_phones) for i,r in tqdm(data.iterrows())]
    # preds_df=pd.concat(preds)
    # preds_df2=pd.concat(preds2)
    # sum(preds_df.pred_phones_audio==preds_df2.pred_phones_audio)/len(preds_df)


    # https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
    from sklearn.ensemble import VotingClassifier

    #  with ensemble
    estimators=[('10 Nearest Neighbors', KNeighborsClassifier(n_neighbors=10, weights='distance')),
    ('QDA', QuadraticDiscriminantAnalysis())]
    eclf = VotingClassifier(estimators=estimators,
                        voting='soft', weights=[1 for _ in estimators])
    model = Wav2Vec2ForFramePrediction('cmu',frame_classifier=eclf)
    model.fit(X, y)

    # pickle.dump(model,open('model_mailabs_pca_0.95_eclf_knn_10_linear_svm_qda.pkl','wb'))


    ###
    
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    # model.fit_phoneme(X_train, y_train)
    # pickle.dump(model,open('model_mailabs_pca_0.95_eclf_knn_10_weighted_dist.pkl','wb'))