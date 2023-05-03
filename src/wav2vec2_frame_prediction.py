import pandas as pd
import numpy as np
import pickle
import os
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.audio_processing import getIntonation, getIntensity, normalize
from src.text_processing import unstress, remove_stress_annots
from sklearn.decomposition import PCA
# from umap.umap_ import UMAP


from time import time
from linetimer import CodeTimer
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.load_data import load_libri_dataset, df_all_frames_to_X_y, load_libri_dataset_audio_timings
# from src.metrics import compute_PER, plot_cf_matrix
from src.dtw_forced_aligner import dtw_forced_aligner
from src.pronunciation_dictionaries import cmu_alphabet, ipa_alphabet, cmu_phones_info, cmu_reducer
from src.text_processing import remove_stress_annots, phonetics_indexed_df_from_formatted_phonetics, unstress, drop_consecutive_duplicate_elements, drop_consecutive_duplicates


global cmu_vowels
# cmu_phones=[el[0] for el in cmu_phones_info]
cmu_vowels=[p[0] for p in cmu_phones_info if p[1][0]=='vowel']
cmu_consonants=[p[0] for p in cmu_phones_info if p[1][0]!='vowel']


# processing functions of df_segmented, which is the output of prediction and forced alignment

def extract_word(df_segmented, phonetics, target_word_idx):
    start_idx=sum([len(p) for p in phonetics][:target_word_idx])
    end_idx=sum([len(p) for p in phonetics][:target_word_idx+1])
    df_word=df_segmented[start_idx:end_idx]
    return df_word

class Wav2Vec2ForFramePrediction:
    # phone_type = 'cmu' or 'ipa'
    def __init__(self, phone_type, w2v2_model_path="hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft", w2v2_model_format="torch", reducer=PCA(n_components=0.95, random_state=42), frame_classifier=KNeighborsClassifier(10)):#, phoneme_classifier=None):
        """
        w2v_format: 'torch' or 'onnx'
        """
        self.status = 'success'
        self.pred_phones_audio = []
        self.fs = 16000
        self.time_per_output=0.02
        
        if phone_type == 'cmu':
            self.alphabet = cmu_alphabet
            self.forced_aligner = dtw_forced_aligner('cmu') 
        elif phone_type == 'ipa':
            self.alphabet = ipa_alphabet
            self.forced_aligner = dtw_forced_aligner('ipa') 
        
        self.id_to_p={i:p for i,p in enumerate(self.alphabet+['[SIL]'])}
        self.p_to_id={p:i for i,p in enumerate(self.alphabet+['[SIL]'])}
        

        self.reducer=reducer
        self.frame_classifier=frame_classifier

        # self.phoneme_classifier=phoneme_classifier
        # if self.phoneme_classifier is not None:
        #     self.phoneme_reducer = PCA(n_components=target_dim, random_state=42)


        # import Wav2Vec2 feature extractor
        self.w2v2_model_format=w2v2_model_format
        if w2v2_model_format=="torch":
            self.model = Wav2Vec2Model.from_pretrained(w2v2_model_path, output_hidden_states=True) 
        else:
            import onnxruntime as rt
            quantized_model_name = "last_hidden_state.quant.onnx"
            quantized_model_path='/'.join([w2v2_model_path,quantized_model_name])
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = rt.InferenceSession(quantized_model_path, sess_options)

        self.processor = Wav2Vec2Processor.from_pretrained(w2v2_model_path)    

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

        if self.w2v2_model_format=="torch":
            with torch.no_grad(): 
                return self.model(input_values).hidden_states[-1]
        else:
            onnx_outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_values.numpy()})[0]
            return onnx_outputs

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

    # from an audio sample, computes the probability matrix of each frame corresponding to every phoneme
    def predict_phone_prob_matrix(self, s, fs):
        self.timestamps = []
        start = time()
        self.lhs = self.get_last_hidden_state(s, fs)
        self.timestamps.append(time()-start)
        
        start = time()
        self.reduced_lhs = self.reduce_lhs_dimension(self.lhs)
        self.timestamps.append(time()-start)

        start = time()
        phone_prob_matrix = self.frame_classifier.predict_proba(self.reduced_lhs)
        self.timestamps.append(time()-start)

        # if during training, the classifier has not seen some of the labels, it won't be in the possible labels, and the proba matrix will have a reduced shape
        # thus here I extract indices that don't have a column in the matrix to then add rows of zeros and have a correct shape
        ids_to_add=[el for el in range(len(self.id_to_p)) if el not in self.frame_classifier.classes_]

        for i in ids_to_add:
            phone_prob_matrix=np.concatenate([phone_prob_matrix[:,:i], np.zeros((phone_prob_matrix.shape[0],1)), phone_prob_matrix[:,i:]], axis=1)
        
        # we defined the silence as the last token, we remove it here. 
        # Silence will be deteted in the forced aligner by checking that the sum of the remaining probablities are not close to 1 (<0.2)
        # phone_prob_matrix = phone_prob_matrix[:,:-1]
        self.phone_prob_matrix=phone_prob_matrix
        
        print('times of get_last_hidden_state, reduce_lhs_dimension, classifier predict_proba')
        print(self.timestamps)
        return phone_prob_matrix

    # predict and get proba means per phoneme alignment and GT
    def predict_with_timings(self, s, target_phonemes):
        phone_prob_matrix = self.predict_phone_prob_matrix(s, self.fs)
        df_segmented=self.forced_aligner.probas_to_df_segmented(phone_prob_matrix, target_phonemes, fs=self.fs)
        self.pred_phones_audio = list(df_segmented.pred_phones_audio.values)

        return df_segmented

def train_Wav2Vec2ForFramePrediction_model():
    

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    
    df_all_instances_select_with_silences=pd.read_pickle('df_all_instances_select_with_silences.pkl')
    X, y = df_all_frames_to_X_y(df_all_instances_select_with_silences)
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_pca_95_knn_10_w')

    
    df_all_instances_select_with_silences=pd.read_pickle('df_all_instances_select_with_silences_no_CH_JH.pkl')
    X, y = df_all_frames_to_X_y(df_all_instances_select_with_silences)
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_pca_95_knn_10_w_no_CH_JH')

    # Basis model Wav2Vec2ForFramePrediction, a 10-NN classifier weighted with distances
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w')

    
    df_all_instances_select_with_silences=pd.read_pickle('df_all_instances_select_with_silences.pkl')
    X, y = df_all_frames_to_X_y(df_all_instances_select_with_silences)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.95, random_state=42), frame_classifier=LinearDiscriminantAnalysis())
    model.fit(X, y)
    model.save(name='model_mailabs_equilibrated_pca_99_lda')



    
    # Basis model Wav2Vec2ForFramePrediction, a 10-NN classifier weighted with distances
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('ipa', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_95_knn_10_w_ipa')

    
    # Basis model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=KNeighborsClassifier(5, weights='distance', metric='cosine'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_99_knn_5_cos_w')

    
    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames=pd.read_pickle('df_all_frames_commonvoice_en_dev.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=LinearDiscriminantAnalysis())
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_lda')

    
    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames=pd.read_pickle('df_all_frames_commonvoice_en_dev.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'))
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_knn')

    
    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames=pd.read_pickle('df_all_frames_commonvoice_en_dev_uk_us_ca_n_1000.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'))
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_knn_uk_us_ca_n_1000')

    
    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    df_all_frames=pd.read_pickle('df_all_frames_commonvoice_en_dev_uk_us_ca_n_1000.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=LinearDiscriminantAnalysis())
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_lda_uk_us_ca_n_1000')

    
    # model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances, UK US FR ES, IPA
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('ipa', frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_cos_w_UK_US_FR_ES')

    
    # Basis model Wav2Vec2ForFramePrediction, PCA 99% variance, logistic regression
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=LogisticRegression(max_iter=1000))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_99_logistic_regression')

    
    # model Wav2Vec2ForFramePrediction, PCA 99% variance, logistic regression, UK US FR ES, IPA
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('ipa', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=LogisticRegression(max_iter=1000))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_99_logistic_regression_UK_US_FR_ES')


    

    # estimators=[('5-NN cosine metric weighted', KNeighborsClassifier(5, weights='distance', metric='cosine')), ('LogisticRegression', LogisticRegression(max_iter=1000))]
    # eclf = VotingClassifier(estimators=estimators, voting='soft', weights=[1,5])
    # reducer=PCA(n_components=0.99, random_state=42)

def inference_demo():
    
    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction
    # default_model_cmu = Wav2Vec2ForFramePrediction('cmu')
    model = Wav2Vec2ForFramePrediction('cmu',w2v2_model_format="onnx")
    # model.load(name='model_mailabs_pca_0.95_knn_10_w')
    model.load(name='model_mailabs_equilibrated_pca_95_knn_10_w')


    # --------------- Inference demo --------------------
    
    # phoneme predictions on a train dataset with forced alignment
    # comment for cmu or ipa
    df_t_train, df_t_test = load_libri_dataset()
    data = load_libri_dataset_audio_timings(df_t_test)
    # data = load_libri_dataset_audio_timings(df_t_train)

    # data[data.apply(lambda r: "ZH" in r.cmu_phones, axis=1)]
    # data = load_test_dataset(df_t_test)

    # # phoneme predictions on a single audio sample with forced alignment
    # pred = model.predict_with_timings(data.s.iloc[0], data.cmu_phones.iloc[0])
    # prob_matrix = model.predict_phone_prob_matrix(data.s.iloc[0], 16000)

    
    # phoneme predictions on a single audio sample with forced alignment
    df_segmented = model.predict_with_timings(data.s.iloc[0], data.cmu_phones.iloc[0])
    prob_matrix = model.predict_phone_prob_matrix(data.s.iloc[0], 16000)

    # from src.label_data_processing import synth_words_data
    from src.audio_processing import read_audio_file
    from src.text_processing import prefill_for_sentence
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    # df=synth_words_data()

    word="sister"
    phonetics=prefill_for_sentence(word)['phonetics']
    path="data/synth_audio/cmu_words/standard/prosody/Joanna/F_US_"+word+".mp3"
    s,fs=read_audio_file(path, fs=16000)

    phone_list=re.sub("[0-9]","",phonetics).replace('|',"_").split('_')


    # phoneme predictions on a single audio sample with forced alignment
    df_segmented = model.predict_with_timings(s,phone_list)
    prob_matrix = model.predict_phone_prob_matrix(s, 16000)

    latentogram=model.reducer.transform(model.lhs[0])
    df_segmented.start_idx.tolist()

    # to have horizontal line in white in the heatmap at the phone starts, I put a 6
    latentogram[df_segmented.start_idx.tolist(),:]=10

    plt.clf()
    sns.heatmap(latentogram)
    plt.savefig('w2v_latentogram_reduced.png')


    return df_segmented, prob_matrix

def use_tests():

    
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.svm import SVC
    # from sklearn.gaussian_process import GaussianProcessClassifier
    # from sklearn.gaussian_process.kernels import RBF
    # from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    
    from sklearn.linear_model import LogisticRegression

    # backup different possible reducers
    # if reducer == "umap":
    #     # parameters advised for clustering: https://umap-learn.readthedocs.io/en/latest/clustering.html
    #     self.reducer = UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=42)
    # elif reducer == "parametric_umap":
    #     from umap.parametric_umap import ParametricUMAP
    #     self.reducer = ParametricUMAP(n_components=target_dim, n_neighbors=30, min_dist=0.0, random_state=42)
    # elif reducer == "pca":
    #     self.reducer = PCA(n_components=target_dim, random_state=42)

    from src.load_data import load_dataset_MAILABS, load_dataset_commonvoice, build_df_all_frames
    from src.wav2vec2_frame_prediction import Wav2Vec2ForFramePrediction

    df_t=load_dataset_commonvoice(lang_codes=['en'], path='./data/cv-corpus-10.0-delta-2022-07-04', split="dev", phone_set='CMU')
    df_t.accents.unique()
    accents=['United States English']
    # accents=['England English']
    df_t=df_t[df_t.accents.isin(accents)]
    df_all_frames = build_df_all_frames(df_t.sample(frac=1, random_state=0), 'phone')
    df_all_frames.to_pickle('df_all_frames_commonvoice_en_dev_us.pkl')
    
    # Basis model Wav2Vec2ForFramePrediction, COMMON VOICE DATA
    # df_all_frames=pd.read_pickle('df_all_frames_commonvoice_en_dev_uk.pkl')

    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', reducer=PCA(n_components=0.99, random_state=42), frame_classifier=KNeighborsClassifier(10, weights='distance', metric='cosine'))
    model.fit(X, y)
    model.save(name='model_commonvoice_pca_99_knn_us')


    # ----------------------------

    # load UK US FR ES in MFA_IPA
    df_t_train_all = load_dataset_MAILABS(['en_US', 'en_UK', 'es_ES', 'fr_FR'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', phone_set='MFA_IPA')
    df_t_train_all=df_t_train_all.dropna()
    df_all_frames_all = build_df_all_frames(df_t_train_all, 'phone')
    df_all_frames_all.to_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')

    
    from src.wav2vec2_utils import plot_reduction
    df_all_frames_all['average_vector']=df_all_frames_all['vector']
    df_all_frames_all.sample(frac=1, random_state=1)
    plot_reduction(df_all_frames_all.sample(frac=1, random_state=1), reduction_technique='umap', base_name='plots/w2v_xlsr_ft_space_phoneme', legend_label='phoneme')
    plot_reduction(df_all_frames_all.sample(frac=1, random_state=1), reduction_technique='umap', base_name='plots/w2v_xlsr_ft_space_genre', legend_label='genre')
    plot_reduction(df_all_frames_all.sample(frac=1, random_state=1), reduction_technique='umap', base_name='plots/w2v_xlsr_ft_space_speaker', legend_label='speaker')
    plot_reduction(df_all_frames_all.sample(frac=1, random_state=1), reduction_technique='umap', base_name='plots/w2v_xlsr_ft_space_language_code', legend_label='language_code')



    # load UK US in MFA_IPA
    # df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', phone_set='MFA_IPA')
    # df_t_train=df_t_train.dropna()
    # df_all_frames = build_df_all_frames(df_t_train, 'phone')
    # df_all_frames.to_pickle('df_all_frames_MAILABS_train_ipa.pkl')


    # load US english in CMU
    df_t_train = load_dataset_MAILABS(['en_US'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS')
    df_t_train=df_t_train.dropna()

    df_all_frames = build_df_all_frames(df_t_train, 'phone')
    df_all_frames.to_pickle('df_all_frames_MAILABS_US_train.pkl')

    # load UK US english in CMU
    df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS')
    df_t_train=df_t_train.dropna()

    df_all_frames = build_df_all_frames(df_t_train, 'phone')
    df_all_frames.to_pickle('df_all_frames_MAILABS_train.pkl')
    
    df_all_frames = build_df_all_frames(df_t_train, 'phone', model_path="hf_models/facebook/wav2vec2-base-960h")
    df_all_frames.to_pickle('df_all_frames_MAILABS_train_w2v_base.pkl')
    
    df_all_frames = build_df_all_frames(df_t_train, 'phone', model_path="hf_models/facebook/wav2vec2-large-xlsr-53")
    df_all_frames.to_pickle('df_all_frames_MAILABS_train_w2v_xlsr_no_ft.pkl')

    df_t_train_all = load_dataset_MAILABS(['en_US', 'en_UK', 'es_ES', 'fr_FR'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', phone_set='MFA_IPA')
    df_all_frames = build_df_all_frames(df_t_train_all, 'phone', model_path="hf_models/facebook/wav2vec2-large-xlsr-53")
    df_all_frames.to_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa_w2v_xlsr_no_ft.pkl')

    from src.wav2vec2_utils import plot_reduction

    df_all_frames['average_vector']=df_all_frames['vector']
    plot_reduction(df_all_frames, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_space_phoneme', legend_label='phoneme')
    plot_reduction(df_all_frames, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_space_genre', legend_label='genre')
    plot_reduction(df_all_frames, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_space_speaker', legend_label='speaker')
    plot_reduction(df_all_frames, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_space_language_code', legend_label='language_code')


    records=[]
    for fn in df_all_frames.filename.unique():
        df_all_frames_fn=df_all_frames[df_all_frames.filename==fn]

        keys=['language_code', 'genre', 'speaker', 'filename']
        d={}
        for k in keys:
            d[k]=df_all_frames_fn.iloc[0][k]

        d['average_vector']=df_all_frames_fn.vector.mean()
        records.append(d)
    df_all_sentences=pd.DataFrame.from_records(records)
    
    # plot_reduction(df_all_sentences, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_sentence_space_phoneme', legend_label='phoneme')
    plot_reduction(df_all_sentences, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_sentence_space_genre', legend_label='genre')
    plot_reduction(df_all_sentences, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_sentence_space_speaker', legend_label='speaker')
    plot_reduction(df_all_sentences, reduction_technique='umap', base_name='plots/w2v_xlsr_no_ft_sentence_space_language_code', legend_label='language_code')



    # # Basis model Wav2Vec2ForFramePrediction, in CMU phoneme set, PCA reduction at 95% variance, and a 10-NN classifier
    # df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    # X, y = df_all_frames_to_X_y(df_all_frames)
    # model = Wav2Vec2ForFramePrediction(0.95,'cmu', reducer="pca")
    # model.fit(X, y)

    # Basis model Wav2Vec2ForFramePrediction, 10-NN classifier weighted with distances, US frames
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_US_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w_US')

    # Basis model Wav2Vec2ForFramePrediction, but a 10-NN classifier weighted with distances
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_mailabs_pca_0.95_knn_10_w')

    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.load(name='model_mailabs_pca_0.95_knn_10_w')
    
    # Basis model Wav2Vec2ForFramePrediction with 10-NN classifier weighted with distances, with w2v2 base model instead of wlsr espeak ft
    df_all_frames=pd.read_pickle('df_all_frames_MAILABS_train_w2v_base.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)
    model = Wav2Vec2ForFramePrediction('cmu', w2v2_model_path="hf_models/facebook/wav2vec2-base-960h",  frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.fit(X, y)
    model.save(name='model_w2v_base_mailabs_pca_0.95_knn_10_w')

    model = Wav2Vec2ForFramePrediction('cmu', frame_classifier=KNeighborsClassifier(10, weights='distance'))
    model.load(name='model_w2v_base_mailabs_pca_0.95_knn_10_w')


    
    
    
    df_all_instances=pd.read_pickle('df_all_phonemes_instances_MAILABS_train.pkl')
    df_all_instances['sum']=df_all_instances.iloc[:,1].apply(lambda r: r.sum())
    df_all_instances=df_all_instances.dropna()
    import numpy as np
    X_train=np.array(list(df_all_instances.iloc[:,1].values))    
    # id_to_p={i:p for i,p in enumerate(cmu_alphabet+['[SIL]'])}
    # p_to_id={p:i for i,p in enumerate(cmu_alphabet+['[SIL]'])}
    # y_train=df_all_instances.phoneme.apply(lambda r: p_to_id[r]).values
    y_train=list(df_all_instances.phoneme.values)




    
    # default_model_ipa = Wav2Vec2ForFramePrediction('ipa')
    # default_model_ipa.load(name='model_mailabs_pca_95_knn_10_w_ipa')

    # # phoneme predictions on a single audio sample with forced alignment
    # pred = default_model_ipa.predict_with_timings(data.s.iloc[0], data.phones.iloc[0])
    # prob_matrix = default_model_ipa.predict_phone_prob_matrix(data.s.iloc[0], 16000)

    
    # # phoneme predictions on a single audio sample with forced alignment
    # pred2 = model2.predict_with_timings(data.s.iloc[0], data.phones.iloc[0])
    # prob_matrix2 = model2.predict_phone_prob_matrix(data.s.iloc[0], 16000)

    # from tqdm import tqdm
    # preds=[model.predict_with_timings(r.s, r.phones) for i,r in tqdm(data.iterrows())]
    # preds2=[model2.predict_with_timings(r.s, r.phones) for i,r in tqdm(data.iterrows())]
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