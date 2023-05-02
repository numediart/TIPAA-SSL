import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import itertools
import json
from operator import itemgetter
from src.text_processing import remove_stress_annots, group_consecutive_duplicates

from src.pronunciation_dictionaries import cmu_alphabet, ipa_alphabet

# https://stackoverflow.com/questions/51269456/pandas-delete-consecutive-duplicates-but-keep-the-first-and-last-value
keep_first_last=lambda s: s[~((s == s.shift(1)) & (s == s.shift(-1)))]

# get the blocks of consecutive identical rows in cols
get_blocks = lambda a,cols: a.loc[(a[cols].shift() == a[cols]).any(axis=1)|(a[cols].shift(-1) == a[cols]).any(axis=1)]


class dtw_forced_aligner:
    def __init__(self, phone_type):
        # self.label_encoder = LabelEncoder()
        if phone_type == 'cmu':
            self.alphabet = cmu_alphabet
        elif phone_type == 'ipa':
            self.alphabet = ipa_alphabet
        # self.label_encoder.fit([phon for phon in self.alphabet])
        self.id_to_p={i:p for i,p in enumerate(self.alphabet+['[SIL]'])}
        self.p_to_id={p:i for i,p in enumerate(self.alphabet+['[SIL]'])}

    def labelize_phonemes(self, phonemes):
        # return np.array(self.label_encoder.transform([phon for phon in phonemes]))
        return np.array([self.p_to_id[el] for el in phonemes])

    # get the columns from the probability matrix which correspond to non-silent frames
    def get_phone_prob_matrix_nonsil(self, phone_prob_matrix):
        phone_prob_matrix = [l for l in phone_prob_matrix]

        # the sum is 1, except if it's a silence, because the column corresponding to that token was removed
        # def condition(vect): return sum(vect)<0.2
        def condition(vect): return vect[-1]>0.8
        a = np.array(phone_prob_matrix)

        silence_frames_idx = [idx for idx, element in enumerate(a) if condition(element)]
        non_silence_frames_idx = [idx for idx, element in enumerate(a) if not condition(element)]
        phone_prob_matrix_nonsil = a[non_silence_frames_idx]
        return phone_prob_matrix_nonsil, silence_frames_idx, non_silence_frames_idx

    # from cost_non_sil and target_phonemes, get the most likely path (forced alignment)
    def get_forced_alignment(self, phone_prob_matrix_nonsil, target_phonemes):
        # target_phonemes = [x[0] for x in groupby(target_phonemes)]
        target_phonemes = remove_stress_annots(target_phonemes)
        target_labels = self.labelize_phonemes(target_phonemes)

        # one_hot_matrix=np.zeros((len(self.id_to_p), len(target_phonemes)))
        # for i in range(len(target_labels)):
        #     one_hot_matrix[target_labels[i],i]=1
        # (np.dot(phone_prob_matrix_nonsil,one_hot_matrix)==phone_prob_matrix_nonsil[:,list(target_labels)]).all()

        # Dynamic Time Warping
        # with 39 phonemes + silence token, phone_prob_matrix_nonsil is of shape T x 40. let's call L the length of the phoneme sequence. 
        # phone_prob_matrix_nonsil[:,list(target_labels)]  is the juxtaposition (horizontal stack) of columns coming from the prob matrix corresponding to each phoneme of the sequence, of shape T x L.
        # for each phoneme of the sequence, we extract a number for each time step that is a similarity measure between the frame proba and the phoneme, i.e. a dot product divided by both their norms. As here we apply that on probability vectors, it is equivalent to a dot product
        # if it is a one-hot, a dot product is equivalent as just taking the element with that index from the prob vector.

        D, wp = librosa.sequence.dtw(C=-phone_prob_matrix_nonsil[:,list(target_labels)], step_sizes_sigma=np.array([[1, 1], [1, 0]]))
        # getting phonemes' labels from forced alignement
        aligned_phones_labels = []
        for index in [i[1] for i in wp]:
            aligned_phones_labels.insert(0, target_labels[index])
        # using the label encoder to find the phoneme
        # aligned_phones = list(self.label_encoder.inverse_transform(aligned_phones_labels))
        aligned_phones = [self.id_to_p[el] for el in aligned_phones_labels]
        
        return aligned_phones

    # forced alignment but with all the audio sample's frames
    def get_alignment_with_silence(self, aligned_phones, silence_frames_idx, non_silence_frames_idx):
        alignment_with_silence=np.array(["     "]*(len(silence_frames_idx)+len(non_silence_frames_idx)))
        alignment_with_silence[silence_frames_idx] = "[SIL]"
        alignment_with_silence[non_silence_frames_idx] = aligned_phones
        return alignment_with_silence

    def predict(self, aligned_phones, phone_prob_matrix_nonsil, target_phonemes):
        # compute the mean of every phoneme alignment
        aligned_preds = list(zip(aligned_phones, phone_prob_matrix_nonsil))
        grouped_aligned_preds = [list(v) for _,v in itertools.groupby(aligned_preds,itemgetter(0))]

        if len(target_phonemes) > len(grouped_aligned_preds):
            for i in range(len(target_phonemes)-1):
                if target_phonemes[i] == target_phonemes[i+1]:
                    index = int(len(grouped_aligned_preds[i])/2)
                    x_1 = grouped_aligned_preds[i][:index]
                    x_2 = grouped_aligned_preds[i][index:]
                    grouped_aligned_preds[i] = x_2
                    grouped_aligned_preds.insert(i, x_1)

        proba_means = []
        for phon in grouped_aligned_preds:
            proba_means.append(np.mean([l[1] for l in phon], axis=0))
            # proba_means.append(np.median([l[1] for l in phon], axis=0))

        # predicted_phones = [self.label_encoder.inverse_transform([np.argmax(i)])[0] for i in proba_means]
        predicted_phones = [self.id_to_p[np.argmax(i)] for i in proba_means]
        
        return predicted_phones, proba_means

    def get_df_segmented(self, alignment_with_silence, predicted_phones, phones, proba_means, fs=16000, time_per_output=0.02):

        start_idx = []
        end_idx = []
        for i in range(len(alignment_with_silence)):
            start_idx.append(i)
            end_idx.append(i+1)

        ph_with_timings = [i for i in list(zip(alignment_with_silence, start_idx, end_idx)) if i[0] != '[SIL]']
        grouped = [list(v) for _,v in itertools.groupby(ph_with_timings,itemgetter(0))]

        if len(grouped)<len(phones):
            for i in range(len(phones)-1):
                if phones[i] == phones[i+1]:
                    grouped.insert(i, grouped[i])

        timings = [(elem[0][0], elem[0][1], elem[-1][2]) for elem in grouped]
        timings_df=pd.DataFrame(timings)
        # df_segmented['phones'] = remove_stress_annots(phones)

        df_segmented = pd.DataFrame()
        df_segmented[['phones', 'start_idx', 'end_idx']]=timings_df
        df_segmented['pred_phones_audio'] = predicted_phones
        df_segmented['proba_means'] = proba_means
        
        # df_segmented['GT_proba']=df_segmented.apply(lambda r: r.proba_means[p_to_id(r.phones)], axis=1)
        # df_segmented['pred_proba']=df_segmented.apply(lambda r: r.proba_means[p_to_id(r.pred_phones_audio)], axis=1)
        df_segmented['GT_proba'] = [df_segmented.proba_means[i][j] for i,j in zip(range(len(df_segmented)), self.labelize_phonemes(df_segmented.phones))]
        df_segmented['pred_proba'] = [df_segmented.proba_means[i][j] for i,j in zip(range(len(df_segmented)), self.labelize_phonemes(df_segmented.pred_phones_audio))]
        df_segmented['start']=df_segmented['start_idx']*time_per_output
        df_segmented['end']=df_segmented['end_idx']*time_per_output

        def collapse_consecutive_duplicates(df):
            df=df.reset_index(drop=True)
            blocks=[]

            groups=df.groupby([(df.phones != df.phones.shift()).cumsum()])
            for i, g in groups:
                r=g.iloc[0]
                r.end=g.iloc[-1].end
                blocks.append(r.to_dict())
            return pd.DataFrame.from_records(blocks)

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
        

        df_segmented2=df_segmented[df_segmented.phones != '[SIL]']
        # collapse_consecutive_duplicates(df_segmented)
        df_segmented2=collapse_consecutive_duplicates(df_segmented2)
        if len(df_segmented)>0:
            df_segmented=divide_consecutive_duplicates(df_segmented2, remove_stress_annots(phones))

        return df_segmented
    
    def probas_to_df_segmented(self, phone_prob_matrix, target_phonemes, fs=16000, time_per_output=0.02):
        phone_prob_matrix_nonsil, silence_frames_idx, non_silence_frames_idx = self.get_phone_prob_matrix_nonsil(phone_prob_matrix)
        aligned_phones = self.get_forced_alignment(phone_prob_matrix_nonsil, target_phonemes)
        if silence_frames_idx:
            alignment_with_silence = self.get_alignment_with_silence(aligned_phones, silence_frames_idx, non_silence_frames_idx)
        else:
            alignment_with_silence = aligned_phones
        predicted_phones, proba_means = self.predict(aligned_phones, phone_prob_matrix_nonsil, target_phonemes)
        df_segmented = self.get_df_segmented(alignment_with_silence, predicted_phones, target_phonemes, proba_means, fs=fs, time_per_output=time_per_output)

        return df_segmented