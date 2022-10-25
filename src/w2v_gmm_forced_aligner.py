import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import itertools
from itertools import groupby
import json
from operator import itemgetter, xor
import cmudict
from src.text_processing import remove_stress_annots, group_consecutive_duplicates

# https://stackoverflow.com/questions/51269456/pandas-delete-consecutive-duplicates-but-keep-the-first-and-last-value
keep_first_last=lambda s: s[~((s == s.shift(1)) & (s == s.shift(-1)))]

# get the blocks of consecutive identical rows in cols
get_blocks = lambda a,cols: a.loc[(a[cols].shift() == a[cols]).any(axis=1)|(a[cols].shift(-1) == a[cols]).any(axis=1)]

global cmu_alphabet
cmu_alphabet = [el[0] for el in cmudict.phones()]

global ipa_alphabet
with open('data/mfa_phones.json', 'r') as openfile: ipa_alphabet = json.load(openfile)

class w2v_gmm_forced_aligner:
    def __init__(self, phone_type):
        self.label_encoder = LabelEncoder()
        if phone_type == 'cmu':
            self.alphabet = cmu_alphabet
        elif phone_type == 'ipa':
            self.alphabet = ipa_alphabet
        self.label_encoder.fit([phon for phon in self.alphabet])

    def labelize_phonemes(self, phonemes):
        return np.array(self.label_encoder.transform([phon for phon in phonemes]))

    # get the columns from the probability matrix which correspond to non-silent frames
    def get_cost_non_sil(self, phone_prob_matrix):
        phone_prob_matrix = [l for l in phone_prob_matrix]

        def condition(vect): return sum(vect)<0.2
        a = np.array(phone_prob_matrix)

        silence_frames_idx = [idx for idx, element in enumerate(a) if condition(element)]
        non_silence_frames_idx = [idx for idx, element in enumerate(a) if not condition(element)]
        cost_nonsil = a[non_silence_frames_idx]
        return cost_nonsil, silence_frames_idx, non_silence_frames_idx

    # from cost_non_sil and target_phonemes, get the most likely path (forced alignment)
    def get_forced_alignment(self, cost_nonsil, target_phonemes):
        # target_phonemes = [x[0] for x in groupby(target_phonemes)]
        target_phonemes = remove_stress_annots(target_phonemes)
        target_labels = self.labelize_phonemes(target_phonemes)
        # Dynamic Time Warping
        D, wp = librosa.sequence.dtw(C=-cost_nonsil[:,list(target_labels)], step_sizes_sigma=np.array([[1, 1], [1, 0]]))
        # getting phonemes' labels from forced alignement
        aligned_phones_labels = []
        for index in [i[1] for i in wp]:
            aligned_phones_labels.insert(0, target_labels[index])
        # using the label encoder to find the phoneme
        aligned_phones = list(self.label_encoder.inverse_transform(aligned_phones_labels))
        return aligned_phones

    # forced alignment but with all the audio sample's frames
    def get_alignment_with_silence(self, aligned_phones, silence_frames_idx, non_silence_frames_idx):
        alignment_with_silence=np.array(["     "]*(len(silence_frames_idx)+len(non_silence_frames_idx)))
        alignment_with_silence[silence_frames_idx] = "[SIL]"
        alignment_with_silence[non_silence_frames_idx] = aligned_phones

        # alignment_with_silence[0] = "[SIL]"
        # alignment_with_silence[-1] = "[SIL]"
        return alignment_with_silence

    def predict(self, aligned_phones, cost_nonsil, target_phonemes):
        # compute the mean of every phoneme alignment
        aligned_preds = list(zip(aligned_phones, cost_nonsil))
        grouped_aligned_preds = [list(v) for _,v in itertools.groupby(aligned_preds,itemgetter(0))]

        if len(target_phonemes) > len(grouped_aligned_preds):
            for i in range(len(target_phonemes)-1):
                if target_phonemes[i] == target_phonemes[i+1]:
                    index = int(len(grouped_aligned_preds[i])/2)
                    x_1 = grouped_aligned_preds[i][:index]
                    x_2 = grouped_aligned_preds[i][index:]
                    grouped_aligned_preds[i] = x_2
                    grouped_aligned_preds.insert(i, x_1)

        probs_means = []
        for phon in grouped_aligned_preds:
            probs_means.append(np.median([l[1] for l in phon], axis=0))

        predicted_phones = [self.label_encoder.inverse_transform([np.argmax(i)])[0] for i in probs_means]
        return predicted_phones, probs_means

    def get_df_segmented(self, alignment_with_silence, predicted_phones, phones, probs_means, fs=16000, time_per_output=0.02):
        df_segmented = pd.DataFrame(columns=['phones', 'pred_phones_audio', 'start', 'end', 'probs_means', 'GT_proba'])
        start = []
        end = []

        for i in range(len(alignment_with_silence)):
            start.append(i*time_per_output)
            end.append((i+1)*time_per_output)

        ph_with_timings = [i for i in list(zip(alignment_with_silence, start, end)) if i[0] != '[SIL]']
        grouped = [list(v) for _,v in itertools.groupby(ph_with_timings,itemgetter(0))]

        if len(grouped)<len(phones):
            for i in range(len(phones)-1):
                if phones[i] == phones[i+1]:
                    grouped.insert(i, grouped[i])

        timings = [(elem[0][0], elem[0][1], elem[-1][2]) for elem in grouped]
        timings_df=pd.DataFrame(timings)
        # df_segmented['phones'] = remove_stress_annots(phones)
        df_segmented[['phones', 'start', 'end']]=timings_df
        df_segmented['pred_phones_audio'] = predicted_phones
        df_segmented['probs_means'] = probs_means
        df_segmented['GT_proba'] = [df_segmented.probs_means[i][j] for i,j in zip(range(len(df_segmented)), self.labelize_phonemes(df_segmented.phones))]
        
        def collapse_consecutive_duplicates(df):
            df=df.reset_index(drop=True)
            blocks=[]

            groups=df.groupby([(df.phones != df.phones.shift()).cumsum()])
            for i, g in groups:#print('---');      print (g);         print (g.phones.tolist());r=g.iloc[0];   r.end=g.iloc[-1].end;   
                r=g.iloc[0]
                r.end=g.iloc[-1].end
                blocks.append(r.to_dict())
            return pd.DataFrame.from_records(blocks)

        def divide_consecutive_duplicates(p_df, phone_list):
            grouped_phone_list=group_consecutive_duplicates(phone_list)
            duplicate_indexes=[i for i,el in enumerate(grouped_phone_list) if el[-1]>1]
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
        

        # df_segmented['start'] = [elem[1] for elem in timings]
        # df_segmented['end'] = [elem[2] for elem in timings]

        # if len(predicted_phones)>len(df_segmented):
        #     # here make sure the index is a range. I will insert using .loc at i+0.5, then reset index every time
        #     # https://stackoverflow.com/questions/15888648/is-it-possible-to-insert-a-row-at-an-arbitrary-position-in-a-dataframe-using-pan?rq=1
        #     df_segmented=df_segmented.reset_index(drop=True)
        #     p_unstressed = remove_stress_annots(phones)
        #     for i in range(len(p_unstressed)-1):
        #         if p_unstressed[i]==p_unstressed[i+1]:
        #             df_segmented.loc[i+0.5]=df_segmented.loc[i]
        #             df_segmented = df_segmented.sort_index()
        #             df_segmented=df_segmented.reset_index(drop=True)

        # if len(predicted_phones)>len(df_segmented):
        #     df_segmented=df_segmented.reset_index(drop=True)
        #     for i in range(len(predicted_phones)-1):
        #         if predicted_phones[i]==predicted_phones[i+1]:
        #             df_segmented.loc[i+0.5]=df_segmented.loc[i]
        #             df_segmented = df_segmented.sort_index()
        #             df_segmented=df_segmented.reset_index(drop=True)

        return df_segmented
    
    
    def probas_to_df_segmented(self, phone_prob_matrix, target_phonemes, fs=16000):
        cost_nonsil, silence_frames_idx, non_silence_frames_idx = self.get_cost_non_sil(phone_prob_matrix)
        aligned_phones = self.get_forced_alignment(cost_nonsil, target_phonemes)
        if silence_frames_idx:
            alignment_with_silence = self.get_alignment_with_silence(aligned_phones, silence_frames_idx, non_silence_frames_idx)
        else:
            alignment_with_silence = aligned_phones
        predicted_phones, probs_means = self.predict(aligned_phones, cost_nonsil, target_phonemes)
        df_segmented = self.get_df_segmented(alignment_with_silence, predicted_phones, target_phonemes, probs_means, fs=fs, time_per_output=0.02)

        return df_segmented