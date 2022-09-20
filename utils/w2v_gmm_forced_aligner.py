import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import itertools
from itertools import groupby
import json
from operator import itemgetter
import cmudict
from utils.text_processing import remove_stress_annots

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

    def get_cost_non_sil(self, phone_prob_matrix): 
        phone_prob_matrix = [l for l in phone_prob_matrix]

        def condition(vect): return sum(vect)<0.2
        a = np.array(phone_prob_matrix)

        silence_frames_idx = [idx for idx, element in enumerate(a) if condition(element)]
        non_silence_frames_idx = [idx for idx, element in enumerate(a) if not condition(element)]
        cost_nonsil = a[non_silence_frames_idx]
        return cost_nonsil, silence_frames_idx, non_silence_frames_idx

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

    def get_alignment_with_silence(self, aligned_phones, silence_frames_idx, non_silence_frames_idx):
        last_idx = max(silence_frames_idx[-1], non_silence_frames_idx[-1])
        alignment_with_silence = np.array(["     " for i in range(last_idx+1)])

        alignment_with_silence[silence_frames_idx] = "[SIL]"
        alignment_with_silence[non_silence_frames_idx] = aligned_phones

        alignment_with_silence[0] = "[SIL]"
        alignment_with_silence[-1] = "[SIL]"
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

        # if len(target_phonemes) > len(grouped_aligned_preds):
        #     target_phonemes = target_phonemes[:len(grouped_aligned_preds)]
        # else if len(target_phonemes) < len(grouped_aligned_preds):
        #     grouped_aligned_preds = grouped_aligned_preds[:len(target_phonemes)]

        probs_means = []
        for phon in grouped_aligned_preds:
            probs_means.append(np.median([l[1] for l in phon], axis=0))

        predicted_phones = [self.label_encoder.inverse_transform([np.argmax(i)])[0] for i in probs_means]

        # if len(predicted_phones)>len(target_phonemes):
        #     print("I'm here")
        #     predicted_phones = predicted_phones[:len(target_phonemes)]
        # else:
        #     print("I'm not here")
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
        timings = [(elem[0][0], elem[0][1], elem[-1][2]) for elem in grouped]
        df_segmented['phones'] = [elem[0] for elem in timings]
        df_segmented['start'] = [elem[1] for elem in timings]
        df_segmented['end'] = [elem[2] for elem in timings]

        if len(predicted_phones)>len(df_segmented):
            # here make sure the index is a range. I will insert using .loc at i+0.5, then reset index every time
            # https://stackoverflow.com/questions/15888648/is-it-possible-to-insert-a-row-at-an-arbitrary-position-in-a-dataframe-using-pan?rq=1
            df_segmented=df_segmented.reset_index(drop=True)
            for i in range(len(phones)-1):
                if phones[i]==phones[i+1]:
                    df_segmented.loc[i+0.5]=df_segmented.loc[i]
                    # df_segmented.loc[i+0.5].start=np.average(df_segmented.loc[i].start, df_segmented.loc[i].end)
                    # df_segmented.loc[i].end=np.average(df_segmented.loc[i].start, df_segmented.loc[i].end)
                    df_segmented=df_segmented.reset_index(drop=True)
                    
        df_segmented['pred_phones_audio'] = predicted_phones
        df_segmented['probs_means'] = probs_means
        df_segmented['GT_proba'] = [df_segmented.probs_means[i][j] for i,j in zip(range(len(df_segmented)), self.labelize_phonemes(df_segmented.phones))]
        
        return df_segmented