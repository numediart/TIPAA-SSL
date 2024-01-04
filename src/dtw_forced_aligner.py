import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import itertools
from operator import itemgetter
from src.text_processing import remove_stress_annots, group_consecutive_duplicates

# https://stackoverflow.com/questions/51269456/pandas-delete-consecutive-duplicates-but-keep-the-first-and-last-value
keep_first_last = lambda s: s[~((s == s.shift(1)) & (s == s.shift(-1)))]

# get the blocks of consecutive identical rows in cols
get_blocks = lambda a, cols: a.loc[
    (a[cols].shift() == a[cols]).any(axis=1) | (a[cols].shift(-1) == a[cols]).any(axis=1)
]


class dtw_forced_aligner:
    """
    The `dtw_forced_aligner` class is designed to perform forced alignment using Dynamic Time Warping (DTW) to align phoneme sequences
    with corresponding probability matrices.
    It provides methods for various tasks including converting phonemes to their corresponding ids,
    getting the most likely alignment path, aligning frames with silence, predicting phonemes from aligned frames,
    converting the alignment information into a segmented DataFrame, retrieving the phone probability matrix for non-silent frames,
    and converting probability matrices to segmented DataFrames.
    The class offers flexibility by allowing the user to specify an alphabet of phonemes and choose between mean or max pooling for collapsing probability vectors.
    Overall, the `dtw_forced_aligner` class provides a comprehensive set of tools for performing forced alignment and analyzing the results.
    """

    def __init__(self, alphabet, collapse_method='mean'):
        """
        Initialize the dtw_forced_aligner class.

        Args:
            alphabet (list): The list of phonemes in the alphabet.
            collapse_method (str, optional): The method used to collapse probability vectors. Defaults to 'mean'.
        """
        self.alphabet = alphabet
        self.id_to_p = {i: p for i, p in enumerate(self.alphabet + ['[SIL]'])}
        self.p_to_id = {p: i for i, p in enumerate(self.alphabet + ['[SIL]'])}
        self.collapse_method = collapse_method

    def labelize_phonemes(self, phonemes):
        """
        Convert a list of phonemes to their corresponding ids.

        Args:
            phonemes (list): The list of phonemes.

        Returns:
            np.array[int]: The array of phoneme ids.
        """
        # return np.array(self.label_encoder.transform([phon for phon in phonemes]))
        return np.array([self.p_to_id[el] for el in phonemes])

    # from phone_prob_matrix_nonsil and target_phonemes, get the most likely path (forced alignment)
    def get_forced_alignment(self, phone_prob_matrix_nonsil, target_phonemes):
        """
        Perform forced alignment to get the most likely path using Dynamic Time Warping.

        With N phonemes + silence token, phone_prob_matrix_nonsil is of shape T x (N+1).
        Let's call L the length of the phoneme sequence.
        phone_prob_matrix_nonsil[:,list(target_labels)]  is the juxtaposition
        (horizontal stack) of columns coming from the prob matrix corresponding to each phoneme of the sequence, of shape T x L.
        For each phoneme of the sequence, we extract a number for each time step that is a
        similarity measure between the frame proba and the phoneme, i.e. a dot product
        divided by both their norms. As here we apply that on probability vectors, it is equivalent to a dot product
        if it is a one-hot, a dot product is equivalent as just taking the element with that index from the prob vector.

        Args:
            phone_prob_matrix_nonsil (np.array[float]): The phone probability matrix for
                non-silent frames. It is of shape T x (N+1), where T is
                the number of frames and N is the number of phonemes.
            target_phonemes (list): The list of target phonemes.

        Returns:
            Tuple[bool, List[str], float]: A tuple containing a boolean indicating the
                success of forced alignment, the list of aligned phonemes, the DTW alignment cost
        """

        # one_hot_matrix=np.zeros((len(self.id_to_p), len(target_phonemes)))
        # for i in range(len(target_labels)):
        #     one_hot_matrix[target_labels[i],i]=1
        # (np.dot(phone_prob_matrix_nonsil,one_hot_matrix)==phone_prob_matrix_nonsil[:,list(target_labels)]).all()

        from librosa.util.exceptions import ParameterError

        target_labels = self.labelize_phonemes(target_phonemes)
        try:
            D, wp = librosa.sequence.dtw(
                C=-phone_prob_matrix_nonsil[:, list(target_labels)],
                step_sizes_sigma=np.array([[1, 1], [1, 0]]),
            )

            # getting phonemes' labels from forced alignement
            aligned_phones_labels = []
            for index in [i[1] for i in wp]:
                aligned_phones_labels.insert(0, target_labels[index])
            # using the label encoder to find the phoneme
            aligned_phones = [self.id_to_p[el] for el in aligned_phones_labels]
            return True, aligned_phones, D[-1, -1]
        except ParameterError:
            # print('DTW failed, most probably the audio is too far from what is expected.')
            aligned_phones = ["[SIL]"] * len(phone_prob_matrix_nonsil)

            return False, aligned_phones, None

    # forced alignment but with all the audio sample's frames
    def get_alignment_with_silence(
        self, aligned_phones, silence_frames_idx, non_silence_frames_idx
    ):
        """
        Get the alignment with silence frames.

        Args:
            aligned_phones (list): The list of aligned phonemes.
            silence_frames_idx (list): The list of indices of silence frames.
            non_silence_frames_idx (list): The list of indices of non-silence frames.

        Returns:
            np.array[str]: The alignment with silence frames.
        """
        alignment_with_silence = np.array(
            ["     "] * (len(silence_frames_idx) + len(non_silence_frames_idx))
        )
        alignment_with_silence[silence_frames_idx] = "[SIL]"
        alignment_with_silence[non_silence_frames_idx] = aligned_phones
        return alignment_with_silence

    # compute a collapsed proba vector of every phoneme alignment
    def predict(self, aligned_phones, phone_prob_matrix_nonsil, target_phonemes):
        """
        Compute the collapsed probability vector of every phoneme alignment.

        Given a set of frames matched to a target phone through the DTW,
        collapse the phone probabilities of the frames to a single probability vector.
        The collapse_method is 'mean' or 'max', ie we either use mean pooling or max
        pooling across time.

        The predicted phones are the phones with the
        highest probability in the collapsed probability vectors.

        Args:
            aligned_phones (list): The list of aligned phonemes.
            phone_prob_matrix_nonsil (np.array[float]): The phone probability matrix for non-silent frames. It is of shape T x (N+1), where T is the number of frames and N is the number of phonemes.
            target_phonemes (list): The list of target phonemes.


        Returns:
            Tuple[List[str], List[np.array[float]]]: A tuple containing the list of predicted phones and the list of collapsed probability vectors.
        """
        aligned_preds = list(zip(aligned_phones, phone_prob_matrix_nonsil))
        grouped_aligned_preds = [
            list(v) for _, v in itertools.groupby(aligned_preds, itemgetter(0))
        ]

        if len(target_phonemes) > len(grouped_aligned_preds):
            for i in range(len(target_phonemes) - 1):
                if target_phonemes[i] == target_phonemes[i + 1]:
                    index = int(len(grouped_aligned_preds[i]) / 2)
                    x_1 = grouped_aligned_preds[i][:index]
                    x_2 = grouped_aligned_preds[i][index:]
                    grouped_aligned_preds[i] = x_2
                    grouped_aligned_preds.insert(i, x_1)

        collapsed_proba_vectors = []
        for phon in grouped_aligned_preds:
            if self.collapse_method == 'mean':
                collapsed_proba_vectors.append(np.mean([l[1] for l in phon], axis=0))
            elif self.collapse_method == 'max':
                collapsed_proba_vectors.append(np.max([l[1] for l in phon], axis=0))
            else:
                raise Exception(
                    "collapse_method in src.dtw_forced_aligner.predict() should be 'mean' or 'max'"
                )
        predicted_phones = [self.id_to_p[np.argmax(i)] for i in collapsed_proba_vectors]

        return predicted_phones, collapsed_proba_vectors

    def get_df_segmented(
        self,
        alignment_with_silence,
        predicted_phones,
        phones,
        proba_means,
        time_per_output=0.02,
    ):
        """
        Convert the alignment information into a segmented DataFrame.

        Args:
            alignment_with_silence (np.array[str]): The alignment with silence frames.
            predicted_phones (list): The list of predicted phones.
            phones (list): The list of target phones.
            proba_means (list): The list of probability means.
            time_per_output (float, optional): The time per output. Defaults to 0.02.

        Returns:
            pd.DataFrame: The segmented DataFrame.
        """

        # print(alignment_with_silence)
        # print(phones)

        start_idx = []
        end_idx = []
        for i in range(len(alignment_with_silence)):
            start_idx.append(i)
            end_idx.append(i + 1)

        ph_with_timings = [
            i
            for i in list(zip(alignment_with_silence, start_idx, end_idx))
            if i[0] != '[SIL]'
        ]
        grouped = [list(v) for _, v in itertools.groupby(ph_with_timings, itemgetter(0))]

        if len(grouped) < len(phones):
            for i in range(len(phones) - 1):
                if phones[i] == phones[i + 1]:
                    grouped.insert(i, grouped[i])

        timings = [(elem[0][0], elem[0][1], elem[-1][2]) for elem in grouped]
        timings_df = pd.DataFrame(timings)

        # print(timings_df)

        df_segmented = pd.DataFrame(columns=['phones', 'start_idx', 'end_idx'])

        df_segmented[['phones', 'start_idx', 'end_idx']] = timings_df
        df_segmented['pred_phones_audio'] = predicted_phones
        df_segmented['proba_means'] = proba_means
        df_segmented['GT_proba'] = [
            df_segmented.proba_means[i][j]
            for i, j in zip(
                range(len(df_segmented)), self.labelize_phonemes(df_segmented.phones)
            )
        ]
        df_segmented['pred_proba'] = [
            df_segmented.proba_means[i][j]
            for i, j in zip(
                range(len(df_segmented)),
                self.labelize_phonemes(df_segmented.pred_phones_audio),
            )
        ]
        df_segmented['start'] = df_segmented['start_idx'] * time_per_output
        df_segmented['end'] = df_segmented['end_idx'] * time_per_output

        def collapse_consecutive_duplicates(df):
            df = df.reset_index(drop=True)

            groups = df.groupby([(df.phones != df.phones.shift()).cumsum()])
            new_df = groups.first().reset_index(drop=True)
            new_df["end"] = groups.last().end.reset_index(drop=True)
            return new_df

        def divide_consecutive_duplicates(p_df, phone_list):
            grouped_phone_list = group_consecutive_duplicates(phone_list)
            n_times = [el[-1] for i, el in enumerate(grouped_phone_list)]

            p_df['n_times'] = n_times
            p_df_full = p_df.loc[p_df.index.repeat(p_df.n_times)]
            p_df_full = p_df_full.reset_index()

            blocks = get_blocks(p_df_full, ['phones'])

            block_starts_ends = keep_first_last(blocks.phones)
            # keep firsts and lasts (thus only when there is two consecutive phonemes)
            starts = block_starts_ends.loc[
                block_starts_ends.shift(-1) == block_starts_ends
            ].index.tolist()
            ends = block_starts_ends.loc[
                block_starts_ends.shift(+1) == block_starts_ends
            ].index.tolist()

            for start_idx, end_idx in zip(starts, ends):
                select = p_df_full.iloc[start_idx : end_idx + 1]
                start = select.start.iloc[0]
                end = select.end.iloc[-1]
                interval = (end - start) / len(select)

                steps = start + np.cumsum([interval] * (len(select) - 1))

                p_df_full.iloc[start_idx + 1 : end_idx + 1].start = steps
                p_df_full.iloc[start_idx:end_idx].end = steps
            p_df_full[['start', 'end']] = p_df_full[['start', 'end']].round(2)
            return p_df_full

        df_segmented2 = df_segmented[df_segmented.phones != '[SIL]']
        df_segmented2 = collapse_consecutive_duplicates(df_segmented2)
        if len(df_segmented) > 0:
            df_segmented = divide_consecutive_duplicates(
                df_segmented2, remove_stress_annots(phones)
            )

        return df_segmented

    def get_phone_prob_matrix_nonsil(self, phone_prob_matrix):
        """
        Get the columns from the probability matrix which correspond to silent and non-silent frames.

        Args:
            phone_prob_matrix (np.array[float]): The phone probability matrix possibly contanining silent frames. It is of shape T x (N+1), where T is the number of frames and N is the number of phonemes.

        Returns:
            Tuple[np.array[float], list, list]: A tuple containing the phone probability matrix of non-silent frames, the list of indices of silence frames, and the list of indices of non-silence frames.
        """

        # Here we want to detect silence frames. The silence token is at the last index. Either we can threshold it, or maybe better: check if it's the max posterior probability
        # def condition(vect): return vect[-1]>0.8
        def condition(vect):
            return vect[-1] == max(vect)

        a = np.array(phone_prob_matrix)

        silence_frames_idx = [idx for idx, element in enumerate(a) if condition(element)]
        non_silence_frames_idx = [
            idx for idx, element in enumerate(a) if not condition(element)
        ]
        phone_prob_matrix_nonsil = a[non_silence_frames_idx]
        return phone_prob_matrix_nonsil, silence_frames_idx, non_silence_frames_idx

    def probas_to_df_segmented(
        self,
        phone_prob_matrix: np.ndarray,
        target_phonemes: list[str],
        time_per_output: float = 0.02,
    ) -> tuple[pd.DataFrame, float | None]:
        """
        Convert probability matrix to a segmented DataFrame.

        Parameters
        ----------
        phone_prob_matrix (np.array[float]):
            The phone probability matrix possibly contanining silent frames.
            It is of shape T x (N+1), where T is the number of frames and N is the number of phones.
            (N+1) because we include the silent sound.
        target_phonemes (list):
            The list of target phonemes.
        time_per_output (float, optional):
            The time between frames in seconds. Defaults to 0.02.

        Returns:
        --------
        pd.DataFrame:
            The segmented DataFrame.
        float:
            the DTW aligment cost value
        """
        (
            phone_prob_matrix_nonsil,
            silence_frames_idx,
            non_silence_frames_idx,
        ) = self.get_phone_prob_matrix_nonsil(phone_prob_matrix)

        status, aligned_phones, dtw_cost = self.get_forced_alignment(
            phone_prob_matrix_nonsil, target_phonemes
        )

        if status:
            if silence_frames_idx:
                alignment_with_silence = self.get_alignment_with_silence(
                    aligned_phones, silence_frames_idx, non_silence_frames_idx
                )
            else:
                alignment_with_silence = aligned_phones
            predicted_phones, proba_means = self.predict(
                aligned_phones, phone_prob_matrix_nonsil, target_phonemes
            )
            df_segmented = self.get_df_segmented(
                alignment_with_silence,
                predicted_phones,
                target_phonemes,
                proba_means,
                time_per_output=time_per_output,
            )
        else:
            df_segmented = pd.DataFrame()
            dtw_cost = None

        return df_segmented, dtw_cost
