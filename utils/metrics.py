import cmudict
import pandas as pd
from jiwer import wer
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils.text_processing import remove_stress_annots

global cmu_alphabet 
cmu_alphabet = [el[0] for el in cmudict.phones()]

def compute_PER(predicted_phones_list, target_phonemes_list):
    PERs = []
    for i in range(len(predicted_phones_list)):
        flat_list = [item for item in predicted_phones_list[i]]
        flat_list_target = remove_stress_annots([item for item in target_phonemes_list[i]])
        PERs.append(wer(" ".join(flat_list), " ".join(flat_list_target)))
    return PERs

def plot_cf_matrix(flat_list_target, flat_list_predictions, labels):
    cf_matrix = confusion_matrix(flat_list_target, flat_list_predictions, normalize='true', labels = labels) 
    ax = sns.heatmap(cf_matrix, xticklabels=labels, yticklabels=labels)

    ax.set_title('Phonemes confusion matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    return ax

def filter_df(df_t_test, target_phonemes):
    df_cmu_phones=df_t_test.apply(lambda r: pd.DataFrame.from_records(r.phone_df).cmu_phone.tolist(), axis=1)
    df_target = pd.DataFrame(columns = df_t_test.columns)
    idxs = []

    for i in range(len(df_cmu_phones)):
        if target_phonemes in df_cmu_phones[i]:
            idxs.append(i)

    df_target = df_t_test.iloc[idxs]
    return df_target

def analyze_phones(df_segmented, target_phonemes):
    idxs = []

    for i in range(len(df_segmented)):
        if any(item in target_phonemes for item in df_segmented.iloc[i]['phones']):
            idxs.append(df_segmented.iloc[i])

    return pd.DataFrame(idxs)