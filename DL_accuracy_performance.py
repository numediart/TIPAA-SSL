from cProfile import label
from utils.libri_phonetization_data import build_librispeech_words_df
from utils.libri_phonetization_data import phonetics_for_row, select_libri
from tqdm import tqdm
import pandas as pd
import pickle
from utils.charsiu_utils import charsiu_phone_forced_aligner
import ast

import librosa
from collections import Counter

from utils.label_data_processing import build_user_data_df, get_errors_examples
# exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')

from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, phonetic_content_analysis, start_end_contrast_from_formatted_phonetics_audio
from utils.audio_processing import prepare_audio_file

from utils.label_data_processing import target_to_alternatives, get_sentenceStress_annotation, get_data_new_content, get_data, actor_recordings, final_s_artificial_data
from utils.text_processing import *
from utils.pronunciation_dictionaries import cmu_vowels, cmu_consonants, cmu_phones

from utils.libri_phonetization_data import *

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
# disable pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

from utils.load_model import load_model
# model = load_model(300,18,'MAILABS')

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_confusion_results(results, name='vowel_contrast_actors_w2v'):
    plt.clf()
    fig, axn = plt.subplots(1, len(results))
    fig.set_size_inches(len(results), 6)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, k in enumerate(results):
        ax=axn.flat[i]
        #  from https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot
        sns.heatmap(results[k], ax=ax, annot=True, cmap='YlGnBu',
                    cbar=i == 0,
                    vmin=0, vmax=100,
                    cbar_ax=None if i else cbar_ax)
    fig.tight_layout(rect=[0, 0, .9, 1])

    # plt.savefig('vowel_contrast_actors_hmm.png')
    plt.savefig(name+'.png')
    # plt.savefig('vowel_contrast_users_w2v.png')



def stress_GE_performance_test(level='sentence'):
    df=get_data_new_content()
    stress_intensities=[]
    stress_binaries=[]
    print('n rows:',len(df))
    for i,row in tqdm(df.iterrows()):
        formatted_phonetics=prefill_for_sentence(row.text)['cmu_phonetics']
        _, rID=prepare_audio_file(row.audio_path)
        
        n_words_by_chunk=chunk_text(text=row.text)
        res=stress_from_formatted_phonetics(rID,phonetics=formatted_phonetics, n_words_by_chunk=n_words_by_chunk, level=level)
        print(res)
        stress_intensities.append(res['stress_intensities'])
        stress_binaries.append(res['stress_binaries'])
    df['stress_intensities']=stress_intensities
    df['stress_binaries']=stress_binaries

    if level=='sentence':
        # Here we filter out sentences with more than a stressed word
        sum_df=df.bins.apply(lambda r:sum(r))
        df=df[sum_df==1]
        
        # get the index of the chunk containing the stress, which is the "chunk of interest"
        n_word_by_chunk=df.text.apply(lambda r:chunk_text(r))
        cumsum=n_word_by_chunk.apply(lambda r:np.cumsum([0]+r))
        idx_1=df.bins.apply(lambda r:r.index(1))

        def get_range(cumsum,idx_1):
            range_df=pd.concat([cumsum,idx_1], axis=1)
            ranges=[]
            for i,r in range_df.iterrows():
                idx=r.bins
                ns=r.text
                for i,n in enumerate(ns[::-1]):
                    if idx>=n: 
                        range_of_i=[r.text[len(ns)-i-1],r.text[len(ns)-i]]
                        ranges.append(range_of_i)
                        break
            range_df['ranges']=ranges
            return range_df
        
        # get the range in terms of words for the "chunk of interest"
        range_df=get_range(cumsum,idx_1)
        df_all=pd.concat([df, range_df[['ranges']]], axis=1)

        # COI = Chunk Of Interest
        df_all['COI_stress_intensities']=df_all.apply(lambda r: r['stress_intensities'][r.ranges[0]:r.ranges[1]], axis=1)
        df_all['COI_bins']=df_all.apply(lambda r: r['bins'][r.ranges[0]:r.ranges[1]], axis=1)
        df_all['COI_stress_binaries']=df_all.apply(lambda r: r['stress_binaries'][r.ranges[0]:r.ranges[1]], axis=1)

        all_bins=sum(df_all.COI_bins.tolist(),[])
        rate_of_1=sum(all_bins)/len(all_bins)
        print('rate_of_1',rate_of_1)

        all_bins_pred=sum(df_all.COI_stress_binaries.tolist(),[])
        rate_of_1_pred=sum(all_bins_pred)/len(all_bins_pred)
        print('rate_of_1_pred',rate_of_1_pred)

        failures=df_all[df_all.COI_stress_intensities.apply(lambda r: len(r))==0]
        rate_of_failures=len(failures)/len(df_all)

        df_all=df_all[df_all.COI_stress_intensities.apply(lambda r: len(r))!=0]

        # check if the 1 in COI_bins was predicted as 1 inside COI_stress_binaries. Let's call it the "Binary of Interest" or BOI
        BOI=df_all.apply(lambda r:r['COI_stress_binaries'][r['COI_bins'].index(1)], axis=1)
        rate_BOI=BOI.sum()/len(BOI)
        print('rate_BOI',rate_BOI)
        return df_all
    elif level=='word':
        phonetics=prefill_for_sentence(row.text)['cmu_phonetics']
        split_phonetics=lambda phonetics: [p.replace('|','_').split('_') for p in phonetics.split(' ')]


        df.index=range(len(df))
        word_stress_binaries=df.text.apply(lambda r: [word_stress_from_cmu(p) for p in split_phonetics(prefill_for_sentence(r)['cmu_phonetics'])])

        # I realized some words have all "syllables" stressed (even though they are > 1 syl). In fact, it corresponds to acronyms
        # here I remove them
        
        # To see them:
        # word_stress_binaries[word_stress_binaries.apply(lambda r: sum([(np.prod(el)==1)&(len(el)>1) for el in r]))>0]

        # only keep whn it's not the case
        word_stress_binaries=word_stress_binaries[word_stress_binaries.apply(lambda r: sum([(np.prod(el)==1)&(len(el)>1) for el in r]))==0]
        df=df.loc[word_stress_binaries.index,:]

        # word_stress_binaries[word_stress_binaries.apply(lambda r: [1,1] in r)]
        # word_stress_binaries[word_stress_binaries.apply(lambda r: [1,1,1,1] in r)]

        def select_by_len(stress_binaries):
            all_word_bins=sum(stress_binaries.tolist(),[])
            max_len=max([len(el) for el in all_word_bins])
            min_len=min([len(el) for el in all_word_bins])
            print('min_len',min_len)
            print('max_len',max_len)
            all_word_bins_by_len=[]
            for l in range(max_len):
                all_word_bins_by_len.append([el for el in all_word_bins if len(el)==l+1])
            return all_word_bins_by_len
        
        # filter out examples with different number of words
        # see them:
        # (word_stress_binaries.apply(lambda r: len(r))!=df['stress_binaries'].apply(lambda r: len(r))).sum()
        word_stress_binaries=word_stress_binaries[word_stress_binaries.apply(lambda r: len(r))==df['stress_binaries'].apply(lambda r: len(r))]
        df=df.loc[word_stress_binaries.index]

        assert len(sum(word_stress_binaries.tolist(),[])) == len(sum(df['stress_binaries'].tolist(),[]))

        GT_by_len=select_by_len(word_stress_binaries)
        preds_by_len=select_by_len(df['stress_binaries'])

        # these should be the same
        print([len(el) for el in GT_by_len])
        print([len(el) for el in preds_by_len])

        for l, (GT,pred) in enumerate(zip(GT_by_len, preds_by_len)):
            error_rate=sum(sum(np.abs(np.array(GT)-np.array(pred))))/np.prod(np.array(pred).shape)
            print('words of len '+str(l+1)+' error rate:'+str(error_rate))


def compute_predictions(selection, target_phones='AO1', tech_function=phonemeContrast_from_formatted_phonetics_audio, basis='IH0_D', alternatives=cmu_vowels, **kwargs):
    phonetic_detections=[]
    records=[]
    print('number of examples:', len(selection))
    if len(selection)>0:
        for i,r in tqdm(selection.iterrows()):
            if type(r.target_word_indexes)==str:
                target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
                target_syllable_idx=ast.literal_eval(r.target_syllable_indexes)[0]
            else:
                target_word_idx=r.target_word_indexes
                target_syllable_idx=r.target_syllable_indexes

            status_audio, rID=prepare_audio_file(r.fpath)
            res=tech_function(rID,
                            phonetics=r.cmu_phonetics,
                            target_word_idx=target_word_idx,
                            target_syllable_idx=target_syllable_idx,
                            target_occurence_idx=0,
                            target_phones=target_phones,
                            basis=basis,
                            alternatives=alternatives,
                            mode='file',
                            **kwargs
                            )
            phonetic_detections.append(res['phonetic_detection'])
            records.append(res)
            result_df=pd.DataFrame.from_records(records)
    else:
        print('Selection to compute_prediction is empty')
        result_df=None
    return result_df


def count_values(phonetic_detections):
    d=Counter(phonetic_detections)
    d = pd.DataFrame.from_dict(d, orient='index')
    if len(d)>0:
        d=d.sort_values(by=0, ascending=False)
        d=d / d.sum()*100
    return d

def pContrast_for_user_data( target_phones='AO1', frac=0.001):
    user_data=build_user_data_df()
    # user_data['module_type']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].module_type.values[0], axis=1)
    user_data['target_phoneme']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_phoneme.values[0], axis=1)
    # user_data['cmu_phonetics']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].cmu_phonetics.values[0], axis=1)
    # user_data['target_word_indexes']=user_data.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_word_indexes.values[0], axis=1)

    selection=user_data[user_data.target_phoneme==target_phones]
    selection['cmu_phonetics']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].cmu_phonetics.values[0], axis=1)
    selection['target_word_indexes']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_word_indexes.values[0], axis=1)
    selection['target_syllable_indexes']=selection.apply(lambda r: exercise_data[exercise_data.exercise_id==r.exercise_id].target_syllable_indexes.values[0], axis=1)
    selection['uid']=selection.apply(lambda r: r.exercise_id+r.audio_file_idx, axis=1)

    selection['audio_file_url']=selection['fpath']
    selections=[]
    n_user=50
    for u in selection.user_id.unique()[:n_user]:
        for ex in selection.exercise_id.unique():
            selections.append(selection[selection.user_id==u][selection.exercise_id==ex])
    df=pd.concat(selections)

    df=df.sample(frac=frac, random_state=1234)

    df['split_phonetics']=df.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)

    _,result_df=compute_predictions(df, target_phones=target_phones)
    phonetic_detections=result_df.phonetic_detection

    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return phonetic_detections, d



def distrib(l):
    x = np.linspace(0, 1, 1000)
    kde = gaussian_kde(l, bw_method = 0.5)
    y = kde(x)
    return x,y

def syllable_contrast_for_actor_recordings():
    # from DL_accuracy_performance import *
    df=actor_recordings()
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']

    df_v=df_pContrast[~df_pContrast.target_phoneme.isin(eds)]
    df_ed=df_pContrast[df_pContrast.target_phoneme.isin(eds)]


    # contrast on all phones with force_and_predict (i.e. predict based on frames allocated to a phoneme)
    pred_dfs=[]
    phonetic_contents=[]
    print(len(df))
    means=[]
    medians=[]
    for i,r in tqdm(df_pContrast.iterrows()):
        s,fs=librosa.load(r.audio_file_url, sr=16000)
        split_phonetics=sum([p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], [])
        _, p_df, phonetic_content = model.align_phones(audio=s,phones=split_phonetics)
        # p_df=charsiu.force_and_predict(s, split_phonetics)
        pred_dfs.append(p_df)
        phonetic_contents.append(phonetic_content)
        # phonetic_content.GT_proba_means.median()
        means.append(p_df[p_df.cmu_phones!='[SIL]'].GT_proba.mean())
        medians.append(p_df[p_df.cmu_phones!='[SIL]'].GT_proba.median())
    correct_proba_means=np.histogram(means)
    correct_proba_medians=np.histogram(medians)

    all_phones_df=pd.concat(pred_dfs)

    x,kde1_x=distrib(means)
    plt.plot(x, kde1_x)
    plt.savefig('gkde.png')

    p_to_id=lambda p: model.charsiu_processor.mapping_phone2id(p)
    # target='IH'

    def plot_vowel_distributions(target):
        p_idx=p_to_id(target)
        plt.cla()
        for v in cmu_vowels:
            l=all_phones_df[all_phones_df.cmu_phones==v].apply(lambda r: r.proba_means[p_idx], axis=1)
            x,y=distrib(l)
            if y[0]<10:
                plt.plot(x,y, label=v)
            else:
                print('vowel', v)
                print('max is', max(y))
        plt.legend()
        plt.title("Proba distributions for "+target)
        plt.savefig('probas_'+target+'.png')

    for v in cmu_vowels:
        plot_vowel_distributions(v)
    np.histogram(all_phones_df[all_phones_df.cmu_phones=="IY"].GT_proba)

    s=all_phones_df[all_phones_df.cmu_phones=="IY"]
    s[(s.GT_proba>0.2)&(s.GT_proba<0.5)]



    
    np.histogram(all_phones_df[all_phones_df.cmu_phones=="EH"].GT_proba)
    np.histogram(all_phones_df[all_phones_df.cmu_phones=="IH"].GT_proba)
    np.histogram(all_phones_df[all_phones_df.cmu_phones=="IY"].GT_proba)

    np.histogram(all_phones_df[all_phones_df.cmu_phones=="AA"].GT_proba)
    np.histogram(all_phones_df[(all_phones_df.cmu_phones=="AA")&(all_phones_df.pred_phones_audio=="AO")].pred_proba)

    np.histogram(all_phones_df[all_phones_df.cmu_phones=="AO"].GT_proba)
    np.histogram(all_phones_df[(all_phones_df.cmu_phones=="AO")&(all_phones_df.pred_phones_audio=="AA")].pred_proba)

    np.histogram(all_phones_df[all_phones_df.cmu_phones=="OW"].GT_proba)

    correct_proba_means=pd.DataFrame(correct_proba_means).T
    correct_proba_means.columns=['N','target proba']
    correct_proba_medians=pd.DataFrame(correct_proba_medians).T
    correct_proba_medians.columns=['N','target proba']

    medians[7]
    df_pContrast.iloc[7]
    pred_dfs[7]

    # mismatches on purpose
    pred_dfs=[]
    phonetic_contents=[]
    print(len(df))
    means=[]
    medians=[]
    for i,r in tqdm(df_pContrast.iterrows()):
        s,fs=librosa.load(r.audio_file_url, sr=16000)
        # take random sentence in df_pContrast that is not the same

        # recursive call until it works, as I pick randomly some phonetics and sometimes it fails to align in DTW algorithm
        def inference():
            try:
                phonetics=df_pContrast[df_pContrast.cmu_phonetics!=r.cmu_phonetics].sample().cmu_phonetics.values[0]
                split_phonetics=sum([p.replace('|','_').split('_') for p in phonetics.split(' ')], [])
                _, p_df, phonetic_content = model.align_phones(audio=s,phones=split_phonetics)
                return p_df, phonetic_content
            except:
                return inference()
        
        p_df, phonetic_content=inference()
        phonetic_content=phonetic_content[phonetic_content.pred_phones_audio!='[SIL]']
        pred_phones_audio=drop_consecutive_duplicate_elements(phonetic_content.pred_phones_audio.tolist())

        # p_df=charsiu.force_and_predict(s, split_phonetics)
        pred_dfs.append(p_df)
        phonetic_contents.append(phonetic_content)
        # phonetic_content.GT_proba_means.median()
        means.append(p_df[p_df.cmu_phones!='[SIL]'].GT_proba.mean())
        medians.append(p_df[p_df.cmu_phones!='[SIL]'].GT_proba.median())
    mismatch_proba_means=np.histogram(means)
    mismatch_proba_medians=np.histogram(medians)

    x,kde2_x=distrib(means)
    plt.plot(x, kde2_x)
    plt.savefig('proba_means_gkde.png')

    # intersection of distributions
    idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()
    optimal_threshold=x[idx]

    mismatch_proba_means=pd.DataFrame(mismatch_proba_means).T
    mismatch_proba_means.columns=['N','target proba']
    mismatch_proba_medians=pd.DataFrame(mismatch_proba_medians).T
    mismatch_proba_medians.columns=['N','target proba']

    match_rate=1-len(pred_df[pred_df.cmu_phones!=pred_df.pred_phones_audio])/len(pred_df)
    pred_df=pred_df[pred_df.pred_phones_audio!='[SIL]']
    match_rate=1-len(pred_df[pred_df.cmu_phones!=pred_df.pred_phones_audio])/len(pred_df)

    
    # final -ed based on force_and_predict logic
    preds=[]
    len(df_ed)
    df_words=[]
    pred_records=[]
    for i,r in tqdm(df_ed.iterrows()):
        target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
        # syl_idx=ast.literal_eval(r.target_syllable_indexes)[0]
        s,fs=librosa.load(r.audio_file_url, sr=16000)
        # split_phonetics=sum([p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], [])
        
        phonetics=r.cmu_phonetics
        _, rID=prepare_audio_file(r.audio_file_url)
        res=termination_contrast(rID,phonetics=phonetics, 
                            target_word_idx=target_word_idx, 
                            target_phones=r.target_phoneme)
        pred_records.append(res)
    
    syl_pred_df=pd.DataFrame.from_records(pred_records)
    success_rate=len(syl_pred_df[syl_pred_df.gibberish_truth==syl_pred_df.gibberish_detected])/len(syl_pred_df)
    syl_pred_df[syl_pred_df.gibberish_truth!=syl_pred_df.gibberish_detected]
    
    # an example of underlying align_phones
    r=df.iloc[2562]
    s,fs=librosa.load(r.audio_file_url, sr=16000)
    phonetics=r.cmu_phonetics
    phones=[[p] for p in sum([p.replace('|','_').split('_') for p in phonetics.split(' ')],[])]
    charsiu.align_phones(s, phones)

    # syllables
    # df=df[df.cmu_phonetics.str.contains('T_IH0_D')]
    # df[df.cmu_phonetics.str.contains('S_T_N')]

    sentences_dfs=[]
    GT_overall_confidences=[]
    pred_overall_confidences=[]
    print(len(df))
    r=df.loc[81]
    r=df.loc[702]
    r=df.loc[1219]
    r=df.iloc[0]
    for i,r in tqdm(df.iterrows()):
        s,fs=librosa.load(r.audio_file_url, sr=16000)
        # phonetic_content=charsiu.analyze_phonetic_content(s, r.cmu_phonetics)
        phonetics=r.cmu_phonetics
        phonetic_content=phonetic_content_analysis(s, phonetics)
        # all_syl_dfs.append(syl_dfs)

        # phonetic_content=pd.concat(syl_dfs)
        sentences_dfs.append(phonetic_content)

        GT_overall_confidence=(phonetic_content['GT_proba_means']*phonetic_content['n_frames']).sum()/sum(phonetic_content.n_frames)
        GT_overall_confidences.append(GT_overall_confidence)
        pred_overall_confidence=(phonetic_content['pred_proba_means']*phonetic_content['n_frames']).sum()/sum(phonetic_content.n_frames)
        pred_overall_confidences.append(pred_overall_confidence)
    
    # make sentence indices
    # sentences_dfs=[pd.concat(dfs) for dfs in all_syl_dfs]
    lens_sentences_dfs=[len(df) for df in sentences_dfs]

    sentence_idxs=[]
    for i,n in enumerate(lens_sentences_dfs):
        sentence_idxs.append([i]*n)
    sentence_idxs=sum(sentence_idxs,[])

    all_phones_df=pd.concat(sentences_dfs)
    all_phones_df['sentence_idx']=sentence_idxs

    all_phones_df.to_csv('performance_results/actor_recordings_all_predictions_post_correction.csv')
    ################### Data analysis of predictions ################
    all_phones_df=pd.read_csv('performance_results/actor_recordings_all_predictions.csv')

    p_counts={}
    for p in all_phones_df.pred_phones.unique():
        d=count_values(all_phones_df[all_phones_df.pred_phones==p].pred_phones_audio)
        d.columns=[p]
        p_counts[p]=d
    
    p_counts['AO']
    p_counts['AA']
    
    errors=all_phones_df[all_phones_df.pred_phones!=all_phones_df.pred_phones_audio]
    errors=errors[errors.pred_phones_audio!='[SIL]']
    # errors=errors[errors.n_frames>1]

    np.histogram(all_phones_df.n_frames)

    overall_phone_rate=1-len(errors)/len(all_phones_df)
    
    # I cannot really remove that afterwards I think, so it would over-estimate the success rate
    # overall_phone_rate=1-len(errors[errors.n_frames>1])/len(all_phones_df[all_phones_df.n_frames>1])
    # If I want to use that, I would have to get pred_phones from ground-truth, and ot altered with post-correction etc. that may remove some rows

    # it was short, like 1 frame = 0.01 second
    # https://reader.elsevier.com/reader/sd/pii/S0095447019305030?token=4B7CC7AB0B052DE372176B0560B43A64AFF8DEBDBC81AC66D7A4024547E51F1333E7BB3F0752B6AC070E0FAE6E917CD9&originRegion=eu-west-1&originCreation=20220401153652

    
    np.histogram(pred_overall_confidences)
    np.histogram(GT_overall_confidences)

    df['GT_overall_confidences']=GT_overall_confidences
    df['pred_overall_confidences']=pred_overall_confidences
    
    df[df.GT_overall_confidences<0.1][['text','audio_file_url']]

    from utils.text_processing import split_phonetics, remove_stress_annots
    # This is to get every syllable
    all_phones_df.index=range(len(all_phones_df))
    syl_indexation_df=all_phones_df[['sentence_idx', 'word_idx',  'syl_idx' ]].drop_duplicates()
    all_syls_records=[]
    print(len(syl_indexation_df))
    for i,r in tqdm(syl_indexation_df.iterrows()):
        syl_df=all_phones_df[all_phones_df.sentence_idx==r.sentence_idx][all_phones_df.word_idx==r.word_idx][all_phones_df.syl_idx==r.syl_idx]

        # GT=drop_consecutive_duplicates(syl_df[['pred_phones']]).pred_phones.tolist()
        # ground-truth should be extracted from original phonetics and not altered dataframe, due to post-correction
        phonetics=df.iloc[r.sentence_idx].cmu_phonetics
        spl_p=split_phonetics(phonetics)[r.word_idx][r.syl_idx]
        GT=remove_stress_annots(spl_p)

        pred=drop_consecutive_duplicates(syl_df[['pred_phones_audio']][syl_df.pred_phones_audio!='[SIL]']).pred_phones_audio.tolist()
        d={'GT':'_'.join(GT),'pred':'_'.join(pred)}
        all_syls_records.append(d)
    all_syls_df=pd.DataFrame.from_records(all_syls_records)
    overall_syl_rate=len(all_syls_df[all_syls_df.GT==all_syls_df.pred])/len(all_syls_df)

    s_counts={}
    for s in all_syls_df.GT.drop_duplicates():
        d=count_values(all_syls_df[all_syls_df.GT==s].pred)
        d.columns=[s]
        s_counts[s]=d 
    s_counts['T_IH_D']
    s_counts['S_T_N']

    
    T_AH_D_idx=all_syls_df[all_syls_df.pred=="T_AH_D"].iloc[0].name
    r=syl_indexation_df.iloc[T_AH_D_idx]
    all_phones_df[all_phones_df.sentence_idx==r.sentence_idx]

    all_syl_dfs[1758]
    all_syl_dfs[2009]

    # 28 mins
    
    

def termination_contrast_for_actor_recordings(target_phones='D'):
    df=actor_recordings()

    df['fpath']=df['audio_file_url']

    selection=df[df.target_phoneme==target_phones]
    result_df=compute_predictions(selection, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio)
    phonetic_detections=result_df.phonetic_detection

    success_rate=len(result_df[result_df.gibberish_truth==result_df.gibberish_detected])/len(result_df)
    print('errors:',result_df[result_df.gibberish_truth!=result_df.gibberish_detected])
    print(success_rate)
    
    d=count_values(phonetic_detections)
    d.columns=[target_phones]
    return phonetic_detections, d
    


def pContrast_for_actor_recordings(target_phones='AO1'):
    df=actor_recordings()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]    
    selection['split_phonetics']=selection.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)
    selection['fpath']=selection.audio_file_url
    # selection['audio_file_idx']=selection.fk_audio_recording_id

    result_df=compute_predictions(selection, target_phones=target_phones)
    phonetic_detections=result_df.phonetic_detection

    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return phonetic_detections, d


def formatted_audiobook_data(selection, libri_words_df, target_phones=None):
    # retrieve phonetics by word thanks to 'phonetics_fot_row'
    selection['split_phonetics']=selection.apply(lambda r: [p.split(' ') for p in phonetics_for_row(r, libri_words_df)], axis=1)
    # Generate formatted_phonetics by syllabifying each word with sonoripy, then joining phonemes, syllables and words with separators
    selection['cmu_phonetics']=selection.apply(lambda r: ' '.join(['|'.join(['_'.join(syl) for syl in SonoriPy(w)[0]]) for w in r.split_phonetics]), axis=1)
    
    # I figured out there are empty elements resulting in double space in formatted phonetics. This is problematic. 
    # As a fix, I insert a "AH0" to have something very neutral close to silence
    selection['cmu_phonetics']=selection.apply(lambda r: r['cmu_phonetics'].replace('  ', ' AH0 '), axis=1)

    selection['target_word_indexes']=selection['word_idx']
    selection['fpath']=selection['wav_path']

    # to extract a target_syllable_index, I use a function that find the first element containing a string
    def find_first(lst, predicate): return next((i for i,j in enumerate(lst) if predicate(j)), float('nan'))
    find_first_containing_target=lambda l: find_first(l, lambda x: target_phones in x.split('_'))

    if target_phones is not None:
        selection['target_syllable_indexes']=selection.apply(lambda r: find_first_containing_target(r['cmu_phonetics'].split(' ')[r['word_idx']].split('|')), axis=1)
        # find_first_containing_target(phonetics.split(' ')[target_word_idx].split('|'))
    else:
        # this is for termination contrasts
        selection['target_syllable_indexes']=-1
    return selection

def termination_contrast_from_audiobook_data(data_set='test-other', target_phones='D', n=None):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    selection=select_libri(select_libri(libri_words_df,'D', option='endswith'), 'ed', column='word', option='endswith')
    selection_id=select_libri(selection, 'IH0 D', option="endswith")
    selection_d=selection[~selection.index.isin(selection_id.index.tolist())]
    selection_t=select_libri(select_libri(libri_words_df,'T', option='endswith'), 'ed', column='word', option='endswith')

    selections={}
    selections['D']=selection_d
    selections['IH0_D']=selection_id
    selections['T']=selection_t
    
    selection=formatted_audiobook_data(selections[target_phones], libri_words_df)

    result_df=compute_predictions(selection, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio)
    phonetic_detections=result_df.phonetic_detection

    success_rate=len(result_df[result_df.gibberish_truth==result_df.gibberish_detected])/len(result_df)

    errors_df=result_df[result_df.gibberish_truth!=result_df.gibberish_detected]
    print('errors:',errors_df)

    if len(errors_df)>0: print("last error:"); errors_df.iloc[-1]
    
    print(success_rate)
    selection=selection.reset_index(drop=True)
    result_df['cmu_phonetics']=selection['cmu_phonetics']
    result_df['fpath']=selection['fpath']

    d=count_values(phonetic_detections)
    d.columns=[target_phones]
    return phonetic_detections, d


def final_s_from_audiobook_data(data_set='dev-clean', n=None):

    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    # words that finish in "s" with phoneme "S" that also exist without an "s" and with last phoneme then not being "S"
    words_in_s=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='S' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='S']
    words_in_s=set(words_in_s)

    words_in_z=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='Z' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='Z']
    words_in_z=set(words_in_z)

    libri_words_list=libri_words_df.word.unique()
    libri_words_in_s=[el for el in libri_words_list if el in words_in_s]
    
    # word=libri_words_in_s[3]

    selections=[]
    selections_s=[]
    print(len(libri_words_in_s))
    for word in tqdm(libri_words_in_s):
        selection, selection_s = selection_with_and_without_s(libri_words_df, word[:-1])

        if len(selection)>0 and len(selection_s)>0:
            selection=formatted_audiobook_data(selection, libri_words_df)
            selection_s=formatted_audiobook_data(selection_s, libri_words_df)

            selections.append(selection)
            selections_s.append(selection_s)

    selections=pd.concat(selections)
    selections_s=pd.concat(selections_s)

    freq_words=count_values(selections.word.tolist()).index[:2].tolist()
    freq_words_s=count_values(selections_s.word.tolist()).index[:2].tolist()

    selections=selections[~selections.word.isin(freq_words)]
    selections_s=selections_s[~selections_s.word.isin(freq_words_s)]

    # filter our first ones like "it" or "its" that are too frequent

    result_df=compute_predictions(selections, target_phones='', tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='S')
    phonetic_detections=result_df.phonetic_detection
    result_df_s=compute_predictions(selections_s, target_phones='S', tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='S')
    phonetic_detections_s=result_df_s.phonetic_detection

    # success_rate=len(result_df[result_df.gibberish_truth==result_df.gibberish_detected])/len(result_df)
    # print('errors:',result_df[result_df.gibberish_truth!=result_df.gibberish_detected])
    # result_df[result_df.gibberish_truth!=result_df.gibberish_detected].iloc[-1]
    # print(success_rate)

    selections=selections.reset_index(drop=True)
    result_df['cmu_phonetics']=selections['cmu_phonetics']
    result_df['fpath']=selections['fpath']

    d=count_values(phonetic_detections)
    d_s=count_values(phonetic_detections_s)

    return phonetic_detections, phonetic_detections_s, d, d_s

def final_s_from_artificial_data():
    df=final_s_artificial_data()    
    # df.apply(lambda r: r.text.split(' ')[r.word_idx], axis=1)
    df_target_word_p=df.apply(lambda r: r.cmu_phonetics.split(' ')[r.word_idx], axis=1)

    df['target_word_indexes']=df['word_idx']
    df['fpath']=df['path']

    df_iz=df[df_target_word_p.str.endswith('AH0_Z')|df_target_word_p.str.endswith('IH0_Z')]
    df_s=df[df_target_word_p.str.endswith('_S')]
    df_z=df[~(df_target_word_p.str.endswith('AH0_Z')|df_target_word_p.str.endswith('IH0_Z'))&~df_target_word_p.str.endswith('_S')]

    selections={
        'IH0_Z':df_iz,
        'S':df_s,
        'Z':df_z
    }

    targets=['S','Z','IH0_Z']
    result_dfs={}
    # predictions={}
    results={}

    for t in targets:
        result_df=compute_predictions(selections[t], target_phones=t, tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='IH_Z')
        result_dfs[t]=result_df
    
    plot_confusion_results(results, name='plots/final_s_artificial_data')

    for t in targets:
        phonetic_detections=result_dfs[t].phonetic_detection
        d=count_values(phonetic_detections)
        d.columns=[t]
        results[t]=d

    for t in targets:
        print('target ',t)
        print('successes:',result_dfs[t][result_dfs[t].phonetic_detection==t])
        print('errors:',result_dfs[t][result_dfs[t].phonetic_detection!=t])
        success_rate=len(result_dfs[t][result_dfs[t].phonetic_detection==t])/len(result_dfs[t])
        print('success_rate:',success_rate)

    
    selections['Z'].reset_index(drop=True)

    selections['Z'].reset_index(drop=True)[result_dfs['Z'].status.str.contains('not')]
    selections['S'].reset_index(drop=True)[result_dfs['S'].status.str.contains('not')]
    selections['IH0_Z'].reset_index(drop=True)[result_dfs['IH0_Z'].status.str.contains('not')]

    selections['IH0_Z'].reset_index(drop=True)[result_dfs['IH0_Z'].phonetic_detection!='IH0_Z']
    # d=count_values(phonetic_detections)
    # d.columns=[target_phones]
    # return phonetic_detections, d

    result_dfs


def pContrast_from_audiobook_data(data_set='test-other', target_phones='AO1', n=None, alternatives=cmu_vowels):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]

    if target_phones!='G':
        # Select words containing a phone. For 'T', to avoid capturing also 'TH', it has to contain 'T '  or end by ' T'
        # same for D/DH, J/JH etc.
        selection=libri_words_df[libri_words_df.phones.str.endswith(' '+target_phones)|libri_words_df.phones.str.contains(target_phones+' ')]
    else:
        # same idea, but only for G/NG
        selection=libri_words_df[libri_words_df.phones.str.startswith(target_phones+' ')|libri_words_df.phones.str.contains(' '+target_phones)]


    selection=formatted_audiobook_data(selection, libri_words_df, target_phones=target_phones)

    result_df=compute_predictions(selection, target_phones=target_phones, alternatives=alternatives)
    phonetic_detections=result_df.phonetic_detection
    
    d=count_values(phonetic_detections)
    d.columns=[target_phones]
    return phonetic_detections, d

def start_end_phoneme_from_audiobook_data(phoneme='HH', basis='HH', contrast="start",data_set='dev-clean', n=100):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    if contrast=="start":
        idx=0
    elif contrast=="end":
        idx=-1

    words_in_p=[el for el in cmudict_dict.keys() if cmudict_dict[el][0][idx]==phoneme]
    words_in_p=set(words_in_p)
    libri_words_list=libri_words_df.word.unique()
    libri_words_in_p=[el for el in libri_words_list if el in words_in_p]

    selections=[]
    print(len(libri_words_in_p))
    for word in tqdm(libri_words_in_p):
        selection=libri_words_df[libri_words_df.word==word]
        if len(selection)>0:
            selection=formatted_audiobook_data(selection, libri_words_df)
            selections.append(selection)
    selections=pd.concat(selections)

    freq_words=count_values(selections.word.tolist()).index[:9].tolist()
    selections=selections[~selections.word.isin(freq_words)]
    
    if basis!=phoneme:
        target_phones=''
    else:
        target_phones=phoneme

    result_df=compute_predictions(selections, target_phones=target_phones, tech_function=start_end_contrast_from_formatted_phonetics_audio, basis=basis, contrast=contrast)
    phonetic_detections=result_df.phonetic_detection
    
    selections=selections.reset_index(drop=True)
    result_df['cmu_phonetics']=selections['cmu_phonetics']
    result_df['fpath']=selections['fpath']
    d=count_values(phonetic_detections)
    d.columns=[phoneme]
    return phonetic_detections,result_df,d


def vowels_confusions_actor_recordings():
    # for actors recordings, take only the true targets
    vowels=['IY1','IH1','AO1','AA1','OW1']

    results={}
    for v in vowels: 
        print("vowel:",v)
        _, rates=pContrast_for_actor_recordings(target_phones=v)
        # _, predictions[v].to_csv('performance_results/vowel_accuracies_'+v+'.csv')
        results[v]=rates

    return results

def vowels_confusions_user_recordings(frac=0.001):
    vowels=['IY1','IH1','AO1','AA1','OW1']
    results={}
    for v in vowels: 
        print("vowel:",v)
        _, rates=pContrast_for_user_data(target_phones=v, frac=frac)
        # pickle.dump(predictions[v], open('vowel_accuracies'+v+'.p','wb'))
        results[v]=rates
    return results

def vowels_consonants_confusions_from_audiobook_data(n=100, data_set='test-other'):
    predictions={}
    results={}
    
    # cs=['T','S','TH']
    # for c in tqdm(cs):
    for c in tqdm(cmu_consonants):
        # predictions[c], results[c]=pContrast_from_audiobook_data(data_set=data_set, target_phones=c, n=n, alternatives=cmu_consonants, model=model)
        try:
            predictions[c], results[c]=pContrast_from_audiobook_data(data_set=data_set, target_phones=c, n=n, alternatives=cmu_consonants)
        except: 
            print("This consonant isn't in dataset")

    for v in tqdm(cmu_vowels): 
        # predictions[v], results[v]=pContrast_from_audiobook_data(data_set=data_set, target_phones=v+'1', n=n, alternatives=cmu_vowels, model=model)
        try:
            predictions[v], results[v]=pContrast_from_audiobook_data(data_set=data_set, target_phones=v+'1', n=n, alternatives=cmu_vowels)
        except: 
            print("This vowel isn't in dataset")
    return predictions, results

def termination_confusions_from_audiobook_data(n=100, data_set='dev-clean'):
    eds=['T','D','IH0_D']

    predictions={}
    results={}
    for p in tqdm(eds): 
        predictions[p], rates=termination_contrast_from_audiobook_data(data_set=data_set, target_phones=p, n=n)
        results[p]=rates
    return predictions, results

def termination_confusions_for_actor_recordings():
    eds=['T','D','IH0_D']

    predictions={}
    results={}
    for p in tqdm(eds): 
        predictions[p], rates=termination_contrast_for_actor_recordings(target_phones=p)
        results[p]=rates
    return predictions, results



def pronunciation_aspects_from_audiobook_data(n=100, data_set='test-other'):
    

    _,_,d=start_end_phoneme_from_audiobook_data(phoneme='HH',basis='HH', n=n, data_set=data_set)
    r=d.T['HH']
    _,_,d=start_end_phoneme_from_audiobook_data(phoneme='EH1',basis='HH', n=1000, data_set=data_set)
    r=d[d.index!='HH'].sum()

    # start_end_phoneme_from_audiobook_data(phoneme='AA', n=10000, data_set=data_set)
    _,_,d=start_end_phoneme_from_audiobook_data(phoneme='S',basis='S', contrast="end", n=n, data_set=data_set, model=model)
    r=d.T['S']
    _,_,d=start_end_phoneme_from_audiobook_data(phoneme='K',basis='S', contrast="end", n=n, data_set=data_set, model=model)
    r=d[d.index!='S'].sum()

    start_end_phoneme_from_audiobook_data(phoneme='Z',basis='Z', contrast="end", n=n, data_set=data_set, model=model)

    predictions, results=vowels_consonants_confusions_from_audiobook_data(n=n, data_set=data_set, model=model)

    with open('vowels_consonant_contrast_audiobook_w2v'+data_set+'_n_'+str(n)+'.pickle', 'wb') as handle:pickle.dump(results,handle)
    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_w2v'+data_set+'_n_'+str(n))



if __name__=='__main__':
    from DL_accuracy_performance import *

    starting_h_from_audiobook_data(data_set='dev-clean', n=100)

    start_end_phoneme_from_audiobook_data(phoneme='HH')
    start_end_phoneme_from_audiobook_data(phoneme='S', basis='S', contrast="end")
    start_end_phoneme_from_audiobook_data(phoneme='Z', basis='Z', contrast="end")

    pContrast_from_audiobook_data(n=100)

    r=termination_contrast_for_actor_recordings()
    r=termination_contrast_for_actor_recordings('T')
    r=termination_contrast_for_actor_recordings("IH0_D")

    pContrast_for_actor_recordings(target_phones='AO1')

    stress_GE_performance_test(level='word')
    stress_GE_performance_test(level='sentence')

    results=vowels_confusions_user_recordings(frac=0.01)
    
    target_phones='AO1'
    df=actor_recordings()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]    
    
    selection['split_phonetics']=selection.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)
    selection['fpath']=selection.audio_file_url

    r=selection.iloc[24]
    
    s,fs=librosa.load(r.fpath, sr=16000)

    target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
    target_syllable_idx=ast.literal_eval(r.target_syllable_indexes)[0]
    df_word=model.predict_word(s, r.split_phonetics, target_word_idx)
    
    word=r.cmu_phonetics.split(' ')[target_word_idx]

    syllables=[syl.split('_') for syl in word.split('|')]
    syllable=syllables[target_syllable_idx]
    
    import cmudict
    phone_df=pd.DataFrame(cmudict.phones())
    vowels=phone_df[phone_df.apply(lambda r: r.iloc[1][0], axis=1)=='vowel'].iloc[:,0].tolist()

    # There is 1 vowel by syllable. Therefore we can find the phoneme index by looking for the vowel
    p_idx_local=[p in vowels for p in syl].index(True)
    len_previous_syllables=sum([len(el) for el in syllables[:target_syllable_idx]])
    p_idx_global=len_previous_syllables+p_idx_local
    phonetic_detection=df_word.iloc[p_idx_global].pred_phones_audio


    r=pContrast_from_audiobook_data(data_set='dev-clean', target_phones='AO1', n=10)

    from time import time
    start=time()
    predictions, results=vowels_consonants_confusions_from_audiobook_data(n=None)
    duration=time()-start

    data_set='test-other'
    n=100
    predictions, results=vowels_consonants_confusions_from_audiobook_data(n=n, data_set=data_set)
    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_w2v'+data_set+'_n_'+str(n))

    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_test-other_w2v_n_10')

    # data_set='dev-clean'
    data_set='test-other'
    n=300
    predictions, results=termination_confusions_from_audiobook_data(n=n, data_set=data_set)
    plot_confusion_results(results, name='termination_contrast_audiobook_no_D_T_dis_'+data_set+'_w2v_n_'+str(n))

    predictions, results=termination_confusions_for_actor_recordings()
    plot_confusion_results(results, name='plots/termination_confusions_for_actor_recordings_D_T_dis_w2v_AH_D_post_corr')

    results=vowels_confusions_actor_recordings()
    plot_confusion_results(results, name='vowel_contrast_proba_means_actors_w2v_thresh_0.2')

    results=vowels_confusions_user_recordings(frac=0.01)
    plot_confusion_results(results, name='vowel_contrast_proba_means_user_data_w2v_thresh_0.2')

    # for i, k in enumerate(predictions):
    #     ax=axn.flat[i]
    #     d=Counter(predictions[k].pred_phones_audio.tolist())
    #     print(d)

    #     df = pd.DataFrame.from_dict(d, orient='index')
    #     df.columns=[k]
    #     df=df.sort_values(by=k, ascending=False)
    #     df=df / df.sum()*100

    #     #  from https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot
    #     sns.heatmap(df, ax=ax, annot=True, cmap='YlGnBu',
    #                 cbar=i == 0,
    #                 vmin=0, vmax=100,
    #                 cbar_ax=None if i else cbar_ax)
    

    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig('vowel_contrast_libri_test_other.png')
    
    
    predictions=pickle.load(open('vowel_accuracies.p','rb'))

    
    flights = sns.load_dataset("flights")

