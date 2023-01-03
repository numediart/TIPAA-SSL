from tqdm import tqdm
import pandas as pd
import pickle
import ast
import librosa

from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, phonetic_content_analysis, start_end_contrast_from_formatted_phonetics_audio, default_model, default_model_charsiu

from src.label_data_processing import actor_recordings, final_s_artificial_data, synth_words_data
from src.text_processing import *
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants

from src.libri_phonetization_data import *

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
# disable pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


from performance_functions import compute_predictions, count_values, pContrast_on_synth_words, \
                                pContrast_for_actor_recordings, start_end_phoneme_from_audiobook_data, \
                                final_ed_for_actor_recordings, stress_GE_performance_test, final_ed_from_audiobook_data, pContrast_from_audiobook_data, pContrast_for_user_data


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



def distrib(l):
    n_points=1000
    x = np.linspace(0, 1, n_points)
    if len(set(l))==1: #all the values are the same means infinite density on this value, and 0 for the rest
        y=np.zeros(len(x))
        idx=int(list(l)[0]*n_points)
        y[idx]=np.inf
    else:
        kde = gaussian_kde(l, bw_method = 0.5)
        y = kde(x)
    return x,y

def GT_proba_distribution_analysis(model=default_model, basename='probas_actors_logistic'):

    # user_data=build_user_data_df()
    # selection=user_data[user_data.target_phoneme==target_phones]
    # df_users=selection

    # from DL_accuracy_performance import *
    df=actor_recordings()
    # df_actors=df[df.target_phoneme==target_phones]

    def compute_predictions(df_pContrast, n_examples=None, model=default_model):
        # contrast on all phones with force_and_predict (i.e. predict based on frames allocated to a phoneme)
        pred_dfs=[]
        # phonetic_contents=[]
        print(len(df_pContrast))
        means=[]
        medians=[]
        for i,r in tqdm(df_pContrast[:n_examples].iterrows()):
            s,fs=librosa.load(r.audio_file_url, sr=16000)
            split_phonetics=sum([p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], [])
            try:
                _, p_df, phonetic_content = model.align_phones(audio=s,phones=split_phonetics)
            except AttributeError:
                p_df=model.predict_with_timings(s, split_phonetics)
            # p_df=charsiu.force_and_predict(s, split_phonetics)
            pred_dfs.append(p_df)
            # phonetic_contents.append(phonetic_content)
            # phonetic_content.GT_proba_means.median()
            means.append(p_df[p_df.phones!='[SIL]'].GT_proba.mean())
            medians.append(p_df[p_df.phones!='[SIL]'].GT_proba.median())
        # correct_proba_means=np.histogram(means)
        # correct_proba_medians=np.histogram(medians)

        return pred_dfs, means, medians

    
    def plot_vowel_distributions(target, all_phones_df, basename='probas'):
        p_idx=p_to_id(target)
        plt.cla()
        for v in cmu_vowels:
            l=all_phones_df[all_phones_df.phones==v].apply(lambda r: r.proba_means[p_idx], axis=1)
            if len(l)>0:
                x,y=distrib(l)
                if y[0]<10 or v==target:
                    plt.plot(x,y, label=v)
                else:
                    print('vowel', v)
                    print('max is', max(y))
        plt.legend()
        plt.title("Proba distributions for "+target)
        plt.savefig(basename+'_'+target+'.png')

    pred_dfs, means, medians=compute_predictions(df, n_examples=100, model=model)

    all_phones_df=pd.concat(pred_dfs)

    x,kde1_x=distrib(means)
    plt.plot(x, kde1_x)
    plt.savefig('gkde_logistic.png')

    # try:
    #     p_to_id=lambda p: model.charsiu_processor.mapping_phone2id(p)
    # except:
    p_to_id=lambda p: model.p_to_id[p]
    # target='IH'

    for v in cmu_vowels:
        plot_vowel_distributions(v, all_phones_df, basename=basename)
        print(np.histogram(all_phones_df[all_phones_df.phones==v].GT_proba))



def syllable_contrast_for_actor_recordings(model=default_model):
    # from DL_accuracy_performance import *
    df=actor_recordings()
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    vowels=['IH','IY','OW','AO','AA']
    eds=['T','D','IH0_D']

    df_v=df_pContrast[~df_pContrast.target_phoneme.isin(eds)]
    df_ed=df_pContrast[df_pContrast.target_phoneme.isin(eds)]

    np.histogram(all_phones_df[all_phones_df.phones=="IY"].GT_proba)

    s=all_phones_df[all_phones_df.phones=="IY"]
    s[(s.GT_proba>0.2)&(s.GT_proba<0.5)]



    
    np.histogram(all_phones_df[all_phones_df.phones=="EH"].GT_proba)
    np.histogram(all_phones_df[all_phones_df.phones=="IH"].GT_proba)
    np.histogram(all_phones_df[all_phones_df.phones=="IY"].GT_proba)

    np.histogram(all_phones_df[all_phones_df.phones=="AA"].GT_proba)
    np.histogram(all_phones_df[(all_phones_df.phones=="AA")&(all_phones_df.pred_phones_audio=="AO")].pred_proba)

    np.histogram(all_phones_df[all_phones_df.phones=="AO"].GT_proba)
    np.histogram(all_phones_df[(all_phones_df.phones=="AO")&(all_phones_df.pred_phones_audio=="AA")].pred_proba)

    np.histogram(all_phones_df[all_phones_df.phones=="OW"].GT_proba)

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
        means.append(p_df[p_df.phones!='[SIL]'].GT_proba.mean())
        medians.append(p_df[p_df.phones!='[SIL]'].GT_proba.median())
    mismatch_proba_means=np.histogram(means)
    mismatch_proba_medians=np.histogram(medians)

    x,kde2_x=distrib(means)
    plt.plot(x, kde2_x)
    plt.savefig('proba_means_gkde.png')

    # # intersection of distributions
    # idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()
    # optimal_threshold=x[idx]

    mismatch_proba_means=pd.DataFrame(mismatch_proba_means).T
    mismatch_proba_means.columns=['N','target proba']
    mismatch_proba_medians=pd.DataFrame(mismatch_proba_medians).T
    mismatch_proba_medians.columns=['N','target proba']

    match_rate=1-len(pred_df[pred_df.phones!=pred_df.pred_phones_audio])/len(pred_df)
    pred_df=pred_df[pred_df.pred_phones_audio!='[SIL]']
    match_rate=1-len(pred_df[pred_df.phones!=pred_df.pred_phones_audio])/len(pred_df)

    
    # an example of underlying align_phones
    r=df.iloc[2562]
    s,fs=librosa.load(r.audio_file_url, sr=16000)
    phonetics=r.cmu_phonetics
    phones=[[p] for p in sum([p.replace('|','_').split('_') for p in phonetics.split(' ')],[])]
    model.align_phones(s, phones)

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

    from src.text_processing import split_phonetics, remove_stress_annots
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


def phoneme_confusions(phonemes=cmu_vowels, performance_function=pContrast_on_synth_words, n=None, accent=None, name='plots/consonants_confusions_on_synth_words'):
    results={}
    predictions={}
    for p in sorted(phonemes):
        if p in cmu_vowels: p+='1'
        preds, rates=performance_function(target_phones=p, n=n, alternatives=phonemes, accent=accent)
        # _, predictions[v].to_csv('performance_results/vowel_accuracies_'+v+'.csv')
        results[p]=rates
        predictions[p]=preds
        print('phoneme')
        print(results[p])
    
    plot_confusion_results(results, name=name+'_'+accent)

    return results

def phoneme_confusion_experiments(label='w2v_pca_99_knn_5_cos_w'):
    # from DL_accuracy_performance import *
    phoneme_confusions(phonemes=cmu_vowels, performance_function=pContrast_on_synth_words, n=100, accent='UK', name='plots/vowels_confusions_on_synth_words_'+label)
    phoneme_confusions(phonemes=cmu_vowels, performance_function=pContrast_on_synth_words, n=100, accent='US', name='plots/vowels_confusions_on_synth_words_'+label)

    phoneme_confusions(phonemes=cmu_consonants, performance_function=pContrast_on_synth_words, n=100, accent='UK', name='plots/consonants_confusions_on_synth_words_'+label)
    phoneme_confusions(phonemes=cmu_consonants, performance_function=pContrast_on_synth_words, n=100, accent='US', name='plots/consonants_confusions_on_synth_words_'+label)


def vowels_confusions_actor_recordings(vowels=['IY1','IH1','AO1','AA1','OW1']):
    # for actors recordings, take only the true targets
    
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

def final_ed_confusions_from_audiobook_data(n=100, data_set='dev-clean'):
    eds=['T','D','IH0_D']

    predictions={}
    results={}
    for p in tqdm(eds): 
        predictions[p], rates=final_ed_from_audiobook_data(data_set=data_set, target_phones=p, n=n)
        results[p]=rates
    return predictions, results

def final_ed_confusions_for_actor_recordings():
    eds=['T','D','IH0_D']

    predictions={}
    results={}
    for p in tqdm(eds): 
        predictions[p], rates=final_ed_for_actor_recordings(target_phones=p)
        results[p]=rates
    return predictions, results


target_to_basis={
    'D':'IH0_D',
    'IH0_D':'IH0_D',
    'T':'IH0_D',
    '':'IH0_Z',
    'S':'IH0_Z',
    'Z':'IH0_Z',
    'IH0_Z':'IH0_Z',
}


def compute_start_end_confusions_from_selections(selections, model=default_model):
    targets=selections.keys()
    result_dfs={}
    results={}
    for t in targets:
        selection=selections[t]
        selection['cmu_phonetics']=selection['phonetics']
        basis=target_to_basis[t] if t in target_to_basis else t
        result_df=compute_predictions(selection, target_phones=t, tech_function=start_end_contrast_from_formatted_phonetics_audio, basis=basis, model=model)
        result_dfs[t]=result_df
    
    for t in targets:
        phonetic_detections=result_dfs[t].phonetic_detection
        d=count_values(phonetic_detections)
        d.columns=[t]
        results[t]=d

    return results, result_dfs

# pronunciation aspect: Initial consonant clusters (e.g.: thr-, pr-, spl-, scr-…)
def start_end_consonant_clusters_on_synth_words(clusters=['P_TH', 'M_P_T', 'N_TH_S'], contrast='end', n=100, model=default_model):
    # clusters=["TH_R", "P_R", "S_P_L", "S_K_R"]
    # 
    df=synth_words_data()

    df['target_word_indexes']=0
    df['target_syllable_indexes']=0
    df['contrast']=contrast
    df['fpath']=df['path']

    df=df.dropna()

    selections={}
    for cluster in clusters:
        if contrast=="start":
            df.phonetics.str.startswith(cluster)
            df_c=df[df.phonetics.str.startswith(cluster)]
        if contrast=="end":
            df.phonetics.str.endswith(cluster)
            df_c=df[df.phonetics.str.endswith(cluster)]
        selections[cluster]=df_c.sample(frac=1, random_state=0)[:n]

    results, result_dfs=compute_start_end_confusions_from_selections(selections, model=model)

    plot_confusion_results(results, name='plots/'+contrast+'_clusters')

# start_end_consonant_clusters_on_synth_words(clusters=['P_TH', 'M_P_T', 'N_TH_S'], n=100, model=default_model_charsiu, contrast='end')
# start_end_consonant_clusters_on_synth_words(clusters=["TH_R", "P_R", "S_P_L", "S_K_R"], n=100, model=default_model_charsiu, contrast='start')


from src.text_processing import unstress
def syl_confusions_on_synth_words():
    
    df=synth_words_data()
    syls=df.syl_p.apply(lambda r: set(['_'.join(el) for el in r])).tolist()
    syls=df.syl_p.apply(lambda r: set(['_'.join([unstress(p) for p in el]) for el in r])).tolist()

    syls_set=set().union(*syls)
    len(syls_set)




# final_ed_on_synth_words(n=100, model=default_model_charsiu, name='plots/final_ed_synth_words_charsiu')
# final_ed_on_synth_words(n=100, model=default_model, name='plots/final_ed_synth_words_pipeline')
def final_ed_on_synth_words(n=100, model=default_model, name='plots/final_ed_synth_words'):

    df=synth_words_data()

    df['target_word_indexes']=0
    df['target_syllable_indexes']=-1
    df['fpath']=df['path']

    df=df.dropna()
    
    df_id=df[(df.phonetics.str.endswith('AH0_D')|df.phonetics.str.endswith('IH0_D'))&df.text.str.endswith('ed')]
    df_t=df[(df.phonetics.str.endswith('_T'))&df.text.str.endswith('ed')]
    df_d=df[(~(df.phonetics.str.endswith('AH0_D')|df.phonetics.str.endswith('IH0_D'))&~df.phonetics.str.endswith('_T'))&df.text.str.endswith('ed')]

    selections={
        # '':df_no_s.sample(frac=1, random_state=0)[:n],
        'IH0_D':df_id.sample(frac=1, random_state=0)[:n],
        'T':df_t.sample(frac=1, random_state=0)[:n],
        'D':df_d.sample(frac=1, random_state=0)[:n],
    }

    
    results, result_dfs=compute_start_end_confusions_from_selections(selections, model=model)

    plot_confusion_results(results, name=name)


# final_s_on_synth_words(n=100, model=default_model_charsiu, name='plots/final_s_synth_words_charsiu')
# final_s_on_synth_words(n=100, model=default_model, name='plots/final_s_synth_words_pipeline')
def final_s_on_synth_words(n=100, model=default_model, name='plots/final_s_synth_words'):

    df=synth_words_data()

    df['target_word_indexes']=0
    df['target_syllable_indexes']=-1
    df['fpath']=df['path']

    df=df.dropna()
    
    df_iz=df[(df.phonetics.str.endswith('AH0_Z')|df.phonetics.str.endswith('IH0_Z'))&df.text.str.endswith('es')]
    df_s=df[(df.phonetics.str.endswith('_S'))&df.text.str.endswith('s')]
    df_z=df[(~(df.phonetics.str.endswith('AH0_Z')|df.phonetics.str.endswith('IH0_Z'))&~df.phonetics.str.endswith('_S'))&df.text.str.endswith('s')]


    words_no_s_candidates=set(list(df_iz.text.str[:-2])+list(df_s.text.str[:-1])+list(df_z.text.str[:-1]))
    df_no_s=df[df.text.isin(words_no_s_candidates)]

    
    selections={
        # '':df_no_s.sample(frac=1, random_state=0)[:n],
        'IH0_Z':df_iz.sample(frac=1, random_state=0)[:n],
        'S':df_s.sample(frac=1, random_state=0)[:n],
        'Z':df_z.sample(frac=1, random_state=0)[:n],
    }

    # targets=['S','Z','IH0_Z']

    
    results, result_dfs=compute_start_end_confusions_from_selections(selections, model=model)

    plot_confusion_results(results, name=name)

def final_ed_s_confusions_on_synth_words():
    final_ed_on_synth_words(n=100, model=default_model_charsiu, name='plots/final_ed_synth_words_charsiu')
    final_ed_on_synth_words(n=100, model=default_model, name='plots/final_ed_synth_words_pipeline')
    final_s_on_synth_words(n=100, model=default_model_charsiu, name='plots/final_s_synth_words_charsiu')
    final_s_on_synth_words(n=100, model=default_model, name='plots/final_s_synth_words_pipeline')

def final_s_from_artificial_data(model=default_model):
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
        result_df=compute_predictions(selections[t], target_phones=t, tech_function=start_end_contrast_from_formatted_phonetics_audio, basis='IH_Z', model=model)
        result_dfs[t]=result_df
    for t in targets:
        phonetic_detections=result_dfs[t].phonetic_detection
        d=count_values(phonetic_detections)
        d.columns=[t]
        results[t]=d
    plot_confusion_results(results, name='plots/final_s_artificial_data')

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




def pronunciation_aspects_from_audiobook_data(n=100, data_set='test-other', model=default_model):
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

    _, results=vowels_consonants_confusions_from_audiobook_data(n=n, data_set=data_set, model=model)

    with open('vowels_consonant_contrast_audiobook_w2v'+data_set+'_n_'+str(n)+'.pickle', 'wb') as handle:pickle.dump(results,handle)
    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_w2v'+data_set+'_n_'+str(n))



if __name__=='__main__':
    from DL_accuracy_performance import *
    pContrast_for_actor_recordings(target_phones='AO1')


    _,results=phoneme_confusions(phonemes=cmu_vowels, performance_function=pContrast_on_synth_words, n=100)
    plot_confusion_results(results, name='plots/vowel_confusions_on_synth_words')


    start_end_phoneme_from_audiobook_data(phoneme='HH')
    start_end_phoneme_from_audiobook_data(phoneme='S', basis='S', contrast="end")
    start_end_phoneme_from_audiobook_data(phoneme='Z', basis='Z', contrast="end")

    r=final_ed_for_actor_recordings()
    r=final_ed_for_actor_recordings('T')
    r=final_ed_for_actor_recordings("IH0_D")

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
    df_word=default_model.predict_word(s, r.split_phonetics, target_word_idx)
    
    word=r.cmu_phonetics.split(' ')[target_word_idx]

    syllables=[syl.split('_') for syl in word.split('|')]
    syllable=syllables[target_syllable_idx]
    
    import cmudict
    phone_df=pd.DataFrame(cmudict.phones())
    vowels=phone_df[phone_df.apply(lambda r: r.iloc[1][0], axis=1)=='vowel'].iloc[:,0].tolist()

    from time import time
    start=time()
    _, results=vowels_consonants_confusions_from_audiobook_data(n=None)
    duration=time()-start

    data_set='test-other'
    n=100
    _, results=vowels_consonants_confusions_from_audiobook_data(n=n, data_set=data_set)
    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_w2v'+data_set+'_n_'+str(n))

    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_test-other_w2v_n_10')

    # data_set='dev-clean'
    data_set='test-other'
    n=300
    _, results=final_ed_confusions_from_audiobook_data(n=n, data_set=data_set)
    plot_confusion_results(results, name='termination_contrast_audiobook_no_D_T_dis_'+data_set+'_w2v_n_'+str(n))

    predictions, results=final_ed_confusions_for_actor_recordings()
    plot_confusion_results(results, name='plots/termination_confusions_for_actor_recordings_D_T_dis_w2v_AH_D_post_corr')

    results=vowels_confusions_actor_recordings()
    # plot_confusion_results(results, name='vowel_contrast_proba_means_actors_w2v_mm_thresh_1_model_mailabs_umap_2_gmm_300')
    # plot_confusion_results(results, name='vowel_contrast_proba_means_actors_w2v_mm_thresh_1_model_mailabs_umap_2_bgmm_300')
    plot_confusion_results(results, name='plots/vowel_contrast_proba_means_actors_w2v_gmm_model_mailabs_umap_neighbors_30_2_gmm_300')

    results=vowels_confusions_user_recordings(frac=0.01)
    plot_confusion_results(results, name='plots/vowel_contrast_proba_means_user_data_w2v_thresh_0.2')
    

