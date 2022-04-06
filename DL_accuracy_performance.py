from utils.libri_phonetization_data import build_librispeech_words_df
from utils.libri_phonetization_data import phonetics_for_row
from tqdm import tqdm
import pandas as pd
import pickle
from utils.charsiu_utils import charsiu_phone_forced_aligner
import ast

import librosa
from collections import Counter

from utils.label_data_processing import build_user_data_df, get_errors_examples
exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')

from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, phonetic_content_analysis
from utils.audio_processing import prepare_audio_file


from utils.label_data_processing import target_to_alternatives, get_sentenceStress_annotation, get_data_new_content, get_data, make_all_phones_annotation_files, actor_recordings
from utils.text_processing import *

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
# disable pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

drop_consecutive_duplicates= lambda df: df.loc[(df.shift()!=df).sum(axis=1).astype(bool)]


def stress_GE_performance_test(level='sentence'):
    df=get_data_new_content()
    stress_intensities=[]
    stress_binaries=[]
    for i,row in df.iterrows():
        formatted_phonetics=prefill_for_sentence(row.text)['cmu_phonetics']
        _, rID=prepare_audio_file(row.audio_path, speech_correction=False)
        res=stress_from_formatted_phonetics(rID,phonetics=formatted_phonetics, text=row.text, level=level)
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
        # df_all.apply(lambda r: r['stress_intensities'], axis=1)

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

        # I realized some words have all "syllables" stresses (even though they are > 1 syl). In fact, it corresponds to acronyms
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


def compute_predictions_with_syl_idx(selection, model, target_phones='AO1'):
    phonetic_detections=[]

    print('number of examples:', len(selection))
    for i,r in tqdm(selection.iterrows()):
        # phonetics=r.cmu_phonetics
        # s,fs=librosa.load(r.fpath, sr=16000)
        target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
        target_syllable_idx=ast.literal_eval(r.target_syllable_indexes)[0]

        status_audio, rID=prepare_audio_file(r.fpath, speech_correction=False) 
        res=phonemeContrast_from_formatted_phonetics_audio(
                                rID,
                                phonetics=r.cmu_phonetics, 
                                target_word_idx=target_word_idx, 
                                target_syllable_idx=target_syllable_idx,
                                target_phones=target_phones
                                )
        phonetic_detections.append(res['phonetic_detection'])
    
    return phonetic_detections

if False:
    # obsolete
    def compute_predictions(selection, model, target_phones='AO1'):
        dfs=[]
        failures=[]
        print('number of examples:', len(selection))
        for i,r in tqdm(selection.iterrows()):
            # phonetics=r.cmu_phonetics
            s,fs=librosa.load(r.fpath, sr=16000)
            target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
            df_word=model.predict_word(s, r.split_phonetics, target_word_idx)
            dfs.append(df_word)
        
        # removing stress info if necessary
        target = target_phones[:-1] if target_phones[-1] in str([0,1,2]) else target_phones

        df_targets=[]
        # select targets in words
        for df_word in dfs:
            df_targets.append(df_word[df_word.cmu_phones==target])

        df_targets=pd.concat(df_targets)
        return df_targets, failures



def count_values(phonetic_detections):
    d=Counter(phonetic_detections)
    d = pd.DataFrame.from_dict(d, orient='index')
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

    charsiu = charsiu_phone_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')
    
    df=df.sample(frac=frac, random_state=1234)

    df['split_phonetics']=df.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)

    # df_targets, _=compute_predictions(df, charsiu, target_phones=target_phones)
    phonetic_detections=compute_predictions_with_syl_idx(df, charsiu, target_phones=target_phones)

    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return phonetic_detections, d


def syllable_contrast_for_actor_recordings():
    df=actor_recordings()

    # contrast on all phones with force_and_predict (i.e. predict based on frames allocated to a phoneme)
    charsiu = charsiu_phone_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')
    pred_dfs=[]
    for i,r in tqdm(df.iterrows()):
        s,fs=librosa.load(r.audio_file_url, sr=16000)
        split_phonetics=sum([p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], [])
        p_df=charsiu.force_and_predict(s, split_phonetics)
        pred_dfs.append(p_df)
    pred_df=pd.concat(pred_dfs)
    match_rate=1-len(pred_df[pred_df.cmu_phones!=pred_df.pred_phones_audio])/len(pred_df)
    pred_df=pred_df[pred_df.pred_phones_audio!='[SIL]']
    match_rate=1-len(pred_df[pred_df.cmu_phones!=pred_df.pred_phones_audio])/len(pred_df)
    
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
    
    




def pContrast_for_actor_recordings(target_phones='AO1'):
    charsiu = charsiu_phone_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')

    # df=pd.read_csv('data/exercise_data_export.csv')
    # df['audio_file_url']='data/scaleway-audio-files/'+df['audio_file_url']
    df=actor_recordings()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]    
    selection['split_phonetics']=selection.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)
    selection['fpath']=selection.audio_file_url
    # selection['audio_file_idx']=selection.fk_audio_recording_id

    phonetic_detections=compute_predictions_with_syl_idx(selection, charsiu, target_phones=target_phones)

    d=count_values(phonetic_detections)
    d.columns=[target_phones]

    return phonetic_detections, d

def pContrast_from_audiobook_data(data_set='dev-clean', target_phones='AO1', n=None):
    libri_words_df=build_librispeech_words_df(data_set=data_set, n=n)
    # there is a tag <unk> when a word is unknown. I filter out the files corresponding to these before performance test
    libri_words_df=libri_words_df[~libri_words_df.file_idx.isin(libri_words_df[libri_words_df.word=='<unk>'].file_idx.unique())]
    selection=libri_words_df[libri_words_df.phones.str.contains(target_phones)]

    # charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
    charsiu = charsiu_phone_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')

    selection['split_phonetics']=selection.apply(lambda r: [p.split(' ') for p in phonetics_for_row(r, libri_words_df)], axis=1)

    dfs=[]
    # inference on librispeech takes approx 1.5s/sentence. (they are long phrases)
    for i,row in tqdm(selection.iterrows()):
        df_word=charsiu.predict_word(row.wav_path, row.split_phonetics, row.word_idx)
        dfs.append(df_word)
    
    # removing stress info if necessary
    target = target_phones[:-1] if target_phones[-1] in str([0,1,2]) else target_phones

    df_targets=[]
    # select targets in words
    for df_word in dfs:
        df_targets.append(df_word[df_word.cmu_phones==target])

    df_targets=pd.concat(df_targets)
    
    d=count_values(df_targets.pred_phones_audio.tolist())
    d.columns=[target_phones]

    return df_targets, d



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

def vowels_consonants_confusions_from_audiobook_data(n=100, data_set='dev-clean'):
    import cmudict
    phones=cmudict.phones()

    cmu_vowels=[p[0]+'1' for p in phones if p[1][0]=='vowel']
    cmu_consonants=[p[0] for p in phones if p[1][0]!='vowel']

    predictions={}
    results={}
    for v in tqdm(cmu_vowels): 
        predictions[v], rates=pContrast_from_audiobook_data(data_set=data_set, target_phones=v, n=n)
        results[v]=rates
    
    for c in tqdm(cmu_consonants):
        predictions[c], rates=pContrast_from_audiobook_data(data_set=data_set, target_phones=c, n=n)
        results[c]=rates

    return predictions, rates


import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_results(results, name='vowel_contrast_actors_w2v'):
    plt.clf()
    fig, axn = plt.subplots(1, len(results))
    fig.set_size_inches(20, 6)
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

if __name__=='__main__':

    pContrast_for_actor_recordings(target_phones='AO1')

    stress_GE_performance_test(level='word')
    stress_GE_performance_test(level='sentence')

    results=vowels_confusions_user_recordings(frac=0.01)
    
    target_phones='AO1'
    df=actor_recordings()
    # those who don't have NaN in target
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    selection=df_pContrast[df_pContrast.target_phoneme==target_phones]    
    charsiu = charsiu_phone_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms', device='cpu')
    selection['split_phonetics']=selection.apply(lambda r: [p.replace('|','_').split('_') for p in r.cmu_phonetics.split(' ')], axis=1)
    selection['fpath']=selection.audio_file_url

    r=selection.iloc[24]
    
    s,fs=librosa.load(r.fpath, sr=16000)

    target_word_idx=ast.literal_eval(r.target_word_indexes)[0]
    target_syllable_idx=ast.literal_eval(r.target_syllable_indexes)[0]
    df_word=charsiu.predict_word(s, r.split_phonetics, target_word_idx)
    
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

    predictions, results=vowels_consonants_confusions_from_audiobook_data(n=30, data_set='test-other')
    plot_confusion_results(results, name='vowels_consonant_contrast_audiobook_test-other_w2v_n_10')


    results=vowels_confusions_actor_recordings()
    plot_confusion_results(results, name='vowel_contrast_proba_means_actors_w2v')

    results=vowels_confusions_user_recordings(frac=0.01)

    


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

