import pandas as pd
import numpy as np
import pdb
from glob import glob
from utils.text_processing import remove_special_characters, remove_stress_annots, prefill_content

import itertools
import os

target_to_alternatives={
    "DH":["DH","TH"],
    "TH":["DH","TH"],
    "AO1":["AO1","OW1"],
    "OW1":["AO1","OW1"],
    "IH1":["IH1","IY1"],
    "IY1":["IH1","IY1"],
    # "IH0 D":['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']
    "IH0 D":['T', 'D', 'IH0 D'],
    "D":['T', 'D', 'IH0 D'],
    "T":['T', 'D', 'IH0 D']
}


graphemes_to_alternatives={
    "ie":["IY1","AY1"],
    "ea":["IY1","EH1"]
}


def process_GE_linguistic_data(path='data/GE_linguistic_data.csv'):
    """We choosed conventions for the target:
    For vowel contrasts:
    -VC1 => IH/IY
    -VC2 => AA/AO/OW

    Example for VC1:
    -If there is only one IH or IY, then it is the target
    -else, the target is the stressed vowel

    Args:
        path (str, optional): [description]. Defaults to 'data/GE_linguistic_data.csv'.

    Returns:
        [type]: [description]
    """
    df=pd.read_csv(path)

    df=df.drop(columns=['Unnamed: 14','Unnamed: 15'])

    df['word_idx']=0
    sents=df[df.text.str.contains('\*')].text.str.split(' ')

    # checks in each word if there is a "*", put one if true in a list. Then I use index() to know where is the 1
    df.loc[sents.index, 'word_idx']=sents.apply(lambda r: [int('*' in el) for i,el in enumerate(r)].index(1))

    df['target']=np.nan
    df['syl_target_idx']=np.nan
    possible_targets={'_VC2':['AO','OW','AA'],  '_VC1':['IH','IY']}
    def set_target(df, module='_VC1'):
        for i,r in df.iterrows():
            if module in r.id:
                p=r.cmu_phonetics.split(' ')[r.word_idx]
                phones=[el.split('_') for el in p.split('|')]
                syl_lens=[len(el) for el in phones]
                syl_ends=np.cumsum(syl_lens)

                chain = itertools.chain(*phones)
                phones=[el for el in chain]
                pure_phones=remove_stress_annots(phones)
                idx_targetable=[i for i,el in enumerate(pure_phones) if el in possible_targets[module]]

                targetable_phones=[phones[idx] for idx in idx_targetable]

                if len(idx_targetable)==1:
                    idx_target=idx_targetable[0]
                    target=phones[idx_target]
                    df.iloc[i, df.columns.get_loc('target')]=target
                else:
                    idx_target=idx_targetable[[i for i,el in enumerate(targetable_phones) if el[-1]=='1'][0]]
                    target=[el for i,el in enumerate(targetable_phones) if el[-1]=='1'][0]
                    df.iloc[i, df.columns.get_loc('target')]=target

                # print(target)
                for i_syl,end in enumerate(syl_ends):
                    if idx_target<end:
                        syl_idx=i_syl
                        break
                df.iloc[i, df.columns.get_loc('syl_target_idx')]=syl_idx
        return df
    
    df=set_target(df, module='_VC1')
    df=set_target(df, module='_VC2')

    # detect mistakes: 
    # -the target was successfully asigned with the code above. This allowed me to detect some annotation errors 
    # (e.g., there was no word in **, or live=layv vs liv)
    
    all_possible_targets=[[el+i for el in possible_targets['_VC1']] for i in ['0','1','2']]
    all_possible_targets = itertools.chain(*all_possible_targets)
    df[df.id.str.contains('_VC1') & ~df.target.isin(all_possible_targets)]
    
    all_possible_targets=[[el+i for el in possible_targets['_VC2']] for i in ['0','1','2']]
    all_possible_targets = itertools.chain(*all_possible_targets)
    df[df.id.str.contains('_VC2') & ~df.target.isin(all_possible_targets)]

    ED_targets=['_T','_D','_IH0_D']
    for i,r in df.loc[df.id.str.contains('_ED')].iterrows():
        p=r.cmu_phonetics.split(' ')[r.word_idx]
        phones=[el.split('_') for el in p.split('|')]
        n_syls=len(phones)

        s=remove_special_characters(r.text.split(' ')[r.word_idx])
        
        # We put T, then D, then OVERWRITE with IH0_D when necessary (so the order is important)
        if sum([p.endswith(t) for t in ED_targets]): #Phonetics of the word has to and with at least one of the terminations
            for target in ED_targets:
                if p.endswith(target):
                    df.iloc[i, df.columns.get_loc('target')]=target[1:]
                    df.iloc[i, df.columns.get_loc('syl_target_idx')]=n_syls-1
        else:
            # -if it is in _ED but the word does not end by one og the above
            # This shows potential annotation mistakes (e.g. if no word was in * *)
            print(s)
            print(r)
    
    # check that df for target!=nan is the same as df of row containing either _VC or _ED
    assert (df.loc[df.id.str.contains('_ED')|df.id.str.contains('_VC')]==df.dropna()).product().product(), 'Error: rows containing _ED or _VC should be the same as df.dropna(), because nans are for rows that do not contain a target'
    
    # set alternatives from targets
    def set_alternatives_from_target(target):
        target_to_alternatives={
            'IH':['IH','IY','AA','AO','AW','AY','ER','OY'], # from   https://docs.google.com/spreadsheets/d/1tzb7ZKQOifCHXh-Aoz4PdAKPlIquThzk80EvXW1UxLw/edit#gid=0
            'IY':['IY','IH','AA','AE','AH','AO','AW','AY','EH','ER','OW','OY','UH'],
            'AO':['AO','OW','AW','EH','ER','EY','IH','IY','OY','UH','UW'],
            'AA':['AA','OW','AW','EH','ER','EY','IH','IY','OY','UH','UW'],
            'OW':['OW','AA','AO','AE','AY','ER','EY','IH','IY','OY','UH'],
            "IH0_D":['T', 'D', 'IH0_D'],
            "D":['T', 'D', 'IH0_D'],
            "T":['T', 'D', 'IH0_D']
        }
        alternatives=float('nan')

        if target[-1] in str([0,1,2]):
            alternatives=' '.join([el+target[-1] for el in target_to_alternatives[target[:-1]]])
        elif '_'+target in ED_targets:
            alternatives=' '.join(target_to_alternatives[target])
        return alternatives
    df['alternatives']=df.target.dropna().apply(lambda r:   set_alternatives_from_target(r))

    assert (df.loc[df.id.str.contains('_ED')|df.id.str.contains('_VC')]==df.dropna()).product().product(), 'Error: rows containing _ED or _VC should be the same as df.dropna(), because nans are for rows that do not contain a target'

    # I should drop duplicates, but keeping one having target and alternatives if there is

    # for i,r in df.iterrows():
    #     if len(r.text.split(' '))>0: df.iloc[i, df.columns.get_loc('pronounciation_guide')]=float('nan')

    # df.appl
    df.to_csv('data/GE_linguistic_data_target_alternatives_syl_idx.csv', index=None)
    return df

# Old database content processing
def get_data(path_to_json='data/audio-with-analysis-ids/data.json'):
    """Get a dataframe containing info of audio recordings with sentence ids

    Args:
        path_to_json (str, optional): [description]. Defaults to 'data/audio-with-analysis-ids/data.json'.

    Returns:
        DataFrame: [description]
    """
    data=pd.read_json(path_to_json)
    return data

def get_iContrast_annotations(path='data/audio-with-analysis-ids/iContrast_data.csv'):
    df=pd.read_csv(path)

    # only those with analysisID in 3 digits have a manual annotation
    df=df[df.analysisId>100]
    annot={}
    word_idx={}
    syl_id={}
    for i,r in df.iterrows():
        annot[r.primaryKey]=r['response']
        word_idx[r.analysisId]=r['word_idx']
        syl_id[r.analysisId]=r['syl_id']
    return annot,word_idx,syl_id

def get_wordStress_annotation(path='data/audio-with-analysis-ids/wordStress_annotations.csv'):
    d=get_data()
    d=d[d.focusType=="wordstress"]

    df=pd.read_csv(path)

    wordStress_annotations={}
    for i,r in d.iterrows():
        analysisId=r.analysisId
        first_digit=int(analysisId/100)

        if first_digit>0:
            col=df[str(first_digit)]
            text=d[d.analysisId==analysisId].text.values[0]
            col_idx=df.columns.tolist().index(str(first_digit))
            match=col[col==text]

            if len(match)>0:
                row_idx=match.index[0]
                bin_annotation=df.iloc[row_idx,col_idx+3]
                bin_annotation_list=[int(el) for el in bin_annotation.split('-')]
                wordStress_annotations[analysisId]=bin_annotation_list
            else:
                print('Text not found')
                print(r)
        else:
            print('index with less than 3 digits')
            print(r)
    
    return wordStress_annotations

def get_sentenceStress_annotation(path='data/audio-with-analysis-ids/learning_content_for_analysis.xlsx'):
    xls = pd.ExcelFile(path)
    # df1 = pd.read_excel(xls, 'Sheet1')
    df2 = pd.read_excel(xls, 'Sheet3')
    df=df2[['SentenceID', 'Sentence stress' , 'binResult']].dropna()
    df.index=df['SentenceID'].astype(int)
    binDict={}
    textDict={}
    for i,r in df.iterrows():
        binDict[i]=[int(el) for el in r['binResult'].split(' ')]
        textDict[i]=r['Sentence stress']
    return binDict, textDict

# New database content processing
def build_data(F, n_F,  df_focus_word, df_data):
    glob_F=glob(F+'/*')
    records=[]
    for i,el in enumerate(n_F):
        # print(i) 
        # print(el)
        id=df_focus_word.iloc[int(el)-1]['id filipe']
        bins=df_data[df_data.id==id].bins.values[0]
        text=df_data[df_data.id==id].text.values[0]
        audio_path=glob_F[i]
        r={'bins':bins, 'text':text, 'audio_path':audio_path, 'n_marker':el, 'id':id}
        records.append(r)
    return pd.DataFrame.from_records(records)

def get_data_new_content():
    F2="data/new_content_audio/F2/Repetition_Tasks_Jessie_AmE/Repetition Task 1 (multivoices)"
    F1="data/new_content_audio/F1/Repetition Task 1"
    M1="data/new_content_audio/M1/Repetition Task 1"
    
    glob_F1=glob(F1+'/*')
    glob_M1=glob(M1+'/*')
    glob_F2=glob(F2+'/*')

    # marker number: odd=question, even=answer
    n_F2=[el.split('/')[-1].split('.')[0].split('_')[-1] for el in glob_F2]
    n_M1=[n if n[0]!='0' else n[1:] for n in [el.split('/')[-1].split('.')[0].split('_')[-1][9:] for el in glob_M1]]
    n_F1=[el.split('/')[-1].split('.')[0].split('_')[-1][6:] for el in glob_F1]

    # glob(F1+'/*')
    # glob(M1+'/*')
    
    df_data=pd.read_csv('data/syllabus_phrases_export_2021-09-15_125529.csv')
    df_data['bins']=df_data.apply(lambda r: [int('*' in el) for el in r.text.split(' ')], axis=1)
    df2=pd.read_csv('data/focus_word_phrases_id.csv')
    
    df_focus_word=df2[df2.name.str.startswith('Task1')]
    df_focus_word.index+=1

    df_F1=build_data(F1, n_F1, df_focus_word, df_data)
    df_F2=build_data(F2, n_F2, df_focus_word, df_data)
    df_M1=build_data(M1, n_M1, df_focus_word, df_data)

    df_tot=pd.concat([df_F1, df_F2, df_M1], axis=0)

    return df_tot

# Actor recordings 31/03 (GE and BE)
def actor_recordings():
    df=pd.read_csv('data/flwc-phrase-audios/flwc-phrase-audios.csv')
    df['audio_file_url']='data/flwc-phrase-audios/'+df['audio_file_url']

    # those who don't have NaN in target_phoneme
    
    df_pContrast=df.loc[df.target_phoneme.dropna().index]
    df_sentence_stress=df.loc[df.stress_category.dropna().index]
    df_word_stress=df.drop(df_pContrast.index).drop(df_sentence_stress.index)

    return df


# User recordings data
def build_user_data_df():
    # there are files at this level (when there is no date), I ignore them for now, it's like 2% of the files
    # all_files=glob('data/flwc-recordings/*/*')

    all_files=glob('data/flwc-recordings/*/*/*')
    len(all_files)

    # different possible extensions in dataset
    extensions=set([f.split('.')[-1] for f in all_files if '.' in f])

    # get filenames
    fpaths=[f for f in all_files if f.split('.')[-1] in extensions]
    user_ids=[f.split('/')[-3] for f in fpaths]
    # user_ids=set([f.split('/')[-2] for f in fpaths])

    fnames=[os.path.split(f.split('.')[0])[-1] for f in all_files if f.split('.')[-1] in extensions]
    len(fpaths)
    len(fnames)

    # big number corresponding to the specific audio file
    audio_file_idx=[f.split('.')[0].split('__')[-1] for f in fnames]

    # remove user info from filename because it can contain '__' which is the separator and thus messes with parsing if not removed
    # and I keep the information inside user_ids list
    # metadata=[f.replace(user_ids[i]+'__', '') for i,f in enumerate(fnames)]
    # len(metadata)

    # split with '__' separator of metadata, discard last columns because user_id may contain '__' which messes the rest of the columns
    df=pd.DataFrame([f.split('__') for f in fnames]).iloc[:,:3]
    df.columns=['exercise_id', 'processing_status', 'answer']
    df['user_id']=user_ids
    df['audio_file_idx']=audio_file_idx
    df['fpath']=fpaths
    df['fname']=fnames

    # 0=error,    1=success
    df.loc[df.processing_status=='P0','processing_status']=0
    df.loc[df.processing_status=='P1','processing_status']=1

    # 0=error,    1=  , 2=  
    df.loc[df.answer=='A0','answer']=0
    df.loc[df.answer=='A1','answer']=1
    df.loc[df.answer=='A2','answer']=2

    

    # df[df.processing_status==0]
    # df[df.processing_status==1]

    # df[df.answer==0]
    # df[df.answer==1]
    # df[df.answer==2]

    # df[df.processing_status==0][df.answer==0]
    # df[df.processing_status==0][df.answer!=0]

    return df

def get_errors_examples():
    
    df=build_user_data_df()
    df_errors=df[df.processing_status==0]
    len(df_errors)/len(df)

    ex_ids=df_errors.exercise_id.unique()

    n_errors={}
    for ex_id in ex_ids:
        n_errors[ex_id]=len(df_errors[df_errors.exercise_id==ex_id])

    exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')
    df_errors['module_type']=df_errors.exercise_id.apply(lambda r: exercise_data[exercise_data.exercise_id==r].module_type.values[0])


    # sort dict by value
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    exs_sort_by_n_errors=list({k: v for k, v in sorted(n_errors.items(), key=lambda item: item[1])})[::-1]

    return df_errors, exs_sort_by_n_errors

def final_s_artificial_data(path="data/Final s - voices for test/exercises_test.csv"):
    df=pd.read_csv(path)
    df['path']=os.path.split(path)[0]+'/audios/'+df.soundfiles_name
    content=prefill_content(df["Text with target"].tolist())
    df['cmu_phonetics']=content.cmu_phonetics
    content['path']=df.path

    content.apply(lambda r: sum(['*' in el for el in r.text.split()]), axis=1)
    
    content['word_idx']=0
    sents=content[content.text.str.contains('\*')].text.str.split(' ')
    # checks in each word if there is a "*", put one if true in a list. Then I use index() to know where is the 1
    content.loc[sents.index, 'word_idx']=sents.apply(lambda r: [int('*' in el) for i,el in enumerate(r)].index(1))

    content["target"]=df["Target phoneme"]

    # content["target_syllable_indexes"]=-1
    content["target_syllable_indexes"]=[[-1]]*len(content)

    to_cmu={"/z/":"Z", "/s/":"S", "/iz/": "IH_Z"}

    content["target_phones"]=content["target"].apply(lambda r: to_cmu[r])

    return content
