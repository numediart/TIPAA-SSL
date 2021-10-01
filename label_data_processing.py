import pandas as pd
import numpy as np
import pdb
from glob import glob
from shutil import copy
import cmudict
import os
from text_processing import phonetics_from_sentence, remove_special_characters, remove_stress_annots
import uuid
from htk_utils import process_grammar
import itertools

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

# Data processing
def get_data(path_to_json='data/audio-with-analysis-ids/data.json'):
    """Get a dataframe containing info of audio recordings with sentence ids

    Args:
        path_to_json (str, optional): [description]. Defaults to '../audio-with-analysis-ids/data.json'.

    Returns:
        DataFrame: [description]
    """
    data=pd.read_json(path_to_json)
    return data

def get_iContrast_annotations(path='../audio-with-analysis-ids/iContrast_data.csv'):
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

def get_wordStress_annotation(path='../audio-with-analysis-ids/wordStress_annotations.csv'):
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

def get_sentenceStress_annotation(path='../audio-with-analysis-ids/learning_content_for_analysis.xlsx'):
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

def make_dct_all_phones_from_phonetics(phonetics, path='test.dct'):
    """This functions generates a dct file for the wordStress module. Phonemes are detailed
    For vowels, the three possibilities of stressed are put as alternatives (0,1,2)

    Args:
        phonetics ([type]): [description]
        path (str, optional): [description]. Defaults to 'test.dct'.
    """
    alternatives=[['AA','AE'],['EH','ER'],['IH','IY'],['AO','OW']] #, ['N','NG'], ['T','S','TH','D']]
    # to access easier a list of alternatives corresponding to one phoneme
    alternatives_dict={}
    for alt in alternatives:
        for el in alt: alternatives_dict[el]=alt

    lines=[]
    n_previous_phonemes=0
    for w_i, word in enumerate(phonetics):
        for p_i, p in enumerate(word):
            if p[-1] in str([0,1,2]):
                # vowels of cmu end by 0,1 or 2
                if p[:-1] in alternatives_dict:
                    for alt in alternatives_dict[p[:-1]]:
                        for n in [0,1,2]: lines.append('p'+str(n_previous_phonemes+p_i)+' ['+'w'+str(w_i)+'_v'+str(n_previous_phonemes+p_i)+'_'+str(n)+'] '+alt+str(n))
                else:
                    for n in [0,1,2]: lines.append('p'+str(n_previous_phonemes+p_i)+' ['+'w'+str(w_i)+'_v'+str(n_previous_phonemes+p_i)+'_'+str(n)+'] '+p[:-1]+str(n))
            else:
                # consonants
                if p in alternatives_dict:
                    for alt in alternatives_dict[p]:
                        lines.append('p'+str(n_previous_phonemes+p_i)+' ['+'w'+str(w_i)+'_c'+str(n_previous_phonemes+p_i)+'_'+str(0)+'] '+alt)
                else:
                    lines.append('p'+str(n_previous_phonemes+p_i)+' ['+'w'+str(w_i)+'_c'+str(n_previous_phonemes+p_i)+'_'+str(0)+'] '+p)
        n_previous_phonemes+=len(word)
    
    # if the last phoneme is a consonant, add an alternative with a "AH0" at the end
    if lines[-1][-1] not in str([0,1,2]): 
        # lines.append('p'+str(n_previous_phonemes)+' ['+'w'+str(w_i)+'_c'+str(n_previous_phonemes)+'_'+str(1)+'] '+p+' AH0')
        lines.append(lines[-1]+' AH0')

    # adding silences and out of vocabulary possibilities
    sil_oov=["sp sp",
        "sil sil",
        "o1 gs1",
        "o2 gss2",
        "o3 gss3",
        "o4 gss4",
        "o5 gss5"]
    lines+=sil_oov
    with open(path, "w") as text_file:
        text_file.write("\n".join(lines)+"\n")

def make_dct_all_phones_from_text(sentence="shopping centre", path='test.dct'):
    phonetics=phonetics_from_sentence(sentence)
    make_dct_all_phones_from_phonetics(phonetics, path=path)

def make_generic_dct_from_phonetics(phonetics=['K AE1 L IH0 K OW0', 'HH EH1 Z IH0 T EY2 T IH0 D'], word_idx=1, target_phones='IH0 D', 
                alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'], path='test.dct'):
    """This function builds a dct file needed for htk model. It consists of a list of words and for one word a list of phoneme.
    Each line is either a word or phoneme (or in fact several phoneme). More generally each line is just one or several phoneme. But 
    in our case, one word of the sentence is detailed in one phoneme or group of phoneme (e.g. IH0 D for the termination -ed). And other words
    are in one line.

    For one phoneme, we put alternatives that can be confused by english learners so that htk model can choose what he recognizes. 
    E.g. AO1, OW1, or the default set for -ed termination

    Args:
        phonetics (list, optional): [description]. Defaults to ['K AE1 L IH0 K OW0', 'HH EH1 Z IH0 T EY2 T IH0 D'].
        word_idx (int, optional): [description]. Defaults to 1.
        target_phones (str, optional): [description]. Defaults to 'IH0 D'.
        alternatives (list, optional): [description]. Defaults to ['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'].
        path (str, optional): [description]. Defaults to 'test.dct'.
    """
    lines=[]
    for i,word in enumerate(phonetics):
        # Here we are at the level of a word of the sentence
        if i!=word_idx:
            # In case it is not the word we want to detail in several lines, we just put its phonetics in one line
            lines.append('w'+str(i)+' ['+'w'+str(i)+'] '+word)
        else:
            # Here we want to detail this specific word
            # split the word to detail in phonemes with the piece with several alternatives
            # e.g., "B L A H B L A H B L A H".split('A H')   -> ['B L ', ' B L ', ' B L ', '']
            phoneme_lists=word.split(target_phones)
            p_idx=0
            if word==target_phones:
                # a particular case fort which the word is only one phoneme and it is the one we study
                alternative_phonemes=['p'+str(p_idx)+' ['+'p'+str(p_idx)+'_'+str(i)+']'+' '+p for i,p in enumerate(alternatives)]
                p_idx+=1
                lines+=alternative_phonemes
            else:
                alternative_phonemes=['p'+str(p_idx)+' ['+'p'+str(p_idx)+'_'+str(i)+']'+' '+p for i,p in enumerate(alternatives)]
                p_idx+=1
                # lines+=alternative_phonemes
                all_phones=alternative_phonemes
                for phoneme_list in phoneme_lists:
                    if phoneme_list!='':
                        # we detail phonemes for the target word
                        phoneme_list=list(filter(None, phoneme_list.split(' ')))
                        # print(phoneme_list)
                        # we list the phonemes
                        # print(p_idx)
                        phonemes=['p'+str(i+p_idx)+' ['+'p'+str(i+p_idx)+']'+' '+p for i,p in enumerate(phoneme_list) if p !='']
                        # print(phonemes)
                        # we list alternatives
                        p_idx+=len(phonemes)
                        # print(p_idx)
                        alternative_phonemes=['p'+str(p_idx)+' ['+'p'+str(p_idx)+'_'+str(i)+']'+' '+p for i,p in enumerate(alternatives)]
                        p_idx+=1
                        all_phones+=(phonemes+alternative_phonemes)
                        
                try:
                    # if we just want tu put Alternative phones between Sequences, we remove the last (first) time  
                    # S A S A S -> S S S -> S A S A S A -> S A S A S
                    # else we let it/them (first, last or both)    A S A S A S A -> S S S -> A S A S A S A 
                    # import pdb;pdb.set_trace()
                    if len(phoneme_lists)>0:
                        if phoneme_lists[-1]!='':
                            all_phones=all_phones[:-len(alternative_phonemes)]
                        if phoneme_lists[0]!='':
                            all_phones=all_phones[len(alternative_phonemes):]
                except NameError:
                    pass
                lines+=all_phones
    # adding silences and out of vocabulary possibilities
    sil_oov=["sp sp",
        "sil sil",
        "o1 gs1",
        "o2 gss2",
        "o3 gss3",
        "o4 gss4",
        "o5 gss5"]
    lines+=sil_oov
    with open(path, "w") as text_file:
        try:
            text_file.write("\n".join(lines)+"\n")
        except TypeError:
            import pdb;pdb.set_trace()

def make_grammar_from_dct(path_dct='test.dct',
                        path_grammar='test.txt'):
    """Make a grammar file from a dct file. We assume that 
    -we are studying one word in the sentence, i.e., one word is
    segmented in phonemes 
    -OR  zero word, i.e., no word is segmented in phonemes (this the case for sentenceStress)
    -OR we work with a sequence of phonemes (that can in fact be one or several words, that does not matter)

    Args:
        p (dict): params from set_params function
    """
    print(path_dct)
    # dct=p['inputPhoneticTranscription']
    df=pd.read_csv(path_dct, header=None)
    # symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
    o_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='o', axis=1)
    p_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='p', axis=1)
    w_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='w', axis=1)

    # drop consecutive duplicates to see if the are phonemes between words
    # word_spots=w_list.loc[w_list.shift() != w_list]

    symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()

    try:
        ps=df[p_list].apply(lambda r:r.str.split(' ')[0][0], axis=1)
        if len(ps)>0:
            ps=ps.unique()
    except:
        pdb.set_trace()
    ws=df[w_list].apply(lambda r:r.str.split(' ')[0][0], axis=1)#.unique()
    
    # the diff evaluates if there is a jump of indices, meaning that there are phoneme between words: words -> phonemes -> words
    listA=np.diff(ws.index)-1
    # this locates where is the jump (we assume here there is one or zero)
    res = [i for i, val in enumerate(listA) if val]
    if len(res)>0:
        # this is the case : words -> phonemes -> words
        ws1=ws.iloc[:res[0]+1].unique()
        ws2=ws.iloc[res[0]+1:].unique()
    else:
        # This is the case  phonemes -> words   OR    words -> phonemes

        if symbols[0][0]=='w':
            # This is the case : words -> phonemes
            ws2=[]
            try:
                if len(ws)>0:
                    ws1=ws.unique()
                else:
                    ws1=[]
            except:
                pdb.set_trace()
        else:
            # This is the case : phonemes -> words
            ws1=[]
            try:
                if len(ws)>0:
                    ws2=ws.unique()
                else:
                    ws2=[]
            except:
                pdb.set_trace()
    try:
        os=df[o_list].apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
    except:
        pdb.set_trace()

    str1="$bla = [o4 o4 o4 o4 o4 o4];"
    if len(ps)>0:
        str2 ="$phrase = (("+' sp '.join(ws1)+" sp (("+' '.join(ps)+") | $bla) sp "+' sp '.join(ws2)+") | ({$bla}));"
    else:
        str2 ="$phrase = (("+' sp '.join(ws1)+" sp "+' sp '.join(ws2)+") | ({$bla}));"
    str3="(({sil} | sp) $phrase ({sil} | sp))"
    # with open(p['inputGrammar'], "w") as text_file:
    with open(path_grammar, "w") as text_file:
        text_file.write("\n".join([str1,str2,str3]))
    return "\n".join([str1,str2,str3])




def make_all_phones_annotation_files_from_phonetics(
    rID,
    phonetics=[['SH', 'AA1', 'P', 'IH0', 'NG'], ['S', 'EH1', 'N', 'T', 'ER0']]):
    make_dct_all_phones_from_phonetics(phonetics, path='inputs/'+rID+'.dct')
    make_grammar_from_dct(path_dct='inputs/'+rID+'.dct',path_grammar='inputs/'+rID+'.txt')
    process_grammar(rID)

def make_pContrast_annotation_files_from_phonetics(rID,
                phonetics=[['T', 'ER1', 'N', 'D'], ['ER0', 'AW1', 'N', 'D']], word_idx=0, target_phones='D', 
                alternatives=target_to_alternatives['D']):
    phonetics=[' '.join(w) for w in phonetics]
    make_generic_dct_from_phonetics(phonetics=phonetics, word_idx=word_idx, target_phones=target_phones, alternatives=alternatives, path='inputs/'+rID+'.dct')
    make_grammar_from_dct(path_dct='inputs/'+rID+'.dct',path_grammar='inputs/'+rID+'.txt')
    process_grammar(rID)



def make_all_phones_annotation_files(
    rID,
    text='I would love to go to ireland !'
    ):
    """This function generates all phones annotation files (dct and grammar) and save them in "inputs" with the rand_fileName
    then updates the default path to point to them in parameters dictionnary

    Args:
        p ([type], optional): [description]. Defaults to set_params().
        text (str, optional): [description]. Defaults to 'I would love to go to ireland !'.

    Returns:
        dict: parameters dictionnary
    """
    make_dct_all_phones_from_text(text, path='inputs/'+rID+'.dct')
    make_grammar_from_dct(path_dct='inputs/'+rID+'.dct',path_grammar='inputs/'+rID+'.txt')
    process_grammar(rID)


def make_pContrast_annotation_files(rID,
                text="turned around",word_idx=0, target_phones='D', 
                alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D']):
    phonetics=phonetics_from_sentence(text)
    make_pContrast_annotation_files_from_phonetics(rID, phonetics=phonetics, word_idx=word_idx, target_phones=target_phones, alternatives=alternatives)



# Obsolete functions backup
if False:    
    def set_params(
        # waveFileAddress='/root/flowchase/sent.wav',
        # waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav',
        # sentenceID=1,
        # basename='phrase_',
        # fs_target = 16000, # the target sampling frequency
        # modelName = 'libri',
        # module='sentenceStress'
        ):
        """Set parameters for an analysis task: wav, dct and grammar files as well as module to use

        Args:
            sentenceID (int, optional): [description]. Defaults to 1.
            fs_target (int, optional): [description]. Defaults to 16000.
            module (str, optional): [description]. Defaults to 'sentenceStress'.

        Returns:
            dict
        """
        # inputPhoneticTranscription_base = './lexicon/'+module+'/dct/'+basename
        # inputGrammar_base = './lexicon/'+module+'/grammar/'+basename
        
        # inputGrammar = '%s.txt' % (inputGrammar_base)
        # inputPhoneticTranscription = '%s.dct' % (inputPhoneticTranscription_base)

        params={}
        # params['waveFileAddress']=waveFileAddress
        # params['fs_target']=fs_target
        # params['inputPhoneticTranscription']=inputPhoneticTranscription
        # params['inputGrammar']=inputGrammar
        # params['modelName']=modelName
        params['rand_fileName']=str(uuid.uuid4())
        # params['sentenceID']=sentenceID

        return params


    
    def sentenceStress_make_grammars(path_dct='lexicon/sentenceStress/dct', path_grammar='lexicon/sentenceStress/grammar'):
        for el in glob(path_dct+'/*.dct'):
            gram=os.path.join(path_grammar,os.path.split(el)[-1].split('.')[0]+'.txt')
            make_grammar_from_dct(el,gram)

    def wordStress_make_dcts_grammars(path_dct='lexicon/wordStress/dct', path_grammar='lexicon/wordStress/grammar'):
        d=get_data()
        d=d[d.focusType=="wordstress"]
        for i,r in d.iterrows():
            make_dct_all_phones_from_text(r.text, path=os.path.join(path_dct, 'phrase_'+str(r.analysisId)+'.dct'))
            make_grammar_from_dct(path_dct=os.path.join(path_dct, 'phrase_'+str(r.analysisId)+'.dct'),  path_grammar=os.path.join(path_grammar, 'phrase_'+str(r.analysisId)+'.txt'))

    def ed_make_grammars(path='/mnt/c/Users/noe_t/Downloads/edAnalysis-20210330T115859Z-001/edAnalysis'):
        for el in glob(path+'/*.dct'):
            make_grammar_from_dct(el)
        IDs=pd.read_csv(path+'/ed_sentenceID.csv')
        for i,r in IDs.iterrows():
            copy(path+'/'+r[0].split('.')[0]+'.dct', path+'/phrase_'+str(r[1])+'.dct')
            copy(path+'/'+r[0].split('.')[0]+'.txt', path+'/phrase_'+str(r[1])+'.txt')

    # This is obsolete compared to make_generic_dct_from_phonetics
    def make_generic_dct_from_text(sentence="I accepted to go to spain", word_idx=1, target_phones='IH0 D', 
                    alternatives=['T', 'D', 'T AH0', 'D AH0', 'IH0 D', 'IH1 D', 'IH2 D', 'EH2 D', 'AH0 D'], path='test.dct'):

        phonetics=[cmudict.dict()[el] for el in remove_special_characters(sentence).split(' ')]
        # phonetics=phonetics_from_sentence(sentence)
        # words=['w'+str(i)+' '+' '.join(word) for i,word in enumerate(phonetics)]

        lines=[]
        for i,alternative_words in enumerate(phonetics):
            # print(alternatives)
            if i!=word_idx:
                for j,word in enumerate(alternative_words):
                    # print(word)
                    lines.append('w'+str(i)+' ['+'w'+str(i)+'_'+str(j)+'] '+' '.join(word))
            else:
                # we detail phonemes for the target word
                # TODO: I take the first alternative, may be I should extract different alternatives for each phoneme... complicated, 
                # we are not even sure it is always the same number of phonemes
                phoneme_list=phonetics[word_idx][0]
                # we list the phonemes up to the termanation
                phonemes=['p'+str(i)+' ['+'p'+str(i)+']'+' '+p for i,p in enumerate(phoneme_list)][:-len(target_phones.split(' '))]
                # we list alternatives
                alternative_phonemes=['p'+str(len(phonemes))+' ['+'p'+str(len(phonemes))+'_'+str(i)+']'+' '+p for i,p in enumerate(alternatives)]
                all_phones=phonemes+alternative_phonemes
                lines+=all_phones
        # adding silences and out of vocabulary possibilities
        sil_oov=["sp sp",
            "sil sil",
            "o1 gs1",
            "o2 gss2",
            "o3 gss3",
            "o4 gss4",
            "o5 gss5"]
        lines+=sil_oov
        with open(path, "w") as text_file:
            text_file.write("\n".join(lines))

        # hmm_phones=pd.read_csv('model/libri/monophones', header=None)
        # hmm_phones[~hmm_phones.isin(phoneme_list)].dropna()
        # other_phones=hmm_phones[~hmm_phones.isin(phoneme_list)].dropna().iloc[:,0].tolist()
        # other_phones=[el+' '+el for el in other_phones]
        # all_phones=phonemes+other_phones
    
    # These two are now generalized with above function "make_grammar_from_dct"
    def make_grammar_from_all_phones_dct(path_dct='test.dct',path_grammar='test.txt'):
        """making grammar files corresponding to dct generated with above function

        Args:
            path_dct (str, optional): [description]. Defaults to 'test.dct'.
            path_grammar (str, optional): [description]. Defaults to 'test.txt'.

        Returns:
            [type]: [description]
        """
        df=pd.read_csv(path_dct, header=None)
        # symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
        o_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='o', axis=1)
        p_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='p', axis=1)
        w_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='w', axis=1)

        ps=df[p_list].apply(lambda r:r.str.split(' ')[0][0], axis=1)
        if len(ps)>0:
            ps=ps.unique()

        # This complicated line extract the index of the word in which is the phoneme (after selecting only phonemes and not silences and o4...)
        # e.g., for a line "p16 [w5_v16_0] AH0", it extract the 5
        w_indxs=df[df.iloc[:,0].str.contains('\[w')].apply(lambda r:r.str.split(' ')[0][1].split('_')[0][-1], axis=1)
        p_indxs=df[df.iloc[:,0].str.contains('\[w')].apply(lambda r:r.str.split(' ')[0][0][1:], axis=1)

        # to int and get diff to have locations where it switch to next word
        diff=w_indxs.apply(lambda r:int(r)).diff()
        step_ups=diff[diff==1].index
        p_step_ups=p_indxs[step_ups].tolist()

        # create a list of ps qith "sp" inserted between phonemes of different words
        ps_sp=[]
        for i,p in enumerate(ps):
            if str(i) in p_step_ups:ps_sp.append('sp')
            ps_sp.append(p)

        str1="$bla = [o4 o4 o4 o4 o4 o4];"
        str2="$phrase = ("+' '.join(ps_sp)+") | {$bla};"
        str3="(({sil} | sp) $phrase ({sil} | sp))"
        # with open(p['inputGrammar'], "w") as text_file:
        with open(path_grammar, "w") as text_file:
            text_file.write("\n".join([str1,str2,str3]))
        return "\n".join([str1,str2,str3])
    # old version with o1 | o2 | o3  kind of format, that was in 1 example of edAnalysis 
    def make_grammar_from_dct_old(path_dct='test.dct',
                            path_grammar='test.txt'):
        """Make a grammar file from a dct file. We assume that 
        -we are studying one word in the sentence, i.e., one word is
        segmented in phonemes 
        -OR  zero word, i.e., no word is segmented in phonemes (this the case for sentenceStress)

        Args:
            p (dict): params from set_params function
        """
        print(path_dct)
        # dct=p['inputPhoneticTranscription']
        df=pd.read_csv(path_dct, header=None)
        # symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
        o_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='o', axis=1)
        p_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='p', axis=1)
        w_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='w', axis=1)

        # drop consecutive duplicates to see if the are phonemes between words
        # word_spots=w_list.loc[w_list.shift() != w_list]

        symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()

        try:
            ps=df[p_list].apply(lambda r:r.str.split(' ')[0][0], axis=1)
            if len(ps)>0:
                ps=ps.unique()
        except:
            pdb.set_trace()
        ws=df[w_list].apply(lambda r:r.str.split(' ')[0][0], axis=1)#.unique()
        
        # the diff evaluates if there is a jump of indices, meaning that there are phoneme between words: words -> phonemes -> words
        listA=np.diff(ws.index)-1
        # this locates where is the jump (we assume here there is one or zero)
        res = [i for i, val in enumerate(listA) if val]
        if len(res)>0:
            # this is the case : words -> phonemes -> words
            ws1=ws.iloc[:res[0]+1].unique()
            ws2=ws.iloc[res[0]+1:].unique()
        else:
            # This is the case  phonemes -> words   OR    words -> phonemes

            if symbols[0][0]=='w':
                # This is the case : words -> phonemes
                ws2=[]
                try:
                    if len(ws)>0:
                        ws1=ws.unique()
                    else:
                        ws1=[]
                except:
                    pdb.set_trace()
            else:
                # This is the case : phonemes -> words
                ws1=[]
                try:
                    if len(ws)>0:
                        ws2=ws.unique()
                    else:
                        ws2=[]
                except:
                    pdb.set_trace()
        try:
            os=df[o_list].apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
        except:
            pdb.set_trace()
        # str1=' | '.join(os)+';\n'
        
        # str1="$bla = [o4 o4 o4 o4 o4 o4];"
        # str2="$phrase = ("+' '.join(ps_sp)+") | {$bla};"

        str1="$other = o1 | o2 | o3 | o4 | o5;\n$other_pho =  o2 | o3 | o4 | o5;"
        if len(ps)>0:
            str2 ="$phrase = (("+' sp '.join(ws1)+" sp (("+' '.join(ps)+") | {sp $other_pho }) sp "+' sp '.join(ws2)+") | {sp ($other )});"
        else:
            str2 ="$phrase = (("+' sp '.join(ws1)+" sp "+' sp '.join(ws2)+") | {sp ($other )});"
        str3="(({sil} | sp) $phrase ({sil} | sp))"
        # with open(p['inputGrammar'], "w") as text_file:
        with open(path_grammar, "w") as text_file:
            text_file.write("\n".join([str1,str2,str3]))
        return "\n".join([str1,str2,str3])

    # This is not needed anymore as I have now a function generating grammars
    def wordStress_correct_grammars(path_grammar='lexicon/wordStress/grammar'):
        """This functions modifies the manual grammars of wordStress module to remove the "$inv" possibility from the "$phrase"
        """
        for el in glob(path_grammar+'/*.txt'):
            gram=os.path.join(path_grammar,os.path.split(el)[-1].split('.')[0]+'.txt')
            df=pd.read_csv(gram, header=None)
            df.iloc[2,0]=df.iloc[2,0].replace('| $inv ','')
            df.to_csv(gram, index=None, header=None)


    
