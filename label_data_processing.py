import pandas as pd
import numpy as np
import pdb
from glob import glob
from shutil import copy

from text_processing import phonetics_from_sentence

# Data processing
def get_data(path_to_json='../audio-with-analysis-ids/data.json'):
    """Get a dataframe containing info of audio recordings with sentence ids

    Args:
        path_to_json (str, optional): [description]. Defaults to '../audio-with-analysis-ids/data.json'.

    Returns:
        DataFrame: [description]
    """
    data=pd.read_json(path_to_json)
    return data

def get_iContrast_annotations(path='../audio-with-analysis-ids/iContrast_data.csv'):
    d=get_data()
    d=d[d.focusType=="shortIlongI"]
    df=pd.read_csv(path)
    annot={}
    for i,r in df.iterrows():
        annot[r.analysisId]=r['response (short=0, long=1)']
    return annot

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

def make_generic_dct_from_text(sentence="Where's the best place to have coffee ?", word_id=None, path='test.dct'):

    hmm_phones=pd.read_csv('model/libri/monophones', header=None)
    phonetics=phonetics_from_sentence(sentence)
    words=['w'+str(i)+' '+' '.join(word) for i,word in enumerate(phonetics)]
    
    phoneme_list=[]
    for el in phonetics:
        phoneme_list+=el
    phonemes=['p'+str(i)+' '+p for i,p in enumerate(phoneme_list)]

    # hmm_phones[~hmm_phones.isin(phoneme_list)].dropna()
    other_phones=hmm_phones[~hmm_phones.isin(phoneme_list)].dropna().iloc[:,0].tolist()
    other_phones=[el+' '+el for el in other_phones]

    all_phones=phonemes+other_phones

    
    with open(path.split('.')[0]+'.txt', "w") as text_file:
        text_file.write("\n".join(all_phones))

    # if word_id is None:

    # elif word_id<len(phonetics):
        
    # else:
    #     print('error: word id is greater than number of words')


def make_grammar_from_dct(path='/mnt/c/Users/noe_t/Downloads/edAnalysis-20210330T115859Z-001/edAnalysis/he decided to go back to spain.dct.txt'):
    """For edAnalysis, make a grammar file from a dct file (may be generalizable in the future if useful)

    Args:
        p (dict): params from set_params function
    """
    print(path)
    # dct=p['inputPhoneticTranscription']
    df=pd.read_csv(path, header=None)
    # symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
    o_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='o', axis=1)
    p_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='p', axis=1)
    w_list=df.apply(lambda r:r.str.split(' ')[0][0][0]=='w', axis=1)

    # drop consecutive duplicates to see if the are phonemes between words
    # word_spots=w_list.loc[w_list.shift() != w_list]

    symbols=df.apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()

    try:
        ps=df[p_list].apply(lambda r:r.str.split(' ')[0][0], axis=1).unique()
    except:
        pdb.set_trace()
    ws=df[w_list].apply(lambda r:r.str.split(' ')[0][0], axis=1)#.unique()
    
    # the diff evaluates if there is a jump of indices, meaning that there are phoneme between words
    listA=np.diff(ws.index)-1
    # this locates where is the jump (we assume here there is one or zero)
    res = [i for i, val in enumerate(listA) if val]
    if len(res)>0:
        ws1=ws.iloc[:res[0]+1].unique()
        ws2=ws.iloc[res[0]+1:].unique()
    else:
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
    str1="$other = o1 | o2 | o3 | o4 | o5;\n\
$other_pho =  o2 | o3 | o4 | o5;"
    str2 ="$phrase = (("+' sp '.join(ws1)+" sp (("+' '.join(ps)+") | {sp $other_pho }) sp "+' sp '.join(ws2)+") | {sp ($other )});"
    str3="\
(\n\
({sil} | sp) $phrase ({sil} | sp)\
)"
    # with open(p['inputGrammar'], "w") as text_file:
    with open(path, "w") as text_file:
        text_file.write("\n".join([str1,str2,str3]))
    return "\n".join([str1,str2,str3])

def ed_make_grammars(path='/mnt/c/Users/noe_t/Downloads/edAnalysis-20210330T115859Z-001/edAnalysis'):
    for el in glob(path+'/*.dct'):
        make_grammar_from_dct(el)
    IDs=pd.read_csv(path+'/ed_sentenceID.csv')

    for i,r in IDs.iterrows(): 
        copy(path+'/'+r[0].split('.')[0]+'.dct', path+'/phrase_'+str(r[1])+'.dct')
        copy(path+'/'+r[0].split('.')[0]+'.txt', path+'/phrase_'+str(r[1])+'.txt')

