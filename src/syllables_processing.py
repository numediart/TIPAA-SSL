
import pandas as pd
from syllabipy.sonoripy import SonoriPy, str_to_list_of_char, define_categories
import os

from src.pronunciation_dictionaries import cmudict_dict

#### Syllables function  ####
def n_syl_SonoriPy(phonetics=['K', 'AA1', 'F', 'IY0']):
    return len(SonoriPy(phonetics)[0])


def syllables_data(syl_sep='|'):
    """This functions builds our dataset truth about text syllables. 
    It uses an existing dataset and add our extension to it. It also get data about the number of syllables according to SonoriPy
    When it is possible we keep only consitent solutions (in terms of n of syllables) and discard others.
    The result is saved in a CSV that is used for syllabification. See "syllabified_text()"

    Args:
        syl_sep (str, optional): [description]. Defaults to '|'.

    Returns:
        [type]: [description]
    """
    # http://www.delphiforfun.org/programs/Syllables.htm
    # syllables=pd.read_csv('Syllables.txt',sep='=', header=None)
    syllables=pd.read_csv('data/mhyph.txt', header=None)
    # mhyph_syl_sep=syllables[0][0][5]
    # syllables.iloc[:,0]=syllables.iloc[:,0].str.replace(mhyph_syl_sep,syl_sep)

    # This file contains additional solutions that we can change. For example, I added "tem|pera|ture"
    # because only tem|pe|ra|ture was present. The following of the function will take care of choosing
    # the right one so that it is consistent with SonoriPy's prediction
    syllables_add=pd.read_csv('data/mhyph_add.txt', header=None)
    syllables=pd.concat([syllables,syllables_add])

    d=cmudict_dict
    syllables=syllables.dropna()  # there is one row that is nan...

    # syllables.columns=['word', 'syllables']
    syllables.columns=['syllables']

    n_syls=[]
    n_vowels_cmu=[]
    n_syls_SonoriPy=[]
    texts=[]
    phonetics=[]
    for i,r in syllables.iterrows():
        text=''.join(r[0].split(syl_sep)).lower()
        texts.append(text)
        try:
            n_syls.append(int(len(r[0].split(syl_sep))))
        except:
            n_syls.append(None)
        try:
            n_syls_SonoriPy.append(n_syl_SonoriPy(d[text][0]))
        except IndexError:
            n_syls_SonoriPy.append(None)

    syllables.syllables=syllables.syllables.str.lower()
    
    syllables['normalized_text']=texts
    # syllables['phonetics']=phonetics

    syllables['n_syls']=n_syls
    syllables['n_syls_SonoriPy']=n_syls_SonoriPy

    syllables.to_csv('data/syllables.csv')

    return syllables

# syllables_data()

def syllabified_text(word, n, syllables_df=pd.read_csv('data/syllables.csv'), lang="en_GB"):
    """Construct syllabified word from a word.

     text with syllable segmentation is done with several rules/steps:
        -Use our syllables dataset
        -If does not exist, check if only 1 vowel (trivial because 1 syllable) ==>  in that case syllable=text
        -If not, fall back to use SonoriPy (sonority sequencing principle) with letters.
    It is less accurate than with phonemes but we use it only on fallback.
    I improved this part with logic in this (trailing "e", "-ed", "-es" ):
    https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word

    Args:
        word ([type]): [description]
        syllables_df ([type]): [description]

    Returns:
        [type]: [description]
    """
    # if phones!=[]:
    if n==1:
        syls_text=word
        used_method='1 vowel = 1 syl'
        return syls_text, used_method
    try:
        # If there exist choices with consistent number of syllables, let's take the first one of these.
        a=syllables_df[syllables_df.n_syls==syllables_df.n_syls_SonoriPy].loc[syllables_df.normalized_text==word]
        if len(a)>0:
            syls_text=a.syllables.values[0]
        else:
            syls_text=syllables_df[syllables_df.normalized_text==word].syllables.values[0]
        used_method='dataset'
    except IndexError:
        # for the final -ed, remove the "e" except if "-ded" or "-ted" or "-ired"
        # We remember if we did to insert back the "e" after syllabification
        modified_ed=False
        # smiles -> remove the e,   raises -> don't
        modified_es=False
        trailing_e=False

        from syllabipy.sonoripy import define_categories
        d=define_categories('letters')
        vowels,nasals,fricatives,affricates,stops=d['vowels'],d['nasals'],d['fricatives'],d['affricates'],d['stops']

        if lang=="en_GB" or lang=="en_US":
            if word[-2:]=="ed" and word[-3] not in ['t','d'] and word[-4:]!="ired":
                word=word[:-2]+'d'
                modified_ed=True
            elif word[-2:]=="es" and word[-3] not in ['s','c','g','x'] and word[-4:]!='ches' and word[-4:]!='shes' and word[-3:]!='les': #this last is treated hereafter because it dependes
                word=word[:-2]+'s'
                modified_es=True
            elif word[-3:]=="les" and word[-4] not in stops: # do it for e.g. "smiles", but not gor "angles, muscles, articles, ..."
                word=word[:-2]+'s'
                modified_es=True
            elif word[-1]=="e" and word[-2:]!='le': #this last one is treated hereafter because it depends
                word=word[:-1]
                trailing_e=True
            elif word[-2:]=="le" and word[-3] not in stops: # do it for e.g. "smile", but not for "angle, muscle, article, ..."
                word=word[:-1]
                trailing_e=True
        
        if lang=="fr_FR":
            if word[-2:]=="es":
                word=word[:-2]+'s'
                modified_es=True
            elif word[-1]=="e": #this last one is treated hereafter because it depends
                word=word[:-1]
                trailing_e=True
        
        letters_by_syl=SonoriPy(str_to_list_of_char(word), mode='letters')[0]
        syls_text='|'.join([''.join(syl) for syl in letters_by_syl])

        if modified_ed:
            syls_text=syls_text[:-1]+"ed"
        if modified_es:
            syls_text=syls_text[:-1]+"es"
        if trailing_e:
            syls_text+='e'
        
        # yet another type of correction for some terminations. Maybe some others from above might be implemented this way and vice versa
        if lang=="en_GB" or lang=="en_US":
            if syls_text[-3:]=="ism":
                syls_text=syls_text[:-3]+"i|sm"
            if syls_text[-4:]=="isms":
                syls_text=syls_text[:-4]+"i|sms"
            if syls_text[-4:]=="ithm":
                syls_text=syls_text[:-4]+"i|thm"
            if syls_text[-5:]=="ithms":
                syls_text=syls_text[:-5]+"i|thms"

        used_method='SonoriPy'
        return syls_text, used_method
    return syls_text, used_method
