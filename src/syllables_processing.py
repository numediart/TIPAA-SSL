
import pandas as pd
from syllabipy.sonoripy import SonoriPy, str_to_list_of_char, define_categories
import os
from tqdm import tqdm
from src.pronunciation_dictionaries import cmudict_dict

unstress = lambda el: el[:-1] if el[-1] in str([0,1,2]) else el

#### Syllables function  ####
def n_syl_SonoriPy(phonetics=['K', 'AA1', 'F', 'IY0'], mode='CMU'):
    return len(SonoriPy(phonetics, mode=mode)[0])

def syllables_data_fr(syl_sep='|'):
    from src.pronunciation_dictionaries import mfa_dicts
    from src.pronunciation_dictionaries import get_augmented_mfa_dict, lang_to_MFA_g2p_models
    mfa_dicts={lang:get_augmented_mfa_dict(lang) for lang in lang_to_MFA_g2p_models}

    # from http://www.lexique.org/  
    df=pd.read_csv('data/Lexique383.tsv', sep='\t')

    columns=['ortho','syll','nbsyll','orthosyll']
    df=df[columns]
    df=df.dropna()

    # filter out words containing a space, and dash (it's the sep, and there is conflict in their file)
    df=df[~df['ortho'].str.contains(' ')]
    df=df[~df['ortho'].str.contains('-')]

    # a weird systematic mistake in the data making me wondering if it was done by spanish native speakers. Starting s- considered as a syllable. I remove the first dash when that happens
    df.loc[(df.orthosyll.str[:2]=='s-'),'orthosyll']=df[(df.orthosyll.str[:2]=='s-')].orthosyll.apply(lambda my_str: my_str[:my_str.index('-')] + my_str[my_str.index('-')+1:])
    df.loc[(df.orthosyll.str[:3]=='ch-'),'orthosyll']=df[(df.orthosyll.str[:3]=='ch-')].orthosyll.apply(lambda my_str: my_str[:my_str.index('-')] + my_str[my_str.index('-')+1:])

    # alone -s- (or -ch-) is also wrong inside a word  (could be generalized for -[consonant_letters]-  . I tried single consonant and applied to those impactes)
    df.loc[df.orthosyll.str.contains('-s-'),'orthosyll']=df.loc[df.orthosyll.str.contains('-s-'),'orthosyll'].str.replace('-s-', '-s')
    df.loc[df.orthosyll.str.contains('-ch-'),'orthosyll']=df.loc[df.orthosyll.str.contains('-ch-'),'orthosyll'].str.replace('-ch-', '-ch')
    df.loc[df.orthosyll.str.contains('-th-'),'orthosyll']=df.loc[df.orthosyll.str.contains('-th-'),'orthosyll'].str.replace('-th-', '-th')
    df.loc[df.orthosyll.str.contains('-n-'),'orthosyll']=df.loc[df.orthosyll.str.contains('-n-'),'orthosyll'].str.replace('-n-', '-n')
    df.loc[df.orthosyll.str.contains('-j-'),'orthosyll']=df.loc[df.orthosyll.str.contains('-j-'),'orthosyll'].str.replace('-j-', '-j')
    df.loc[df.orthosyll.str.contains('-r-'),'orthosyll']=df.loc[df.orthosyll.str.contains('-r-'),'orthosyll'].str.replace('-r-', '-r')

    
    # from tqdm import tqdm
    # tqdm.pandas()
    # n_syl_in_text=df.progress_apply(lambda r: n_syl_SonoriPy(mfa_dicts['fr_FR'][r.ortho][0], mode='MFA_IPA') if r.ortho in mfa_dicts['fr_FR'] else len(r.orthosyll.split('-')), axis=1)

    # check number of syls in text == number of syls in phonetics
    n_syl_in_text=df.orthosyll.str.split('-').apply(lambda r: len(r))

    df_good_n_syl=df[df.nbsyll==n_syl_in_text]
    df_bad_n_syl=df[df.nbsyll!=n_syl_in_text]

    # extract words for which the mistake of number of syllables comes from trailing -e, -es, -ent. They are annotated with n+1 syllables
    wrong_n_syl_in_text=df_bad_n_syl.orthosyll.str.split('-').apply(lambda r: len(r))
    
    df_e=df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll+1)&(df_bad_n_syl.orthosyll.str[-1]=='e')]
    df_es=df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll+1)&(df_bad_n_syl.orthosyll.str[-2:]=='es')]
    df_ent=df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll+1)&(df_bad_n_syl.orthosyll.str[-3:]=='ent')]

    # words in -ement  . According to mfa, mfa_dicts['fr_FR']['lentement'] = [['l', 'ɑ̃', 't', 'm', 'ɑ̃']]
    # other sources like https://fr.wiktionary.org/wiki/lentement let the 2 possibilities  \lɑ̃t.mɑ̃\ ou \lɑ̃.tə.mɑ̃\  
    df_ement=df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll+1)&(df_bad_n_syl.orthosyll.str[-6:]=='e-ment')]

    # a weird systematic mistake in the dataset that looks like plural were not corrently processed
    df_s=df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll+1)&(df_bad_n_syl.orthosyll.str[-2:]=='-s')]

    
    def remove_last_syl_sep(df_n_plus_1):
        # in each syllabified word: 
        # -take all characters before and after the last "-"
        df_n_plus_1['orthosyll']=df_n_plus_1.orthosyll.apply(lambda my_str: my_str[:my_str.rfind('-')] + my_str[my_str.rfind('-')+1:])
        return df_n_plus_1
    df_e=remove_last_syl_sep(df_e)
    df_es=remove_last_syl_sep(df_es)
    df_ent=remove_last_syl_sep(df_ent)
    df_s=remove_last_syl_sep(df_s)
    df_ement=remove_last_syl_sep(df_ement)

    df_new_good_n_syl=pd.concat([df_good_n_syl, df_e, df_es, df_ent, df_s, df_ement])
    
    # not n+1 inconsistencies
    df_bad_n_syl[(wrong_n_syl_in_text!=df_bad_n_syl.nbsyll+1)]

    # mostly hiatus words
    df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll-1)]
    n_syl_SonoriPy(mfa_dicts['fr_FR']['exagérément'][0], mode='MFA_IPA')

    # SonoriPy(mfa_dicts['fr_FR']['exagérément'][0], mode='MFA_IPA')[0]
    # SonoriPy('exagérément', mode='letters')[0]

    # df_rest_bad_n_syl=df_bad_n_syl[(wrong_n_syl_in_text==df_bad_n_syl.nbsyll+1)&(df_bad_n_syl.orthosyll.str[-1]!='e')&(df_bad_n_syl.orthosyll.str[-2:]!='es')&(df_bad_n_syl.orthosyll.str[-3:]!='ent')&(df_bad_n_syl.orthosyll.str[-2:]!='-s')]


    syllables=df_new_good_n_syl[['orthosyll','ortho','nbsyll']]

    syllables.columns=['syllables','normalized_text','n_syls']  #,'n_syls_SonoriPy']

    syllables.loc[:,'syllables']=syllables.loc[:,'syllables'].str.replace('-',syl_sep)

    syllables['n_syls_SonoriPy']=None

    # d=mfa_dicts['fr_FR']
    # n_syls=[]
    # n_syls_SonoriPy=[]
    # texts=[]
    # for i,r in tqdm(syllables.iterrows()):
    #     text=''.join(r[0].split(syl_sep)).lower()
    #     texts.append(text)
    #     try:
    #         n_syls.append(int(len(r[0].split(syl_sep))))
    #     except:
    #         n_syls.append(None)
    #     try:
    #         n_syls_SonoriPy.append(n_syl_SonoriPy(d[text][0], mode='MFA_IPA'))
    #     except:
    #         n_syls_SonoriPy.append(None)

    syllables.to_csv('data/syllables_fr_FR.csv')

    return syllables



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
    n_syls_SonoriPy=[]
    texts=[]
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


def n_vowels(phonetics=['K', 'AA1', 'F', 'IY0'], mode="CMU"):
    d=define_categories(mode=mode)
    n=0
    for el in phonetics:
        # if the first symbol of a phoneme corresponds to a vowel, I count it as a vowel.
        # I need this for espeak
        # 'start'-> 's_t_ˈɑːɹ_t',   'engineer' -> "ˌɛ_n_dʒ_ɪ_n_ˈɪɹ"
        if (unstress(el.lower()) in d['vowels']) or (unstress(el.lower()[0]) in d['vowels']): n+=1
    return n


def use_tests():
    from src.pronunciation_dictionaries import get_augmented_mfa_dict, lang_to_MFA_g2p_models
    mfa_dicts={lang:get_augmented_mfa_dict(lang) for lang in lang_to_MFA_g2p_models}

    word='dépendance'
    phonetic_dict=mfa_dicts['fr_FR']
    nv=n_vowels(phonetic_dict[word][0], mode="MFA_IPA")
    syllabified_text(word, nv, syllables_df=pd.read_csv('data/syllables_fr_FR.csv'), lang="fr_FR")

    syllabified_text('dépendance', pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']))[0]