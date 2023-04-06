
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
    # from src.pronunciation_dictionaries import mfa_dicts
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
    n_syl_SonoriPy(mfa_dicts['fr_FR']['évoluez'][0], mode='MFA_IPA')

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

    # syllables.iloc[:,0].str.replace('|','').drop_duplicates()   
    # syllables[syllables.iloc[:,0].str.replace('|','').duplicated()]

    # syllables.dropna()[syllables.dropna().iloc[:,0].str.replace('|','').str.startswith('ingredien')]

    d=cmudict_dict
    syllables=syllables.dropna()  # there is one row that is nan... (maybe just a \n at the end of one of the files)

    # syllables.columns=['word', 'syllables']
    syllables.columns=['syllables']

    # correct systematic mistake in words with "ire" in them
    # 869                 ac|quired taste
    # 870                   ac|quire|ment
    # 934                     ac|ro|spire
    # 1338                        ad|mire
    # 1339                       ad|mired
    # 2495                   all-fired|ly
    # 3072               al|ley|fired|est
    # 7323                  Ar|gyll|shire
    # 7688                   ar|thro|dire
    # 8031                        as|pire
    # 8032                       as|pired
    # 8642                        at|tire
    syllables_ire=syllables[syllables['syllables'].str.lower().str.contains('ire')&~syllables['syllables'].str.lower().str.contains('aire')&~syllables['syllables'].str.lower().str.contains('oire')]
    syllables.loc[syllables_ire.index,'syllables']=syllables_ire['syllables'].str.replace('ire','i|re').str.replace('Ire', "I|re")

    syllables_ism=syllables[syllables['syllables'].str.lower().str.contains('ism')]
    syllables.loc[syllables_ism.index,'syllables']=syllables_ism['syllables'].str.replace('ism','i|sm')

    # correct systematic mistake in words with "ithm" in them
    syllables_ithm=syllables[syllables['syllables'].str.lower().str.contains('ithm')]
    syllables.loc[syllables_ithm.index,'syllables']=syllables_ithm['syllables'].str.replace('ithm','i|thm')

    syllables_ythm=syllables[syllables['syllables'].str.lower().str.contains('ythm')]
    syllables.loc[syllables_ythm.index,'syllables']=syllables_ythm['syllables'].str.replace('ythm','y|thm')


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
from src.phonemizer_utils import phonetize
import librosa
import numpy as np
import pandas as pd
def combined_sonoripy(word="dépendanc", orig_word="dépendance", lang="fr_FR"):
    stress_symbol="ˈ"
    second_stress_symbol="ˌ"
    phonetics=phonetize(orig_word, lang=lang).replace(stress_symbol,"").replace(second_stress_symbol,"").split('_')

    letter_by_syl, sonorities_letters = SonoriPy(str_to_list_of_char(word), mode='letters')
    p_by_syl, sonorities_phonemes = SonoriPy(phonetics, mode='MFA_IPA')

    x=[el[-1] for el in sonorities_letters]
    y=[el[-1] for el in sonorities_phonemes]

    D, wp = librosa.sequence.dtw(X=np.array(x), Y=np.array(y), metric='euclidean')

    lens=[len(el) for el in p_by_syl]
    syl_starts=[0]+[el for el in np.cumsum(lens)][:-1]

    aligned_tokens=pd.DataFrame(wp)[::-1]
    aligned_tokens.columns=["letters","phonemes"]

    syl_starts_letters=[(aligned_tokens[aligned_tokens.phonemes==syl_start].letters.iloc[0]) for syl_start in syl_starts]


    syls_letters=[]
    letters=str_to_list_of_char(word)
    syl_starts_letters.append(len(letters))

    for i in range(len(syl_starts_letters)-1):
        syls_letters.append(letters[syl_starts_letters[i]:syl_starts_letters[i+1]])

    return syls_letters






def syllabified_text(word, n, syllables_df=pd.read_csv('data/syllables.csv'), lang="en_GB", combined_sonoripy_dtw=True):
    """Construct syllabified word from a word.

     text with syllable segmentation is done with several rules/steps:
        -check if only 1 vowel (trivial because 1 syllable) ==>  in that case syllable=text
        -If more, use our syllables dataset
        -If does not exist,  fall back to use SonoriPy (sonority sequencing principle) with letters.
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
    orig_word=word
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
            if word[-2:]=="ed" and word[-3] not in ['t','d'] and word[-4:]!="ired" and word[-3:]!='led': #this last is treated hereafter because it dependes
                word=word[:-2]+'d'
                modified_ed=True
            
            # reduction for e.g. "smiled, filed" (vowel before l), but not for "angled, muscled, ..."
            elif word[-3:]=="led" and word[-4] in vowels:
                word=word[:-2]+'d'
                modified_ed=True

            # sounds before -es corresponding to sibilant sounds
            elif word[-2:]=="es" and word[-3] not in ['s','c','g','x','z','j'] and word[-4:]!='ches' and word[-4:]!='shes' and word[-4:]!="ires" and word[-3:]!='les': #this last is treated hereafter because it dependes
                word=word[:-2]+'s'
                modified_es=True
            
            # reduction for e.g. "smiles" (vowel before l), but not for "angles, muscles, articles, ...", and not for "alleles,angeles,anopheles,isosceles"
            elif word[-3:]=="les" and word[-4] in vowels and word[-4]!='e':
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
            elif word[-1]=="e":
                word=word[:-1]
                trailing_e=True
        
        if combined_sonoripy_dtw:
            letters_by_syl=combined_sonoripy(word, orig_word, lang=lang)
        else:
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


def n_vowels(word="coffee", lang="en_GB"):
    """use phonemizer with espeak backend and count vowels

    Args:
        word (str, optional): _description_. Defaults to "coffee".
        lang (str, optional): _description_. Defaults to "en_GB".

    Returns:
        _type_: _description_
    """
    # print("in n_vowels")
    # print(lang.split('_')[0])
    # I choose cmudict as a favourite ground truth for the number of syllables, and fall back to espeak
    # words like science -> 's_ˈaɪə_n_s', client -> 'k_l_ˈaɪə_n_t'  have a weird phoneme separation in espeak. The schwa is glued to a vowel. I prefer cmudict's conventions.
    # I therefore make the assumption that en_GB and en_US have the same number of syllables for now, and I hope we can actually choose alternatives accordingly
    if lang.split('_')[0]=="en":
        if len(cmudict_dict[word])>0:
            d=define_categories(mode="CMU")
            phonetics=cmudict_dict[word][0]
            n=0
            for el in phonetics:
                if (unstress(el.lower()) in d['vowels']) : n+=1
            # print(word, " n of vowels, from cmudict: ", n)
            return n


    d=define_categories(mode="MFA_IPA")
    stress_symbol="ˈ"
    second_stress_symbol="ˌ"
    phonetics=phonetize(word, lang=lang).replace(stress_symbol,"").replace(second_stress_symbol,"").split('_')
    
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
    # phonetic_dict=mfa_dicts['fr_FR']
    nv=n_vowels(word, lang="fr_FR")
    syllabified_text(word, nv, syllables_df=pd.read_csv('data/syllables_fr_FR.csv'), lang="fr_FR")

    syllabified_text('dépendance', pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']))[0]