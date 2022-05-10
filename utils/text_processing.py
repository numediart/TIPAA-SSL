
import re
import cmudict
from tqdm import tqdm
import pandas as pd

import numpy as np
import itertools
from syllabipy.sonoripy import SonoriPy, str_to_list_of_char

from g2p_en.expand import normalize_numbers
from g2p_en import G2p
from itertools import groupby

g2p = G2p()

drop_consecutive_duplicates= lambda df: df.loc[(df.shift()!=df).sum(axis=1).astype(bool)]
drop_consecutive_duplicate_elements= lambda L: [key for key, _group in groupby(L)]
unstress = lambda el: el[:-1] if el[-1] in str([0,1,2]) else el
split_phonetics = lambda phonetics: [[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]


cmudict_dict=cmudict.dict()

syllables_df=pd.read_csv('data/syllables.csv')

cmu_to_gibberish={'AA':'o',
                'AE':'a',
                'AH':'uh',
                'AO':'aw',
                'AW':'au',
                'AY':'ay',
                'B':'b',
                'CH':'ch',
                'D':'d',
                'DH':'th',
                'EH':'e',
                'ER':'uhr',
                'EY':'ey',
                'F':'f',
                'G':'g',
                'HH':'h',
                'IH':'i',
                'IY':'ee',
                'JH':'dj',
                'K':'k',
                'L':'l',
                'M':'m',
                'N':'n',
                'NG':'ng',
                'OW':'ow',
                'OY':'oy',
                'P':'p',
                'R':'r',
                'S':'s',
                'SH':'sh',
                'T':'t',
                'TH':'th',
                'UH':'u',
                'UW':'oo',
                'V':'v',
                'W':'w',
                'Y':'y',
                'Z':'z',
                'ZH':'j'}

# from https://github.com/kosuke-kitahara/xlsr-wav2vec2-phoneme-recognition/blob/main/Fine_tuning_XLSR_Wav2Vec2_for_Phoneme_Recognition.ipynb
# IPA
# ref: https://en.wikipedia.org/wiki/ARPABET
arpabet_to_ipa = {
    'aa': 'ɑ',
    'ae': 'æ',
    'ah':'ʌ',
    'ao':'ɔ',
    'aw':'W',
    'ax':'ə',
    'axr':'ɚ',
    'ay':'Y',
    'eh':'ɛ',
    'er':'ɝ',
    'ey':'e',
    'ih':'ɪ',
    'ix':'ɨ',
    'iy':'i',
    'ow':'o',
    'oy':'O',
    'uh':'ʊ',
    'uw':'u',
    'ux':'ʉ',
    'b':'b',
    'ch':'C',
    'd':'d',
    'dh':'ð',
    'dx':'ɾ',
    'el':'l̩',
    'em':'m̩',
    'en':'n̩',
    'f':'f',
    'g':'g',
    'hh':'h',
    'h':'h',
    'jh':'J',
    'k':'k',
    'l':'l',    
    'm':'m',    
    'n':'n',    
    'ng':'ŋ',    
    'nx':'ɾ̃',    
    'p':'p',    
    'q':'ʔ',    
    'r':'ɹ',    
    's':'s',    
    'sh':'ʃ',    
    't':'t',    
    'th':'θ',    
    'v':'v',    
    'w':'w',    
    'wh':'ʍ',    
    'y':'j',    
    'z':'z',    
    'zh':'ʒ',    
    'ax-h':'ə̥',    
    'bcl':'b̚',    
    'dcl':'d̚',    
    'eng':'ŋ̍',    
    'gcl':'ɡ̚',    
    'hv':'ɦ',    
    'kcl':'k̚',    
    'pcl':'p̚',    
    'tcl':'t̚',
    'epi':'S', 
    'pau':'P',   
}

cmu_phones=[el[0] for el in cmudict.phones()]

phones=cmudict.phones()
cmu_vowels=[p[0] for p in phones if p[1][0]=='vowel']
cmu_consonants=[p[0] for p in phones if p[1][0]!='vowel']

# CMU is a subset of arpabet
cmu_1_char={}
cmu_1_char_to_gibberish={}
for p in cmu_phones:
    cmu_1_char[p]=arpabet_to_ipa[p.lower()]
    cmu_1_char_to_gibberish[cmu_1_char[p]]=cmu_to_gibberish[p]

def show_alternatives_distributions():
    import numpy as np
    lens=[]
    for k,v in cmudict_dict.items():
        lens.append(len(v))
    print(np.histogram(lens, bins=[0,1,2,3,4,5,6,7]))



# expand some words as Mr or Mrs to Mister and Misses
expand_dict={'mr':'mister',
            'mrs':'misses',
            'Mr':'Mister',
            'Mrs':'Misses'
            }

def get_cmudict_info(word='university'):
    """get the first possible phonetisation of a word from cmudict

    Args:
        word (str, optional): input. Defaults to 'university'.
    Returns:
        list: phonemes and a number for each vowel indicating stress: 0=no stress, 1=primary stress, 2=secondary stress
    """
    return cmudict_dict[word][0]

def remove_special_characters(sentence="Where's the best place to have coffee?", lowercase=True, chars_to_ignore_regex = '[\,\?\.\!\;\:\"\*\{\}]'):
    """Normalize text by lowercasing (if option is True), and remove a set of punctuation characters

    Args:
        sentence (str, optional): [description]. Defaults to "Where's the best place to have coffee ?".
        lowercase (bool, optional): [description]. Defaults to True.
        chars_to_ignore_regex (str, optional): [description]. Defaults to '[\,\?\.\!\;\:\"\*]'.

    Returns:
        str: normalized sentence
    """
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    sentence = re.sub(chars_to_ignore_regex, '', sentence)

    # I saw this weird quote show up and mess the rest up
    sentence=sentence.replace("’","'")

    if lowercase:
        sentence=sentence.lower()

    # This is to make sure there will not be empty strings after a splitting. So here I split, remove Nones, and rejoin
    sentence=' '.join(list(filter(None, sentence.split(' '))))
    return sentence

def get_chunks(text, chunking_chars=[',',';','.','!','?', ':', '/']):
    for c in chunking_chars:
        text=text.replace(c, chunking_chars[0])
    chunks=text.split(chunking_chars[0])
    chunks = list(filter(None, chunks)) # remove empty string
    return chunks

def chunk_text(text, chunking_chars=[',',';','.','!','?', ':', '/']):
    chunks=get_chunks(text, chunking_chars=chunking_chars)
    # split each chunk in words, remove empty strings, get length (to know the n of words in each chunk)
    n_words_by_chunk=[len(list(filter(None, el.split(' ')))) for el in chunks]
    # assert sum(n_words_by_chunk)==len(list(filter(None, text.split(' ')))), "Checking number of words is the same after chunking"
    return n_words_by_chunk
    
def phonetics_from_sentence(sentence="Where's the best place to have coffee?"):
    sentence=remove_special_characters(sentence)
    words=sentence.split(' ')
    # drop empty strings
    words = list(filter(None, words))
    words_phones=[]
    for w in tqdm(words):
        phones=get_cmudict_info(w)
        words_phones.append(phones)
    return words_phones


def remove_stress_annots(transcription=['K', 'AA1', 'F', 'IY0']):
    return [unstress(el) for el in transcription]




def phonetics_indexed_df_from_formatted_phonetics(phonetics):
    word_split_phonetics=[p.replace('|','_').split('_') for p in phonetics.split(' ')]
    n_phone_by_word=[len(w) for w in word_split_phonetics]
    
    word_indices=[]
    for i,n in enumerate(n_phone_by_word): word_indices+= [i]*n
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    n_phone_by_syl=[[len(s) for s in w] for w in split_phonetics]

    all_syl_indices=[]
    for n_phones in n_phone_by_syl:
        syl_indices=[]
        for i,n in enumerate(n_phones): syl_indices+= [i]*n
        all_syl_indices+=syl_indices

    phone_list=sum(sum(split_phonetics, []),[])

    phonetics_indexed_df=pd.DataFrame()
    phonetics_indexed_df['phones']=phone_list
    phonetics_indexed_df['word_idx']=word_indices
    phonetics_indexed_df['syl_idx']=all_syl_indices

    phonetics_indexed_df.apply(lambda r: r['phones'], axis=1)

    # # to get the number of rows of each sequence of syl_idx
    # # I look at differences of indices. For the last one, we need to make diff with len(df)
    # d=drop_consecutive_duplicates(phonetics_indexed_df[['syl_idx']])
    # n_rows=np.diff(d.index).tolist()
    # n_rows.append(len(phonetics_indexed_df)-d.index[-1])
    # # build phone index which are indeices inside a syllable
    # p_idx=sum([np.arange(n).tolist() for n in n_rows],[])
    # phonetics_indexed_df['p_idx']=p_idx
    
    # remove_stress_annots(phone_list)

    return phonetics_indexed_df



def check_phonemes(phonemes):
    """This function returns a non existing phoneme if it happens. else it returns None
    """
    ps=remove_stress_annots(phonemes)
    for p in ps:
        if p not in cmu_phones: return p
        
def n_vowels(phonetics=['K', 'AA1', 'F', 'IY0']):
    n=0
    for el in phonetics:
        if el[-1] in str([0,1,2]): n+=1
    return n


def n_syl_SonoriPy(phonetics=['K', 'AA1', 'F', 'IY0']):
    return len(SonoriPy(phonetics)[0])



def word_stress_from_cmu(phonetics=['K', 'AA1', 'F', 'IY0']):
    # cmu vowels end by a number : 0, 1 or 2.   0= no stress, 1 = primary stress, 2 = secondary stress
    # consonants do not end by a number
    # here I return a list that is one if primary stressed and else 0
    return [1 if p[-1]==str(1) else 0 for p in phonetics if p[-1] in str([0,1,2])]
    
def word_stress_from_text(sentence="Where's the best place to have coffee ?"):
    phonetics=phonetics_from_sentence(sentence) #return a list of phoneme list (by word)
    result = []
    for el in phonetics:
        result+=el
    binResult=word_stress_from_cmu(result)
    return binResult


def x_in_y(query, base):
    """Check if a (query) is a subsequence of another list (base)

    Args:
        query (list): subsequence to find
        base (list): main list

    Returns:
        Boolean: True if subsequence found in list, else False
    """
    # from https://stackoverflow.com/questions/33392219/how-to-check-subsequence-exists-in-a-list
    try:
        l = len(query)
    except TypeError:
        l = 1
        query = type(base)((query,))

    for i in range(len(base)):
        if base[i:i+l] == query:
            return True
    return False

def get_words_that_end_with(phones=['IH0', 'D']):
    """Goes through cmudict items and those who end by "phones"

    Args:
        phones (list, optional): list of cmu phonemes. Defaults to ['IH0', 'D'].

    Returns:
        dict: words that end by phones
    """
    # cmudict_first_alternatives={}
    selection={}
    for k,v in cmudict_dict.items():
        # cmudict_first_alternatives[k]=v[0]
        if len(v[0])>=len(phones):
            if v[0][-len(phones):]==phones:
                selection[k]=v[0]
    return selection

def words_that_contains(phones=['IH0', 'D']):
    """Goes through cmudict items and those who end by "phones"

    Args:
        phones (list, optional): list of cmu phonemes. Defaults to ['IH0', 'D'].

    Returns:
        dict: words that end by phones
    """
    selection={}
    for k,v in cmudict_dict.items():
        # cmudict_first_alternatives[k]=v[0]
        if v!=[]:
            if len(v[0])>=len(phones):
                if x_in_y(phones, v[0]):
                    selection[k]=v[0]
    return selection


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

    # http://hindson.com.au/info/free/free-english-language-hyphenation-dictionary/
    # syllables=pd.read_csv('data/EnglishHyphDict_v108.txt', header=None, sep=' ')
    # syllables.iloc[:,1]=syllables.iloc[:,1].str.strip(';')
    # syllables.iloc[:,1]=syllables.iloc[:,1].str.replace('-','_')
    # # syllables.iloc[:,0]=syllables.iloc[:,1]
    # syllables=pd.DataFrame(syllables.iloc[:,1])
    # syllables.columns=[0]

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
        # try:
        #     phonetics.append(' '.join(d[text][0]))
        #     n_vowels_cmu.append(int(n_vowels(d[text][0])))
        # except IndexError:
        #     n_vowels_cmu.append(None)
        #     phonetics.append(None)
        try:
            n_syls_SonoriPy.append(n_syl_SonoriPy(d[text][0]))
        except IndexError:
            n_syls_SonoriPy.append(None)

    syllables.syllables=syllables.syllables.str.lower()
    
    syllables['normalized_text']=texts
    # syllables['phonetics']=phonetics

    syllables['n_syls']=n_syls
    syllables['n_syls_SonoriPy']=n_syls_SonoriPy
    # syllables['n_vowels_cmu']=n_vowels_cmu

    # syllables[syllables.n_vowels_cmu.isnull()].normalized_text.tolist()
    # len(syllables[~syllables.n_vowels_cmu.isnull()].normalized_text.tolist())

    if False:
        syllables=syllables.dropna()

        # syllables[syllables.n_syls==syllables.n_vowels_cmu]
        # syllables[syllables.n_syls!=syllables.n_syls_SonoriPy]
        # syllables[syllables.n_vowels_cmu!=syllables.n_syls_SonoriPy]

        # syllables[syllables.normalized_text=="really"]

        # Some words have several possibilities of text syllable segmentation.
        # For a given word, if there are some that for which n_syls!=n_syls_SonoriPy, and others for which n_syls==n_syls_SonoriPy
        # then I only keep those for which n_syls==n_syls_SonoriPy

        inconsistent_syls=syllables[syllables.n_syls!=syllables.n_syls_SonoriPy]
        lens=[]
        for i,r in inconsistent_syls.iterrows():
            lens.append(len(syllables[syllables.normalized_text==r.normalized_text]))
        
        inconsistent_syls['n_syl_text_alternatives']=lens

        # Select the ones who have potentially another solution with a consistent number of syls
        candidates_for_good_alt=inconsistent_syls[inconsistent_syls['n_syl_text_alternatives']>1]

        # If there are possibilities with consistent number of syllables, remove the inconsistent ones
        idx_to_remove=[]
        for i,r in candidates_for_good_alt.iterrows():
            alts=syllables[syllables.normalized_text==r.normalized_text]
            if len(alts[alts.n_syls==alts.n_syls_SonoriPy])>0:
                idx_to_remove+=alts[alts.n_syls!=alts.n_syls_SonoriPy].index.tolist()
        syllables=syllables.drop(idx_to_remove)

    syllables.to_csv('data/syllables.csv')

    return syllables

def generate_syl_phonetics_alternatives_from_word(word="before"):
    """"Looks up in cmudict for phonetic alternatives, and apply SonoriPy on all alternatives

    Args:
        word (str, optional): [description]. Defaults to "before".

    Returns:
        list: the nested list contains alternatives -> words -> syllables -> phones.  
        for default input: [[['B', 'IH0'], ['F', 'AO1', 'R']], [['B', 'IY2'], ['F', 'AO1', 'R']]]
    """

    # fallbacks
    if '-' in word:
        w_parts=word.split('-')
        p_parts=[]
        for w_part in w_parts:
            p_part=cmudict_dict[w_part]
            if p_part==[]:
                p_part=[g2p(w_part)]
            p_parts.append(p_part)

        # Here I generate all alternatives of combination of word parts
        # It corresponds to cmu alternatives for other words
        ps=list(itertools.product(*p_parts))
        syls_ps=[[SonoriPy(p)[0] for p in alt] for alt in ps]
        
        # This puts 
        # '_' between phonemes
        # '-' between word components (hyphenated words and acronyms, not to have more than 1 stressed syllable by word component)
        # '|' between syllables
        # spaces between words 
        # to have less degrees of nested list and be compatible with the database
        syls_ps_formatted=['-'.join(['|'.join(['_'.join(s) for s in w]) for w in alt]) for alt in syls_ps]
        syls_gs_formatted=['-'.join(['|'.join(['_'.join([cmu_to_gibberish[unstress(el)] for el in s] ) for s in w]) for w in alt]) for alt in syls_ps]

    else:
        ps=cmudict_dict[word]
        if ps==[]:
            ps=[g2p(word)]
        
        # Rule for verbs in -ded or -ted: we want to get rid of the "AH0_D" alternative
        for p in ps:
            if (p[-3:]==[ 'T', 'AH0', 'D'] or p[-3:]==[ 'D', 'AH0', 'D']) and (word[-3:]=='ted' or word[-3:]=='ded'):
                p[-2:]=['IH0','D']
        
        syls_ps=[SonoriPy(p)[0] for p in ps]
        syls_ps_formatted=['|'.join(['_'.join(s) for s in w]) for w in syls_ps]
        syls_gs_formatted=['|'.join(['_'.join([cmu_to_gibberish[unstress(el)] for el in s]) for s in w]) for w in syls_ps]
    return syls_ps_formatted, syls_gs_formatted


def syllabified_text(word, syllables_df=pd.read_csv('data/syllables.csv')):
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
    if cmudict_dict[word]!=[]:
        if n_vowels(cmudict_dict[word][0])==1:
            syls_text=word
            used_method='1 syl in cmu'
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
        _,vowels,nasals,fricatives,affricates,stops=define_categories()

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
        
        letters_by_syl=SonoriPy(str_to_list_of_char(word), mode='letters')[0]
        syls_text='|'.join([''.join(syl) for syl in letters_by_syl])

        if modified_ed:
            syls_text=syls_text[:-1]+"ed"
        if modified_es:
            syls_text=syls_text[:-1]+"es"
        if trailing_e:
            syls_text+='e'

        used_method='SonoriPy'
        return syls_text, used_method
    return syls_text, used_method


def find(s, ch=['|']):
    return [i for i, ltr in enumerate(s) if ltr in ch]

def insert(s, ch, i):
    return s[:i] + ch + s[i:]


def insert_seps_in_cased_text(s, s_case, syl_sep='|'):
    """To have syllable parts with capital letters, this function compare segmented syllable string to original text.
    Indeed the segmented version had to be lowercased to look-up in a word dataset without being sensitive to case.
    Here we get the position of syllable separators "|" and put them at their position in original cased text.

    Args:
        s (str): text segmented in syllables
        s_case (str): cased text
        syl_sep (str, optional): syllable separator. Defaults to '|'.

    Returns:
        str: cased text segmented in syllables
    """
    s_case_sep=[]
    for w,w_case in zip(s.split(' '),s_case.split(' ')):
        idxs=find(w, [syl_sep])
        # now that we found indices of where the syllable separators are, we need to find after which character to insert them
        # as if the separators weren't there. So we need to substract the number of separators there was before, i.e. its index in the list
        char_idxs=[el-i for i,el in enumerate(idxs)]

        # here we insert the separator in the cased text thanks to the indices. It is important to do that
        # starting from the end, so that the other indices are not afftected by these insertions
        for idx in char_idxs[::-1]:
            w_case=insert(w_case, syl_sep, idx)
        s_case_sep.append(w_case)
    return ' '.join(s_case_sep)


# def add_special_char(s_orig, s_modified):
#     """This function adds punctuation marks to modified text (here with syllable separation symbols "|") 
#     at the end of words from an original sentence.
#     This assumes that punctuation marks are glued to words, which is the case in english. 
#     This assumption allows us to just check if the first and last characters are the same in original and modified text
#     "Hello, my name is John."
#     "hello my name is john"
#     -> hello and john do not have the last same character.

#     "*Hello*," / "hello", we extract "*" and "*,"

#     Example:
#     s_orig="I'm taking a Spanish class."
#     s_modified="I'm tak|ing a Span|ish class"

#     output="I'm tak|ing a Span|ish class."

#     Args:
#         s_orig (str): original text
#         s_modified (str): modified text

#     Returns:
#         str: modified text with punctuation marks
#     """
#     s_modified_with_special_chars=[]
#     for w_orig, w_modified in zip(s_orig.split(' '), s_modified.split(' ')):
#         # n_spec_char=
#         def get_n_spec_char(location='start'):
#             # get the number of special character at the start or at the end of the word, 
#             # by looking at every character of the orig (containing the special characters)
#             for i in range(len(w_orig)):
#                 if location=="end":
#                     if w_orig[-i]==w_modified[-1]:return i
#                 if location=="start":
#                     if w_orig[i]==w_modified[0]:return i
#         # glue the special characters the to the body
#         if not get_n_spec_char(location='end'):
#             s_modified_with_special_chars.append(w_orig[:get_n_spec_char()]+w_modified+w_orig[-get_n_spec_char(location='end'):][1:])
#         else:
#             s_modified_with_special_chars.append(w_orig[:get_n_spec_char()]+w_modified)


#     return ' '.join(s_modified_with_special_chars)


def prefill_for_sentence(sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up!", syllables_df=pd.read_csv('data/syllables.csv'), syl_sep='|', special_chars = [',','?','.','!',';',':','"', '{', '}']):
    """This function extract information of syllabified texts and phonetics. 
    It uses a combination of datasets (CMUdict, data from syllable_data() ) and algorithm (SonoriPy)

    Args:
        sentence (str): a phrase to be processed. It can contain captial letters and punctuation.
        syllables_df (DataFrame): The syllables dataset built from syllables_data()
        syl_sep (str, optional): [description]. Defaults to '|'.

    Returns:
        dict: see structure a the end of the function
    """
    # there shouldn't be a space before a special char, they must be glued to words (in english)
    # correct that if it's not the case
    for c in special_chars: sentence=sentence.replace(' '+c, c)

    # this takes care of e.g. "2021", "$110"
    # curly braces for numbers, if they are in several words
    norm_sent_list=[]
    for word in sentence.split(' '):
        n_word=normalize_numbers(word)
        if ' ' in n_word:
            el="{"+n_word+"}"
        else: el=n_word
        norm_sent_list.append(el)

    norm_sent=" ".join(norm_sent_list)

    # memorize special characters glued before and after words
    special_chars_dict_end={}
    special_chars_dict_start={}
    for i,w in enumerate(norm_sent.split(' ')):
        if w[-1] in special_chars:
            special_chars_dict_end[i]=w[-1] 
        if w[0] in special_chars:
            special_chars_dict_start[i]=w[0] 

    
    words=remove_special_characters(norm_sent, lowercase=False).split(' ')

    # little dictionnary mapping e.g. Mr -> Mister
    words=[expand_dict[word] if word in expand_dict else word for word in words]
    norm_words=words
    
    # If all letters are capital (acronym), put '-' between all letters
    # I need to do that before putting in lowercase, that is why I cannot put that in e.g. syllabified_text()
    ws=[]
    acronym_idxs=[]
    for w_idx,w in enumerate(words):
        if w.isupper():
            acronym_idxs.append(w_idx)
            w='-'.join(w)
            words[w_idx]=w
        ws.append(w)

    words=[word.lower() for word in ws]

    syls_texts=[]
    used_method_syllables=[]
    for word in words:
        # # This split and rejoin will apply only if there is a dash and then allow to treat separately parts of a word with dash
        # if False:
        if '-' in word:
            syls_texts_parts=[]
            used_method_syllables_parts=[]
            for w_part in word.split('-'):
                syls_texts_parts.append(syllabified_text(w_part, syllables_df)[0])
                used_method_syllables_parts.append(syllabified_text(w_part, syllables_df)[1])
            # syls_texts.append('|-'.join(syls_texts_parts))
            syls_texts.append('-'.join(syls_texts_parts))
            used_method_syllables.append('-'.join(used_method_syllables_parts))
        else:
            syls_texts.append(syllabified_text(word, syllables_df)[0])
            used_method_syllables.append(syllabified_text(word, syllables_df)[1])
    
    # put back acronyms in syls_texts, and ignore what was there
    for i in acronym_idxs:
        syls_texts[i]=words[i]
        used_method_syllables[i]='acronym'

    # for idx in acronym_idxs:
    #     norm_words[idx]='{'+ws[idx]+'}'

    case_syls_texts=insert_seps_in_cased_text(' '.join(syls_texts), ' '.join(norm_words), syl_sep=syl_sep)
    case_syls_texts_special_chars=case_syls_texts.split(' ')

    def add_special_chars(split_text, special_chars_dict_start, special_chars_dict_end):
        for k in special_chars_dict_end:
            split_text[k]+=special_chars_dict_end[k]
        for k in special_chars_dict_start:
            split_text[k]=special_chars_dict_start[k]+split_text[k]
        return split_text

    def acronyms_hyphen_to_compound(split_text, acronym_idxs):
        # remove '-' in acronyms
        # And if it was only 1 letter, then it's not a compound word.
        for i,el in enumerate(split_text):
            if i in acronym_idxs:
                if '-' in el:
                    split_text[i]='{'+el.replace('-',' ')+'}'
                else:
                    split_text[i]=el.replace('{','').replace('}','')
        return split_text

    case_syls_texts_special_chars=acronyms_hyphen_to_compound(case_syls_texts_special_chars, acronym_idxs)
    case_syls_texts_special_chars=add_special_chars(case_syls_texts_special_chars, special_chars_dict_start, special_chars_dict_end)
    

    segmented_text=' '.join(case_syls_texts_special_chars)

    # p -> phonetics
    # g -> gibberish

    p=[generate_syl_phonetics_alternatives_from_word(word)[0] for word in words]

    # I separate the acronym in letters and lookup cmudict. I take the last, because
    # there is only one alternative except for letter 'A' which has  [['AH0'], ['EY1']]
    for i in acronym_idxs:  p[i]=['-'.join(['_'.join(cmudict_dict[w][-1]) for w in words[i].split('-')])]

    stress_inconsistencies=[]
    for w_i,w in enumerate(p):
        for alt_i,alt in enumerate(w):
            # if there is several primary stress inside a word part, it is a problem. It breaks our assumptions, so we memorize its index to treat it
            # below
            if alt.count('1')>alt.count('-')+1: 
                # print(alt)
                # print(w)
                d={'w_i':w_i,'alt_i':alt_i,'n_alt':len(w)}
                stress_inconsistencies.append(d)
    stress_inconsistencies=pd.DataFrame(stress_inconsistencies)
    
    n_alternatives=[len(el) for el in p]
    n_syls_in_p=[[len(alt.split(syl_sep)) for alt in word] for word in p]
    n_syls_in_text=[len(word.split(syl_sep)) for word in syls_texts]

    # among alternatives of phonetics (or gibberish), take the first index for which the number of syllables is equal (i.e. difference=0)
    # if there are none, just take the first alternative (index=0)
    alternative_idxs_syls_consistent=[]
    # memorize words for which we couldn't find any consistent alternative
    word_idxs_inconsitencies={}
    word_idxs_inconsitencies['stress']=[]
    word_idxs_inconsitencies['n_syl']=[]
    for i,(syl_g,syl_t) in enumerate(zip(n_syls_in_p, n_syls_in_text)):
        inconsistency=(np.array(syl_g)-syl_t).tolist()
        if len(stress_inconsistencies)>0:
            if i in stress_inconsistencies.w_i.tolist():
                # if there is a stress inconsistency in this word, add a one to the corresponding element in the inconsistency vector
                for alt_idx in stress_inconsistencies[stress_inconsistencies.w_i==i].alt_i.tolist():
                    inconsistency[alt_idx]+=1
        # searching a 0 in the consitency vector to select a consistent alternative
        if 0 in inconsistency:  
            alternative_idxs_syls_consistent.append(inconsistency.index(0))
        else:
            # I select the first alternative as there is none that is consistent
            alternative_idxs_syls_consistent.append(0)
            # Then I check what inconsistencies were present
            if syl_g[0]!=syl_t:
                word_idxs_inconsitencies['n_syl'].append(i)
            if p[i][0].count('1')>p[i][0].count('-')+1:
                word_idxs_inconsitencies['stress'].append(i)

    # Keep first alternative. 
    try:
        p_0=' '.join([el[alternative_idxs_syls_consistent[i]] for i,el in enumerate(p)])
    except IndexError:
        p_0=''


    split_phonetics = lambda phonetics: [[[s.split('_') for s in sub_w.split('|')] for sub_w in w.split('-')] for w in phonetics.split(' ')]
    g_0=' '.join(['-'.join(['|'.join(['_'.join([cmu_to_gibberish[unstress(p)] for p in s]) for s in sub_w]) for sub_w in w]) for w in split_phonetics(p_0)])
    

    brace_dict_start=dict(filter(lambda el: el[1] in ['{','}'], special_chars_dict_start.items()))
    brace_dict_end=dict(filter(lambda el: el[1] in ['{','}'], special_chars_dict_end.items()))

    p_0_special_chars=acronyms_hyphen_to_compound(p_0.split(' '), acronym_idxs)
    p_0_special_chars=add_special_chars(p_0_special_chars, brace_dict_start, brace_dict_end)
    p_0_special_chars=' '.join(p_0_special_chars)

    g_0_special_chars=acronyms_hyphen_to_compound(g_0.split(' '), acronym_idxs)
    g_0_special_chars=add_special_chars(g_0_special_chars, brace_dict_start, brace_dict_end)
    g_0_special_chars=' '.join(g_0_special_chars)

    record={'text':sentence,
        'cmu_phonetics':p_0_special_chars,
        'pronounciation_guide':g_0_special_chars,
        'pronounciation_guide_hr':g_0_special_chars.replace('_',''),
        'segmented_text':segmented_text,
        # 'n_syl_mismatch':len(word_idxs_inconsitencies['n_syl']),
        'n_syl_mismatches':word_idxs_inconsitencies['n_syl'],
        'n_stress_inconsistencies':word_idxs_inconsitencies['stress'],
        'used_method_for_syl_text':used_method_syllables,
        'cmu_phonetics_alt':p,
        # 'pronounciation_guide_alt':g,
        # 'pronounciation_guide_hr_alt':g_hr,
        'n_alternatives':n_alternatives
        }
    return record

def prefill_content(sentences, syl_sep='|'):
    """This function extract information of syllabified texts and phonetics using prefill_for_sentence on a list of sentences.
    The result is saved in a DataFrame.

    Args:
        sentences ([type]): list of sentences (can contain special characters and capital letters)
        syl_sep (str, optional): [description]. Defaults to '|'.

    Returns:
        [type]: [description]
    """
    records=[]
    print("n sentences", len(sentences))
    for i,s in tqdm(enumerate(sentences)):
        try:
            record=prefill_for_sentence(s, syllables_df, syl_sep=syl_sep)
        except:
            print('Error with sentence: '+s)
            
        records.append(record)
    df=pd.DataFrame.from_records(records)
    return df


def generate_prefill_csv(                            # path='data/phrases_speaking_activities.txt', 
                            path='data/phrases_dynamoDB.txt',
                            syl_sep='|'#, 
                            # out_path='prefill_test.csv'
                            ):
    """Reads a text file containing phrases and uses prefill_content to return a CSV of syllabified texts and phonetics

    Args:
        path (str, optional): [description]. Defaults to 'phrases_speaking_activities.txt'.
        syl_sep (str, optional): [description]. Defaults to '|'.
        out_path (str, optional): [description]. Defaults to 'prefill_test.csv'.

    Returns:
        [type]: [description]
    """
    sentences=pd.read_csv(path, sep='/', header=None)
    df=prefill_content(sentences.iloc[:,0].tolist(), syl_sep=syl_sep)
    df.text=sentences

    print('syl mismatches')
    df.loc[df.apply(lambda r: bool(len(r.n_syl_mismatches)), axis=1)]
    print('stress inconsistencies')
    df.loc[df.apply(lambda r: bool(len(r.n_stress_inconsistencies)), axis=1)]

    return df

def word_selection():
    # words that finish in "s" with phoneme "S" that also exist without an "s" and with last phoneme then not being "S"
    words_in_s=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='S' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='S']
    words_in_z=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='Z' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='Z']
    
    
if __name__ == "__main__":
    from utils.text_processing import *
    prefill_for_sentence()
    
    from utils.label_data_processing import actor_recordings
    df_phrases=actor_recordings()
    df_phrases=df_phrases.loc[df_phrases.phrase_id.drop_duplicates().index]
    df_phrases=df_phrases.reset_index(drop=True)
    sentences=df_phrases.text.tolist()
    df=prefill_content(sentences)

    df['phrase_id']=df_phrases['phrase_id']

    print('syl mismatches')
    df.loc[df.apply(lambda r: bool(len(r.n_syl_mismatches)), axis=1)]
    print('stress inconsistencies')
    df.loc[df.apply(lambda r: bool(len(r.n_stress_inconsistencies)), axis=1)]

    df_bkp=pd.read_csv('prefill_test_phrases_export')

    df_bkp[df.cmu_phonetics!=df_bkp.cmu_phonetics]
    df[df.cmu_phonetics!=df_bkp.cmu_phonetics]
    

    df=generate_prefill_csv('data/export_phrases.txt')
    df.to_csv('prefill_test_phrases_export_new')

    words=['comfortable','table','professional', 'analysis', 'temperature', 'personal', 'government']
    sentences=words+['yesterday morning','coffee','seek','take the lead','worked','started a company','think','visited']
    # from syllabipy.sonoripy import generate_gibberish_alternatives
    # g=generate_gibberish_alternatives(sentences)

    df=prefill_content(sentences)
    
    df.iloc[:,-5:]

    from label_data_processing import get_data
    a=get_data()
    sentences=a.text.apply(lambda r: remove_special_characters(r)).tolist()
    df=prefill_content(sentences)
    df.text=a.text
    df.to_csv('prefill_test.csv')

    df.iloc[:,-3:]
    df.iloc[:,-5:-1]

    sentences=pd.read_csv('data/phrases-for-noe.txt', sep='/', header=None)
    sentences=sentences.iloc[:,0].apply(lambda r: remove_special_characters(r)).tolist()
    df=prefill_content(sentences)

    cmu_words=list(cmudict_dict.keys())
    df=prefill_content(cmu_words)

    word=words[0]
    scores_p=[el[-1] for el in SonoriPy(str_to_list_of_char(word), mode='letters')[-1]]

    word=cmudict_dict['comfortable'][0]
    scores_t=[el[-1] for el in SonoriPy(word)[-1]]

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import xticks
    plt.plot(scores_t)
    plt.savefig('sonoripy_comfortable.png')
    xticks(np.arange(len(word)), word)

    word='enjoyed'
    syls_text=syllables_df[syllables_df.normalized_text==word].syllables.values[0]
    print(syls_text)

    syllabified_text(word, syllables_df)

    df=generate_prefill_csv()
    df[df.n_syl_mismatch>0][['syllable_parts', 'pronounciation_guide_hr','used_method_for_syl_text']]

    # df[df.n_syl_mismatch>0][['syllable_parts', 'pronounciation_guide_hr','used_method_for_syl_text']]
    df[df.n_syl_mismatches.apply(lambda r: np.sum(r))>0].n_syl_mismatches
    df[df.n_syl_mismatches.apply(lambda r: np.sum(r))>0]
    
    df.n_alternatives.max()

    df.iloc[df.n_alternatives.argmax()]
    df.iloc[96]
    
    sentence=df.iloc[96].text
    words=remove_special_characters(df.iloc[96].text).split(' ')
    # len(syl_phonetics_alternatives(words))