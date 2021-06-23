
import re
import cmudict
from tqdm import tqdm
import pandas as pd

import numpy as np
import itertools
from syllabipy.sonoripy import SonoriPy, str_to_list_of_char

cmudict_dict=cmudict.dict()


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


def show_alternatives_distributions():
    import numpy as np
    lens=[]
    for k,v in cmudict_dict.items():
        lens.append(len(v))
    print(np.histogram(lens, bins=[0,1,2,3,4,5,6,7]))



def get_cmudict_info(word='university'):
    """get the first possible phonetisation of a word from cmudict

    Args:
        word (str, optional): input. Defaults to 'university'.
    Returns:
        list: phonemes and a number for each vowel indicating stress: 0=no stress, 1=primary stress, 2=secondary stress
    """
    return cmudict_dict[word][0]

def remove_special_characters(sentence="Where's the best place to have coffee ?"):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    sentence = re.sub(chars_to_ignore_regex, '', sentence).lower()
    return sentence

def phonetics_from_sentence(sentence="Where's the best place to have coffee ?"):
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
    l=[]
    for el in transcription:
        if el[-1] in str([0,1,2]): l.append(el[:-1])
        else: l.append(el)
    return l

def n_vowels(phonetics=['K', 'AA1', 'F', 'IY0']):
    n=0
    for el in phonetics:
        if el[-1] in str([0,1,2]): n+=1
    return n

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
    for k,v in cmudict.dict().items():
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
        if len(v[0])>=len(phones):
            if x_in_y(phones, v[0]):
                selection[k]=v[0]
    return selection

def syllables_data():
    # http://www.delphiforfun.org/programs/Syllables.htm
    # syllables=pd.read_csv('Syllables.txt',sep='=', header=None)
    syllables=pd.read_csv('mhyph.txt', header=None)

    syl_sep=syllables[0][0][5]
    syllables.iloc[:,0]=syllables.iloc[:,0].str.replace(syl_sep,'_')

    syl_sep='_'

    d=cmudict_dict
    syllables=syllables.dropna()  # there is one row that is nan...

    # syllables.columns=['word', 'syllables']
    syllables.columns=['syllables']

    n_syls=[]
    n_vowels_cmu=[]
    texts=[]
    phonetics=[]
    for i,r in syllables.iterrows():
        text=''.join(r[0].split(syl_sep)).lower()
        texts.append(text)
        try:
            n_syls.append(int(len(r[0].split(syl_sep))))
        except:
            n_syls.append(None)
        # print(d[r[0]][0])
        try:
            phonetics.append(' '.join(d[text][0]))
            n_vowels_cmu.append(int(n_vowels(d[text][0])))
        except IndexError:
            n_vowels_cmu.append(None)
            phonetics.append(None)

    # len([el for el in n_vowels_cmu if el!=None])
    
    syllables['normalized_text']=texts
    syllables['phonetics']=phonetics

    syllables['n_syls']=n_syls
    syllables['n_vowels_cmu']=n_vowels_cmu

    syllables[syllables.n_vowels_cmu.isnull()].normalized_text.tolist()
    len(syllables[~syllables.n_vowels_cmu.isnull()].normalized_text.tolist())

    syllables=syllables.dropna()
    syllables[syllables.n_syls==syllables.n_vowels_cmu]
    syllables[syllables.n_syls!=syllables.n_vowels_cmu]

    syllables.to_csv('syllables.csv')

    return syllables


def generate_phonetics_from_words(words, indxs):
    phonetics=[]
    for i,word in enumerate(words):
        phonetics.append(SonoriPy(cmudict_dict[word][int(indxs[i])])) 
        # SonoriPy(phonetics)   
        # gibberish=[[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in l] for l in SonoriPy(phonetics)]

    return phonetics


def phonetics_alternatives(words):
    lens=[]
    for i,word in enumerate(words):
        lens.append(len(cmudict_dict[word]))
    lists_indxs=[list(np.arange(el)) for el in lens]
    alternative_combinations=list(itertools.product(*lists_indxs))
    alternative_phonetics=[]
    for indxs in alternative_combinations:
        alternative_phonetics.append(generate_phonetics_from_words(words, indxs))
    return alternative_phonetics

def generate_phonetics_alternatives(sentences):
    phonetics=[]
    for sent in sentences:
        words=sent.split(' ')
        word_phonetics=phonetics_alternatives(words)
        phonetics.append(word_phonetics)
    return phonetics

def syllabified_text(word, syllables_df):
    try:
        syls_text=syllables_df[syllables_df.normalized_text==word].syllables.values[0]
    except IndexError:
        # if it did not exist in the manually syllabified data we have:
        # we can first say that words of one syllable is a trivial case, and we can detect that by checking if there is only one syllable
        # if not, I use SonoriPy (sonority sequencing principle) based on letters (it is less accurate than with phonemes but we use it only on fallback)
        if cmudict_dict[word]!=[]:
            if n_vowels(cmudict_dict[word][0])==1:
                syls_text=word
                return syls_text
        letters_by_syl=SonoriPy(str_to_list_of_char(word), mode='letters')
        syls_text='_'.join([''.join(syl) for syl in letters_by_syl])
        return syls_text
    return syls_text


def prefill_content(sentences):
    syllables=syllables_data()

    syllables_texts=[]
    for s in sentences:
        words=s.split(' ')
        syls_texts=[]
        for word in words:
            syls_texts.append(syllabified_text(word, syllables))
        # syls_text=' '.join(syls_texts)
        syllables_texts.append(' '.join(syls_texts))
    
    ps=generate_phonetics_alternatives(sentences)
    # go through levels of the list (alternatives, words, syllables, phones) and then convert every phoneme in gibberish
    gs=[[[[[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in syl] for syl in word] for word in alt] for alt in phonetics] for phonetics in ps]

    # This puts 
    # '|' between phonemes
    # '_' between syllables
    # spaces between words 
    # to have less degrees of nested list and be compatible with the database
    ps2=[[' '.join(['_'.join(['|'.join(syl) for syl in word]) for word in alt]) for alt in phonetics] for phonetics in ps]
    gs2=[[' '.join(['_'.join(['|'.join(syl) for syl in word]) for word in alt]) for alt in phonetics] for phonetics in gs]

    df=pd.DataFrame()
    df['sentence']=sentences
    df['syllables_text']=syllables_texts
    df['CMU']=ps2
    df['gibberish']=gs2

    # use syllables_data. But modify it to check if in cmu, another alternative has the same number of syllables instead
    # of only checking with the first alternative

    return df

if __name__ == "__main__":
    
    
    words=['comfortable','table','professional', 'analysis', 'temperature', 'personal', 'government']
    sentences=words+['yesterday morning','coffee','seek','take the lead','worked','started a company','think','visited']
    # from syllabipy.sonoripy import generate_gibberish_alternatives
    # g=generate_gibberish_alternatives(sentences)

    df=prefill_content(sentences)