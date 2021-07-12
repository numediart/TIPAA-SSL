
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

# special_characters=[',','?','!','-',';',':','"']

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

def remove_special_characters(sentence="Where's the best place to have coffee ?", lowercase=True, chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'):
    """Normalize text by lowercasing (if option is True), and remove a set of punctuation characters

    Args:
        sentence (str, optional): [description]. Defaults to "Where's the best place to have coffee ?".
        lowercase (bool, optional): [description]. Defaults to True.
        chars_to_ignore_regex (str, optional): [description]. Defaults to '[\,\?\.\!\-\;\:\"]'.

    Returns:
        str: normalized sentence
    """
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    sentence = re.sub(chars_to_ignore_regex, '', sentence)

    if lowercase:
        sentence=sentence.lower()

    # This is to make sure there will not be empty strings after a splitting. So here I split, remove Nones, and rejoin
    sentence=' '.join(list(filter(None, sentence.split(' '))))
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
        if v!=[]:
            if len(v[0])>=len(phones):
                if x_in_y(phones, v[0]):
                    selection[k]=v[0]
    return selection


def syllables_data():
    # http://www.delphiforfun.org/programs/Syllables.htm
    # syllables=pd.read_csv('Syllables.txt',sep='=', header=None)
    syllables=pd.read_csv('data/mhyph.txt', header=None)
    syl_sep=syllables[0][0][5]
    syllables.iloc[:,0]=syllables.iloc[:,0].str.replace(syl_sep,'|')

    # http://hindson.com.au/info/free/free-english-language-hyphenation-dictionary/
    # syllables=pd.read_csv('EnglishHyphDict_v108.txt', header=None, sep=' ')
    # syllables.iloc[:,1]=syllables.iloc[:,1].str.strip(';')
    # syllables.iloc[:,1]=syllables.iloc[:,1].str.replace('-','_')
    # # syllables.iloc[:,0]=syllables.iloc[:,1]
    # syllables=pd.DataFrame(syllables.iloc[:,1])
    # syllables.columns=[0]

    syl_sep='|'

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

    syllables.syllables=syllables.syllables.str.lower()
    
    syllables['normalized_text']=texts
    syllables['phonetics']=phonetics

    syllables['n_syls']=n_syls
    syllables['n_vowels_cmu']=n_vowels_cmu

    syllables[syllables.n_vowels_cmu.isnull()].normalized_text.tolist()
    len(syllables[~syllables.n_vowels_cmu.isnull()].normalized_text.tolist())

    syllables=syllables.dropna()
    syllables[syllables.n_syls==syllables.n_vowels_cmu]
    syllables[syllables.n_syls!=syllables.n_vowels_cmu]

    syllables.to_csv('data/syllables.csv')

    return syllables

def generate_phonetics_from_words(words, indxs):
    phonetics=[]
    for i,word in enumerate(words):
        phonetics.append(SonoriPy(cmudict_dict[word][int(indxs[i])])[0])
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

def sentence_phonetics_alternatives(sentence):
    words=sentence.split(' ')
    word_phonetics=phonetics_alternatives(words)
    return word_phonetics

def generate_phonetics_alternatives(sentences):
    phonetics=[]
    for sent in sentences:
        word_phonetics=sentence_phonetics_alternatives(sent)
        phonetics.append(word_phonetics)
    return phonetics

def syllabified_text(word, syllables_df):
    if cmudict_dict[word]!=[]:
            if n_vowels(cmudict_dict[word][0])==1:
                syls_text=word
                used_method='1 syl in cmu'
                return syls_text, used_method
    try:
        syls_text=syllables_df[syllables_df.normalized_text==word].syllables.values[0]
        used_method='dataset'
    except IndexError:
        # if it did not exist in the manually syllabified data we have:
        # we can first say that words of one syllable is a trivial case, and we can detect that by checking if there is only one syllable
        # if not, I use SonoriPy (sonority sequencing principle) based on letters (it is less accurate than with phonemes but we use it only on fallback)
        # Maybe I could improve this part with logic in this (trailing "e", "-ed" ):
        # https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word

        # for the final -ed, remove the "e" except if "-ded" or "-ted"
        # We remember if we did to insert back the "e" after syllabification
        modified_ed=False
        # smiles -> remove the e,   raises -> don't
        modified_es=False
        trailing_e=False

        if word[-2:]=="ed" and word[-3] not in ['t','d']:
            word=word[:-2]+'d'
            modified_ed=True
        elif word[-2:]=="es" and word[-3] not in ['s']:
            word=word[:-2]+'s'
            modified_es=True
        elif word[-1]=="e":
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


def add_special_char(s_orig, s_modified):
    """This function adds punctuation marks to modified text (here with syllable separation symbols "|") 
    at the end of words from an original sentence.
    This assumes that punctuation marks are glued to words, which is the case in english. 
    This assumption allows us to just check if the last charcter is the same in original and modified text
    "Hello, my name is John."
    "hello my name is john"
    -> hello and john no not have the last same character.

    Example:
    s_orig="I'm taking a Spanish class."
    s_modified="I'm tak|ing a Span|ish class"

    output="I'm tak|ing a Span|ish class."

    Args:
        s_orig (str): original text
        s_modified (str): modified text

    Returns:
        str: modified text with punctuation marks
    """
    s_modified_with_special_chars=[]
    for w_orig, w_modified in zip(s_orig.split(' '), s_modified.split(' ')):
        if w_orig[-1]!=w_modified[-1]:
            s_modified_with_special_chars.append(w_modified+w_orig[-1])
        else:
            s_modified_with_special_chars.append(w_modified)
    return ' '.join(s_modified_with_special_chars)

def prefill_for_sentence(sentence, syllables_data, syl_sep='|'):
    words=remove_special_characters(sentence).split(' ')
    syls_texts=[]
    used_method_syllables=[]
    for word in words:
        syls_texts.append(syllabified_text(word, syllables_data)[0])
        used_method_syllables.append(syllabified_text(word, syllables_data)[1])
    case_syls_texts=insert_seps_in_cased_text(' '.join(syls_texts), remove_special_characters(sentence, lowercase=False), syl_sep=syl_sep)        

    # p -> phonetics
    p=sentence_phonetics_alternatives(remove_special_characters(sentence))

    # g -> gibberish
    # go through levels of the list (alternatives, words, syllables, phones) and then convert every phoneme in gibberish
    g=[[[[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in syl] for syl in word] for word in alt] for alt in p]
    
    # This puts 
    # '|' between phonemes
    # '_' between syllables
    # spaces between words 
    # to have less degrees of nested list and be compatible with the database
    p2=[' '.join(['|'.join(['_'.join(syl) for syl in word]) for word in alt]) for alt in p]
    g2=[' '.join(['|'.join(['_'.join(syl) for syl in word]) for word in alt]) for alt in g]

    # Keep first alternative. Maybe in the future I can store all the alternatives in another variable
    try:
        p2=p2[0]
    except IndexError:
        p2=''
    try:
        g2=g2[0]
    except IndexError:
        g2=''
    
    g_hr=g2.replace('_','')
    n_syl_mismatch=int(len(case_syls_texts.split('|'))!=len(p2.split('|')))

    case_syls_texts=add_special_char(sentence, case_syls_texts)

    record={'text':sentence,
        'cmu_phonetics':p2,
        'pronounciation_guide':g2,
        'pronounciation_guide_hr':g_hr,
        'syllable_parts':case_syls_texts,
        'n_syl_mismatch':n_syl_mismatch,
        'used_method_for_syl_text':used_method_syllables}
    return record

def prefill_content(sentences, syl_sep='|'):
    # syllables=syllables_data()
    syllables=pd.read_csv('data/syllables.csv')
    records=[]
    for s in sentences:
        record=prefill_for_sentence(s, syllables, syl_sep=syl_sep)
        records.append(record)
    df=pd.DataFrame.from_records(records)
    return df


def generate_prefill_csv(path='phrases_speaking_activities.txt', syl_sep='|', out_path='prefill_test.csv'):
    sentences=pd.read_csv(path, sep='/', header=None)
    df=prefill_content(sentences.iloc[:,0].tolist(), syl_sep=syl_sep)
    df.text=sentences

    syl_parts_with_special_characters=[]
    for _,r in df.iterrows():
        syl_parts_with_special_characters.append(add_special_char(r.text, r.syllable_parts))
    
    df.syllable_parts=syl_parts_with_special_characters
    df.to_csv(out_path) 
    return df

if __name__ == "__main__":
    
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

    sentences=pd.read_csv('phrases-for-noe.txt', sep='/', header=None)
    sentences=sentences.iloc[:,0].apply(lambda r: remove_special_characters(r)).tolist()
    df=prefill_content(sentences)

    word=words[0]
    scores_p=[el[-1] for el in SonoriPy(str_to_list_of_char(word), mode='letters')[-1]]

    word=cmudict_dict['comfortable'][0]
    scores_t=[el[-1] for el in SonoriPy(word)[-1]]

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import xticks
    plt.plot(scores_t)
    plt.savefig('sonoripy_comfortable.png')
    xticks(np.arange(len(word)), word)

    syllables_df=syllables_data()
    word='enjoyed'
    syls_text=syllables_df[syllables_df.normalized_text==word].syllables.values[0]
    print(syls_text)

    syllabified_text(word, syllables_df)

    df=generate_prefill_csv()
    df[df.n_syl_mismatch==1][['syllable_parts', 'pronounciation_guide_hr','used_method_for_syl_text']]