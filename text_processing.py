
import re
import cmudict
from tqdm import tqdm

cmudict_dict=cmudict.dict()

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
    for k,v in cmudict.dict().items():
        # cmudict_first_alternatives[k]=v[0]
        if len(v[0])>=len(phones):
            if x_in_y(phones, v[0]):
                selection[k]=v[0]
    return selection


cmu_to_gibberish={'AA':'oh',
                'AE':'ah',
                'AH':'uh',
                'AO':'aw',
                'AW':'au',
                'AY':'ay',
                'B':'b',
                'CH':'ch',
                'D':'d',
                'DH':'th',
                'EH':'eh',
                'ER':'uh',
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

def gibberish_from_phonetics(phonetics=['P', 'R', 'AH0', 'F', 'EH1', 'SH', 'AH0', 'N', 'AH0', 'L']):

    gibberish=[]
    for p in phonetics:
        if p[-1] =='0':
            gibberish.append('uh')
        else:
            if p[-1] in str([0,1,2]): gibberish.append(cmu_to_gibberish[p[:-1]])
            else: gibberish.append(cmu_to_gibberish[p])
            
    return ' '.join(gibberish)

if __name__ == "__main__":
    gibberish_from_phonetics()

    words=['professional', 'analysis', 'temperature', 'personal', 'government']

    gibberishes=[]
    for w in words:
        gibberishes.append(gibberish_from_phonetics(phonetics_from_sentence(w)[0]))