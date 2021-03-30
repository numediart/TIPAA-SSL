
import re
import cmudict
from tqdm import tqdm

def get_cmudict_info(word='university'):
    """get the first possible phonetisation of a word from cmudict

    Args:
        word (str, optional): input. Defaults to 'university'.
    Returns:
        list: phonemes and a number for each vowle indicating stress: 0=no stress, 1=primary stress, 2=secondary stress
    """
    return cmudict.dict()[word][0]


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
