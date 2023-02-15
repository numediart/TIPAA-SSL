import os, psutil;print_memory_usage=lambda stage: print(stage + ": "+ str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
print_memory_usage('RAM - text_processing start')

import re
from tqdm import tqdm
import pandas as pd
from glob import glob
import numpy as np
import itertools
from syllabipy.sonoripy import SonoriPy, str_to_list_of_char, define_categories
from g2p_en.expand import normalize_numbers
from g2p_en import G2p
from itertools import groupby
from num2words import num2words
import unidecode

print_memory_usage('RAM - text_processing after external libraries')


g2p = G2p()
print_memory_usage('RAM - text_processing after g2p model')

drop_consecutive_duplicates= lambda df: df.loc[(df.shift()!=df).sum(axis=1).astype(bool)]
drop_consecutive_duplicate_elements= lambda L: [key for key, _group in groupby(L)]
unstress = lambda el: el[:-1] if el[-1] in str([0,1,2]) else el
split_phonetics = lambda phonetics: [[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
group_consecutive_duplicates= lambda L:[(k, sum(1 for i in g)) for k,g in groupby(L)]

print_memory_usage('RAM - text_processing after lambda functions')
from src.pronunciation_dictionaries import get_augmented_mfa_dict, cmudict_dict, lang_to_MFA_g2p_models, mfa_g2p, cmu_phones, cmu_to_gibberish
print_memory_usage('RAM - text_processing after pronunciation_dictionaries')
# mfa_dicts={lang:get_augmented_mfa_dict(lang) for lang in lang_to_MFA_g2p_models}
# print_memory_usage('RAM - text_processing after mfa_dicts')
from src.syllables_processing import syllabified_text, n_vowels
print_memory_usage('RAM - text_processing after syllable_processsing')


syllables_df={
            'en_GB':pd.read_csv('data/syllables.csv'),
            'en_US':pd.read_csv('data/syllables.csv'),
            'fr_FR':pd.read_csv('data/syllables_fr_FR.csv'),
            'es_ES':pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']),
            'es_LA':pd.DataFrame(columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']),
            }

print_memory_usage('RAM - text_processing after syllables_df')


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
def remove_special_characters(sentence="Where's the best place to have coffee?", lowercase=True, chars_to_ignore_regex = '[\,\?\.\!\¡\;\:\"\*\{\}]'):
    """Normalize text by lowercasing (if option is True), and remove a set of punctuation characters

    Args:
        sentence (str, optional): [description]. Defaults to "Where's the best place to have coffee ?".
        lowercase (bool, optional): [description]. Defaults to True.
        chars_to_ignore_regex (str, optional): [description]. Defaults to '[\,\?\.\!\¡\;\:\"\*]'.

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

def get_chunks(text, chunking_chars=[',',';','.','!','¡','?', ':', '/']):
    for c in chunking_chars:
        text=text.replace(c, chunking_chars[0])
    chunks=text.split(chunking_chars[0])
    chunks = list(filter(None, chunks)) # remove empty string
    return chunks

def chunk_text(text, chunking_chars=[',',';','.','!','¡','?', ':', '/']):
    chunks=get_chunks(text, chunking_chars=chunking_chars)
    # split each chunk in words, remove empty strings, get length (to know the n of words in each chunk)
    n_words_by_chunk=[len(list(filter(None, el.split(' ')))) for el in chunks]
    # assert sum(n_words_by_chunk)==len(list(filter(None, text.split(' ')))), "Checking number of words is the same after chunking"
    return n_words_by_chunk
    
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
        



def generate_syl_phonetics_alternatives_from_word_ipa(word="teórico-práctico", word_dict=get_augmented_mfa_dict('es_ES'), g2p_model="spanish_spain_mfa"):
    # fallbacks
    if '-' in word:
        w_parts=word.split('-')
        p_parts=[]
        for w_part in w_parts:
            try:
                p_part=word_dict[w_part]
            except KeyError: p_part=[]
            if p_part==[]: #raise "phonetics empty, not in pronunciation dictionary"
                # trying first if the word without accents axist in the dict, if not fallback to g2p
                unaccented_string = unidecode.unidecode(w_part)
                try: p_part=word_dict[unaccented_string]
                except KeyError: p_part=mfa_g2p(w_part, model=g2p_model)[w_part]
            p_parts.append(p_part)

        # Here I generate all alternatives of combination of word parts
        # It corresponds to cmu alternatives for other words
        ps=list(itertools.product(*p_parts))
        syls_ps=[[SonoriPy(p, mode='MFA_IPA')[0] for p in alt] for alt in ps]
        
        # This puts 
        # '_' between phonemes
        # '-' between word components (hyphenated words and acronyms, not to have more than 1 stressed syllable by word component)
        # '|' between syllables
        # spaces between words 
        # to have less degrees of nested list and be compatible with the database
        syls_ps_formatted=['-'.join(['|'.join(['_'.join(s) for s in w]) for w in alt]) for alt in syls_ps]
    else:
        try:
            ps=word_dict[word]
        except KeyError:
            ps=[]
        if ps==[]: #raise "phonetics empty, not in pronunciation dictionary"
            # ps=[g2p(word)]
            # trying first if the word without accents axist in the dict, if not fallback to g2p
            unaccented_string = unidecode.unidecode(word)
            try: ps=word_dict[unaccented_string]
            except KeyError: ps=mfa_g2p(word, model=g2p_model)[word]
        
        # Rule for verbs in -ded or -ted: we want to get rid of the "AH0_D" alternative
        # for p in ps:
        #     if (p[-3:]==[ 'T', 'AH0', 'D'] or p[-3:]==[ 'D', 'AH0', 'D']) and (word[-3:]=='ted' or word[-3:]=='ded'):
        #         p[-2:]=['IH0','D']
        
        syls_ps=[SonoriPy(p, mode='MFA_IPA')[0] for p in ps]
        syls_ps_formatted=['|'.join(['_'.join(s) for s in w]) for w in syls_ps]
    return syls_ps_formatted

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
            if (p[-3:]==[ 'T', 'AH0', 'D'] or p[-3:]==['D', 'AH0', 'D']) and (word[-3:]=='ted' or word[-3:]=='ded'):
                p[-2:]=['IH0','D']
        
        syls_ps=[SonoriPy(p)[0] for p in ps]
        syls_ps_formatted=['|'.join(['_'.join(s) for s in w]) for w in syls_ps]
        syls_gs_formatted=['|'.join(['_'.join([cmu_to_gibberish[unstress(el)] for el in s]) for s in w]) for w in syls_ps]
    return syls_ps_formatted, syls_gs_formatted

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


def normalize_sentence_numbers(sentence, lang="en_GB", mode="CMU"):
    # this takes care of e.g. "2021", "$110"
    # curly braces for numbers, if they are in several words
    # also add curly braces in original sentence around numbers

    def normalize(word, lang="en"):
        if mode=="CMU":
            n_word=normalize_numbers(word)
        else:
            try:
                w=float(word)
                n_word=num2words(w, lang=lang)
            except ValueError:
                n_word=word
        return n_word

    norm_sent_list=[]
    sent_list=[]
    for word in sentence.split(' '):
        n_word=normalize(word, lang=lang)
        if ' ' in n_word:
            n_el="{"+n_word+"}"
            el="{"+word+"}"
        else: 
            n_el=n_word
            el=word
        norm_sent_list.append(n_el)
        sent_list.append(el)
    norm_sent=" ".join(norm_sent_list)
    sent=" ".join(sent_list)
    return norm_sent, sent

def extract_special_chars(norm_sent, special_chars):
    # memorize special characters glued before and after words
    special_chars_dict_end={}
    special_chars_dict_start={}
    for i,w in enumerate(norm_sent.split(' ')):
        is_special_char=[c in special_chars for c in w]

        # count number of repeated values to know how much special chars is start and at end if any
        special_char_s_in_word = [(v, sum(1 for _ in group)) for v, group in groupby(is_special_char)]
        if special_char_s_in_word[0][0]:
            special_chars_dict_start[i]=w[:special_char_s_in_word[0][1]]
        if special_char_s_in_word[-1][0]:
            special_chars_dict_end[i]=w[-special_char_s_in_word[-1][1]:]
    return special_chars_dict_start, special_chars_dict_end

get_acronyms_idxs=lambda words: [w_idx for w_idx,w in enumerate(words) if w.isupper()]


# sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una empresa emergente de 30000 dólares en etapa inicial, ¡luego con el CTO!"
# sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una start-up de 30000 dólares en etapa inicial, ¡luego con el CTO!"
# sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
def prefill_for_sentence(
    sentence="I paid a $3000 bill when visiting UCLA, it's an expensive hotel, for the 21st century!",
    # sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!", 
                        syllables_df=pd.read_csv('data/syllables.csv'), 
                        syl_sep='|', 
                        special_chars = [',','?','.','!','¡',';',':','"', '{', '}', '*'],
                        lang="en_US",
                        mode='CMU'):  # "CMU" or "MFA_IPA"
    """This function extract information of syllabified texts and phonetics. 
    It uses a combination of datasets (CMUdict, data from syllable_data() ) and algorithm (SonoriPy)

    Args:
        sentence (str): a phrase to be processed. It can contain captial letters and punctuation.
        syllables_df (DataFrame): The syllables dataset built from syllables_data()
        syl_sep (str, optional): [description]. Defaults to '|'.

    Returns:
        dict: see structure a the end of the function
    """
    if mode=="MFA_IPA":
        word_dict=get_augmented_mfa_dict(lang)
    else:
        word_dict=cmudict_dict

    sentence=sentence.strip()
    # there shouldn't be a space before a special char, they must be glued to words (in english)
    # correct that if it's not the case
    for c in special_chars: 
        if c not in ['¡', '*']:  # these punctuation mark can be at the beginning of a word (spanish exception, and our asterisk mark for target words)
            sentence=sentence.replace(' '+c, c)
    
    norm_sent, c=normalize_sentence_numbers(sentence, lang=lang, mode=mode)
    special_chars_dict_start, special_chars_dict_end=extract_special_chars(norm_sent, special_chars)
    words=remove_special_characters(norm_sent, lowercase=False).split(' ')

    # little dictionnary mapping e.g. Mr -> Mister
    words=[expand_dict[word] if word in expand_dict else word for word in words]
    norm_words=words
    
    # If all letters are capital (acronym), put '-' between all letters
    # I need to do that before putting in lowercase, that is why I cannot put that in e.g. syllabified_text()
    acronym_idxs=get_acronyms_idxs(words)
    ws=[]
    for w_idx,w in enumerate(words):
        if w.isupper():
            w='-'.join(w)
            words[w_idx]=w
        ws.append(w)
    words=[word.lower() for word in ws]

    # p -> phonetics
    # g -> gibberish
    if mode!='MFA_IPA':
        p=[generate_syl_phonetics_alternatives_from_word(word)[0] for word in words]
        # I separate the acronym in letters and lookup cmudict. I take the last, because
        # there is only one alternative except for letter 'A' which has  [['AH0'], ['EY1']]
        for i in acronym_idxs:  p[i]=['-'.join(['_'.join(cmudict_dict[w][-1]) for w in words[i].split('-')])]
    else:
        p=[generate_syl_phonetics_alternatives_from_word_ipa(word, word_dict=word_dict, g2p_model=lang_to_MFA_g2p_models[lang]) for word in words]


    syls_texts=[]
    used_method_syllables=[]
    for word, phones in zip(words,p):
        # # This split and rejoin will apply only if there is a dash and then allow to treat separately parts of a word with dash
        # if False:
        if '-' in word:
            syls_texts_parts=[]
            used_method_syllables_parts=[]
            for w_part, phones_part in zip(word.split('-'),phones[0].split('-')):
                n=n_vowels(phones_part.replace('|','_').split('_'), mode=mode)
                # print("w phones_part:",phones_part,', n_vowels:', n)
                syls_texts_parts.append(syllabified_text(w_part, n, syllables_df, lang=lang)[0])
                used_method_syllables_parts.append(syllabified_text(w_part, n, syllables_df, lang=lang)[1])
            # syls_texts.append('|-'.join(syls_texts_parts))
            syls_texts.append('-'.join(syls_texts_parts))
            used_method_syllables.append('-'.join(used_method_syllables_parts))
        else:
            n=n_vowels(phones[0].replace('|','_').split('_'), mode=mode)
            # print("w phones:",phones[0],', n_vowels:', n)
            syls_texts.append(syllabified_text(word, n, syllables_df, lang=lang)[0])
            used_method_syllables.append(syllabified_text(word, n, syllables_df, lang=lang)[1])
    
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
    def acronyms_to_compound(split_text, acronym_idxs):
            # remove '-' in acronyms
            # And if it was only 1 letter, then it's not a compound word.
            for i,el in enumerate(split_text):
                if i in acronym_idxs:
                    if len(el)>1:
                        split_text[i]='{'+el.replace('-',' ')+'}'
                    else:
                        split_text[i]=el.replace('{','').replace('}','')
            return split_text
    
    case_syls_texts_special_chars=acronyms_hyphen_to_compound(case_syls_texts_special_chars, acronym_idxs)
    # this adds punctuation and the curly brackets:
    case_syls_texts_special_chars=add_special_chars(case_syls_texts_special_chars, special_chars_dict_start, special_chars_dict_end)

    segmented_text=' '.join(case_syls_texts_special_chars)

    # Here I want to remove only punctuation
    segmented_text=remove_special_characters(sentence=segmented_text, lowercase=False, chars_to_ignore_regex = '[\,\?\.\!\¡\;\:\"\*]')
    


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
    
    brace_dict_start=dict(filter(lambda el: el[1] in ['{','}'], special_chars_dict_start.items()))
    brace_dict_end=dict(filter(lambda el: el[1] in ['{','}'], special_chars_dict_end.items()))

    p_0_special_chars=acronyms_hyphen_to_compound(p_0.split(' '), acronym_idxs)
    p_0_special_chars=add_special_chars(p_0_special_chars, brace_dict_start, brace_dict_end)
    p_0_special_chars=' '.join(p_0_special_chars)

    if mode!='MFA_IPA':
        g_0=' '.join(['-'.join(['|'.join(['_'.join([cmu_to_gibberish[unstress(p)] for p in s]) for s in sub_w]) for sub_w in w]) for w in split_phonetics(p_0)])
        g_0_special_chars=acronyms_hyphen_to_compound(g_0.split(' '), acronym_idxs)
        g_0_special_chars=add_special_chars(g_0_special_chars, brace_dict_start, brace_dict_end)
        g_0_special_chars=' '.join(g_0_special_chars)

    # processing on the raw text to add curly brackets around compound words
    special_chars_dict_start_raw, special_chars_dict_end_raw=extract_special_chars(sentence, special_chars)
    sent_=acronyms_to_compound(remove_special_characters(sentence=sentence, lowercase=False, chars_to_ignore_regex = '[\,\?\.\!\¡\;\:\"\*\{\}]').split(' '), get_acronyms_idxs(remove_special_characters(sentence, lowercase=False).split(' ')))
    sent_brackets=' '.join(add_special_chars(sent_, special_chars_dict_start_raw, special_chars_dict_end_raw))

    if mode!='MFA_IPA':
        record={'text':sent_brackets,
            'cmu_phonetics':p_0_special_chars,
            'pronounciation_guide_hr':g_0_special_chars.replace('_',''),
            'segmented_text':segmented_text,
            'n_syl_mismatches':word_idxs_inconsitencies['n_syl'],
            'n_stress_inconsistencies':word_idxs_inconsitencies['stress'],
            'used_method_for_syl_text':used_method_syllables,
            'cmu_phonetics_alt':p,
            'n_alternatives':n_alternatives
            }
    else:
        record={'text':sent_brackets,
            'cmu_phonetics':p_0_special_chars,
            'segmented_text':segmented_text,
            'n_syl_mismatches':word_idxs_inconsitencies['n_syl'],
            'n_stress_inconsistencies':word_idxs_inconsitencies['stress'],
            'used_method_for_syl_text':used_method_syllables,
            'cmu_phonetics_alt':p,
            'n_alternatives':n_alternatives
            }
    return record

def prefill_content(sentences, syl_sep='|', lang='en_US', mode='CMU'):
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
            record=prefill_for_sentence(s, syllables_df[lang], syl_sep=syl_sep, mode=mode, lang=lang)
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



def word_stress_from_cmu(phonetics=['K', 'AA1', 'F', 'IY0']):
    # cmu vowels end by a number : 0, 1 or 2.   0= no stress, 1 = primary stress, 2 = secondary stress
    # consonants do not end by a number
    # here I return a list that is one if primary stressed and else 0
    return [1 if p[-1]==str(1) else 0 for p in phonetics if p[-1] in str([0,1,2])]
    
# unused functions
if False:
    def word_selection():
        # words that finish in "s" with phoneme "S" that also exist without an "s" and with last phoneme then not being "S"
        words_in_s=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='S' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='S']
        words_in_z=[el for el in cmudict_dict.keys() if el[-1]=='s' and cmudict_dict[el][0][-1]=='Z' and el[:-1] in cmudict_dict and cmudict_dict[el[:-1]][0][-1]!='Z']
    def get_cmudict_info(word='university'):
        """get the first possible phonetisation of a word from cmudict

        Args:
            word (str, optional): input. Defaults to 'university'.
        Returns:
            list: phonemes and a number for each vowel indicating stress: 0=no stress, 1=primary stress, 2=secondary stress
        """
        return cmudict_dict[word][0]

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


print_memory_usage('RAM - text_processing after all function declarations')


def use_tests():
    # from src.text_processing import *
    prefill_for_sentence()

    
    sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una empresa emergente de 30000 dólares en etapa inicial, ¡luego con el CTO!"
    sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!"
    # sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    sentence="A 22 heures, j'ai rendez-vous avec le CEO, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"

    lang="fr_FR"
    r=prefill_for_sentence(
                        sentence=sentence,
                        syllables_df=syllables_df[lang], 
                        lang=lang,
                        mode='MFA_IPA')  # "CMU" or "MFA_IPA"
    
    from src.label_data_processing import actor_recordings
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

    content=pd.read_csv('/mnt/c/Users/noe_t/Downloads/All content minus audio 2023-02-14 - Sheet1.csv')
    sentences=content.words.apply(lambda r: remove_special_characters(r)).tolist()



    # https://www.angmohdan.com/22-words-with-british-and-american-pronunciations-that-may-confuse-you/
    words="Advertisement,Bald,Clique,Either,Envelope,Esplanade,Leisure,Mobile,Missile,Neither,Niche,Often,Parliament,Privacy,Semi,Schedule,Scone,Stance,Tomato,Vase,Vitamin,Wrath".split(',')
    df_us_mfa=prefill_content(words, lang="en_US", mode="MFA_IPA")
    df_gb_mfa=prefill_content(words, lang="en_GB", mode="MFA_IPA")
    df_us_cmu=prefill_content(words)

    # ' '.join(['|'.join(['_'.join([arpabet_to_2_char_ipa[unstress(p).lower()] for p in syl]) for syl in word]) for word in split_phonetics('AE0|D_V_ER1|T_AH0|Z_M_AH0_N_T')])

    from src.pronunciation_dictionaries import arpabet_to_2_char_ipa, mfa_to_display_ipa

    
    import unicodedata
    # https://stackoverflow.com/questions/61811872/how-do-i-remove-subscript-superscript-in-python
    remove_superscripts = lambda s: "".join(c for c in s if (unicodedata.category(c) not in ["No", "Lo", "Lm"]) or (c=="ː"))

    map_phonemes= lambda mapper, formatted_phonetics: ' '.join(['|'.join(['_'.join([mapper[unstress(p).lower()] for p in syl]) for syl in word]) for word in split_phonetics(formatted_phonetics)])

    df_us_cmu_to_ipa=df_us_cmu.cmu_phonetics.apply(lambda r:map_phonemes(arpabet_to_2_char_ipa, r))
    df_us_mfa_disp=df_us_mfa.cmu_phonetics.apply(lambda r:map_phonemes(mfa_to_display_ipa, r)).apply(lambda r: remove_superscripts(r))
    df_gb_mfa_disp=df_gb_mfa.cmu_phonetics.apply(lambda r:map_phonemes(mfa_to_display_ipa, r)).apply(lambda r: remove_superscripts(r))

    df_all_variations=pd.DataFrame([df_us_cmu.text, df_us_cmu_to_ipa, df_us_mfa.cmu_phonetics, df_us_mfa_disp, df_gb_mfa.cmu_phonetics,df_gb_mfa_disp]).T
    df_all_variations.columns=['text','cmu_to_ipa','mfa_us','mfa_us_disp','mfa_gb','mfa_gb_disp']
    df_all_variations.replace('_','', regex=True).replace('\|','', regex=True).to_csv('ipa_variations.csv')

    df_us_mfa_disp.apply(lambda r: remove_superscripts(r))



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