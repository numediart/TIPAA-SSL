import os, psutil

print_memory_usage = lambda stage: print(
    stage + ": " + str(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
)
print_memory_usage('RAM - text_processing start')

import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
from syllabipy.sonoripy import SonoriPy, str_to_list_of_char, define_categories
from src.numbers_processing import normalize_numbers

# from g2p_en.expand import normalize_numbers
from g2p_en import G2p
from itertools import groupby
from num2words import num2words
import unidecode

from src.phonemizer_utils import word_to_stressed_syl

print_memory_usage('RAM - text_processing after external libraries')


g2p = G2p()
print_memory_usage('RAM - text_processing after g2p model')

drop_consecutive_duplicates = lambda df: df.loc[
    (df.shift() != df).sum(axis=1).astype(bool)
]
drop_consecutive_duplicate_elements = lambda L: [key for key, _group in groupby(L)]
split_phonetics = lambda phonetics: [
    [s.split('_') for s in w.split('|')] for w in phonetics.split(' ')
]
group_consecutive_duplicates = lambda L: [(k, len(list(g))) for k, g in groupby(L)]

from src.pronunciation_dictionaries import (
    unstress,
    remove_stress_annots,
    differs_by_one_insertion,
)


print_memory_usage('RAM - text_processing after lambda functions')
from src.pronunciation_dictionaries import (
    get_augmented_mfa_dict,
    cmudict_dict,
    lang_to_MFA_g2p_models,
    mfa_g2p,
    cmu_phones,
    cmu_to_gibberish,
    normalize_termination,
)

print_memory_usage('RAM - text_processing after pronunciation_dictionaries')
# mfa_dicts={lang:get_augmented_mfa_dict(lang) for lang in lang_to_MFA_g2p_models}
# print_memory_usage('RAM - text_processing after mfa_dicts')
from src.syllables_processing import syllabified_text, n_vowels

print_memory_usage('RAM - text_processing after syllable_processsing')


syllables_dfs = {
    'en_GB': pd.read_csv('data/syllables.csv'),
    'en_US': pd.read_csv('data/syllables.csv'),
    'fr_FR': pd.read_csv('data/syllables_fr_FR.csv'),
    'es_ES': pd.DataFrame(
        columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']
    ),
    'es_LA': pd.DataFrame(
        columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']
    ),
}

print_memory_usage('RAM - text_processing after syllables_df')


# class Phonetics(str):
#     pass


# class PhoneList(list[str]):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs


def show_alternatives_distributions():
    import numpy as np

    lens = []
    for k, v in cmudict_dict.items():
        lens.append(len(v))
    print(np.histogram(lens, bins=[0, 1, 2, 3, 4, 5, 6, 7]))


# expand some words as Mr or Mrs to Mister and Misses
expand_dict = {
    'mr': 'mister',
    'mrs': 'misses',
    'dr': 'docteur',
    'Mr': 'Mister',
    'Mrs': 'Misses',
    'Dr': 'Docteur',
    'etc': 'etcetera',
}


def remove_special_characters(
    sentence="Where's the best place to have coffee?",
    lowercase=True,
    chars_to_ignore_regex='[\,\?\.\!\¡\;\:\"\*\{\}]',
):
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
    sentence = sentence.replace("’", "'")

    if lowercase:
        sentence = sentence.lower()

    # This is to make sure there will not be empty strings after a splitting. So here I split, remove Nones, and rejoin
    sentence = ' '.join(list(filter(None, sentence.split(' '))))
    return sentence


def get_chunks(text, chunking_chars=[',', ';', '.', '!', '¡', '?', ':', '/']):
    for c in chunking_chars:
        text = text.replace(c, chunking_chars[0])
    chunks = text.split(chunking_chars[0])
    chunks = list(filter(None, chunks))  # remove empty string
    return chunks


def chunk_text(text, chunking_chars=[',', ';', '.', '!', '¡', '?', ':', '/']):
    chunks = get_chunks(text, chunking_chars=chunking_chars)
    # split each chunk in words, remove empty strings, get length (to know the n of words in each chunk)
    n_words_by_chunk = [len(list(filter(None, el.split(' ')))) for el in chunks]
    # assert sum(n_words_by_chunk)==len(list(filter(None, text.split(' ')))), "Checking number of words is the same after chunking"
    return n_words_by_chunk


def phonetics_indexed_df_from_formatted_phonetics(phonetics: str) -> pd.DataFrame:
    """This function takes a formatted phonetics string (sentence) and returns a dataframe
    with the phones, the word index and the syllable index.

    Parameters
    ----------
    phonetics : str
        formatted phonetics string
        Example: "HH_AW1 L_AH1|V_L_IY0" ("how lovely")

    Returns
    -------
    pandas.DataFrame
        dataframe with the phones, the word index and the syllable index, example:

          phones  word_idx  syl_idx
        0     HH         0        0
        1    AW1         0        0
        2      L         1        0
        3    AH1         1        0
        4      V         1        1
        5      L         1        1
        6    IY0         1        1
    """
    word_split_phonetics = [p.replace('|', '_').split('_') for p in phonetics.split(' ')]
    n_phone_by_word = [len(w) for w in word_split_phonetics]

    word_indices = []
    for i, n in enumerate(n_phone_by_word):
        word_indices += [i] * n
    split_phonetics = [[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    n_phone_by_syl = [[len(s) for s in w] for w in split_phonetics]

    all_syl_indices = []
    for n_phones in n_phone_by_syl:
        syl_indices = []
        for i, n in enumerate(n_phones):
            syl_indices += [i] * n
        all_syl_indices += syl_indices

    phone_list = sum(sum(split_phonetics, []), [])

    phonetics_indexed_df = pd.DataFrame()
    phonetics_indexed_df['phones'] = phone_list
    phonetics_indexed_df['word_idx'] = word_indices
    phonetics_indexed_df['syl_idx'] = all_syl_indices
    phonetics_indexed_df.apply(lambda r: r['phones'], axis=1)

    return phonetics_indexed_df


def cmu_ensure_phonetics_consistency(phonetics: str) -> str:
    """Replace CMU 'double' phones (CH, JH) by their
    single-phone equivalents (T_SH, D_ZH)"""
    return phonetics.replace('CH', 'T_SH').replace('JH', 'D_ZH')


def check_phonemes(phonemes):
    """This function returns a non existing phoneme if it happens. else it returns None"""
    ps = remove_stress_annots(phonemes)
    for p in ps:
        if p not in cmu_phones:
            return p


# adding stresses to vowels thanks to "word_to_stressed_syl" that got us stressed syllables from espeak sonoripyed
def add_stress(syls_ps, word_stress):
    vowels = define_categories(mode='MFA_IPA')['vowels']
    for alt_idx, alt in enumerate(syls_ps):
        # Here I only put the stresses if there is consistency in terms of number of syllables.
        # This will be detected in stress inconsistencies and can be manually corrected
        if len(alt) == word_stress[-1]:  # , "The number of syllables should be the same"
            for syl_idx, syl_p in enumerate(alt):
                # print(syl_p)
                stress_mark = "1" if syl_idx == word_stress[0] else "0"
                syl_p = [p + stress_mark if p in vowels else p for p in syl_p]
                syls_ps[alt_idx][syl_idx] = syl_p


# adding stresses to vowels thanks to "word_to_stressed_syl" that got us stressed syllables from espeak sonoripyed
def add_stress_w_parts(syls_ps, word_stress_syls_parts):
    vowels = define_categories(mode='MFA_IPA')['vowels']
    for alt_idx, alt in enumerate(syls_ps):
        assert len(word_stress_syls_parts) == len(
            alt
        ), "The number of word parts should be the same"
        for w_part_idx, syls_p in enumerate(alt):
            # Here I only put the stresses if there is consistency in terms of number of syllables.
            # This will be detected in stress inconsistencies and can be manually corrected
            if (
                len(syls_p) == word_stress_syls_parts[w_part_idx][-1]
            ):  # , "The number of syllables should be the same"
                for syl_idx, syl_p in enumerate(syls_p):
                    # print(syl_p)
                    stress_mark = (
                        "1" if syl_idx == word_stress_syls_parts[w_part_idx][0] else "0"
                    )
                    syl_p = [p + stress_mark if p in vowels else p for p in syl_p]
                    syls_ps[alt_idx][w_part_idx][syl_idx] = syl_p


# word_dict=get_augmented_mfa_dict('en_US')
# generate_syl_phonetics_alternatives_from_word_ipa(word="start", lang='en_US', word_dict=word_dict, g2p_model=lang_to_MFA_g2p_models['en_US'])
# generate_syl_phonetics_alternatives_from_word_ipa(word="powered", lang='en_US', word_dict=word_dict, g2p_model=lang_to_MFA_g2p_models['en_US'])
def generate_syl_phonetics_alternatives_from_word_ipa(
    word="teórico-práctico",
    lang='es_ES',
    word_dict=get_augmented_mfa_dict('es_ES'),
    g2p_model="spanish_spain_mfa",
):
    # fallbacks
    if '-' in word:
        w_parts = word.split('-')
        p_parts = []
        word_stress_syls_parts = []
        for w_part in w_parts:
            try:
                p_part = word_dict[w_part]
            except KeyError:
                p_part = []
            if p_part == []:  # raise "phonetics empty, not in pronunciation dictionary"
                # trying first if the word without accents axist in the dict, if not fallback to g2p
                unaccented_string = unidecode.unidecode(w_part)
                try:
                    p_part = word_dict[unaccented_string]
                except KeyError:
                    p_part = mfa_g2p(w_part, model=g2p_model)[w_part]
            p_parts.append(p_part)
            word_stress_syls_parts.append(word_to_stressed_syl(w_part, lang=lang))

        # Here I generate all alternatives of combination of word parts
        # It corresponds to MFA_IPA alternatives for other words
        ps = list(itertools.product(*p_parts))
        syls_ps = [[SonoriPy(p, mode='MFA_IPA')[0] for p in alt] for alt in ps]

        # adding stresses, by reference, to vowels thanks to "word_to_stressed_syl" that got us stressed syllables from espeak sonoripyed
        add_stress_w_parts(syls_ps, word_stress_syls_parts)

        # This puts
        # '_' between phonemes
        # '-' between word components (hyphenated words and acronyms, not to have more than 1 stressed syllable by word component)
        # '|' between syllables
        # spaces between words
        # to have less degrees of nested list and be compatible with the database
        syls_ps_formatted_corr = [
            '-'.join(['|'.join(['_'.join(s) for s in w]) for w in alt]) for alt in syls_ps
        ]
    else:
        try:
            ps = word_dict[word]
        except KeyError:
            ps = []
        if ps == []:  # raise "phonetics empty, not in pronunciation dictionary"
            # ps=[g2p(word)]
            # trying first if the word without accents axist in the dict, if not fallback to g2p
            unaccented_string = unidecode.unidecode(word)
            try:
                ps = word_dict[unaccented_string]
            except KeyError:
                ps = mfa_g2p(word, model=g2p_model)[word]

        word_stress = word_to_stressed_syl(word, lang=lang)

        # Note: for -ed and -es ending, MFA_IPA seems consistent for the ['ɪ', 'd'] and ['ɪ', 'z'] on the contrary to CMU, so no normalization needed

        syls_ps = [SonoriPy(p, mode='MFA_IPA')[0] for p in ps]
        syls_ps_formatted = ['|'.join(['_'.join(s) for s in w]) for w in syls_ps]

        # I am trying to figure out the best decision in terms of consistency from the different variations of english and phoneme/phone sets for a specific case: the words containing what corresponds to our gibberish "eer", like in "fear, beer, fierce, here, career, ..."

        # GB:
        # f_ɪ_ə
        # b_ɪ_ə
        # f_ɪ_ə_s

        # US:
        # f_ɪ_ɹ
        # b_ɪ_ɹ
        # f_ɪ_ɹ_s

        # When I do syllabification, I always consider that there should be exactly 1 vowel in each syllable if we consider diphtongs (like ay, oy, ow, aw) as a single vowel.
        # However here, these are not considered diphtongs in MFA IPA set. So in GB it would be 2 syllables and in US

        # I could've change the convention and say that ɪə can be a single vowel, i.e. it's considered as a diphtong
        # However I am not sure what this change would imply on a tech trained on IPA. It means either post-processing alignment frames to merge the ɪ_ə sequence in GB, inside the training corpus for the pipeline tech, or do the merge at inference (when using the model)

        # The other solution, that I choose here, is to do this change only in the notation. As I use this notation in at inference, it would work.
        # To avoid having a double vowel in 1 syllable, that would violate the rule of 1 vowel per syllable, this schwa is considered in this context, not a "real" vowel...
        # I thus put the stress pattern on the ɪ, and nothing on the scwha: fear -> f_ɪ1_ə

        # .replace("ɪ|ə", "ɪ_ə")
        syls_ps_formatted2 = [
            el.replace("ɪ|ə", "ɪ_ə").replace("i|ə", "i_ə") for el in syls_ps_formatted
        ]
        syls_ps_corr = [[s.split('_') for s in w.split('|')] for w in syls_ps_formatted2]
        add_stress(syls_ps_corr, word_stress)

        # If the word_stress extracted by espeak does not contain a schwa, and there are alternatives that are just 1 more syllable and contain a schwa
        # I want to add stress by keeping the correct index. For that, I need to know in which syllable is the schwa and either add 1 to the index if it is after the schwa
        # or keep the index as is if it is before
        if word_stress[0] is not None:
            has_n_syls_idxs = []
            has_n_plus_one_syls_idxs = []
            for i, alt in enumerate(syls_ps_corr):
                if len(alt) == word_stress[-1]:
                    # print(alt)
                    has_n_syls_idxs.append(i)
                elif len(alt) == word_stress[-1] + 1:
                    has_n_plus_one_syls_idxs.append(i)
            for n_plus_one_idx in has_n_plus_one_syls_idxs:
                if 'ə' in ps[n_plus_one_idx]:
                    for n_idx in has_n_syls_idxs:
                        b, phone, insert_idx = differs_by_one_insertion(
                            ps[n_plus_one_idx], ps[n_idx]
                        )
                        if b:
                            schwa_idx = [
                                'ə' in remove_stress_annots(syl)
                                for syl in syls_ps_corr[n_plus_one_idx]
                            ].index(True)
                            if word_stress[0] >= schwa_idx:
                                add_stress(
                                    syls_ps_corr, (word_stress[0] + 1, word_stress[1] + 1)
                                )
                            else:
                                add_stress(
                                    syls_ps_corr, (word_stress[0], word_stress[1] + 1)
                                )

        # words like "realize", "idea" also have and "ɪ|ə" for which we want to keep the alternative
        # espeak will typically have 1 syllable less because it consider a diphtong while I don't (it's not in cmudict, nor in MFA). idea -> "aɪ_d_ˈiə"
        syls_ps_ie = [
            [s.split('_') for s in w.split('|')]
            for w in syls_ps_formatted
            if "ɪ|ə" in w or "i|ə" in w
        ]
        syls_ps_corr += syls_ps_ie
        add_stress(syls_ps_corr, (word_stress[0], word_stress[1] + 1))
        # this won't overwrite previous because stressed vowels wouldn't be recognized as vowels, but will fill optional possibilities like for "media"
        add_stress(syls_ps_corr, (word_stress[0], word_stress[1]))

        # if there is no stress found (happens in short words like "at, the, of, our"), also add the (None, 1) possibility.
        # Because for e.g. "our", espeak outputs 2 syllables, although there are also 1 syllables alternatives that we want to keep
        if word_stress[0] is None:
            add_stress(syls_ps_corr, (None, 1))

        syls_ps_formatted_corr = [
            '|'.join(['_'.join(s) for s in w]) for w in syls_ps_corr
        ]
        syls_ps_formatted_corr = [
            el.replace("ɪ1_ə1", "ɪ1_ə")
            .replace("ɪ0_ə0", "ɪ0_ə")
            .replace("i1_ə1", "i1_ə")
            .replace("i0_ə0", "i0_ə")
            for el in syls_ps_formatted_corr
        ]

    return syls_ps_formatted_corr


def generate_syl_phonetics_alternatives_from_word(word="before"):
    """ "Looks up in cmudict for phonetic alternatives, and apply SonoriPy on all alternatives

    Args:
        word (str, optional): [description]. Defaults to "before".

    Returns:
        list: the nested list contains alternatives -> words -> syllables -> phones.
        for default input: [[['B', 'IH0'], ['F', 'AO1', 'R']], [['B', 'IY2'], ['F', 'AO1', 'R']]]
    """

    # fallbacks
    if '-' in word:
        w_parts = word.split('-')
        p_parts = []
        for w_part in w_parts:
            p_part = cmudict_dict[w_part]
            if p_part == []:
                p_part = [g2p(w_part)]
            p_parts.append(p_part)

            normalize_termination(p_part, w_part)

        # Here I generate all alternatives of combination of word parts
        # It corresponds to cmu alternatives for other words
        ps = list(itertools.product(*p_parts))
        syls_ps = [[SonoriPy(p)[0] for p in alt] for alt in ps]

        # This puts
        # '_' between phonemes
        # '-' between word components (hyphenated words and acronyms, not to have more than 1 stressed syllable by word component)
        # '|' between syllables
        # spaces between words
        # to have less degrees of nested list and be compatible with the database
        syls_ps_formatted = [
            '-'.join(['|'.join(['_'.join(s) for s in w]) for w in alt]) for alt in syls_ps
        ]
        syls_gs_formatted = [
            '-'.join(
                [
                    '|'.join(
                        [
                            '_'.join([cmu_to_gibberish[unstress(el)] for el in s])
                            for s in w
                        ]
                    )
                    for w in alt
                ]
            )
            for alt in syls_ps
        ]

    else:
        ps = cmudict_dict[word]
        if ps == []:
            ps = [g2p(word)]

        normalize_termination(ps, word)

        syls_ps = [SonoriPy(p)[0] for p in ps]
        syls_ps_formatted = ['|'.join(['_'.join(s) for s in w]) for w in syls_ps]
        syls_gs_formatted = [
            '|'.join(['_'.join([cmu_to_gibberish[unstress(el)] for el in s]) for s in w])
            for w in syls_ps
        ]
    return syls_ps_formatted, syls_gs_formatted


def find(s, ch=['|']):
    return [i for i, ltr in enumerate(s) if ltr in ch]


def insert(s, ch, i):
    return s[:i] + ch + s[i:]


# processing steps for "prefill_for_sentence"
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
    s_case_sep = []
    for w, w_case in zip(s.split(' '), s_case.split(' ')):
        idxs = find(w, [syl_sep])
        # now that we found indices of where the syllable separators are, we need to find after which character to insert them
        # as if the separators weren't there. So we need to substract the number of separators there was before, i.e. its index in the list
        char_idxs = [el - i for i, el in enumerate(idxs)]

        # here we insert the separator in the cased text thanks to the indices. It is important to do that
        # starting from the end, so that the other indices are not afftected by these insertions
        for idx in char_idxs[::-1]:
            w_case = insert(w_case, syl_sep, idx)
        s_case_sep.append(w_case)
    return ' '.join(s_case_sep)


def normalize_sentence_numbers(sentence, lang="en_GB"):
    # this takes care of e.g. "2021", "$110"
    # curly braces for numbers, if they are in several words
    # also add curly braces in original sentence around numbers
    def normalize(word, lang="en_GB"):
        if lang.split('_')[0] == "en":
            n_word = normalize_numbers(word)
        else:
            try:
                w = float(word)
                n_word = num2words(w, lang=lang)
            except ValueError:
                n_word = word
        return n_word

    norm_sent_list = []
    sent_list = []
    for word in sentence.split(' '):
        n_word = normalize(word, lang=lang)
        if ' ' in n_word:
            n_el = "{" + n_word + "}"
            el = "{" + word + "}"
        else:
            n_el = n_word
            el = word
        norm_sent_list.append(n_el)
        sent_list.append(el)
    norm_sent = " ".join(norm_sent_list)
    sent = " ".join(sent_list)
    return norm_sent, sent


def extract_special_chars(norm_sent, special_chars):
    # memorize special characters glued before and after words
    special_chars_dict_end = {}
    special_chars_dict_start = {}
    for i, w in enumerate(norm_sent.split(' ')):
        is_special_char = [c in special_chars for c in w]

        # count number of repeated values to know how much special chars is start and at end if any
        special_char_s_in_word = [
            (v, sum(1 for _ in group)) for v, group in groupby(is_special_char)
        ]
        if special_char_s_in_word[0][0]:
            special_chars_dict_start[i] = w[: special_char_s_in_word[0][1]]
        if special_char_s_in_word[-1][0]:
            special_chars_dict_end[i] = w[-special_char_s_in_word[-1][1] :]
    return special_chars_dict_start, special_chars_dict_end


get_acronyms_idxs = lambda words: [w_idx for w_idx, w in enumerate(words) if w.isupper()]


def add_special_chars(split_text, special_chars_dict_start, special_chars_dict_end):
    for k in special_chars_dict_end:
        split_text[k] += special_chars_dict_end[k]
    for k in special_chars_dict_start:
        split_text[k] = special_chars_dict_start[k] + split_text[k]
    return split_text


def acronyms_hyphen_to_compound(split_text, acronym_idxs):
    # remove '-' in acronyms
    # And if it was only 1 letter, then it's not a compound word.
    for i, el in enumerate(split_text):
        if i in acronym_idxs:
            if '-' in el:
                split_text[i] = '{' + el.replace('-', ' ') + '}'
            else:
                split_text[i] = el.replace('{', '').replace('}', '')
    return split_text


def acronyms_to_compound(split_text, acronym_idxs):
    # remove '-' in acronyms
    # And if it was only 1 letter, then it's not a compound word.
    for i, el in enumerate(split_text):
        if i in acronym_idxs:
            if len(el) > 1:
                split_text[i] = '{' + el.replace('-', ' ') + '}'
            else:
                split_text[i] = el.replace('{', '').replace('}', '')
    return split_text


def text_normalization(
    sentence="I paid a $3000 bill when visiting UCLA… it's an expensive hotel, for the 21st century!",
    lang="en_US",
    special_chars=[',', '?', '.', '!', '¡', ';', ':', '"', '{', '}', '*', '…'],
    syllables_df=pd.read_csv('data/syllables.csv'),
    syl_sep='|',
):
    """Text normalization:
        -process special characters
        -process numbers to text. Ordinal work only in english
        -process acronyms
        -syllabify text

    Args:
        sentence (str, optional): _description_. Defaults to "I paid a  bill when visiting UCLA… it's an expensive hotel, for the 21st century!".
        lang (str, optional): _description_. Defaults to "en_US".

    Returns:
    lower_words, segmented_text, n_syls_in_text, acronym_idxs, special_chars_dict_start, special_chars_dict_end, used_method_syllables
    """

    sentence = sentence.strip()
    # there shouldn't be a space before a special char, they must be glued to words (in english)
    # correct that if it's not the case
    for c in special_chars:
        if c not in [
            '¡',
            '*',
            '"',
        ]:  # these punctuation mark can be at the beginning of a word (spanish exception, and our asterisk mark for target words)
            sentence = sentence.replace(' ' + c, c)

    # handle multiple space by removing empty strings (none) after splitting with space, and rejoining
    sentence = ' '.join(list(filter(None, sentence.split(' '))))

    norm_sent, c = normalize_sentence_numbers(sentence, lang=lang)
    special_chars_dict_start, special_chars_dict_end = extract_special_chars(
        norm_sent, special_chars
    )
    words = remove_special_characters(
        norm_sent,
        lowercase=False,
        chars_to_ignore_regex='[' + '\\'.join(special_chars) + ']',
    ).split(' ')

    # expand_dict is a little dictionnary mapping e.g. Mr -> Mister
    words = [expand_dict[word] if word in expand_dict else word for word in words]
    norm_words = words

    # If all letters are capital (acronym), put '-' between all letters
    # I need to do that before putting in lowercase, that is why I cannot put that in e.g. syllabified_text()
    acronym_idxs = get_acronyms_idxs(words)
    ws = []
    for w_idx, w in enumerate(words):
        if w.isupper():
            w = '-'.join(w)
            words[w_idx] = w
        ws.append(w)
    lower_words = [word.lower() for word in ws]

    syls_texts = []
    used_method_syllables = []
    # for word, phones in zip(lower_words,p):
    for word in lower_words:
        # # This split and rejoin will apply only if there is a dash and then allow to treat separately parts of a word with dash
        # if False:
        if '-' in word:
            syls_texts_parts = []
            used_method_syllables_parts = []
            # for w_part, phones_part in zip(word.split('-'),phones[0].split('-')):
            for w_part in word.split('-'):
                n = n_vowels(w_part, lang=lang)
                # n=n_vowels(phones_part.replace('|','_').split('_'), mode=mode)
                # print("w phones_part:",phones_part,', n_vowels:', n)
                syls_texts_parts.append(
                    syllabified_text(w_part, n, syllables_df, lang=lang)[0]
                )
                used_method_syllables_parts.append(
                    syllabified_text(w_part, n, syllables_df, lang=lang)[1]
                )
            # syls_texts.append('|-'.join(syls_texts_parts))
            syls_texts.append('-'.join(syls_texts_parts))
            used_method_syllables.append('-'.join(used_method_syllables_parts))
        else:
            n = n_vowels(word, lang=lang)
            # n=n_vowels(phones[0].replace('|','_').split('_'), mode=mode)
            # print("w phones:",phones[0],', n_vowels:', n)
            syls_texts.append(syllabified_text(word, n, syllables_df, lang=lang)[0])
            used_method_syllables.append(
                syllabified_text(word, n, syllables_df, lang=lang)[1]
            )

    # put back acronyms in syls_texts, and ignore what was there
    for i in acronym_idxs:
        syls_texts[i] = lower_words[i]
        used_method_syllables[i] = 'acronym'

    case_syls_texts = insert_seps_in_cased_text(
        ' '.join(syls_texts), ' '.join(norm_words), syl_sep=syl_sep
    )
    case_syls_texts_special_chars = case_syls_texts.split(' ')
    case_syls_texts_special_chars = acronyms_hyphen_to_compound(
        case_syls_texts_special_chars, acronym_idxs
    )
    # this adds punctuation and the curly brackets:
    case_syls_texts_special_chars = add_special_chars(
        case_syls_texts_special_chars, special_chars_dict_start, special_chars_dict_end
    )
    segmented_text = ' '.join(case_syls_texts_special_chars)

    # Here I want to remove only punctuation, i.e. all special chars minus curly brackets used forcomound words
    segmented_text = remove_special_characters(
        sentence=segmented_text,
        lowercase=False,
        chars_to_ignore_regex='[' + '\\'.join(set(special_chars) - {"}", "{"}) + ']',
    )

    n_syls_in_text = [len(word.split(syl_sep)) for word in syls_texts]

    return (
        sentence,
        lower_words,
        segmented_text,
        n_syls_in_text,
        acronym_idxs,
        special_chars_dict_start,
        special_chars_dict_end,
        used_method_syllables,
    )


# sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una empresa emergente de 30000 dólares en etapa inicial, ¡luego con el CTO!"
# sentence="A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una start-up de 30000 dólares en etapa inicial, ¡luego con el CTO!"
# sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
def prefill_for_sentence(
    sentence="I paid a $3000 bill when visiting UCLA… it's an expensive hotel, for the 21st century!",
    # sentence="At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!",
    syllables_df=pd.read_csv('data/syllables.csv'),
    syl_sep='|',
    special_chars=[',', '?', '.', '!', '¡', ';', ':', '"', '{', '}', '*', '…'],
    lang="en_US",
    mode='CMU',
    word_dict=cmudict_dict,
):  # "CMU" or "MFA_IPA"
    """This function extract information of syllabified texts and phonetics.
    It uses a combination of datasets (CMUdict, data from syllable_data() ) and algorithm (SonoriPy)

    Args:
        sentence (str): a phrase to be processed. It can contain captial letters and punctuation.
        syllables_df (DataFrame): The syllables dataset built from syllables_data()
        syl_sep (str, optional): [description]. Defaults to '|'.

    Returns:
        dict: see structure a the end of the function
    """
    (
        sentence,
        lower_words,
        segmented_text,
        n_syls_in_text,
        acronym_idxs,
        special_chars_dict_start,
        special_chars_dict_end,
        used_method_syllables,
    ) = text_normalization(
        sentence=sentence,
        lang=lang,
        special_chars=special_chars,
        syllables_df=syllables_df,
        syl_sep=syl_sep,
    )

    # p -> phonetics
    # g -> gibberish
    if mode != 'MFA_IPA':
        p = [
            generate_syl_phonetics_alternatives_from_word(word)[0] for word in lower_words
        ]
        # I separate the acronym in letters and lookup cmudict. I take the last, because
        # there is only one alternative except for letter 'A' which has  [['AH0'], ['EY1']]
        for i in acronym_idxs:
            p[i] = [
                '-'.join(
                    ['_'.join(cmudict_dict[w][-1]) for w in lower_words[i].split('-')]
                )
            ]
    else:
        p = [
            generate_syl_phonetics_alternatives_from_word_ipa(
                word,
                word_dict=word_dict,
                lang=lang,
                g2p_model=lang_to_MFA_g2p_models[lang],
            )
            for word in lower_words
        ]

    stress_inconsistencies = []
    for w_i, w in enumerate(p):
        for alt_i, alt in enumerate(w):
            # if there is several primary stress inside a word part, it is a problem. It breaks our assumptions, so we memorize its index to treat it
            # below
            if alt.count('1') > alt.count('-') + 1:
                d = {'w_i': w_i, 'alt_i': alt_i, 'n_alt': len(w)}
                stress_inconsistencies.append(d)

            # Also, if there is neither 0 or 1 in the phonetics, the algorithm was not able to find a consistent way of extracting a stress pattern from different sources:
            # This is for MFA_IPA and extraction of the stress pattern from espeak annotation
            for w_part in alt.split('-'):
                if w_part.count('1') + w_part.count('0') == 0:
                    d = {'w_i': w_i, 'alt_i': alt_i, 'n_alt': len(w)}
                    stress_inconsistencies.append(d)

    stress_inconsistencies = pd.DataFrame(stress_inconsistencies)

    n_alternatives = [len(el) for el in p]
    n_syls_in_p = [[len(alt.split(syl_sep)) for alt in word] for word in p]

    # among alternatives of phonetics (or gibberish), take the first index for which the number of syllables is equal (i.e. difference=0)
    # if there are none, just take the first alternative (index=0)
    alternative_idxs_syls_consistent = []
    # memorize words for which we couldn't find any consistent alternative
    word_idxs_inconsitencies = {}
    word_idxs_inconsitencies['stress'] = []
    word_idxs_inconsitencies['n_syl'] = []
    for i, (syl_g, syl_t) in enumerate(zip(n_syls_in_p, n_syls_in_text)):
        inconsistency = np.abs(np.array(syl_g) - syl_t).tolist()
        if len(stress_inconsistencies) > 0:
            if i in stress_inconsistencies.w_i.tolist():
                # if there is a stress inconsistency in this word, add a one to the corresponding element in the inconsistency vector
                for alt_idx in stress_inconsistencies[
                    stress_inconsistencies.w_i == i
                ].alt_i.tolist():
                    inconsistency[alt_idx] += 1
        # searching a 0 in the consitency vector to select a consistent alternative
        if 0 in inconsistency:
            alternative_idxs_syls_consistent.append(inconsistency.index(0))
        else:
            # I select the first alternative as there is none that is consistent
            alternative_idxs_syls_consistent.append(0)
            # Then I check what inconsistencies were present
            if syl_g[0] != syl_t:
                word_idxs_inconsitencies['n_syl'].append(i)
            if p[i][0].count('1') > p[i][0].count('-') + 1:
                word_idxs_inconsitencies['stress'].append(i)

            for p_part in p[i][0].split('-'):
                # This is to check and detect if there are indeed stress marks in vowels, because the add_stress() can fail in MFA_IPA
                if p_part.count('0') + p_part.count('1') + p_part.count('2') == 0:
                    word_idxs_inconsitencies['stress'].append(i)

    # Keep first alternative.
    try:
        p_0 = ' '.join(
            [el[alternative_idxs_syls_consistent[i]] for i, el in enumerate(p)]
        )
    except IndexError:
        p_0 = ''

    split_phonetics = lambda phonetics: [
        [[s.split('_') for s in sub_w.split('|')] for sub_w in w.split('-')]
        for w in phonetics.split(' ')
    ]

    brace_dict_start = dict(
        filter(lambda el: el[1] in ['{', '}'], special_chars_dict_start.items())
    )
    brace_dict_end = dict(
        filter(lambda el: el[1] in ['{', '}'], special_chars_dict_end.items())
    )

    p_0_special_chars = acronyms_hyphen_to_compound(p_0.split(' '), acronym_idxs)
    p_0_special_chars = add_special_chars(
        p_0_special_chars, brace_dict_start, brace_dict_end
    )
    p_0_special_chars = ' '.join(p_0_special_chars)

    if mode != 'MFA_IPA':
        g_0 = ' '.join(
            [
                '-'.join(
                    [
                        '|'.join(
                            [
                                '_'.join([cmu_to_gibberish[unstress(p)] for p in s])
                                for s in sub_w
                            ]
                        )
                        for sub_w in w
                    ]
                )
                for w in split_phonetics(p_0)
            ]
        )
        g_0_special_chars = acronyms_hyphen_to_compound(g_0.split(' '), acronym_idxs)
        g_0_special_chars = add_special_chars(
            g_0_special_chars, brace_dict_start, brace_dict_end
        )
        g_0_special_chars = ' '.join(g_0_special_chars)

    # processing on the raw text to add curly brackets around compound words
    special_chars_dict_start_raw, special_chars_dict_end_raw = extract_special_chars(
        sentence, special_chars
    )
    sent_ = acronyms_to_compound(
        remove_special_characters(
            sentence=sentence,
            lowercase=False,
            chars_to_ignore_regex='[' + '\\'.join(special_chars) + ']',
        ).split(' '),
        get_acronyms_idxs(
            remove_special_characters(sentence, lowercase=False).split(' ')
        ),
    )
    sent_brackets = ' '.join(
        add_special_chars(sent_, special_chars_dict_start_raw, special_chars_dict_end_raw)
    )

    if mode != 'MFA_IPA':
        record = {
            'text': sent_brackets,
            'phonetics': p_0_special_chars,
            'pronounciation_guide_hr': g_0_special_chars.replace('_', ''),
            'segmented_text': segmented_text,
            'n_syl_mismatches': word_idxs_inconsitencies['n_syl'],
            'n_stress_inconsistencies': word_idxs_inconsitencies['stress'],
            'used_method_for_syl_text': used_method_syllables,
            'phonetics_alt': p,
            'n_alternatives': n_alternatives,
        }
    else:
        record = {
            'text': sent_brackets,
            'phonetics': p_0_special_chars,
            'segmented_text': segmented_text,
            'n_syl_mismatches': word_idxs_inconsitencies['n_syl'],
            'n_stress_inconsistencies': word_idxs_inconsitencies['stress'],
            'used_method_for_syl_text': used_method_syllables,
            'phonetics_alt': p,
            'n_alternatives': n_alternatives,
        }
    return record


# If I want to test multithreading
# records=[]
# error_records=[]
# print("n sentences", len(sentences))

# def try_my_operation(s):
#     try:
#         record=prefill_for_sentence(s.replace('{','').replace('}',''), syllables_dfs[lang], syl_sep=syl_sep, mode=mode, lang=lang, word_dict=word_dict)
#         records.append(record)
#         return record
#     except:
#         print('Error with sentence: '+s)
#         err=internal_error()
#         err["sentence"]=s
#         error_records.append(err)
#         return err


# import concurrent
# executor = concurrent.futures.ProcessPoolExecutor(10)
# futures = [executor.submit(try_my_operation, s) for i,s in tqdm(enumerate(sentences))]
# concurrent.futures.wait(futures)

# records = [future.result() for future in concurrent.futures.as_completed(futures)]


from src.code_utils import internal_error
from datetime import datetime


def prefill_content(sentences, syl_sep='|', lang='en_US', mode='CMU'):
    """This function extract information of syllabified texts and phonetics using prefill_for_sentence on a list of sentences.
    The result is saved in a DataFrame.

    Args:
        sentences ([type]): list of sentences (can contain special characters and capital letters)
        syl_sep (str, optional): [description]. Defaults to '|'.

    Returns:
        [type]: [description]
    """

    if mode == "MFA_IPA":
        word_dict = get_augmented_mfa_dict(lang)
    else:
        word_dict = cmudict_dict

    records = []
    error_records = []
    print("n sentences", len(sentences))
    for i, s in tqdm(enumerate(sentences)):
        try:
            record = prefill_for_sentence(
                s.replace('{', '').replace('}', ''),
                syllables_dfs[lang],
                syl_sep=syl_sep,
                mode=mode,
                lang=lang,
                word_dict=word_dict,
            )
        except:
            print('Error with sentence: ' + s)
            err = internal_error()
            err["sentence"] = s
            error_records.append(err)
        records.append(record)

    df_errors = pd.DataFrame.from_records(error_records)
    # now=datetime.now()
    # date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
    # df_errors.to_csv('prefill_content_errors_'+date_time+'.csv')

    df = pd.DataFrame.from_records(records)
    return df, df_errors


def generate_prefill_csv(  # path='data/phrases_speaking_activities.txt',
    path='data/phrases_dynamoDB.txt',
    syl_sep='|'  # ,
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
    sentences = pd.read_csv(path, sep='/', header=None)
    df, df_errors = prefill_content(sentences.iloc[:, 0].tolist(), syl_sep=syl_sep)
    df.text = sentences

    print('syl mismatches')
    df.loc[df.apply(lambda r: bool(len(r.n_syl_mismatches)), axis=1)]
    print('stress inconsistencies')
    df.loc[df.apply(lambda r: bool(len(r.n_stress_inconsistencies)), axis=1)]

    return df


def word_stress_from_cmu(phonetics=['K', 'AA1', 'F', 'IY0']):
    # cmu vowels end by a number : 0, 1 or 2.   0= no stress, 1 = primary stress, 2 = secondary stress
    # consonants do not end by a number
    # here I return a list that is one if primary stressed and else 0
    return [1 if p[-1] == str(1) else 0 for p in phonetics if p[-1] in str([0, 1, 2])]


print_memory_usage('RAM - text_processing after all function declarations')


def use_tests():
    # from src.text_processing import *
    prefill_for_sentence(sentence)

    sentence = "A las 22 en punto, tengo una *reunión* con el CEO, Indya, y un ingeniero de una empresa emergente de 30000 dólares en etapa inicial, ¡luego con el CTO!"

    # sentence="A 22 heures, j'ai rendez-vous avec le CEO, Indya, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    sentence = "A 22 heures, j'ai rendez-vous avec le CEO, et un ingénieur d'une start-up à 300 k dollars, puis avec le CTO !"
    sentence = "At 22 o'clock, I have a *meeting* with the CEO, Indya, and an engineer of a 300 k dollars early-stage start-up, then with the CTO!"
    # sentence="twenty"
    # lang="fr_FR"

    sentence = "*My* email campaigns have a better opening rate."
    sentence = "If you don't mind, I'll start."
    sentence = "Great, I'll be grateful to have your advice on *it*."
    sentence = "Great! I'll send you an invite for *10* AM."
    sentence = "Mmh, it might be true then. Do you *know* what happened?"
    sentence = "Amazing, would next Thursday the 16th at 2 PM work for you?"
    sentence = "And it did. I'm sure they'll take the deal you offered."

    sentence = "I'm at UCLA too, *actually*."

    sentence = "I bumped into your sister at the mall"

    sentence = "solar powered"
    sentence = "started"
    sentence = "into"
    sentence = "client"
    sentence = "circumstancial"
    sentence = "our"
    sentence = "every"
    sentence = "lawyer"
    sentence = "technically"
    sentence = "temperature"
    sentence = "idea"
    sentence = "hypothetically"
    sentence = "pie"
    sentence = "indya"
    sentence = "false"

    lang = "en_GB"

    mfa_gb = get_augmented_mfa_dict("en_GB")
    mfa_us = get_augmented_mfa_dict("en_US")

    syllables_df_gb = syllables_dfs["en_GB"]
    syllables_df_us = syllables_dfs["en_US"]

    # r=prefill_for_sentence(sentence=sentence)
    sentence = "nut"
    r_gb = prefill_for_sentence(
        sentence=sentence,
        syllables_df=syllables_df_gb,
        lang="en_GB",
        mode='MFA_IPA',
        word_dict=mfa_gb,
    )  # "CMU" or "MFA_IPA"
    r_us = prefill_for_sentence(
        sentence=sentence,
        syllables_df=syllables_df_us,
        lang="en_US",
        mode='MFA_IPA',
        word_dict=mfa_us,
    )  # "CMU" or "MFA_IPA"
    r_gb
    r_us

    sentence = "circumstancial"
    lang = "en_US"
    syllables_df = syllables_dfs[lang]
    # r=prefill_for_sentence(sentence=sentence)
    r = prefill_for_sentence(
        sentence=sentence,
        syllables_df=syllables_df,
        lang=lang,
        mode='CMU',
        word_dict=cmudict_dict,
    )  # "CMU" or "MFA_IPA"

    db = pd.read_csv('data/query_results-2023-02-21_102331.csv')
    sentences = db.words.tolist()
    df, df_errors = prefill_content(
        sentences, lang='en_US', mode='CMU', output_errors=True
    )
    df.to_csv('prefill_export_2023-02-21_CMU_en_US.csv')
    df, df_errors = prefill_content(
        sentences, lang='en_US', mode='MFA_IPA', output_errors=True
    )
    df.to_csv('prefill_export_2023-02-21_MFA_IPA_en_US.csv')
    df, df_errors = prefill_content(
        sentences, lang='en_GB', mode='MFA_IPA', output_errors=True
    )
    df.to_csv('prefill_export_2023-02-21_MFA_IPA_en_GB.csv')

    df_MFA_IPA_US = pd.read_csv('prefill_export_2023-02-21_MFA_IPA_en_US.csv')
    df_MFA_IPA_GB = pd.read_csv('prefill_export_2023-02-21_MFA_IPA_en_GB.csv')
    df_CMU = pd.read_csv('prefill_export_2023-02-21_CMU_en_US.csv')

    df_MFA_IPA_US

    df_compare = pd.DataFrame()
    df_compare['db'] = db[df_CMU.phonetics != db.phonetics].phonetics
    df_compare['prefill'] = df_CMU[df_CMU.phonetics != db.phonetics].phonetics

    # import difflib
    # diff_string = lambda case_a, case_b : [li for li in difflib.ndiff(case_a, case_b) if li[0] != ' ']
    # df_compare.apply(lambda r: diff_string(r.db, r.prefill), axis=1)

    df_compare.apply(
        lambda r: [
            el for el in zip(r.db.split(' '), r.prefill.split(' ')) if el[0] != el[1]
        ],
        axis=1,
    )

    diff_iz = df_compare.apply(
        lambda r: [
            el
            for el in zip(r.db.split(' '), r.prefill.split(' '))
            if (el[0] != el[1] and el[1].endswith("_IH0_Z"))
        ],
        axis=1,
    )
    diff_non_iz = df_compare.apply(
        lambda r: [
            el
            for el in zip(r.db.split(' '), r.prefill.split(' '))
            if el[0] != el[1] and (not el[1].endswith("_IH0_Z"))
        ],
        axis=1,
    )

    diff_non_iz = diff_non_iz[diff_non_iz.apply(lambda r: len(r)) > 0]
    diff_iz = diff_iz[diff_iz.apply(lambda r: len(r)) > 0]

    db.loc[diff_non_iz.index, :]
    iz_modifs = db.loc[diff_iz.index, :]

    iz_modifs['modification'] = diff_iz

    df[df.apply(lambda r: len(r.n_stress_inconsistencies), axis=1) > 0]
    df[df.apply(lambda r: len(r.n_syl_mismatches), axis=1) > 0]

    df[df.apply(lambda r: len(r.n_syl_mismatches), axis=1) > 0].iloc[0].segmented_text
    df[df.apply(lambda r: len(r.n_syl_mismatches), axis=1) > 0].iloc[
        0
    ].used_method_for_syl_text

    from src.label_data_processing import actor_recordings

    df_phrases = actor_recordings()
    df_phrases = df_phrases.loc[df_phrases.phrase_id.drop_duplicates().index]
    df_phrases = df_phrases.reset_index(drop=True)
    sentences = df_phrases.text.tolist()
    df, df_errors = prefill_content(sentences)

    df['phrase_id'] = df_phrases['phrase_id']

    print('syl mismatches')
    df.loc[df.apply(lambda r: bool(len(r.n_syl_mismatches)), axis=1)]
    print('stress inconsistencies')
    df.loc[df.apply(lambda r: bool(len(r.n_stress_inconsistencies)), axis=1)]

    df_bkp = pd.read_csv('prefill_test_phrases_export')

    df_bkp[df.cmu_phonetics != df_bkp.cmu_phonetics]
    df[df.cmu_phonetics != df_bkp.cmu_phonetics]

    df = generate_prefill_csv('data/export_phrases.txt')
    df.to_csv('prefill_test_phrases_export_new')

    words = [
        'comfortable',
        'table',
        'professional',
        'analysis',
        'temperature',
        'personal',
        'government',
    ]
    sentences = words + [
        'yesterday morning',
        'coffee',
        'seek',
        'take the lead',
        'worked',
        'started a company',
        'think',
        'visited',
    ]
    # from syllabipy.sonoripy import generate_gibberish_alternatives
    # g=generate_gibberish_alternatives(sentences)

    content = pd.read_csv(
        '/mnt/c/Users/noe_t/Downloads/All content minus audio 2023-02-14 - Sheet1.csv'
    )
    sentences = content.words.apply(lambda r: remove_special_characters(r)).tolist()

    # https://www.angmohdan.com/22-words-with-british-and-american-pronunciations-that-may-confuse-you/
    words = "Advertisement,Bald,Clique,Either,Envelope,Esplanade,Leisure,Mobile,Missile,Neither,Niche,Often,Parliament,Privacy,Semi,Schedule,Scone,Stance,Tomato,Vase,Vitamin,Wrath".split(
        ','
    )
    df_us_mfa, df_errors = prefill_content(words, lang="en_US", mode="MFA_IPA")
    df_gb_mfa, df_errors = prefill_content(words, lang="en_GB", mode="MFA_IPA")
    df_us_cmu, df_errors = prefill_content(words)

    # ' '.join(['|'.join(['_'.join([arpabet_to_2_char_ipa[unstress(p).lower()] for p in syl]) for syl in word]) for word in split_phonetics('AE0|D_V_ER1|T_AH0|Z_M_AH0_N_T')])

    from src.pronunciation_dictionaries import arpabet_to_2_char_ipa, mfa_to_display_ipa

    import unicodedata

    # https://stackoverflow.com/questions/61811872/how-do-i-remove-subscript-superscript-in-python
    remove_superscripts = lambda s: "".join(
        c for c in s if (unicodedata.category(c) not in ["No", "Lo", "Lm"]) or (c == "ː")
    )

    map_phonemes = lambda mapper, formatted_phonetics: ' '.join(
        [
            '|'.join(
                ['_'.join([mapper[unstress(p).lower()] for p in syl]) for syl in word]
            )
            for word in split_phonetics(formatted_phonetics)
        ]
    )

    df_MFA_IPA_US = pd.read_csv('prefill_export_2023-02-21_MFA_IPA_en_US.csv')
    df_MFA_IPA_GB = pd.read_csv('prefill_export_2023-02-21_MFA_IPA_en_GB.csv')

    df_MFA_IPA_US_disp = df_MFA_IPA_US.cmu_phonetics.apply(
        lambda r: map_phonemes(
            mfa_to_display_ipa, r.replace('{', '').replace('}', '').replace('-', ' ')
        )
    )
    df_MFA_IPA_GB_disp = df_MFA_IPA_GB.cmu_phonetics.apply(
        lambda r: map_phonemes(
            mfa_to_display_ipa, r.replace('{', '').replace('}', '').replace('-', ' ')
        )
    )

    df_db_display = pd.DataFrame([db.words, df_MFA_IPA_US_disp, df_MFA_IPA_GB_disp]).T
    df_db_display.replace('_', '')

    df_us_cmu_to_ipa = df_us_cmu.cmu_phonetics.apply(
        lambda r: map_phonemes(arpabet_to_2_char_ipa, r)
    )
    df_us_mfa_disp = df_us_mfa.cmu_phonetics.apply(
        lambda r: map_phonemes(mfa_to_display_ipa, r)
    ).apply(lambda r: remove_superscripts(r))
    df_gb_mfa_disp = df_gb_mfa.cmu_phonetics.apply(
        lambda r: map_phonemes(mfa_to_display_ipa, r)
    ).apply(lambda r: remove_superscripts(r))

    df_all_variations = pd.DataFrame(
        [
            df_us_cmu.text,
            df_us_cmu_to_ipa,
            df_us_mfa.cmu_phonetics,
            df_us_mfa_disp,
            df_gb_mfa.cmu_phonetics,
            df_gb_mfa_disp,
        ]
    ).T
    df_all_variations.columns = [
        'text',
        'cmu_to_ipa',
        'mfa_us',
        'mfa_us_disp',
        'mfa_gb',
        'mfa_gb_disp',
    ]
    df_all_variations.replace('_', '', regex=True).replace('\|', '', regex=True).to_csv(
        'ipa_variations.csv'
    )

    df_us_mfa_disp.apply(lambda r: remove_superscripts(r))

    df, df_errors = prefill_content(sentences)

    df.iloc[:, -5:]

    from label_data_processing import get_data

    a = get_data()
    sentences = a.text.apply(lambda r: remove_special_characters(r)).tolist()
    df, df_errors = prefill_content(sentences)
    df.text = a.text
    df.to_csv('prefill_test.csv')

    df.iloc[:, -3:]
    df.iloc[:, -5:-1]

    sentences = pd.read_csv('data/phrases-for-noe.txt', sep='/', header=None)
    sentences = (
        sentences.iloc[:, 0].apply(lambda r: remove_special_characters(r)).tolist()
    )
    df, df_errors = prefill_content(sentences)

    cmu_words = list(cmudict_dict.keys())
    df, df_errors = prefill_content(cmu_words)

    word = words[0]
    scores_p = [el[-1] for el in SonoriPy(str_to_list_of_char(word), mode='letters')[-1]]

    word = cmudict_dict['comfortable'][0]
    scores_t = [el[-1] for el in SonoriPy(word)[-1]]

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import xticks

    plt.plot(scores_t)
    plt.savefig('sonoripy_comfortable.png')
    xticks(np.arange(len(word)), word)

    word = 'enjoyed'
    syls_text = syllables_df[syllables_df.normalized_text == word].syllables.values[0]
    print(syls_text)

    df = generate_prefill_csv()
    df[df.n_syl_mismatch > 0][
        ['syllable_parts', 'pronounciation_guide_hr', 'used_method_for_syl_text']
    ]

    # df[df.n_syl_mismatch>0][['syllable_parts', 'pronounciation_guide_hr','used_method_for_syl_text']]
    df[df.n_syl_mismatches.apply(lambda r: np.sum(r)) > 0].n_syl_mismatches
    df[df.n_syl_mismatches.apply(lambda r: np.sum(r)) > 0]

    df.n_alternatives.max()

    df.iloc[df.n_alternatives.argmax()]
    df.iloc[96]

    sentence = df.iloc[96].text
    words = remove_special_characters(df.iloc[96].text).split(' ')
    # len(syl_phonetics_alternatives(words))
