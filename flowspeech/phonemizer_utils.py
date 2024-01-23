import os, psutil

print_memory_usage = lambda stage: print(
    stage + ": " + str(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
)
print_memory_usage('RAM - phonemizer_utils start')
import numpy as np
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

from flowspeech.pronunciation_dictionaries import cmudict_dict
from syllabipy.sonoripy import SonoriPy

# https://bootphon.github.io/phonemizer/python_examples.html

# words=list(cmudict_dict.keys())

backend_dict = {
    'en_GB': EspeakBackend('en-gb', with_stress=True),
    'en_US': EspeakBackend('en-us', with_stress=True),
    'fr_FR': EspeakBackend('fr-fr', with_stress=True),
    'es_ES': EspeakBackend('es', with_stress=True),
    'es_LA': EspeakBackend('es', with_stress=True),
}

print_memory_usage('RAM - phonemizer_utils after backend dict')


# EspeakBackend(lang, with_stress=True)


def phonetize(word, lang="en_GB"):
    # initialize the espeak backend for English
    backend = backend_dict[lang]

    # separate phones by a space and ignoring words boundaries
    separator = Separator(phone='_', word="")
    # the strip is weirdly, if you do this without the strip, it can sometimes start with an underscore:   phonetize('e',"fr_FR") -> "_ˈə"  phonetize('y',"fr_FR") -> "i_ɡ_ʁ_ˈɛ_k"
    return backend.phonemize([word], separator=separator, strip=True)[0].strip('_')


# lang="fr-fr"
# lang="es"
def word_to_stressed_syl(word, lang="en_US"):
    """This function uses a combination of phonemizer with espeak backend providing phonetic transcriptions with a stress symbol
    and SonoriPy to syllabify the transcription. It outputs a stress pattern

    Args:
        word (_type_): _description_
        lang (str, optional): language codes for espeak: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md . Defaults to "en".

    Returns:
        (int, int):  stress index, n of syllables. If there is no stress symbol in phonetics, the stress index is set to None
    """
    stress_symbol = "ˈ"
    second_stress_symbol = "ˌ"

    p = phonetize(word, lang=lang)
    syl_p = SonoriPy(
        p.replace(stress_symbol, '').replace(second_stress_symbol, '').split('_'),
        mode="MFA_IPA",
    )[0]

    # there can be no stress in little words like "at"
    if stress_symbol in p:
        p_stress_idx = [stress_symbol in el for el in p.split('_')].index(1)
        n_p_by_syl = [len(syl) for syl in syl_p]
        n_p_by_syl_cumsum = np.cumsum(n_p_by_syl)

        stress_index = 0
        for i, el in enumerate(n_p_by_syl_cumsum):
            if p_stress_idx < el:
                stress_index = i
                break
    else:
        stress_index = None

    return stress_index, len(syl_p)


def words_to_lexicon(words, lang="en"):
    # initialize the espeak backend for English
    backend = EspeakBackend(lang, with_stress=True)

    # separate phones by a space and ignoring words boundaries
    separator = Separator(phone='', word="")
    phonetize_word = lambda word: backend.phonemize(
        [word], separator=separator, strip=True
    )[0]

    phonetize_syllabify_word = lambda word: '|'.join(
        [''.join(syl) for syl in SonoriPy(phonetize_word(word), mode="MFA_IPA")[0]]
    )

    # build the lexicon by phonemizing each word one by one. The backend.phonemize
    # function expect a list as input and outputs a list.
    lexicon = {word: phonetize_syllabify_word(word) for word in words}

    return lexicon


def use_tests():
    words = ['address', 'adult', 'advertisement', 'ballet', 'café']
    words_to_lexicon(words)
    words_to_lexicon(words, lang="en")
