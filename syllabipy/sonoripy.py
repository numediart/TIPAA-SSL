from __future__ import unicode_literals  # for python2 compatibility
# -*- coding: utf-8 -*-
# created at UC Berkeley 2015
# Authors: Christopher Hench & Alex Estes © 2015-2019

import sys
from syllabipy.util import cleantext
from datetime import datetime
import json
import cmudict

cmu_phones=cmudict.phones()
unstress = lambda el: el[:-1] if el[-1] in str([0,1,2]) else el

# Python code to convert string to list character-wise
# https://www.geeksforgeeks.org/python-program-convert-string-list/
def str_to_list_of_char(string):
    list1=[]
    list1[:0]=string
    return list1

# SONORITY HIERARCHY, MODIFY FOR LANGUAGE BELOW
# categories should be collapsed into more general groups

def define_categories(mode='CMU'):
    if mode=='CMU':
        approximants,vowels,nasals,fricatives,affricates,stops=[],[],[],[],[],[]
        for p in cmu_phones:
            if p[-1][0]=='vowel':
                # I add with and without stress so that it works with both converntions
                vowels.append(p[0].lower())
                vowels.append(p[0].lower()+str(0))
                vowels.append(p[0].lower()+str(1))
                vowels.append(p[0].lower()+str(2))
            if p[-1][0]=='nasal' or p[-1][0]=='liquid' or p[-1][0]=='semivowel':
                nasals.append(p[0].lower())
            if p[-1][0]=='fricative':
                fricatives.append(p[0].lower())
            if p[-1][0]=='affricate':
                affricates.append(p[0].lower())
            if p[-1][0]=='stop':
                stops.append(p[0].lower())
    elif mode=='letters':
        vowels = str_to_list_of_char('aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ')
        approximants = str_to_list_of_char('')
        nasals = str_to_list_of_char('lmnrw')
        fricatives = str_to_list_of_char('zvsfh')
        affricates = str_to_list_of_char('')
        stops = str_to_list_of_char('bcdgtkpqxhj')
    elif mode=='IPA':
        vowels = str_to_list_of_char('aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ')
        approximants = str_to_list_of_char('')
        nasals = str_to_list_of_char('lmnrw')
        fricatives = str_to_list_of_char('zvsfhʃ')
        affricates = str_to_list_of_char('')
        stops = str_to_list_of_char('bcdgtkpqxhj')
        
        additional_vowels='ɝəaɔʌãeéẽɛøoõiu'
        vowels+=str_to_list_of_char(additional_vowels)
    elif mode=="MFA_IPA":
        with open('data/mfa_phones.json', 'r') as openfile: mfa_phones = json.load(openfile)
        # https://mfa-models.readthedocs.io/en/refactor/mfa_phone_set.html
        cats={}
        cats['vowels']=str_to_list_of_char('aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ')+str_to_list_of_char('ɝəaɔʌãeéẽɛøoõiu')
        cats['vowels']+=['ɐ', 'ɑ', 'ɑː', 'ɑ̃', 'ɒ', 'ɒː', 'ɚ', 'ɜ', 'ɜː','ʉ', 'ʉː','ʊ','ɪ','ɫ̩','m̩','n̩'] 
        cats['vowels']+=["ᵻ"] # discovered this one for espeak in final -ed when uning en_US.
        # Try :  from src.phonemizer_utils import phonetize;phonetize("started", "en_US")  -> "s_t_ˈɑːɹ_ɾ_ᵻ_d"

        # the last two have a little AH0 before the consonant, that's why I have to put them in vowels.
        # https://memcauliffe.com/bootstrapping-an-ipa-dictionary-for-english-using-montreal-forced-aligner-20.html
        # maybe a better solution would be to add a vowel before it. It was with a schwa in wikipron

        cats['vowels']+=['aʊ','aɪ','eɪ','oʊ','əʊ','ɔɪ']

        
        cats['approximants']=str_to_list_of_char('')
        cats['nasals']=str_to_list_of_char('lmnr')
        cats['fricatives']=str_to_list_of_char('zvsfhʃ')
        cats['affricates']=str_to_list_of_char('')
        cats['stops']=str_to_list_of_char('bcdgtkpqxh')

        # for diphtongs vowel or consonants
        # it's not exactly correct because it will put tʃ in stops instead of affricates, but it does not impact the syllable cutting
        remaining_phones=[el for el in mfa_phones if el not in sum(cats.values(),[])]
        for k in cats:
            additionals=[el for el in remaining_phones if el[0] in cats[k]]
            cats[k]+=additionals

        # classifying remaining ones https://en.wikipedia.org/wiki/International_Phonetic_Alphabet

        # https://en.wikipedia.org/wiki/Approximant
        cats['approximants']+=['ɹ','ɥ', 'ɫ','w','j']
        cats['fricatives']+=['ʒ','ʝ', 'β', 'θ', 'ç', 'ð','ɣ','ʁ']
        cats['stops']+=['ʔ', 'ɡ', 'ɟ', 'ɾ', 'ɾʲ']
        cats['nasals']+=['ɱ', 'ɲ','ŋ','ʎ']
        cats['affricates']+=['ɟʝ']

        # remaining_phones=[el for el in mfa_phones if el not in sum(cats.values(),[])]
        # print(remaining_phones)

        vowels = cats['vowels']
        approximants = cats['approximants']
        nasals = cats['nasals']
        fricatives = cats['fricatives']
        affricates = cats['affricates']
        stops = cats['stops']
    else:
        print('This mode of categories for sonoripy does not exist')

    d={"approximants":approximants,"vowels":vowels,"nasals":nasals,"fricatives":fricatives,"affricates":affricates,"stops":stops}
    return d




def SonoriPy(word, mode='CMU'):
    '''
    This program syllabifies words based on the Sonority Sequencing Principle (SSP)
    The different modes are defined in the "define_categories" function.
    Input can be either a string, either a list of phonemes. 
    
    The use with "letters" is more a toy than anything for english, but it might be helpful for syllabifying text in spanish or italian.

    The principle is however very reliable when applied on actual phonetics (CMU or IPA).

    Note that the mode "MFA_IPA" might work well on letters, because I added as many symbols as possible to be accepted in the categories.
    I also should work well on other IPA standards. So maybe the "IPA" mode is now obsolete.
    For example, using it on phonetics coming from "phonemizer" with e.g. espeak backend should work well with MFA_IPA mode. 
    I use this to find stressed syllables in many languages.

    >>> SonoriPy("justification")
    ['jus', 'ti', 'fi', 'ca', 'tion']
    '''

    def no_syll_no_vowel(ss):
        '''
        cannot be a syllable without a vowel
        '''
        nss = []

        # as the function can be used either on strings or list of phonemes
        if len(ss)>0:
            front = "" if type(ss[0])==str else []
        else:
            front=[]
        for syll in ss:
            # if following syllable doesn't have vowel,
            # add it to the current one
            
            # if the first symbol of a phoneme corresponds to a vowel, I count it as a vowel.
            # I need this for espeak
            # 'start'-> 's_t_ˈɑːɹ_t',   'engineer' -> "ˌɛ_n_dʒ_ɪ_n_ˈɪɹ"
            if not any(((char.lower() in vowels) or (char.lower()[0] in vowels)) for char in syll):
                if len(nss) == 0:
                    front += syll
                else:
                    nss = nss[:-1] + [nss[-1] + syll]
            else:
                if len(nss) == 0:
                    nss.append(front + syll)
                else:
                    nss.append(syll)

        return nss

    d=define_categories(mode=mode)
    approximants,vowels,nasals,fricatives,affricates,stops=d['approximants'],d['vowels'],d['nasals'],d['fricatives'],d['affricates'],d['stops']

    # The original version did not process properly "hiatus" words.
    # The algorithm extract oscilating sonority values (assigned by categories), and basically cut syllables with locals minimas.
    # If there are two consecutive vowels, it wouldn't cut that in separate syllables because there is no local minimum between them.
    # Although there shouldn't be 2 vowels in 1 syllable. So what I do is insert an approximant just after each vowel (wich is the level just below vowels).
    # That way, it I insert a local minimum there, but don't modify the rest of the trend for the rest. After cutting, I remove the inserted approximants to get the final result.

    # processing: insert an approximant after each vowel so that e.g. "going" can be in two syllables. "théorie"
    # this works well in phonetics, but in letters, there are too much exceptions. I have a better accuracy without it
    if mode!='letters':
        idx_to_insert_approximants=[]
        for i,el in enumerate(word):
            # if the first symbol of a phoneme corresponds to a vowel, I count it as a vowel.
            # I need this for espeak
            # 'start'-> 's_t_ˈɑːɹ_t',   'engineer' -> "ˌɛ_n_dʒ_ɪ_n_ˈɪɹ"
            if (el.lower() in vowels) or (el.lower()[0] in vowels):# and el[1].lower() in vowel_ending_in_consonant:
                idx_to_insert_approximants.append(i)
        for idx in idx_to_insert_approximants[::-1]:
            # insert approximant, e.g. 'w', after vowels.
            # I remove it afterwards, because I actually use the output of this to get syllabified phonetics...
            if type(word)==str:
                word=word[:idx+1] + 'W' + word[idx+1:]
            else:
                word=word[:idx+1] + ['W'] + word[idx+1:]
    
        # compute the resulting places of inserted Ws
        W_indices=[el+i+1 for i,el in enumerate(idx_to_insert_approximants)]

    # assign numerical values to phonemes (characters)
    vowelcount = 0  # if vowel count is 1, syllable is automatically 1
    sylset = []  # to collect letters and corresponding values
    for letter in word:
        # if the first symbol of a phoneme corresponds to a vowel, I count it as a vowel.
        # I need this for espeak
        # 'start'-> 's_t_ˈɑːɹ_t',   'engineer' -> "ˌɛ_n_dʒ_ɪ_n_ˈɪɹ"
        if (letter.lower() in vowels) or (letter.lower()[0] in vowels):
            sylset.append((letter, 5))
            vowelcount += 1
        elif letter.lower() in approximants:
            sylset.append((letter, 4))
        elif letter.lower() in nasals:
            sylset.append((letter, 3))
        elif letter.lower() in fricatives:
            sylset.append((letter, 2))
        elif letter.lower() in affricates:
            sylset.append((letter, 1))
        elif letter.lower() in stops:
            sylset.append((letter, 0))
        else:
            sylset.append((letter, 0))

    # SSP syllabification follows
    final_sylset = []
    if vowelcount == 1:  # finalize word immediately if monosyllabic
        final_sylset.append(word)
    if vowelcount != 1:
        syllable = []  # prepare empty syllable to build upon
        for i, tup in enumerate(sylset):
            if i == 0:  # if it's the first letter, append automatically
                syllable.append(tup[0])

            else:
                # add whatever is left at end of word, last letter
                if i == len(sylset) - 1:
                    syllable.append(tup[0])
                    final_sylset.append(syllable)
                    # print(syllable)

                # MAIN ALGORITHM BELOW

                # these cases DO NOT trigger syllable breaks
                elif (i < len(sylset) - 1) and tup[1] < sylset[i + 1][1] and \
                        tup[1] > sylset[i - 1][1]:
                    syllable.append(tup[0])
                elif (i < len(sylset) - 1) and tup[1] > sylset[i + 1][1] and \
                        tup[1] < sylset[i - 1][1]:
                    syllable.append(tup[0])
                elif (i < len(sylset) - 1) and tup[1] > sylset[i + 1][1] and \
                        tup[1] > sylset[i - 1][1]:
                    syllable.append(tup[0])
                elif (i < len(sylset) - 1) and tup[1] > sylset[i + 1][1] and \
                        tup[1] == sylset[i - 1][1]:
                    syllable.append(tup[0])
                elif (i < len(sylset) - 1) and tup[1] == sylset[i + 1][1] and \
                        tup[1] > sylset[i - 1][1]:
                    syllable.append(tup[0])
                elif (i < len(sylset) - 1) and tup[1] < sylset[i + 1][1] and \
                        tup[1] == sylset[i - 1][1]:
                    syllable.append(tup[0])

                # these cases DO trigger syllable break
                # if phoneme value is equal to value of preceding AND following
                # phoneme
                elif (i < len(sylset) - 1) and tup[1] == sylset[i + 1][1] and \
                        tup[1] == sylset[i - 1][1]:
                    syllable.append(tup[0])
                    # append and break syllable BEFORE appending letter at
                    # index in new syllable
                    final_sylset.append(syllable)
                    # print(syllable)
                    syllable = []

                # if phoneme value is less than preceding AND following value
                # (trough)
                elif (i < len(sylset) - 1) and tup[1] < sylset[i + 1][1] and \
                        tup[1] < sylset[i - 1][1]:
                    # append and break syllable BEFORE appending letter at
                    # index in new syllable
                    final_sylset.append(syllable)
                    # print(syllable)
                    syllable = []
                    syllable.append(tup[0])

                # if phoneme value is less than preceding value AND equal to
                # following value
                elif (i < len(sylset) - 1) and tup[1] == sylset[i + 1][1] and \
                        tup[1] < sylset[i - 1][1]:
                    syllable.append(tup[0])
                    # append and break syllable BEFORE appending letter at
                    # index in new syllable
                    final_sylset.append(syllable)
                    # print(syllable)
                    syllable = []

    final_sylset = no_syll_no_vowel(final_sylset)

    if mode!='letters':
        # Here we remove back the "W" approximants added for syllable separation purpose 
        # We stored W indices in original list, here we have to build a new list of syllable list
        # by excluding W indices (that are at the word level without syllable segmentation )
        final_sylset_new=[]
        idx=0
        for syl in final_sylset:
            temp_syl=[]
            for phone in syl:
                if idx not in W_indices:
                    temp_syl.append(phone)
                idx+=1
            final_sylset_new.append(temp_syl)
        
        
        sylset_new=[]
        idx=0
        for phone in sylset:
            if idx not in W_indices:
                sylset_new.append(phone)
            idx+=1
        final_sylset=final_sylset_new
        sylset=sylset_new

    return final_sylset, sylset


# command line usage
if __name__ == '__main__':
    print("\n\nSonoriPy-ing...\n")

    sfile = sys.argv[1]  # input text file to syllabify
    with open(sfile, 'r', encoding='utf-8') as f:
        text = f.read()

    sylls = [SonoriPy(w) for w in cleantext(text).split()]

    toprint = ""
    for word in sylls:
        for syll in word:
            if syll != word[-1]:
                toprint += syll
                toprint += "-"
            else:
                toprint += syll
        toprint += " "

    fmt = '%Y/%m/%d %H:%M:%S'
    date = "SonoriPyed on " + str(datetime.now().strftime(fmt))

    finalwrite = date + "\n\n" + toprint

    with open('SonoriPyed.txt', 'w', encoding='utf-8') as f:
        f.write(finalwrite)

    print("\nResults saved to SonoriPyed.txt\n\n")
