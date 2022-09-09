from __future__ import unicode_literals  # for python2 compatibility
# -*- coding: utf-8 -*-
# created at UC Berkeley 2015
# Authors: Christopher Hench & Alex Estes © 2015-2019

import sys
from syllabipy.util import cleantext
from datetime import datetime
import json
from ordered_set import OrderedSet
import numpy as np

import itertools
import cmudict

cmu_phones=cmudict.phones()

cmudict_dict=cmudict.dict()
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
        approximates,vowels,nasals,fricatives,affricates,stops=[],[],[],[],[],[]
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
        approximates = str_to_list_of_char('')
        nasals = str_to_list_of_char('lmnrw')
        fricatives = str_to_list_of_char('zvsfh')
        affricates = str_to_list_of_char('')
        stops = str_to_list_of_char('bcdgtkpqxhj')
    elif mode=='IPA':
        vowels = str_to_list_of_char('aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ')
        approximates = str_to_list_of_char('')
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
        # the last two have a little AH0 before the consonant, that's why I have to put them in vowels.
        # https://memcauliffe.com/bootstrapping-an-ipa-dictionary-for-english-using-montreal-forced-aligner-20.html
        # maybe a better solution would be to add a vowel before it. It was with a schwa in wikipron

        cats['approximates']=str_to_list_of_char('')
        cats['nasals']=str_to_list_of_char('lmnrw')
        cats['fricatives']=str_to_list_of_char('zvsfhʃ')
        cats['affricates']=str_to_list_of_char('')
        cats['stops']=str_to_list_of_char('bcdgtkpqxhj')

        remaining_phones=[el for el in mfa_phones if el not in sum(cats.values(),[])]
        for k in cats:
            additionals=[el for el in remaining_phones if el[0] in cats[k]]
            cats[k]+=additionals

        # classifying remaining ones https://en.wikipedia.org/wiki/International_Phonetic_Alphabet
        cats['approximates']+=['ɹ','ɥ', 'ɫ']
        cats['fricatives']+=['ʒ','ʝ', 'β', 'θ', 'ç', 'ð','ɣ','ʁ']
        cats['stops']+=['ʔ', 'ɡ', 'ɟ', 'ɾ', 'ɾʲ']
        cats['nasals']+=['ɱ', 'ɲ','ŋ','ʎ']
        cats['affricates']+=['ɟʝ']

        # remaining_phones=[el for el in mfa_phones if el not in sum(cats.values(),[])]
        # print(remaining_phones)

        vowels = cats['vowels']
        approximates = cats['approximates']
        nasals = cats['nasals']
        fricatives = cats['fricatives']
        affricates = cats['affricates']
        stops = cats['stops']
    else:
        print('This mode of categories for sonoripy does not exist')

    d={"approximates":approximates,"vowels":vowels,"nasals":nasals,"fricatives":fricatives,"affricates":affricates,"stops":stops}
    return d




def SonoriPy(word, mode='CMU'):
    '''
    This program syllabifies words based on the Sonority Sequencing Principle (SSP)

    >>> SonoriPy("justification")
    ['jus', 'ti', 'fi', 'ca', 'tion']
    '''

    def no_syll_no_vowel(ss):
        '''
        cannot be a syllable without a vowel
        '''
        nss = []
        front = []
        for i, syll in enumerate(ss):
            # if following syllable doesn't have vowel,
            # add it to the current one
            if not any(char.lower() in vowels for char in syll):
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


    # SONORITY HIERARCHY for IPA. I am not using it, so I put False instead of a flag "IPA"
    if False:
        vowels = str_to_list_of_char('aeiouyàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūůÿ')

        additional_vowels='ɝə'
        vowels+=str_to_list_of_char(additional_vowels)

        # categories can be collapsed into more general groups
        vowelcount = 0  # if vowel count is 1, syllable is automatically 1
        sylset = []  # to collect letters and corresponding values
        for letter in word.strip(".:;?!)('ˈ" + '"'):
            if letter.lower() in 'aɔʌã':
                sylset.append((letter, 9))
                vowelcount += 1  # to check for monosyllabic words
            elif letter.lower() in 'eéẽɛøoõ'+additional_vowels:
                sylset.append((letter, 8))
                vowelcount += 1  # to check for monosyllabic words
            elif letter.lower() in 'iu':
                sylset.append((letter, 7))
                vowelcount += 1  # to check for monosyllabic words
            elif letter.lower() in 'jwɥh':
                sylset.append((letter, 6))
            elif letter.lower() in 'rl':
                sylset.append((letter, 5))
            elif letter.lower() in 'mn':
                sylset.append((letter, 4))
            elif letter.lower() in 'zvðʒ':
                sylset.append((letter, 3))
            elif letter.lower() in 'sfθʃ':
                sylset.append((letter, 2))
            elif letter.lower() in 'bdg':
                sylset.append((letter, 1))
            elif letter.lower() in 'ptkx':
                sylset.append((letter, 0))
            else:
                sylset.append((letter, 0))

    else:
        d=define_categories(mode=mode)
        approximates,vowels,nasals,fricatives,affricates,stops=d['approximates'],d['vowels'],d['nasals'],d['fricatives'],d['affricates'],d['stops']

        if mode=='CMU':
        # if False:
            # processing: insert a consonant for vowels like "OW", "ER", etc. so that e.g. "going" can be in two syllables
            vowel_ending_in_consonant=['w','y','r']
            idx_to_insert_liquid=[]
            for i,el in enumerate(word):
                # print(el)
                # print(el[1])
                if el.lower() in vowels and el[1].lower() in vowel_ending_in_consonant:
                    idx_to_insert_liquid.append(i)
            for idx in idx_to_insert_liquid[::-1]:
                # insert liquid, e.g. 'r', it does not really matter that it is the true one. Because it will just 
                # be replaced by a score afterwards
                # TODO: PROBLEM: I need to remove it afterwards, because I actually use the output of this to get syllabified phonetics...
                word=word[:idx+1] + ['R'] + word[idx+1:]
            
            # compute the resulting places of inserted Rs
            R_indices=[el+i+1 for i,el in enumerate(idx_to_insert_liquid)]

        # assign numerical values to phonemes (characters)
        vowelcount = 0  # if vowel count is 1, syllable is automatically 1
        sylset = []  # to collect letters and corresponding values
        for letter in word:
            if letter.lower() in vowels:
                sylset.append((letter, 5))
                vowelcount += 1
            elif letter.lower() in approximates:
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

    if mode=="CMU":
        # Here we remove back the "R" liquids added for syllable separation purpose 
        # We stored R indices in original list, here we have to build a new list of syllable list
        # by excluding R indices (that are at the word level without syllable segmentation )
        final_sylset_new=[]
        idx=0
        for syl in final_sylset:
            temp_syl=[]
            for phone in syl:
                if idx not in R_indices:
                    temp_syl.append(phone)
                idx+=1
            final_sylset_new.append(temp_syl)
        final_sylset=final_sylset_new

    return (final_sylset), sylset


if False:
    
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

    def generate_gibberish_from_words(words, indxs):
        sylwords=[]
        word_gibberishes=[]
        for i,word in enumerate(words):
            phonetics=cmudict_dict[word][int(indxs[i])]
            sylwords.append(SonoriPy(phonetics))
            gibberish=[[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in l] for l in SonoriPy(phonetics)]
            word_gibberishes.append(gibberish)
        word_gibberishes=['-'.join([''.join(el) for el in gibberish]) for gibberish in word_gibberishes]
        return word_gibberishes

    def gibberish_alternatives(words):
        lens=[]
        for i,word in enumerate(words):
            lens.append(len(cmudict_dict[word]))
        lists_indxs=[list(np.arange(el)) for el in lens]
        alternative_combinations=list(itertools.product(*lists_indxs))
        alternative_gibberishes=[]
        for indxs in alternative_combinations:
            alternative_gibberishes.append(generate_gibberish_from_words(words, indxs))
        
        return [' '.join(el) for el in alternative_gibberishes]

    # THIS IS the general function
    def generate_gibberish_alternatives(sentences):
        gibberishes=[]
        for sent in sentences:
            words=sent.split(' ')
            word_gibberishes=gibberish_alternatives(words)
            # making it unique (duplicates come from the vowel with different stresses that we do not care about)
            word_gibberishes=list(OrderedSet(word_gibberishes))
            gibberishes.append(word_gibberishes)
        return gibberishes

    def generate_gibberishes(sentences):
        gibberishes=[]
        for sent in sentences:
            words=sent.split(' ')
            word_gibberishes=generate_gibberish_from_words(words)
            gibberishes.append(word_gibberishes)
        return [' '.join(g) for g in gibberishes]

    # def generate_syllables():
    #     words=['professional', 'analysis', 'temperature', 'personal', 'government']
    #     sentences=['yesterday morning','coffee','seek','take the lead','worked','started a company','think','visited']+words
    #     for sent in sentences:
    #         words=sent.split(' ')
    #         sylwords=[]
    #         # word_gibberishes=[]
    #         for word in words:
    #             print(word)
    #             sylwords.append(SonoriPy(word))
    #     return sylwords

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
