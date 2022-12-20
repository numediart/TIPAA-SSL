import cmudict
import pandas as pd
from glob import glob
import json

from io import StringIO
import subprocess
def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        item = d[key]
        # Check if in the inverted dict the key exists
        if item not in inverse: 
            # If not create a new list
            inverse[item] = [key] 
        else: 
            inverse[item].append(key) 
    return inverse

# standard from FB http://fbdevwiki.com/wiki/Locales
lang_to_MFA_g2p_models={'en_GB':'english_uk_mfa',
'en_US':'english_us_mfa',
'fr_FR':'french_mfa',
'es_ES':'spanish_spain_mfa',
'es_LA':'spanish_latin_america_mfa'
}

def get_mfa_df(path='data/english_us_mfa.dict'):
    df=pd.read_csv(path, sep='\t', header=None, encoding='utf8').dropna()
    df.columns=['text', 'ipa']
    df=df[~df.text.str.contains(']')]
    df=df[~df.text.str.contains('<')]    

    ipa=df.apply(lambda r: r.ipa.split(' '), axis=1)
    df['ipa']=ipa

    # df.index=df.text
    return df

def mfa_g2p(word, model="english_us_mfa"):
    #  use process substitution to avoid writing a text file for input and output
    # https://www.gnu.org/software/bash/manual/bash.html#Process-Substitution
    # <code>doit <(echo "hello") >(cat)  
    # e.g. mfa g2p french_mfa <(echo "salut")  >(cat)

    cmd="mfa g2p "+model+" "+'<(echo "'+word+'")'+" >(cat)"
    cmd_list=['bash', '-c',cmd]
    p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
    out, err = p.communicate()

    # the lines not being "INFO" messages correspond to the output that would have been written to the file
    data='\n'.join([el for el in out.decode('utf8').split('\n') if not 'INFO' in el])
    print("mfa g2p for ", word, ' :',data)
    df = pd.read_csv(StringIO(data), sep="\t", header=None, encoding='utf8')
    df.columns=['text', 'ipa']
    ipa=df.apply(lambda r: r.ipa.split(' '), axis=1)
    df['ipa']=ipa

    df['ipa']=df.apply(lambda r: [r.ipa], axis=1)
    ipa_dict=df.groupby(['text']).sum().to_dict()['ipa']

    return ipa_dict

def get_mfa_dict(path='data/spanish_spain_mfa.dict'):
    df=get_mfa_df(path)
    df['ipa']=df.apply(lambda r: [r.ipa], axis=1)
    ipa_dict=df.groupby(['text']).sum().to_dict()['ipa']
    return ipa_dict

def generate_acronym_letter_dicts():
    
    def acro_dict(letters, lang):
        acronym_dict={}
        for l in letters[lang]:
            try:
                acronym_dict[l]=mfa_g2p(l, lang_to_MFA_g2p_models[lang])[l]
            except:
                acronym_dict[l]=[]
        return acronym_dict
    
    letters={}

    # spanish letters https://en.wikipedia.org/wiki/Spanish_orthography: E A O S R N I D L C T U M P B G V Y Q H F Z J Ñ X W K
    lang="es_ES"
    letters[lang]='E A O S R N I D L C T U M P B G V Y Q H F Z J Ñ X W K'.lower().split(' ')
    
    acronym_dict=acro_dict(letters, lang)
    acronym_dict['h']=mfa_g2p('ache', lang_to_MFA_g2p_models[lang])['ache']
    acronym_dict['z']=mfa_g2p('zeta', lang_to_MFA_g2p_models[lang])['zeta']
    acronym_dict['w']=mfa_g2p('uvedoble', lang_to_MFA_g2p_models[lang])['uvedoble']
    
    acronym_dict_long={k:[max(lst, key=len)] for k,lst in acronym_dict.items()}
    acronym_dict_long['g']= ['g', 'e']

    acronym_dict_LA=acronym_dict
    acronym_dict_LA['z']=mfa_g2p('seta', lang_to_MFA_g2p_models[lang])['seta']
    acronym_dict_long_LA={k:[max(lst, key=len)] for k,lst in acronym_dict_LA.items()}
    acronym_dict_long_LA['g']= ['g', 'e']

    with open('data/acronyms_es_ES_mfa.dict','w') as f: f.write(json.dumps(acronym_dict_long))
    with open('data/acronyms_es_LA_mfa.dict','w') as f: f.write(json.dumps(acronym_dict_long_LA))

    # english letters: https://en.wikipedia.org/wiki/English_alphabet
    lang="en_GB"
    letters[lang]="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".lower().split(' ')
    acronym_dict=acro_dict(letters, lang)
    acronym_dict_long={k:[max(lst, key=len)] for k,lst in acronym_dict.items()}
    # as the first phoneme in "age"
    # acronym_dict_long['a']=[mfa_g2p('age', lang_to_MFA_g2p_models[lang])['age'][0][0]]
    acronym_dict_long['a']=[['ej']]
    acronym_dict_long['e']=[['iː']]
    # as the word "are", US Version //!\\
    acronym_dict_long['r']=[mfa_g2p('are', lang_to_MFA_g2p_models['en_US'])['are'][0]]
    acronym_dict_long['z']=mfa_g2p('zee', lang_to_MFA_g2p_models[lang])['zee']
    
    acronym_dict_long_US=acronym_dict_long
    acronym_dict_long_US['o']=mfa_g2p('o', lang_to_MFA_g2p_models['en_US'])['o']

    with open('data/acronyms_en_GB_mfa.dict','w') as f: f.write(json.dumps(acronym_dict_long))
    with open('data/acronyms_en_US_mfa.dict','w') as f: f.write(json.dumps(acronym_dict_long_US))
    
    lang="fr_FR"
    letters[lang]="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".lower().split(' ')
    acronym_dict=acro_dict(letters, lang)
    acronym_dict['h']=mfa_g2p('ache', lang_to_MFA_g2p_models[lang])['ache']
    acronym_dict_long={k:[max(lst, key=len)] for k,lst in acronym_dict.items()}
    acronym_dict_long['e']=[['ə']]
    acronym_dict_long['n']=[['ɛ', 'n']]
    acronym_dict_long['r']=[['ɛ', 'ʁ']]
    acronym_dict_long['t']=[['t', 'e']]

    acronym_dict_long['w']=[max(mfa_g2p('doublevé', lang_to_MFA_g2p_models[lang])['doublevé'], key=len)]
    acronym_dict_long['x']=[['i', 'k', 's']]
    acronym_dict_long['y']=[max(mfa_g2p('igrec', lang_to_MFA_g2p_models[lang])['igrec'], key=len)]
    acronym_dict_long['z']=[['z', 'ɛ', 'd']]

    with open('data/acronyms_fr_FR_mfa.dict','w') as f: f.write(json.dumps(acronym_dict_long))



def get_augmented_mfa_dict(lang='es_ES'):
    d=get_mfa_dict(path='data/'+lang_to_MFA_g2p_models[lang]+'.dict')
    with open('data/acronyms_'+lang+'_mfa.dict','r') as f: acronyms=json.loads(f.read())
    for k in acronyms: d[k]=acronyms[k]
    return d



mfa_dicts={lang:get_augmented_mfa_dict(lang) for lang in lang_to_MFA_g2p_models}

# mfa_dicts['en_US']
# len(mfa_dicts['en_US'])


def build_mfa_phone_set():
    dict_paths=glob('data/*mfa.dict')
    dfs=[get_mfa_df(p) for p in dict_paths]
    df=pd.concat(dfs)
    phones=sorted(list(set().union(*df.ipa.apply(lambda r: set(r)).tolist())))

    # Writing to sample.json
    with open("data/mfa_phones.json", "w") as outfile: outfile.write(json.dumps(phones))

    return phones

def get_augmented_cmudict():
    cmudict_dict=cmudict.dict()

    inconsistent_word_stresses={}
    for k in cmudict_dict:
        for alt in cmudict_dict[k]:
            if '-' not in k:
                if sum(['1' in p for p in alt])>1:
                    inconsistent_word_stresses[k]=alt
    # len(inconsistent_word_stresses)
    # There are too much to be corrected, and too few to really care, it's less than 1% nd most probalably unfrequend words...

    # len([k for k in inconsistent_word_stresses if k[:2]=='re'])
    # {k:inconsistent_word_stresses[k] for k in inconsistent_word_stresses if k[:2]!='re'}

    corrections={
        'areas':[['EH1','R','IH0','AH0','Z']],
        'live':[['L', 'IH1', 'V']],
        'drawing':[['D','R','AO1','W','IH0','NG']],
        'drawings':[['D','R','AO1','W','IH0','NG','Z']],
        'ph':[['P', 'IY1', 'EY2', 'CH']],
        'pH':[['P', 'IY1', 'EY2', 'CH']],
        'laboratory':[['L', 'AE1', 'B', 'AH0', 'R', 'AH0', 'T', 'AO2', 'R', 'IY0']],
        'fourteen':[['F', 'AO2', 'R', 'T', 'IY1', 'N']],
        'thirteen':[['TH', 'ER2', 'T', 'IY1', 'N']],
        'fifteen':[['F', 'IH2', 'F', 'T', 'IY1', 'N']],
        'sixteen':[['S', 'IH2', 'K', 'S', 'T', 'IY1', 'N']],
        'seventeen':[['S', 'EH2', 'V', 'AH0', 'N', 'T', 'IY1', 'N']],
        'eighteen':[['EY0', 'T', 'IY1', 'N'], ['EY2', 'T', 'IY1', 'N']],
        'nineteen':[['N', 'AY2', 'N', 'T', 'IY1', 'N']],
        'engineer':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R']],
        'engineers':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R', 'Z']],
        'engineering':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R', 'IH0', 'NG']],
        'downstairs':[['D', 'AW0', 'N', 'S', 'T', 'EH1', 'R', 'Z']],
        'trainee':[['T', 'R', 'EY0', 'N', 'IY1']],
        'outside':[['AW0', 'T', 'S', 'AY1', 'D']],
        'trespasser':[['T', 'R', 'EH0', 'S', 'P', 'AE1', 'S', 'ER0']],
        'trespassers':[['T', 'R', 'EH0', 'S', 'P', 'AE1', 'S', 'ER0', 'Z']],
        'outdoors':[['AW1', 'T', 'D', 'AO2', 'R', 'Z']]
	}
    for k in corrections:
        cmudict_dict[k]=corrections[k]
    return cmudict_dict



cmudict_dict=get_augmented_cmudict()


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
arpabet_to_1_char_ipa = {
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

cmu_phones_info=cmudict.phones()
cmu_phones=set([el[0] for el in cmu_phones_info])
cmu_vowels=set([p[0] for p in cmu_phones_info if p[1][0]=='vowel'])
cmu_consonants=set([p[0] for p in cmu_phones_info if p[1][0]!='vowel'])

cmu_stressed_vowels=set(cmudict.symbols())-cmu_phones

# CMU is a subset of arpabet
cmu_1_char={}
cmu_1_char_to_gibberish={}
for p in cmu_phones:
    cmu_1_char[p]=arpabet_to_1_char_ipa[p.lower()]
    cmu_1_char_to_gibberish[cmu_1_char[p]]=cmu_to_gibberish[p]


cmu_alphabet = [el[0] for el in cmudict.phones()]
with open('data/mfa_phones.json', 'r') as openfile: ipa_alphabet = json.load(openfile)

# csv built from tables in https://en.wikipedia.org/wiki/ARPABET  and adapted by looking at some transcriptions in cmudict of the examples in a spreadsheet
cmu_reducer_df=pd.read_csv('data/cmu_reducer.csv')
cmu_reducer=dict(zip(cmu_reducer_df.IPA, cmu_reducer_df.CMU))

# doc on mfa phone set, an opinionated ipa phone set: https://mfa-models.readthedocs.io/en/latest/mfa_phone_set.html
# UK US english
# consonants
cmu_reducer['ɲ']=cmu_reducer['n']
cmu_reducer['c']=cmu_reducer['k']
cmu_reducer['ʎ']=cmu_reducer['l']
cmu_reducer['ɟ']=cmu_reducer['ɡ']
cmu_reducer['ç']=cmu_reducer['h']

cmu_reducer['pʰ']=cmu_reducer['p']
cmu_reducer['tʰ']=cmu_reducer['t']
cmu_reducer['cʰ']=cmu_reducer['k']
cmu_reducer['kʰ']=cmu_reducer['k']

cmu_reducer['ɫ̩']='L'
cmu_reducer['ɫ']='L'

cmu_reducer['ɱ']=cmu_reducer['m']
cmu_reducer['t̪']=cmu_reducer['t']

cmu_reducer['ʔ']='T'  # "butter", "uh-oh" sound. UK accent has instances of "T" pronounced like that


# vowels
cmu_reducer['aw']=cmu_reducer['aʊ']
cmu_reducer['aj']=cmu_reducer['aɪ']
cmu_reducer['ow']=cmu_reducer['oʊ']
cmu_reducer['ej']=cmu_reducer['eɪ']
cmu_reducer['ɔj']=cmu_reducer['ɔɪ']

cmu_reducer['əw']=cmu_reducer['ow']

cmu_reducer['ɐ']='AA'


# nigerian
cmu_reducer['a']='AA'

# https://en.wikipedia.org/wiki/Open-mid_central_unrounded_vowel
cmu_reducer['ɜ']=cmu_reducer['ə']




long_vowels=[el for el in ipa_alphabet if el[-1]=='ː']
# cmu does not have a symbol to differentiate long and short vowels with the same acoustics
for lv in long_vowels: cmu_reducer[lv]=cmu_reducer[lv[:-1]]

j_consonants=[el for el in ipa_alphabet if el[-1]=='ʲ']
for jc in j_consonants: cmu_reducer[jc]=cmu_reducer[jc[:-1]]

# print(set(ipa_alphabet)-set(cmu_reducer))
# len(set(ipa_alphabet)-set(cmu_reducer))

# invert_dict(cmu_reducer)

# https://en.wiktionary.org/wiki/%C9%A3
# cmu_reducer['ɣ']=cmu_reducer['g']

# categorize ipa phonemes in vowels and consonants, first thanks to the associations done towards CMU, and then filling the missing ones

ipa_vowels=set()
ipa_consonants=set()
for p in ipa_alphabet:
    if p in cmu_reducer:
        if cmu_reducer[p] in cmu_vowels:
            ipa_vowels.add(p)
        elif cmu_reducer[p] in cmu_consonants:
            ipa_consonants.add(p)

# print(set(ipa_alphabet)-set(cmu_reducer))

ipa_remainings=set(ipa_alphabet)-set(cmu_reducer)

# information on the remainings from https://mfa-models.readthedocs.io/en/latest/mfa_phone_set.html
ipa_vowels_add={'œ', 'ø', 'ɛ̃', 'y', 'o', 'e', 'ɔ̃', 'ɑ̃'}

ipa_consonants_add=ipa_remainings-ipa_vowels_add

ipa_vowels=ipa_vowels.union(ipa_vowels_add)
ipa_consonants=ipa_consonants.union(ipa_consonants_add)

ipa_to_gibberish={k:cmu_to_gibberish[cmu_reducer[k]] for k in cmu_reducer}