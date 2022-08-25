import cmudict
import pandas as pd
from glob import glob
import json
from g2p_en import G2p
from itertools import groupby

from io import StringIO
import subprocess

# If I use this, it should be en_GB instead of en_UK http://fbdevwiki.com/wiki/Locales
lang_to_MFA_g2p_models={'en_UK':'english_uk_mfa',
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

    corrections={
        'fourteen':[['F', 'AO2', 'R', 'T', 'IY1', 'N']],
        'thirteen':[['TH', 'ER2', 'T', 'IY1', 'N']],
        'fifteen':[['F', 'IH2', 'F', 'T', 'IY1', 'N']],
        'sixteen':[['S', 'IH2', 'K', 'S', 'T', 'IY1', 'N']],
        'seventeen':[['S', 'EH2', 'V', 'AH0', 'N', 'T', 'IY1', 'N']],
        'eighteen':[['EY0', 'T', 'IY1', 'N'], ['EY2', 'T', 'IY1', 'N']],
        'nineteen':[['N', 'AY2', 'N', 'T', 'IY1', 'N']],
        'engineer':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R']],
        'downstairs':[['D', 'AW0', 'N', 'S', 'T', 'EH1', 'R', 'Z']],
        'trainee':[['T', 'R', 'EY0', 'N', 'IY1']],
        'outside':[['AW0', 'T', 'S', 'AY1', 'D']],
        'trespasser':[['T', 'R', 'EH0', 'S', 'P', 'AE1', 'S', 'ER0']],
        'trespassers':[['T', 'R', 'EH0', 'S', 'P', 'AE1', 'S', 'ER0', 'Z']]
	}
    for k in corrections:
        cmudict_dict[k]=corrections[k]
    return cmudict_dict


cmudict_dict=get_augmented_cmudict()

mfa_dicts={lang:get_mfa_dict(path='data/'+lang_to_MFA_g2p_models[lang]+'.dict') for lang in lang_to_MFA_g2p_models}



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
arpabet_to_ipa = {
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
cmu_phones=[el[0] for el in cmu_phones_info]
cmu_vowels=[p[0] for p in cmu_phones_info if p[1][0]=='vowel']
cmu_consonants=[p[0] for p in cmu_phones_info if p[1][0]!='vowel']

# CMU is a subset of arpabet
cmu_1_char={}
cmu_1_char_to_gibberish={}
for p in cmu_phones:
    cmu_1_char[p]=arpabet_to_ipa[p.lower()]
    cmu_1_char_to_gibberish[cmu_1_char[p]]=cmu_to_gibberish[p]