import os, psutil;print_memory_usage=lambda stage: print(stage + ": "+ str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
print_memory_usage('RAM - pronunciation_dictionaries start')
import cmudict
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
from syllabipy.sonoripy import SonoriPy

unstress = lambda el: el[:-1] if (len(el)>0 and el[-1]) in ['0','1','2'] else el
def remove_stress_annots(transcription=['K', 'AA1', 'F', 'IY0']):     return [unstress(el) for el in transcription]


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

print_memory_usage('RAM - pronunciation_dictionaries after external libraries')

# standard from FB http://fbdevwiki.com/wiki/Locales
lang_to_MFA_g2p_models={
    'en_GB':'english_uk_mfa',
    'en_US':'english_us_mfa',
    'fr_FR':'french_mfa',
    'es_ES':'spanish_spain_mfa',
    'es_LA':'spanish_latin_america_mfa'
}


def normalize_termination(ps=[['S', 'T', 'AA1', 'R', 'T', 'AH0', 'D']], word="started"):
    """normalize final -ed and final -s terminations for CMU to remove the variations "AH0_D" vs "IH0_D" and "AH0_Z" vs "IH0_Z"
    It does that on all phonetics alternatives

    Args:
        ps (list, optional): _description_. Defaults to [['S', 'T', 'AA1', 'R', 'T', 'AH0', 'D']].
        word (str, optional): _description_. Defaults to "started".
    """
    termination_correction_data=[
        ("ed", "AH0_D", "IH0_D"),
        ("es", "AH0_Z", "IH0_Z"),
    ]

    for t_data in termination_correction_data:
        # Rule for verbs in -ded or -ted: we want to get rid of the "AH0_D" alternative
        for p in ps:
            if (p[-2:]==t_data[1].split('_')) and (word[-2:]==t_data[0]):
                p[-2:]= t_data[-1].split('_') # ['IH0','D']


def get_augmented_cmudict():
    cmudict_dict=cmudict.dict()
    
    for k in tqdm(cmudict_dict):
        normalize_termination(cmudict_dict[k], k)

    inconsistent_word_stresses={}
    for k in cmudict_dict:
        for alt in cmudict_dict[k]:
            if '-' not in k:
                if sum(['1' in p for p in alt])>1:
                    inconsistent_word_stresses[k]=alt
    # len(inconsistent_word_stresses)
    # There are too much to be corrected, and too few to really care, it's less than 1% and most probalably unfrequent words...

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
        'thirteen':[['TH', 'ER2', 'T', 'IY1', 'N']],
        'fourteen':[['F', 'AO2', 'R', 'T', 'IY1', 'N']],
        'fifteen':[['F', 'IH2', 'F', 'T', 'IY1', 'N']],
        'sixteen':[['S', 'IH2', 'K', 'S', 'T', 'IY1', 'N']],
        'seventeen':[['S', 'EH2', 'V', 'AH0', 'N', 'T', 'IY1', 'N']],
        'eighteen':[['EY0', 'T', 'IY1', 'N'], ['EY2', 'T', 'IY1', 'N']],
        'nineteen':[['N', 'AY2', 'N', 'T', 'IY1', 'N']],

        'thirteenth':[['TH', 'ER2', 'T', 'IY1', 'N', 'TH']],
        'fourteenth':[['F', 'AO2', 'R', 'T', 'IY1', 'N', 'TH']],
        'fifteenth':[['F', 'IH2', 'F', 'T', 'IY1', 'N', 'TH']],
        'sixteenth':[['S', 'IH2', 'K', 'S', 'T', 'IY1', 'N', 'TH']],
        'seventeenth':[['S', 'EH2', 'V', 'AH0', 'N', 'T', 'IY1', 'N', 'TH']],
        'eighteenth':[['EY0', 'T', 'IY1', 'N', 'TH'], ['EY2', 'T', 'IY1', 'N', 'TH']],
        'nineteenth':[['N', 'AY2', 'N', 'T', 'IY1', 'N', 'TH']],

        'engineer':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R']],
        'engineers':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R', 'Z']],
        'engineering':[['EH2', 'N', 'JH', 'AH0', 'N', 'IH1', 'R', 'IH0', 'NG']],
        'downstairs':[['D', 'AW0', 'N', 'S', 'T', 'EH1', 'R', 'Z']],
        'trainee':[['T', 'R', 'EY0', 'N', 'IY1']],
        'outside':[['AW0', 'T', 'S', 'AY1', 'D']],
        'trespasser':[['T', 'R', 'EH0', 'S', 'P', 'AE1', 'S', 'ER0']],
        'trespassers':[['T', 'R', 'EH0', 'S', 'P', 'AE1', 'S', 'ER0', 'Z']],
        'outdoors':[['AW1', 'T', 'D', 'AO2', 'R', 'Z']],
        'unreasonable':[['AH0', 'N', 'R', 'IY1', 'Z', 'AH0', 'N', 'AH0', 'B', 'AH0', 'L']],
        'coworker':[['K', 'OW1', 'W', 'ER0', 'K', 'ER0']],
        'coworkers':[['K', 'OW1', 'W', 'ER0', 'K', 'ER0', 'Z']],
        'multitasker':[['M', 'AH1', 'L', 'T', 'IY0', 'T', 'AE2', 'S', 'K', 'ER0']],
        'domestically':[['D', 'AH0', 'M', 'EH1', 'S', 'T', 'IH0', 'K', 'AH0', 'L', 'IY0']],

        'hm':cmudict_dict['hum'],
        'hmm':cmudict_dict['hum'],
        'mmh':cmudict_dict['hum'],
        'ok':cmudict_dict['okay']

	}
    for k in corrections:
        cmudict_dict[k]=corrections[k]
    return cmudict_dict



cmudict_dict=get_augmented_cmudict()

def differs_by_one_insertion(seq1, seq2):
    """check if two sequences only differs by one insertion, and what was the element inserted

    Args:
        seq1 (_type_): _description_
        seq2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if abs(len(seq2)  - len(seq1))!=  1: return False, None, None
    if len(seq2)>len(seq1): big_seq=seq2; small_seq=seq1
    else: big_seq=seq1; small_seq=seq2
    
    i = 0
    j = 0
    inserted_token = None
    inserted_idx = None

    while i < len(small_seq):
        if small_seq[i] != big_seq[j]:
            if inserted_token is not None: 
                return False, None, None
            inserted_token = big_seq[j]
            inserted_idx = j
            j += 1
        else:
            i += 1
            j += 1
    

    if inserted_token is None:
        inserted_token = seq2[-1]
        inserted_idx = len(seq2)-1
    # explicit check that the only difference is the inserted token
    if small_seq != big_seq[:inserted_idx]+big_seq[inserted_idx+1:]: return False, None, None
    return True, inserted_token, inserted_idx

def select_keys_with_one_diff(my_dict, token="AH0"):
    """go through a dictionnary of alternatives, and check if they contain two alternatives that differs only by 1 token that is "AH0" by default.
    This is, e.g., to look for words like "typically" or "listening" that can differs only by the presence of absence of a schwa and that modify the susequent number of syllables

    Args:
        my_dict (_type_): _description_
        token (str, optional): _description_. Defaults to "AH0".

    Returns:
        _type_: _description_
    """
    result_keys = []
    values_small = []
    inserted_idxs = []

    for key, value in my_dict.items():
        if len(value) < 2:
            continue
        for i in range(len(value)):
            for j in range(i+1, len(value)):
                is_one_diff, inserted_token, inserted_idx = differs_by_one_insertion(value[i], value[j])
                if is_one_diff and inserted_token==token:
                    # print(f"Key '{key}' meets the criteria. Inserted token: '{inserted_token}'.")
                    result_keys.append(key)
                    if len(value[i])<len(value[j]): small_value=value[i]
                    else: small_value=value[j]
                    values_small.append(small_value)
                    inserted_idxs.append(inserted_idx)
                    break
            if key in result_keys:
                break
    return result_keys, values_small, inserted_idxs


def get_formatted_cmudict(phonetic_dict=cmudict_dict, mode='CMU'):
    
    df=pd.DataFrame()
    df['text']=phonetic_dict.keys()

    print('get_formatted_cmudict')
    from tqdm import tqdm
    tqdm.pandas()
    df['phonetics']=df.progress_apply(lambda r: phonetic_dict[r.text] if r.text in phonetic_dict else float('nan'), axis=1)

    df=df.dropna()

    # not sure why, it seems there are empty entries in cmudict
    df=df[df.apply(lambda r: len(r.phonetics), axis=1)>0]
    
    df['syl_p']=df['phonetics'].progress_apply(lambda r: SonoriPy(r[0], mode=mode)[0])
    df['formatted_phonetics']=df['syl_p'].apply(lambda p: '|'.join(['_'.join(syl) for syl in p]))

    return df



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
    # e.g. mfa g2p <(echo "salut") french_mfa >(cat)

    cmd="mfa g2p "+'<(echo "'+word+'") '+model+" >(cat)"
    cmd_list=['bash', '-c',cmd]
    p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
    out, err = p.communicate()

    # the lines not being "INFO" messages correspond to the output that would have been written to the file
    data='\n'.join([el for el in out.decode('utf8').split('\n') if not 'INFO' in el])
    data='\n'.join(data.split('\n')[1:])
    print("mfa g2p for ", word, ' :',data)

    df = pd.read_csv(StringIO(data), sep="\t", header=None, encoding='utf8')
    df.columns=['text', 'ipa']
    ipa=df.apply(lambda r: r.ipa.split(' '), axis=1)
    df['ipa']=ipa

    df['ipa']=df.apply(lambda r: [r.ipa], axis=1)
    ipa_dict=df.groupby(['text']).sum().to_dict()['ipa']

    return ipa_dict

def mfa_english_add_schwa_alternative(mfa_d):
    """Take keys from cmudict that have 2 alternatives for which the only difference is an inserted schwa (AH0 in CMU)
    And add in mfa this alternative if the word exists in the dictionary and has the same reduced phonetics of the small alternative (without schwa)
    """
    # mfa_d=get_augmented_mfa_dict(lang='en_US')
    result_keys, values_small, inserted_idxs=select_keys_with_one_diff(cmudict_dict, token="AH0")
    for k, v, idx in zip(result_keys, values_small, inserted_idxs):
        if k in mfa_d:
            for alt in mfa_d[k]:
                if [cmu_reducer[p] for p in alt] == remove_stress_annots(v):
                    alt_with_schwa=alt[:idx]+["ə"]+alt[idx:]
                    mfa_d[k].append(alt_with_schwa)




def get_mfa_dict(path='data/spanish_spain_mfa.dict'):
    df=get_mfa_df(path)
    df['ipa']=df.apply(lambda r: [r.ipa], axis=1)
    ipa_dict=df.groupby(['text']).sum().to_dict()['ipa']
    return ipa_dict


def generate_acronym_letter_mfa_dicts():
    
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
    acronym_dict_long['k']=[['k', 'a']]

    acronym_dict_long['w']=[max(mfa_g2p('doublevé', lang_to_MFA_g2p_models[lang])['doublevé'], key=len)]
    acronym_dict_long['x']=[['i', 'k', 's']]
    acronym_dict_long['y']=[max(mfa_g2p('igrec', lang_to_MFA_g2p_models[lang])['igrec'], key=len)]
    acronym_dict_long['z']=[['z', 'ɛ', 'd']]

    with open('data/acronyms_fr_FR_mfa.dict','w') as f: f.write(json.dumps(acronym_dict_long))

def process_diphtongs_r(mfa_us):
    # Here it is explained that diphtong + r-colored schwa was replaced by diphtong + 'ɹ'. For consistency in syllables, I prefer put it back to r-colored one
    # https://mfa-models.readthedocs.io/en/latest/mfa_phone_set.html#:~:text=Diphthong%20%2B%20rhotic%20standardization%3A

    from syllabipy.sonoripy import define_categories
    d=define_categories(mode="MFA_IPA")
    mfa_diphtongs={'aw','aj','ej','ow','əw','oj', 'ɔj'}
    for k in mfa_us:
        # for words like powered
        for idx_alt in range(len(mfa_us[k])):
            # print(idx_alt, "for len", len(mfa_us[k]))
            if len(mfa_us[k][idx_alt])>=3:
                char_list=mfa_us[k][idx_alt]
                for i in range(len(char_list) - 2):
                    if char_list[i] in mfa_diphtongs and char_list[i+1] == 'ɹ' and (char_list[i+2] not in d['vowels']):
                        # print("Found 'r' after reference character", char_list[i])
                        # print(k, ': ', mfa_us[k])
                        # print("after char:", char_list[i+2])
                        char_list[i+1]="ɚ"
                        mfa_us[k][idx_alt]=char_list
                        # print(k, ': ', mfa_us[k])
            # words like power
            if len(mfa_us[k][idx_alt])>=2:
                if mfa_us[k][idx_alt][-1]=='ɹ' and (mfa_us[k][idx_alt][-2] in mfa_diphtongs):
                    # print(k, ': ', mfa_us[k])
                    mfa_us[k][idx_alt]=mfa_us[k][idx_alt][:-1]+["ɚ"]
                    # print(k, ': ', mfa_us[k])


def get_augmented_mfa_dict(lang='es_ES'):
    from syllabipy.sonoripy import define_categories
    vowels=define_categories(mode="MFA_IPA")['vowels']

    d=get_mfa_dict(path='data/'+lang_to_MFA_g2p_models[lang]+'.dict')
    with open('data/acronyms_'+lang+'_mfa.dict','r') as f: acronyms=json.loads(f.read())
    for k in acronyms: d[k]=acronyms[k]

    
    if lang.split('_')[0]=="fr":
        with open('data/add_dict_'+lang+'.dict','r') as f: add_dict=json.loads(f.read())
        for k in add_dict: d[k]=add_dict[k]

    if lang.split('_')[0]=="en":
        with open('data/add_dict_'+lang+'.dict','r') as f: add_dict=json.loads(f.read())
        for k in add_dict: d[k]=add_dict[k]

        mfa_english_add_schwa_alternative(d)

        # corrections={
        # 'areas':[['EH1','R','IH0','AH0','Z']],
        # 'live':[['L', 'IH1', 'V']],
        # }
        # for k in corrections: d[k]=corrections[k]

        if lang=="en_US":
            process_diphtongs_r(d)

        # "manual" corrections
        d["have"]=[['h', 'æ', 'v']]
        d["i'll"]=[['aj', 'ɫ'], ['ɑː', 'ɫ'], ['ɫ̩']]
        d["they'll"]=[['ð', 'ɫ̩']]
        d["i've"]=[['aj', 'v']]
        d["mmh"]=[['m̩']]
        d['every']=[['ɛ', 'v', 'ə', 'ɹ', 'i'],['ɛ', 'v', 'ɹ', 'i']]

        if lang=="en_US":
            d["salesperson"]=[['s', 'ej', 'l', 'z', 'p', 'ɝ', 's', 'ə', 'n']]
            d["salespersons"]=[['s', 'ej', 'l', 'z', 'p', 'ɝ', 's', 'ə', 'n', 'z']]
            d["salesperson's"]=[['s', 'ej', 'l', 'z', 'p', 'ɝ', 's', 'ə', 'n', 'z']]
            d["hypothetical"]=[['h', 'aj', 'p', 'ə', 'θ', 'ɛ', 'tʲ', 'ɪ', 'k', 'ɫ̩'], ['h', 'aj', 'p', 'ə', 'θ', 'ɛ', 'ɾʲ', 'ɪ', 'k', 'ɫ̩']]


            # [k for k in mfa_us if (k.endswith('ior') and mfa_us[k][0][-2:]==mfa_us['superior'][0][-2:])]
            # similar to process diphtong+r, it seems that 
            for k in d:
                if (k.endswith('ior') and d[k][0][-2:]==d['superior'][0][-2:]):
                    d[k]=[alt[:-1]+['ɚ'] for alt in d[k]]


        if lang=="en_GB":
            mfa_us=get_mfa_dict(path='data/'+lang_to_MFA_g2p_models["en_US"]+'.dict')
            d["salesperson"]=[['s', 'ej', 'l', 'z', 'p', 'ɜː', 's', 'ə', 'n']]
            d["salespersons"]=[['s', 'ej', 'l', 'z', 'p', 'ɜː', 's', 'ə', 'n', 'z']]
            d["salesperson's"]=[['s', 'ej', 'l', 'z', 'p', 'ɜː', 's', 'ə', 'n', 'z']]
            d['assets']=[['ə', 's', 'ɛ', 't', 's']]
            d['pie']=mfa_us['pie']


            # for some, to correct easily, take the us version that was checked to be okay
            # this group is generally a removed schwa
            d['difference']=mfa_us['difference']
            d['differences']=mfa_us['differences']
            d['confidential']=mfa_us['confidential']
            d['favourite']=mfa_us['favourite']
            d['favorite']=mfa_us['favorite']
            d['january']=mfa_us['january']
            d['february']=mfa_us['february']

            # these are kind of diphtongs separated into two syllables
            d['against']=mfa_us['against']
            d['proteins']=[alt+['z'] for alt in d['protein']]

            d["hypothetical"]=[['h', 'aj', 'p', 'ə', 'θ', 'ɛ', 't', 'ɪ', 'k', 'ɫ̩']]

            
            
            # [k for k in mfa_gb if (k.endswith('tion') and mfa_gb[k][0][-2:]==mfa_gb['immigration'][0][-2:])]
            # [k for k in mfa_gb if (k.endswith('tial') and mfa_gb[k][0][-2:]==mfa_gb['confidential'][0][-2:])]
            # [k for k in mfa_gb if (k.endswith('cal') and mfa_gb[k][0][-2:]==mfa_gb['hypothetical'][0][-2:])]
            

            # [k for k in mfa_gb if (k in mfa_us and mfa_us[k][0][-1]=='ɫ̩' and mfa_gb[k][0][-1]=='ɫ' and mfa_gb[k][0][-2] not in vowels)]
            # [k for k in mfa_gb if (k in mfa_us and mfa_us[k][0][-1]=='m̩' and mfa_gb[k][0][-1]=='m' and mfa_gb[k][0][-2] not in vowels)]
            # [k for k in mfa_gb if (k in mfa_us and mfa_us[k][0][-1]=='n̩' and mfa_gb[k][0][-1]=='n' and mfa_gb[k][0][-2] not in vowels)]
            # [k for k in mfa_gb if (k in mfa_us and mfa_us[k][0][0]=='θ' and mfa_gb[k][0][0]=='f')]

            # missing schwaed-l,  schwaed-n or schwaed-m, and weird
            for k in d:
                if (k in mfa_us and mfa_us[k][0][-1]=='ɫ̩' and d[k][0][-1]=='ɫ' and d[k][0][-2] not in vowels):
                    d[k]=[alt[:-1]+['ɫ̩'] for alt in d[k]]
                if (k in mfa_us and mfa_us[k][0][-1]=='m̩' and d[k][0][-1]=='m' and d[k][0][-2] not in vowels):
                    d[k]=[alt[:-1]+['ɫ̩'] for alt in d[k]]
                if (k in mfa_us and mfa_us[k][0][-1]=='n̩' and d[k][0][-1]=='n' and d[k][0][-2] not in vowels):
                    d[k]=[alt[:-1]+['ɫ̩'] for alt in d[k]]
                if (k in mfa_us and mfa_us[k][0][0]=='θ' and d[k][0][0]=='f'):
                    d[k]=[['θ']+alt[1:] for alt in d[k]]


    return d




from collections import ChainMap
def add_mfa_dicts():
    formatted_cmudict_df=get_formatted_cmudict()
    apostroph_s_words=formatted_cmudict_df[formatted_cmudict_df.text.str.endswith("'s")]

    apostroph_d_words=formatted_cmudict_df[formatted_cmudict_df.text.str.endswith("'d")]


    # to have all words ending in "'s" in mfa dicts in english, I select all such words from cmudict and look at the end for knowing if it's a "S" or "Z" sound, and take the word in correponding mfa_dict
    apostroph_s_words["apostroph_s_phone"]=apostroph_s_words.formatted_phonetics.str.split('_').apply(lambda r:r[-1].lower())
    lang="en_GB"
    mfa_d=get_augmented_mfa_dict(lang)
    new_words=apostroph_s_words.apply(lambda r: {r['text']:[el+[r['apostroph_s_phone']] for el in mfa_d[r['text'][:-2]]]} if r['text'][:-2] in mfa_d else float('nan'), axis=1).dropna()
    add_dict=dict(ChainMap(*new_words))

    d_dict={}
    for w in apostroph_d_words.text.tolist():
        d_dict[w]=[el+["d"] for el in mfa_d[w.split("'")[0]]]
    corr_d_dict={
                "it'd": [['ɪ', 't','ɪ', 'd'], ['ɪ', 'ʔ','ɪ', 'd']],
                "that'd": [['d̪', 'æ', 't','ɪ', 'd'],
                            ['d̪', 'æ', 'ʔ','ɪ', 'd'],
                            ['ð', 'æ', 't','ɪ', 'd'],
                            ['ð', 'æ', 'ʔ','ɪ', 'd']],
                "what'd": [['w', 'ɒ', 't','ɪ', 'd'], ['w', 'ɒ', 'ʔ','ɪ', 'd']],
                }
    
    for k in corr_d_dict: d_dict[k]=corr_d_dict[k]
    for k in d_dict: add_dict[k]=d_dict[k]

    with open('data/add_dict_en_GB.dict','w') as f: f.write(json.dumps(add_dict))

    lang="en_US"
    mfa_d=get_augmented_mfa_dict(lang)
    new_words=apostroph_s_words.apply(lambda r: {r['text']:[el+[r['apostroph_s_phone']] for el in mfa_d[r['text'][:-2]]]} if r['text'][:-2] in mfa_d else float('nan'), axis=1).dropna()
    add_dict=dict(ChainMap(*new_words))

    
    d_dict={}
    for w in apostroph_d_words.text.tolist():
        d_dict[w]=[el+["d"] for el in mfa_d[w.split("'")[0]]]
    corr_d_dict={
        "it'd": [['ɪ', 't','ɪ', 'd'], ['ɪ', 'ʔ','ɪ', 'd']],
        "that'd": [['d̪', 'æ', 't','ɪ', 'd'],
                    ['d̪', 'æ', 'ʔ','ɪ', 'd'],
                    ['ð', 'æ', 't','ɪ', 'd'],
                    ['ð', 'æ', 'ʔ','ɪ', 'd']],
        "what'd": [['w', 'ɐ', 't','ɪ', 'd'], ['w', 'ɐ', 'ɾ','ɪ', 'd'], ['w', 'ɐ', 'ʔ','ɪ', 'd']],
    }
    
    for k in corr_d_dict: d_dict[k]=corr_d_dict[k]
    for k in d_dict: add_dict[k]=d_dict[k]

    with open('data/add_dict_en_US.dict','w') as f: f.write(json.dumps(add_dict))

    # "l":"l",  "s":"s",  j'  m' "n":"n",  "d":"d",  "qu":"k",  "t":"t"  
    
    lang="fr_FR"
    mfa_d=get_augmented_mfa_dict(lang)

    # TODO: here this additional dict takes 28Mb which seems excessive when you know the original french_mfa dict takes only 5Mb ...
    # but it avoids mfa_g2p be called all the time easily. Maybe it would be better to implement that in mfa_g2p function
    from syllabipy.sonoripy import define_categories
    d=define_categories(mode="MFA_IPA")
    apostroph_letter_prefixes={"l":"l",  "s":"s",  "c":"s",  "j":"ʒ",  "m":"m", "n":"n",  "d":"d",  "qu":"k",  "t":"t"}
    add_dict={}
    for k in mfa_d: 
        if mfa_d[k][0][0] in d['vowels']:
            for letter in apostroph_letter_prefixes:
                add_dict["'".join([letter,k])]=[[apostroph_letter_prefixes[letter]] + el for el in mfa_d[k]]

    with open('data/add_dict_fr_FR.dict','w') as f: f.write(json.dumps(add_dict))





def build_mfa_phone_set():
    dict_paths=glob('data/*mfa.dict')
    dfs=[get_mfa_df(p) for p in dict_paths]
    df=pd.concat(dfs)
    phones=sorted(list(set().union(*df.ipa.apply(lambda r: set(r)).tolist())))

    # Writing to sample.json
    with open("data/mfa_phones.json", "w") as outfile: outfile.write(json.dumps(phones))

    return phones

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

arpabet_to_2_char_ipa=arpabet_to_1_char_ipa

arpabet_to_2_char_ipa['aw']='aʊ'
arpabet_to_2_char_ipa['ay']='aɪ'
arpabet_to_2_char_ipa['ey']='eɪ'
arpabet_to_2_char_ipa['ow']='oʊ'
arpabet_to_2_char_ipa['oy']='ɔɪ'

arpabet_to_2_char_ipa['ch']='tʃ'
arpabet_to_2_char_ipa['jh']='dʒ'


cmu_phones_info=cmudict.phones()
cmu_phones=set([el[0] for el in cmu_phones_info])
cmu_vowels=set([p[0] for p in cmu_phones_info if p[1][0]=='vowel'])
cmu_consonants=set([p[0] for p in cmu_phones_info if p[1][0]!='vowel'])

cmu_diphtongs=[p[0] for p in cmu_phones_info if (p[1][0]=='vowel' and p[0][-1] in cmu_consonants)]


cmu_stressed_vowels=set(cmudict.symbols())-cmu_phones

cmu_stressed_alphabet=sorted(list(cmu_stressed_vowels)+list(cmu_consonants))

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
mfa_simplifier={k:k for k in ipa_alphabet}
mfa_simplifier['ɲ']='n'
mfa_simplifier['c']='k'
mfa_simplifier['ʎ']='l'
mfa_simplifier['ɫ']='l'
mfa_simplifier['ɟ']='ɡ'
mfa_simplifier['ç']='h'

mfa_simplifier['pʰ']='p'
mfa_simplifier['tʰ']='t'
mfa_simplifier['cʰ']='k'
mfa_simplifier['kʰ']='k'

# https://en.wikipedia.org/wiki/International_Phonetic_Alphabet_chart_for_English_dialects#Chart
# https://easypronunciation.com/en/american-english-pronunciation-ipa-chart

mfa_simplifier['ʔ']='t'
# this can be either "t" or "d". In fact, I hope that its presence is generally not in the first alternatives in mfa dicts, that will generally have the "t" or "d"
mfa_simplifier['d̪']='ð'
mfa_simplifier['t̪']='θ'

mfa_simplifier['ʉ']='u'
mfa_simplifier['ʉː']='uː'


for p in ipa_alphabet:
    if p[-1]=="ʲ":
        mfa_simplifier[p]=p[:-1]


simple_mfa=list(set([mfa_simplifier[p] for p in ipa_alphabet]))

# print(sorted(simple_mfa))

mfa_to_display_ipa={k:mfa_simplifier[k] for k in mfa_simplifier}
mfa_to_display_ipa['aw']='aʊ'
mfa_to_display_ipa['aj']='aɪ'
mfa_to_display_ipa['ej']='eɪ'
mfa_to_display_ipa['ow']='oʊ'
mfa_to_display_ipa['əw']='əʊ'
mfa_to_display_ipa['oj']='ɔɪ'

# with open('data/mfa_to_display_ipa.json','w') as f: f.write(json.dumps(mfa_to_display_ipa))

# keys=list(mfa_to_display_ipa.keys())
# values=[mfa_to_display_ipa[k] for k in keys]

# import pandas as pd
# mfa_to_display_ipa_df=pd.DataFrame()
# mfa_to_display_ipa_df['complex']=keys
# mfa_to_display_ipa_df['simple']=values

# the schwa+consonant ones
for p in mfa_simplifier:
    if p[-1]=='̩':
        mfa_to_display_ipa[p]="ə"+p[:-1]

simple_mfa_display=list(set([mfa_to_display_ipa[p] for p in ipa_alphabet]))
# print(sorted(simple_mfa_display))


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
long_vowels_ascii=[el[:-1]+':' for el in long_vowels]
# cmu does not have a symbol to differentiate long and short vowels with the same acoustics
for lv in long_vowels: cmu_reducer[lv]=cmu_reducer[lv[:-1]]
for lv in long_vowels_ascii: cmu_reducer[lv]=cmu_reducer[lv[:-1]]

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

def use_tests():
    # just to take a look at terminations in MFA_IPA
    mfa_d=get_augmented_mfa_dict(lang='en_US')
    for word in mfa_d:
        p=mfa_d[word]
        if (word[-2:]=="es"): print(word, p)