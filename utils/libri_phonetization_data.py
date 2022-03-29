import cmudict
from pandas.io import pickle
import textgrid
# doc: https://github.com/kylebgorman/textgrid
from itertools import compress

from tqdm import tqdm
import pandas as pd
import os
from glob import glob

from utils.text_processing import remove_stress_annots, cmu_1_char

def get_phone_timings(f='data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid',word_idx=8):
    """Uses the (start,end) of a word and (starts,ends) of phonemes to retrieve phonemes corresponding to a word

    Args:
        f (str, optional): [description]. Defaults to 'data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'.
        word_idx (int, optional): [description]. Defaults to 8.

    Returns:
        list of Intervals: each element is a phoneme with attributes element.min, element.max and element.mark
    """
    tg = textgrid.TextGrid.fromFile(f)
    filter=[tg[0][word_idx].overlaps(el) for el in tg[1]]
    return list(compress(tg[1], filter))

def get_all_phone_with_timings(f='data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'):
    """get all phonemes of a sentence located in tg[1], and filter silence and empty parts, then convert to DataFrame

    Args:
        f (str, optional): [description]. Defaults to 'data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'.
        word_idx (int, optional): [description]. Defaults to 8.

    Returns:
        [type]: [description]
    """
    tg = textgrid.TextGrid.fromFile(f)
    # get phones and drop "sp", "sil" and empty strings
    phones=[[el.minTime, el.maxTime, el.mark] for el in tg[1] if el.mark not in ['sil','sp','','spn']]
    phones=pd.DataFrame(phones)
    phones.columns=["start", "end", "phone"]
    return phones

def get_sentence(f='data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'):
    """get all words of a sentence located in tg[0]

    Args:
        f (str, optional): [description]. Defaults to 'data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'.

    Returns:
        [type]: [description]
    """
    tg = textgrid.TextGrid.fromFile(f)
    # get words and drop empty strings
    words=[el.mark for el in tg[0] if el.mark]
    return ' '.join(words)


def phonetics_for_row(row, libri_words_df):
    """
    retrieving phonetics by word that is not available directly from textgrids, but can be extracted from the dataframe
    as I already extracted phonemes for each word using overlapping in timings
    it is important to use file_idx to be sure that the phonetic transcription of a word is correct. Because
    words can have several phonetic transcriptions depending on the context. e.g., the -> DH AH0, DH IY0
    In fact, even doing that may lead to some mistake, if the word is several times in the same sentence with different pronunciations...


    Args:
        row ([type]): [description]
        libri_words_df ([type]): [description]

    Returns:
        [type]: [description]
    """
    sentence=get_sentence(row.path)
    phonetics=[]
    for w_idx,w in enumerate(sentence.split(' ')):
        try:
            phonetics.append(libri_words_df[(libri_words_df.file_idx==row.file_idx) &(libri_words_df.word_idx==w_idx) & (libri_words_df.word==w)].iloc[0,:].phones)
        except IndexError:
            import pdb;pdb.set_trace()
    return phonetics



def build_librispeech_words_df(
        data_set='dev-clean',
        basepath='data/librispeech_alignments',
        audio_path='data/LibriSpeech/',
        n=None
        ):
    path=os.path.join(basepath, data_set)
    files = glob(path+'/*/*/*.TextGrid')
    if n is not None: files=files[:n]
    records=[]
    for f_idx,f in tqdm(enumerate(files)):
        tg = textgrid.TextGrid.fromFile(f)
        tg_words=tg[0]
        # filter out None words
        tg_words=[el for el in tg_words if el.mark]
        tg_phones=tg[1]
        for i,el in enumerate(tg_words):
            # this takes all phonemes that overlap with the word. I found overlap function in Interval class:
            # https://github.com/kylebgorman/textgrid/blob/master/textgrid/textgrid.py
            filter=[el.overlaps(p) for p in tg[1]]
            phone_intervals=list(compress(tg[1], filter))
            phones=[el.mark for el in phone_intervals]
            word=el.mark
            start=el.minTime
            end=el.maxTime
            wav_path=os.path.join(audio_path,'/'.join(f.split('/')[2:]).split('.')[0]+'.flac')
            d={'word':word, 'phones':" ".join(phones), 'file_idx':f_idx, 'word_idx':i, 'start':start, 'end':end, 'path':f, 'wav_path':wav_path}
            # d={'word':word, 'phones':" ".join(phones), 'file_idx':f_idx, 'word_idx':i, 'start':start, 'end':end, 'path':f, 'wav_path':wav_path, 'sentence':get_sentence(f)}

            records.append(d)
    libri_words_df=pd.DataFrame.from_records(records)
    return libri_words_df

def libri_phonetics(
        data_set='dev-clean',
        basepath='data/librispeech_alignments',
        audio_path='data/LibriSpeech/',
        # audio_path='/mnt/c/Users/noe_t/Downloads/LibriSpeech/',
        n=None
        ):
    """Build a dataframe witha all sentences of the set. columns are ['text', 'phone_df', 'path', 'wav_path']
    where phone_df is itself a dataframe for which columns are ["start", "end", "cmu_phone", "ipa_phone"]

    Args:
        data_set (str, optional): [description]. Defaults to 'dev-clean'.
        basepath (str, optional): [description]. Defaults to '/data/librispeech_alignments'.
        audio_path (str, optional): [description]. Defaults to '/data/LibriSpeech/'.
        n ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    path=os.path.join(basepath, data_set)
    files = glob(path+'/*/*/*.TextGrid')
    if n is not None: files=files[:n]


    records=[]
    records_with_timings=[]

    for f in files:
        wav_path=os.path.join(audio_path,'/'.join(f.split('/')[-4:]).split('.')[0]+'.flac')
        phone_df=get_all_phone_with_timings(f)
        text=get_sentence(f)
        d={'text':text, 'phones':' '.join(phone_df['ipa_phone'].tolist()), 'path':f, 'wav_path':wav_path}
        d_with_timings={'text':text, 'phone_df':phone_df, 'path':f, 'wav_path':wav_path}
        records.append(d)
        records_with_timings.append(d_with_timings)
    libri_phonetics_df=pd.DataFrame.from_records(records)
    libri_phonetics_df_with_timings=pd.DataFrame.from_records(records_with_timings)


    libri_phonetics_df_with_timings.to_json('data/libri_text_ipa_timings_'+data_set+'.json')
    libri_phonetics_df.to_json('data/libri_text_ipa_'+data_set+'.json')

    return libri_phonetics_df_with_timings, libri_phonetics_df

def libri_phonetics_data(
        data_set='dev-clean',
        # audio_path='/mnt/c/Users/noe_t/Downloads/LibriSpeech/',
        ):
    path_df_t='data/libri_text_ipa_timings_'+data_set+'.json'
    path_df='data/libri_text_ipa_'+data_set+'.json'
    if os.path.exists(path_df_t) and os.path.exists(path_df):
        libri_phonetics_df_with_timings=pd.read_json(path_df_t)
        libri_phonetics_df=pd.read_json(path_df)
    else:
        libri_phonetics_df_with_timings, libri_phonetics_df=libri_phonetics(data_set=data_set)

    return libri_phonetics_df_with_timings, libri_phonetics_df



def cmu_ascii_mappings():
    #  if needed to lowercase:
    # [el.lower() for el in cmudict.symbols()]

    # I do a transliteration to ASCII characters that makes no sense for a human 
    # but use it only as an intermediate to be able to fine tune a hugging face model.
    # This is to be able to follow the same procedure as this:
    # https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
    # https://huggingface.co/elgeish/wav2vec2-base-timit-asr
    # https://github.com/elgeish/transformers/blob/cfc0bd01f2ac2ea3a5acc578ef2e204bf4304de7/examples/research_projects/wav2vec2/finetune_base_timit_asr.sh

    # I start to chr(33) because it is the first "visible" character (not an arrow or shift etc.) 
    # and not a space that I probably need to separate words
    # https://python-reference.readthedocs.io/en/latest/docs/str/ASCII.html

    cmudict.symbols()

    ascii_encoding={}
    ascii_decoding={}
    for i,el in enumerate(cmudict.symbols()):
        ascii_encoding[el]=chr(33+i)
        ascii_decoding[chr(33+i)]=el
    
    return ascii_encoding, ascii_decoding

def learning_content(selection, n=20):
    """
    -count the occurences of words to have an idea of their frequence in english
    -take the n first
    """
    counts=selection.word.value_counts().iloc[:n]
    frequent_selection=selection[selection.word.isin(counts.index)]
    frequent_selection_unique=frequent_selection.drop_duplicates(subset=['word'])
    # frequent_selection_unique[['word','phones']]
    frequent_selection_unique['frequency among examples']=counts[frequent_selection_unique.word].values/len(selection)*100
    return frequent_selection_unique[['word','phones','frequency among examples']].sort_values('frequency among examples', ascending=False)

def selection_with_and_without_s(libri_words_df, word='speak'):
    selection=libri_words_df[libri_words_df.word==word]

    phonetics=[]
    for i,r in selection.iterrows():
        p=phonetics_for_row(r, libri_words_df)
        # if the word is not the last one of the sentence, look at the next to see if it starts with an "S"
        if r.word_idx<len(p)-1:
            if p[r.word_idx+1].split(' ')[0]=="S":
                phonetics.append(float('nan'))
            else:
                phonetics.append(p)
        else:
            phonetics.append(p)

    selection['phonetics']=phonetics

    selection_s=libri_words_df[libri_words_df.word==word+'s']
    phonetics=[]
    for i,r in selection_s.iterrows():
        p=phonetics_for_row(r, libri_words_df)
        phonetics.append(p)
    
    selection_s['phonetics']=phonetics

    return selection, selection_s
    


def frequent_selection_containing(libri_words_df, phones=['AO0','IY1'], letters=[], n=20, option="contains"):
    """select from libri_words_df with criteria, then does a selection based on frequence in the list, sorted

    Args:
        phones (list, optional): [description]. Defaults to ['AO0','IY1'].
        letters (list, optional): [description]. Defaults to [].
        n (int, optional): [description]. Defaults to 20.
        option (str, optional): [description]. Defaults to "contains". can be 'endswith' or 'startstwith'

    Returns:
        [type]: [description]
    """
    
    def select(selection, phone, column='phones'):
        # apply one selection criteria
        if option=='contains':
            try:
                selection=selection[selection[column].str.contains(phone)]
            except:import pdb;pdb.set_trace()
        elif option=='endswith':
            selection=selection[selection[column].str.endswith(phone)]
        elif option=='startswith':
            selection=selection[selection[column].str.startswith(phone)]
        else:
            print('option should be in: ["contains","endwith","startswith"]')
        return selection
    
    selection=libri_words_df
    for phone in phones: selection=select(selection,phone)
    for l in letters:selection=select(selection,l, 'word')

    content=learning_content(selection, n=n)
    content.index=range(len(content))
    filename=option
    
    if phones!=[]:filename+='_phones_'+'_'.join(phones)
    if letters!=[]:filename+='_letters_'+'_'.join(letters)
    filename+='_n_'+str(n)

    selection.to_csv('results/'+filename+'.csv')

    return content


def frequent_word_selections_for_phones(libri_words_df,
                            phones=['IH0','IH1','IH2','IY0','IY1','IY2','AA0','AA1','AA2','AO0','AO1','AO2','OW0','OW1','OW2'], n=200):
    dfs=[]
    for phone in phones: 
        dfs.append(frequent_selection_containing(libri_words_df,[phone], n=n).word)
    df=pd.concat(dfs,axis=1)
    df.columns=phones
    return df


def frequent_ed_word_selections_by_termation(libri_words_df, n=200):
    from module_performance import get_phone_termination_dict
    phone_termination_dict, correct_alternatives=get_phone_termination_dict()
    import pickle
    res=pickle.load(open('performance_results/ed_performance_dev-clean.p','rb'))
    good_preterminations=[k for k in res['performances'] if res['performances'][k]>=0.8]
    bad_preterminations=[k for k in res['performances'] if res['performances'][k]<0.8]

    terminations=list(correct_alternatives.keys())

    for term in terminations:
        preterm_selection=[pret for pret in good_preterminations if phone_termination_dict[pret]==term]
        dfs=[]
        for preterm in preterm_selection:
            full_termination=' '+' '.join([preterm,term])
            dfs.append(frequent_selection_containing(libri_words_df, phones=[full_termination], letters=['ed'], option='endswith', n=n).word)
        df=pd.concat(dfs,axis=1)
        df.columns=preterm_selection
        df.to_csv('results/ed_in_'+term+'.csv')

if __name__ == "__main__":
    libri_words_df=build_librispeech_words_df()
    # libri_words_df=build_librispeech_words_df(n=10000)

    libri_words_df[libri_words_df.word=='speaks']
    selection=libri_words_df[libri_words_df.word=='speak']

    
    phonetics=[]
    for i,r in selection.iterrows():
        p=phonetics_for_row(r, libri_words_df)
        phonetics.append(p)


    libri_words_df[libri_words_df.word=='low']
    selection=libri_words_df[libri_words_df.phones.str.endswith(' IH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' T IH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' T EH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' T AH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' D IH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' D AH0 D')]

    libri_words_df[libri_words_df.phones.str.endswith(' S')]
    libri_words_df[libri_words_df.phones.str.endswith(' S') & libri_words_df.word.str.endswith('s')]


    libri_words_df[libri_words_df.phones.str.endswith(' V D')]
    libri_words_df[libri_words_df.phones.str.endswith(' M D')]
    libri_words_df[libri_words_df.phones.str.endswith(' NG D')]
    selection=libri_words_df[libri_words_df.phones.str.endswith(' DH D')]

    # if not os.path.exists('file_selection'): os.makedirs('file_selection')
    # for i,r in selection.iterrows():
    #     shutil.copy(r.wav_path, 'file_selection')

    libri_words_df[libri_words_df.phones.str.endswith('AO1')].word.unique()
    libri_words_df[libri_words_df.phones.str.contains('AO1')].word.unique()
    libri_words_df[libri_words_df.phones.str.contains('AO0')&libri_words_df.phones.str.contains('IY1')]
    libri_words_df[libri_words_df.phones.str.contains('OW1')]
    libri_words_df[libri_words_df.phones.str.endswith(' B D') & libri_words_df.word.str.endswith('bed')]
    libri_words_df[libri_words_df.phones.str.endswith(' D') & libri_words_df.word.str.endswith('ied')]

    libri_words_df[libri_words_df.phones.str.endswith(' P T') & libri_words_df.word.str.endswith('ped')]
    libri_words_df[libri_words_df.phones.str.endswith(' K T') & libri_words_df.word.str.endswith('ked')]
    libri_words_df[libri_words_df.phones.str.endswith(' SH T')]
    libri_words_df[libri_words_df.phones.str.endswith(' F T')]


    selection=libri_words_df[libri_words_df.phones.str.endswith(' P T') & libri_words_df.word.str.endswith('ped')]

    frequent_selection_containing(libri_words_df,['IH1'], n=200)


    libri_words_df[libri_words_df.phones.str.contains(' IY1 ') & libri_words_df.word.str.contains('ea')]
    libri_words_df[libri_words_df.phones.str.contains(' UH1 ') & libri_words_df.word.str.contains('oo')]
    libri_words_df[libri_words_df.phones.str.contains(' UW1 ') & libri_words_df.word.str.contains('oo')]
    libri_words_df[libri_words_df.phones.str.contains(' AW1 ') & libri_words_df.word.str.contains('ou')]
    libri_words_df[libri_words_df.phones.str.contains(' UW1 ') & libri_words_df.word.str.contains('ou')]

    
    selection=libri_words_df[libri_words_df.word=='public']

    sents=[]
    for i,r in selection.iterrows():
        sents.append(get_sentence(r.path))
    example=selection.iloc[0,:]

    get_phone_timings(f=example.path, word_idx=example.word_idx)