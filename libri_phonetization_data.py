import cmudict
import textgrid
# doc: https://github.com/kylebgorman/textgrid
from itertools import compress

from tqdm import tqdm
import pandas as pd
import os
from glob import glob
import shutil

def get_phone_timings(f='librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid',word_idx=8):
    """Uses the (start,end) of a word and (starts,ends) of phonemes to retrieve phonemes corresponding to a word

    Args:
        f (str, optional): [description]. Defaults to 'librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'.
        word_idx (int, optional): [description]. Defaults to 8.

    Returns:
        list of Intervals: each element is a phoneme with attributes element.min, element.max and element.mark
    """
    tg = textgrid.TextGrid.fromFile(f)
    tg[0][word_idx]
    filter=[tg[0][word_idx].overlaps(el) for el in tg[1]]
    return list(compress(tg[1], filter))

def get_sentence(f='librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'):
    tg = textgrid.TextGrid.fromFile(f)
    words=[el.mark for el in tg[0]]
    # drop empty strings
    words = [x for x in words if x]
    return ' '.join(words)

# def get_phonetics(f='librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'):


def build_librispeech_words_df(
        data_set='dev-clean',
        basepath='librispeech_alignments',
        audio_path='/mnt/c/Users/noe_t/Downloads/LibriSpeech/'
        ):
    path=os.path.join(basepath, data_set)
    files = glob(path+'/*/*/*.TextGrid')
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
            wav_path=os.path.join(audio_path,'/'.join(f.split('/')[1:]).split('.')[0]+'.flac')
            d={'word':word, 'phones':" ".join(phones), 'file_idx':f_idx, 'word_idx':i, 'start':start, 'end':end, 'path':f, 'wav_path':wav_path}
            # d={'word':word, 'phones':" ".join(phones), 'file_idx':f_idx, 'word_idx':i, 'start':start, 'end':end, 'path':f, 'sentence':get_sentence(f)}

            records.append(d)
    libri_words_df=pd.DataFrame.from_records(records)
    return libri_words_df

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


if __name__ == "__main__":
    libri_words_df=build_librispeech_words_df()

    libri_words_df[libri_words_df.word=='low']
    libri_words_df[libri_words_df.phones.str.endswith(' IH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' T IH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' T EH0 D')]
    libri_words_df[libri_words_df.phones.str.endswith(' D IH0 D')]

    libri_words_df[libri_words_df.phones.str.endswith(' V D')]
    libri_words_df[libri_words_df.phones.str.endswith(' M D')]
    libri_words_df[libri_words_df.phones.str.endswith(' NG D')]
    selection=libri_words_df[libri_words_df.phones.str.endswith(' DH D')]

    # if not os.path.exists('file_selection'): os.makedirs('file_selection')
    # for i,r in selection.iterrows():
    #     shutil.copy(r.wav_path, 'file_selection')

    libri_words_df[libri_words_df.phones.str.endswith('AO1')].word.unique()
    libri_words_df[libri_words_df.phones.str.contains('AO1')].word.unique()
    libri_words_df[libri_words_df.phones.str.contains('AO1')]
    libri_words_df[libri_words_df.phones.str.contains('OW1')]
    libri_words_df[libri_words_df.phones.str.endswith(' B D') & libri_words_df.word.str.endswith('bed')]
    libri_words_df[libri_words_df.phones.str.endswith(' D') & libri_words_df.word.str.endswith('ied')]


    libri_words_df[libri_words_df.phones.str.endswith(' P T') & libri_words_df.word.str.endswith('ped')]
    libri_words_df[libri_words_df.phones.str.endswith(' K T') & libri_words_df.word.str.endswith('ked')]
    libri_words_df[libri_words_df.phones.str.endswith(' SH T')]
    libri_words_df[libri_words_df.phones.str.endswith(' F T')]


    selection=libri_words_df[libri_words_df.phones.str.endswith(' P T') & libri_words_df.word.str.endswith('ped')]

    example=selection.iloc[0,:]

    get_phone_timings(f=example.path, word_idx=example.word_idx)


    # tg[0] -> words
    # tg[1] -> phones
    words=[el.mark for el in tg[0]]
    phones=[el.mark for el in tg[1]]