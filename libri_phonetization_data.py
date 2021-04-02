import cmudict
import textgrid
# doc: https://github.com/kylebgorman/textgrid
from itertools import compress

from tqdm import tqdm
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

data_set='dev-clean'
import os
path=os.path.join('librispeech_alignments', data_set)

audio_path='/mnt/c/Users/noe_t/Downloads/LibriSpeech/'

from glob import glob
files = glob(path+'/*/*/*.TextGrid')

f=files[0]

records=[]
for f_idx,f in tqdm(enumerate(files)):
    tg = textgrid.TextGrid.fromFile(f)
    tg_words=tg[0]
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
        d={'word':word, 'phones':" ".join(phones), 'file_idx':f_idx, 'word_idx':i, 'start':start, 'end':end, 'path':f}
        records.append(d)
libri_words_df=pd.DataFrame.from_records(records)

libri_words_df[libri_words_df.word=='moved']
libri_words_df[libri_words_df.phones.str.endswith('IH0 D')]
libri_words_df[libri_words_df.phones.str.endswith('T IH0 D')]
libri_words_df[libri_words_df.phones.str.endswith('D IH0 D')]
libri_words_df[libri_words_df.phones.str.endswith('V D')]
libri_words_df[libri_words_df.phones.str.endswith('M D')]

libri_words_df[libri_words_df.phones.str.endswith('P T') & libri_words_df.word.str.endswith('ped')]
libri_words_df[libri_words_df.phones.str.endswith('K T') & libri_words_df.word.str.endswith('ked')]
libri_words_df[libri_words_df.phones.str.endswith('SH T')]

f_idx=1252
word_idx=8
tg = textgrid.TextGrid.fromFile(files[f_idx])
tg[0][word_idx]

filter=[tg[0][word_idx].overlaps(el) for el in tg[1]]
list(compress(tg[1], filter))

# tg[0] -> words
# tg[1] -> phones
words=[el.mark for el in tg[0]]
phones=[el.mark for el in tg[1]]