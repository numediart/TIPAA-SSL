import cmudict
import textgrid

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

from glob import glob
files = glob(path+'/*/*/*.TextGrid')

f=files[0]

# doc: https://github.com/kylebgorman/textgrid
tg = textgrid.TextGrid.fromFile(f)

# tg[0] -> words
# tg[1] -> phones

words=[el.mark for el in tg[0]]
phones=[el.mark for el in tg[1]]