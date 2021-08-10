import os
import pandas as pd
import re

def remove_special_characters(sentence="Where's the best place to have coffee ?", lowercase=True, chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'):
    """Normalize text by lowercasing (if option is True), and remove a set of punctuation characters

    Args:
        sentence (str, optional): [description]. Defaults to "Where's the best place to have coffee ?".
        lowercase (bool, optional): [description]. Defaults to True.
        chars_to_ignore_regex (str, optional): [description]. Defaults to '[\,\?\.\!\-\;\:\"]'.

    Returns:
        str: normalized sentence
    """
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    sentence = re.sub(chars_to_ignore_regex, '', sentence)

    if lowercase:
        sentence=sentence.lower()

    # This is to make sure there will not be empty strings after a splitting. So here I split, remove Nones, and rejoin
    sentence=' '.join(list(filter(None, sentence.split(' '))))
    return sentence


# df=pd.read_csv('data/BE_PickStressedWord_1.csv')
# col=df.iloc[:,1]


df=pd.read_csv('data/Business English-Vocabulary_all.csv')
col=df.iloc[:10,2]

def replace_with_tag(sentence, tag='emphasis', options='level="strong"'):
    idx=0
    tag_start=True #binary
    idx=sentence.find('*')
    while idx!=-1:
        if tag_start:
            sentence=sentence[:idx]+'<'+tag+' '+options+'>'+sentence[idx+1:]
        else:
            sentence=sentence[:idx]+'</'+tag+'>'+sentence[idx+1:]
        tag_start=not tag_start
        idx=sentence.find('*')

    sentence="<speak>"+sentence+"</speak>"
    return sentence


def synthesize(sentence, tag='prosody', options='rate="70%" volume="+20dB"',root_folder="synth_audio",synth_technique='standard',voice_id="Joanna"):
    text=replace_with_tag(sentence, tag=tag, options=options)
    name='_'.join(remove_special_characters(sentence).split(' ')).replace('*','+')
    path='/'.join([root_folder,synth_technique,tag,voice_id])+'/'

    if not os.path.exists(path): os.makedirs(path)

    print('name',name)
    print('text',text)

    cmd= "aws polly synthesize-speech \
    --text-type ssml \
    --text '"+text+"' \
    --output-format mp3 \
    --voice-id "+voice_id+" \
    --engine "+synth_technique+" \
    "+path+name+".mp3"

    os.system(cmd)
    

root_folder="synth_audio"
synth_technique='standard'

# tag=''
# options=''

tag='prosody'
options='rate="70%" volume="+20dB pitch="+10%"'
# options='rate="70%" volume="+20dB"'

# tag='emphasis'
# options='level="strong"'

voice_id="Joanna"

for sentence in col:
    synthesize(sentence, tag=tag, options=options,root_folder=root_folder,synth_technique=synth_technique,voice_id=voice_id)

