import os
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor


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


def synthesize(sentence, tag='prosody', options='rate="70%" volume="+20dB"',root_folder="synth_audio",synth_technique='standard',voice_id="Joanna", name="sample"):
    sentence=sentence.replace("'","&apos;").replace('"','&quot;')
    text=replace_with_tag(sentence, tag=tag, options=options)
    
    path='/'.join([root_folder,synth_technique,tag,voice_id])+'/'

    if not os.path.exists(path): os.makedirs(path)

    # reserved characters : https://docs.aws.amazon.com/polly/latest/dg/escapees.html
    
    # print('name',name)
    # print('text',text)

    cmd= "aws polly synthesize-speech \
    --text-type ssml \
    --text '"+text+"' \
    --output-format mp3 \
    --region us-east-1 \
    --voice-id "+voice_id+" \
    --engine "+synth_technique+" \
    "+path+name+".mp3  >/dev/null 2>&1"

    if not os.path.exists(path+name+".mp3"):    os.system(cmd)
    

def synthesize_cmu(root_folder="synth_audio/cmu_words", voice_id="Joanna", spk_id="F_US"):
    import cmudict
    from tqdm import tqdm
    words=list(cmudict.dict().keys())

    executor = ProcessPoolExecutor(max_workers=25)    
    futures = []
    for w in tqdm(words):
        name=spk_id+'_'+remove_special_characters(w, chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\']')
        # synthesize(w, root_folder=root_folder, synth_technique='standard',voice_id=voice_id, name=name)

        futures.append(executor.submit(
            synthesize, w, tag='prosody', options='rate="70%" volume="+20dB"',root_folder=root_folder,synth_technique='standard',voice_id=voice_id, name=name))

    proc_list = [future.result() for future in tqdm(futures)]

if __name__ == "__main__":
    # df=pd.read_csv('data/BE_PickStressedWord_1.csv')
    # col=df.iloc[:,1]

    # df=pd.read_csv('data/Business English-Vocabulary_all.csv')
    # col=df.iloc[:10,2]
    # df=pd.read_csv('../data/GE_linguistic_data_target_alternatives.csv')
    df=pd.read_csv("../data/AWS_audio_for_tutorials.csv")
    col=df.text
    root_folder="synth_audio/tutorials"
    synth_technique='neural' # "standard" or "neural"

    # tag=''
    # options=''

    tag='prosody'
    # options='rate="70%" volume="+20dB" pitch="+10%"'
    options='rate="70%" volume="+20dB"'

    # tag='emphasis'
    # options='level="strong"'


    voices={'Joanna':'F_US', "Amy":"F_UK", "Matthew":"M_US", "Brian":"M_UK"}

    for k in voices:
        voice_id=k
        spk_id=voices[k]
        synthesize_cmu(voice_id=voice_id, spk_id=spk_id)

    voice_id="Joanna"
    spk_id="F_US"

    # voice_id="Amy"
    # spk_id="F_UK"


    # voice_id="Matthew"
    # spk_id="M_US"

    # voice_id="Brian"
    # spk_id="M_UK"

    for i,sentence in col.iteritems():
        # name='_'.join(remove_special_characters(sentence).split(' ')).replace('*','+')
        # name=spk_id+'_'+df.iloc[i, df.columns.get_loc('id')]
        name=df.iloc[i, df.columns.get_loc('id')]
        print(name)
        synthesize(sentence, tag=tag, options=options,root_folder=root_folder,synth_technique=synth_technique,voice_id=voice_id,name=name)

