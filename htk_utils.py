import uuid
from scipy.io.wavfile import read, write
import numpy as np
import os
import pandas as pd


def clean_htk_files(p):
    """Clean the files generated for and by the HTK model (as it uses input and output files)
    TODO: This is probably dangerous when we use it in parallel, multithreading... 
    In the future, just delete the specific files after processing, by putting name in parameters (inputs and results files)
    """
    os.system('rm inputs/'+p['rand_fileName']+'*')
    os.system('rm results/'+p['rand_fileName']+'*')

# HTK related functions
def process_grammar(inputGrammar, rand_fileName):
    """process grammar of a sentence

    Args:
        inputGrammar (string): path of grammar file
        rand_fileName (string): filename of the output of HParse

    Returns:
        int: 0 if success, -1 otherwise
    """    
    cmd1 = 'HParse ' +inputGrammar +' ./inputs/'+ rand_fileName +'.net'
    a=os.system(cmd1)
    return a

def htk_recognition(modelName, rand_fileName, inputPhoneticTranscription):
    """This function uses an HMM model to do speech recognition or forced alignment. 
    It uses an HTK command that output a result file, then reads this file and filter some information.

    Args:
        modelName (string): name of HMM model
        rand_fileName (string): temp filname
        inputPhoneticTranscription (string): input phonetic transcription for forced alignment (I think)

    Returns:
        dataframe: timings of start and end of phonemes (or words ?)
        out2: std output of the command
    """
    cmd2 = 'HVite -A -T 1 -a -C ./model/' +modelName+ '/Align.cfg -H ./model/' +modelName+ '/hmm-mono -H \
    ./model/generalSpeech/hmm-gs_1 -H ./model/generalSpeech/hmm-gss_2 -H ./model/generalSpeech/hmm-gss_3 -H ./model/generalSpeech/hmm-gss_4 \
    -H ./model/generalSpeech/hmm-gss_5 -w ./inputs/'+ rand_fileName+ '.net -l ./results -o N ' +inputPhoneticTranscription+ ' ./model/' +modelName+ '/monophones \
    ./inputs/' +rand_fileName+ '.wav'

    out2 = os.popen(cmd2).read()

    # out2=os.system(cmd2)
    print('out2', out2)

    # read rec file (i.e., the alignment outcome)
    textgridData = pd.read_csv('./results/'+ rand_fileName +'.rec', sep=' ', header=None)

    # convert timings in seconds
    textgridData.iloc[:,0]/=10000000
    textgridData.iloc[:,1]/=10000000

    # remove phones with zero duration (e.g., sp)
    textgridData=textgridData[textgridData.iloc[:,0]!=textgridData.iloc[:,1]]

    # compensate a 10ms biased in the results
    if (textgridData.iloc[:,1].iloc[-1]-textgridData.iloc[:,0].iloc[-1])>0.01:
        textgridData.iloc[:,0].iloc[1:] += 0.01
        textgridData.iloc[:,1].iloc[:-1] += 0.01

    return textgridData, out2

def get_textgrid_data(s,fs,p):
    """This function write necessary files to then call HMM model (htk_recognition) and the filter out silences

    Args:
        s (numpy array): signal waveform of the audio recording
        fs (int): frequency of sampling of s
        p (dict): parameters (paths of grammar, phonetics and files etc.)

    Returns:
        DataFrame: data of duration and phonetic characteristics of words or phonemes 
    """    
    rand_fileName = p['rand_fileName']
    write('./inputs/'+ rand_fileName+ '.wav', fs, (s*32767).astype(np.int16))

    process_grammar(p['inputGrammar'], rand_fileName)

    textgridData, out2=htk_recognition(p['modelName'], rand_fileName, p['inputPhoneticTranscription'])

    # filter out silences
    textgridData=textgridData[(textgridData.iloc[:,2]!='sil')&(textgridData.iloc[:,2]!='sp')]

    # reset indices of dataframe
    textgridData.index=range(len(textgridData))
    clean_htk_files(p)
    return textgridData, out2

