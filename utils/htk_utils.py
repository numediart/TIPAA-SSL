import os
import pandas as pd
import subprocess

def clean_htk_files(rand_fileName):
    """Clean the files generated for and by the HTK model (as it uses input and output files)
    In the future, just delete the specific files after processing, by putting name in parameters (inputs and results files)
    """
    print('rand_fileName',rand_fileName)
    input_extensions=['.net', '.wav', '.rec', '.dct', '.txt']
    # input_extensions=['.wav']
    output_extensions=['.rec']

    for ext in input_extensions:
        f='inputs/'+rand_fileName+ext
        if os.path.exists(f):
            os.system('rm '+f)
    for ext in output_extensions:
        f='results/'+rand_fileName+ext
        if os.path.exists(f):
            os.system('rm '+f)

# HTK related functions
def process_grammar(rand_fileName):
    """process grammar of a sentence

    Args:
        inputGrammar (string): path of grammar file
        rand_fileName (string): filename of the output of HParse

    Returns:
        int: 0 if success, -1 otherwise
    """    
    cmd1 = 'HParse ' +'./inputs/'+ rand_fileName +'.txt' +' ./inputs/'+ rand_fileName +'.net'
    a=os.system(cmd1)
    return a

def htk_recognition(modelName, rand_fileName, wav_name):
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
    
    inputPhoneticTranscription='./inputs/' +rand_fileName+ '.dct'

    cmd2 = 'HVite -A -T 1 -a -C ./model/' +modelName+ '/Align.cfg -H ./model/' +modelName+ '/hmm-mono -H \
./model/generalSpeech/hmm-gs_1 -H ./model/generalSpeech/hmm-gss_2 -H ./model/generalSpeech/hmm-gss_3 -H ./model/generalSpeech/hmm-gss_4 \
-H ./model/generalSpeech/hmm-gss_5 -w ./inputs/'+ rand_fileName+ '.net -l ./results -o N ' +inputPhoneticTranscription+ ' ./model/' +modelName+ '/monophones \
./inputs/' +wav_name+ '.wav'

    # out2 = os.popen(cmd2).read()
    out2=subprocess.run(cmd2.split(' '), capture_output=True)
    # out2.stdout
    print('stderr',out2.stderr)
    # out2=os.system(cmd2)
    # print('out2', out2)
    if out2.stderr.decode('utf-8')!='':
        return [], out2
    # read rec file (i.e., the alignment outcome)
    textgridData = pd.read_csv('./results/'+ wav_name +'.rec', sep=' ', header=None)

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

def get_textgrid_data(rand_fileName, wav_name, modelName = 'libri'):
    """This function write necessary files to then call HMM model (htk_recognition) and the filter out silences

    Args:
        rand_fileName ([type]): [description]
        modelName (str, optional): [description]. Defaults to 'libri'.

    Returns:
        DataFrame: data of duration and phonetic characteristics of words or phonemes 
    """
    inputPhoneticTranscription='./inputs/' +rand_fileName+ '.dct'

    textgridData, out2=htk_recognition(modelName, rand_fileName, wav_name)

    
    if out2.stderr.decode('utf-8')!='':
        return [], out2

    # filter out silences
    textgridData=textgridData[(textgridData.iloc[:,2]!='sil')&(textgridData.iloc[:,2]!='sp')]

    # reset indices of dataframe
    textgridData.index=range(len(textgridData))
    
    phonetics=pd.read_csv(inputPhoneticTranscription, header=None, sep='(\] |\[)', engine='python')

    detected_transcription=[]
    for r in textgridData.iloc[:,2]:
        corresponding_transcription=phonetics[phonetics.iloc[:,2]==r][4]
        if len(corresponding_transcription)>0:
            detected_transcription.append(corresponding_transcription.values[0])
        else:
            # print('out of vocabulary: no transcription')
            detected_transcription.append('out_of_vocabulary')

    # if len(textgridData)!=len(detected_transcription):
    #     print("part or all the phrase was not recognized in expected phonemes")
    #     return [], out2
    
    textgridData['detected_transcription']=detected_transcription
    # print(textgridData)
    clean_htk_files(wav_name)
    return textgridData, out2
