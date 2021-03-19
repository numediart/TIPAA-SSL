import uuid
from scipy.io.wavfile import read, write
import librosa
import numpy as np
import os
import pandas as pd
import pyworld as pw
import matplotlib.pyplot as plt
import cmudict
import pytsmod as tsm
import soundfile as sf

def clean_temp_files():
    os.system('rm inputs/*')
    os.system('rm results/*')

def set_params(
    # waveFileAddress='/root/flowchase/sent.wav',
    waveFileAddress="/media/adminpc/DATA/databases/AmuS_clean/SpkA/neutral/wav/SpkA_pr_0001.wav",
    sentenceID=1,
    keep_logs = 0, # 1 if we want to keep logs of all processed files (e.g., for troubleshooting), 0 otherwise
    fs_target = 16000, # the target sampling frequency
    binResult = [],
    status = -1,
    result = [],
    modelName = 'libri',
    module='sentenceStress'):

    inputPhoneticTranscription_base = './lexicon/'+module+'/dct/phrase_'
    inputGrammar_base = './lexicon/'+module+'/grammar/phrase_'
    
    inputGrammar = '%s%d.txt' % (inputGrammar_base, sentenceID)
    inputPhoneticTranscription = '%s%d.dct' % (inputPhoneticTranscription_base, sentenceID)
    params={}
    params['waveFileAddress']=waveFileAddress
    params['sentenceID']=sentenceID
    params['keep_logs']=keep_logs
    params['fs_target']=fs_target
    params['inputPhoneticTranscription_base']=inputPhoneticTranscription_base
    params['inputGrammar_base']=inputGrammar_base
    params['binResult']=binResult
    params['status']=status
    params['result']=result
    params['inputPhoneticTranscription']=inputPhoneticTranscription
    params['inputGrammar']=inputGrammar
    params['modelName']=modelName

    return params

def createJSON(binResult, status, rand_fileName):
    """prepare json output message

    Args:
        binResult (): Result to send
        status (int): 0 if nothing went wrong, other values (negative) if not
        rand_fileName (string): random filename for temp files

    Returns:
        string: JSON string
    """
    
    if status != 0:
        res = '{"status": "%d"}'% status
        '%s%d.dct' % (inputPhoneticTranscription_base, sentenceID)
    else:
        oneString = '%d,' % binResult # string with all values separated by a comma
        res = '{"status": "%d", "sequence of words and pauses (marked as zeros)": "%s"}'% (status, oneString[:-1]) # (end-1): strip final comma
    
    return res
    
def load_audio(waveFileAddress):
    """Load audio, remove DC and normalize waveform

    Args:
        waveFileAddress (string): wav file address

    Returns:
        numpy array, int: waveform signal and frequency of sampling
    """
    # fs, s = read(waveFileAddress)
    s,fs=librosa.load(waveFileAddress, sr=16000)

    #trim silences
    s, index = librosa.effects.trim(s, top_db=20)
    # remove DC
    s = s - s[int(0.15*len(s)):int(0.85*len(s))].mean() # we exclude 15% at each side that might contain buffer initialization/release noises
    # normalization
    s = 0.90*s/max(abs(s))
    return s, fs

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

    # r='/home/adminpc/Dropbox/contracts_info/flowchase/CONFIDENTIAL_Flowchase_transfer/CONFIDENTIAL_Flowchase_transfer/octave_analysis_modules/sentenceStress'
    # rand_fileName2='oct-tGp3LF'

    # cmd2 = 'HVite -A -T 1 -a -C ./model/' +modelName+ '/Align.cfg -H ./model/' +modelName+ '/hmm-mono -H \
    # ./model/generalSpeech/hmm-gs_1 -H ./model/generalSpeech/hmm-gss_2 -H ./model/generalSpeech/hmm-gss_3 -H ./model/generalSpeech/hmm-gss_4 \
    # -H ./model/generalSpeech/hmm-gss_5 -w ./inputs/'+ rand_fileName+ '.net -l ./results -o N ' +inputPhoneticTranscription+ ' ./model/' +modelName+ '/monophones \
    # '+r+'/inputs/' +rand_fileName2+ '.wav'

    # cmd2 = 'HVite -A -T 1 -a -C ./model/' +modelName+ '/Align.cfg -H ./model/' +modelName+ '/hmm-mono -H \
    # ./model/generalSpeech/hmm-gs_1 -H ./model/generalSpeech/hmm-gss_2 -H ./model/generalSpeech/hmm-gss_3 -H ./model/generalSpeech/hmm-gss_4 \
    # -H ./model/generalSpeech/hmm-gss_5 -w ./inputs/'+ rand_fileName+ '.net -l ./results -o N ./inputs/' +rand_fileName+ '.dct ./model/' +modelName+ '/monophones \
    # ./inputs/' +rand_fileName+ '.wav'


    # cmd2 = 'HVite -A -T 1 -a -C '+r+'/model/' +modelName+ '/Align.cfg -H '+r+'/model/'+ modelName+ '/hmm-mono -H \
    # '+r+'/model/generalSpeech/hmm-gs_1 -H '+r+'/model/generalSpeech/hmm-gss_2 -H '+r+'/model/generalSpeech/hmm-gss_3 -H '+r+'/model/generalSpeech/hmm-gss_4 \
    # -H '+r+'/model/generalSpeech/hmm-gss_5 -w '+r+'/inputs/'+ rand_fileName2 +'.net -l '+r+'/results -o N '+r+'/'+ inputPhoneticTranscription +' '+r+'/model/'+ modelName+ \
    # '/monophones '+r+'/inputs/' +rand_fileName2+ '.wav'

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
    rand_fileName = str(uuid.uuid4())
    write('./inputs/'+ rand_fileName+ '.wav', fs, (s*32767).astype(np.int16))

    process_grammar(p['inputGrammar'], rand_fileName)

    textgridData, out2=htk_recognition(p['modelName'], rand_fileName, p['inputPhoneticTranscription'])

    # filter out silences
    textgridData=textgridData[(textgridData.iloc[:,2]!='sil')&(textgridData.iloc[:,2]!='sp')]

    # reset indices of dataframe
    textgridData.index=range(len(textgridData))

    return textgridData, out2


# signal processing (pitch, instensity, normalization...)
def getIntonation(s, fs):
    #  the /2**15  is for converting 16bit PCM to double between -1 and 1
    f0, sp, ap = pw.wav2world(s.astype(np.float64)/2**15, fs)

    # convert in semitones
    f0Frames=40*np.log10(f0)

    # replace any inf due to the log operation with nan (which are treaded below)
    f0Frames[f0Frames==-np.inf]=np.nan

    from scipy.interpolate import interp1d

    x=np.arange(len(f0Frames))
    f = interp1d(x, f0Frames, kind='linear')

    xnew = np.floor(np.arange(len(s))/len(s)*len(f0Frames))
    f0Samples = f(xnew)

    # replace nans with minimum value
    f0Samples=np.nan_to_num(f0Samples, nan= np.nanmin(f0Samples))

    # from scipy import signal
    # ynew = signal.resample(f0Frames, len(s))

    return f0Samples

def smooth(x,beta, window_len=11):
    """ kaiser window smoothing """
    # from https://dzone.com/articles/computing-convolution-using

    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    # return y[5:len(y)-5]
    
    # replace nans with minimum value
    y=np.nan_to_num(y, nan= np.nanmin(y))

    return y[int(window_len/2):-int(window_len/2)+1]

def getIntensity(s, fs):

    # this is done similarly to praat:
    minimum_pitch = 70; # in Hz
    analysis_win = round((3.2/minimum_pitch)*fs)

    # smoothing the signal power using a moving average
    #  the /2**15  is for converting 16bit PCM to double between -1 and 1

    intensity=smooth((s/2**15)**2, 20, 2*analysis_win)

    # convert in db
    intensity = intensity / 4.0e-10
    int_db = 10*np.log10(intensity)

    #  remove any inf due to the log operation and replace them with the minimum value of intensity
    # int_db(isinf(int_db)) = min(int_db(~isinf(int_db)));
    return int_db

def normalize(x):
    y=x-min(x)
    return y/max(y)

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter

def slow_down(input_filename='audio_recordings/WS_111_toothpaste.wav', output_filename='audio_recordings/WS_111_toothpaste_slow2.wav', fs=1600, speed_rate=0.7):
    
    # Note: don't know why, but this commented approach does not work...
    # x,fs=librosa.load(input_filename, sr=fs)
    # path=np.array([[0,len(x)],[0,int(len(x)/speed_rate)]])
    # x_speed_rate = tsm.wsola(x, path)
    # sf.write(output_filename, x_speed_rate,  fs)
    
    with WavReader(input_filename) as reader:
        with WavWriter(output_filename, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=speed_rate)
            tsm.run(reader, writer)



def align_audios(reference, recording, fs=16000):
    """Performs Dynamic Time Warping on the mel-spectrograms of a reference audio and a recording
    from a user. It computes a path of alignments of timings. This path is then used 
    for a time stretching of the user resording. The 2 audio can then be summed to have a resulting audio
    in which we can hear both voices at the same time

    Args:
        reference (numpy array): waveform of reference
        recording (numpy array): waveform of user recording
        fs (int, optional): frequency of sampling. Defaults to 16000.

    Returns:
        numpy array: waveform of the merged audio that are time-aligned
    """

    # reference,fs=librosa.load('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', sr=16000)
    # recording,fs=librosa.load('audio_recordings/SS_1_i_would_love_to_go_to_ireland_FR.wav', sr=16000)

    # reference=normalize(reference)*0.7
    # recording=normalize(recording)*0.7
    ref_mel=librosa.feature.melspectrogram(y=reference, sr=fs)
    rec_mel=librosa.feature.melspectrogram(y=recording, sr=fs)
    D, wp = librosa.sequence.dtw(X=ref_mel, Y=rec_mel)

    s_ap=(wp[::-1].T*len(reference)/ref_mel.shape[-1]).astype(int)

    recording_aligned = tsm.wsola(recording, s_ap[::-1])

    max_len=max(len(recording_aligned), len(reference))

    def pad_zeros_to_len(a, length):
        return np.pad(a, (0, (length-len(a))), 'constant', constant_values=(0, 0))
    
    reference=pad_zeros_to_len(reference, max_len)
    recording_aligned=pad_zeros_to_len(recording_aligned, max_len)
    out=recording_aligned+reference

    # sf.write('audio_recordings/SS_1_i_would_love_to_go_to_ireland_merge.wav', out,  fs)

    return out



# Modules
def chunking():
    p=set_params(module='chunking')

    s,fs = load_audio(p['waveFileAddress'])

    rand_fileName = str(uuid.uuid4())
    # write('./inputs/'+ rand_fileName+ '.wav', fs, (s*32767).astype(np.int16))
    write('./inputs/'+ rand_fileName+ '.wav', fs, s)

    process_grammar(p['inputGrammar'], rand_fileName)

    textgridData, out2=htk_recognition(p['modelName'], rand_fileName, p['inputPhoneticTranscription'])

    minSilDur = 0.090 # we ask for at least 90ms of silence

    silence_durations=textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,1]-textgridData[textgridData.iloc[:,2]=='sil'].iloc[:,0]
    idx_to_filter=silence_durations[silence_durations>minSilDur].index

    status=0

def sentenceStress(p=set_params(sentenceID=1, waveFileAddress='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', module='sentenceStress')):
    s,fs = load_audio(p['waveFileAddress'])
    textgridData, cmdout2=get_textgrid_data(s,fs,p)

    # find the number of words and phonemes per word in the input phrase
    # the '_' delimits the information of the word and number of phonemes, e.g. for "w2_7", there are 7 phonemes
    # if there is no '_', it is e.g. "o4". We put 1 in that case and else, the number of phonemes
    l=[el.split('_') for el in textgridData.iloc[:,2].tolist()]
    phonemesPerWord=[1 if len(el)==1 else int(el[-1]) for el in l]

    # each word start and end position expressed in samples
    startPositions_samples = (round(fs*textgridData.iloc[:,0])+1).astype(int)
    stopPositions_samples = round(fs*textgridData.iloc[:,1]).astype(int)

    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples.iloc[-1]<len(s), "The end of the last phoneme should be inside the signal"

    # TODO: verification of alignment

    # f0Samples=getIntonation(s, fs)
    f0Samples=getIntonation(s.astype(np.float64), fs)
    intensity=getIntensity(s, fs)

    # plt.plot(f0Samples)
    # plt.show()
    
    # plt.plot(intensity)
    # plt.show()

    Dur=(textgridData.iloc[:,1]-textgridData.iloc[:,0])/(np.array(phonemesPerWord)+1)  # +1 assuming stressed phonemes = 2*other phonemes
    Dur=np.array(Dur.tolist())
    # plt.plot(Dur)
    # plt.show()

    startPositions_samples=startPositions_samples.tolist()
    stopPositions_samples=stopPositions_samples.tolist()
    
    Imax=np.zeros(len(textgridData))
    Fmax=np.zeros(len(textgridData))
    for i in range(len(textgridData)):
        if phonemesPerWord[i] == 1:
            Dur[i] = 0.6 * Dur[i]
        elif phonemesPerWord[i] == 2:
            Dur[i] = 0.8 * Dur[i]
        temp_sort = sorted(intensity[startPositions_samples[i]:stopPositions_samples[i]], reverse = True)
        Imax[i] = np.median(temp_sort[0:np.round(0.05*len(temp_sort)).astype(int)])
        temp_sort = sorted(f0Samples[startPositions_samples[i]:stopPositions_samples[i]], reverse = True)
        Fmax[i] = np.median(temp_sort[0:np.round(0.05*len(temp_sort)).astype(int)])
    
    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zFmax = normalize(Fmax)
    zDur = normalize(Dur)

    # plt.plot(zImax)
    # plt.show()
    
    # plt.plot(zFmax)
    # plt.show()

    # combine the features into a final result
    # weighted_score = (0.6*zImax + 0.4*zFmax + 0.2*zDur)/1.2;
    weighted_score = (0.6*zImax + 0.4*zFmax)/1.0 # needs fine-tuning once enough user data are available - in the long term consider additional futures and train a classifier with annotated user data
    rateThreshold = 1.01
    nWords=len(textgridData)
    binResult = np.zeros(nWords)
    
    plt.plot(weighted_score)
    plt.savefig('sentence_curve.png')

    if nWords == 1:
        binResult[0] = 1
    elif nWords == 2:
        sWS_id=np.argsort(weighted_score)[::-1]
        sWS_val=weighted_score[sWS_id]
        # TODO : I have to check if this make any sense. 
        # in the case with only two words, we ckeck if the higher is at least 1% higher than the other and put 1 there... (why this 1% ?)
        if sWS_val[0] > rateThreshold*sWS_val[1]:
            binResult[sWS_id[0]] = 1
    else:
        #TODO : this mean score has to be adapted because he use a normalization that led to values in a small range such
        # meanScore = 0.90*max(weighted_score)
        # meanScore= weighted_score.mean()
        meanScore=0.2
        if (weighted_score[0] > rateThreshold*weighted_score[1]) and (weighted_score[0] > meanScore):
            binResult[0] = 1
        elif (weighted_score[1] > rateThreshold*max(weighted_score[[0, 2]])) and (weighted_score[1] > meanScore):
            binResult[1] = 1
        elif (weighted_score[-1] > rateThreshold*weighted_score[-2]) and (weighted_score[-1] > meanScore):
            binResult[-1] = 1
        if nWords > 3:
            for i in range(2,nWords-1):
                if (weighted_score[i] > rateThreshold*max(weighted_score[[i-1, i+1]])) and (weighted_score[i] > meanScore):
                    binResult[i] = 1
    
    return binResult
        
def wordStress(p=set_params(sentenceID=111, waveFileAddress='audio_recordings/WS_111_toothpaste.wav', module='wordStress')):
    s,fs = load_audio(p['waveFileAddress'])
    textgridData, cmdout2=get_textgrid_data(s,fs,p)

    # TODO: verification of alignment
    # TODO: check if all phonemes were found in the speech signal
    nEntries = len(textgridData)

    # get the line showing phoneme sequence
    phonetics=pd.read_csv(p['inputPhoneticTranscription'], header=None)

    dict_phones={}
    for i,r in phonetics.iterrows():
        #lines startint by d correspond to phonemes
        if r[0][0]=='d':
            line=r.values[0].split(' ')
            phone=line[0]
            info=line[1][1:-1]
            # print(phone)
            # print(info)
            dict_phones[info]=phone

    phone_seq=[dict_phones[r[2]] for i,r in textgridData.iterrows()]
    phone_set=set(phone_seq)
    count_total = len(phone_seq)
    count_unique = len(phone_set)

    # TODO: use these for verification 


    #extract the number of words by taking the index of the last one
    nWords=int(textgridData[2].iloc[-1].split('_')[0][1:])

    # for each entry, the second column is something like "w1_v1_1" or "w1_g1_1". the v is for vowel.
    # I tranform that to a table
    phone_df=pd.DataFrame([r[2].split('_') for i,r in textgridData.iterrows()])

    nVowelsPerWord=np.zeros(nWords, dtype=int)
    # for each entry of textgridData, we look at the word index, and if the phone is a vowel, we increment its number of vowels
    for i,r in phone_df.iterrows():
        word_index=int(r[0][1:])-1
        phone_type=r[1][0]
        # print(word_index)
        # print(phone_type)
        if phone_type=='v':
            nVowelsPerWord[word_index]+=1

    is_vowel=[r[2].split('_')[1][0]=='v' for i,r in textgridData.iterrows()]
    indxVowels = [i for i, val in enumerate(is_vowel) if val] 
    nVowels=len(indxVowels)

    # TODO: I do not know what this is but this is set to 1 when vowel ELSE corresponds to the integer at the end of "w1_v1_1"
    phoPerEntry=[int(r[2].split('_')[-1]) for i,r in textgridData.iterrows()]
    
    # TODO: check words duration
    # TODO: check acoustic scores

    

    # % check words duration
    # potErrors = 0;
    # grosErrors = 0;
    # for i = 1:nEntries
    #     if (textgridData{2}(i) - textgridData{1}(i) < 0.040*phoPerEntry(i)) || (textgridData{2}(i) - textgridData{1}(i) > 0.230*phoPerEntry(i))
    #         potErrors = potErrors + 1;
    #     end
    #     if textgridData{2}(i) - textgridData{1}(i) > 0.450*phoPerEntry(i)
    #         grosErrors = grosErrors + 1;
    #     end
    # end
    # if doVerification == 1 && (potErrors > 0.5*nEntries || grosErrors > 0)
    #     status = -500;
    #     result = createJSON(binResult, status, rand_fileName);
    #     return;
    # end

    # % check acoustic scores
    # if doVerification == 1 && mean(textgridData{4}) < 8.5
    #     status = -600;
    #     result = createJSON(binResult, status, rand_fileName);
    #     return;
    # end


    f0Samples=getIntonation(s, fs)
    intensity=getIntensity(s, fs)

    # extract features

    # each word start and end position expressed in samples
    startPositions_samples = (round(fs*textgridData.iloc[:,0])+1).astype(int).tolist()
    stopPositions_samples = round(fs*textgridData.iloc[:,1]).astype(int).tolist()
    
    # to make sure we don t go beyond the end of the signal
    assert stopPositions_samples[-1]<len(s), "The end of the last phoneme should be inside the signal"

    Imax,Imean,Fmax,Fmean,Dur=[],[],[],[],[]
    sylType=np.zeros(nVowels)
    for i in range(nVowels):
        range_vowel=range(startPositions_samples[indxVowels[i]], stopPositions_samples[indxVowels[i]])
        Ivowel=intensity[range_vowel]
        Fvowel=f0Samples[range_vowel]
        
        Imax.append(max(Ivowel))
        Imean.append(np.mean(Ivowel))
        Fmax.append(max(Fvowel))
        Fmean.append(np.mean(Fvowel))

        Dur.append(textgridData[1].iloc[indxVowels[i]]-textgridData[0].iloc[indxVowels[i]])

        textgridData[2].iloc[indxVowels[i]]

        syltype_phone=int(phone_df[2].iloc[indxVowels[i]])
        if syltype_phone == 2:  # the sylType is 0 for unstressed, 0.5 for secondary stressed syllables and 1 for primary stressed syllables
            sylType[i] = 0.5
        else:
            sylType[i]=syltype_phone
    
    # normalization of features (projection to [0 1] range)
    zImax = normalize(Imax)
    zImean = normalize(Imean)
    zFmax = normalize(Fmax)
    zFmean = normalize(Fmean)
    zDur = normalize(Dur)

    # combine the features
    weighted_score = (zImax + 0.2*zImean + zFmax + 0.2*zFmean + 0.8*zDur + 0.4*sylType)/3.6  # needs fine-tuning once enough user data are available - in the long term train a classifier with annotated user data

    plt.plot(weighted_score)
    plt.savefig('word_curve.png')
    # chose prominent vowel per word
    binResult=np.zeros(len(weighted_score))
    for i in range(nWords):
        # make the indices range of syllables (or vowels) in the sentence
        if i == 0:
            syl_id = np.arange(nVowelsPerWord[i])
        else:
            syl_id = np.arange(nVowelsPerWord[i])+sum(nVowelsPerWord[:i])
        maxscore = max(weighted_score[syl_id]) # the max score over a word
        for j in range(len(syl_id)):
            if weighted_score[syl_id[j]] == maxscore:
                binResult[syl_id[j]] = 1
            elif weighted_score[syl_id[j]] >= 0.98*maxscore: # syllables close to the max score (i.e., 98% of its value) are also considered as stressed
                binResult[syl_id[j]] = 1
            else:
                binResult[syl_id[j]] = 0
    
    return binResult

def iConstrast(p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")):
    
    #p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_sleep.wav', module="iContrast")
    s,fs = load_audio(p['waveFileAddress'])
    textgridData, cmdout2=get_textgrid_data(s,fs,p)

    # find the number of words and phonemes per word in the input phrase
    # the '_' delimits the information of the word and number of phonemes, e.g. for "w2_7", there are 7 phonemes
    # if there is no '_', it is e.g. "o4". We put 1 in that case and else, the number of phonemes
    # l=[el.split('_') for el in textgridData.iloc[:,2].tolist()]
    # phonemesPerWord=[1 if len(el)==1 else int(el[-1]) for el in l]

    def number_and_indices(char='w'):
        #take index when the first character is char
        idxs=[i if el[0]==char else np.nan for i,el in enumerate(textgridData.iloc[:,2].tolist())]
        # remove nans (x=nan is the only value such that x!=x)
        idxs = [x for x in idxs if x==x]
        n=len(idxs)
        return n, idxs
    # TODO: this is for the verification and it is not finished
    nWords, indxWords = number_and_indices('w')
    nPho, indxPho = number_and_indices('p')
    nSil, indxSil = number_and_indices('s')

    # TODO: weird, he does a loop for every phoneme, check if it is a short or long, and repeat, but does not record results.
    # only the last result will be kept (maybe it works because there is only one vowel that needs to be checked)
    for i,el in enumerate(textgridData.iloc[:,2].tolist()):
        print(el[-2:])
        if el[-2:]=='*s':
            binResult=0
        elif el[-2:]=='*l':
            binResult=1
            if textgridData.iloc[i, 1]-textgridData.iloc[i, 0]<0.07:
                binResult=0
    
    return binResult

def oConstrast():#p=set_params(sentenceID=111, waveFileAddress='audio_recordings/iC_111_slip.wav', module="iContrast")):
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Laaw_MP3.mp3', module="oContrast")
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Laaw_WAV.wav', module="oContrast")
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Law_WAV.wav', module="oContrast")
    p=set_params(sentenceID=111, waveFileAddress='audio_recordings/Low_WAV.wav', module="oContrast")
    s,fs = load_audio(p['waveFileAddress'])
    textgridData, cmdout2=get_textgrid_data(s,fs,p)


def edAnalysis():
    p=set_params(sentenceID=1, waveFileAddress='audio_recordings/ed_accepted.wav', module="edAnalysis")
    s,fs = load_audio(p['waveFileAddress'])
    textgridData, cmdout2=get_textgrid_data(s,fs,p)

# Data processing
def get_data(path_to_json='../audio-with-analysis-ids/data.json'):
    """Get a dataframe containing info of audio recordings with sentence ids

    Args:
        path_to_json (str, optional): [description]. Defaults to '../audio-with-analysis-ids/data.json'.

    Returns:
        DataFrame: [description]
    """
    data=pd.read_json(path_to_json)
    return data

def get_wordStress_annotation(path='../audio-with-analysis-ids/wordStress_annotations.csv'):
    d=get_data()
    d=d[d.focusType=="wordstress"]

    df=pd.read_csv(path)

    wordStress_annotations={}
    for i,r in d.iterrows():
        analysisId=r.analysisId
        first_digit=int(analysisId/100)

        if first_digit>0:
            col=df[str(first_digit)]

            text=d[d.analysisId==analysisId].text.values[0]

            col_idx=df.columns.tolist().index(str(first_digit))
            match=col[col==text]

            if len(match)>0:
                row_idx=match.index[0]

                bin_annotation=df.iloc[row_idx,col_idx+3]
                bin_annotation_list=[int(el) for el in bin_annotation.split('-')]

                wordStress_annotations[analysisId]=bin_annotation_list
            else:
                print('Text not found')
                print(r)
        else:
            print('index with less than 3 digits')
            print(r)
    
    return wordStress_annotations

def get_cmudict_info(word='university'):
    """get the first possible phonetisation of a word from cmudict

    Args:
        word (str, optional): input. Defaults to 'university'.
    Returns:
        list: phonemes and a number for each vowle indicating stress: 0=no stress, 1=primary stress, 2=secondary stress
    """
    return cmudict.dict()[word][0]

if __name__ == "__main__":
    # execute only if run as a script

    # Test performance of wordStress module

    d=get_data()
    #d[d.analysisId==232][d.focusType=='wordstress']

    a=get_wordStress_annotation()

    d=d[d.focusType=='wordstress']
    audio_path="../audio-with-analysis-ids/audio/"

    preds=[]
    GTs=[]
    errors=[]
    p_errors=[]
    for id,bin in a.items():
        # print(bin)
        row=d[d.analysisId==id]  #[d.focusType=='wordstress']
        ground_truth=a[id]
        path=os.path.join(audio_path, row.primaryKey.values[0]+'.wav')
        p=set_params(sentenceID=id, waveFileAddress=path, module='wordStress')
        try:
            pred=wordStress(p)
            # print('Ground truth:')
            # print(ground_truth)
            # print('prediction:')
            # print(pred)
            preds.append(pred)
            GTs.append(ground_truth)
        except:
            print('Error with:')
            print(row)
            errors.append(row)
            p_errors.append(p)
            # import pdb;pdb.set_trace()
    
    print(preds)
    print(GTs)
    errors=pd.concat(errors)

    clean_temp_files()