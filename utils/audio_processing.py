import pyworld as pw
import soundfile as sf
import numpy as np
import librosa
import pytsmod as tsm

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from utils.DL_audio_processing import melgan_analysis_synthesis, speech_enhancement
from scipy.interpolate import interp1d
import uuid
from scipy.io.wavfile import write, read
import os
import base64
import io
def load_audio(waveFileAddress, fs=16000):
    """Load audio, remove DC and normalize waveform
    speech_correction refers to the use of MetricGAN+. A speech enhancement system based on an adversarial loss and PESQ/STOI metrics 
    to improve audio quality. 

    Args:
        waveFileAddress (string): wav file address
    Returns:
        numpy array, int: waveform signal and frequency of sampling
    """
    # fs, s = read(waveFileAddress)
    s,fs=librosa.load(waveFileAddress, sr=fs)

    # if speech_correction:
    #     s=speech_enhancement(s)

    # trim silences
    # s, index = librosa.effects.trim(s, top_db=20)
    # remove DC
    s = s - s[int(0.15*len(s)):int(0.85*len(s))].mean() # we exclude 15% at each side that might contain buffer initialization/release noises
    # normalization
    s = 0.90*s/max(abs(s))
    return s, fs

def prepare_audio_file(audio_file, fs=16000):
    rID=str(uuid.uuid4())
    if os.path.exists(audio_file):
        s,fs = load_audio(audio_file, fs=fs)
    else:
        return "error: "+audio_file+" could not be loaded", None
    write('./inputs/'+ rID+ '.wav', fs, (s*32767).astype(np.int16))

    return "success", rID

# signal processing (pitch, instensity, normalization...)
def getf0Samples(s, fs):
    """Uses pyworld vocoder to extract fundamental frequency of the signal in Hz 
    and converts it in semitones. And then upsample up to signal length

    Args:
        s (numpy array): audio waveform signal
        fs (int): frequency of sampling

    Returns:
        numpy array: upsampled f0 contour in semitones
    """
    #  the /2**15  is for converting 16bit PCM to double between -1 and 1
    f0, sp, ap = pw.wav2world(s.astype(np.float64)/2**15, fs)

    f0+=10**-10 #to avoid zeros going in the log, add a tiny number

    # convert in semitones
    f0Frames=40*np.log10(f0)

    # replace any inf due to the log operation with nan (which are treated below)
    f0Frames[f0Frames==-np.inf]=np.nan
    x=np.arange(len(f0Frames))
    f = interp1d(x, f0Frames, kind='linear')

    xnew = np.floor(np.arange(len(s))/len(s)*len(f0Frames))
    f0Samples = f(xnew)

    return f0Samples

def getIntonation(s, fs):
    f0Samples=getf0Samples(s,fs)
    # replace nans with minimum value
    f0Samples=np.nan_to_num(f0Samples, nan= np.nanmin(f0Samples))
    return f0Samples

def smooth(x,beta, window_len=11):
    """ kaiser window smoothing
    Args:
        x (numpy array): curve to be smoothed
        beta (float): parameter of kaiser window
        window_len (int, optional): length of the window used to smooth. Defaults to 11.

    Returns:
        numpy array: [description]
    """
    # from https://dzone.com/articles/computing-convolution-using

    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    
    # replace nans with minimum value
    y=np.nan_to_num(y, nan= np.nanmin(y))

    return y[int(window_len/2):-int(window_len/2)+1]

def getIntensity(s, fs):
    """computes intensity of the signal (squared signal, smoothed) in dB

    Args:
        s (numpy array): audio signal
        fs (int): frequency of sampling

    Returns:
        float: intensity of signal
    """
    # this is done similarly to praat:
    minimum_pitch = 70; # in Hz
    analysis_win = round((3.2/minimum_pitch)*fs)

    # smoothing the signal power using a moving average
    #  the /2**15  is for converting 16bit PCM to double between -1 and 1
    intensity=smooth((s/2**15)**2, 20, 2*analysis_win)

    # convert in db
    intensity /= 4.0e-10
    int_db = 10*np.log10(intensity+10**-10)

    #  remove any inf due to the log operation and replace them with the minimum value of intensity
    int_db[int_db==-np.inf]=min(int_db[int_db>-np.inf])
    return int_db

def normalize(x):
    """normalizes a signal between 0 and 1

    Args:
        x (numpy array): signal to normalize

    Returns:
        numpy array: normalized signal
    """
    y=x-min(x)
    return y/max(y)


def slow_down_audiotsm(input_filename='data/audio_recordings/WS_111_toothpaste.wav', output_filename='data/audio_recordings/WS_111_toothpaste_0.8.wav', speed_rate=0.8):    
    with WavReader(input_filename) as reader:
        with WavWriter(output_filename, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=speed_rate)
            tsm.run(reader, writer)

def slow_down(input_filename='data/audio_recordings/WS_111_toothpaste.wav', output_filename='data/audio_recordings/WS_111_toothpaste_pytsmod_0.8.wav', speed_rate=0.8):
    """change speed of audio without modifying the pitch. 

    Args:
        input_filename (str, optional): Defaults to 'data/audio_recordings/WS_111_toothpaste.wav'.
        output_filename (str, optional): Defaults to 'data/audio_recordings/WS_111_toothpaste_pytsmod_0.8.wav'.
        speed_rate (float, optional): Defaults to 0.8.
    """
    x, sr = sf.read(input_filename)
    x = x.T
    x_length = x.shape[-1]  # length of the audio sequence x.
    s_fixed = 1/speed_rate  # stretch the audio signal 1.3x times.
    x_s_fixed = tsm.wsola(x, s_fixed)
    sf.write(output_filename,x_s_fixed, sr)

def align_audios(
    reference_path='../audio-with-analysis-ids/audio/dbb2b8be-50f6-4b69-8ec1-a53a4bb307bf.wav', 
    # recording_path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland_FR.wav', 
    recording_path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav',
    fs=16000
    ):
    """Performs Dynamic Time Warping on the mel-spectrograms of a reference audio and a recording
    from a user. It computes a path of alignments of timings. This path is then used 
    for a time stretching of the user resording. I use pytsmod, but sometimes it produces errors.
    So maybe I should either find another ola library, or use a deep learning based vocoder on warped mel spectrogram.
    
    The 2 audio can then be summed to have a resulting audio
    in which we can hear both voices at the same time

    Args:
        reference (numpy array): waveform of reference
        recording (numpy array): waveform of user recording
        fs (int, optional): frequency of sampling. Defaults to 16000.

    Returns:
        numpy array: waveform of the merged audio that are time-aligned
    """

    # reference,fs=librosa.load('data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', sr=16000)
    # recording,fs=librosa.load('data/audio_recordings/SS_1_i_would_love_to_go_to_ireland_FR.wav', sr=16000)
    
    
    # reference,fs=load_audio(reference_path)
    # recording,fs=load_audio(recording_path)
    reference,fs=librosa.load(reference_path, sr=fs)
    recording,fs=librosa.load(recording_path, sr=fs)
    
    # reference/=max(abs(reference))
    # reference*=0.7
    # recording/=max(abs(recording))
    # recording*=0.8

    # reference=(2*normalize(reference)-1)*0.9
    # recording=(2*normalize(recording)-1)*0.9
    ref_mel=librosa.feature.melspectrogram(y=reference, sr=fs)
    rec_mel=librosa.feature.melspectrogram(y=recording, sr=fs)

    
    # from dtw import *
    # alignment = dtw(rec_mel.T, ref_mel.T, keep_internals=True)
    D, wp = librosa.sequence.dtw(X=ref_mel, Y=rec_mel)

    # from fastdtw import fastdtw
    # distance, path = fastdtw(ref_mel.T, rec_mel.T)#, dist=euclidean)

    # s, index = librosa.effects.trim(s, top_db=20)

    s_ap=(wp[::-1].T*len(reference)/ref_mel.shape[-1]).astype(int)
    # s_ap=(np.array(path).T*len(reference)/ref_mel.shape[-1]).astype(int)

    recording_aligned = tsm.ola(recording, s_ap[::-1])

    max_len=max(len(recording_aligned), len(reference))

    def pad_zeros_to_len(a, length):
        return np.pad(a, (0, (length-len(a))), 'constant', constant_values=(0, 0))
    
    reference=pad_zeros_to_len(reference, max_len)
    recording_aligned=pad_zeros_to_len(recording_aligned, max_len)
    out=recording_aligned+reference

    sf.write('data/audio_recordings/SS_1_i_would_love_to_go_to_ireland_merge.wav', out,  fs)

    return out

def audio64_from_file(path, fs=16000):
    # writing bytes of an ogg file with virtual io, then encoding with base64
    s,fs=librosa.load(path, sr=fs)
    with io.BytesIO() as fio:
        sf.write(fio, s, samplerate=fs, format='ogg')
        audio_string = fio.getvalue()
    encode_string = base64.b64encode(audio_string)
    return encode_string



def test_audio_file_like():
    import io

    from six.moves.urllib.request import urlopen

    # url = "https://raw.githubusercontent.com/librosa/librosa/master/tests/data/test1_44100.wav"
    # url="https://filesamples.com/samples/audio/caf/sample3.caf"
    url="https://filesamples.com/samples/audio/m4a/sample3.m4a"

    

    import base64
    path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    path='temp.ogg'
    # path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'
    # path='data/audio_recordings/turned_around.mp3'
    encode_string = base64.b64encode(open(path, "rb").read())

    # this is an audio full of zeros
    # encode_string="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"
    
    decode_string = base64.b64decode(encode_string)

    import soundfile as sf
    sf.read(io.BytesIO(decode_string))
    s,fs=librosa.load(io.BytesIO(decode_string), sr=fs)



    wav_file = open(path, "wb")
    wav_file.write(decode_string)
    wav_file.close()