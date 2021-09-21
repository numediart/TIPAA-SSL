import pyworld as pw
import soundfile as sf
import numpy as np
import librosa
import pytsmod as tsm

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from DL_speech_tech import melgan_analysis_synthesis, speech_enhancement
def load_audio(waveFileAddress, fs=16000, speech_correction=True):
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

    if speech_correction:
        s=speech_enhancement(s)

    # trim silences
    # s, index = librosa.effects.trim(s, top_db=20)
    # remove DC
    s = s - s[int(0.15*len(s)):int(0.85*len(s))].mean() # we exclude 15% at each side that might contain buffer initialization/release noises
    # normalization
    s = 0.90*s/max(abs(s))
    return s, fs


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

    # convert in semitones
    f0Frames=40*np.log10(f0)

    # replace any inf due to the log operation with nan (which are treaded below)
    f0Frames[f0Frames==-np.inf]=np.nan

    from scipy.interpolate import interp1d

    x=np.arange(len(f0Frames))
    f = interp1d(x, f0Frames, kind='linear')

    xnew = np.floor(np.arange(len(s))/len(s)*len(f0Frames))
    f0Samples = f(xnew)

    # from scipy import signal
    # ynew = signal.resample(f0Frames, len(s))

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
    # return y[5:len(y)-5]
    
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
    int_db = 10*np.log10(intensity)

    #  remove any inf due to the log operation and replace them with the minimum value of intensity
    # int_db(isinf(int_db)) = min(int_db(~isinf(int_db)));
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


def slow_down_audiotsm(input_filename='audio_recordings/WS_111_toothpaste.wav', output_filename='audio_recordings/WS_111_toothpaste_0.8.wav', speed_rate=0.8):    
    with WavReader(input_filename) as reader:
        with WavWriter(output_filename, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=speed_rate)
            tsm.run(reader, writer)

def slow_down(input_filename='audio_recordings/WS_111_toothpaste.wav', output_filename='audio_recordings/WS_111_toothpaste_pytsmod_0.8.wav', speed_rate=0.8):
    """change speed of audio without modifying the pitch. 

    Args:
        input_filename (str, optional): Defaults to 'audio_recordings/WS_111_toothpaste.wav'.
        output_filename (str, optional): Defaults to 'audio_recordings/WS_111_toothpaste_pytsmod_0.8.wav'.
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
    # recording_path='audio_recordings/SS_1_i_would_love_to_go_to_ireland_FR.wav', 
    recording_path='audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav',
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

    # reference,fs=librosa.load('audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav', sr=16000)
    # recording,fs=librosa.load('audio_recordings/SS_1_i_would_love_to_go_to_ireland_FR.wav', sr=16000)
    
    
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

    sf.write('audio_recordings/SS_1_i_would_love_to_go_to_ireland_merge.wav', out,  fs)

    return out

