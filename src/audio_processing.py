import array
import base64
import io

import librosa
import numpy as np
import pytsmod as tsm
import pyworld as pw
import soundfile as sf
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from pydub import AudioSegment
from pydub.utils import get_array_type
from scipy.interpolate import interp1d
from soundfile import LibsndfileError


# signal processing (pitch, instensity, normalization...)
def getf0Samples(s: np.ndarray, fs: float) -> np.ndarray:
    """Uses pyworld vocoder to extract fundamental frequency of the signal in Hz
    and converts it in semitones. And then upsample up to signal length

    Args:
        s (numpy array): audio waveform signal
        fs (float): frequency of sampling

    Returns:
        numpy array: upsampled f0 contour in semitones
    """

    # pyin works as a replacement of pyworld, but I saw a slightly lower performance with it, so I'm not changing that for now
    # f0, voiced_flag, voiced_prob=librosa.pyin(s, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=fs)
    f0, sp, ap = pw.wav2world(s.astype(np.float64), fs)  # type: ignore

    f0 += 10**-10  # to avoid zeros going in the log, add a tiny number

    # convert in semitones
    f0Frames = 40 * np.log10(f0)

    # replace any inf due to the log operation with nan (which are treated below)
    f0Frames[f0Frames == -np.inf] = np.nan
    x = np.arange(len(f0Frames))
    f = interp1d(x, f0Frames, kind='linear')

    xnew = np.floor(np.arange(len(s)) / len(s) * len(f0Frames))
    f0Samples = f(xnew)

    return f0Samples


def getIntonation(s: np.ndarray, fs: float) -> np.ndarray:
    f0Samples = getf0Samples(s, fs)
    # replace nans with minimum value
    f0Samples = np.nan_to_num(f0Samples, nan=np.nanmin(f0Samples))
    return f0Samples


def smooth(x, beta, window_len=11):
    """kaiser window smoothing
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
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')

    # replace nans with minimum value
    y = np.nan_to_num(y, nan=np.nanmin(y))

    return y[int(window_len / 2) : -int(window_len / 2) + 1]


def getIntensity(s: np.ndarray, fs: float) -> np.ndarray:
    """computes intensity of the signal (squared signal, smoothed) in dB

    Args:
        s (numpy array): audio signal
        fs (int): frequency of sampling

    Returns:
        float: intensity of signal
    """
    # this is done similarly to praat:
    minimum_pitch = 70  # in Hz
    analysis_win = round((3.2 / minimum_pitch) * fs)

    # smoothing the signal power using a moving average
    intensity = smooth((s) ** 2, 20, 2 * analysis_win)
    if len(intensity) == 0:
        return np.array([])

    # convert in db
    intensity /= 4.0e-10
    int_db = 10 * np.log10(intensity + 10**-10)

    #  remove any inf due to the log operation and replace them with the minimum value of intensity
    int_db[int_db == -np.inf] = min(int_db[int_db > -np.inf])
    return int_db


def normalize(x: np.ndarray | list) -> np.ndarray:
    """normalizes a signal between 0 and 1

    Args:
        x (numpy array): signal to normalize

    Returns:
        numpy array: normalized signal
    """
    if len(x) == 0:
        return np.array([])
    x = np.array(x)
    y = x - x.min()
    ymax = y.max()
    return y / ymax if ymax > 0 else y


def slow_down_audiotsm(
    input_filename='data/audio_recordings/WS_111_toothpaste.wav',
    output_filename='data/audio_recordings/WS_111_toothpaste_0.8.wav',
    speed_rate=0.8,
):
    with WavReader(input_filename) as reader:
        with WavWriter(output_filename, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=speed_rate)
            tsm.run(reader, writer)


def slow_down(
    input_filename='data/audio_recordings/WS_111_toothpaste.wav',
    output_filename='data/audio_recordings/WS_111_toothpaste_pytsmod_0.8.wav',
    speed_rate=0.8,
):
    """change speed of audio without modifying the pitch.

    Args:
        input_filename (str, optional): Defaults to 'data/audio_recordings/WS_111_toothpaste.wav'.
        output_filename (str, optional): Defaults to 'data/audio_recordings/WS_111_toothpaste_pytsmod_0.8.wav'.
        speed_rate (float, optional): Defaults to 0.8.
    """
    x, sr = sf.read(input_filename)
    x = x.T
    x_length = x.shape[-1]  # length of the audio sequence x.
    s_fixed = 1 / speed_rate  # stretch the audio signal 1.3x times.
    x_s_fixed = tsm.wsola(x, s_fixed)
    sf.write(output_filename, x_s_fixed, sr)


def align_audios(
    reference_path='../audio-with-analysis-ids/audio/dbb2b8be-50f6-4b69-8ec1-a53a4bb307bf.wav',
    # recording_path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland_FR.wav',
    recording_path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.wav',
    fs=16000,
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
    reference, fs = librosa.load(reference_path, sr=fs)
    recording, fs = librosa.load(recording_path, sr=fs)

    # reference/=max(abs(reference))
    # reference*=0.7
    # recording/=max(abs(recording))
    # recording*=0.8

    # reference=(2*normalize(reference)-1)*0.9
    # recording=(2*normalize(recording)-1)*0.9
    ref_mel = librosa.feature.melspectrogram(y=reference, sr=fs)
    rec_mel = librosa.feature.melspectrogram(y=recording, sr=fs)

    # from dtw import *
    # alignment = dtw(rec_mel.T, ref_mel.T, keep_internals=True)
    D, wp = librosa.sequence.dtw(X=ref_mel, Y=rec_mel)

    # from fastdtw import fastdtw
    # distance, path = fastdtw(ref_mel.T, rec_mel.T)#, dist=euclidean)

    # s, index = librosa.effects.trim(s, top_db=20)

    s_ap = (wp[::-1].T * len(reference) / ref_mel.shape[-1]).astype(int)
    # s_ap=(np.array(path).T*len(reference)/ref_mel.shape[-1]).astype(int)

    recording_aligned = tsm.ola(recording, s_ap[::-1])

    max_len = max(len(recording_aligned), len(reference))

    def pad_zeros_to_len(a, length):
        return np.pad(a, (0, (length - len(a))), 'constant', constant_values=(0, 0))

    reference = pad_zeros_to_len(reference, max_len)
    recording_aligned = pad_zeros_to_len(recording_aligned, max_len)
    out = recording_aligned + reference

    sf.write(
        'data/audio_recordings/SS_1_i_would_love_to_go_to_ireland_merge.wav', out, fs
    )

    return out


def read_audio_file(
    audio_file: str | io.IOBase, fs: float = 16000
) -> tuple[np.ndarray, float]:
    """
    -audio file is a path or file-like object
    -then read that with "soundfile" when possible, else with "pydub"
    -resample to fs

    refs:
    https://stackoverflow.com/questions/55352789/how-can-i-convert-any-random-byte-string-to-a-playable-mp3-file-in-python3-6
    https://stackoverflow.com/questions/32373996/pydub-raw-audio-data
    """
    try:
        s, orig_sr = sf.read(audio_file)
    except LibsndfileError:
        audio = AudioSegment.from_file(audio_file)

        bit_depth = audio.sample_width * 8
        array_type = get_array_type(bit_depth)
        numeric_array = array.array(array_type, audio._data)
        s = np.array(numeric_array) / 2**15
        orig_sr = audio.frame_rate

    # if the signal is a 2D array, we take the average of both channels (stereo)
    if len(s.shape) > 1 and s.shape[-1] == 2:  # noqa: PLR2004
        s = np.mean(s, axis=1)
    # if len(s) is 0, stop here, don't try to resample it, it will throw an error
    if len(s) == 0:
        return s, fs
    new_s = librosa.resample(s, orig_sr=orig_sr, target_sr=fs)

    return new_s, fs


def read_audio_bytes(
    audio_bytes: bytes | bytearray, fs: float = 16000
) -> tuple[np.ndarray, float]:
    """
    -put into a file-like object with "io",
    -then calls read_audio_file that reads with "soundfile" when possible, else with "pydub"
    """
    return read_audio_file(io.BytesIO(audio_bytes), fs=fs)


def read_audio_string(
    encoded_string: str | bytearray | bytes, fs: float = 16000
) -> tuple[np.ndarray, float]:
    """
    -decodes base64,
    -call read_audio_bytes
    """
    audio_bytes = base64.b64decode(encoded_string)
    s, fs = read_audio_bytes(audio_bytes, fs=fs)
    return s, fs


def audio64_from_file(path: str, fs: float = 16000) -> bytes:
    # writing bytes of an ogg file with virtual io, then encoding with base64
    s, fs = read_audio_file(path, fs=fs)
    with io.BytesIO() as fio:
        sf.write(fio, s, samplerate=fs, format='ogg')
        audio_string = fio.getvalue()
    encode_string = base64.b64encode(audio_string)
    return encode_string


def test_pyin():
    path = 'data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    s, fs = librosa.load(path, sr=16000)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        s,
        fmin=float(librosa.note_to_hz('C2')),
        fmax=float(librosa.note_to_hz('C7')),
        sr=fs,
    )

    f0 += 10**-10  # to avoid zeros going in the log, add a tiny number

    # convert in semitones
    f0Frames = 40 * np.log10(f0)

    # replace any inf due to the log operation with nan (which are treated below)
    f0Frames[f0Frames == -np.inf] = np.nan
    x = np.arange(len(f0Frames))
    f = interp1d(x, f0Frames, kind='linear')

    xnew = np.floor(np.arange(len(s)) / len(s) * len(f0Frames))
    f0Samples = f(xnew)

    # f0_pw, sp, ap = pw.wav2world(s.astype(np.float64), fs)

    f0Samples_pw = getf0Samples(s, fs)

    f0Samples_pw[f0Samples_pw == -400] = np.nan

    import matplotlib.pyplot as plt

    plt.cla()
    plt.plot(f0Samples, label="pyin")
    plt.plot(f0Samples_pw, label="pyworld")
    plt.legend()
    plt.savefig('f0.png')


def test_audio_file_like():
    # from six.moves.urllib.request import urlopen

    # url = "https://raw.githubusercontent.com/librosa/librosa/master/tests/data/test1_44100.wav"
    # url="https://filesamples.com/samples/audio/caf/sample3.caf"
    # url="https://filesamples.com/samples/audio/m4a/sample3.m4a"

    import base64

    path = 'data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.caf'
    path = 'temp.ogg'
    # path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'
    # path='data/audio_recordings/turned_around.mp3'

    sf.write(path, [], 44100)
    encode_string = base64.b64encode(open(path, "rb").read())

    # this is an audio full of zeros
    encode_string = "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"

    # this is an audio with 0 sample at 44100Hz
    encode_string = "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EErKnINQSkqZcxBKSSmUkkpKrYVSSkqptQIAAAocAAACbNCUWByg0JCVAEAqAIDBcSzL80RRNWXZsSTPE0XTVFXbdizL80TRNFXVti3PE0XTVFXX1XXL80TRVFXVdXXdE0XVVFXXlWXf90TRNFXVdWXZ903TdFXXlWXb9n3TNFXXdWVZtn1hdVXXlWXb1m1jWFXXdWXZtm1dOW7d1nXhF4ZhmNq67vu+LwzH8EwDAMATHACACmxYHeGkaCyw0JCVAEAGAABhDEIGIYUMQkghhZRCSCklAABgwAEAIMCEMlBoyEoAIBUAACDEWmuttdZaYqm11lprrbWGSmuttdZaa6211lprrbXWWmuttdZaa6211lprrbXWWmuttZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKaWUUkoppZRSSimllFJKKRUA6FfhAOD/YMPqCCdFY4GFhqwEAMIBAABjlGIMOukkpNQw5RiEUlJJpZVGMecglJJSSq1VzklIpaXWWouxck5KSSm1FluMHYSUWmotxhhj7CCklFprMcYYYyilpRhjrDHWWkNJqbUYY4w111pSai3GWmutufeSUosxxlxr7rmX1mKsteacc849tRZjrTXn3HPwqbUYY8619957UK3FWGuuOQfhewEA3A0OABAJNs6wknRWOBpcaMhKACAkAIBAiDHGnHMOQgghREox5pxzEEIIIYRIKcaccw5CCCGEkDHmnHMQQgihlFIyxpxzDkIIJZRQSuaccxBCCKGUUkrJnHMOQgghlFJKKR10EEIIoZRSSimlcw5CCKGUUkoppYQQQiillFJKKaWUEEIIpZRSSimllBJCCKWUUkoppZRSQgihlFJKKaWkUkoIoZRSSimllFJKCSGUUkoppZRSSimhhFJKKaWUUkopJZRQSimllFJKKqUUAABw4AAAEGAEnWRUWYSNJlx4AAoNWQkAAAEAIM5abClGRjHnIIbIIMQghgopxZy1DCmDHKZMKYSUlc4xhoiTFlsLFQMAAEAQAEAgZAKBAigwkAEABwgJUgBAYYGhQ4QIEKPAwLi4tAEACEJkhkhELAaJCdVAUTEdACwuMOQDQIbGRtrFBXQZ4IIu7joQQhCCEMTiAApIwMEJNzzxhifc4ASdolIHAQAAAABwAAAPAADHBhAR0RxHh8cHSIjICElJAAAAAADgAMAHAMBhAkRENMfR4fEBEiIyQlISAAAAAAAAAAAABAQEAAAAAAACAAAABARPZ2dTAAQAAAAAAAAAAOSvZQQCAAAAIz0MRgEBAA=="

    decode_string = base64.b64decode(encode_string)

    sf.read(io.BytesIO(decode_string))[0]

    s, fs = librosa.load(io.BytesIO(decode_string), sr=fs)

    wav_file = open(path, "wb")
    wav_file.write(decode_string)
    wav_file.close()


def use_tests(path='data/audio_recordings/SS_1_i_would_love_to_go_to_ireland.m4a'):
    import array

    from pydub import AudioSegment
    from pydub.utils import get_array_type

    with open('base64_test.txt', 'r') as f:
        encoded_string = f.readlines()
    encoded_string = '\n'.join(encoded_string)
    decode_string = base64.b64decode(encoded_string)

    audio = AudioSegment.from_file(path, format="mp3")

    bit_depth = audio.sample_width * 8
    array_type = get_array_type(bit_depth)
    numeric_array = array.array(array_type, audio._data)
    s = np.array(numeric_array) / 2**15

    fs = 16000

    librosa.resample(s, orig_sr=audio.frame_rate, target_sr=fs)

    import io

    f = io.BytesIO()
    f = audio.export(f, format='mp3')

    test = AudioSegment.from_file(f, format="mp3")
