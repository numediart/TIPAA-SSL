import ffmpeg
from IPython import display
from ipywebrtc import AudioRecorder, CameraStream
from numpy import ndarray

from flowspeech.audio_processing import read_audio_file


def record_audio():
    """Record audio from microphone using a widget. Use as:

    recorder = record_audio(); recorder
    """
    camera = CameraStream(constraints={"audio": True, "video": False})
    recorder = AudioRecorder(stream=camera)
    return recorder


def get_recording(
    recorder: AudioRecorder, output: str = "recording.wav", sr: float = 16_000
) -> tuple[ndarray, float]:
    """From a recording, return the audio signal and the sample rate.
    Also write it to a file if specified.

    sound, sr = get_recording(recorder)
    """
    with open("temp.webm", "wb") as f_webm:
        f_webm.write(recorder.audio.value)
    ffmpeg.input("temp.webm").output(output).run(overwrite_output=True, quiet=True)
    sig, sr = read_audio_file(output, sr)
    return sig, sr


def play_audio(path_or_sound: str | ndarray, sr: float = 16_000):
    """Play audio in a notebook using a widget.

    Args:
        path_or_sound (str | ndarray): path to audio file or audio signal
        sr (int, optional): sample rate. Defaults to 16_000.
    """
    if isinstance(path_or_sound, str):
        sound, sr = read_audio_file(path_or_sound, sr)
    else:
        sound = path_or_sound
    return display.Audio(data=sound, rate=sr)
