import os
import pandas as pd
from pydub import AudioSegment

# folder_path = '/mnt/c/Users/49174/flowspeech_new/data/EDACC/edacc_v1.0/data_Copy/'
folder_path = 'data/EDACC/edacc_v1.0/Data_Test'

# Output folder where the split segments will be saved
output_folder = 'Output_TEST/'

# Read your DataFrame with start and stop times
df = pd.read_csv(
    'data/EDACC/edacc_v1.0/test/segments_copy',
    sep=' ',
    header=None,
    names=['EDACC', 'WAV_File', 'Start_Time', 'Stop_Time'],
)
# df = pd.read_csv('/mnt/c/Users/49174/flowspeech_new/data/EDACC/edacc_v1.0/dev/text', sep=' ', header=None, names=['WAV_File', 'sentence'])

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize variables for tracking the current file and segment number
current_file = None
segment_number = 0

# Iterate through DataFrame entries
for index, row in df.iterrows():
    # Get WAV file name from the DataFrame
    wav_file = row['WAV_File']
    if not wav_file.endswith('.wav'):
        wav_file += '.wav'

    # Load the WAV file
    sound = AudioSegment.from_wav(os.path.join(folder_path, wav_file))

    # Extract start and stop times from the dataframes
    start_time = row['Start_Time'] * 1000
    stop_time = row['Stop_Time'] * 1000

    # Check if a new file is being processed
    if wav_file != current_file:
        current_file = wav_file  # Update the current file
        segment_number = 0  # Reset the segment number for the new file

    # Extract the segment based on start and stop times
    segment = sound[start_time:stop_time]

    # Generate the output filename with the updated segment number
    output_filename = os.path.join(
        output_folder, f"{os.path.splitext(wav_file)[0]}-0{segment_number}.wav"
    )

    # Export the segment to a new WAV file
    segment.export(output_filename, format="wav")

    # Increment the segment number for the next segment
    segment_number += 1

    # Print information about the current iteration
    print(
        f'Split {wav_file} - Start: {start_time}, Stop: {stop_time}, Saved: {output_filename}'
    )
