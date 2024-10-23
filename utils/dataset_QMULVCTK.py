# import os
# import librosa
# import soundfile as sf
# import numpy as np
# import scikit.learn 

# # Make train_valid_test_split
# from sklearn.model_selection import train_test_split

# def process_directory(start_dir, output_dir):
#     # Walk through the directory and process all .wav files
# 	for root, dir, files in os.walk(os.path.join(start_dir)):
# 		for file in files:
# 			if file.endswith('mic1.flac'):
# 				file_path = os.path.join(root, file)
# 				process_wav_file(file_path, os.path.splitext(os.path.basename(file))[0], e_name, output_dir)
        
# import os
# import random
# import shutil
# from typing import List, Tuple

# def split_files(source_folder: str, output_folder: str, split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2)):
#     """
#     Split files from source folder into train, validation, and test folders.
    
#     :param source_folder: Path to the folder containing source files
#     :param output_folder: Path to the folder where train, validation, and test folders will be created
#     :param split_ratio: Tuple of (train_ratio, validation_ratio, test_ratio)
#     """
#     # Validate split ratio
#     if sum(split_ratio) != 1.0:
#         raise ValueError("Split ratios must sum to 1.0")

#     # Create output folders
#     train_folder = os.path.join(output_folder, 'train')
#     val_folder = os.path.join(output_folder, 'val')
#     test_folder = os.path.join(output_folder, 'test')
    
#     for folder in [train_folder, val_folder, test_folder]:
#         os.makedirs(folder, exist_ok=True)

#     # Get all files from source folder
#     all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
#     all_files.sort() 
#     random.seed(42)
#     random.shuffle(all_files)

#     # Calculate split indices
#     total_files = len(all_files)
#     train_split = int(total_files * split_ratio[0])
#     val_split = int(total_files * (split_ratio[0] + split_ratio[1]))

#     # Split files
#     train_files = all_files[:train_split]
#     val_files = all_files[train_split:val_split]
#     test_files = all_files[val_split:]

#     # Copy files to respective folders
#     for file_list, folder in zip([train_files, val_files, test_files], [train_folder, val_folder, test_folder]):
#         for file in file_list:
#             shutil.copy2(os.path.join(source_folder, file), os.path.join(folder, file))

#     print(f"Split complete. \nTrain: {len(train_files)} files\nValidation: {len(val_files)} files\nTest: {len(test_files)} files")

# # Example usage
# if __name__ == "__main__":
#     source_folder = "/mnt/data3/SDE/raw_IR/QMUL/16khz"
#     output_folder = "/mnt/data3/SDE/raw_IR/QMUL/low"
#     split_ratio = (0.7, 0.1, 0.2)  # 70% train, 20% validation, 10% test
    
#     split_files(source_folder, output_folder, split_ratio)


import os
import random
import shutil
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve
import soundfile as sf
import librosa
from pathlib import Path
# def split_speakers(source_dir, output_dir, split_ratio=(0.7, 0.1, 0.2)):
#     speakers = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
#     speakers.sort()
#     random.seed(42)
#     random.shuffle(speakers)
    
#     total = len(speakers)
#     train_split = int(total * split_ratio[0])
#     val_split = int(total * (split_ratio[0] + split_ratio[1]))
    
#     train_speakers = speakers[:train_split]
#     val_speakers = speakers[train_split:val_split]
#     test_speakers = speakers[val_split:]
    
#     # for split, split_speakers in zip(['train', 'valid', 'test'], [train_speakers, val_speakers, test_speakers]):
#     #     split_dir = os.path.join(output_dir, split)
#     #     os.makedirs(split_dir, exist_ok=True)
#     #     for speaker in split_speakers:
#     #         shutil.copytree(os.path.join(source_dir, speaker), os.path.join(split_dir, speaker))
    
#     return train_speakers, val_speakers, test_speakers

# def apply_filters(filter_dir, speech_dir, output_dir, speakers, num_samples=10, duration=10):
#     os.makedirs(output_dir, exist_ok=True)
    
#     for filter_file in os.listdir(filter_dir):
#         if filter_file.endswith('.wav'):
            
#             filter_path = os.path.join(filter_dir, filter_file)
#             filter_data, filter_rate = librosa.load(filter_path, sr=None)
            
#             for _ in range(num_samples):
#                 speaker = random.choice(speakers)
#                 speaker_dir = os.path.join(speech_dir, speaker)
#                 speech_files = [f for f in os.listdir(speaker_dir) if f.endswith('mic1.flac')]
#                 speech_file = random.choice(speech_files)
#                 speech_path = os.path.join(speaker_dir, speech_file)
                
#                 speech_data, speech_rate = librosa.load(speech_path, sr=None)
                
#                 # Ensure both signals have the same sample rate
#                 if speech_rate != filter_rate:
#                     raise ValueError("Sample rates do not match")
                
#                 # Crop silence
#                 speech_data = crop_silence(speech_data)
#                 speech_data = repeat_to_len(speech_data, duration)
#                 # Convolve the signals
#                 convolved = convolve(speech_data, filter_data)[:duration*filter_rate]
                
#                 # Normalize the output
#                 if np.max(np.abs(convolved)) > 0.99:
#                     convolved = convolved / np.max(np.abs(convolved))
                
#                 # Generate output filename
#                 output_filename = f"{os.path.splitext(filter_file)[0]}_{'_'.join(os.path.splitext(speech_file)[0].split('_')[0:2])}.wav"
#                 output_path = os.path.join(output_dir, output_filename)
#                 print(output_path)
                
#                 # Save the convolved audio
#                 sf.write(output_path, convolved, speech_rate)

# def crop_silence(y, top_db=35, frame_length=2048, hop_length=512, sr=48000):
#     # Load the audio file
    
#     # Trim silence
#     yt, index = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

#     # Save the trimmed audio    
#     # print(f"Original duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
#     # print(f"Cropped duration: {librosa.get_duration(y=yt, sr=sr):.2f} seconds")
    
#     return yt

# def repeat_to_len(y, duration=10, sr=48000):
#     if len(y) < duration*sr:
#         y = np.tile(y, (duration*sr)//len(y) + 1)
#     return y


def process_wav_file(file_path, file, output_dir, split_dir, target_sr=16000):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Resample to 16 kHz
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Prepare the output file path, maintaining the same directory structure
    # relative_path = os.path.relpath(file_p ath, start_dir)
    # output_file_path = os.path.join(output_dir, relative_path)
    filename = file.split('_')
    filename[4] = '16khz'
    filename = '_'.join(filename)
    print(split_dir)
    output_file_path = os.path.join(output_dir, split_dir, filename)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    print(output_file_path)
    # Save the processed file as WAV
    sf.write(output_file_path, resampled_audio, target_sr)

def process_directory(start_dir, output_dir):
    # Walk through the directory and process all .wav files
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                process_wav_file(file_path, file, output_dir, os.path.basename(os.path.dirname(file_path) ))


start_dir = '/mnt/data3/SDE/real_datasets/QMULVCTK'  # Directory to scan for .wav files
output_dir = '/mnt/data3/SDE/real_datasets/wav16'  # Directory to save the processed files

process_directory(start_dir, output_dir)

    
# # Main execution
# if __name__ == "__main__":
#     source_dir = "/mnt/data3/VCTK-Corpus/wav48_silence_trimmed"
#     output_dir = "/mnt/data3/SDE/real_datasets/QMULVCTK"
#     filter_dir = "/mnt/data3/SDE/raw_IR/QMUL/high"
    
#     # Split speakers
#     train_speakers, val_speakers, test_speakers = split_speakers(source_dir, output_dir)
#     # Apply filters to each split
#     for split, speakers in zip(['val', 'test'], [val_speakers, test_speakers]):
#         split_dir = os.path.join(source_dir)
#         filter_output_dir = os.path.join(output_dir, f"{split}")
#         filter_dir_2 = os.path.join(filter_dir, split)
#         apply_filters(filter_dir_2, split_dir, filter_output_dir, speakers)

#     print("Processing complete!")

