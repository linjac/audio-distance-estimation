import os
import librosa
import soundfile as sf
import numpy as np

def process_wav_file(file_path, file, output_dir, target_duration=10.0, target_sr=16000):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Get the duration of the audio in seconds
    duration = librosa.get_duration(y=audio, sr=sr)
    
    dataset = np.random.random_sample()
    if dataset <= 0.7:
        noise_dir = "training_noise"
    elif dataset <= 0.8:
        noise_dir = "val_noise"
    else:
        noise_dir = "test_noise"

    # Process only if the duration is 10 seconds or more
    if duration >= target_duration:
        # Truncate the audio to the first 10 seconds
        for l in range(int(min(4, duration//target_duration))):
            s_idx = int(l * target_duration * sr)
            e_idx = int((l+1) * target_duration * sr)
            truncated_audio = audio[s_idx:e_idx]
            # truncated_audio = audio[:int(target_duration * sr)]
            
            # Resample to 16 kHz
            resampled_audio = librosa.resample(truncated_audio, orig_sr=sr, target_sr=target_sr)
            
            # Prepare the output file path, maintaining the same directory structure
            # relative_path = os.path.relpath(file_p ath, start_dir)
            # output_file_path = os.path.join(output_dir, relative_path)

            output_file_path = os.path.join(output_dir, noise_dir, file + '_' + str(l).zfill(1) + '.wav')
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Save the processed file as WAV
            sf.write(output_file_path, resampled_audio, target_sr)

def process_directory(start_dir, output_dir):
    # Walk through the directory and process all .wav files
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                process_wav_file(file_path, os.path.splitext(os.path.basename(file))[0], output_dir)

# Example usage:
np.random.seed(42)

start_dir = '/mnt/data3/high_res_wham/audio'  # Directory to scan for .wav files
output_dir = '/mnt/data3/SDE/noise_dataset'  # Directory to save the processed files

process_directory(start_dir, output_dir)
