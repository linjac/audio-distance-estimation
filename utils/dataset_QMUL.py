import os
import librosa
import soundfile as sf
import numpy as np

# def process_directory(start_dir, output_dir):
#     # Walk through the directory and process all .wav files
#     for root, dir, files in os.walk(start_dir):
        # for file in files:
        #     if file.endswith('.wav'):
        #         file_path = os.path.join(root, file)
        #         process_wav_file(file_path, os.path.splitext(os.path.basename(file))[0], output_dir)
def get_distance(xn, yn, environment_name):
    from math import sqrt
    from numpy import round

    if environment_name == "greathall" or environment_name=="octagon":
        x = int(xn[1:])
        y = int(yn[1:])
        xc = 6
        y0 = 2
        x_real = (x - xc)
        y_real = y + y0
        dist = round(sqrt(x_real**2 + y_real**2),3)
        return  str(x).zfill(2) + 'x_' + str(y).zfill(2) + 'y_' + f'{dist:.3f}' + 'm'
    elif environment_name == "classroom":
        x = int(xn[:-1])
        y = int(yn[:-1])
        xc = 30
        y0 = 1.5
        x_real = (x - xc) * 0.1
        y_real = y*0.1 + y0
        dist = round(sqrt(x_real**2 + y_real**2),3)
        return str(x).zfill(2) + 'x_' + str(y).zfill(2) + 'y_' + f'{dist:.3f}' + 'm'
    

def process_wav_file(file_path, filename, environment_name, output_dir):
    
    audio, sr = librosa.load(file_path, sr=None)
    
    x = filename[0:3]
    y = filename[3:6]
    fname = '_'.join([get_distance(x, y, environment_name), environment_name + 'Omni'])

    # downsample to 16kHz
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    output_file_path = os.path.join(output_dir, '16khz', fname + '_16khz.wav')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    sf.write(output_file_path, resampled_audio, 16000)
    print(output_file_path)

    # resample to 48kHz
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    output_file_path = os.path.join(output_dir, '48khz', fname + '_48khz.wav')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    print(output_file_path)
    sf.write(output_file_path, resampled_audio, 48000)
    
    return None
    
def process_directory(start_dir, output_dir):
    # Walk through the directory and process all .wav files
    environments = ["greathall", "octagon", "classroom"]
    for e_name in environments:
        for root, dir, files in os.walk(os.path.join(start_dir, e_name)):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    process_wav_file(file_path, os.path.splitext(os.path.basename(file))[0], e_name, output_dir)
            
start_dir = '/mnt/data3/SDE/raw_IR/C4DM'
end_dir = '/mnt/data3/SDE/raw_IR/QMUL'
process_directory(start_dir, end_dir)