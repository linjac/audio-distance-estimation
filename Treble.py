import torch
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule

class Treble(Dataset):
    def __init__(self,
                path_audios,
                path_Noises = None, 
                dbNoise = None,
                fs = 16000
        ):
        self.path_audios = path_audios
        self.fs = fs
        self.pathNoises = pathNoises
        self.dBNoise = dBNoise
        self.list_all_files = [f for f in listdir(self.path_audios) if isfile(join(self.path_audios, f))]
        if self.pathNoises != None:
            self.list_all_noises = [f for f in listdir(self.pathNoises) if isfile(join(self.pathNoises, f))]

    def __getitem__(self, index):
        path = join(self.path_audios, self.list_all_files[index])
        sound, _ = lb.load(path, sr = self.fs, mono = True, res_type = "kaiser_fast")
        if self.dBNoise is not None:
            print(f"List all noises {len(self.list_all_noises)}")
            random_index_noise = np.random.randint(low = 0, high = len(self.list_all_noises))
            selected_noise_file = self.list_all_noises[random_index_noise]
            audio_noise, _ = lb.load(join(self.pathNoises, selected_noise_file), sr = self.fs, mono = True, res_type = "kaiser_fast")

            RMS_s = np.sqrt(np.mean(np.power(sound,2)))
            if self.dBNoise == "Random": # extractly randomly an SNR
                random_SNR = np.random.rand() * 50
                RMS_n = np.sqrt(np.power(RMS_s,2) / np.power(10, random_SNR/10))
            else:
                RMS_n = np.sqrt(np.power(RMS_s,2) / np.power(10, self.dBNoise/10))

            RMS_n_current = np.sqrt(np.mean(np.power(audio_noise,2)))

            
            audio_noise = audio_noise * (RMS_n / RMS_n_current)

            sound = sound.squeeze() + audio_noise
        else:
            sound = sound.squeeze()

        #### Get distance
        distance = self.list_all_files[index].split('_')[2]
        distance = float(distance[:-1])

        
        return {
                "audio": torch.tensor(sound).float(), 
                "label": torch.tensor(distance).float(),
                "id": self.list_all_files[index]
            }
    
    def __len__(self):
        return len(self.list_all_files) 
    
    def get_mean_distances(self):
        distance = 0
        for i in range(len(self.list_all_files)):
            returned_values = self.__getitem__(i)
            distance += returned_values['label']
        return distance / len(self.list_all_files)
    
    
    def get_distribution(self):
        distances = []
        for i in range(len(self.list_all_files)):
            returned_values = self.__getitem__(i)
            distances.append(returned_values['label'].numpy())
        distances = np.array(distances)
        plt.figure()
        plt.hist(distances, edgecolor = 'k', alpha = 0.65)
        plt.axvline(distances.mean(), color='r', linestyle='dashed', linewidth=1)
        _, max_ylim = plt.ylim()
        plt.text(distances.mean()*1.05, max_ylim*0.9, 'Mean: {:.2f} m'.format(distances.mean()))
        plt.grid(alpha = 0.2)
        plt.title("Treble distance distribution")
        plt.xlabel("Distance [m]")
        plt.ylabel("Occurrences")
        plt.savefig("Treble test.pdf", transparent = True)
        plt.show()
        
class TrebleDataModule(LightningDataModule):
    
    def __init__(self, path_dataset, batch_size, pathNoisesTraining = None, pathNoisesVal = None, pathNoisesTest = None, db = None, fs = 16000):
        super().__init__()
        self.path_dataset = path_dataset
        self.fs = fs
        self.batch_size = batch_size
        # regarding addind background noises
        self.pathNoiseTraining = pathNoisesTraining
        self.pathNoiseVal = pathNoisesVal
        self.pathNoiseTest = pathNoisesTest
        self.dBNoise = db
        
    def prepare_data(self):
        '''
            Do something. Convolve?
        '''
        self.audio_train = 
        self.audio_val = 
        self.audio_test = 
        
    
    def setup(self, stage = None):
        '''
            Do something?
        '''
        
    def train_dataloader(self):
        return Dataloader(Treble(join(self.dataset_)))