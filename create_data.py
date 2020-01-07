import os
import shutil
from pathlib import Path
from fastai.vision import * 

class CreateData:
    def __init__(self):
        self.data_path = Path('/home/ubuntu/kevin_folder/Audio_classification/UrbanSound8K/data/')
        self.spectrogram_path = Path('/home/ubuntu/kevin_folder/Audio_classification/UrbanSound8K/spectrogram/')
        self.labels = ['air_conditioner', 'car_horn', 'children_playing',
                       'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

    def create_fold_directory(self, fold):
        png_files = list(Path(self.spectrogram_path / fold).glob('*.png'))
        #print(png_files)
        os.mkdir(self.data_path / fold)
        os.mkdir(self.data_path / fold / 'train')
        os.mkdir(self.data_path / fold / 'valid')
        for label in self.labels:
            os.mkdir(self.data_path / fold / 'train' / label)
            os.mkdir(self.data_path / fold / 'valid' / label)

        for file in png_files:
            label = file.as_posix().split('-')[1]
            shutil.copyfile(file, self.data_path / fold / 'valid' / self.labels[int(label)] / file.name)
        for i in range(1, 11):
            if 'fold'+str(i) == fold:
                continue
            fold_name = 'fold' + str(i)
            png_files = list(Path(self.spectrogram_path / fold_name).glob('*.png'))
            for file in png_files:
                label = file.as_posix().split('-')[1]
                shutil.copyfile(file, self.data_path / fold / 'train' / self.labels[int(label)] / file.name)


if __name__ == '__main__':
    create_data_operator = CreateData()
    for i in range(1, 11):
        fold_name = 'fold' + str(i)
        create_data_operator.create_fold_directory(fold_name)
        
