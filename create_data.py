import os
import shutil
from pathlib import Path


class CreateData:
    def __init__(self):
        self.data_path = Path('/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/')
        self.spectrogram_path = Path('/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/spectrogram/')
        self.labels = ['air_conditioner', 'car_horn', 'children_playing',
                       'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

    def create_fold_directory(self, fold):
        png_files = list(Path(self.spectrogram_path / fold).glob('*.png'))
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
            if str(i) == fold:
                continue
            png_files = list(Path(self.spectrogram_path / str(i)).glob('*.png'))
            for file in png_files:
                label = file.as_posix().split('-')[1]
                shutil.copyfile(file, self.data_path / fold / 'train' / self.labels[int(label)] / file.name)


if __name__ == '__main__':
    create_data_operator = CreateData()
    for i in range(1, 11):
        create_data_operator.create_fold_directory(str(i))
