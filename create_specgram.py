from fastai.vision import *
import librosa.display
import matplotlib.pyplot as plt
import librosa
import numpy as np


class CreateSpectrogram:
    def __init__(self):
        self.spectrogram_path = Path('/home/ubuntu/kevin_folder/Audio_classification/UrbanSound8K/spectrogram/')
        self.audio_path = Path('/home/ubuntu/kevin_folder/Audio_classification/UrbanSound8K/audio/')

    def create_fold_spectrogram(self, fold):
        print(f'Processing {fold}')
        os.mkdir(self.spectrogram_path/fold)
        for audio_file in list(Path(self.audio_path/f'{fold}').glob('*.wav')):
            print(audio_file)
            samples, sample_rate = librosa.load(audio_file)
            fig = plt.figure(figsize=[0.72, 0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            filename = self.spectrogram_path / fold / Path(audio_file).name.replace('.wav', '.png')
            spec_fig = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(spec_fig, ref=np.max))
            plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')


if __name__ == '__main__':
    create_spectrogram_operator = CreateSpectrogram()
    for i in range(1, 11):
        fold_name = 'fold' + str(i) 
        print(fold_name)
        create_spectrogram_operator.create_fold_spectrogram(fold_name)
