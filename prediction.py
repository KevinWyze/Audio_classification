from fastai import *
from fastai.vision import *
import argparse
import librosa


class ConvertSpectrogram:
    def __init__(self, audio_file):
        self.audio_file = audio_file

    def convert_img_spectrogram(self):
        samples, sample_rate = librosa.load(self.audio_file)
        fig = plt.figure(figsize=[0.72, 0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        filename = Path(self.audio_file).name.replace('.wav', '.png')
        spec_fig = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(spec_fig, ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_audio_path", type=str, default="", help="Path to audio to predict")
    parser.add_argument("--checkpoint_model", type=str, default="saved_model/stage-1", help="Path to saved model")

    opt = parser.parse_args()
    print(opt)

    converter = ConvertSpectrogram(opt.predict_audio_path)
    converter.convert_img_spectrogram()

    learner = load_learner(opt.checkpoint_model)

    img_name = Path(opt.pred_audio_path).name.replace('.wav', '.png')
    predict_result = learner.predict(img_name)
    print(predict_result)
