from fastai.vision import *
from fastai.vision import image as im
import argparse
import librosa


class Prediction:
    def __init__(self, dataset_path, audio_file, model_path):
        self.audio_file = audio_file
        self.data_dir = Path(self.dataset_path)
        self.data = ImageDataBunch.from_folder(data_dir, ds_tfms=[], size=224)
        self.learn = cnn_learner(data, models.resnet34, metrics=error_rate)
        self.learn = self.learn.load(model_path)
        self.img_name = Path(self.audio_file).name.replace('.wav', '.png')


    def convert_audio_img(self):
        samples, sample_rate = librosa.load(self.audio_file)
        fig = plt.figure(figsize=[0.72, 0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        spec_fig = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(spec_fig, ref=np.max))
        plt.savefig(img_name, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def predict_on_img(self):
        img = im.open_image(self.image_name)
        pred_class, pred_idx, outputs = self.learn.predict(img)
        return(pred_class, outputs[pred_idx])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type = str, default = "/home/ubuntu/kevin_folder/Audio_classifcation/UrbanSound8K/data/fold1", help ="Path to dataset")
    parser.add_argument("--predict_audio_path", type=str, default="", help="Path to audio to predict")
    parser.add_argument("--checkpoint_model", type=str, default="/home/ubuntu/kevin_folder/Audio_classifcation/UrbanSound8K/saved_model/trained_model", help="Path to a saved model")


    opt = parser.parse_args()
    print(opt)


    predictor = ConvertSpectrogram(opt.dataset_path, opt.predict_audio_path, opt.checkpoint_model)
    predictor.convert_audio_img()
    pred_class, pred_prob = predictor.predict_on_img()
    print(f'The prediction class is {pred_class} with probability {pred_prob}')
