from fastai.vision import *
from fastai.vision import image as im
import argparse
import librosa
from fastai import * 
import os 
import matplotlib.pyplot as plt 
import librosa.display

class Prediction:
    def __init__(self, dataset_path, audio_file, model_path):
        self.audio_file = audio_file
        #self.data_dir = Path(dataset_path)
        self.data_dir = Path(dataset_path)
        self.data = ImageDataBunch.from_folder(self.data_dir, size=224)
        self.learn = cnn_learner(self.data, models.resnet34, metrics=error_rate)
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
        plt.savefig(self.img_name, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def predict_on_img(self):
        img = im.open_image(self.img_name)
        pred_class, pred_idx, outputs = self.learn.predict(img)
        os.remove(self.img_name)
        return(pred_class, pred_idx, outputs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type = str, default = "/Users/kli/Desktop/Audio_classification/audio_cnn/data", help ="Path to dataset")
    parser.add_argument("--predict_audio_path", type=str, default="/Users/kli/Desktop/Audio_classification/audio_cnn/test_clips/7061-6-0-0.wav", help="Path to audio to predict")
    parser.add_argument("--checkpoint_model", type=str, default="/Users/kli/Desktop/Audio_classification/audio_cnn/saved_model/trained_model", help="Path to a saved model")


    opt = parser.parse_args()
    print(opt)


    predictor = Prediction(opt.dataset_path, opt.predict_audio_path, opt.checkpoint_model)
    predictor.convert_audio_img()
    pred_class, pred_idx, pred_prob = predictor.predict_on_img()
    actual_class = opt.predict_audio_path.split('-')[1]
    sound_label = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine_idling", "gun shot", "jackhammer", "siren", "street music"]
    pred_idx = int(pred_idx)
    actual_class = int(actual_class)

    if pred_idx == actual_class:
        print(f'The prediction is correct, the sound is {pred_class}, with confidence {outputs[pred_idx]}')
    else:
        print(f'We fail the prediction, the prediction is {pred_class} with probability {outputs[pred_idx]}. However, the actual sounds is {sound_label[actual_class]}, our model probability is {outputs[actual_class]}.')
