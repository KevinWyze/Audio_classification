from fastai.vision import *
class CnnLearner():
    def __init__(self, bs, data_dir):
        self.bs = bs
        self.data_dir = Path(data_dir)

    def train_fold(self, fold):
        data = ImageDataBunch.from_folder(self.data_dir / fold, ds_tfms=[], size=224, bs = self.bs)
        data.normalize(imagenet_stats)
        learn = cnn_learner(data, model.resnet34, metrics = error_rate)
        learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

if __name__ == '__main__':
    create_spectrogram_operator = CreateSpectrogram()
    for i in range(1, 11):
        create_spectrogram_operator.create_fold_spectrogram(str(i))

