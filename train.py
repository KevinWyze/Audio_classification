import argparse
from fastai.vision import *
from fastai.metrics import error_rate

class CnnLearner():
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def train_fold(self, fold, bs, epoch):
        data_train = ImageDataBunch.from_folder(self.data_dir / fold, ds_tfms=[], size=224, bs=bs)
        data_train.normalize(imagenet_stats)
        learn = cnn_learner(data_train, models.resnet34, metrics=error_rate)
        learn.fit_one_cycle(epoch, max_lr=slice(1e-6, 1e-4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data",
                        help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs for each fold")
    opt = parser.parse_args()
    print(opt)

    learner = CnnLearner(opt.dataset_path)

    for i in range(1, 10):
        learner.train_fold(str(i), opt.batch_size, opt.num_epochs)

    learner.export('saved_model/trained_model.pkl')


