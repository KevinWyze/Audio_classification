import argparse
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import * 

class CnnLearner():
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def train_fold(self, fold, bs, epoch):
        data_train = ImageDataBunch.from_folder(self.data_dir / fold, ds_tfms=[], size=224, bs=bs)
        data_train.normalize(imagenet_stats)
        self.learn = cnn_learner(data_train, models.resnet34, metrics=error_rate)
        self.learn.fit_one_cycle(epoch, max_lr=slice(3e-6, 3e-4))
    
    def predict(self, fold, model_path, image_path):
        data_train = ImageDataBunch.from_folder(self.data_dir / fold, size = 224)
        learn = cnn_learner(data_train, models.resnet34, metrics = error_rate)
        learn.load(model_path)
        print(learn.predict(image_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/ubuntu/kevin_folder/Audio_classification/UrbanSound8K/data",
                        help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs for each fold")
    opt = parser.parse_args()
    print(opt)

    learner = CnnLearner(opt.dataset_path)

    #for i in range(1, 10):
    #    learner.train_fold('fold'+str(i), opt.batch_size, opt.num_epochs)

    #learner.learn.save('/home/ubuntu/kevin_folder/Audio_classification/saved_model/trained_model')
    
    learner.predict('fold1', '/saved_model/trained_model', '/home/ubuntu/kevin_folder/Audio_classification/UrbanSound8K/data/fold1/train/air_conditioner/166942-0-0-0.png')

        
