import os
from distutuls.dir_util import copy_tree
import shutil

if __name__ == '__main__':
    path_list = os.listdir()
    path_list = [item for item in path_list if len(item) == 1]
    os.mkdir('train')
    os.mkdir('valid')

    fromDirectory = "/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/1/train"
    toDirectory = "/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/train"
    copy_tree(fromDirectory, toDirectory)

    fromDirectory = "/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/1/valid"
    toDirectory = "/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/valid"
    copy_tree(fromDirectory, toDirectory)

    labels = ['air_conditioner', 'car_horn', 'children_playing',
              'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

    for i in range(2, 10):
        for label in labels:
            source = '/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/{}/train/{}/'.format(i, label)
            dest = '/Users/kli/Desktop/AI tasks/urbansound/UrbanSound8K/data/train/{}/'.format(label)
            files = os.listdir(source)
            for f in files:
                new_name = '{}-'.format(i) + f
                os.rename(r'{}{}'.format(source, f), r'{}{}'.format(source, new_name))
                shutil.move(source + '/' + new_name, dest)