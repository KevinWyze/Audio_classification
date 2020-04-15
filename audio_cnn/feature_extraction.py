#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Feature extractions with augmentations 

Three augmentations are considered:
    Time effect shifting (range from 0.80 to 1.20)
    Pitch shifting (an integer from [-2, -1, 1, 2]) 
    Noise adding (noises from (0.005, 0.008))
"""
import os 
import librosa
from tqdm import tqdm
import numpy as np 
import pandas as pd
import pickle
import argparse

"""
aug_rate: time effect shifting parameters for augmentation per audio
aug_tone: pitch shifting parameters for augmentation per audio
noise_num: add random noises in the audio 
"""


class feature_extractor:
    
    def __init__(self, aug_rate, aug_tone, max_padding, audio_path, data, feature_type, feature_dim):
        self.aug_rate = aug_rate
        self.aug_tone = aug_tone
        self.max_padding = max_padding
        self.audio_path = audio_path
        self.data = data
        self.feature_type = feature_type
        self.feature_dim = feature_dim
        self.feature = []
        self.label = []
        self.test_feature =[]
        self.test_label = []
    
    def collect_feature_training(self):
        for i in tqdm(range(self.data.shape[0])):
            file_path = str(self.data['ID'][i]) + '.wav'
            label = self.data['Class'][i]
            audio = os.path.join(self.audio_path, file_path)
            signal, rate = librosa.load(audio)
            normalized_signal = librosa.util.normalize(signal)
            try:
                self.normalize_feature(normalized_signal, rate, label)
            except:
                print('normalized_feature failed')
            try:
                self.time_effect_feature(normalized_signal, label)
            except:
                print('time_effect_feature failed')
            try:
                self.pitch_shift_feature(normalized_signal,rate, label)
            except:
                print('pitch_shift_feature failed')
            try:
                self.add_noise_feature(normalized_signal,rate, label)
            except:
                print('add_noise_feature failed')

        return self.feature, self.label
    
    def normalize_feature(self,normalized_signal, rate, label):
        if self.feature_type == 'log_mel':
            S = librosa.feature.melspectrogram(normalized_signal, sr=rate, n_mels= self.feature_dim)
            mel = librosa.power_to_db(S, ref=np.max)
            # Convert sound intensity to log amplitude:
            mel_db = librosa.amplitude_to_db(abs(mel))
            # Normalize between -1 and 1
            normalized_feature = librosa.util.normalize(mel_db)
        if self.feature_type =='mfcc':
            mfcc = librosa.feature.mfcc(normalized_signal, sr=rate, n_mfcc= self.feature_dim)
            normalized_feature = librosa.util.normalize(mfcc)
        
        self.add_feature(normalized_feature, label)
    def time_effect_feature(self, normalized_signal,label):
        for rate in self.aug_rate:
            time_effect_signal = librosa.effects.time_stretch(normalized_signal, rate=rate)

            if self.feature_type == 'log_mel':                
                S = librosa.feature.melspectrogram(time_effect_signal, sr=rate, n_mels=self.feature_dim)
                mel = librosa.power_to_db(S, ref=np.max)
                # Convert sound intensity to log amplitude:
                mel_db = librosa.amplitude_to_db(abs(mel))
                # Normalize between -1 and 1
                time_effect_feature = librosa.util.normalize(mel_db)
                self.add_feature(time_effect_feature, label)  
            if self.feature_type == 'mfcc':
                mfcc = librosa.feature.mfcc(time_effect_signal, sr=rate, n_mfcc= self.feature_dim)
                time_effect_feature = librosa.util.normalize(mfcc)
                self.add_feature(time_effect_feature, label)  

    def pitch_shift_feature(self, normalized_signal, rate, label):
        for tone in self.aug_tone:
            pitch_shift_signal = librosa.effects.pitch_shift(normalized_signal, sr= rate, n_steps=tone)
            if self.feature_type == 'log_mel':
                S = librosa.feature.melspectrogram(pitch_shift_signal, sr=rate, n_mels=self.feature_dim)
                mel = librosa.power_to_db(S, ref=np.max)
                # Convert sound intensity to log amplitude:
                mel_db = librosa.amplitude_to_db(abs(mel))
                # Normalize between -1 and 1
                pitch_shift_feature = librosa.util.normalize(mel_db)
            if self.feature_type == 'mfcc':
                mfcc = librosa.feature.mfcc(pitch_shift_signal, sr=rate, n_mfcc= self.feature_dim)
                pitch_shift_feature = librosa.util.normalize(mfcc)
            self.add_feature(pitch_shift_feature, label)  
    def add_noise_feature(self, normalized_signal, rate, label):
        noised_signal = self.add_noise(normalized_signal)
        if self.feature_type == 'log_mel':
            S = librosa.feature.melspectrogram(noised_signal, sr=rate, n_mels= self.feature_dim)
            mel = librosa.power_to_db(S, ref=np.max)
            # Convert sound intensity to log amplitude:
            mel_db = librosa.amplitude_to_db(abs(mel))
            # Normalize between -1 and 1
            noised_feature = librosa.util.normalize(mel_db)
        if self.feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(noised_signal, sr=rate, n_mfcc= self.feature_dim)
            noised_feature = librosa.util.normalize(mfcc)
                
        self.add_feature(noised_feature, label)  
        
    
    def add_noise(self, normalized_signal):
        noise = np.random.rand(len(normalized_signal))
        noise_amp = np.random.uniform(0.005, 0.008)
        noised_signal = normalized_signal + (noise_amp * noise)
        return(noised_signal)
    
    
    def add_feature(self, feat, lab):
        shape = feat.shape[1]
        if (self.max_padding > 0 and shape < self.max_padding):
            xDiff = self.max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            feat = np.pad(feat, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        if (self.max_padding > 0 and shape > self.max_padding):
            feat = feat[:,0:174]
        self.feature.append(feat)
        self.label.append(lab)
    
    def collect_feature_test(self, test_audio, test_data, labeled):
        

        
        for i in tqdm(range(test_data.shape[0])):
            file_path = str(test_data.iloc[i]['ID']) + '.wav'
            audio = os.path.join(test_audio, file_path)
            
            signal, rate = librosa.load(audio)
            
            normalized_signal = librosa.util.normalize(signal)

            if self.feature_type == 'log_mel':
                S = librosa.feature.melspectrogram(normalized_signal, sr=rate, n_mels= self.feature_dim)
                mel = librosa.power_to_db(S, ref=np.max)
                # Convert sound intensity to log amplitude:
                mel_db = librosa.amplitude_to_db(abs(mel))
                # Normalize between -1 and 1
                normalized_feature = librosa.util.normalize(mel_db)
            if self.feature_type =='mfcc':
                mfcc = librosa.feature.mfcc(normalized_signal, sr=rate, n_mfcc= self.feature_dim)
                normalized_feature = librosa.util.normalize(mfcc)
            shape = normalized_feature.shape[1]

            if (self.max_padding > 0 and shape < self.max_padding):
                xDiff = self.max_padding - shape
                xLeft = xDiff//2
                xRight = xDiff-xLeft
                normalized_feature = np.pad(normalized_feature, pad_width=((0,0), (xLeft, xRight)), mode='constant')
            self.test_feature.append(normalized_feature)
            
            if labeled == True:
                self.test_label.append(test_data.iloc[i]['Class'])

        return self.test_feature, self.test_label
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type = str, default = "/efs/kevin/audio/train.csv", help ="Path to train dataset")
    #parser.add_argument("--test_data", type = str, default = "/efs/kevin/audio/test.csv", help ="Path to test dataset")
    
    parser.add_argument("--train_audio", type=str, default="/efs/kevin/audio/Train/", help="Path to train audio folder")
    parser.add_argument("--test_audio", type=str, default="/efs/kevin/audio/Train/", help="Path to test audio folder")
    
    
    parser.add_argument("--feature_folder", type=str, default= '/efs/kevin/audio/features/', help="Path to feature folder")
    parser.add_argument("--feature_type", type=str, default="mfcc", help="Feature type to extract")
    parser.add_argument("--feature_num", type=int, default= 40, help="Number of features to extract")

    opt = parser.parse_args()
    print(opt)


    aug_rate = [0.81, 1.07]
    aug_tone = [-1, -2, 1, 2]
    data = pd.read_csv(opt.train_data)
    train_data = data[0:4435]
    test_data = data[4435:5535]

    
    feature_extract = feature_extractor(aug_rate, aug_tone, 174, opt.train_audio, train_data, opt.feature_type, opt.feature_dim)
    X_train, y_train = feature_extract.collect_feature_training()
    X_test, y_test = feature_extract.collect_feature_test(opt.test_audio, test_data, True)

    train_feature = opt.feature_folder + 'X_train_' + opt.feature_type + '_' + str(opt.feature_dim) + '_aug' + '.pickle'
    train_label = opt.feature_folder + 'Y_train' + '.pickle'
    
    X_feature = open(train_feature,"wb")
    pickle.dump(X_train, X_feature)
    X_feature.close()
    Y_label = open(train_label,"wb")
    pickle.dump(y_train, Y_label)
    Y_label.close()
    
    test_feature = opt.feature_folder + 'X_test_' + opt.feature_type + '_' + str(opt.feature_dim) + '.pickle'
    test_label = opt.feature_folder + 'Y_test' + '.pickle'
    X_feature = open(test_feature,"wb")
    pickle.dump(X_test, X_feature)
    X_feature.close()
    Y_label = open(test_label,"wb")
    pickle.dump(y_test, Y_label)
    Y_label.close()






