# ran into this issue
# https://github.com/librosa/librosa/issues/219
# if using mac/linux I had to install ffmpeg to parse the audio. If using windows you might need to do something else. Librosa uses a 3rd party app to load audio files, and it doesn't tell you how to fix errors if there are any with loading 
# the audio files. 

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load


import librosa
import librosa.display
import sklearn

import numpy as np
from collections import Counter, defaultdict
import time
import math
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import xml.dom.minidom
import os
from os.path import exists
from collections import defaultdict

respack_folders = []
for directory, sub_directories, files in os.walk('./Respacks/'):
    for folder in sub_directories:
        if folder != 'Songs' and folder != 'characters' and folder != 'Images':
            respack_folders.append(folder)
print(respack_folders)

rhythm_map = defaultdict(str)
build_up_rhythm_map = defaultdict(str)

for folder in respack_folders:
    path_name = './Respacks/' + folder + '/songs.xml'
    print(path_name)
    if exists(path_name):
        song_xml = xml.dom.minidom.parse(path_name)
        songs = song_xml.getElementsByTagName('song')
        for song in songs:
            # parsing rhythms
            curr_song_name = song.getAttribute('name')
            if len(song.getElementsByTagName('rhythm')) > 0:
                rhythm_map[curr_song_name] = song.getElementsByTagName('rhythm')[0].firstChild.nodeValue
            if len(song.getElementsByTagName('buildup')) > 0:
                build_up_name = song.getElementsByTagName('buildup')[0].firstChild.nodeValue
                if len(song.getElementsByTagName('buildupRhythm')) > 0:
                    build_up_rhythm_map[build_up_name] = song.getElementsByTagName('buildupRhythm')[0].firstChild.nodeValue
            
    else:
        print('bad path name ' + str(path_name))

print(rhythm_map.items())
print(build_up_rhythm_map.items())

   
def parseSong(y, sr):
    T = 30.0    # seconds
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz
    D = librosa.stft(y)
    ret_dict = dict()
    ret_dict['D_harmonic4'], ret_dict['D_percussive4'] = librosa.decompose.hpss(D, margin=4)
    ret_dict['D_harmonic16'], ret_dict['D_percussive16'] = librosa.decompose.hpss(D, margin=16)
    # ret_dict['spectral_centroids'] = librosa.feature.spectral_centroid(x, sr=sr)[0]
    # ret_dict['spectral_rolloff'] = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    
    
    #spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    #spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
    #spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
    #ret_dict['spectral_bandwidth_2'] = spectral_bandwidth_2
    #ret_dict['spectral_bandwidth_3'] = spectral_bandwidth_3
    #ret_dict['spectral_bandwidth_4'] = spectral_bandwidth_4
    return ret_dict

# dict(audio_name -> dict(features -> values))
audio_rhythm_feature_map = defaultdict(dict)
audio_build_up_rhythm_feature_map = defaultdict(dict)

for folder in respack_folders:
    path_name = './Respacks/' + folder + '/songs.xml'
    print(path_name)
    if exists(path_name):
        song_xml = xml.dom.minidom.parse(path_name)
        songs = song_xml.getElementsByTagName('song')
        for song in songs:
            # parsing rhythms
            curr_song_name = song.getAttribute('name')
            song_path_name = None
            is_rhythm = False
            is_build_up = False
            if len(song.getElementsByTagName('rhythm')) > 0:
                is_rhythm = True
                song_path_name = path_name = './Respacks/' + folder +'/Songs/' + curr_song_name + '.mp3'
            
            if len(song.getElementsByTagName('buildup')) > 0:
                is_build_up - True
                build_up_name = song.getElementsByTagName('buildup')[0].firstChild.nodeValue
                song_path_name = path_name = './Respacks/' + folder +'/Songs/' + build_up_name + '.mp3'
                    
            # load the song
            if exists(song_path_name):
                print('parsing ' + str(song_path_name))
                try:
                    y, sr = librosa.load(song_path_name, duration=100)
                    if is_rhythm:
                        audio_rhythm_feature_map[curr_song_name] = parseSong(y,sr)
                    elif is_build_up:
                        audio_build_up_rhythm_feature_map[build_up_name] = parseSong(y,sr)
                except Exception as e:
                    print('exception occurred')
                    print(e)
            else:
                print('file name doesn\'t exist ' + rhythm_song_path_name)
    else:
        print('bad path name ' + str(path_name))
        
# features = audio features
# labels = rhythm/buildup
features = []
labels = []
for song_name in rhythm_map.keys():
    if song_name in audio_rhythm_feature_map:
        # if no features were extracted for some reason, then we can't process it.
        if len(audio_rhythm_feature_map[song_name]) != 0:
            list_to_add = list()
            for feature in audio_rhythm_feature_map[song_name]:
                audio_features_flattened = np.array(audio_rhythm_feature_map[song_name][feature]).flatten()
                for x in audio_features_flattened:
                    complex_to_real = x.real + x.imag
                    list_to_add.append(complex_to_real)
            if len(list_to_add) == 885600:
                labels.append(rhythm_map[song_name])
                features.append(list_to_add)
print(len(labels))
print(len(features))



songs_to_process = len(labels)
post_processed_labels = []
post_processed_features = []
for index in range(songs_to_process):
    for beat_index in range(len(labels[index])):
        list_to_add = []
        beat_length = len(features[index])//len(labels[index])
        for feature_index in range(beat_index*beat_length, (beat_index+1)*beat_length):
            value = features[index][feature_index]
            list_to_add.append(value)
        if len(list_to_add) == 3459:
            post_processed_labels.append(labels[index][beat_index])
            post_processed_features.append(list_to_add)
            
    #print(len(post_processed_labels[index]))
    #print(len(post_processed_features[index]))
print(len(post_processed_labels))
print(len(post_processed_features))

scaler = StandardScaler()
scaler.fit(post_processed_features)
X_train = scaler.transform(post_processed_features)
# print(X_train[0:10])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000000)
clf.fit(X_train, post_processed_labels)

dump(clf, 'neural_net_harmonic_and_percussive_3459.joblib')




post_processed_labels = []
post_processed_features = []
for index in range(songs_to_process):
    for beat_index in range(len(labels[index])):
        list_to_add = []
        beat_length = len(features[index])//len(labels[index])
        for feature_index in range(beat_index*beat_length, (beat_index+1)*beat_length):
            value = features[index][feature_index]
            list_to_add.append(value)
        if len(list_to_add) == 317:
            post_processed_labels.append(labels[index][beat_index])
            post_processed_features.append(list_to_add)
            
    #print(len(post_processed_labels[index]))
    #print(len(post_processed_features[index]))
print(len(post_processed_labels))
print(len(post_processed_features))

scaler = StandardScaler()
scaler.fit(post_processed_features)
X_train = scaler.transform(post_processed_features)
# print(X_train[0:10])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000000)
clf.fit(X_train, post_processed_labels)

dump(clf, 'neural_net_harmonic_and_percussive_317.joblib')




post_processed_labels = []
post_processed_features = []
for index in range(songs_to_process):
    for beat_index in range(len(labels[index])):
        list_to_add = []
        beat_length = len(features[index])//len(labels[index])
        for feature_index in range(beat_index*beat_length, (beat_index+1)*beat_length):
            value = features[index][feature_index]
            list_to_add.append(value)
        if len(list_to_add) == 1729:
            post_processed_labels.append(labels[index][beat_index])
            post_processed_features.append(list_to_add)
            
    #print(len(post_processed_labels[index]))
    #print(len(post_processed_features[index]))
print(len(post_processed_labels))
print(len(post_processed_features))

scaler = StandardScaler()
scaler.fit(post_processed_features)
X_train = scaler.transform(post_processed_features)
# print(X_train[0:10])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000000)
clf.fit(X_train, post_processed_labels)

dump(clf, 'neural_net_harmonic_and_percussive_1729.joblib')


post_processed_labels = []
post_processed_features = []
for index in range(songs_to_process):
    for beat_index in range(len(labels[index])):
        list_to_add = []
        beat_length = len(features[index])//len(labels[index])
        for feature_index in range(beat_index*beat_length, (beat_index+1)*beat_length):
            value = features[index][feature_index]
            list_to_add.append(value)
        if len(list_to_add) == 1581:
            post_processed_labels.append(labels[index][beat_index])
            post_processed_features.append(list_to_add)
            
    #print(len(post_processed_labels[index]))
    #print(len(post_processed_features[index]))
print(len(post_processed_labels))
print(len(post_processed_features))

scaler = StandardScaler()
scaler.fit(post_processed_features)
X_train = scaler.transform(post_processed_features)
# print(X_train[0:10])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000000)
clf.fit(X_train, post_processed_labels)

dump(clf, 'neural_net_harmonic_and_percussive_1581.joblib')