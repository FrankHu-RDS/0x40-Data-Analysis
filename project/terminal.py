# ran into this issue
# https://github.com/librosa/librosa/issues/219
# if using mac/linux I had to install ffmpeg to parse the audio. If using windows you might need to do something else. Librosa uses a 3rd party app to load audio files, and it doesn't tell you how to fix errors if there are any with loading 
# the audio files. 

import numpy as np
import matplotlib.pyplot as plt

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


from xml.dom import minidom
import os
from os.path import exists
from collections import defaultdict
from joblib import dump, load


import pickle

### Initializations ###
song_dictionary = defaultdict(dict)
folder_path = './songs'
clf_3459 = load('neural_net_harmonic_and_percussive_3459.joblib')
clf_317 = load('neural_net_harmonic_and_percussive_317.joblib')
clf_1729 = load('neural_net_harmonic_and_percussive_1729.joblib')
clf_1581 = load('neural_net_harmonic_and_percussive_1581.joblib')
def ddict():
    return defaultdict(ddict)
beatmap_dictionary = ddict()
feature_process = 'D_harmonic16'
curr_clf = clf_3459
curr_length = 3459         
beat_glossary = dict()
beat_glossary['x'] = 'Vertical blur (snare)'
beat_glossary['o'] = 'Horizontal blur (bass)'
beat_glossary['-'] = 'No blur'
beat_glossary['+'] = 'Blackout'
beat_glossary['¤'] = 'Whiteout'
beat_glossary['|'] = 'Short blackout'
beat_glossary[':'] = 'Color only'
beat_glossary['*'] = 'Image only'
beat_glossary['X'] = 'Vertical blur only'
beat_glossary['O'] = 'Horizontal blur only'
beat_glossary[')'] = 'Trippy cirle in'
beat_glossary['('] = 'Trippy circle out'
beat_glossary['~'] = 'Fade color'
beat_glossary['='] = 'Fade and change image'
beat_glossary['i'] = 'Invert all colours'
beat_glossary['I'] = 'Invert & change image'
beat_glossary['s'] = 'Horizontal slice'
beat_glossary['S'] = 'Horizontal slice and change image'
beat_glossary['v'] = 'Vertical slice'
beat_glossary['V'] = 'Vertical slice and change image'
beat_glossary['@'] = 'Double slice and change image'

### FUNCTIONS ###
# Set up a loop where users can choose what they'd like to do.
def display_title_bar():
    # Clears the terminal screen, and displays a title bar.
    os.system('clear')
    print("\t**********************************************")
    print("\t***   Hues AI System - Beatmap Generator   ***")
    print("\t**********************************************")
    
def get_user_choice():
    # Let users know what they can do.
    print("\n[1] Select a Folder path. Current folder path is " + folder_path)
    print("[2] Select feature, current feature : " + feature_process )
    print("[3] Choose beat frequency. current frequency : " + str(curr_length))
    print("[4] Process Songs.")
    print("[5] Create Beatmaps in songs.xml file.")
    print("[q] Quit.")
    
    return input("What would you like to do? ")
    
def select_feature():
    global feature_process
    print("\n[1] Harmonic")
    print("[2] Percussive")
    print('[3] Mixture')
    print('Other audio features not implemented yet : spectral_centroids, spectral_rolloff, spectral_bandwidths')
    # Let users know what they can do.
    user_input = input("Select feature to parse audio by ")
    if user_input == '1':
        feature_process = 'D_harmonic16'
    elif user_input == '2':
        feature_process = 'D_percussive16'
    else:
        feature_process = 'Mixture'

def get_folder_path():
    # Asks the user for a new name, and stores the name if we don't already
    #  know about this person.
    global folder_path
    folder_path = input("\nPlease put in your folder path: ")
   
def parseSong(y, sr):
    T = 30.0    # seconds
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz
    D = librosa.stft(y)
    ret_dict = dict()
    # ret_dict['D_harmonic4'], ret_dict['D_percussive4'] = librosa.decompose.hpss(D, margin=4)
    ret_dict['D_harmonic16'], ret_dict['D_percussive16'] = librosa.decompose.hpss(D, margin=16)
    # ret_dict['spectral_centroids'] = librosa.feature.spectral_centroid(x, sr=sr)[0]
    # ret_dict['spectral_rolloff'] = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    # spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    # spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
    # spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
    # ret_dict['spectral_bandwidth_2'] = spectral_bandwidth_2
    # ret_dict['spectral_bandwidth_3'] = spectral_bandwidth_3
    # ret_dict['spectral_bandwidth_4'] = spectral_bandwidth_4
    return ret_dict

def merge():
    print('merge function')
    for song_name in song_dictionary.keys():
        harmonic_res = beatmap_dictionary['D_harmonic16'][song_name]
        percussive_res = beatmap_dictionary['D_percussive16'][song_name]
        final_res = []
        print(percussive_res)
        for index in range(len(percussive_res)):
            if percussive_res[index] != ':' and percussive_res[index] != '.':
                final_res.append(percussive_res[index])
            else:
                final_res.append(harmonic_res[index])
        beatmap_dictionary['Mixture'][song_name] = final_res[:]

def process_songs():
    for filename in os.listdir(folder_path):
        print(filename)
        try:
            print(folder_path + '/' + filename)
            y, sr = librosa.load(folder_path + '/' + filename)
            song_dictionary[filename] = parseSong(y, sr)
        except Exception as e:
            print('exception occurred with ' + str(filename))
            print(e)
    if feature_process == 'D_harmonic16':
        predict_with_feature('D_harmonic16')
    if feature_process == 'D_percussive16':
        predict_with_feature('D_percussive16')
    if feature_process == 'Mixture':
        predict_with_feature('D_percussive16')
        predict_with_feature('D_harmonic16')
        merge()


def post_process(res):
    print(''.join(res))
    range_length = 6
    for index in range(0,len(res),range_length):
        num_of_colons = 0
        for index_2 in range(index,index+range_length):
            if index_2 < len(res):
                if res[index_2] == ':':
                    num_of_colons += 1
        if num_of_colons >= 4:
            for index_2 in range(index,index+range_length):
                if index_2 < len(res):
                    if res[index_2] == ':':
                        res[index_2] = '.'

    # let's give it some flavor and convert some of those items into os
    for index in range(0,len(res)):
        if res[index] == 'x' or res[index] == ':':
            random = np.random.randint(3)
            if random == 1:
                res[index] = np.random.choice(list(beat_glossary.keys()))
                
    # let's give it some flavor and convert some of those items into os
    for index in range(0,len(res)):
        if res[index] == ':':
            random = np.random.randint(9)
            if random == 3:
                res[index] = np.random.choice(list(beat_glossary.keys()))
    
    for index in range(0,len(res)):
        if np.random.randint(3) == 2:
            res[index] = '.'
    song_length_cut = 7
    for index in range(len(res)-song_length_cut,len(res)):
        res[index] = '.'
    for index in range(0,song_length_cut):
        res[index] = '.'
    print('post processed')
    for index in range(len(res)):
        print(res[index], end='')
    

def predict_with_feature(feature_type):
    for song_name in song_dictionary.keys():
        flattened_feature_input = []
        single_feature_input = []

        single_song = song_dictionary[song_name]
        flattened_feature_input.extend(single_song[feature_type])

        flattened_feature_input = np.array(flattened_feature_input).flatten()

        for index in range(0, len(flattened_feature_input), curr_length):
            single_feature_input.append([])
            for x in flattened_feature_input[index:index+curr_length]:
                complex_to_real = x.real + x.imag
                single_feature_input[-1].append(complex_to_real)
        single_feature_input.pop()

        scaler = StandardScaler()
        scaler.fit(single_feature_input)
        single_feature_normalized = scaler.transform(single_feature_input)
        print(feature_type)
        res = curr_clf.predict(np.array(single_feature_normalized))
        post_process(res)
        print(res)
        beatmap_dictionary[feature_type][song_name] = res[:]


'''
<songs>
  <song name="loop_ThisCityisKillingMe">
    <title>Dusty Brown - This City is Killing Me</title>
    <rhythm>o..:.:o.x...x...o.:.:.:.x.:.:.:.o.:.-.o.x...x.:.o...o...x...x...o..:.:o.x...x...o.+..o..x.:...o.o+.x-.o.x.x..x..o...x.:.:...:...o..:.:o.x...x...o.:.:.:.x.:.:.:.o.:.-.o.x...o.:.o.:.:.:.:...-+..o.:....x....o...o---x-.-x.....:.o...-.o.x...o.:.o...o.:.o...:.:.</rhythm>
  </song>
</songs>
'''
def create_beatmaps():
    # just print it out
    '''
    f = open("song_output.txt", "w")
    for song_name in song_dictionary.keys():
        beats = ''.join(beatmap_dictionary[feature_process][song_name])
        print(beats)
        f.write(song_name + ' : ')
        f.write(beats)
        f.write('\n')
    '''
    root = minidom.Document()
    xml = root.createElement('songs')
    for song_name in song_dictionary.keys():
        beat_map = ''.join(beatmap_dictionary[feature_process][song_name])
        song_child = root.createElement('song')
        song_child.setAttribute('name', song_name[0:len(song_name)-4])

        title = root.createElement('title')
        title_text = root.createTextNode(song_name[0:len(song_name)-4])
        title.appendChild(title_text)
        rhythm = root.createElement('rhythm')
        rhythm_text = root.createTextNode(beat_map)
        rhythm.appendChild(rhythm_text)
        song_child.appendChild(title)
        song_child.appendChild(rhythm)
        xml.appendChild(song_child)
    root.appendChild(xml)
    xml_str = root.toprettyxml(indent ="\t") 
    with open('./songs.xml', "w") as f:
        f.write(xml_str) 
    
def choose_frequency():
    global curr_clf
    global curr_length
    print("\n[1] low beat frequency")
    print("[2] medium_1729 beat frequency")
    print("[3] medium_1581 beat frequency")
    print('[4] high beat frequency')
    print('Other audio features not implemented yet : spectral_centroids, spectral_rolloff, spectral_bandwidths')
    # Let users know what they can do.
    choice = input("Select feature to parse audio by ")
    if choice == '1':
        curr_clf = clf_317
        curr_length = 317
    elif choice == '2': # medium
        curr_clf = clf_1729
    elif choice == '3': # medium
        curr_clf = clf_1581
        curr_length = 1581
    elif choice == '4': # high
        curr_clf = clf_3459
        curr_length = 3459
### MAIN PROGRAM ###

choice = ''
display_title_bar()
while choice != 'q':    
    choice = get_user_choice()
    # Respond to the user's choice.
    display_title_bar()
    if choice == '1':
        get_folder_path()
    elif choice == '2':
        select_feature()
    elif choice == '3': # choose frequency
        choose_frequency()
    elif choice == '4': # Process songs
        process_songs()
    elif choice == '5': # create beatmaps
        create_beatmaps()
    elif choice == 'q':
        print("\nGood bye")
    else:
        print("\nI didn't understand that choice.\n")