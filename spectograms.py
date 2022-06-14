import numpy as np
import librosa.display

import matplotlib.pyplot as plt
import os

from pydub import AudioSegment

os.makedirs('generated_data/spectrograms3sec')

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']

for genre in genres:
    os.makedirs(os.path.join('generated_data/3sec', genre))
    os.makedirs(os.path.join('generated_data/spectrograms3sec', genre))

for genre in genres:
    for it, filename in enumerate(os.listdir(os.path.join('Data/genres_original', genre))):
        song = os.path.join('Data/genres_original/'+genre, filename)
        newAudio = AudioSegment.from_wav(song)
        for w in range(10):
            new = newAudio[w*3000:(w+1)*3000]
            new.export('generated_data/3sec/'+genre+'/'+genre + str(it) + str(w)+'.wav', format="wav")

for genre in genres:
    for it, filename in enumerate(os.listdir(os.path.join('content/3sec', genre))):
        song = os.path.join('generated_data/3sec/genre', filename)
        y, sr = librosa.load(song, duration=3)
        plt.imshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max))
        plt.axis("off")
        plt.savefig('./generated_data/spectrograms3sec/'+genre+'/'+genre + str(it)+'.png', bbox_inches="tight", pad_inches=0)
