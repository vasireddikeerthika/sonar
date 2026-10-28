import glob, os
import numpy as np
import librosa

EMO = {'01':'neutral','02':'calm','03':'happy','04':'sad',
       '05':'angry','06':'fearful','07':'disgust','08':'surprised'}

def features(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)
    def stats(m):
        return np.hstack([m.mean(axis=1), m.std(axis=1)])
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    dmf    = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel    = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
    zcr    = librosa.feature.zero_crossing_rate(y)
    rms    = librosa.feature.rms(y=y)
    return np.hstack([stats(mfcc), stats(dmf), stats(chroma),
                      stats(mel), stats(zcr), stats(rms)])

X, y, actors = [], [], []
files = glob.glob("data/ravdess/Actor_*/*.wav")

for i, p in enumerate(files):
    parts = os.path.basename(p)[:-4].split("-")
    X.append(features(p))
    y.append(EMO[parts[2]])
    actors.append(int(parts[6]))
    if i % 100 == 0:
        print(i, "/", len(files))

np.save("X.npy", np.array(X))
np.save("y.npy", np.array(y))
np.save("actors.npy", np.array(actors))
print("Done:", np.array(X).shape)