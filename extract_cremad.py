import glob, os
import numpy as np
import librosa

FOLDER = r"C:\Users\KEERTHIKA SAI\OneDrive\Desktop\datasets\cremad\AudioWAV"
EMO = {'NEU':'neutral','HAP':'happy','SAD':'sad',
       'ANG':'angry','FEA':'fearful','DIS':'disgust'}

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

files = sorted(glob.glob(os.path.join(FOLDER, "*.wav")))
print("Files found:", len(files))

X, y, actors, skipped = [], [], [], 0
for i, p in enumerate(files):
    parts = os.path.basename(p)[:-4].split("_")
    try:
        X.append(features(p))
        y.append(EMO[parts[2]])
        actors.append(int(parts[0]))
    except Exception as e:
        skipped += 1
        print("Skipped:", os.path.basename(p), e)
    if i % 500 == 0:
        print(i, "/", len(files))

np.save("X_c.npy", np.array(X))
np.save("y_c.npy", np.array(y))
np.save("actors_c.npy", np.array(actors))
print("Done:", np.array(X).shape, "| skipped:", skipped)