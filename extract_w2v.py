import numpy as np, torch, librosa, glob, os, sys
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

NAME = "facebook/wav2vec2-base"
fe = Wav2Vec2FeatureExtractor.from_pretrained(NAME)
mdl = Wav2Vec2Model.from_pretrained(NAME).eval()

def embed(path):
    y, _ = librosa.load(path, sr=16000, duration=3, offset=0.5)
    inp = fe(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = mdl(**inp).last_hidden_state[0]
    return torch.cat([h.mean(0), h.std(0)]).numpy()

EMO = {'01':'neutral','02':'calm','03':'happy','04':'sad',
       '05':'angry','06':'fearful','07':'disgust','08':'surprised'}

files = sorted(glob.glob("data/ravdess/Actor_*/*.wav"))
print("Files:", len(files))
X, y, actors = [], [], []
for i, p in enumerate(files):
    parts = os.path.basename(p)[:-4].split("-")
    X.append(embed(p)); y.append(EMO[parts[2]]); actors.append(int(parts[6]))
    if i % 100 == 0: print(i, "/", len(files), flush=True)

np.save("Xw.npy", np.array(X))
np.save("yw.npy", np.array(y))
np.save("actorsw.npy", np.array(actors))
print("Done:", np.array(X).shape)