import numpy as np, torch, librosa, glob, os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

FOLDER = r"C:\Users\KEERTHIKA SAI\datasets\cremad\AudioWAV"
NAME = "facebook/wav2vec2-base"
fe = Wav2Vec2FeatureExtractor.from_pretrained(NAME)
mdl = Wav2Vec2Model.from_pretrained(NAME).eval()

EMO = {'NEU':'neutral','HAP':'happy','SAD':'sad',
       'ANG':'angry','FEA':'fearful','DIS':'disgust'}

def embed(path):
    y, _ = librosa.load(path, sr=16000, duration=3, offset=0.5)
    inp = fe(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        h = mdl(**inp).last_hidden_state[0]
    return torch.cat([h.mean(0), h.std(0)]).numpy()

files = sorted(glob.glob(os.path.join(FOLDER, "*.wav")))
print("Files:", len(files), flush=True)
X, y, actors, skipped = [], [], [], 0
for i, p in enumerate(files):
    parts = os.path.basename(p)[:-4].split("_")
    try:
        X.append(embed(p)); y.append(EMO[parts[2]]); actors.append(int(parts[0]))
    except Exception as e:
        skipped += 1; print("Skipped:", os.path.basename(p), e)
    if i % 200 == 0: print(i, "/", len(files), flush=True)
    if i % 1000 == 999:
        np.save("Xwc_part.npy", np.array(X))   # safety checkpoint

np.save("Xwc.npy", np.array(X))
np.save("ywc.npy", np.array(y))
np.save("actorswc.npy", np.array(actors))
print("Done:", np.array(X).shape, "| skipped:", skipped)