import numpy as np, torch, librosa, glob
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

files = glob.glob("data/ravdess/Actor_01/*.wav")[:3]
import time; t = time.time()
for f in files:
    print(embed(f).shape)
print("sec per clip:", round((time.time() - t) / 3, 2))