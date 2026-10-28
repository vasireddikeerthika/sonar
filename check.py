import glob
files = glob.glob("data/ravdess/Actor_*/*.wav")
print(len(files))
print(files[0])