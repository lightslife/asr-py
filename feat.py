import librosa

y, sr = librosa.load('./sample.wav', sr=16000)
# 提取 MFCC feature
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

print(mfccs.shape)        # (40, 65)
print(mfccs)    
