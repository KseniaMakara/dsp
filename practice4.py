import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = 'birdsMountain.wav'
y, sr = librosa.load(audio_path)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')

plt.subplot(2, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

plt.tight_layout()
plt.show()

bird_freq_range = (1000, 5000)  

def frequency_filter(signal, sr, freq_range):
    nyquist = 0.5 * sr
    low, high = freq_range[0] / nyquist, freq_range[1] / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal)
    return filtered_signal

filtered_audio = frequency_filter(y, sr, bird_freq_range)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
librosa.display.waveshow(filtered_audio, sr=sr)
plt.title('Filtered Waveform')

plt.subplot(2, 1, 2)
D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(filtered_audio)), ref=np.max)
librosa.display.specshow(D_filtered, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Filtered Spectrogram')

plt.tight_layout()
plt.show()
