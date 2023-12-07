import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def hanning_window(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

def spectrogram(signal, window_size, overlap):
    hop_size = window_size - overlap
    num_windows = int(np.ceil(len(signal) / hop_size))
    padded_length = num_windows * hop_size + window_size
    signal = np.pad(signal, (0, padded_length - len(signal)))

    specgram = []

    for i in range(0, len(signal) - window_size + 1, hop_size):
        windowed_signal = signal[i:i + window_size] * hanning_window(window_size)
        spectrum = np.abs(np.fft.fft(windowed_signal))[:window_size // 2]
        specgram.append(spectrum)

    return np.array(specgram)

# Завантаження аудіофайлу
sample_rate, signal = wavfile.read("sample.wav")

# Параметри спектрограми
window_size = 512
overlap = 256

# Створення та відображення спектрограми
specgram = spectrogram(signal, window_size, overlap)
plt.imshow(np.log(specgram.T + 1), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Log Magnitude')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Custom Spectrogram')
plt.show()