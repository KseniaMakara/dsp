import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def create_high_frequency_noise(duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    noise = np.random.normal(0, 1, len(t))
    signal_frequency = 5000.0  
    high_frequency_signal = np.sin(2.0 * np.pi * signal_frequency * t)
    noisy_signal = high_frequency_signal + noise
    return t, noisy_signal

def create_fir_filter(cutoff_frequency, num_taps):
    nyquist_rate = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_rate
    fir_filter = signal.firwin(num_taps, normalized_cutoff, window='hamming')
    return fir_filter

def add_mirror_padding(signal_data, num_taps):
    mirrored_data = np.concatenate((signal_data[-num_taps+1:][::-1], signal_data, signal_data[:num_taps-1][::-1]))
    return mirrored_data

duration = 5.0  
sampling_rate = 44100  
cutoff_frequency = 1000.0 
num_taps = 101 

time, noisy_signal = create_high_frequency_noise(duration, sampling_rate)

fir_filter = create_fir_filter(cutoff_frequency, num_taps)

padded_signal = add_mirror_padding(noisy_signal, num_taps)

filtered_signal = signal.lfilter(fir_filter, 1.0, padded_signal)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(time, noisy_signal, label='Noisy Signal')
plt.title('Original Noisy Signal')

plt.subplot(3, 1, 2)
plt.plot(fir_filter, label='FIR Filter Coefficients')
plt.title('FIR Filter')

plt.subplot(3, 1, 3)
plt.plot(time, filtered_signal[num_taps-1:-num_taps+1], label='Filtered Signal', color='green')
plt.title('Filtered Signal')

plt.tight_layout()
plt.show()