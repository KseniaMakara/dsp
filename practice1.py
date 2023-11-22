import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def create_signal(srate, duration, signal_type='rand', **kwargs):
    xtime = np.arange(0, duration, 1/srate)

    if signal_type == 'sine':
        frequency = kwargs.get('frequency', 1)
        amplitude = kwargs.get('amplitude', 1)
        phase = kwargs.get('phase', 0)
        DC = kwargs.get('DC', 0)

        signal = np.zeros_like(xtime)
        for freq, amp, ph, dc in zip(np.atleast_1d(frequency), np.atleast_1d(amplitude),
                                     np.atleast_1d(phase), np.atleast_1d(DC)):
            signal += amp * np.sin(2 * np.pi * freq * xtime + ph) + dc

    elif signal_type == 'rand':
        poles = kwargs.get('poles', 10)
        signal = np.interp(np.linspace(1, poles, len(xtime)), np.arange(poles), np.random.randn(poles))

    else:
        raise ValueError("Wrong signal type. Use 'sine' or 'rand'")

    lintrend = kwargs.get('lintrend', False)
    if lintrend:
        top_margin = np.max(signal)
        bot_margin = np.min(signal)
        ampl = max(2, abs(int(top_margin - bot_margin)))

        trend_line = np.random.randint(ampl-1) + ampl
        signal += np.linspace(-trend_line, trend_line, len(xtime))

    return xtime, signal

def add_noise(signal, noise_amplitude):
    noise = noise_amplitude * np.random.randn(len(signal))
    return signal + noise

def remove_noise(signal, noise_template):
    def model_func(t, *params):
        return params[0] * noise_template

    popt, _ = curve_fit(model_func, np.arange(len(noise_template)), signal)
    fitted_noise = model_func(np.arange(len(signal)), *popt)
    denoised_signal = signal - fitted_noise

    return denoised_signal

# Parameters
srate = 1000  # Hz
duration = 1  # sec
frequency = [1, 2, 3]  # Add more frequencies if needed
amplitude = [1, 0.5, 0.2]  # Corresponding amplitudes
phase = [0, np.pi/2, np.pi]  # Corresponding phases
DC = [0, 0.5, 1]  # Corresponding DC offsets
lintrend = True

# Create signal
xtime, signal = create_signal(srate, duration, signal_type='sine', frequency=frequency,
                              amplitude=amplitude, phase=phase, DC=DC, lintrend=lintrend)

window_size = 100
etime = np.linspace(0, 2, window_size)
artifact = 2 * np.sin(2 * np.pi * 3 * etime) * np.exp(-3 * etime)

idx = np.random.randint(len(signal) - window_size)
noise_template = np.zeros_like(signal)
noise_template[idx : idx + window_size] = artifact
noise_template = add_noise(noise_template, noise_amplitude=0.02)

plt.figure()
plt.plot(noise_template)
plt.legend(['Artifact signal'])
plt.title('Artifact Signal')
plt.show()

# Add noise
artifact_signal = signal + noise_template
artifact_signal = add_noise(artifact_signal, noise_amplitude=0.1)

# Plot signal with noise
plt.figure()
plt.plot(artifact_signal)
plt.title('Artifact Signal with Noise')
plt.show()

# Remove noise using curve_fit
denoised_signal = remove_noise(artifact_signal, noise_template)

# Plot original vs denoised signal
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(xtime, signal, label='Original Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(xtime, denoised_signal, label='Denoised Signal')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(xtime, denoised_signal - signal, label='Difference')
plt.legend()

plt.tight_layout()
plt.show()
