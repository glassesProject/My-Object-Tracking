
import numpy as np
from scipy.io.wavfile import write




sample_rate = 44100
duration = 0.1
frequency = 440

t = np.linspace(0, duration, int(sample_rate * duration), False)
wave = np.sin(2 * np.pi * frequency * t)

audio = (wave * 32767).astype(np.int16)

write("Audio/beep.wav", sample_rate, audio)
