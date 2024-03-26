import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import chirp

def add_sweep_to_speech_dynamic_sr(file_path, sweep_duration, start_freq, end_freq):
    """
    Adds a frequency sweep to the beginning of a speech file using the file's sample rate and saves the result.

    Parameters:
    - file_path: Path to the original speech audio file.
    - sweep_duration: Duration of the sweep in seconds.
    - start_freq: Starting frequency of the sweep in Hz.
    - end_freq: Ending frequency of the sweep in Hz.

    Returns:
    - Path to the resulting audio file with the sweep added.
    """
    # Load the original audio file to get its sample rate
    original_sr, original_signal = read(file_path)

    # Ensure the original signal is in the correct format if it's stereo
    if original_signal.ndim > 1:
        original_signal = original_signal[:, 0]  # Use the first channel if it's stereo
    
    original_signal = np.float64(original_signal) / np.max(np.abs(original_signal)) * 32767
    original_signal_int16 = np.int16(original_signal)
    
    # Generate time axis for the sweep with the original file's sample rate
    t = np.linspace(0, sweep_duration, int(sweep_duration * original_sr))

    # Generate sweep signal
    sweep_signal = chirp(t, f0=start_freq, f1=end_freq, t1=sweep_duration, method='linear')

    # Apply Hann window to the sweep signal
    hann_window = np.hamming(len(sweep_signal))
    sweep_signal = sweep_signal * hann_window

    # Normalize and convert to int16 for WAV format
    sweep_signal_int16 = np.int16(sweep_signal / np.max(np.abs(sweep_signal)) * 32767)

    # Concatenate sweep and original signals
    combined_signal = np.concatenate((sweep_signal_int16, original_signal_int16))

    # Save the combined signal to a new file
    params_str = f"_with_sweep_sd{int(sweep_duration)}sec_sf{int(start_freq)}Hz_ef{int(end_freq)}Hz"
    output_file_path = file_path.rsplit('.', 1)[0] + params_str + ".wav"
    write(output_file_path, original_sr, combined_signal)

    return output_file_path

# Parameters for the sweep
sweep_duration = 30  # seconds
start_freq = 200  # Hz
end_freq = 3000  # Hz
file_path = 'snds/OSR_us_000_0010_8k.wav'
output_file_path = add_sweep_to_speech_dynamic_sr(file_path, sweep_duration, start_freq, end_freq)
