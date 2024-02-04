## https://alphacephei.com/vosk/install

import wave
import json
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import speech_recognition as sr
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

SetLogLevel(-1)  # Reduce Vosk log verbosity

# Define paths
model_path = "C:/Users/User/Downloads/SpeechQ/vosk-model-en-us-0.42-gigaspeech"
audio_file_path = "C:/Users/User/Downloads/SpeechQ/1 Minute Aphasia Tips： Music and Speech.wav"

def check_and_adjust_audio(audio_file_path):
    try:
        with wave.open(audio_file_path, "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()

            adjusted_file_path = audio_file_path

            if channels != 1 or framerate != 16000:
                print("Adjusting audio file.")
                audio = AudioSegment.from_wav(audio_file_path)
                if channels != 1:
                    audio = audio.set_channels(1)
                if framerate != 16000:
                    audio = audio.set_frame_rate(16000)
                adjusted_file_path = audio_file_path.replace(".wav", "_adjusted.wav")
                audio.export(adjusted_file_path, format="wav")

            return adjusted_file_path
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

def apply_noise_reduction(audio_file_path):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            r.adjust_for_ambient_noise(source)
            audio_data = r.record(source)

        denoised_file_path = audio_file_path.replace(".wav", "_denoised.wav")
        with wave.open(denoised_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data.get_wav_data())

        return denoised_file_path
    except Exception as e:
        print(f"Error applying noise reduction: {e}")
        return None

def recognize_speech_vosk(audio_file_path, model_path):
    print(audio_file_path)
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    with wave.open(audio_file_path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
        else:
            result = json.loads(rec.FinalResult())
    return result

def visualize_with_words(audio_file_path, result):
    signal, sr = librosa.load(audio_file_path, sr=None)
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    
    if 'result' in result:
        for word_info in result['result']:
            word = word_info['word']
            start_time = word_info['start']
            end_time = word_info['end']
            conf = word_info['conf']
            print(f"\'{word}\', [{start_time}, {end_time}], conf. {conf}")
            # Annotate word and confidence with larger font size
            plt.annotate(f"{word}\n{conf:.2f}", (start_time + 0.05, 2000), color='white', fontsize=8, ha='center')
            # Draw a line for the word's duration
            plt.plot([start_time, end_time], [1900, 1900], color=np.random.rand(3,) * 0.8 + 0.2, linewidth=2)

    plt.tight_layout()
    plt.show()

def estimate_audio_quality(audio_file_path, result):
    signal, sr = librosa.load(audio_file_path, sr=None)
    non_word_signal = np.array([])
    word_signal = np.array([])
    word_durations = [(word_info['start'], word_info['end']) for word_info in result.get('result', [])]
    total_words = len(word_durations)

    # Split signal into word and non-word segments
    for i, (start_time, end_time) in enumerate(word_durations):
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        word_signal = np.concatenate((word_signal, signal[start_sample:end_sample]))
        
        if i == 0:
            non_word_signal = np.concatenate((non_word_signal, signal[:start_sample]))
        else:
            prev_end_sample = int(word_durations[i-1][1] * sr)
            non_word_signal = np.concatenate((non_word_signal, signal[prev_end_sample:start_sample]))
        
        if i == len(word_durations) - 1:
            non_word_signal = np.concatenate((non_word_signal, signal[end_sample:]))

    # Calculate noise level in non-word segments
    noise_rms = librosa.feature.rms(y=non_word_signal)
    noise_level = np.sum(noise_rms) / np.sqrt(len(non_word_signal))

    # Calculate volume in word segments
    word_rms = librosa.feature.rms(y=word_signal)
    adjusted_word_rms = word_rms - noise_level
    volume = np.mean(adjusted_word_rms)

    # Calculate median confidence
    confidences = [word_info['conf'] for word_info in result.get('result', [])]
    median_confidence = np.percentile(confidences, 10) if max(confidences)>0 else 0

    # Calculate words per minute
    total_duration_minutes = len(signal) / sr / 60
    words_per_minute = total_words / total_duration_minutes if total_duration_minutes > 0 else 0

    print(f"Noise Level: {noise_level}, Volume: {volume}, Median Confidence: {median_confidence}, Words per Minute: {words_per_minute}")

    return noise_level, volume, median_confidence, words_per_minute


# Process audio file
adjusted_file_path = check_and_adjust_audio(audio_file_path)
denoised_file_path = apply_noise_reduction(adjusted_file_path)

# Recognize speech
result = recognize_speech_vosk(denoised_file_path, model_path)

# Visualize spectrogram with annotated words
visualize_with_words(denoised_file_path, result)

noise_level, volume, median_confidence, words_per_minute = estimate_audio_quality(audio_file_path, result)
