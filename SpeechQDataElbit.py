"""
This script processes audio files, specifically targeting multi-channel 'sig.wav' files. 
It adjusts audio properties, applies noise reduction, recognizes speech using the Vosk model, 
visualizes spectrograms segmented by time, estimates audio quality, and saves the results in various formats.

Functions:
- convert_wav_to_mp3: Converts WAV files to MP3 format and saves them in a specified folder.
- check_and_adjust_audio: Checks and adjusts the audio file's channels and framerate for consistency.
- apply_noise_reduction: Applies noise reduction to the audio file to improve speech recognition accuracy.
- recognize_speech_vosk: Uses the Vosk model to perform speech recognition on the audio file.
- visualize_with_words_segmented: Creates and saves segmented spectrogram visualizations with word annotations.
- estimate_audio_quality: Estimates the audio quality based on noise level, volume, confidence, and words per minute.
- save_results_to_text: Saves speech recognition results and audio quality estimates to text files.
- process_channels_and_save_results: Processes each channel of 'sig.wav' files separately and saves the results.
- process_folders: Recursively processes folders containing audio files, applying all the above functions.

This script is designed for audio analysis and transcription projects, facilitating detailed audio processing and analysis.
"""

import wave
import json
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import speech_recognition as sr
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import math

# Reduce Vosk log verbosity
SetLogLevel(-1)

def convert_wav_to_mp3(wav_file_path, file_specific_res_folder):
    mp3_file_path = file_specific_res_folder + '\\snd.mp3'
    if not os.path.exists(mp3_file_path):
        sound = AudioSegment.from_wav(wav_file_path)
        sound.export(mp3_file_path, format="mp3", bitrate="192k")

def check_and_adjust_audio(audio_file_path):
    try:
        with wave.open(audio_file_path, "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            adjusted_file_path = audio_file_path
            if channels != 1 or framerate != 16000:
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

def visualize_with_words_segmented(audio_file_path, result, output_folder):
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    signal, sr = librosa.load(audio_file_path, sr=None)
    total_duration = librosa.get_duration(y=signal, sr=sr)
    segment_duration = 30
    segments = math.ceil(total_duration / segment_duration)
    for segment in range(segments):
        start_time = segment * segment_duration
        end_time = min((segment + 1) * segment_duration, total_duration)
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        S = librosa.feature.melspectrogram(y=signal[start_sample:end_sample], sr=sr, n_mels=64, fmax=4000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(16, 8))
        plt.get_current_fig_manager().full_screen_toggle()
        librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', fmax=4000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-frequency spectrogram: Segment {segment + 1}')
        if 'result' in result:
            for word_info in result['result']:
                word_start = word_info['start']
                word_end = word_info['end']
                if start_time <= word_start <= end_time:
                    adjusted_start = word_start - start_time
                    conf = word_info['conf']
                    plt.annotate(f"{word_info['word']}\n{conf:.2f}", (adjusted_start + 0.05, 2000), color='white', fontsize=8, ha='center')
                    plt.plot([adjusted_start, adjusted_start + (word_end - word_start)], [1900, 1900], color=np.random.rand(3,) * 0.8 + 0.2, linewidth=2)
        plt.tight_layout()
        fig_file_path = os.path.join(output_folder, f"{base_name}_spectrogram_segment_{segment + 1}.png")
        plt.savefig(fig_file_path, bbox_inches='tight')
        plt.close()

def estimate_audio_quality(audio_file_path, result):
    signal, sr = librosa.load(audio_file_path, sr=None)
    non_word_signal, word_signal = np.array([]), np.array([])
    word_durations = [(word_info['start'], word_info['end']) for word_info in result.get('result', [])]
    for i, (start_time, end_time) in enumerate(word_durations):
        start_sample, end_sample = int(start_time * sr), int(end_time * sr)
        word_signal = np.concatenate((word_signal, signal[start_sample:end_sample]))
        if i == 0:
            non_word_signal = np.concatenate((non_word_signal, signal[:start_sample]))
        else:
            prev_end_sample = int(word_durations[i-1][1] * sr)
            non_word_signal = np.concatenate((non_word_signal, signal[prev_end_sample:start_sample]))
        if i == len(word_durations) - 1:
            non_word_signal = np.concatenate((non_word_signal, signal[end_sample:]))
    noise_level = np.std(non_word_signal) / np.sqrt(len(non_word_signal)) if len(non_word_signal) > 0 else 0
    volume = np.std(word_signal) / np.sqrt(len(word_signal)) if len(word_signal) > 0 else 0
    confidences = [word_info['conf'] for word_info in result.get('result', [])]
    confidence_percentile_10 = np.percentile(confidences, 10) if confidences else 0
    words_per_minute = len(word_durations) / (librosa.get_duration(y=signal, sr=sr) / 60)
    return noise_level, volume, confidence_percentile_10, words_per_minute

def save_results_to_text(output_folder, filename_prefix, result, noise_level, volume, confidence_percentile_10, words_per_minute):
    results_path = os.path.join(output_folder, f"{filename_prefix}_recognition_results.txt")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)
    audio_quality_path = os.path.join(output_folder, f"{filename_prefix}_audio_quality.txt")
    with open(audio_quality_path, "w") as f:
        f.write(f"Noise Level: {noise_level}\nVolume: {volume}\nConfidence percentile 10: {confidence_percentile_10}\nWords per Minute: {words_per_minute}")
    srt_path = os.path.join(output_folder, f"{filename_prefix}_subtitles.srt")
    with open(srt_path, "w") as srt_file:
        counter = 1
        for word_info in result.get('result', []):
            start_time, end_time = word_info['start'], word_info['end']
            start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
            srt_file.write(f"{counter}\n{start_srt} --> {end_srt}\n{word_info['word']}\n\n")
            counter += 1

def process_channels_and_save_results(audio_file_path, subdir, res_folder, model_path):
    sound = AudioSegment.from_file(audio_file_path)
    channels = sound.split_to_mono()
    
    for i, channel in enumerate(channels, start=1):
        channel_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_ch{i}.wav"
        channel_path = os.path.join(subdir, channel_filename)
        channel.export(channel_path, format="wav")
        
        # Convert channel WAV to MP3 and check if MP3 exists before converting
        convert_wav_to_mp3(channel_path)

        # Construct a folder and filename prefix for each channel's results
        file_specific_res_folder = os.path.join(res_folder, os.path.splitext(channel_filename)[0])
        if not os.path.exists(file_specific_res_folder):
            os.makedirs(file_specific_res_folder)
        
        print(f"Processing channel {i} of {os.path.basename(audio_file_path)}")
        
        # Adjust and denoise audio for each channel
        adjusted_file_path = check_and_adjust_audio(channel_path)
        denoised_file_path = apply_noise_reduction(adjusted_file_path)
        if adjusted_file_path != channel_path:
            os.remove(adjusted_file_path)

        # Recognize speech for each channel
        result = recognize_speech_vosk(denoised_file_path, model_path)
        
        # Estimate audio quality for each channel
        noise_level, volume, confidence_percentile_10, words_per_minute = estimate_audio_quality(denoised_file_path, result)
        
        # Delete the denoised file after its use
        os.remove(denoised_file_path)
        
        # Save results for each channel
        filename_prefix = os.path.splitext(channel_filename)[0]
        save_results_to_text(file_specific_res_folder, filename_prefix, result, noise_level, volume, confidence_percentile_10, words_per_minute)
        visualize_with_words_segmented(channel_path, result, file_specific_res_folder)
        
        # Optionally, delete the channel-specific WAV file to clean up if not needed further
        os.remove(channel_path)

# Main processing loop
def process_folders(root_path, model_path):
    res_base_folder = os.path.join(root_path, "Res")
    if not os.path.exists(res_base_folder):
        os.makedirs(res_base_folder)
    for subdir, dirs, files in os.walk(root_path):
        # if "SPEECH_06032024032156" not in subdir.upper():
        #     continue
        if 'Data' in subdir:
            res_folder = os.path.join(res_base_folder, os.path.relpath(subdir, start=root_path).replace("Data" + os.sep, ""))
                        
            for filename in files:
                if filename.endswith(".wav") and "adjusted" not in filename:
                    audio_file_path = os.path.join(subdir, filename)
                    name = audio_file_path.split('\\')[2]

                    # Check if filename is 'sig.wav' and process each channel
                    if filename == "sig.wav":
                        sound = AudioSegment.from_file(audio_file_path)
                        channels = sound.split_to_mono()
                        
                        for i, channel in enumerate(channels, start=1):
                            channel_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_ch{i}.wav"
                            channel_path = os.path.join(subdir, channel_filename)
                            channel.export(channel_path, format="wav")
                            
                            file_specific_res_folder = os.path.join(res_folder, os.path.splitext(channel_filename)[0])
                            if not os.path.exists(file_specific_res_folder):
                                os.makedirs(file_specific_res_folder)
                            
                            # Convert channel WAV to MP3 and check if MP3 exists before converting
                            convert_wav_to_mp3(channel_path, file_specific_res_folder)
                            
                            print(f"Processing {name}: channel {i} of {filename}")
                            
                            # Adjust and denoise audio for each channel
                            adjusted_file_path = check_and_adjust_audio(channel_path)
                            denoised_file_path = apply_noise_reduction(adjusted_file_path)
                            if adjusted_file_path != channel_path:
                                os.remove(adjusted_file_path)

                            # Recognize speech for each channel
                            result = recognize_speech_vosk(denoised_file_path, model_path)
                            
                            # Estimate audio quality for each channel
                            noise_level, volume, confidence_percentile_10, words_per_minute = estimate_audio_quality(denoised_file_path, result)
                            
                            # Delete the denoised file after its use
                            os.remove(denoised_file_path)
                            
                            # Save results for each channel
                            filename_prefix = os.path.splitext(channel_filename)[0]
                            save_results_to_text(file_specific_res_folder, filename_prefix, result, noise_level, volume, confidence_percentile_10, words_per_minute)
                            visualize_with_words_segmented(channel_path, result, file_specific_res_folder)
                            
                            # Optionally, delete the channel-specific WAV file to clean up
                            os.remove(channel_path)
                    else:
                        # Process other wav files as usual (if you have any specific processing for them)
                        pass

# Adjust the paths as necessary
root_path = "."
model_path = "models/vosk-model-small-en-us-0.15"
# model_path = "C:/Users/User/Downloads/vosk-model-en-us-0.42-gigaspeech"
process_folders(root_path, model_path)
