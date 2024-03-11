# Speech Recognition and Audio Quality Analysis

This project utilizes a combination of speech recognition and audio processing libraries in Python to analyze audio files. It focuses on recognizing speech using the Vosk library, adjusting audio quality, and visualizing spectrograms with annotated words. Additionally, it estimates audio quality based on noise level, volume, median confidence, and words per minute.

## Features

- **Audio File Adjustment**: Checks and adjusts the audio file's channel and framerate to meet Vosk's requirements.
- **Noise Reduction**: Applies noise reduction to improve speech recognition accuracy.
- **Speech Recognition**: Utilizes Vosk to convert speech in audio files to text, including word confidence levels.
- **Spectrogram Visualization**: Generates a Mel-frequency spectrogram of the audio file, with speech words annotated.
- **Audio Quality Estimation**: Estimates audio quality metrics including noise level, volume, median word confidence, and words per minute.

## Installation

Before running the project, ensure you have Python 3.x installed. Then, install the required libraries using pip:

`pip install vosk pydub SpeechRecognition matplotlib librosa numpy wave json`


Download the Vosk model suitable for your language from [Vosk Models](https://alphacephei.com/vosk/models) and extract it to the `models` directory in your project folder. Or use the model in the GIT folder.

## Usage

1. **Prepare Audio Files**: Place your audio files in the `snds` directory.
2. **Update Configuration**: Modify the `audio_file_path` and `model_path` variables in the script to point to your audio file and the Vosk model directory, respectively.
3. **Run the Script**: Execute the script in your terminal or command prompt.

`python SpeechQTst.py`

4. **Review the Output**: The script will adjust the audio file if necessary, apply noise reduction, perform speech recognition, visualize the spectrogram with annotated words, and estimate audio quality.
