import os
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import pandas as pd
import sounddevice as sd

class YAMNetAudioClassifier:
    def __init__(self, model_handle='https://tfhub.dev/google/yamnet/1', sample_rate=16000, duration=1):
        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_length = self.sample_rate * self.duration
        
        # Load the YAMNet model
        self.model = hub.load(model_handle)
        
        # Load class names
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        self.class_names = list(pd.read_csv(class_map_path)['display_name'])
    
    @tf.function
    def load_wav_16k_mono(self, filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav
    
    @tf.function
    def load_wav_16k_mono_audio(self, audio):
        """ Convert an audio tensor to a float tensor, resample to 16 kHz single-channel audio. """
        wav, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav
    
    def extract_embedding(self, wav_data, label, fold):
        """ Run YAMNet to extract embeddings from the wav data """
        scores, embeddings, spectrogram = self.model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return embeddings, tf.repeat(label, num_embeddings), tf.repeat(fold, num_embeddings)
    
    def predict(self, audio):
        """ Predict the main sound class from the provided audio """
        sd.wait()
        sample_rate = tf.constant(self.sample_rate, dtype=tf.int64)
        wav = tf.squeeze(audio, axis=-1)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        
        scores, embeddings, spectrogram = self.model(wav)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(class_scores)
        inferred_class = self.class_names[top_class]
        top_score = class_scores[top_class]
        
        print(f'[YAMNet] The main sound is: {inferred_class} ({top_score})')