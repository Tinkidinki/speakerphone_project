from pydub import AudioSegment
import librosa
import numpy as np

def mp3_to_vectors(mp3_path, segment_length_ms=600):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_path)

    # Split the audio into one-minute segments
    segments = [audio[i:i+segment_length_ms] for i in range(0, len(audio), segment_length_ms)]

    # Process each segment into vectors
    vectors = []
    for segment in segments[:-1]:
        # Export segment to a temporary WAV file
        segment.export("temp.wav", format="wav")

        # Load the segment using librosa
        y, sr = librosa.load("temp.wav", sr=None)

        # Extract features (e.g., MFCCs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        # Append the features to the vectors list
        vectors.append(mfcc)

    return np.array(vectors)

# Use the function
speaker_path = 'recording_test.mp3'
nonspeaker_path = 'recording_test.mp3'

vectors_class_0 = mp3_to_vectors(nonspeaker_path)
vectors_class_1 = mp3_to_vectors(speaker_path)

labels_class_0 = np.zeros(len(vectors_class_0))
labels_class_1 = np.ones(len(vectors_class_1))

vectors_unshuffled = np.concatenate((vectors_class_0, vectors_class_1))
labels_unshuffled = np.concatenate((labels_class_0, labels_class_1))

indices = np.arange(len(vectors_unshuffled))
np.random.shuffle(indices)
vectors = vectors_unshuffled[indices]
labels = labels_unshuffled[indices]


#---------------------------