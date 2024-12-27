import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Set Dataset Path
dataset_path = "MakalahAlgeo/dataset"
if os.path.exists(dataset_path):
    print("Dataset path is valid.")
else:
    print("Dataset path is invalid.")


output_dir = "output" 
os.makedirs(output_dir, exist_ok=True)

audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.mp3') or f.endswith('.wav')]
for idx, audio_file in enumerate(audio_files[:400]): #400 Files
    file_path = os.path.join(dataset_path, audio_file)

    print(f"Processing file {idx + 1}/{min(10, len(audio_files))}: {audio_file}")

    # Load Audio
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # Generate Spectrogram
    n_fft = 1024
    hop_length = 512
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Original Spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label="Amplitude (dB)")
    plt.title(f"Original Spectrogram: {audio_file}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{audio_file}_Original_Spectrogram.png"))
    plt.close()

    # Apply Singular Value Decomposition (SVD)
    U, sigma, VT = np.linalg.svd(spectrogram, full_matrices=False)

    # Threshold Singular Values
    threshold = 0.1 * np.max(sigma)  # retain only dominant singular values
    sigma_filtered = np.where(sigma > threshold, sigma, 0)

    # Reconstruct Filtered Spectrogram
    filtered_spectrogram = np.dot(U, np.dot(np.diag(sigma_filtered), VT))

    # Plot Filtered Spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(filtered_spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label="Amplitude (dB)")
    plt.title(f"Filtered Spectrogram (After SVD): {audio_file}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{audio_file}_Filtered_Spectrogram.png"))
    plt.close()

    # Reconstruct Audio from Filtered Spectrogram
    reconstructed_audio = librosa.istft(filtered_spectrogram, hop_length=hop_length)

    # Save Reconstructed Audio
    output_audio_file = os.path.join(output_dir, f"{audio_file}_Reconstructed_Audio.wav")
    sf.write(output_audio_file, reconstructed_audio, sr)

    print(f"Saved results for {audio_file}")

print("Processing complete. Results saved to:", output_dir)