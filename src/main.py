import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Set Dataset Path
dataset_path = "src/dataset"
if os.path.exists(dataset_path):
    print("Dataset path is valid.")
else:
    print("err: Dataset path is invalid.")
    exit()

output_dir = "output" 
os.makedirs(output_dir, exist_ok=True)

audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.mp3') or f.endswith('.wav')]
cumulative_spectrogram = None
file_count = 0
target_shape = None

for idx, audio_file in enumerate(audio_files[:400]): # 400 Files
    file_path = os.path.join(dataset_path, audio_file)

    print(f"Processing file {idx + 1}/{len(audio_files[:400])}: {audio_file}")

    # Load Audio
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # Generate Spectrogram
    n_fft = 1024
    hop_length = 512
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Determine the target shape for the spectrogram
    if target_shape is None:
        target_shape = spectrogram.shape

    # Resize or pad the spectrogram to match the target shape
    padded_spectrogram = np.zeros(target_shape)
    padded_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram

    # Original Spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(padded_spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label="Amplitude (dB)")
    plt.title(f"Original Spectrogram: {audio_file}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{audio_file}_Original_Spectrogram.png"))
    plt.close()

    # Apply Singular Value Decomposition (SVD)
    U, sigma, VT = np.linalg.svd(padded_spectrogram, full_matrices=False)

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

    # Add to cumulative spectrogram
    if cumulative_spectrogram is None:
        cumulative_spectrogram = padded_spectrogram
    else:
        cumulative_spectrogram += padded_spectrogram

    file_count += 1

if cumulative_spectrogram is not None and file_count > 0:
    average_spectrogram = cumulative_spectrogram / file_count

    # Plot and Save Average Spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(average_spectrogram, ref=np.max),
        sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(label="Amplitude (dB)")
    plt.title("Average Spectrogram of Dataset")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "Average_Spectrogram.png"))
    plt.close()

    print("Average spectrogram saved to:", os.path.join(output_dir, "Average_Spectrogram.png"))
else:
    print("err: no files processed.")

print("Processing complete. Results saved to:", output_dir)
