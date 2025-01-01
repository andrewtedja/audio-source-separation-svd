# Enhanced Audio Source Separation Using Singular Value Decomposition and Custom Thresholding Techniques

> Tugas Makalah Algeo 2024

This project applies **Singular Value Decomposition (SVD)** and **custom thresholding techniques** to separate vocals and instruments in audio tracks. The implementation involves spectrogram analysis, SVD-based decomposition, and audio reconstruction with thresholding to enhance separation performance.

## Applications on real life

Audio source separation is essential in applications such as:

-   Karaoke systems
-   Music remixing
-   Sound engineering

This project leverages SVD to decompose spectrograms into components, isolating vocals and instruments separately.

## Features

-   **Audio Source Separation**: Isolates vocals and instruments from mixed tracks.
-   **Custom Thresholding**: Filters out noise for clearer separation.
-   **Spectrogram Visualization**: Generates spectrogram plots before and after processing.
-   **Cumulative Analysis**: Computes the average spectrogram across a dataset.

## How to run

Install the necessary Python libraries:

```bash
pip install librosa numpy matplotlib soundfile
```

### Usage

```bash
python src/main.py
```

## Workflow

1. Preprocessing

    - Convert audio to mono at 16 kHz.
    - Generate spectrograms using Short-Time Fourier Transform (STFT).

2. SVD Decomposition

    - Decompose the spectrogram matrix \( S \) into \( U \Sigma V^T \).

3. Thresholding

    - Apply a threshold to filter singular values in \( \Sigma \).

4. Reconstruction

    - Reconstruct the spectrogram and convert it back to the audio signal.

5. Visualization
    - Save spectrograms (original, filtered, and average) for analysis.

## References

```markdown
1. Kim, J.H., & Park, H.M. (2024). [Multiple Sound Source Localization Using SVD-PHAT-ATV on Blind Source Separation](https://ieeexplore.ieee.org/abstract/document/10597381/). IEEE Access.

2. Basir, S., Hosen, M.S., & Hossain, M.N. (2024). [Enhanced Speech Separation Through a Supervised Approach Using Bidirectional Long Short-Term Memory in Dual Domains](https://www.sciencedirect.com/science/article/pii/S0045790624002921). Computers and Mathematics with Applications.

3. Gorodetska, N., & Oliynik, V. (2024). [An SSA-Based Strategy for Extraction of Cardiac Sounds from Composite Auscultation Records](https://ieeexplore.ieee.org/abstract/document/10756895/). IEEE International Conference on Engineering in Medicine and Biology.

4. Sun, P. (2015). [Comparison of STFT and Wavelet Transform in Time-Frequency Analysis](https://www.diva-portal.org/smash/get/diva2:793176/FULLTEXT01.pdf). Diva Portal.

5. Liu, Q., Wang, W., Jackson, P. J. B., & Barnard, M. (2013). [Source separation of convolutive and noisy mixtures using audio-visual dictionary learning and probabilistic time-frequency masking](https://personalpages.surrey.ac.uk/w.wang/papers/LiuWBJKC_TSP_2013.pdf). IEEE International Conference on Signal Processing.

6. TU Dortmund Audio Benchmark Dataset: [https://www-ai.cs.tu-dortmund.de/audio.html](https://www-ai.cs.tu-dortmund.de/audio.html).
```
