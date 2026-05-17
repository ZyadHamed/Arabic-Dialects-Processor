import io
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> io.BytesIO:
    """
    Compute a linear-frequency STFT spectrogram and render it to a PNG buffer.

    Args:
        y:          Mono audio signal as a float32 numpy array.
        sr:         Sample rate of the signal.
        n_fft:      FFT window size (default 2048).
        hop_length: Hop length between frames (default 512).

    Returns:
        A seeked BytesIO buffer containing the PNG image — ready to stream or write.
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    duration = librosa.get_duration(y=y, sr=sr)
    fig_w = max(8, min(20, duration / 2))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("STFT Spectrogram", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> io.BytesIO:
    """
    Compute a Mel-scale spectrogram and render it to a PNG buffer.

    Args:
        y:          Mono audio signal as a float32 numpy array.
        sr:         Sample rate of the signal.
        n_mels:     Number of Mel filter banks (default 128).
        hop_length: Hop length between frames (default 512).
        fmin:       Minimum frequency in Hz (default 0).
        fmax:       Maximum frequency in Hz (default sr/2).

    Returns:
        A seeked BytesIO buffer containing the PNG image — ready to stream or write.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    duration = librosa.get_duration(y=y, sr=sr)
    fig_w = max(8, min(20, duration / 2))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel Spectrogram ({n_mels} bands)", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Mel)")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf