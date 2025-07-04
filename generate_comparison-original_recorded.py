import os
import csv
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr

from scipy import signal
import matplotlib.pyplot as plt
import pyloudnorm as pyln
from tqdm import tqdm
import logging
from typing import Tuple, Optional


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Directory ensured: {path}")




def enhance_audio(input_path: str, output_path: str, sr: int = 22050) -> np.ndarray:
    """
    Comprehensive audio enhancement pipeline for budget recordings
    Applies phase correction, noise reduction, EQ, and loudness normalization
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save enhanced audio
        sr: Target sample rate
        
    Returns:
        Enhanced audio array
    """
    try:
        # Load audio with resampling and force mono
        y, orig_sr = librosa.load(input_path, sr=sr, mono=True)
        logger.info(f"Processing: {os.path.basename(input_path)} - Length: {len(y)/sr:.2f}s")

        # Processing pipeline
        y = correct_phase(y)
        y = reduce_noise(y, sr)
        y = apply_highpass(y, sr, cutoff=60)
        y = apply_speech_eq(y, sr)
        y = apply_compression(y, sr)
        y = normalize_loudness(y, sr, target_lufs=-16.0)
        y = limiter(y, threshold=0.95)

        # Save enhanced audio
        sf.write(output_path, y, sr, subtype='PCM_16')
        logger.info(f"Enhanced audio saved to {output_path}")
        return y
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise


def correct_phase(y: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detect and correct phase inversion based on mean polarity"""
    if np.mean(y) < threshold:
        return -y
    return y


def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """Advanced noise reduction with adaptive noise profile"""
    # Extract noise profile from silent segments
    rms = librosa.feature.rms(y=y)[0]
    noise_threshold = np.percentile(rms, 25)
    noise_frames = np.where(rms < noise_threshold)[0]
    
    if len(noise_frames) > 10:
        frame_length = 2048
        noise_samples = []
        for f in noise_frames[:100]:  # Use up to 100 frames
            start = f * frame_length
            end = start + frame_length
            if end < len(y):
                noise_samples.append(y[start:end])
        
        if noise_samples:
            noise_profile = np.concatenate(noise_samples)
            return nr.reduce_noise(y=y, y_noise=noise_profile, sr=sr, stationary=False)
    
    # Fallback to basic noise reduction
    return nr.reduce_noise(y=y, sr=sr)


def apply_highpass(y: np.ndarray, sr: int, cutoff: int = 80) -> np.ndarray:
    """Remove low-frequency rumble with minimum phase filter"""
    sos = signal.butter(6, cutoff, 'highpass', fs=sr, output='sos')
    return signal.sosfiltfilt(sos, y)


def apply_speech_eq(y: np.ndarray, sr: int) -> np.ndarray:
    """Multi-band EQ optimized for speech clarity"""
    # Parametric EQ bands
    bands = [
        {'type': 'lowpass', 'freq': 10000, 'gain': -3, 'Q': 0.7},
        {'type': 'highshelf', 'freq': 3000, 'gain': 4, 'Q': 0.7},
        {'type': 'peaking', 'freq': 500, 'gain': 2, 'Q': 1.0},
        {'type': 'peaking', 'freq': 1500, 'gain': 3, 'Q': 2.0}
    ]
    
    sos = np.array([])
    for band in bands:
        btype = 'highshelf' if band['type'] == 'highshelf' else 'peak'
        b, a = signal.iirfilter(
            2, 
            band['freq']/(sr/2), 
            btype=btype, 
            ftype='butter',
            q=band['Q'],
            gain=band['gain']
        )
        band_sos = signal.tf2sos(b, a)
        sos = band_sos if sos.size == 0 else np.vstack([sos, band_sos])
    
    return signal.sosfilt(sos, y)


def apply_compression(y: np.ndarray, sr: int, ratio: float = 3.0, 
                      threshold: float = -20.0, attack: float = 0.01, 
                      release: float = 0.1) -> np.ndarray:
    """Simple dynamic range compressor for consistent volume"""
    # Convert to dB
    envelope = np.abs(signal.hilbert(y))
    db = 20 * np.log10(envelope + 1e-7)
    
    # Compression curve
    gain_reduction = np.zeros_like(db)
    above_threshold = db > threshold
    gain_reduction[above_threshold] = ((db[above_threshold] - threshold) * (1 - 1/ratio))
    
    # Smooth with attack/release
    smoothed_reduction = np.zeros_like(gain_reduction)
    alpha_a = np.exp(-1/(attack * sr))
    alpha_r = np.exp(-1/(release * sr))
    
    for i in range(1, len(gain_reduction)):
        alpha = alpha_a if gain_reduction[i] > smoothed_reduction[i-1] else alpha_r
        smoothed_reduction[i] = alpha * smoothed_reduction[i-1] + (1 - alpha) * gain_reduction[i]
    
    # Apply gain reduction
    gain = 10 ** (-smoothed_reduction / 20)
    return y * gain


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """EBU R128 loudness normalization with fallback"""
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness) and loudness > -70:
            return pyln.normalize.loudness(y, loudness, target_lufs)
    except Exception as e:
        logger.warning(f"Loudness normalization failed: {str(e)}")
    
    # RMS fallback normalization
    rms = np.sqrt(np.mean(y**2))
    target_rms = 0.05  # -26 dB
    return y * (target_rms / (rms + 1e-7))


def limiter(y: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Look-ahead limiter to prevent clipping"""
    # Simple look-ahead (1ms window)
    lookahead = 10
    smoothed = np.convolve(np.abs(y), np.ones(lookahead)/lookahead, mode='same')
    gain_reduction = np.where(smoothed > threshold, threshold / (smoothed + 1e-7), 1.0)
    return y * gain_reduction


def calculate_audio_metrics(y: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """Calculate SNR, RMS, and spectral flatness"""
    # SNR estimation
    S = np.abs(librosa.stft(y))
    signal_power = np.median(np.mean(S, axis=1))
    noise_power = np.percentile(np.mean(S, axis=1), 10)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10) + 1e-7)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    
    # Spectral flatness (noise indicator)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    return snr, avg_rms, flatness


def generate_quality_report(input_path: str, output_path: str, report_dir: str) -> None:
    """Generate CSV and spectrogram comparison plots"""
    gfx_dir = os.path.join(report_dir, 'spectrograms')
    ensure_dir(gfx_dir)

    try:
        y_raw, sr = librosa.load(input_path, sr=22050, mono=True)
        y_enh, _ = librosa.load(output_path, sr=sr)
        
        # Calculate metrics
        snr_raw, rms_raw, flatness_raw = calculate_audio_metrics(y_raw, sr)
        snr_enh, rms_enh, flatness_enh = calculate_audio_metrics(y_enh, sr)
        
        # Write to CSV
        csv_path = os.path.join(report_dir, 'quality_metrics.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    'Filename', 'SNR_raw', 'SNR_enh', 'RMS_raw', 'RMS_enh',
                    'Flatness_raw', 'Flatness_enh', 'SNR_improvement'
                ])
            
            snr_imp = snr_enh - snr_raw
            writer.writerow([
                os.path.basename(input_path),
                f"{snr_raw:.2f}", f"{snr_enh:.2f}",
                f"{rms_raw:.4f}", f"{rms_enh:.4f}",
                f"{flatness_raw:.4f}", f"{flatness_enh:.4f}",
                f"{snr_imp:.2f}"
            ])
        
        # Generate comparison plot
        basename = os.path.splitext(os.path.basename(input_path))[0]
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Original spectrogram
        S_raw = librosa.amplitude_to_db(np.abs(librosa.stft(y_raw)), ref=np.max)
        img = librosa.display.specshow(S_raw, sr=sr, x_axis='time', 
                                       y_axis='log', ax=ax[0])
        ax[0].set(title=f'Original - SNR: {snr_raw:.2f} dB, Flatness: {flatness_raw:.3f}')
        
        # Enhanced spectrogram
        S_enh = librosa.amplitude_to_db(np.abs(librosa.stft(y_enh)), ref=np.max)
        librosa.display.specshow(S_enh, sr=sr, x_axis='time', 
                                 y_axis='log', ax=ax[1])
        ax[1].set(title=f'Enhanced - SNR: {snr_enh:.2f} dB, Flatness: {flatness_enh:.3f}')
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        # Save and close
        plt.savefig(os.path.join(gfx_dir, f'comparison_{basename}.png'), dpi=150)
        plt.close()
        logger.info(f"Generated report for {basename}")
        
    except Exception as e:
        logger.error(f"Error generating report for {input_path}: {str(e)}")


def process_all_recordings(version: str, input_dir: str, output_dir: str) -> None:
    """Batch process all recordings and generate reports"""
    ensure_dir(output_dir)
    report_dir = os.path.join('quality_reports', f'v{version}')
    ensure_dir(report_dir)
    
    valid_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    if not valid_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(valid_files)} files in v{version}")
    
    for fname in tqdm(valid_files, desc=f"Processing v{version}"):
        inp = os.path.join(input_dir, fname)
        outp = os.path.join(output_dir, f'enhanced_{fname}')
        
        try:
            enhance_audio(inp, outp)
            generate_quality_report(inp, outp, report_dir)
        except Exception as e:
            logger.error(f"Failed to process {fname}: {str(e)}")
    
    logger.info(f"✅ Completed processing. Enhanced files: {output_dir}\nReports: {report_dir}")


if __name__ == '__main__':
    version = 1
    in_dir = os.path.join('audios', 'grabados', f'v{version}')
    out_dir = os.path.join('audios', 'enhanced', f'v{version}')
    
    if not os.path.exists(in_dir):
        logger.error(f"Input directory not found: {in_dir}")
    else:
        process_all_recordings(version, in_dir, out_dir)