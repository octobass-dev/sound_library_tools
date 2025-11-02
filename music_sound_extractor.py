"""
Experimental Sound Library Extractor

Extracts unique, interesting sounds from audio/video files for sampling.
Identifies transients, sustained tones, drones, and other experimental sounds.

Dependencies:
pip install librosa soundfile numpy scipy pydub noisereduce essentia-tensorflow

This tool imports and uses classes from the previous audio processing tools.
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import signal
from scipy.stats import kurtosis, skew
import subprocess
from collections import defaultdict
import hashlib


# ============================================================================
# Sound Classification and Categories
# ============================================================================

@dataclass
class SoundSample:
    """Represents an extracted sound sample"""
    file_path: str
    source_file: str
    start_time: float
    end_time: float
    duration: float
    
    # Audio characteristics
    category: str  # 'transient', 'sustained', 'drone', 'percussive', 'tonal', 'noise'
    subcategory: str  # More specific classification
    
    # Quality metrics
    loudness_lufs: float
    dynamic_range_db: float
    noise_floor_db: float
    signal_to_noise_db: float
    
    # Pitch information
    has_pitch: bool
    pitch_hz: Optional[float]
    pitch_note: Optional[str]
    pitch_stability: float  # 0-1, how stable the pitch is
    
    # Spectral characteristics
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float  # 0-1, how noise-like vs tonal
    zero_crossing_rate: float
    
    # Temporal characteristics
    attack_time_ms: float  # Time to reach peak
    decay_time_ms: float
    onset_strength: float
    
    # Uniqueness score
    uniqueness_score: float  # 0-1, how unique/interesting this sound is
    
    # Tags for searchability
    tags: List[str]


class SoundCategory:
    """Sound category definitions"""
    
    TRANSIENT = "transient"
    SUSTAINED = "sustained"
    DRONE = "drone"
    PERCUSSIVE = "percussive"
    TONAL = "tonal"
    NOISE = "noise"
    VOCAL = "vocal"
    MECHANICAL = "mechanical"
    
    # Subcategories
    SUBCATEGORIES = {
        TRANSIENT: [
            "impact", "click", "pop", "snap", "crack",
            "glass_break", "door_slam", "footstep",
            "clap", "snap_finger", "surprised_gasp"
        ],
        SUSTAINED: [
            "laughter", "scream", "cry", "sigh",
            "vehicle_engine", "machine_hum", "alarm",
            "musical_note", "vocal_sustain"
        ],
        DRONE: [
            "wind", "rumble", "hum", "buzz",
            "ambient_noise", "resonance", "feedback",
            "tunnel_echo", "room_tone"
        ],
        PERCUSSIVE: [
            "drum_hit", "wood_knock", "metal_clang",
            "hand_clap", "stomp", "thud"
        ],
        TONAL: [
            "whistle", "bell", "chime", "singing",
            "musical_instrument", "tone", "beep"
        ],
        NOISE: [
            "white_noise", "static", "crackle",
            "rain", "crowd", "texture"
        ]
    }


# ============================================================================
# Audio Feature Extraction
# ============================================================================

class AudioFeatureExtractor:
    """Extract detailed audio features for classification"""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def extract_features(self, y: np.ndarray, sr: int = None) -> Dict:
        """Extract comprehensive audio features"""
        if sr is None:
            sr = self.sr
        
        features = {}
        
        # Basic features
        features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
        features['peak_amplitude'] = float(np.max(np.abs(y)))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Temporal features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_strength_mean'] = float(np.mean(onset_env))
        features['onset_strength_max'] = float(np.max(onset_env))
        
        # MFCC for timbre
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
        features['mfcc_std'] = mfcc.std(axis=1).tolist()
        
        # Chroma for pitch content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = chroma.mean(axis=1).tolist()
        features['chroma_std'] = chroma.std(axis=1).tolist()
        
        # Statistical features
        features['kurtosis'] = float(kurtosis(y))
        features['skewness'] = float(skew(y))
        
        return features
    
    def calculate_loudness_lufs(self, y: np.ndarray, sr: int) -> float:
        """Calculate loudness in LUFS (approximation)"""
        # Simple approximation using RMS
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            lufs = 20 * np.log10(rms) - 23.0  # Rough approximation
        else:
            lufs = -100.0
        return float(lufs)
    
    def calculate_dynamic_range(self, y: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y**2))
        
        if rms > 0:
            dr = 20 * np.log10(peak / rms)
        else:
            dr = 0.0
        
        return float(dr)
    
    def estimate_noise_floor(self, y: np.ndarray, percentile: float = 10) -> float:
        """Estimate noise floor using lower percentile of energy"""
        frame_length = 2048
        hop_length = 512
        
        # Calculate energy per frame
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Noise floor is estimated from low-energy frames
        noise_floor = np.percentile(energy, percentile)
        
        if noise_floor > 0:
            noise_floor_db = 10 * np.log10(noise_floor)
        else:
            noise_floor_db = -100.0
        
        return float(noise_floor_db)
    
    def calculate_snr(self, y: np.ndarray, sr: int) -> float:
        """Calculate signal-to-noise ratio"""
        noise_floor_db = self.estimate_noise_floor(y)
        signal_power = np.mean(y**2)
        
        if signal_power > 0:
            signal_db = 10 * np.log10(signal_power)
            snr = signal_db - noise_floor_db
        else:
            snr = 0.0
        
        return float(snr)
    
    def detect_pitch_stability(self, y: np.ndarray, sr: int) -> Tuple[bool, Optional[float], float]:
        """
        Detect if sound has stable pitch
        
        Returns:
            (has_pitch, pitch_hz, stability_score)
        """
        # Use librosa's pyin for pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Filter out unvoiced regions
        valid_f0 = f0[voiced_flag]
        
        if len(valid_f0) < 5:
            return False, None, 0.0
        
        # Calculate pitch statistics
        mean_f0 = np.nanmean(valid_f0)
        std_f0 = np.nanstd(valid_f0)
        
        if np.isnan(mean_f0) or mean_f0 <= 0:
            return False, None, 0.0
        
        # Stability score based on coefficient of variation
        cv = std_f0 / mean_f0
        stability = np.exp(-cv * 5)  # 0-1 scale
        
        # Has pitch if >50% of frames are voiced and stability is reasonable
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        has_pitch = voiced_ratio > 0.5 and stability > 0.3
        
        return has_pitch, float(mean_f0), float(stability)
    
    def calculate_envelope(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Calculate amplitude envelope"""
        # Hilbert transform for envelope
        analytic_signal = signal.hilbert(y)
        envelope = np.abs(analytic_signal)
        
        # Smooth the envelope
        window_size = int(0.01 * sr)  # 10ms window
        if window_size % 2 == 0:
            window_size += 1
        
        envelope = signal.savgol_filter(envelope, window_size, 3)
        
        return envelope
    
    def measure_attack_decay(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Measure attack and decay times in milliseconds"""
        envelope = self.calculate_envelope(y, sr)
        
        # Find peak
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]
        
        if peak_value == 0:
            return 0.0, 0.0
        
        # Attack time: time from start to 90% of peak
        threshold = 0.9 * peak_value
        attack_indices = np.where(envelope[:peak_idx] > threshold)[0]
        
        if len(attack_indices) > 0:
            attack_samples = attack_indices[0]
        else:
            attack_samples = peak_idx
        
        attack_time_ms = (attack_samples / sr) * 1000
        
        # Decay time: time from peak to 10% of peak
        threshold = 0.1 * peak_value
        decay_indices = np.where(envelope[peak_idx:] < threshold)[0]
        
        if len(decay_indices) > 0:
            decay_samples = decay_indices[0]
        else:
            decay_samples = len(envelope) - peak_idx
        
        decay_time_ms = (decay_samples / sr) * 1000
        
        return float(attack_time_ms), float(decay_time_ms)


# ============================================================================
# Sound Detection and Segmentation
# ============================================================================

class SoundDetector:
    """Detect and segment interesting sounds in audio"""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.feature_extractor = AudioFeatureExtractor(sr=sr)
    
    def detect_sound_segments(self, y: np.ndarray, sr: int = None,
                             min_duration: float = 0.05,
                             max_duration: float = 10.0,
                             min_silence_duration: float = 0.1,
                             energy_threshold_db: float = -40) -> List[Tuple[float, float]]:
        """
        Detect sound segments based on energy and silence
        
        Returns:
            List of (start_time, end_time) tuples
        """
        if sr is None:
            sr = self.sr
        
        # Calculate energy envelope
        hop_length = 512
        frame_length = 2048
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find segments above threshold
        above_threshold = rms_db > energy_threshold_db
        
        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
        
        # Find contiguous regions
        segments = []
        in_segment = False
        start_time = 0
        
        for i, (is_sound, t) in enumerate(zip(above_threshold, times)):
            if is_sound and not in_segment:
                # Start of segment
                start_time = t
                in_segment = True
            elif not is_sound and in_segment:
                # End of segment
                end_time = t
                duration = end_time - start_time
                
                if min_duration <= duration <= max_duration:
                    segments.append((start_time, end_time))
                
                in_segment = False
        
        # Handle last segment
        if in_segment:
            end_time = times[-1]
            duration = end_time - start_time
            if min_duration <= duration <= max_duration:
                segments.append((start_time, end_time))
        
        # Merge segments that are too close
        merged_segments = self._merge_close_segments(
            segments, min_silence_duration
        )
        
        return merged_segments
    
    def _merge_close_segments(self, segments: List[Tuple[float, float]],
                             min_gap: float) -> List[Tuple[float, float]]:
        """Merge segments separated by less than min_gap"""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            
            if start - prev_end < min_gap:
                # Merge with previous segment
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        
        return merged
    
    def detect_onsets(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """Detect onset times (transient events)"""
        if sr is None:
            sr = self.sr
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True
        )
        
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        return onset_times
    
    def has_surrounding_silence(self, y: np.ndarray, sr: int,
                               silence_duration: float = 0.1,
                               silence_threshold_db: float = -50) -> Tuple[bool, bool]:
        """
        Check if sound is surrounded by silence
        
        Returns:
            (has_leading_silence, has_trailing_silence)
        """
        silence_samples = int(silence_duration * sr)
        
        if len(y) < 2 * silence_samples:
            return False, False
        
        # Check leading silence
        leading = y[:silence_samples]
        leading_rms = np.sqrt(np.mean(leading**2))
        leading_db = 20 * np.log10(leading_rms + 1e-10)
        has_leading = leading_db < silence_threshold_db
        
        # Check trailing silence
        trailing = y[-silence_samples:]
        trailing_rms = np.sqrt(np.mean(trailing**2))
        trailing_db = 20 * np.log10(trailing_rms + 1e-10)
        has_trailing = trailing_db < silence_threshold_db
        
        return has_leading, has_trailing


# ============================================================================
# Sound Classification
# ============================================================================

class SoundClassifier:
    """Classify sounds into categories"""
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
    
    def classify_sound(self, y: np.ndarray, sr: int,
                      features: Dict = None) -> Tuple[str, str, List[str]]:
        """
        Classify sound into category and subcategory
        
        Returns:
            (category, subcategory, tags)
        """
        if features is None:
            features = self.feature_extractor.extract_features(y, sr)
        
        # Get additional temporal features
        attack_ms, decay_ms = self.feature_extractor.measure_attack_decay(y, sr)
        has_pitch, pitch_hz, pitch_stability = self.feature_extractor.detect_pitch_stability(y, sr)
        
        duration = len(y) / sr
        
        tags = []
        
        # Decision tree for classification
        
        # TRANSIENT: Short, fast attack
        if duration < 0.5 and attack_ms < 50:
            category = SoundCategory.TRANSIENT
            
            if features['spectral_flatness_mean'] > 0.7:
                subcategory = "impact"
                tags.extend(["sharp", "percussive", "hit"])
            elif features['spectral_centroid_mean'] > 3000:
                subcategory = "click"
                tags.extend(["short", "bright", "snap"])
            else:
                subcategory = "pop"
                tags.extend(["short", "transient"])
        
        # PERCUSSIVE: Short to medium, rhythmic potential
        elif duration < 2.0 and attack_ms < 100 and decay_ms < 500:
            category = SoundCategory.PERCUSSIVE
            
            if features['spectral_centroid_mean'] > 2000:
                subcategory = "metal_clang"
                tags.extend(["metallic", "bright", "resonant"])
            elif features['spectral_centroid_mean'] < 500:
                subcategory = "thud"
                tags.extend(["dull", "low", "impact"])
            else:
                subcategory = "drum_hit"
                tags.extend(["percussive", "rhythmic"])
        
        # DRONE: Long, sustained, stable
        elif duration > 1.0 and attack_ms > 200 and features['rms_energy'] > 0.01:
            if pitch_stability > 0.6:
                category = SoundCategory.TONAL
                subcategory = "musical_note"
                tags.extend(["sustained", "tonal", "pitched"])
            else:
                category = SoundCategory.DRONE
                
                if features['spectral_flatness_mean'] > 0.6:
                    subcategory = "noise"
                    tags.extend(["textural", "atmospheric"])
                elif features['spectral_centroid_mean'] < 500:
                    subcategory = "rumble"
                    tags.extend(["low", "deep", "bass"])
                else:
                    subcategory = "hum"
                    tags.extend(["drone", "sustained"])
        
        # TONAL: Has clear pitch
        elif has_pitch and pitch_stability > 0.5:
            category = SoundCategory.TONAL
            
            if attack_ms < 50:
                subcategory = "bell"
                tags.extend(["bright", "ringing", "tonal"])
            elif features['spectral_centroid_mean'] > 2000:
                subcategory = "whistle"
                tags.extend(["high", "pure", "tonal"])
            else:
                subcategory = "musical_instrument"
                tags.extend(["tonal", "melodic"])
        
        # SUSTAINED: Medium to long duration
        elif duration > 0.5:
            category = SoundCategory.SUSTAINED
            
            # Check for vocal characteristics
            if 300 < features['spectral_centroid_mean'] < 3000 and features['spectral_flatness_mean'] < 0.4:
                if features['onset_strength_max'] > 0.5:
                    subcategory = "laughter"
                    tags.extend(["vocal", "human", "organic"])
                else:
                    subcategory = "vocal_sustain"
                    tags.extend(["vocal", "human"])
            elif features['spectral_centroid_mean'] > 1000:
                subcategory = "machine_hum"
                tags.extend(["mechanical", "electronic"])
            else:
                subcategory = "ambient_noise"
                tags.extend(["atmospheric", "background"])
        
        # NOISE: High spectral flatness
        elif features['spectral_flatness_mean'] > 0.7:
            category = SoundCategory.NOISE
            subcategory = "texture"
            tags.extend(["noisy", "textural", "abstract"])
        
        # Default
        else:
            category = SoundCategory.SUSTAINED
            subcategory = "unknown"
            tags.append("unclassified")
        
        # Add pitch-related tags
        if has_pitch and pitch_hz:
            if pitch_hz < 200:
                tags.append("low_pitch")
            elif pitch_hz < 1000:
                tags.append("mid_pitch")
            else:
                tags.append("high_pitch")
        
        # Add spectral tags
        if features['spectral_centroid_mean'] > 3000:
            tags.append("bright")
        elif features['spectral_centroid_mean'] < 500:
            tags.append("dark")
        
        # Add dynamic tags
        if attack_ms < 20:
            tags.append("fast_attack")
        elif attack_ms > 200:
            tags.append("slow_attack")
        
        return category, subcategory, tags
    
    def calculate_uniqueness_score(self, y: np.ndarray, sr: int,
                                   features: Dict = None) -> float:
        """
        Calculate how unique/interesting a sound is for sampling
        
        Factors:
        - Spectral uniqueness
        - Dynamic range
        - Pitch characteristics
        - Onset strength
        """
        if features is None:
            features = self.feature_extractor.extract_features(y, sr)
        
        score = 0.0
        
        # High spectral centroid = interesting timbre
        centroid_score = min(features['spectral_centroid_mean'] / 5000, 1.0) * 0.2
        score += centroid_score
        
        # Moderate spectral flatness = good balance
        flatness = features['spectral_flatness_mean']
        flatness_score = 1.0 - abs(flatness - 0.5) * 2  # Peak at 0.5
        score += flatness_score * 0.15
        
        # High onset strength = impactful
        onset_score = min(features['onset_strength_max'], 1.0) * 0.2
        score += onset_score
        
        # Clear pitch = more musical
        has_pitch, pitch_hz, pitch_stability = self.feature_extractor.detect_pitch_stability(y, sr)
        if has_pitch:
            pitch_score = pitch_stability * 0.15
        else:
            pitch_score = 0.1  # Atonal sounds can still be interesting
        score += pitch_score
        
        # Good dynamic range
        dr = self.feature_extractor.calculate_dynamic_range(y)
        dr_score = min(dr / 30, 1.0) * 0.15
        score += dr_score
        
        # High SNR = clean sample
        snr = self.feature_extractor.calculate_snr(y, sr)
        snr_score = min(max(snr / 40, 0), 1.0) * 0.15
        score += snr_score
        
        return float(np.clip(score, 0, 1))


# ============================================================================
# Sound Library Builder
# ============================================================================

class SoundLibraryBuilder:
    """Build library of unique sounds from video/audio files"""
    
    def __init__(self, output_dir: str = "./sound_library",
                 sr: int = 44100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sr = sr
        self.detector = SoundDetector(sr=sr)
        self.classifier = SoundClassifier()
        self.feature_extractor = AudioFeatureExtractor(sr=sr)
        
        # Quality thresholds
        self.min_loudness_lufs = -40.0
        self.min_snr_db = 10.0
        self.min_uniqueness_score = 0.3
        self.min_duration = 0.05
        self.max_duration = 10.0
        
        # Deduplication
        self.extracted_hashes = set()
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file"""
        audio_output = str(self.output_dir / "temp_audio.wav")
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.sr), '-ac', '1',
            '-y', audio_output
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_output
    
    def process_file(self, file_path: str,
                    min_uniqueness: float = None) -> List[SoundSample]:
        """
        Process a single audio/video file and extract sounds
        
        Returns:
            List of extracted SoundSample objects
        """
        if min_uniqueness is None:
            min_uniqueness = self.min_uniqueness_score
        
        print(f"\nProcessing: {Path(file_path).name}")
        
        # Extract audio if video
        is_video = str(file_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            audio_path = self.extract_audio_from_video(file_path)
        else:
            audio_path = file_path
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Detect sound segments
        segments = self.detector.detect_sound_segments(
            y, sr,
            min_duration=self.min_duration,
            max_duration=self.max_duration
        )
        
        print(f"  Found {len(segments)} potential segments")
        
        samples = []
        
        for i, (start_time, end_time) in enumerate(segments):
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            # Quality checks
            loudness = self.feature_extractor.calculate_loudness_lufs(segment_audio, sr)
            snr = self.feature_extractor.calculate_snr(segment_audio, sr)
            
            # Check if loud enough and clean enough
            if loudness < self.min_loudness_lufs or snr < self.min_snr_db:
                continue
            
            # Check for surrounding silence (preferred but not required)
            has_leading, has_trailing = self.detector.has_surrounding_silence(
                segment_audio, sr
            )
            
            # Extract features
            features = self.feature_extractor.extract_features(segment_audio, sr)
            
            # Classify
            category, subcategory, tags = self.classifier.classify_sound(
                segment_audio, sr, features
            )
            
            # Calculate uniqueness
            uniqueness = self.classifier.calculate_uniqueness_score(
                segment_audio, sr, features
            )
            
            if uniqueness < min_uniqueness:
                continue
            
            # Bonus for surrounding silence
            if has_leading and has_trailing:
                uniqueness = min(uniqueness * 1.1, 1.0)
                tags.append("clean")
            
            # Check for duplicates
            audio_hash = self._hash_audio(segment_audio)
            if audio_hash in self.extracted_hashes:
                continue
            self.extracted_hashes.add(audio_hash)
            
            # Get pitch info
            has_pitch, pitch_hz, pitch_stability = self.feature_extractor.detect_pitch_stability(
                segment_audio, sr
            )
            
            if has_pitch and pitch_hz:
                from __main__ import AudioProcessor
                processor = AudioProcessor()
                pitch_note = processor.hz_to_note(pitch_hz)
            else:
                pitch_note = None
            
            # Get temporal characteristics
            attack_ms, decay_ms = self.feature_extractor.measure_attack_decay(segment_audio, sr)
            
            # Calculate other metrics
            dr = self.feature_extractor.calculate_dynamic_range(segment_audio)
            noise_floor = self.feature_extractor.estimate_noise_floor(segment_audio)
            
            # Save audio file
            sound_filename = f"{Path(file_path).stem}_{i:04d}_{category}_{subcategory}.wav"
            sound_path = self.output_dir / category / sound_filename
            sound_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(sound_path), segment_audio, sr)
            
            # Create sample object
            sample = SoundSample(
                file_path=str(sound_path),
                source_file=file_path,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                category=category,
                subcategory=subcategory,
                loudness_lufs=loudness,
                dynamic_range_db=dr,
                noise_floor_db=noise_floor,
                signal_to_noise_db=snr,
                has_pitch=has_pitch,
                pitch_hz=pitch_hz,
                pitch_note=pitch_note,
                pitch_stability=pitch_stability,
                spectral_centroid_hz=features['spectral_centroid_mean'],
                spectral_rolloff_hz=features['spectral_rolloff_mean'],
                spectral_flatness=features['spectral_flatness_mean'],
                zero_crossing_rate=features['zcr_mean'],
                attack_time_ms=attack_ms,
                decay_time_ms=decay_ms,
                onset_strength=features['onset_strength_max'],
                uniqueness_score=uniqueness,
                tags=tags
            )
            
            samples.append(sample)
        
        print(f"  Extracted {len(samples)} high-quality samples")
        
        # Cleanup temp audio
        if is_video and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return samples
    
    def _hash_audio(self, y: np.ndarray) -> str:
        """Create hash of audio for deduplication"""
        # Use perceptual hash based on MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Create hash from quantized MFCC
        mfcc_bytes = (mfcc_mean * 100).astype(np.int32).tobytes()
        return hashlib.md5(mfcc_bytes).hexdigest()
    
    def build_library_from_directory(self, directory: str,
                                     max_files: int = None) -> List[SoundSample]:
        """
        Build sound library from directory of audio/video files
        
        Returns:
            List of all extracted samples
        """
        print("\n" + "=" * 80)
        print("BUILDING SOUND LIBRARY")
        print("=" * 80)
        
        # Find all media files
        extensions = ['.mp3', '.wav', '.m4a', '.flac', '.mp4', '.avi', '.mov', '.mkv']
        files = []
        
        for ext in extensions:
            files.extend(Path(directory).rglob(f'*{ext}'))
        
        if max_files:
            files = files[:max_files]
        
        print(f"Found {len(files)} files to process")
        
        all_samples = []
        
        for file_path in files:
            try:
                samples = self.process_file(str(file_path))
                all_samples.extend(samples)
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        print(f"\n{'='*80}")
        print(f"LIBRARY BUILD COMPLETE")
        print(f"Total samples extracted: {len(all_samples)}")
        print(f"{'='*80}")
        
        # Save library metadata
        self._save_library_metadata(all_samples)
        
        return all_samples
    
    def _save_library_metadata(self, samples: List[SoundSample]):
        """Save library metadata to JSON"""
        metadata_path = self.output_dir / "library_metadata.json"
        
        metadata = {
            'total_samples': len(samples),
            'by_category': self._group_by_category(samples),
            'samples': [asdict(s) for s in samples]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_path}")
    
    def _group_by_category(self, samples: List[SoundSample]) -> Dict:
        """Group samples by category"""
        grouped = defaultdict(lambda: {'count': 0, 'subcategories': defaultdict(int)})
        
        for sample in samples:
            grouped[sample.category]['count'] += 1
            grouped[sample.category]['subcategories'][sample.subcategory] += 1
        
        return {k: dict(v) for k, v in grouped.items()}


# ============================================================================
# Sound Library Search and Browse
# ============================================================================

class SoundLibraryBrowser:
    """Search and browse the sound library"""
    
    def __init__(self, library_dir: str = "./sound_library"):
        self.library_dir = Path(library_dir)
        self.samples = []
        self.load_library()
    
    def load_library(self):
        """Load library metadata"""
        metadata_path = self.library_dir / "library_metadata.json"
        
        if not metadata_path.exists():
            print(f"No library found at {self.library_dir}")
            return
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        self.samples = [
            SoundSample(**s) for s in data['samples']
        ]
        
        print(f"Loaded {len(self.samples)} samples from library")
    
    def search(self, category: str = None,
               subcategory: str = None,
               tags: List[str] = None,
               min_duration: float = None,
               max_duration: float = None,
               has_pitch: bool = None,
               min_uniqueness: float = None) -> List[SoundSample]:
        """Search library with filters"""
        results = self.samples
        
        if category:
            results = [s for s in results if s.category == category]
        
        if subcategory:
            results = [s for s in results if s.subcategory == subcategory]
        
        if tags:
            results = [
                s for s in results
                if any(tag in s.tags for tag in tags)
            ]
        
        if min_duration:
            results = [s for s in results if s.duration >= min_duration]
        
        if max_duration:
            results = [s for s in results if s.duration <= max_duration]
        
        if has_pitch is not None:
            results = [s for s in results if s.has_pitch == has_pitch]
        
        if min_uniqueness:
            results = [s for s in results if s.uniqueness_score >= min_uniqueness]
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get library statistics"""
        stats = {
            'total_samples': len(self.samples),
            'by_category': defaultdict(int),
            'by_subcategory': defaultdict(int),
            'duration_stats': {},
            'uniqueness_stats': {},
            'pitch_distribution': {
                'pitched': 0,
                'unpitched': 0
            }
        }
        
        durations = []
        uniqueness_scores = []
        
        for sample in self.samples:
            stats['by_category'][sample.category] += 1
            stats['by_subcategory'][sample.subcategory] += 1
            durations.append(sample.duration)
            uniqueness_scores.append(sample.uniqueness_score)
            
            if sample.has_pitch:
                stats['pitch_distribution']['pitched'] += 1
            else:
                stats['pitch_distribution']['unpitched'] += 1
        
        stats['duration_stats'] = {
            'mean': float(np.mean(durations)),
            'median': float(np.median(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations))
        }
        
        stats['uniqueness_stats'] = {
            'mean': float(np.mean(uniqueness_scores)),
            'median': float(np.median(uniqueness_scores)),
            'min': float(np.min(uniqueness_scores)),
            'max': float(np.max(uniqueness_scores))
        }
        
        return dict(stats)
    
    def print_statistics(self):
        """Print library statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("SOUND LIBRARY STATISTICS")
        print("=" * 80)
        
        print(f"\nTotal Samples: {stats['total_samples']}")
        
        print("\nBy Category:")
        for category, count in sorted(stats['by_category'].items()):
            print(f"  {category}: {count}")
        
        print("\nBy Subcategory (Top 10):")
        top_subcats = sorted(
            stats['by_subcategory'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for subcat, count in top_subcats:
            print(f"  {subcat}: {count}")
        
        print("\nDuration Statistics:")
        for key, value in stats['duration_stats'].items():
            print(f"  {key}: {value:.3f}s")
        
        print("\nUniqueness Statistics:")
        for key, value in stats['uniqueness_stats'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nPitch Distribution:")
        print(f"  Pitched: {stats['pitch_distribution']['pitched']}")
        print(f"  Unpitched: {stats['pitch_distribution']['unpitched']}")
    
    def export_pack(self, output_path: str,
                   category: str = None,
                   min_uniqueness: float = 0.6,
                   max_samples: int = 100):
        """
        Export a sample pack (collection of sounds)
        
        Args:
            output_path: Directory to export to
            category: Filter by category
            min_uniqueness: Minimum uniqueness score
            max_samples: Maximum number of samples
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Search for samples
        samples = self.search(
            category=category,
            min_uniqueness=min_uniqueness
        )
        
        # Sort by uniqueness
        samples.sort(key=lambda x: x.uniqueness_score, reverse=True)
        samples = samples[:max_samples]
        
        print(f"\nExporting {len(samples)} samples to {output_path}")
        
        # Copy files and create index
        pack_info = {
            'name': f"Sample Pack - {category or 'All'}",
            'samples': []
        }
        
        for i, sample in enumerate(samples):
            # Copy file
            src = Path(sample.file_path)
            dst = output_dir / f"{i:04d}_{src.name}"
            
            import shutil
            shutil.copy2(src, dst)
            
            pack_info['samples'].append({
                'filename': dst.name,
                'category': sample.category,
                'subcategory': sample.subcategory,
                'duration': sample.duration,
                'uniqueness': sample.uniqueness_score,
                'tags': sample.tags,
                'pitch_note': sample.pitch_note
            })
        
        # Save pack info
        with open(output_dir / "pack_info.json", 'w') as f:
            json.dump(pack_info, f, indent=2)
        
        print(f"Sample pack exported to: {output_path}")


# ============================================================================
# Integration with Previous Tools
# ============================================================================

class IntegratedSoundExtractor:
    """Integrated tool using both video library and sound extraction"""
    
    def __init__(self):
        # Import from previous tools
        try:
            from __main__ import VideoLibraryDownloader, VideoLibraryFinder
            self.video_downloader = VideoLibraryDownloader()
            self.video_finder = VideoLibraryFinder()
        except ImportError:
            print("Warning: Previous tool classes not available")
            self.video_downloader = None
            self.video_finder = None
        
        self.sound_builder = SoundLibraryBuilder()
    
    def build_from_video_library(self, video_library_dir: str = "./video_library",
                                 max_files: int = None):
        """Build sound library from existing video library"""
        print("\n" + "=" * 80)
        print("BUILDING SOUND LIBRARY FROM VIDEO LIBRARY")
        print("=" * 80)
        
        samples = self.sound_builder.build_library_from_directory(
            video_library_dir,
            max_files=max_files
        )
        
        return samples
    
    def download_and_extract(self, source_name: str,
                            max_videos: int = 20):
        """Download videos from source and extract sounds"""
        if not self.video_downloader or not self.video_finder:
            print("Video downloader not available")
            return []
        
        # Get source
        source = self.video_finder.get_source_by_name(source_name)
        if not source:
            print(f"Source not found: {source_name}")
            return []
        
        # Download videos
        print(f"\nDownloading from: {source_name}")
        downloaded = self.video_downloader.download_from_source(source, max_videos)
        
        # Extract sounds
        print(f"\nExtracting sounds from {len(downloaded)} videos")
        
        all_samples = []
        for video_path in downloaded:
            try:
                samples = self.sound_builder.process_file(video_path)
                all_samples.extend(samples)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
        
        return all_samples


# ============================================================================
# Advanced Sound Processing
# ============================================================================

class SoundProcessor:
    """Additional processing for extracted sounds"""
    
    @staticmethod
    def normalize_loudness(input_path: str, output_path: str,
                          target_lufs: float = -16.0):
        """Normalize audio to target LUFS"""
        cmd = [
            'ffmpeg', '-i', input_path,
            '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    @staticmethod
    def trim_silence(input_path: str, output_path: str,
                    threshold_db: float = -50.0):
        """Trim silence from beginning and end"""
        cmd = [
            'ffmpeg', '-i', input_path,
            '-af', f'silenceremove=start_periods=1:start_threshold={threshold_db}dB:stop_periods=1:stop_threshold={threshold_db}dB',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    @staticmethod
    def apply_fade(input_path: str, output_path: str,
                  fade_in_ms: float = 10.0,
                  fade_out_ms: float = 50.0):
        """Apply fade in/out"""
        y, sr = librosa.load(input_path, sr=None)
        
        fade_in_samples = int(fade_in_ms * sr / 1000)
        fade_out_samples = int(fade_out_ms * sr / 1000)
        
        # Apply fades
        if fade_in_samples > 0:
            fade_curve = np.linspace(0, 1, fade_in_samples)
            y[:fade_in_samples] *= fade_curve
        
        if fade_out_samples > 0:
            fade_curve = np.linspace(1, 0, fade_out_samples)
            y[-fade_out_samples:] *= fade_curve
        
        sf.write(output_path, y, sr)
    
    @staticmethod
    def batch_normalize(library_dir: str, target_lufs: float = -16.0):
        """Normalize all sounds in library"""
        library_path = Path(library_dir)
        
        for wav_file in library_path.rglob('*.wav'):
            normalized_path = wav_file.parent / f"{wav_file.stem}_normalized.wav"
            
            try:
                SoundProcessor.normalize_loudness(
                    str(wav_file),
                    str(normalized_path),
                    target_lufs
                )
                
                # Replace original
                wav_file.unlink()
                normalized_path.rename(wav_file)
                
            except Exception as e:
                print(f"Error normalizing {wav_file}: {e}")


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Experimental Sound Library Extractor'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build library command
    build_parser = subparsers.add_parser('build', help='Build sound library')
    build_parser.add_argument('directory', help='Directory with audio/video files')
    build_parser.add_argument('--output', default='./sound_library',
                             help='Output directory')
    build_parser.add_argument('--max-files', type=int,
                             help='Maximum files to process')
    build_parser.add_argument('--min-uniqueness', type=float, default=0.3,
                             help='Minimum uniqueness score')
    
    # Browse library command
    browse_parser = subparsers.add_parser('browse', help='Browse sound library')
    browse_parser.add_argument('--library', default='./sound_library',
                              help='Library directory')
    browse_parser.add_argument('--category', help='Filter by category')
    browse_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show library statistics')
    stats_parser.add_argument('--library', default='./sound_library',
                             help='Library directory')
    
    # Export pack command
    export_parser = subparsers.add_parser('export', help='Export sample pack')
    export_parser.add_argument('output', help='Output directory')
    export_parser.add_argument('--library', default='./sound_library',
                              help='Library directory')
    export_parser.add_argument('--category', help='Filter by category')
    export_parser.add_argument('--min-uniqueness', type=float, default=0.6,
                              help='Minimum uniqueness score')
    export_parser.add_argument('--max-samples', type=int, default=100,
                              help='Maximum samples to export')
    
    # Process single file
    process_parser = subparsers.add_parser('process', help='Process single file')
    process_parser.add_argument('file', help='Audio/video file to process')
    process_parser.add_argument('--output', default='./sound_library',
                               help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        builder = SoundLibraryBuilder(output_dir=args.output)
        builder.min_uniqueness_score = args.min_uniqueness
        builder.build_library_from_directory(
            args.directory,
            max_files=args.max_files
        )
    
    elif args.command == 'browse':
        browser = SoundLibraryBrowser(library_dir=args.library)
        results = browser.search(
            category=args.category,
            tags=args.tags
        )
        
        print(f"\nFound {len(results)} samples")
        for sample in results[:20]:  # Show first 20
            print(f"\n  {Path(sample.file_path).name}")
            print(f"    Category: {sample.category} / {sample.subcategory}")
            print(f"    Duration: {sample.duration:.2f}s")
            print(f"    Uniqueness: {sample.uniqueness_score:.2f}")
            print(f"    Tags: {', '.join(sample.tags)}")
    
    elif args.command == 'stats':
        browser = SoundLibraryBrowser(library_dir=args.library)
        browser.print_statistics()
    
    elif args.command == 'export':
        browser = SoundLibraryBrowser(library_dir=args.library)
        browser.export_pack(
            args.output,
            category=args.category,
            min_uniqueness=args.min_uniqueness,
            max_samples=args.max_samples
        )
    
    elif args.command == 'process':
        builder = SoundLibraryBuilder(output_dir=args.output)
        samples = builder.process_file(args.file)
        print(f"\nExtracted {len(samples)} samples")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# ============================================================================
# Usage Examples
# ============================================================================

USAGE_EXAMPLES = """
SOUND LIBRARY EXTRACTOR - USAGE EXAMPLES
=========================================

1. BUILD LIBRARY FROM DIRECTORY:
   python sound_library.py build ./video_library --output ./sounds

2. BUILD WITH QUALITY FILTERS:
   python sound_library.py build ./videos --min-uniqueness 0.5 --max-files 50

3. BROWSE LIBRARY:
   python sound_library.py browse --category transient --tags sharp bright

4. VIEW STATISTICS:
   python sound_library.py stats --library ./sounds

5. EXPORT SAMPLE PACK:
   python sound_library.py export ./my_pack --category transient --min-uniqueness 0.7

6. PROCESS SINGLE FILE:
   python sound_library.py process video.mp4 --output ./sounds

7. PROGRAMMATIC USAGE:
   
   from sound_library import SoundLibraryBuilder, SoundLibraryBrowser
   
   # Build library
   builder = SoundLibraryBuilder()
   samples = builder.build_library_from_directory("./videos")
   
   # Browse and search
   browser = SoundLibraryBrowser()
   transients = browser.search(category="transient", min_uniqueness=0.6)
   
   # Export pack
   browser.export_pack("./transient_pack", category="transient")

8. INTEGRATED WITH VIDEO LIBRARY:
   
   from sound_library import IntegratedSoundExtractor
   
   extractor = IntegratedSoundExtractor()
   
   # Build from existing video library
   samples = extractor.build_from_video_library("./video_library")
   
   # Download and extract in one go
   samples = extractor.download_and_extract("TED Talks", max_videos=10)
"""
