"""
Audio/Video Phrase Matching and Composition Tool

This tool processes audio/video clips to create a searchable database of speech phrases
with musical characteristics, then uses this database to reconstruct vocal tracks by
finding and stitching together matching phrases.

Dependencies:
pip install whisper-openai librosa numpy scipy soundfile ffmpeg-python pydub \
            crepe essentia music21 pretty_midi midiutil

Requirements:
- ffmpeg installed on system
- CUDA recommended for faster Whisper processing
"""

import os
import json
import pickle
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess
from dataclasses import dataclass, asdict
from collections import defaultdict
import whisper
import crepe
from scipy.spatial.distance import euclidean
from scipy.signal import resample
import hashlib


@dataclass
class WordTiming:
    """Store word timing and pitch information"""
    word: str
    start: float
    end: float
    pitch_hz: float
    pitch_note: str
    confidence: float


@dataclass
class PhraseData:
    """Store phrase information"""
    phrase: str
    clip_path: str
    start_time: float
    end_time: float
    words: List[WordTiming]
    avg_pitch_hz: float
    pitch_note: str
    musicality_score: float  # 0-1, higher = more musical
    duration: float


class AudioProcessor:
    """Process audio files to extract speech and musical features"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the audio processor
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.whisper_model = whisper.load_model(model_size)
        
    def extract_audio_from_video(self, video_path: str, output_path: str) -> str:
        """Extract audio from video file"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def transcribe_with_word_timestamps(self, audio_path: str) -> Dict:
        """Transcribe audio with word-level timestamps using Whisper"""
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language='en'
        )
        return result
    
    def extract_pitch_contour(self, audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch contour using CREPE
        
        Returns:
            time, frequency, confidence arrays
        """
        y, sr = librosa.load(audio_path, sr=sr)
        time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True)
        return time, frequency, confidence
    
    def hz_to_note(self, hz: float) -> str:
        """Convert frequency in Hz to musical note"""
        if hz <= 0:
            return "N/A"
        
        A4 = 440.0
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        semitones_from_a4 = 12 * np.log2(hz / A4)
        midi_note = 69 + semitones_from_a4
        note_index = int(round(midi_note)) % 12
        octave = int(round(midi_note)) // 12 - 1
        
        return f"{note_names[note_index]}{octave}"
    
    def calculate_musicality(self, pitch_contour: np.ndarray, confidence: np.ndarray) -> float:
        """
        Calculate how musical a phrase is (0-1 score)
        Based on pitch stability, confidence, and harmonic content
        """
        if len(pitch_contour) == 0:
            return 0.0
        
        # Filter out low confidence predictions
        valid_pitches = pitch_contour[confidence > 0.5]
        
        if len(valid_pitches) < 3:
            return 0.0
        
        # Pitch stability (lower variance = more musical)
        pitch_std = np.std(valid_pitches)
        pitch_mean = np.mean(valid_pitches)
        cv = pitch_std / (pitch_mean + 1e-6)  # Coefficient of variation
        stability_score = np.exp(-cv)  # 0-1, higher = more stable
        
        # Confidence score
        confidence_score = np.mean(confidence)
        
        # Combine scores
        musicality = 0.6 * stability_score + 0.4 * confidence_score
        
        return float(np.clip(musicality, 0, 1))
    
    def get_pitch_at_time(self, target_time: float, pitch_times: np.ndarray, 
                          pitch_freqs: np.ndarray, pitch_conf: np.ndarray) -> Tuple[float, float]:
        """Get pitch frequency and confidence at specific time"""
        idx = np.argmin(np.abs(pitch_times - target_time))
        return pitch_freqs[idx], pitch_conf[idx]
    
    def process_clip(self, clip_path: str, n_gram: int = 3) -> List[PhraseData]:
        """
        Process a single audio/video clip
        
        Args:
            clip_path: Path to audio or video file
            n_gram: Number of words per phrase
            
        Returns:
            List of PhraseData objects
        """
        # Extract audio if video
        is_video = clip_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            audio_path = clip_path.rsplit('.', 1)[0] + '_temp.wav'
            self.extract_audio_from_video(clip_path, audio_path)
        else:
            audio_path = clip_path
        
        # Transcribe with word timestamps
        transcript = self.transcribe_with_word_timestamps(audio_path)
        
        # Extract pitch contour
        pitch_times, pitch_freqs, pitch_conf = self.extract_pitch_contour(audio_path)
        
        phrases = []
        
        # Process each segment
        for segment in transcript['segments']:
            if 'words' not in segment:
                continue
                
            words = segment['words']
            
            # Create n-grams
            for i in range(len(words) - n_gram + 1):
                ngram_words = words[i:i + n_gram]
                
                start_time = ngram_words[0]['start']
                end_time = ngram_words[-1]['end']
                phrase_text = ' '.join([w['word'].strip() for w in ngram_words])
                
                # Get pitch for each word
                word_timings = []
                phrase_pitches = []
                
                for word_data in ngram_words:
                    word_start = word_data['start']
                    word_end = word_data['end']
                    word_mid = (word_start + word_end) / 2
                    
                    pitch_hz, confidence = self.get_pitch_at_time(
                        word_mid, pitch_times, pitch_freqs, pitch_conf
                    )
                    
                    if confidence > 0.3 and pitch_hz > 50:
                        phrase_pitches.append(pitch_hz)
                    
                    word_timings.append(WordTiming(
                        word=word_data['word'].strip(),
                        start=word_start,
                        end=word_end,
                        pitch_hz=pitch_hz,
                        pitch_note=self.hz_to_note(pitch_hz),
                        confidence=confidence
                    ))
                
                # Calculate phrase characteristics
                if phrase_pitches:
                    avg_pitch = np.mean(phrase_pitches)
                else:
                    avg_pitch = 0.0
                
                # Get pitch contour for phrase duration
                phrase_mask = (pitch_times >= start_time) & (pitch_times <= end_time)
                phrase_pitch_contour = pitch_freqs[phrase_mask]
                phrase_confidence = pitch_conf[phrase_mask]
                
                musicality = self.calculate_musicality(phrase_pitch_contour, phrase_confidence)
                
                phrase = PhraseData(
                    phrase=phrase_text,
                    clip_path=clip_path,
                    start_time=start_time,
                    end_time=end_time,
                    words=word_timings,
                    avg_pitch_hz=avg_pitch,
                    pitch_note=self.hz_to_note(avg_pitch),
                    musicality_score=musicality,
                    duration=end_time - start_time
                )
                
                phrases.append(phrase)
        
        # Clean up temp audio file
        if is_video and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return phrases


class PhraseDatabase:
    """Searchable database of audio phrases"""
    
    def __init__(self, db_path: str = "phrase_database.pkl"):
        self.db_path = db_path
        self.phrases: List[PhraseData] = []
        self.phrase_index: Dict[str, List[int]] = defaultdict(list)
        
    def add_phrases(self, phrases: List[PhraseData]):
        """Add phrases to database"""
        start_idx = len(self.phrases)
        self.phrases.extend(phrases)
        
        # Index phrases
        for i, phrase in enumerate(phrases, start=start_idx):
            phrase_lower = phrase.phrase.lower()
            self.phrase_index[phrase_lower].append(i)
    
    def save(self):
        """Save database to disk"""
        with open(self.db_path, 'wb') as f:
            pickle.dump({
                'phrases': self.phrases,
                'phrase_index': dict(self.phrase_index)
            }, f)
    
    def load(self):
        """Load database from disk"""
        with open(self.db_path, 'rb') as f:
            data = pickle.load(f)
            self.phrases = data['phrases']
            self.phrase_index = defaultdict(list, data['phrase_index'])
    
    def search(self, query: str, target_pitch_hz: Optional[float] = None,
               target_duration: Optional[float] = None,
               max_results: int = 10) -> List[Tuple[PhraseData, float]]:
        """
        Search for matching phrases
        
        Args:
            query: Text to search for
            target_pitch_hz: Target pitch in Hz
            target_duration: Target duration in seconds
            max_results: Maximum number of results
            
        Returns:
            List of (PhraseData, score) tuples, sorted by relevance
        """
        query_lower = query.lower()
        
        # Exact matches
        if query_lower in self.phrase_index:
            candidates = [self.phrases[i] for i in self.phrase_index[query_lower]]
        else:
            # Fuzzy matching - find phrases containing query words
            query_words = set(query_lower.split())
            candidates = []
            
            for phrase in self.phrases:
                phrase_words = set(phrase.phrase.lower().split())
                overlap = len(query_words & phrase_words)
                if overlap > 0:
                    candidates.append(phrase)
        
        # Score candidates
        scored_results = []
        for phrase in candidates:
            score = self._calculate_match_score(
                phrase, query, target_pitch_hz, target_duration
            )
            scored_results.append((phrase, score))
        
        # Sort by score (higher is better)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:max_results]
    
    def _calculate_match_score(self, phrase: PhraseData, query: str,
                               target_pitch_hz: Optional[float],
                               target_duration: Optional[float]) -> float:
        """Calculate match score for a phrase"""
        score = 0.0
        
        # Text similarity (40%)
        query_words = set(query.lower().split())
        phrase_words = set(phrase.phrase.lower().split())
        
        if len(query_words) > 0:
            text_similarity = len(query_words & phrase_words) / len(query_words)
            score += 0.4 * text_similarity
        
        # Pitch similarity (30%)
        if target_pitch_hz and phrase.avg_pitch_hz > 0:
            # Use semitone distance
            semitone_diff = abs(12 * np.log2(phrase.avg_pitch_hz / target_pitch_hz))
            pitch_similarity = np.exp(-semitone_diff / 6)  # Decay over 6 semitones
            score += 0.3 * pitch_similarity
        
        # Duration similarity (20%)
        if target_duration:
            duration_ratio = min(phrase.duration, target_duration) / max(phrase.duration, target_duration)
            score += 0.2 * duration_ratio
        
        # Musicality bonus (10%)
        score += 0.1 * phrase.musicality_score
        
        return score


class VocalTrackComposer:
    """Compose video from database phrases matching a vocal track"""
    
    def __init__(self, database: PhraseDatabase, processor: AudioProcessor):
        self.database = database
        self.processor = processor
        
    def align_phrase_timing(self, target_start: float, target_duration: float,
                           phrase: PhraseData) -> Tuple[float, float]:
        """
        Calculate time stretch factor to align phrase with target
        
        Returns:
            (stretch_factor, adjusted_start_time)
        """
        stretch_factor = target_duration / phrase.duration
        return stretch_factor, target_start
    
    def compose_from_vocal_track(self, vocal_track_path: str,
                                 output_dir: str = "composed",
                                 n_gram: int = 3) -> Dict[float, Dict]:
        """
        Create composition dictionary from vocal track
        
        Returns:
            Dictionary mapping {time: clip_info}
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the vocal track
        print("Processing vocal track...")
        vocal_phrases = self.processor.process_clip(vocal_track_path, n_gram=n_gram)
        
        composition = {}
        
        for i, target_phrase in enumerate(vocal_phrases):
            print(f"Finding match for phrase {i+1}/{len(vocal_phrases)}: '{target_phrase.phrase}'")
            
            # Search for best match
            matches = self.database.search(
                query=target_phrase.phrase,
                target_pitch_hz=target_phrase.avg_pitch_hz,
                target_duration=target_phrase.duration,
                max_results=5
            )
            
            if matches:
                best_match, score = matches[0]
                
                stretch_factor, adjusted_start = self.align_phrase_timing(
                    target_phrase.start_time,
                    target_phrase.duration,
                    best_match
                )
                
                composition[target_phrase.start_time] = {
                    'phrase': target_phrase.phrase,
                    'source_clip': best_match.clip_path,
                    'source_start': best_match.start_time,
                    'source_end': best_match.end_time,
                    'target_start': target_phrase.start_time,
                    'target_end': target_phrase.end_time,
                    'stretch_factor': stretch_factor,
                    'match_score': score,
                    'pitch_shift_semitones': 12 * np.log2(target_phrase.avg_pitch_hz / best_match.avg_pitch_hz) if best_match.avg_pitch_hz > 0 else 0
                }
            else:
                print(f"  No match found for '{target_phrase.phrase}'")
        
        return composition
    
    def stitch_video(self, composition: Dict[float, Dict], 
                    output_path: str = "output.mp4",
                    fallback_image: Optional[str] = None,
                    fallback_video: Optional[str] = None):
        """
        Stitch together video from composition dictionary using ffmpeg
        
        Args:
            composition: Dictionary from compose_from_vocal_track
            output_path: Output video path
            fallback_image: Path to still image for clips without video
            fallback_video: Path to looping video for clips without video
        """
        temp_dir = "temp_clips"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create file list for concatenation
        concat_list = []
        
        sorted_times = sorted(composition.keys())
        
        for i, start_time in enumerate(sorted_times):
            clip_info = composition[start_time]
            
            source_clip = clip_info['source_clip']
            source_start = clip_info['source_start']
            source_end = clip_info['source_end']
            stretch = clip_info['stretch_factor']
            pitch_shift = clip_info['pitch_shift_semitones']
            
            temp_output = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
            
            # Check if source has video
            is_video = source_clip.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            if is_video:
                # Extract and process video clip
                self._extract_clip_with_effects(
                    source_clip, temp_output,
                    source_start, source_end,
                    stretch, pitch_shift
                )
            else:
                # Audio only - use fallback visuals
                self._create_audio_with_visuals(
                    source_clip, temp_output,
                    source_start, source_end,
                    stretch, pitch_shift,
                    fallback_image, fallback_video
                )
            
            concat_list.append(f"file '{os.path.abspath(temp_output)}'")
        
        # Write concat list
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_list))
        
        # Concatenate all clips
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy', '-y', output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"Output video created: {output_path}")
    
    def _extract_clip_with_effects(self, source_path: str, output_path: str,
                                   start: float, end: float,
                                   stretch: float, pitch_shift: float):
        """Extract clip with time stretching and pitch shifting"""
        duration = end - start
        
        # Build ffmpeg filter
        filters = []
        
        # Time stretch (atempo can only handle 0.5-2.0 range)
        if abs(stretch - 1.0) > 0.01:
            if 0.5 <= stretch <= 2.0:
                filters.append(f"atempo={stretch}")
            else:
                # Chain multiple atempo filters
                remaining = stretch
                while remaining > 2.0:
                    filters.append("atempo=2.0")
                    remaining /= 2.0
                while remaining < 0.5:
                    filters.append("atempo=0.5")
                    remaining /= 0.5
                if abs(remaining - 1.0) > 0.01:
                    filters.append(f"atempo={remaining}")
        
        # Pitch shift (using rubberband if available, otherwise asetrate)
        if abs(pitch_shift) > 0.1:
            # Simple pitch shift using asetrate (changes tempo too, so apply after atempo)
            semitone_ratio = 2 ** (pitch_shift / 12)
            filters.append(f"asetrate=44100*{semitone_ratio},aresample=44100")
        
        filter_str = ','.join(filters) if filters else "anull"
        
        cmd = [
            'ffmpeg', '-ss', str(start), '-i', source_path,
            '-t', str(duration),
            '-filter_complex', f"[0:a]{filter_str}[a]",
            '-map', '0:v', '-map', '[a]',
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _create_audio_with_visuals(self, audio_path: str, output_path: str,
                                   start: float, end: float,
                                   stretch: float, pitch_shift: float,
                                   fallback_image: Optional[str],
                                   fallback_video: Optional[str]):
        """Create video from audio with fallback visuals"""
        duration = end - start
        
        # Similar audio processing as above
        filters = []
        if abs(stretch - 1.0) > 0.01:
            if 0.5 <= stretch <= 2.0:
                filters.append(f"atempo={stretch}")
        
        if abs(pitch_shift) > 0.1:
            semitone_ratio = 2 ** (pitch_shift / 12)
            filters.append(f"asetrate=44100*{semitone_ratio},aresample=44100")
        
        filter_str = ','.join(filters) if filters else "anull"
        
        if fallback_video:
            # Use looping video
            cmd = [
                'ffmpeg',
                '-stream_loop', '-1', '-i', fallback_video,
                '-ss', str(start), '-i', audio_path,
                '-t', str(duration),
                '-filter_complex', f"[1:a]{filter_str}[a]",
                '-map', '0:v', '-map', '[a]',
                '-shortest', '-y', output_path
            ]
        elif fallback_image:
            # Use static image
            cmd = [
                'ffmpeg',
                '-loop', '1', '-i', fallback_image,
                '-ss', str(start), '-i', audio_path,
                '-t', str(duration),
                '-filter_complex', f"[1:a]{filter_str}[a]",
                '-map', '0:v', '-map', '[a]',
                '-shortest', '-y', output_path
            ]
        else:
            # Just audio with black video
            cmd = [
                'ffmpeg',
                '-f', 'lavfi', '-i', f'color=c=black:s=1280x720:d={duration}',
                '-ss', str(start), '-i', audio_path,
                '-filter_complex', f"[1:a]{filter_str}[a]",
                '-map', '0:v', '-map', '[a]',
                '-y', output_path
            ]
        
        subprocess.run(cmd, check=True, capture_output=True)


def build_database_from_directory(directory: str, db_path: str = "phrase_database.pkl",
                                  n_gram: int = 3, model_size: str = "base"):
    """
    Build phrase database from directory of audio/video files
    
    Args:
        directory: Directory containing audio/video files
        db_path: Path to save database
        n_gram: Number of words per phrase
        model_size: Whisper model size
    """
    processor = AudioProcessor(model_size=model_size)
    database = PhraseDatabase(db_path=db_path)
    
    # Find all audio/video files
    extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.flac']
    files = []
    
    for ext in extensions:
        files.extend(Path(directory).rglob(f'*{ext}'))
    
    print(f"Found {len(files)} files to process")
    
    for i, file_path in enumerate(files, 1):
        print(f"\nProcessing {i}/{len(files)}: {file_path.name}")
        try:
            phrases = processor.process_clip(str(file_path), n_gram=n_gram)
            database.add_phrases(phrases)
            print(f"  Added {len(phrases)} phrases")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    database.save()
    print(f"\nDatabase saved with {len(database.phrases)} total phrases")


def compose_video_from_vocal(vocal_track: str, database_path: str,
                             output_path: str = "output.mp4",
                             fallback_image: Optional[str] = None,
                             fallback_video: Optional[str] = None,
                             n_gram: int = 3,
                             model_size: str = "base"):
    """
    Compose video from vocal track using phrase database
    
    Args:
        vocal_track: Path to vocal track audio/video
        database_path: Path to phrase database
        output_path: Output video path
        fallback_image: Path to fallback image
        fallback_video: Path to fallback video
        n_gram: Phrase size
        model_size: Whisper model size
    """
    processor = AudioProcessor(model_size=model_size)
    database = PhraseDatabase(db_path=database_path)
    database.load()
    
    composer = VocalTrackComposer(database, processor)
    
    print("Creating composition...")
    composition = composer.compose_from_vocal_track(vocal_track, n_gram=n_gram)
    
    # Save composition as JSON
    composition_json = {
        str(k): {**v, 'phrase': str(v['phrase'])}  # Ensure serializable
        for k, v in composition.items()
    }
    
    with open('composition.json', 'w') as f:
        json.dump(composition_json, f, indent=2)
    
    print("\nStitching video...")
    composer.stitch_video(composition, output_path, fallback_image, fallback_video)


# ============================================================================
# EXTENSION 1: Generate missing phrases using Whisper TTS
# ============================================================================

class WhisperTTSGenerator:
    """Generate missing vocal phrases (requires external TTS system)"""
    
    def __init__(self):
        # Note: Whisper is speech-to-text only
        # For TTS, you'd need to integrate with systems like:
        # - Coqui TTS, Bark, or commercial APIs (ElevenLabs, Azure TTS)
        pass
    
    def generate_phrase(self, text: str, target_pitch_hz: float,
                       output_path: str) -> str:
        """
        Generate synthetic speech for missing phrase
        
        This is a placeholder - integrate with actual TTS system
        """
        print(f"[TTS] Would generate: '{text}' at {target_pitch_hz:.1f}Hz")
        # Implementation would call TTS API here
        return output_path


# ============================================================================
# EXTENSION 2: Repitch audio samples
# ============================================================================

class AudioRepitcher:
    """Repitch audio samples to match target key"""
    
    @staticmethod
    def repitch_audio(input_path: str, output_path: str,
                     semitone_shift: float):
        """
        Repitch audio file by semitone amount using rubberband or sox
        
        Args:
            input_path: Input audio file
            output_path: Output audio file
            semitone_shift: Semitones to shift (positive = up, negative = down)
        """
        try:
            # Try using rubberband (best quality)
            cmd = [
                'rubberband',
                '--pitch', str(semitone_shift),
                input_path,
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  Repitched by {semitone_shift:.2f} semitones using rubberband")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to sox
            try:
                cmd = [
                    'sox', input_path, output_path,
                    'pitch', str(semitone_shift * 100)  # sox uses cents
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  Repitched by {semitone_shift:.2f} semitones using sox")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to ffmpeg asetrate (changes tempo too)
                semitone_ratio = 2 ** (semitone_shift / 12)
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-filter:a', f'asetrate=44100*{semitone_ratio},aresample=44100',
                    '-y', output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  Repitched by {semitone_shift:.2f} semitones using ffmpeg")
    
    @staticmethod
    def repitch_clip_segment(input_path: str, output_path: str,
                            start: float, end: float,
                            semitone_shift: float):
        """Extract segment and repitch it"""
        temp_segment = "temp_segment.wav"
        
        # Extract segment
        cmd = [
            'ffmpeg', '-ss', str(start), '-i', input_path,
            '-t', str(end - start),
            '-y', temp_segment
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Repitch
        AudioRepitcher.repitch_audio(temp_segment, output_path, semitone_shift)
        
        # Cleanup
        if os.path.exists(temp_segment):
            os.remove(temp_segment)


# ============================================================================
# EXTENSION 3: Fill gaps with generative video (placeholder)
# ============================================================================

class GenerativeVideoFiller:
    """
    Fill gaps between video clips using generative AI
    
    This is a placeholder for integration with video generation models like:
    - Stable Video Diffusion, Gen-2, Pika, etc.
    """
    
    def __init__(self, model_name: str = "stable-video-diffusion"):
        self.model_name = model_name
        print(f"[Note] Generative video requires external API: {model_name}")
    
    def generate_transition(self, prev_frame_path: str, next_frame_path: str,
                           duration: float, output_path: str) -> str:
        """
        Generate transition video between two frames
        
        Args:
            prev_frame_path: Last frame of previous clip
            next_frame_path: First frame of next clip
            duration: Duration of transition in seconds
            output_path: Output video path
            
        Returns:
            Path to generated video
        """
        print(f"[GenVideo] Would generate {duration}s transition")
        
        # Placeholder: Create simple crossfade instead
        cmd = [
            'ffmpeg',
            '-loop', '1', '-t', str(duration/2), '-i', prev_frame_path,
            '-loop', '1', '-t', str(duration/2), '-i', next_frame_path,
            '-filter_complex',
            f'[0:v][1:v]xfade=transition=fade:duration={duration/2}:offset={duration/2}',
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path


# ============================================================================
# EXTENSION 4: Extract MIDI from audio
# ============================================================================

class MIDIExtractor:
    """Extract MIDI information from audio files"""
    
    def __init__(self):
        try:
            import pretty_midi
            from music21 import converter, instrument, note, stream, tempo
            self.pretty_midi = pretty_midi
            self.music21 = (converter, instrument, note, stream, tempo)
        except ImportError:
            print("Warning: Install pretty_midi and music21 for full MIDI support")
    
    def extract_midi_from_audio(self, audio_path: str, output_midi: str,
                                hop_length: int = 512):
        """
        Extract MIDI from monophonic audio
        
        Args:
            audio_path: Input audio file
            output_midi: Output MIDI file path
            hop_length: Hop length for pitch detection
        """
        import pretty_midi
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract pitch using CREPE
        time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True)
        
        # Filter by confidence
        valid_mask = confidence > 0.5
        time = time[valid_mask]
        frequency = frequency[valid_mask]
        confidence = confidence[valid_mask]
        
        # Convert to MIDI notes
        midi_notes = 69 + 12 * np.log2(frequency / 440.0)
        midi_notes = np.round(midi_notes).astype(int)
        
        # Extract note onsets and durations
        notes = self._extract_notes_from_contour(
            time, midi_notes, confidence
        )
        
        # Create MIDI file
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        for note_info in notes:
            note = pretty_midi.Note(
                velocity=note_info['velocity'],
                pitch=note_info['pitch'],
                start=note_info['start'],
                end=note_info['end']
            )
            instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        midi.write(output_midi)
        
        print(f"Extracted {len(notes)} notes to {output_midi}")
        
        return output_midi
    
    def _extract_notes_from_contour(self, time: np.ndarray,
                                    midi_notes: np.ndarray,
                                    confidence: np.ndarray) -> List[Dict]:
        """Extract discrete notes from pitch contour"""
        notes = []
        
        if len(midi_notes) == 0:
            return notes
        
        # Group consecutive similar pitches into notes
        current_pitch = midi_notes[0]
        current_start = time[0]
        current_confidences = [confidence[0]]
        
        for i in range(1, len(midi_notes)):
            # New note if pitch changes by more than 0.5 semitones
            if abs(midi_notes[i] - current_pitch) > 0.5:
                # Save previous note
                avg_confidence = np.mean(current_confidences)
                velocity = int(np.clip(avg_confidence * 127, 40, 127))
                
                notes.append({
                    'pitch': int(current_pitch),
                    'start': float(current_start),
                    'end': float(time[i-1]),
                    'velocity': velocity
                })
                
                # Start new note
                current_pitch = midi_notes[i]
                current_start = time[i]
                current_confidences = [confidence[i]]
            else:
                current_confidences.append(confidence[i])
        
        # Add final note
        if len(current_confidences) > 0:
            avg_confidence = np.mean(current_confidences)
            velocity = int(np.clip(avg_confidence * 127, 40, 127))
            
            notes.append({
                'pitch': int(current_pitch),
                'start': float(current_start),
                'end': float(time[-1]),
                'velocity': velocity
            })
        
        return notes
    
    def extract_advanced_midi(self, audio_path: str, output_midi: str):
        """
        Extract MIDI with more advanced features (polyphonic, ADSR)
        
        Note: Basic monophonic extraction. For polyphonic, consider:
        - Omnizart library
        - Basic Pitch by Spotify
        - Commercial APIs
        """
        # Use basic extraction
        self.extract_midi_from_audio(audio_path, output_midi)
        
        # ADSR envelope extraction would require additional analysis
        # This would involve detecting:
        # - Attack: onset detection
        # - Decay: amplitude envelope analysis
        # - Sustain: stable amplitude period
        # - Release: offset detection
        
        print("Note: ADSR envelope extraction requires additional audio analysis")


# ============================================================================
# Enhanced Composer with Extensions
# ============================================================================

class EnhancedVocalTrackComposer(VocalTrackComposer):
    """Extended composer with repitching and TTS generation"""
    
    def __init__(self, database: PhraseDatabase, processor: AudioProcessor,
                 enable_tts: bool = False, enable_repitch: bool = True):
        super().__init__(database, processor)
        self.tts_generator = WhisperTTSGenerator() if enable_tts else None
        self.enable_repitch = enable_repitch
        self.repitcher = AudioRepitcher()
    
    def compose_from_vocal_track(self, vocal_track_path: str,
                                 output_dir: str = "composed",
                                 n_gram: int = 3,
                                 max_pitch_diff_semitones: float = 2.0) -> Dict[float, Dict]:
        """
        Enhanced composition with repitching
        
        Args:
            max_pitch_diff_semitones: Maximum pitch difference before repitching
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the vocal track
        print("Processing vocal track...")
        vocal_phrases = self.processor.process_clip(vocal_track_path, n_gram=n_gram)
        
        composition = {}
        
        for i, target_phrase in enumerate(vocal_phrases):
            print(f"Finding match for phrase {i+1}/{len(vocal_phrases)}: '{target_phrase.phrase}'")
            
            # Search for best match
            matches = self.database.search(
                query=target_phrase.phrase,
                target_pitch_hz=target_phrase.avg_pitch_hz,
                target_duration=target_phrase.duration,
                max_results=5
            )
            
            if matches:
                best_match, score = matches[0]
                
                # Calculate pitch difference
                pitch_shift = 0.0
                if best_match.avg_pitch_hz > 0 and target_phrase.avg_pitch_hz > 0:
                    pitch_shift = 12 * np.log2(target_phrase.avg_pitch_hz / best_match.avg_pitch_hz)
                
                # Repitch if needed
                source_clip = best_match.clip_path
                if self.enable_repitch and abs(pitch_shift) > max_pitch_diff_semitones:
                    print(f"  Repitching by {pitch_shift:.2f} semitones")
                    repitched_path = os.path.join(output_dir, f"repitched_{i:04d}.wav")
                    
                    self.repitcher.repitch_clip_segment(
                        best_match.clip_path,
                        repitched_path,
                        best_match.start_time,
                        best_match.end_time,
                        pitch_shift
                    )
                    
                    source_clip = repitched_path
                    pitch_shift = 0.0  # Already corrected
                
                stretch_factor, adjusted_start = self.align_phrase_timing(
                    target_phrase.start_time,
                    target_phrase.duration,
                    best_match
                )
                
                composition[target_phrase.start_time] = {
                    'phrase': target_phrase.phrase,
                    'source_clip': source_clip,
                    'source_start': 0.0 if 'repitched' in source_clip else best_match.start_time,
                    'source_end': target_phrase.duration if 'repitched' in source_clip else best_match.end_time,
                    'target_start': target_phrase.start_time,
                    'target_end': target_phrase.end_time,
                    'stretch_factor': stretch_factor,
                    'match_score': score,
                    'pitch_shift_semitones': pitch_shift,
                    'was_repitched': abs(pitch_shift) < 0.1
                }
            else:
                print(f"  No match found for '{target_phrase.phrase}'")
                
                # Try TTS generation if enabled
                if self.tts_generator:
                    print(f"  Generating with TTS...")
                    # TTS integration would go here
        
        return composition


# Example usage
if __name__ == "__main__":
    # Step 1: Build database from directory
    print("=" * 50)
    print("Building phrase database...")
    print("=" * 50)
    build_database_from_directory(
        directory="./source_clips",
        db_path="phrase_database.pkl",
        n_gram=3,
        model_size="base"
    )
    
    # Step 2: Compose video with extensions
    print("\n" + "=" * 50)
    print("Composing video with enhanced features...")
    print("=" * 50)
    
    processor = AudioProcessor(model_size="base")
    database = PhraseDatabase(db_path="phrase_database.pkl")
    database.load()
    
    # Use enhanced composer with repitching
    composer = EnhancedVocalTrackComposer(
        database, processor,
        enable_tts=False,
        enable_repitch=True
    )
    
    composition = composer.compose_from_vocal_track(
        "./vocal_track.mp3",
        n_gram=3
    )
    
    composer.stitch_video(
        composition,
        output_path="output.mp4",
        fallback_image="./fallback.jpg"
    )
    
    # Step 3: Extract MIDI from audio (Extension 4)
    print("\n" + "=" * 50)
    print("Extracting MIDI...")
    print("=" * 50)
    
    midi_extractor = MIDIExtractor()
    midi_extractor.extract_midi_from_audio(
        "./vocal_track.mp3",
        "output.mid"
    )
