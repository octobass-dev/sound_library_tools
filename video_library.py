"""
Video Library Downloader & Vocal Replacement System

This tool finds, downloads, and processes speech-heavy video libraries,
then uses them to create alternate vocal tracks for YouTube videos.

Dependencies:
pip install yt-dlp demucs internetarchive requests beautifulsoup4 lxml tqdm

Additional requirements from previous tool:
pip install whisper-openai librosa numpy scipy soundfile ffmpeg-python pydub \
            crepe essentia music21 pretty_midi
"""

import os
import json
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time
from tqdm import tqdm
import hashlib


# ============================================================================
# Video Library Sources
# ============================================================================

@dataclass
class VideoSource:
    """Information about a video source"""
    name: str
    url: str
    source_type: str  # 'archive.org', 'youtube_channel', 'dataset'
    description: str
    estimated_hours: float


class VideoLibraryFinder:
    """Find and catalog freely available video libraries with speech"""
    
    # Curated list of high-quality speech video sources
    SPEECH_SOURCES = [
        VideoSource(
            name="C-SPAN Video Library",
            url="https://archive.org/details/cspan",
            source_type="archive.org",
            description="Public affairs programming, congressional sessions, speeches",
            estimated_hours=1000
        ),
        VideoSource(
            name="American Rhetoric Movie Speeches",
            url="https://archive.org/details/great_speeches",
            source_type="archive.org",
            description="Famous movie speeches and dialogues",
            estimated_hours=50
        ),
        VideoSource(
            name="Public Domain Movies",
            url="https://archive.org/details/feature_films",
            source_type="archive.org",
            description="Classic public domain films with dialogue",
            estimated_hours=5000
        ),
        VideoSource(
            name="Prelinger Archives",
            url="https://archive.org/details/prelinger",
            source_type="archive.org",
            description="Educational films, advertisements, industrial videos",
            estimated_hours=2000
        ),
        VideoSource(
            name="Television Archive",
            url="https://archive.org/details/tv",
            source_type="archive.org",
            description="TV news broadcasts and shows",
            estimated_hours=10000
        ),
        VideoSource(
            name="TED Talks",
            url="https://archive.org/details/TedTalks",
            source_type="archive.org",
            description="Public speaking and presentations",
            estimated_hours=500
        ),
        VideoSource(
            name="Educational Videos",
            url="https://archive.org/details/opensource_movies",
            source_type="archive.org",
            description="Open source educational content",
            estimated_hours=1000
        ),
        VideoSource(
            name="Political Campaign Ads Archive",
            url="https://archive.org/details/political_campaign_ads",
            source_type="archive.org",
            description="Political advertisements with speech",
            estimated_hours=100
        ),
        VideoSource(
            name="NASA Video Archive",
            url="https://archive.org/details/nasa",
            source_type="archive.org",
            description="NASA videos with technical speech",
            estimated_hours=500
        ),
        VideoSource(
            name="Internet Archive TV News",
            url="https://archive.org/details/tv?&and[]=mediatype%3A%22movies%22",
            source_type="archive.org",
            description="News broadcasts from various networks",
            estimated_hours=50000
        )
    ]
    
    def list_sources(self) -> List[VideoSource]:
        """Get list of available video sources"""
        return self.SPEECH_SOURCES
    
    def print_sources(self):
        """Print available sources in a readable format"""
        print("\n" + "=" * 80)
        print("AVAILABLE SPEECH VIDEO SOURCES")
        print("=" * 80)
        
        for i, source in enumerate(self.SPEECH_SOURCES, 1):
            print(f"\n{i}. {source.name}")
            print(f"   URL: {source.url}")
            print(f"   Type: {source.source_type}")
            print(f"   Description: {source.description}")
            print(f"   Estimated Hours: ~{source.estimated_hours:.0f}")
    
    def get_source_by_name(self, name: str) -> Optional[VideoSource]:
        """Get source by name"""
        for source in self.SPEECH_SOURCES:
            if source.name.lower() == name.lower():
                return source
        return None


# ============================================================================
# Video Downloader for Multiple Sources
# ============================================================================

class VideoLibraryDownloader:
    """Download videos from various sources"""
    
    def __init__(self, output_dir: str = "./video_library"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import internetarchive
        try:
            import internetarchive as ia
            self.ia = ia
            self.ia_available = True
        except ImportError:
            print("Warning: internetarchive not installed. Install with: pip install internetarchive")
            self.ia_available = False
    
    def download_from_archive_org(self, collection_id: str, max_items: int = 100,
                                  min_duration: float = 60.0,
                                  max_duration: float = 3600.0) -> List[str]:
        """
        Download videos from Internet Archive collection
        
        Args:
            collection_id: Archive.org collection identifier
            max_items: Maximum number of items to download
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            
        Returns:
            List of downloaded file paths
        """
        if not self.ia_available:
            print("Internet Archive library not available")
            return []
        
        downloaded = []
        
        try:
            # Search collection
            search_results = self.ia.search_items(
                f'collection:{collection_id}',
                fields=['identifier', 'title', 'mediatype']
            )
            
            print(f"\nDownloading from collection: {collection_id}")
            print(f"Maximum items: {max_items}")
            
            count = 0
            for result in search_results:
                if count >= max_items:
                    break
                
                identifier = result['identifier']
                
                try:
                    # Get item details
                    item = self.ia.get_item(identifier)
                    
                    # Find video files
                    video_files = [
                        f for f in item.files
                        if f.get('format', '').lower() in ['mpeg4', 'h.264', 'mp4', 'avi', 'mov']
                    ]
                    
                    if not video_files:
                        continue
                    
                    # Get largest video file (usually best quality)
                    video_file = max(video_files, key=lambda x: int(x.get('size', 0)))
                    
                    # Check duration if available
                    duration = float(video_file.get('length', 0))
                    if duration > 0 and (duration < min_duration or duration > max_duration):
                        continue
                    
                    # Download
                    output_path = self.output_dir / collection_id / f"{identifier}.mp4"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_path.exists():
                        print(f"  Skipping (already exists): {identifier}")
                        downloaded.append(str(output_path))
                        count += 1
                        continue
                    
                    print(f"  Downloading: {identifier} ({video_file.get('size', 0) / 1e6:.1f} MB)")
                    
                    # Download using ia command line (more reliable)
                    cmd = [
                        'ia', 'download', identifier,
                        '--glob', video_file['name'],
                        '--destdir', str(output_path.parent)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Move/rename file
                        downloaded_file = output_path.parent / identifier / video_file['name']
                        if downloaded_file.exists():
                            downloaded_file.rename(output_path)
                            # Clean up directory
                            downloaded_file.parent.rmdir()
                            downloaded.append(str(output_path))
                            count += 1
                    else:
                        print(f"    Error downloading: {result.stderr}")
                
                except Exception as e:
                    print(f"  Error processing {identifier}: {e}")
                
                # Rate limiting
                time.sleep(1)
        
        except Exception as e:
            print(f"Error searching collection: {e}")
        
        print(f"\nDownloaded {len(downloaded)} videos")
        return downloaded
    
    def download_youtube_playlist(self, playlist_url: str, max_items: int = 50) -> List[str]:
        """
        Download videos from YouTube playlist
        
        Args:
            playlist_url: YouTube playlist URL
            max_items: Maximum videos to download
        """
        output_path = self.output_dir / "youtube_playlists"
        output_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'yt-dlp',
            '--playlist-end', str(max_items),
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', str(output_path / '%(id)s.%(ext)s'),
            '--no-playlist-download-archive',
            playlist_url
        ]
        
        print(f"Downloading YouTube playlist...")
        subprocess.run(cmd, check=True)
        
        # Get list of downloaded files
        downloaded = list(output_path.glob('*.mp4'))
        return [str(f) for f in downloaded]
    
    def download_from_source(self, source: VideoSource, max_items: int = 100) -> List[str]:
        """Download videos from a VideoSource"""
        if source.source_type == "archive.org":
            # Extract collection ID from URL
            collection_id = source.url.split('/')[-1].split('?')[0]
            return self.download_from_archive_org(collection_id, max_items)
        elif source.source_type == "youtube_channel":
            return self.download_youtube_playlist(source.url, max_items)
        else:
            print(f"Unsupported source type: {source.source_type}")
            return []


# ============================================================================
# YouTube Audio Downloader with Vocal Separation
# ============================================================================

class YouTubeVocalProcessor:
    """Download YouTube audio and separate vocals"""
    
    def __init__(self, output_dir: str = "./youtube_processing"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_audio(self, youtube_url: str) -> str:
        """
        Download audio from YouTube using yt-dlp
        
        Returns:
            Path to downloaded audio file
        """
        # Generate output filename
        video_id = self._extract_video_id(youtube_url)
        output_path = self.output_dir / f"{video_id}_original.wav"
        
        if output_path.exists():
            print(f"Audio already downloaded: {output_path}")
            return str(output_path)
        
        print(f"Downloading audio from: {youtube_url}")
        
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',
            '--audio-quality', '0',  # Best quality
            '-o', str(self.output_dir / f"{video_id}_original.%(ext)s"),
            youtube_url
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Downloaded: {output_path}")
        
        return str(output_path)
    
    def download_video(self, youtube_url: str) -> str:
        """Download video from YouTube"""
        video_id = self._extract_video_id(youtube_url)
        output_path = self.output_dir / f"{video_id}_video.mp4"
        
        if output_path.exists():
            print(f"Video already downloaded: {output_path}")
            return str(output_path)
        
        print(f"Downloading video from: {youtube_url}")
        
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', str(output_path),
            youtube_url
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Downloaded: {output_path}")
        
        return str(output_path)
    
    def separate_vocals(self, audio_path: str, model: str = "htdemucs") -> Dict[str, str]:
        """
        Separate vocals using Demucs
        
        Args:
            audio_path: Path to audio file
            model: Demucs model to use (htdemucs, htdemucs_ft, htdemucs_6s)
                   htdemucs_6s separates into: drums, bass, other, vocals, guitar, piano
        
        Returns:
            Dictionary with paths to separated stems
        """
        print(f"\nSeparating vocals using Demucs ({model})...")
        
        # Demucs output directory
        demucs_output = self.output_dir / "demucs_output"
        
        cmd = [
            'demucs',
            '--out', str(demucs_output),
            '--name', model,
            '-n', model,
            audio_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Find output files
        audio_name = Path(audio_path).stem
        model_dir = demucs_output / model / audio_name
        
        stems = {}
        
        # Standard 4-stem output
        for stem in ['vocals', 'drums', 'bass', 'other']:
            stem_path = model_dir / f"{stem}.wav"
            if stem_path.exists():
                stems[stem] = str(stem_path)
        
        # Extended 6-stem output
        if model == 'htdemucs_6s':
            for stem in ['guitar', 'piano']:
                stem_path = model_dir / f"{stem}.wav"
                if stem_path.exists():
                    stems[stem] = str(stem_path)
        
        print(f"Separated stems: {list(stems.keys())}")
        
        return stems
    
    def create_instrumental(self, stems: Dict[str, str], output_path: str) -> str:
        """
        Create instrumental track by mixing all non-vocal stems
        
        Args:
            stems: Dictionary of stem paths
            output_path: Output path for instrumental
        """
        print("\nCreating instrumental track...")
        
        # Mix all non-vocal stems
        non_vocal_stems = [path for name, path in stems.items() if name != 'vocals']
        
        if not non_vocal_stems:
            print("No non-vocal stems found")
            return None
        
        if len(non_vocal_stems) == 1:
            # Just copy the single stem
            subprocess.run(['cp', non_vocal_stems[0], output_path], check=True)
        else:
            # Mix multiple stems using ffmpeg
            inputs = []
            for stem in non_vocal_stems:
                inputs.extend(['-i', stem])
            
            filter_complex = f"amix=inputs={len(non_vocal_stems)}:duration=longest"
            
            cmd = ['ffmpeg'] + inputs + [
                '-filter_complex', filter_complex,
                '-y', output_path
            ]
            
            subprocess.run(cmd, check=True)
        
        print(f"Created instrumental: {output_path}")
        return output_path
    
    def _extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        # Handle various YouTube URL formats
        if 'youtu.be/' in youtube_url:
            return youtube_url.split('youtu.be/')[-1].split('?')[0]
        elif 'youtube.com/watch?v=' in youtube_url:
            return youtube_url.split('v=')[-1].split('&')[0]
        else:
            # Use hash of URL as fallback
            return hashlib.md5(youtube_url.encode()).hexdigest()[:12]


# ============================================================================
# Complete Pipeline: Database + Vocal Replacement
# ============================================================================

class VocalReplacementPipeline:
    """Complete pipeline for downloading, processing, and vocal replacement"""
    
    def __init__(self, library_dir: str = "./video_library",
                 database_path: str = "./phrase_database.pkl",
                 work_dir: str = "./work"):
        self.library_dir = Path(library_dir)
        self.database_path = database_path
        self.work_dir = Path(work_dir)
        
        self.library_finder = VideoLibraryFinder()
        self.downloader = VideoLibraryDownloader(str(self.library_dir))
        self.youtube_processor = YouTubeVocalProcessor(str(self.work_dir / "youtube"))
        
        # Import from previous tool
        try:
            from __main__ import (
                AudioProcessor, PhraseDatabase, 
                EnhancedVocalTrackComposer, build_database_from_directory
            )
            self.AudioProcessor = AudioProcessor
            self.PhraseDatabase = PhraseDatabase
            self.EnhancedVocalTrackComposer = EnhancedVocalTrackComposer
            self.build_database_from_directory = build_database_from_directory
        except ImportError:
            print("Warning: Previous tool modules not available")
    
    def setup_library(self, sources: List[str] = None, max_items_per_source: int = 50):
        """
        Download and process video library
        
        Args:
            sources: List of source names to download from (None = all)
            max_items_per_source: Max items to download per source
        """
        print("\n" + "=" * 80)
        print("SETTING UP VIDEO LIBRARY")
        print("=" * 80)
        
        # Get sources
        all_sources = self.library_finder.list_sources()
        
        if sources:
            selected_sources = [
                s for s in all_sources 
                if s.name in sources
            ]
        else:
            # Use a curated subset for quick setup
            selected_sources = [
                s for s in all_sources
                if s.name in [
                    "American Rhetoric Movie Speeches",
                    "TED Talks",
                    "Prelinger Archives"
                ]
            ]
        
        print(f"\nSelected {len(selected_sources)} sources:")
        for source in selected_sources:
            print(f"  - {source.name}")
        
        # Download from each source
        all_downloaded = []
        
        for source in selected_sources:
            print(f"\n{'='*80}")
            print(f"Processing: {source.name}")
            print(f"{'='*80}")
            
            try:
                downloaded = self.downloader.download_from_source(
                    source,
                    max_items=max_items_per_source
                )
                all_downloaded.extend(downloaded)
            except Exception as e:
                print(f"Error downloading from {source.name}: {e}")
        
        print(f"\n{'='*80}")
        print(f"Total videos downloaded: {len(all_downloaded)}")
        print(f"{'='*80}")
        
        return all_downloaded
    
    def build_phrase_database(self, n_gram: int = 3, model_size: str = "base"):
        """Build phrase database from downloaded library"""
        print("\n" + "=" * 80)
        print("BUILDING PHRASE DATABASE")
        print("=" * 80)
        
        # Build database from library directory
        self.build_database_from_directory(
            directory=str(self.library_dir),
            db_path=self.database_path,
            n_gram=n_gram,
            model_size=model_size
        )
        
        print(f"\nDatabase saved to: {self.database_path}")
    
    def process_youtube_video(self, youtube_url: str, 
                             output_name: str = "output",
                             n_gram: int = 3,
                             enable_repitch: bool = True,
                             demucs_model: str = "htdemucs") -> str:
        """
        Complete pipeline: Download YouTube, replace vocals, combine with video
        
        Args:
            youtube_url: YouTube video URL
            output_name: Base name for output files
            n_gram: Phrase size for matching
            enable_repitch: Enable pitch correction
            demucs_model: Demucs model (htdemucs, htdemucs_ft, htdemucs_6s)
        
        Returns:
            Path to final output video
        """
        print("\n" + "=" * 80)
        print("PROCESSING YOUTUBE VIDEO")
        print("=" * 80)
        print(f"URL: {youtube_url}")
        
        # Step 1: Download audio and video
        print("\n[1/6] Downloading from YouTube...")
        audio_path = self.youtube_processor.download_audio(youtube_url)
        video_path = self.youtube_processor.download_video(youtube_url)
        
        # Step 2: Separate vocals with Demucs
        print("\n[2/6] Separating vocals with Demucs...")
        stems = self.youtube_processor.separate_vocals(audio_path, model=demucs_model)
        
        vocal_path = stems['vocals']
        
        # Step 3: Create instrumental track
        instrumental_path = str(self.work_dir / "youtube" / f"{output_name}_instrumental.wav")
        self.youtube_processor.create_instrumental(stems, instrumental_path)
        
        # Step 4: Generate alternate vocal using phrase database
        print("\n[3/6] Generating alternate vocal track...")
        
        processor = self.AudioProcessor(model_size="base")
        database = self.PhraseDatabase(db_path=self.database_path)
        database.load()
        
        composer = self.EnhancedVocalTrackComposer(
            database, processor,
            enable_tts=False,
            enable_repitch=enable_repitch
        )
        
        # Create composition
        composition = composer.compose_from_vocal_track(
            vocal_path,
            output_dir=str(self.work_dir / "composed"),
            n_gram=n_gram
        )
        
        # Save composition info
        composition_json = str(self.work_dir / f"{output_name}_composition.json")
        with open(composition_json, 'w') as f:
            json.dump({
                str(k): {**v, 'phrase': str(v.get('phrase', ''))}
                for k, v in composition.items()
            }, f, indent=2)
        
        # Create alternate vocal track (audio only, no video yet)
        alternate_vocal_path = str(self.work_dir / f"{output_name}_alternate_vocal.wav")
        self._create_alternate_vocal_audio(composition, alternate_vocal_path)
        
        # Step 5: Mix alternate vocal with instrumental
        print("\n[4/6] Mixing alternate vocal with instrumental...")
        mixed_audio_path = str(self.work_dir / f"{output_name}_mixed.wav")
        self._mix_audio_tracks(alternate_vocal_path, instrumental_path, mixed_audio_path)
        
        # Step 6: Combine with original video (or composed video clips)
        print("\n[5/6] Creating video with composed clips...")
        clips_video_path = str(self.work_dir / f"{output_name}_clips.mp4")
        composer.stitch_video(
            composition,
            output_path=clips_video_path,
            fallback_image=None,
            fallback_video=video_path  # Use original video as fallback
        )
        
        # Step 7: Create final output with mixed audio
        print("\n[6/6] Creating final output...")
        final_output = str(self.work_dir / f"{output_name}_final.mp4")
        self._combine_video_audio(clips_video_path, mixed_audio_path, final_output)
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Final output: {final_output}")
        print(f"Alternate vocal: {alternate_vocal_path}")
        print(f"Instrumental: {instrumental_path}")
        print(f"Mixed audio: {mixed_audio_path}")
        print(f"Composition info: {composition_json}")
        
        return final_output
    
    def _create_alternate_vocal_audio(self, composition: Dict, output_path: str):
        """Create audio-only alternate vocal track from composition"""
        temp_dir = Path("temp_audio_clips")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract audio segments
        concat_list = []
        sorted_times = sorted(composition.keys())
        
        for i, start_time in enumerate(sorted_times):
            clip_info = composition[start_time]
            
            source_clip = clip_info['source_clip']
            source_start = clip_info['source_start']
            source_end = clip_info['source_end']
            
            temp_audio = temp_dir / f"clip_{i:04d}.wav"
            
            # Extract audio segment
            cmd = [
                'ffmpeg', '-ss', str(source_start),
                '-i', source_clip,
                '-t', str(source_end - source_start),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2',
                '-y', str(temp_audio)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            concat_list.append(f"file '{temp_audio.absolute()}'")
        
        # Concatenate audio clips
        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_list))
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True)
    
    def _mix_audio_tracks(self, vocal_path: str, instrumental_path: str,
                         output_path: str, vocal_gain: float = 1.0,
                         instrumental_gain: float = 1.0):
        """Mix vocal and instrumental tracks"""
        cmd = [
            'ffmpeg',
            '-i', vocal_path,
            '-i', instrumental_path,
            '-filter_complex',
            f'[0:a]volume={vocal_gain}[a1];[1:a]volume={instrumental_gain}[a2];[a1][a2]amix=inputs=2:duration=longest',
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True)
    
    def _combine_video_audio(self, video_path: str, audio_path: str, output_path: str):
        """Combine video with new audio track"""
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True)


# ============================================================================
# Command-line Interface
# ============================================================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Video Library Downloader & Vocal Replacement System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List sources command
    list_parser = subparsers.add_parser('list-sources', help='List available video sources')
    
    # Setup library command
    setup_parser = subparsers.add_parser('setup-library', help='Download and setup video library')
    setup_parser.add_argument('--max-items', type=int, default=50,
                             help='Max items per source')
    setup_parser.add_argument('--sources', nargs='+',
                             help='Specific sources to download')
    
    # Build database command
    build_parser = subparsers.add_parser('build-database', help='Build phrase database')
    build_parser.add_argument('--n-gram', type=int, default=3,
                             help='N-gram size')
    build_parser.add_argument('--model-size', default='base',
                             choices=['tiny', 'base', 'small', 'medium', 'large'],
                             help='Whisper model size')
    
    # Process YouTube command
    youtube_parser = subparsers.add_parser('process-youtube',
                                          help='Process YouTube video')
    youtube_parser.add_argument('url', help='YouTube URL')
    youtube_parser.add_argument('--output', default='output',
                               help='Output name')
    youtube_parser.add_argument('--n-gram', type=int, default=3,
                               help='N-gram size')
    youtube_parser.add_argument('--no-repitch', action='store_true',
                               help='Disable pitch correction')
    youtube_parser.add_argument('--demucs-model', default='htdemucs',
                               choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s'],
                               help='Demucs model')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline',
                                       help='Run complete pipeline')
    full_parser.add_argument('url', help='YouTube URL')
    full_parser.add_argument('--max-items', type=int, default=30,
                            help='Max items per source')
    full_parser.add_argument('--output', default='output',
                            help='Output name')
    
    args = parser.parse_args()
    
    pipeline = VocalReplacementPipeline()
    
    if args.command == 'list-sources':
        pipeline.library_finder.print_sources()
    
    elif args.command == 'setup-library':
        pipeline.setup_library(
            sources=args.sources,
            max_items_per_source=args.max_items
        )
    
    elif args.command == 'build-database':
        pipeline.build_phrase_database(
            n_gram=args.n_gram,
            model_size=args.model_size
        )
    
    elif args.command == 'process-youtube':
        pipeline.process_youtube_video(
            youtube_url=args.url,
            output_name=args.output,
            n_gram=args.n_gram,
            enable_repitch=not args.no_repitch,
            demucs_model=args.demucs_model
        )
    
    elif args.command == 'full-pipeline':
        print("\n" + "=" * 80)
        print("RUNNING FULL PIPELINE")
        print("=" * 80)
        
        # Step 1: Setup library
        print("\nStep 1: Setting up video library...")
        pipeline.setup_library(max_items_per_source=args.max_items)
        
        # Step 2: Build database
        print("\nStep 2: Building phrase database...")
        pipeline.build_phrase_database()
        
        # Step 3: Process YouTube video
        print("\nStep 3: Processing YouTube video...")
        pipeline.process_youtube_video(
            youtube_url=args.url,
            output_name=args.output
        )
        
        print("\n" + "=" * 80)
        print("FULL PIPELINE COMPLETE!")
        print("=" * 80)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# ============================================================================
# Example Usage Scripts
# ============================================================================

def example_quick_start():
    """Quick start example"""
    print("""
    QUICK START GUIDE
    =================
    
    1. List available video sources:
       python video_library_tool.py list-sources
    
    2. Download video library (recommended sources):
       python video_library_tool.py setup-library --max-items 30
    
    3. Build phrase database:
       python video_library_tool.py build-database --n-gram 3 --model-size base
    
    4. Process a YouTube video:
       python video_library_tool.py process-youtube "https://youtube.com/watch?v=..." --output my_video
    
    5. Or run everything at once:
       python video_library_tool.py full-pipeline "https://youtube.com/watch?v=..." --max-items 30
    
    """)


def example_programmatic_usage():
    """Example of programmatic usage"""
    
    # Initialize pipeline
    pipeline = VocalReplacementPipeline()
    
    # Option 1: Use existing library
    # pipeline.library_finder.print_sources()
    
    # Option 2: Setup new library from specific sources
    pipeline.setup_library(
        sources=[
            "American Rhetoric Movie Speeches",
            "TED Talks"
        ],
        max_items_per_source=20
    )
    
    # Build database
    pipeline.build_phrase_database(n_gram=3, model_size="base")
    
    # Process YouTube video
    output = pipeline.process_youtube_video(
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        output_name="rick_roll_remix",
        n_gram=3,
        enable_repitch=True,
        demucs_model="htdemucs"
    )
    
    print(f"Final video created: {output}")


# ============================================================================
# Advanced Features
# ============================================================================

class AdvancedLibraryManager:
    """Advanced library management features"""
    
    def __init__(self, library_dir: str = "./video_library"):
        self.library_dir = Path(library_dir)
    
    def analyze_library_stats(self) -> Dict:
        """Analyze library statistics"""
        stats = {
            'total_videos': 0,
            'total_size_gb': 0,
            'by_source': {},
            'total_duration_hours': 0
        }
        
        for video_file in self.library_dir.rglob('*.mp4'):
            stats['total_videos'] += 1
            stats['total_size_gb'] += video_file.stat().st_size / 1e9
            
            # Get source from directory structure
            source = video_file.parent.name
            if source not in stats['by_source']:
                stats['by_source'][source] = {
                    'count': 0,
                    'size_gb': 0
                }
            
            stats['by_source'][source]['count'] += 1
            stats['by_source'][source]['size_gb'] += video_file.stat().st_size / 1e9
        
        return stats
    
    def print_library_stats(self):
        """Print library statistics"""
        stats = self.analyze_library_stats()
        
        print("\n" + "=" * 80)
        print("LIBRARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal Videos: {stats['total_videos']}")
        print(f"Total Size: {stats['total_size_gb']:.2f} GB")
        
        print("\nBy Source:")
        for source, source_stats in stats['by_source'].items():
            print(f"  {source}:")
            print(f"    Videos: {source_stats['count']}")
            print(f"    Size: {source_stats['size_gb']:.2f} GB")
    
    def cleanup_corrupted_files(self):
        """Remove corrupted or invalid video files"""
        print("\nScanning for corrupted files...")
        
        corrupted = []
        
        for video_file in self.library_dir.rglob('*.mp4'):
            # Try to get video info
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_file)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode != 0 or not result.stdout.strip():
                    corrupted.append(video_file)
            except Exception:
                corrupted.append(video_file)
        
        if corrupted:
            print(f"\nFound {len(corrupted)} corrupted files:")
            for f in corrupted:
                print(f"  {f}")
                f.unlink()  # Delete corrupted file
            print("Corrupted files removed.")
        else:
            print("No corrupted files found.")
    
    def filter_by_duration(self, min_duration: float = 60.0,
                          max_duration: float = 600.0):
        """Filter videos by duration"""
        print(f"\nFiltering videos (duration: {min_duration}-{max_duration}s)...")
        
        filtered_out = []
        
        for video_file in self.library_dir.rglob('*.mp4'):
            # Get duration
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_file)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                duration = float(result.stdout.strip())
                
                if duration < min_duration or duration > max_duration:
                    filtered_out.append(video_file)
                    # Move to filtered directory instead of deleting
                    filtered_dir = self.library_dir / "_filtered_out"
                    filtered_dir.mkdir(exist_ok=True)
                    video_file.rename(filtered_dir / video_file.name)
            
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        
        print(f"Filtered out {len(filtered_out)} videos")
    
    def extract_representative_frames(self, output_dir: str = "./preview_frames"):
        """Extract preview frames from each video"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\nExtracting preview frames...")
        
        for video_file in self.library_dir.rglob('*.mp4'):
            frame_output = output_path / f"{video_file.stem}.jpg"
            
            if frame_output.exists():
                continue
            
            # Extract frame at 10% of video duration
            cmd = [
                'ffmpeg',
                '-ss', '00:00:05',  # 5 seconds in
                '-i', str(video_file),
                '-frames:v', '1',
                '-q:v', '2',
                '-y', str(frame_output)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, timeout=10)
            except Exception as e:
                print(f"Error extracting frame from {video_file}: {e}")
        
        print(f"Preview frames saved to: {output_path}")


# ============================================================================
# Additional Download Sources
# ============================================================================

class AdditionalVideoSources:
    """Additional methods to find and download video sources"""
    
    @staticmethod
    def download_common_voice_dataset(output_dir: str = "./common_voice"):
        """
        Download Mozilla Common Voice dataset (audio only)
        Note: Requires manual download from commonvoice.mozilla.org
        """
        print("""
        Mozilla Common Voice Dataset
        =============================
        
        To use Common Voice:
        1. Visit: https://commonvoice.mozilla.org/datasets
        2. Select English dataset
        3. Download the validated clips
        4. Extract to: {output_dir}
        
        This dataset contains thousands of voice recordings with transcriptions.
        """)
    
    @staticmethod
    def download_librispeech(output_dir: str = "./librispeech"):
        """
        Download LibriSpeech dataset
        """
        print("\nDownloading LibriSpeech dataset...")
        
        # LibriSpeech is available via direct download
        base_url = "https://www.openslr.org/resources/12/"
        
        # Start with smaller dev-clean dataset
        datasets = [
            "dev-clean.tar.gz",  # ~350 MB
            # "train-clean-100.tar.gz",  # ~6 GB
        ]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for dataset in datasets:
            url = base_url + dataset
            output_file = output_path / dataset
            
            if output_file.exists():
                print(f"Already downloaded: {dataset}")
                continue
            
            print(f"Downloading {dataset}...")
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            # Extract
            print(f"Extracting {dataset}...")
            subprocess.run(['tar', '-xzf', str(output_file), '-C', str(output_path)])
        
        print(f"LibriSpeech downloaded to: {output_path}")
    
    @staticmethod
    def search_youtube_channels(query: str, max_results: int = 10) -> List[str]:
        """
        Search for YouTube channels with speech content
        
        Returns list of channel URLs
        """
        print(f"\nSearching YouTube for: {query}")
        print("Note: Use YouTube API or manual search for best results")
        
        # Recommended speech-heavy channels
        recommended = [
            "https://www.youtube.com/@TED",
            "https://www.youtube.com/@TEDx",
            "https://www.youtube.com/@Toastmasters",
            "https://www.youtube.com/@CrashCourse",
            "https://www.youtube.com/@CGPGrey",
            "https://www.youtube.com/@Kurzgesagt",
            "https://www.youtube.com/@3blue1brown",
        ]
        
        print("\nRecommended speech-heavy channels:")
        for channel in recommended:
            print(f"  {channel}")
        
        return recommended


# ============================================================================
# Batch Processing Utilities
# ============================================================================

class BatchProcessor:
    """Process multiple YouTube videos in batch"""
    
    def __init__(self, pipeline: VocalReplacementPipeline):
        self.pipeline = pipeline
    
    def process_youtube_playlist(self, playlist_url: str, 
                                 max_videos: int = 10,
                                 output_dir: str = "./batch_output"):
        """Process entire YouTube playlist"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get playlist video URLs using yt-dlp
        cmd = [
            'yt-dlp',
            '--flat-playlist',
            '--get-id',
            '--playlist-end', str(max_videos),
            playlist_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_ids = result.stdout.strip().split('\n')
        
        print(f"\nProcessing {len(video_ids)} videos from playlist...")
        
        results = []
        
        for i, video_id in enumerate(video_ids, 1):
            print(f"\n{'='*80}")
            print(f"Processing video {i}/{len(video_ids)}: {video_id}")
            print(f"{'='*80}")
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            try:
                output = self.pipeline.process_youtube_video(
                    youtube_url=video_url,
                    output_name=f"video_{i:03d}_{video_id}"
                )
                results.append({
                    'video_id': video_id,
                    'status': 'success',
                    'output': output
                })
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                results.append({
                    'video_id': video_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save results
        results_file = output_path / "batch_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing complete. Results saved to: {results_file}")
        
        return results
    
    def process_url_list(self, url_file: str, output_dir: str = "./batch_output"):
        """Process list of URLs from file"""
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"\nProcessing {len(urls)} URLs from file...")
        
        results = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*80}")
            print(f"Processing {i}/{len(urls)}: {url}")
            print(f"{'='*80}")
            
            try:
                video_id = self.pipeline.youtube_processor._extract_video_id(url)
                output = self.pipeline.process_youtube_video(
                    youtube_url=url,
                    output_name=f"video_{i:03d}_{video_id}"
                )
                results.append({
                    'url': url,
                    'status': 'success',
                    'output': output
                })
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results.append({
                    'url': url,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results


# ============================================================================
# Quality Control and Validation
# ============================================================================

class QualityControl:
    """Quality control for processed videos"""
    
    @staticmethod
    def validate_output_video(video_path: str) -> Dict:
        """Validate output video quality"""
        validation = {
            'exists': False,
            'has_video': False,
            'has_audio': False,
            'duration': 0,
            'resolution': None,
            'audio_sync': True
        }
        
        if not os.path.exists(video_path):
            return validation
        
        validation['exists'] = True
        
        # Check streams
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_streams',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        for stream in data.get('streams', []):
            if stream['codec_type'] == 'video':
                validation['has_video'] = True
                validation['resolution'] = f"{stream.get('width')}x{stream.get('height')}"
            elif stream['codec_type'] == 'audio':
                validation['has_audio'] = True
        
        # Get duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            validation['duration'] = float(result.stdout.strip())
        except:
            pass
        
        return validation
    
    @staticmethod
    def print_validation_report(video_path: str):
        """Print validation report"""
        validation = QualityControl.validate_output_video(video_path)
        
        print("\n" + "=" * 80)
        print("VIDEO VALIDATION REPORT")
        print("=" * 80)
        print(f"File: {video_path}")
        print(f"Exists: {'✓' if validation['exists'] else '✗'}")
        print(f"Has Video: {'✓' if validation['has_video'] else '✗'}")
        print(f"Has Audio: {'✓' if validation['has_audio'] else '✗'}")
        print(f"Duration: {validation['duration']:.2f}s")
        print(f"Resolution: {validation['resolution']}")
        print(f"Audio Sync: {'✓' if validation['audio_sync'] else '✗'}")
        print("=" * 80)


# Example usage documentation
USAGE_EXAMPLES = """
COMPREHENSIVE USAGE EXAMPLES
============================

1. QUICK START (Full Pipeline):
   -----------------------------
   python video_library_tool.py full-pipeline "https://youtube.com/watch?v=VIDEO_ID" --max-items 30

2. STEP-BY-STEP WORKFLOW:
   -----------------------
   # List available video sources
   python video_library_tool.py list-sources
   
   # Download video library
   python video_library_tool.py setup-library --max-items 50 --sources "TED Talks" "Prelinger Archives"
   
   # Build phrase database
   python video_library_tool.py build-database --n-gram 3 --model-size base
   
   # Process YouTube video
   python video_library_tool.py process-youtube "https://youtube.com/watch?v=VIDEO_ID" --output my_remix

3. PROGRAMMATIC USAGE:
   --------------------
   from video_library_tool import VocalReplacementPipeline
   
   pipeline = VocalReplacementPipeline()
   pipeline.setup_library(max_items_per_source=20)
   pipeline.build_phrase_database(n_gram=3)
   output = pipeline.process_youtube_video("https://youtube.com/watch?v=VIDEO_ID")

4. ADVANCED FEATURES:
   ------------------
   # Use 6-stem separation (includes guitar, piano)
   python video_library_tool.py process-youtube "URL" --demucs-model htdemucs_6s
   
   # Disable pitch correction
   python video_library_tool.py process-youtube "URL" --no-repitch
   
   # Larger phrases for more context
   python video_library_tool.py process-youtube "URL" --n-gram 5

5. LIBRARY MANAGEMENT:
   -------------------
   from video_library_tool import AdvancedLibraryManager
   
   manager = AdvancedLibraryManager()
   manager.print_library_stats()
   manager.cleanup_corrupted_files()
   manager.filter_by_duration(min_duration=60, max_duration=300)

6. BATCH PROCESSING:
   -----------------
   from video_library_tool import BatchProcessor, VocalReplacementPipeline
   
   pipeline = VocalReplacementPipeline()
   batch = BatchProcessor(pipeline)
   batch.process_youtube_playlist("PLAYLIST_URL", max_videos=10)

7. QUALITY CONTROL:
   ----------------
   from video_library_tool import QualityControl
   
   QualityControl.print_validation_report("output_final.mp4")
"""
