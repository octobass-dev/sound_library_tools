The tool creates a complete workflow from finding free video sources to producing a final video with replaced vocals, all while maintaining synchronization and video quality. In addition, it creates a sound library with musical clips.

# Core Features:
1. Video Library Discovery & Download
- Curated Sources: 10+ pre-configured speech-heavy video sources including:
- C-SPAN (political speeches)
- TED Talks (presentations)
- Public domain movies (dialogues)
- Prelinger Archives (advertisements, educational content)
- TV news broadcasts
- Political campaign ads
- NASA technical videos
- Download Methods:
- Internet Archive collections via ia CLI
- YouTube playlists via yt-dlp
- Automatic filtering by duration and quality

2. YouTube Processing Pipeline
- Downloads audio using yt-dlp
- Downloads video for later use
- Separates vocals using Demucs (3 models available):
- htdemucs: Standard 4-stem (vocals, drums, bass, other)
- htdemucs_ft: Fine-tuned version
- htdemucs_6s: Extended 6-stem (adds guitar, piano)
- Creates instrumental track from non-vocal stems

3. Vocal Replacement
- Uses phrase database to find matching clips
- Generates alternate vocal track from video library phrases
- Supports pitch correction and time stretching
- Aligns phrases temporally with original timing

4. Video Composition
- Stitches together video clips matching the phrases
- Uses original video as fallback when clips don't have video
- Combines new vocals with instrumental track
- Creates final mixed output
- Advanced Features:
- Library Management
- Statistics analysis (count, size, duration by source)
- Corrupted file detection and cleanup
- Duration-based filtering
- Preview frame extraction
- Batch Processing
- Process entire YouTube playlists
- Process URLs from text file
- Automatic error handling and logging
- Quality Control
- Video validation (streams, duration, resolution)
- Audio sync verification
- Detailed validation reports
- Usage Examples:

# Quick start - full pipeline
```
python video_library_tool.py full-pipeline "https://youtube.com/watch?v=dQw4w9WgXcQ" --max-items 30
```

# Step-by-step
```
python video_library_tool.py list-sources
python video_library_tool.py setup-library --max-items 50
python video_library_tool.py build-database --n-gram 3
python video_library_tool.py process-youtube "YOUTUBE_URL" --output remix
```

# Advanced options
```
python video_library_tool.py process-youtube "URL" --demucs-model htdemucs_6s --n-gram 5 --no-repitch
```
Programmatic Usage:
```python
from video_library_tool import VocalReplacementPipeline

pipeline = VocalReplacementPipeline()
```

# Setup library with specific sources
```python
pipeline.setup_library(
    sources=["TED Talks", "American Rhetoric Movie Speeches"],
    max_items_per_source=30
)
```

# Build database
```python
pipeline.build_phrase_database(n_gram=3, model_size="base")
```

# Process YouTube video
```python
output = pipeline.process_youtube_video(
    youtube_url="https://youtube.com/watch?v=VIDEO_ID",
    output_name="my_remix",
    enable_repitch=True,
    demucs_model="htdemucs"
)
```

# Musical Sounds Library Tool
The tool automatically organizes sounds by category, tags them intelligently, and creates searchable metadata perfect for experimental music production!
1. Sound Detection & Segmentation
- Detects sound segments based on energy analysis
- Filters by duration (0.05-10 seconds)
- Merges segments separated by brief gaps
- Detects onset times for transient events
- Checks for surrounding silence (preferred for clean samples)

2. Audio Feature Extraction
- Spectral: Centroid, rolloff, flatness, zero-crossing rate
- Temporal: Attack/decay times, onset strength, envelope
- Pitch: Stability, frequency, musical note
- Quality: Loudness (LUFS), dynamic range, SNR, noise floor
- Timbre: MFCC, chroma features
- Statistics: Kurtosis, skewness

3. Sound Classification
- Six main categories with subcategories:
- TRANSIENT (short, fast attack):
- Impact, click, pop, snap, crack, glass break, surprised gasp
- SUSTAINED (medium-long duration):
- Laughter, scream, cry, vehicle engine, machine hum, alarm
- DRONE (long, stable):
- Wind, rumble, hum, buzz, tunnel echo, room tone
- PERCUSSIVE (rhythmic potential):
- Drum hit, wood knock, metal clang, hand clap, stomp
- TONAL (clear pitch):
- Whistle, bell, chime, singing, musical instrument
- NOISE (textural):
- White noise, static, crackle, rain, crowd

4. Quality Filtering
- Minimum loudness threshold (-40 LUFS)
- Minimum SNR (10 dB)
- Pitch consistency check
- Surrounding silence detection
- Perceptual uniqueness scoring (0-1)

5. Uniqueness Scoring
- Factors:
- Spectral characteristics (40%)
- Pitch clarity (15%)
- Onset strength (20%)
- Dynamic range (15%)
- Signal-to-noise ratio (15%)

6. Deduplication
- Perceptual hashing using MFCC
- Prevents duplicate samples in library
- Advanced Features:
- Library Browser
- Search by category, subcategory, tags
- Filter by duration, pitch, uniqueness
- Statistics generation
- Sample pack export

## Integration with Previous Tools
```python
from sound_library import IntegratedSoundExtractor
extractor = IntegratedSoundExtractor()

# Build from existing video library
samples = extractor.build_from_video_library("./video_library")

# Download videos and extract sounds
samples = extractor.download_and_extract("TED Talks", max_videos=20)
```

Usage Examples:
```python
# Build library from video directory
python sound_library.py build ./video_library --min-uniqueness 0.4

# Browse library
python sound_library.py browse --category transient --tags sharp bright

# View statistics
python sound_library.py stats

# Export high-quality transients
python sound_library.py export ./transient_pack --category transient --min-uniqueness 0.7

# Process single file
python sound_library.py process movie.mp4
Programmatic Usage:
from sound_library import SoundLibraryBuilder, SoundLibraryBrowser

# Build library
builder = SoundLibraryBuilder(output_dir="./my_sounds")
builder.min_uniqueness_score = 0.5
samples = builder.build_library_from_directory("./videos")

# Search library
browser = SoundLibraryBrowser(library_dir="./my_sounds")
browser.print_statistics()

# Find specific sounds
impacts = browser.search(
    category="transient",
    subcategory="impact",
    min_uniqueness=0.6
)

# Export curated pack
browser.export_pack(
    "./impact_pack",
    category="transient",
    min_uniqueness=0.7,
    max_samples=50
)
```
## Features
- Sound Processing
- Loudness normalization
- Silence trimming
- Fade in/out
- Batch processing


Output Structure:
sound_library/
├── transient/
│   ├── video1_0001_transient_impact.wav
│   └── video1_0023_transient_click.wav
├── sustained/
│   └── video2_0005_sustained_laughter.wav
├── drone/
│   └── video3_0012_drone_rumble.wav
├── library_metadata.json
└── ...
