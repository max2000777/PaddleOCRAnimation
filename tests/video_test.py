from pathlib import Path

vid = Path(__file__).parent.parent /'examples'/'data'/'testVidExtrait.mkv'

def video_file_exists():
    assert vid.exists(), 'the video file needs to exists'