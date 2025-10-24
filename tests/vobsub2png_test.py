from PaddleOCRAnimation.video.sub.vobsub2png import vobsub2png
from pathlib import Path
from shutil import rmtree

def test_vobsub2png():
    sub_path = Path(__file__).parent.parent/'examples'/'data'/'subs'/'tiny.idx'
    output_path = Path(__file__).parent/'tmp'/'vobsub2png_results'
    if output_path.exists():
        rmtree(output_path)
    
    vobsub2png(
        idx_path=str(sub_path),
        outputdir=str(output_path)
    )
    index_path = output_path / 'index.json'
    last_image_path = output_path / '0000.png'

    assert index_path.exists()
    assert last_image_path.exists()
    rmtree(output_path)
    