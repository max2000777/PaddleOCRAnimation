from PaddleOCRAnimation.video.sub.vobsub2png import vobsub2png, vobsubpng_to_eventWithPilList
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

def test_vobsubpng_to_eventWithPilList():
    idx_path = Path(__file__).parent.parent/'examples'/'data'/'subs'/'tiny.idx'
    str_path= Path(__file__).parent.parent/'examples'/'data'/'subs'/'tiny.srt'
    output_path = Path(__file__).parent/'tmp'/'vobsub2png_results'
    if output_path.exists():
        rmtree(output_path)
    
    vobsub2png(
        idx_path=str(idx_path),
        outputdir=str(output_path)
    )

    b = vobsubpng_to_eventWithPilList(
        path_to_sub=str_path,
        path_to_vobsubpng_folder=output_path
    )
    assert len(b)==1
    assert b[0].events[0].Event.text == ','
