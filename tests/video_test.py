from pathlib import Path
from shutil import rmtree

vid_path = Path(__file__).parent.parent /'examples'/'data'/'testVidExtrait.mkv'
temp_folder_path = Path(__file__).parent /'tmp'

from PaddleOCRAnimation.video.Video import Video

def test_make_video():
    vid = Video.make_video(vid_path)
    print(str(Path(__file__).parent))
    assert vid.duree==43.057, "the test vid is about 43 sec long"

def test_extractSub():
    if temp_folder_path.exists():
        rmtree(temp_folder_path)

    vid = Video.make_video(vid_path)
    vid.extractSub(0, str(Path(__file__).parent/'tmp'/'subtest'))
    subpath=Path(__file__).parent/'tmp'/'subtest.ass'
    assert vid.extracted_sub_path[0]== str(subpath), "The subpath should be registered"
    assert subpath.exists(), "the subfile should be extracted"


def test_dump_attachement():
    if temp_folder_path.exists():
        rmtree(temp_folder_path)

    vid = Video.make_video(vid_path)
    attachement_path= Path(__file__).parent/'tmp'/'attachement'
    vid.dumpAtachement(dossier=str(attachement_path))
    assert vid.attachement_path == str(attachement_path), 'the attachement path should be registered'

    font_path = attachement_path/'MyriadPro-Semibold.ttf'

    assert font_path.exists(), 'The font should be extracted'