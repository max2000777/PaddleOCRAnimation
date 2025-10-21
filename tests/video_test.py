from pathlib import Path
from shutil import rmtree
from PIL import Image

vid_path = Path(__file__).parent.parent /'examples'/'data'/'testVidExtrait.mkv'
temp_folder_path = Path(__file__).parent /'tmp'

from PaddleOCRAnimation.video.Video import Video
from PaddleOCRAnimation.video.sub.RendererClean import Context, Renderer

def initialise_video() -> tuple[Video, Context, Renderer]:
    if temp_folder_path.exists():
        rmtree(temp_folder_path)

    vid = Video.make_video(vid_path)
    attachement_path= Path(__file__).parent/'tmp'/'attachement'
    vid.dumpAtachement(dossier=str(attachement_path))
    vid.extractSub(0, str(Path(__file__).parent/'tmp'/'subtest'))
    SIZE = (1920, 1080)
    ctx = Context()
    ctx.fonts_dir = str(attachement_path).encode('utf-8')
    r = ctx.make_renderer()
    r.set_fonts(fontconfig_config="\0")
    r.set_all_sizes(SIZE)
    return vid, ctx, r

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

def test_initialise_function():
    vid, ctx, r = initialise_video()
    assert isinstance(vid, Video) and isinstance(ctx, Context) and isinstance(r, Renderer)

def test_get_subtitle_boxes():
    vid, ctx, r = initialise_video()
    event_list = vid.get_subtitle_boxes(
        timestamp=26,
        SIZE=(0, 0),
        renderer=r,
        context=ctx,
        piste=0,
        multiline=False
    )

    assert str(event_list[3].events[1].Boxes) == '[[0, 65], [872, 65], [872, 135], [0, 135]]'
    assert isinstance(event_list[2].image, Image.Image)

def test_eventWithPilList_topil():
    vid, ctx, r = initialise_video()
    event_list = vid.get_subtitle_boxes(
        timestamp=26,
        SIZE=(1920, 1080),
        renderer=r,
        context=ctx,
        piste=0,
        multiline=False
    )

    assert isinstance(event_list.to_pil(), Image.Image)
    assert event_list.to_pil().size == (1920, 1080)

