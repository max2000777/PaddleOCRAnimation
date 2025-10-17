from pathlib import Path
from datetime import timedelta
import PIL.Image

sub = Path(__file__).parent.parent / "examples"/"data"/"subs"/"subs.ass"
def sub_file_exists():
    assert sub.exists(), 'The sub file needs to exists'
# TODO : faire tests box

from PaddleOCRAnimation.video.sub.DocumentPlus import DocumentPlus
def test_sort_event():
    doc = DocumentPlus.parse_file_plus(sub, sort=False)
    assert doc.events[0].text =="{\\i1}Ici Shinobu Oshino !", "Without sorting, the first event should be this one"
    doc = doc.sort_events()

    assert doc.events[0].text !="{\\i1}Ici Shinobu Oshino !", "With sorting, the first event should not be this one"


def test_document_nb_event_dans_frame():
    doc = DocumentPlus.parse_file_plus(sub)
    assert doc.nb_event_dans_frame(frame=timedelta(minutes=00, seconds=4)) == 3, 'there should be 3 events at that timing'

def test_doc_event_precis():
    doc = DocumentPlus.parse_file_plus(sub)
    doc = doc.doc_event_precis(frame=timedelta(seconds=20), event_id=1)

    assert doc.nb_event_dans_frame(frame=timedelta(seconds=20)) ==1, "only one event should be in the copy"
    assert doc.nb_event_dans_frame(frame=timedelta(seconds=4)) == 0, "there should not be any event there after nb_event_dans_frame"

def test_event_to_pil():
    doc = DocumentPlus.parse_file_plus(sub)
    results = doc.event_to_pil(
        frame=timedelta(minutes=00, seconds=4),
        event_id=2,
        size=(1920, 1080)
    )

    assert len(results) == 4, "There should be 4 bitmap at that timing (multiline are separated)"

    assert isinstance(results.to_pil((1920, 1080)), PIL.Image.Image), "The image sequence should be able to be turned into a PIL image"
