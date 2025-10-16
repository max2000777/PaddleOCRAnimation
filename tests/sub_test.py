from pathlib import Path
from datetime import timedelta

sub = Path(__file__).parent.parent / "examples"/"data"/"subs"/"subs.ass"
# TODO : faire tests box

from PaddleOCRAnimation.video.sub.DocumentPlus import DocumentPlus

def test_documentplus():
    doc = DocumentPlus.parse_file_plus(sub)
    assert doc.nb_event_dans_frame(frame=timedelta(minutes=00, seconds=4)) == 3, 'there should be 3 events at that timing'

