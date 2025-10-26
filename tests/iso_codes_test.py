from PaddleOCRAnimation.video.iso_codes import iso_639_dict

def test_iso_639_dict():
    assert iso_639_dict['fr'] == 'fre'