from os.path import exists, dirname, join
import json
from paddleocr import PaddleOCR  


def paddle_api(
        input: str | list,
        text_recognition_model_dir: str = '../../PaddleOCR_output/french_PP-OCRv5_mobile_rec/latest/inference',
        text_detection_model_dir: str = '../../PaddleOCR_output/PP-OCRv5_server_det_anime/latest/inference'
) -> dict:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_model_dir=text_recognition_model_dir,
        text_recognition_model_name='latin_PP-OCRv5_mobile_rec',
        text_detection_model_dir=text_detection_model_dir,
        text_detection_model_name='PP-OCRv5_server_det',
        lang='fr',
        text_rec_score_thresh=0.8,
    )
    results = ocr.predict(input)
    return results



def ocr_vobsub(
    json_sub: str,
    root_path: str | None = None
):
    if not exists(json_sub):
        raise FileNotFoundError(f'The file {json_sub} does not exists')
    
    root_path = dirname(json_sub) if not root_path else root_path
    
    with open(json_sub, 'r') as file:
        sub = json.load(file)
    
    if 'subtitles' not in sub:
        raise ValueError(f'The key subtitles is not in the json file')
    
    sub_list: list[dict] = sub['subtitles']

    path_list: list[str] = []

    for s in sub_list:
        image_path = s.get('path')
        if image_path is None: 
            continue
        path_list.append(join(root_path, image_path))
    
    results = paddle_api(sub_list)

    return results


