from os.path import exists, dirname, join
import json
from paddleocr import PaddleOCR  


def paddle_api(
        input: str | list,
        text_recognition_model_dir: str = '/home/maxim/code/SubProject/PaddleOCR_output/french_PP-OCRv5_mobile_rec/latest/inference',
        text_detection_model_dir: str = '/home/maxim/code/SubProject/PaddleOCR_output/PP-OCRv5_server_det_anime/latest/inference'
) -> list[dict]:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_model_dir=text_recognition_model_dir,
        text_recognition_model_name='latin_PP-OCRv5_mobile_rec',
        text_detection_model_dir=text_detection_model_dir,
        text_detection_model_name='PP-OCRv5_server_det',
        text_det_box_thresh=0.7,
        lang='fr',
        text_rec_score_thresh=0.8,
    )
    results = ocr.predict(input)
    return results



def ocr_vobsub(
    json_sub: str,
    srt_save_path: str,
    root_path: str | None = None
):
    def to_srt_time(seconds: float) -> str:
        total_ms = int(round(seconds * 1000))  # arrondi au ms
        ms = total_ms % 1000
        total_s = total_ms // 1000

        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60

        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


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
    
    results = paddle_api(path_list)

    for i, s in enumerate(results):
        sub_list[i]['rec_text'] = '\n'.join(s.get('rec_texts', []))

    with open(srt_save_path, mode='w', encoding='utf-8') as f:
        for i, line in enumerate(sub_list):
            f.write(f'{i+1}\n')
            f.write(f'{to_srt_time(line.get('start'))} --> {to_srt_time(line.get('end'))}\n')
            f.write(f'{line.get('rec_text')}\n\n')

    return sub_list


