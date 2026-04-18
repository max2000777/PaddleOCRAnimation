from shapely import Polygon
from numpy.typing import NDArray
from os import makedirs
from os.path import exists, join, dirname, basename, abspath
from ...dataset.detDataset import detDataset
from paddleocr._pipelines.ocr import PaddleOCR
import logging
import jiwer
from collections import Counter
from paddlex.inference.pipelines.ocr.result import OCRResult
from PIL import Image
from shutil import rmtree
from tqdm.auto import tqdm
from json import dump as jsonDump


logger = logging.getLogger(__name__)

def are_two_boxes_the_same(
        box1: list[list[int]] | tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | NDArray, 
        box2: list[list[int]] | tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | NDArray,
    ) -> float:
    """
    Compute the Intersection-over-Union (IoU) between two quadrilateral boxes.

    The boxes are interpreted as polygons defined by 4 corner points (x, y).
    The point order should describe a valid, non self-intersecting polygon
    (left, top, right, bottom).

    Args:
        box1: First box as 4 points of shape (4, 2), e.g. [[x1, y1], ..., [x4, y4]].
        box2: Second box as 4 points of shape (4, 2), e.g. [[x1, y1], ..., [x4, y4]].

    Raises:
        ValueError: If a box does not contain exactly 4 points, or if a point is not of length 2.

    Returns:
        The IoU value in [0, 1]. Returns 0.0 if the union area is zero.
    """
    for box in [box1, box2]:
        if not len(box) ==4 or not all([len(x) ==2 for x in box]):
            raise ValueError(f"boxes should be of len 4 and points of len 2, here {type(box)}, \n {box}")
    
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union ==0:
        return 0
    iou = inter / union

    return iou

def match_greedy_iou(
    b_comparaison: list[list[float]],
    iou_thresh: float,
) -> list[int | None]:
    """
    Greedily match ground-truth boxes to detections using IoU.

    Args:
        b_comparaison: IoU matrix of shape (n_truth, n_det).
        iou_thresh: Minimum IoU required to create a match.

    Returns:
        list: For each ground-truth index, the matched detection index, or None if unmatched.
    """
    n_truth = len(b_comparaison)
    n_det = len(b_comparaison[0]) if n_truth else 0

    pairs = []
    for t in range(n_truth):
        for d in range(n_det):
            iou = b_comparaison[t][d]
            if iou > iou_thresh:
                pairs.append((iou, t, d))

    pairs.sort(reverse=True, key=lambda x: x[0])

    truth_used = [False] * n_truth
    det_used = [False] * n_det
    assign: list[int | None] = [None] * n_truth

    for iou, t, d in pairs:
        if (not truth_used[t]) and (not det_used[d]):
            assign[t] = d
            truth_used[t] = True
            det_used[d] = True

    return assign


def detection_boxes_comparaison(
        truth_boxes: list[NDArray] | list[tuple] | list[list],
        detected_boxes: list[NDArray] | list[tuple] | list[list],
        iou_thresh: float = 0.6,
    ) -> tuple[list[list[float]], list[int | None]]:
    """
    Build the IoU matrix between ground-truth and detected boxes, then compute a 1-to-1
    greedy matching based on IoU.

    Each box is expected to be a quadrilateral defined by 4 (x, y) points.

    Args:
        truth_boxes: List of ground-truth boxes (n_truth), each of shape (4, 2).
        detected_boxes: List of detected boxes (n_det), each of shape (4, 2).
        iou_thresh: Minimum IoU required for a pair to be considered matchable.

    Returns:
        tuple: A tuple (iou_matrix, assignment) where:
          - iou_matrix is a list of lists with shape (n_truth, n_det) containing IoU values.
          - assignment is a list of length n_truth where assignment[i] is the matched detection
            index for truth box i, or None if no detection matched above threshold.
    """
    if not truth_boxes:
        return [], []
    if not detected_boxes:
        return [[] for _ in truth_boxes], [None] * len(truth_boxes)

    b_comparaison = []
    detected_box_for_each_truth = []
    for i, truth_box in enumerate(truth_boxes):
        b_comparaison.append([])
        for y, detected_box in enumerate(detected_boxes):
            iou = are_two_boxes_the_same(box1=truth_box, box2=detected_box)
            b_comparaison[i].append(
                iou
            )
    
    detected_box_for_each_truth = match_greedy_iou(
        b_comparaison=b_comparaison, iou_thresh=iou_thresh
    )

    return b_comparaison, detected_box_for_each_truth


def detection_eval_metrics_each_image(
    n_detected_boxes: int,
    detected_box_for_each_truth: list[int | None],
) -> dict[str, float]:
    """
    Compute per-image detection counts (TP/FP/FN) from a GT-to-detection assignment.

    Args:
        n_detected_boxes: Number of predicted boxes for the image.
        detected_box_for_each_truth: For each ground-truth box, the matched detection index,
            or None if unmatched.

    Returns:
        dict: A dict with keys "det_TP", "det_FP", "det_FN" as floats.
    """
    TP = sum(x is not None for x in detected_box_for_each_truth)
    FP = n_detected_boxes - TP
    FN = len(detected_box_for_each_truth) - TP

    return {
        "det_TP": float(TP),
        "det_FP": float(FP),
        "det_FN": float(FN),
    }

def calculate_image_det_metrics(
        ocr_result_entry: dict,
        dataset_entry_annotations: list[dict],
        aggregate_counts: dict[str, float],
        iou_thresh: float,
    )->tuple[dict[str, float], list[int | None]]:
    """
    Compute detection metrics for a single image and update global det_TP/det_FP/det_FN accumulators.

    Ground-truth boxes are extracted from `dataset_entry_annotations[*]["points"]` and
    matched to predicted boxes from `ocr_result_entry["dt_polys"]` using greedy IoU
    matching (thresholded by `iou_thresh`).

    Args:
        ocr_result_entry: OCR pipeline output for one image. Must contain key "dt_polys"
            (list of detected quadrilateral boxes).
        dataset_entry_annotations: List of ground-truth annotations for the image. Each
            annotation must contain key "points" (quadrilateral box).
        aggregate_counts: Running totals of {"det_TP", "det_FP", "det_FN"} updated in-place.
        iou_thresh: Minimum IoU required to match a detection to a ground-truth box.

    Returns:
        tuple: A tuple (aggregate_counts, detected_box_for_each_truth) where:
        * aggregate_counts is the updated {"det_TP","det_FP","det_FN"} accumulator.
        * detected_box_for_each_truth maps each ground-truth index to a matched detection
            index, or None if no match was found.
    """
    dataset_boxes = [b['points'] for b in dataset_entry_annotations]
    iou_matrix, detected_box_for_each_truth = detection_boxes_comparaison(
        truth_boxes=dataset_boxes,
        detected_boxes=ocr_result_entry['dt_polys'],
        iou_thresh=iou_thresh
    )
    image_metr = detection_eval_metrics_each_image(
        n_detected_boxes=len(ocr_result_entry['dt_polys']), detected_box_for_each_truth=detected_box_for_each_truth
    )
    aggregate_counts["det_TP"] += image_metr["det_TP"]
    aggregate_counts["det_FP"] += image_metr["det_FP"]
    aggregate_counts["det_FN"] += image_metr["det_FN"]

    return aggregate_counts, detected_box_for_each_truth

def _dot_runs_by_alnum_pos(s: str) -> dict[int, int]:
    """
    Map runs of '.' to their start position measured in non-dot characters.

    Args:
        s: Input string.

    Returns:
        dict: A dict {pos: run_len} where `pos` is the count of non-dot characters seen
        before the run starts, and `run_len` is the number of consecutive dots.
    """
    pos = 0
    runs: dict[int, int] = {}
    i = 0
    while i < len(s):
        if s[i] == ".":
            j = i
            while j < len(s) and s[j] == ".":
                j += 1
            runs[pos] = j - i
            i = j
        else:
            pos += 1
            i += 1
    return runs

def three_dots_metric(
        truth_text:str,
        rec_text: str
)-> Counter | None:
    """
    Evaluate how well the recognizer preserves occurrences of "..." (three consecutive dots).

    The metric is only computed when the two strings match under a "hard" normalization
    (case/whitespace/punctuation-insensitive), so that differences are assumed to come
    mainly from dot handling. Text is then softly normalized (lowercasing, collapsing
    spaces, converting '…' to '...', and removing non-alnum characters except dots).
    Expected and predicted "..." runs are located by their position measured in
    non-dot characters.

    Args:
        truth_text (str): Ground-truth transcription.
        rec_text (str): Recognized transcription.

    Returns:
        counter: A Counter with keys:
          - three_dots_TP / FP / FN: counts of correctly predicted / spurious / missed "..."
          - three_dots_wrong_len: expected "..." present but with a different dot-run length
          - three_dots_missing: expected "..." not present at all
          - three_dots_n_expected: number of expected "..." occurrences
        Returns None if the metric is not applicable (no "..." in GT, or strings differ
        beyond dot placement/length).
    """
    if '...' not in truth_text:
        return None
    hard_tr = jiwer.Compose([
        jiwer.RemoveWhiteSpace(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase()
    ])
    if hard_tr(truth_text) != hard_tr(rec_text):
        #hard to compute, the problem is not the three dots
        return None
    soft_tr = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({
            r"…": r"...",
        }),
        jiwer.SubstituteRegexes({
            r"[^0-9a-z\.]+": r"",
        }),
    ])

    t:str = soft_tr(truth_text)
    r:str = soft_tr(rec_text)

    if t.replace(".", "") != r.replace(".", ""):
        return None
    
    t_runs = _dot_runs_by_alnum_pos(t)
    r_runs = _dot_runs_by_alnum_pos(r)

    expected_pos = {p for p, ln in t_runs.items() if ln == 3}
    predicted_pos = {p for p, ln in r_runs.items() if ln == 3}

    TP = len(expected_pos & predicted_pos)
    FP = len(predicted_pos - expected_pos)

    wrong_len = sum(1 for p in expected_pos if (p in r_runs and r_runs[p] != 3))
    missing = sum(1 for p in expected_pos if (p not in r_runs))
    FN = wrong_len + missing

    return Counter({
        "three_dots_TP": TP,
        "three_dots_FP": FP,
        "three_dots_FN": FN,
        "three_dots_wrong_len": wrong_len,
        "three_dots_missing": missing,
        "three_dots_n_expected": len(expected_pos),
    })

def calculate_image_rec_metrics(
        detected_box_for_each_truth: list[int | None],
        dataset_entry_annotations: list[dict],
        ocr_result_entry: dict,
    )->tuple:
    """
    Compute recognition metrics for a single image using the GT-to-detection assignment.

    For each ground-truth annotation, the matched recognized text is taken from
    `ocr_result_entry["rec_texts"]`. Unmatched ground-truth boxes yield None entries.
    Also accumulates the custom "three dots" statistics over matched pairs.

    Args:
        detected_box_for_each_truth (list): For each ground-truth box, the matched detection index,
            or None if unmatched.
        dataset_entry_annotations (list): Ground-truth annotations for the image. Each must contain
            key "transcription".
        ocr_result_entry (dict): OCR pipeline output for one image. Must contain key "rec_texts"
            aligned with detected boxes.

    Returns:
        tuple: A tuple (correctly_rec_text, truth_text_list, rec_text_list, three_dots_metrics) where:
          - correctly_rec_text: list of bool or None (None if GT box unmatched).
          - truth_text_list: list of GT strings or None (None if unmatched).
          - rec_text_list: list of recognized strings or None (None if unmatched).
          - three_dots_metrics: Counter aggregating three_dots_* counts for this image.
    """
    correctly_rec_text = []
    truth_text_list = []
    rec_text_list = []
    three_dots_metrics = Counter({
        "three_dots_TP": 0,
        "three_dots_FP": 0,
        "three_dots_FN": 0,
        "three_dots_wrong_len": 0,
        "three_dots_missing": 0,
        "three_dots_n_expected": 0,
    })
    for i in range(len(detected_box_for_each_truth)):
        if detected_box_for_each_truth[i] is None:
            correctly_rec_text.append(None)
            truth_text_list.append(None)
            rec_text_list.append(None)
            continue
        truth_text = dataset_entry_annotations[i]['transcription']
        rec_text = ocr_result_entry['rec_texts'][detected_box_for_each_truth[i]]
        correctly_rec_text.append(
            truth_text == rec_text
        )
        truth_text_list.append(truth_text)
        rec_text_list.append(rec_text)
        m = three_dots_metric(truth_text, rec_text)
        if m is not None:
            three_dots_metrics.update(m)

    return correctly_rec_text, truth_text_list, rec_text_list, three_dots_metrics


def global_rec_eval_metrics(
        correctly_rec_text: list[bool | None],
        truth_text_list : list[str | None],
        rec_text_list: list[str | None],
        three_dots_metrics:dict
)-> dict[str, int | float]:
    """
    Compute global recognition metrics (CER/WER/accuracy) and aggregate "three dots" scores.

    The recognition metrics are computed on matched GT/predicted text pairs only (entries
    where the GT box had no match are filtered out). "three dots" precision/recall/F1 are
    derived from the provided `three_dots_metrics` counters.

    Args:
        correctly_rec_text (list): Per-GT correctness flags (True/False) or None if unmatched.
        truth_text_list (list): Per-GT reference strings or None if unmatched.
        rec_text_list (list): Per-GT hypothesis strings or None if unmatched.
        three_dots_metrics (dict): Dict containing at least the keys:
            {"three_dots_TP","three_dots_FP","three_dots_FN","three_dots_wrong_len",
             "three_dots_missing","three_dots_n_expected"}.

    Returns:
        dict: A dict containing CER, WER, recognition accuracy, and three-dots precision/recall/F1
        along with the underlying three-dots counts.
    """
    tr = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords()
    ])
    required = {"three_dots_TP", "three_dots_FP",
        "three_dots_FN", "three_dots_wrong_len",
        "three_dots_missing", "three_dots_n_expected"}
    
    if not isinstance(three_dots_metrics, dict) or not required.issubset(three_dots_metrics):
        raise ValueError("three_dots_TP, three_dots_FP and three_dots_FN should all be entries of the dict")
    TP, FP, FN = three_dots_metrics['three_dots_TP'], three_dots_metrics['three_dots_FP'], three_dots_metrics['three_dots_FN']
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    true_correctly_rec_text = [x for x in correctly_rec_text if x is not None]
    Acc_rec = sum(true_correctly_rec_text)/len(correctly_rec_text) if len(correctly_rec_text) !=0 else 0

    true_truth_text_list = [se for se in truth_text_list if se is not None]
    true_rec_text_list = [se for se in rec_text_list if se is not None]
    if len(true_truth_text_list)!=len(true_rec_text_list):
        logger.error("The length of the truth text list is not the same as the length of the rec text list")
        cer = 0
        wer = 0
    else:
        cer = jiwer.cer(reference=true_truth_text_list, hypothesis=true_rec_text_list)
        wer = jiwer.wer(reference=true_truth_text_list, hypothesis=true_rec_text_list, hypothesis_transform=tr,reference_transform=tr)
    
    return {
        "cer" : cer,
        "wer": wer,
        "Acc_rec": Acc_rec,
        "three_dots_Precision":precision,
        "three_dots_Recall":recall,
        "three_dots_f1_score":f1,
        "three_dots_TP": TP,
        "three_dots_FP": FP,
        "three_dots_FN": FN,
        "three_dots_wrong_len":  three_dots_metrics['three_dots_wrong_len'],
        "three_dots_missing": three_dots_metrics['three_dots_missing'],
        "three_dots_n_expected": three_dots_metrics['three_dots_n_expected'],
    }
def global_det_eval_metrics(
        aggregate_counts: dict[str, float],
)->dict[str, float | int]:
    """
    Compute global detection precision/recall/F1 from aggregated TP/FP/FN counts.

    Args:
        aggregate_counts (dict): Dict containing cumulative `det_TP`, `det_FP`, and `det_FN` counts.

    Returns:
        dict: A dict with keys "Precision", "Recall", "F1-score" (floats) and `det_TP`,`det_FP`,`det_FN` (ints).
    """
    required = {"det_TP", "det_FP", "det_FN"}
    if not isinstance(aggregate_counts, dict) or not required.issubset(aggregate_counts):
        raise ValueError("det_TP, det_FP and det_FN should all be entries of the dict")
    TP, FP, FN = aggregate_counts['det_TP'], aggregate_counts['det_FP'], aggregate_counts['det_FN']
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "det_TP": int(TP),
        "det_FP": int(FP),
        "det_FN": int(FN),
    }


def prepare_dataset(
        det_dataset_txt_path: str
    )-> detDataset:
    """
    Load and validate a detection dataset from a dataset list file.

    Args:
        det_dataset_txt_path (str): Path to the dataset text file.

    Returns:
        detDataset: A verified `detDataset` instance.
    """
    if not exists(det_dataset_txt_path):
        raise ValueError(f'The file {det_dataset_txt_path} does not exist')
    
    det_dataset = detDataset.make_dataset(det_dataset_txt_path)
    det_dataset.verify_dataset()

    return det_dataset

def prepare_ocr_pipeline(**kwargs)-> PaddleOCR:
    """
    Create a PaddleOCR pipeline from keyword arguments.
    Args:
        **kwargs: Passed directly to `PaddleOCR(**kwargs)`.
    Returns:
        An initialized `PaddleOCR` instance.
    """
    if 'text_recognition_model_dir' not in kwargs and 'text_detection_model_dir' not in kwargs:
        logger.warning(
            'the pipeline is created without a custom model, the original model is evaluated'+
            'to change this provide \'text_recognition_model_dir\' or \'text_detection_model_dir\''
            )
    ocr_pipeline = PaddleOCR(**kwargs)
    return ocr_pipeline

def save_wrong_image(
        save_path:str,
        ocr_result_image: OCRResult,
        dataset_image:Image.Image,
        detected_box_for_each_truth: list[int | None],
        image_correctly_rec_text: list[bool | None],
    ):
    if not isinstance(detected_box_for_each_truth, list) or not isinstance(ocr_result_image['dt_polys'], list):
        raise ValueError
    if (
        not None in detected_box_for_each_truth and 
        all(x is True for x in image_correctly_rec_text if x is not None) and
        len(ocr_result_image['dt_polys']) == len(detected_box_for_each_truth)
    ):
        # the image was successfuly OCRed, no need to save it
        return True
    ocr_final_image = ocr_result_image._to_img()['ocr_res_img']

    w1, h1 = dataset_image.size
    w2, h2 = ocr_final_image.size

    out = Image.new("RGBA", (w1 + w2, max(h1, h2)), (0, 0, 0, 0))
    out.paste(dataset_image, (0, 0))
    out.paste(ocr_final_image, (w1, 0))

    out.save(save_path)
    return False

    
def folder_prep(
        image_dir_path:str,
        remove_existing:bool = False,
        folder_name:str ="saving", 
) -> str:
    full_path = join(image_dir_path, folder_name)
    if remove_existing and exists(full_path):
        rmtree(full_path)
    makedirs(full_path, exist_ok=True)
    return full_path

def create_wrong_image_dict(
        dataset_image: dict,
        ocr_result_image: dict,
        correctly_rec_text: list[bool | None],
        detected_box_for_each_truth:list[int | None]
) -> dict[str, list|str]:
    wrong_image_dict = {}
    wrong_image_dict['image_name'] = basename(dataset_image['image_path'])
    wrong_image_dict['dataset_annotation'] = dataset_image['annotations']
    wrong_image_dict['dt_polys'] = [arr.astype(int).tolist() for arr in ocr_result_image['dt_polys']]
    wrong_image_dict['detected_box_for_each_truth'] = detected_box_for_each_truth
    wrong_image_dict['correctly_rec_text'] = correctly_rec_text
    wrong_image_dict['rec_texts'] = ocr_result_image['rec_texts']
    wrong_image_dict['rec_scores'] = ocr_result_image['rec_scores']

    return wrong_image_dict


def eval_paddleOCR(
        det_dataset_txt_path: str,
        image_dir_path: str,
        iou_thresh: float = 0.6,
        remove_existing:bool = False,
        user_text_rec_score_thresh: float = 0.75,
        **kwargs
):
    det_dataset = prepare_dataset(det_dataset_txt_path)

    ocr_pipeline =prepare_ocr_pipeline(**kwargs)
    full_saving_path = folder_prep(
        image_dir_path=image_dir_path,
        remove_existing=remove_existing
    )

    ocr_results = ocr_pipeline.predict(
        [join(dirname(det_dataset_txt_path), entry['image_path']) for entry in det_dataset.images]
    )

    if len(det_dataset) != len(ocr_results):
        raise ValueError(f"The length of the eval dataset ({len(det_dataset)}) is not equal to the length of the ocr result ({len(ocr_results)})")
    aggregate_det_metrics = {
        "det_TP": 0.0, "det_FP": 0.0, "det_FN": 0.0,
    }
    three_dots_metrics=Counter({"three_dots_TP": 0, "three_dots_FP": 0,
        "three_dots_FN": 0, "three_dots_wrong_len": 0,
        "three_dots_missing": 0, "three_dots_n_expected": 0,
    })
    correctly_rec_text, truth_text_list, rec_text_list, wrong_images_dict =[],[],[], []


    for i in tqdm(range(len(ocr_results)),desc="ocr"):
        aggregate_det_metrics, detected_box_for_each_truth = calculate_image_det_metrics(
            ocr_result_entry=ocr_results[i],
            dataset_entry_annotations=det_dataset[i]['annotations'],
            aggregate_counts=aggregate_det_metrics,
            iou_thresh=iou_thresh
        )
        image_correctly_rec_text, image_truth_text_list, image_rec_text_list, image_three_dots_metrics = calculate_image_rec_metrics(
            dataset_entry_annotations=det_dataset[i]['annotations'],
            detected_box_for_each_truth=detected_box_for_each_truth,
            ocr_result_entry=ocr_results[i]
        )
        three_dots_metrics.update(image_three_dots_metrics)
        correctly_rec_text.extend(image_correctly_rec_text)
        truth_text_list.extend(image_truth_text_list)
        rec_text_list.extend(image_rec_text_list)

        correct_or_not = save_wrong_image(
            save_path=join(full_saving_path, basename(det_dataset[i]['image_path'])),
            ocr_result_image=ocr_results[i],
            dataset_image=det_dataset.renderImageWithBox(i),
            detected_box_for_each_truth=detected_box_for_each_truth,
            image_correctly_rec_text=image_correctly_rec_text
        )

        if not correct_or_not:
            wrong_images_dict.append(create_wrong_image_dict(
                dataset_image=det_dataset[i],
                ocr_result_image=ocr_results[i],
                correctly_rec_text=image_correctly_rec_text,
                detected_box_for_each_truth=detected_box_for_each_truth
            ))
    
    final_eval_det_metrics = global_det_eval_metrics(aggregate_det_metrics)
    final_eval_rec_metrics = global_rec_eval_metrics(
        correctly_rec_text,
        truth_text_list,
        rec_text_list,
        three_dots_metrics
    )
    return_dict = final_eval_det_metrics | final_eval_rec_metrics
    return_dict['n_images'] = len(det_dataset)

    with open(join(full_saving_path,"data.json"), "w", encoding="utf-8") as f:
        jsonDump(wrong_images_dict, f, ensure_ascii=False, indent=1)
    
    print(f'Images saved in {abspath(full_saving_path)}')

    return return_dict

