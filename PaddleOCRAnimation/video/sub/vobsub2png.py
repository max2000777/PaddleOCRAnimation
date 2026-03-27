import subprocess
from os.path import exists, dirname, abspath, join, relpath
from os import makedirs
import importlib.resources
from PIL import Image, ImageDraw
from pathlib import Path
import logging
import json
from platform import system
from .DocumentPlus import DocumentPlus, split_dialogue
from .RendererClean import Box
from ..Video import eventWithPil, FrameToBoxEvent, eventWithPilList, padded_box_from_xyxy
from datetime import datetime
from ..classes import dataset_image
from typing import Literal
from ..utilis import detect_text_line_xyxy, adjust_box_to_baseline
from fractions import Fraction
import xml.etree.ElementTree as ET
import random



logger = logging.getLogger(__name__)




def vobsub2png(idx_path: str, outputdir: str | None = None):
    """
    Convertit un fichier de sous-titres VobSub (`.idx`/`.sub`) en une série d'images `PNG` à l'aide
    d'un binaire externe. Créer aussi un fichier json avec les timing et la position des images.

    Args:
        idx_path (str): Chemin vers le fichier `.idx` à convertir.
        outputdir (str | None, optional): Dossier de sortie pour les fichiers `PNG`.
            Si non spécifié, les images seront générées dans le dossier courant, dans un dossier
            avec le nom du fichier `.idx`.

    Raises:
        RuntimeError: Si le système d'exploitation n'est pas Windows ou Linux.
        FileNotFoundError: Si le fichier `.idx` (ou `.sub`) n'existe pas.
        TypeError: Si le fichier fourni n'est pas un fichier `.idx` valide.
    """
    plateforme = system()
    base_dir = dirname(dirname(abspath(__file__)))
    if plateforme == 'Windows':
        binary_path = str(importlib.resources.files("PaddleOCRAnimation.libs.Windows")/ "vobsub2png")
    elif plateforme == 'Linux':
        binary_path = str(importlib.resources.files("PaddleOCRAnimation.libs.linux")/ "vobsub2png")
    else:
        raise RuntimeError(
            f"La plateforme {plateforme} n'est pas supportée"
        )

    if not exists(idx_path):
        raise FileNotFoundError(
            f"Le fichier {idx_path} n'existe pas"
        )
    elif not idx_path.endswith('.idx'):
        raise TypeError(
            f"Le fichier {idx_path} n'est pas un fichier .idx"
        )
    elif not exists(idx_path[:-4] + '.sub'):
        raise FileNotFoundError(
            f"Le fichier {idx_path[:-4] + '.sub'} (associé au .idx) est manquant"
        )

    if outputdir is not None:
        command = [
            binary_path,
            '-o', outputdir,
            idx_path
        ]
    else:
        command = [
            binary_path,
            idx_path
        ]

    try:
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        raise ChildProcessError(e.stderr)

def xml_index_to_json_index(path_to_folder: Path | str, rounding: int = 3) -> dict | None:
    """
    Convert a BDSup2Sub BDN XML index (stored in a folder with PNG files) into a unified subtitle index
    compatible with the JSON-like structure produced by vobsub2png (subtitleRS).

    The function expects the input folder to contain exactly one `.xml` file (the index).
    It parses the framerate, converts event timecodes (InTC/OutTC) to seconds, and extracts
    each event's bitmap geometry from the `<Graphic>` node.

    Args:
        path_to_folder: Path to the folder containing the BDSup2Sub XML index.
        rounding: Number of decimals used to round start/end times (in seconds).

    Returns:
        A dict with the following structure:
            {"subtitles": [
                {
                    "start": float,     # seconds
                    "end": float,       # seconds
                    "force": bool,      # True if Forced="True"
                    "position": [x, y], # top-left position in pixels
                    "size": [w, h],     # [width, height] in pixels
                    "path": str,        # referenced graphic filename/path from XML
                },
                ...
            ]}

    Raises:
        ValueError: If the input path is not a directory, if the framerate cannot be read,
            or if the `<Events>` section is missing.
    """
    def tc_to_seconds(tc: str, fps: Fraction) -> float:
        hh, mm, ss, ff = tc.split(":")
        hh, mm, ss, ff = int(hh), int(mm), int(ss), int(ff)

        total = Fraction(hh * 3600 + mm * 60 + ss, 1) + Fraction(ff, 1) / fps
        return float(total)

    def parse_fps(root: ET.Element) -> Fraction:
        fmt = root.find("./Description/Format")
        if fmt is None or fmt.get("FrameRate") is None:
            raise ValueError("Impossible de trouver Description/Format/@FrameRate dans le XML.")
        fps_str = fmt.get("FrameRate").strip()

        if fps_str == "23.976":
            return Fraction(24000, 1001)

        return Fraction(fps_str)
    if isinstance(path_to_folder, str):
        path_to_folder = Path(path_to_folder)
    if not path_to_folder.is_dir():
        raise ValueError('path_to_foler is not a directory')
    
    xml_file = [p for p in path_to_folder.iterdir() if p.is_file() and p.suffix.lower() == ".xml"]
    if len(xml_file) != 1:
        # There should be only one xml file (the index)
        return None
    
    xml_file = xml_file[0]
    tree = ET.parse(xml_file)
    root = tree.getroot()

    fps = parse_fps(root)
    events_node = root.find("./Events")
    if events_node is None:
        raise ValueError("Impossible de trouver la section /BDN/Events dans le XML.")

    subtitles = []
    for ev in events_node.findall("./Event"):
        graphic = ev.find("./Graphic")
        if graphic is None:
            # Event sans Graphic : on ignore ou on lève une erreur selon ton besoin
            continue

        start_s = tc_to_seconds(ev.get("InTC", "00:00:00:00"), fps)
        end_s   = tc_to_seconds(ev.get("OutTC", "00:00:00:00"), fps)

        w = int(graphic.get("Width", "0"))
        h = int(graphic.get("Height", "0"))
        x = int(graphic.get("X", "0"))
        y = int(graphic.get("Y", "0"))

        forced = (ev.get("Forced", "False").strip().lower() == "true")

        subtitles.append({
            "start": round(start_s, rounding),
            "end": round(end_s, rounding),
            "force": forced,
            "position": [x, y],
            "size": [w, h],
            "path": graphic.text,
        })

    return {"subtitles": subtitles}

def simulate_CreateDoubleBorder(
        img: Image.Image, borderSize: int = 10, p_change_color:float = 0.3
    ) -> Image.Image:
    """Try to replicate the default prepocess : https://github.com/SubtitleEdit/subtitleedit/blob/669eb2ca0ebc1950707fff3dcac600f97d7b5602/src/UI/Features/Ocr/Engines/PaddleOcr.cs#L388
    """
    img = img.convert("RGBA")

    w, h = img.size
    totalBorder = borderSize * 2
    finalWidth = w + totalBorder * 2
    finalHeight = h + totalBorder * 2

    result = Image.new("RGBA", (finalWidth, finalHeight), (0, 0, 0, 0))

    draw = ImageDraw.Draw(result)

    left = borderSize
    top = borderSize
    right = finalWidth - borderSize
    bottom = finalHeight - borderSize

    rect_color = (0, 0, 0, 255)

    if random.random() < p_change_color:
        rect_color = rect_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            255,
        )

    draw.rectangle([left, top, right, bottom], fill=rect_color)

    result.paste(img, (totalBorder, totalBorder), img)

    return result




def vobsubpng_to_eventWithPilList(
        path_to_vobsubpng_folder: str | Path,
        path_to_sub: str | Path, 
        multiline: bool = True,
        padding: tuple[int,int,int,int] = (7,2,3,1),
        p_CreateDoubleBorder: float = 0.4,
)->eventWithPilList:
    """
    Build an image/text dataset by pairing VobSub PNG subtitles with timed subtitle events
    and per-line bounding boxes.

    The function expects a folder produced by vobsub2png (or similar) containing PNG images
    and an `index.json`. If `index.json` is missing, it attempts to convert a single BDSup2Sub
    BDN XML index found in the folder.

    For each subtitle entry in the index, the corresponding PNG is loaded and text line boxes
    are detected from transparency (alpha). Subtitle events are parsed from `path_to_sub`
    (`.ass` / `.srt`) and matched to images by comparing start times (rounded to 2 decimals)
    using a forward scan.

    Args:
        path_to_vobsubpng_folder: Folder containing subtitle PNGs and either `index.json`
            or a single BDN `.xml` index.
        path_to_sub: Subtitle text file to parse (typically `.ass` or `.srt`).
        multiline: If True, treat each PNG as a single subtitle block (one event matched
            to one detected box set). If False, split the matched event into multiple
            line events and align them with detected boxes.
        padding: (left, top, right, bottom) padding in pixels applied to each detected
            bounding box (clamped to image bounds).

    Returns:
        eventWithPilList: A list-like container of `eventWithPil` objects. Each element
        contains the loaded `PIL.Image` and a list of `FrameToBoxEvent` items linking:
            - an `Event` (parsed from the subtitle file),
            - and a `Box` (geometry around a detected text line).

    Raises:
        FileNotFoundError: If the folder / subtitle file is missing, or if neither
            `index.json` nor a usable XML index is found.
        ValueError: If subtitle parsing yields no events, if the index has no "subtitles"
            key, or if the number of detected boxes does not match the number of text lines.
        IndexError: If a PNG subtitle entry cannot be aligned to any subtitle event by
            start time.

    Notes:
        - Alignment is based on start timestamps only; end times are not used.
        - If multiple subtitle events share the same start time, alignment can be ambiguous.
        - PNGs whose paths are missing from the index or missing on disk are skipped (warning).
    """
    def dynamic_padding(
            event_text: str, padding:tuple[int, int, int, int],
            three_dots_padding: int = 0, point_padding: int = 0
        ):
        if three_dots_padding==0 and point_padding ==0:
            return padding
        l, t, r, b = padding
        if event_text.endswith('...') or event_text.endswith('…'):
            r += three_dots_padding
        elif event_text.endswith('.'):
            r+= point_padding
        
        return (l, t, r, b)

    if isinstance(path_to_vobsubpng_folder, str):
        path_to_vobsubpng_folder = Path(path_to_vobsubpng_folder)
    if isinstance(path_to_sub, str):
        path_to_sub = Path(path_to_sub)
    if not isinstance(path_to_vobsubpng_folder, Path):
        raise ValueError(f'path_to_vobsubpng_folder should be a str or a Path, here {type(path_to_vobsubpng_folder)}')
    if not isinstance(path_to_sub, Path):
        raise ValueError(f'path_to_sub should be a str or a Path, here {type(path_to_sub)}')
    if not path_to_vobsubpng_folder.is_dir():
        raise ValueError('path_to_vobsubpng_folder should be a path to a folder, containing the png files and the json index')
    if not path_to_vobsubpng_folder.exists():
        raise FileNotFoundError(f'The folder {path_to_vobsubpng_folder.absolute()} does not exist')
    if not path_to_sub.exists():
        raise FileNotFoundError(f'The file {path_to_sub.absolute()} does not exist')
    path_to_index= path_to_vobsubpng_folder / 'index.json'
    if not path_to_index.exists():
        # there is no index.json in the folder, maybe there is a xml file
        index = xml_index_to_json_index(path_to_folder=path_to_vobsubpng_folder)
        if index is None:
            raise FileNotFoundError(
                f"The folder exists but the index that should come with it does not, it should be a .json file (named inde.json) or a .xml file"
            )
    else: 
        with open(path_to_index) as f:
            index = json.load(f)

    
    document = DocumentPlus.parse_file_plus(str(path_to_sub))
    if len(document.events) <1: 
        raise ValueError(f'The subfile was parsed but no event were detected')
    
    if 'subtitles' not in index:
        raise ValueError('Prasing of index.json successfull but "subtitles" not  in the json')
    
    index['subtitles'] = sorted(index['subtitles'], key=lambda x: x['start'])
    last_found = 0
    event_with_pil_list = []
    tol_ms = 200 if len(index['subtitles']) != len(document.events) else 2000 # if the number of events is the same we are way less strict
    for i, sub in enumerate(index['subtitles']):
        if 'path' not in sub:
            logger.warning(f'the sub {i} does not have a path')
            continue
        sub_image_path = path_to_vobsubpng_folder / sub['path']
        if not sub_image_path.exists():
            logger.warning(f'The file {sub_image_path} does not exists, sub {i} skiped')
            continue

        sub_image = Image.open(sub_image_path)
        if sub['size'][0] != sub_image.size[0] or sub['size'][1] != sub_image.size[1]:
            logger.warning(f"The size of the sub n{i} in the index ({sub['size']}) is not the same as the real size ({sub_image.size})")
        
        # Here we need to find the event corresponding to the image
        # the index is not always a good indicator beacause events are sorted by time
        # start timing are the best information we have
        # sadly the end timing is often not the same in the .idx and the .sub (or vobsub2png does not write the correct end timing is dont know)
        # so if two sub have the exact same start timing, they can be swaped
        # most of the time, this does not happen in sub/idx files 
        corresponding_event = None
        j=last_found

        for event in document.events[last_found:]:
            event_ms = int(round(event.start.total_seconds() * 1000))
            json_ms   = int(round(sub["start"] * 1000))
            if json_ms - tol_ms <= event_ms <= json_ms + tol_ms:
                corresponding_event = event
                last_found=j+1
                break
            elif event_ms > json_ms + tol_ms:
                # because events are sorted by default, this means the corresponding event cannot be found
                raise IndexError(f'the corresponding event for sub {i} cannot be found')
            j+=1
        boxes = detect_text_line_xyxy(sub_image, multiline) # try to isolate the text 
        boxes.sort(key=lambda x: x[1], reverse=False) # we sort boxes by their top coord
        if multiline:
            corresponding_event = [corresponding_event]
        else:
            corresponding_event = split_dialogue(corresponding_event)
        if len(corresponding_event) != len(boxes):
            raise ValueError(f'The number of lines detected for the sub {i} ({len(boxes)} lines) is not the same as the number of lines in the text ({len(corresponding_event)} lines)')
        event_list: list[FrameToBoxEvent] = []
        for j, bbox in enumerate(boxes):
            d_padding = dynamic_padding(
                corresponding_event[j].text, padding=padding, 
                three_dots_padding=3, point_padding=2
            )
            b = padded_box_from_xyxy(bbox, sub_image.size, d_padding)
            baseline_b = adjust_box_to_baseline(sub_image, box=b) if multiline == False else None
            event_list.append(FrameToBoxEvent(Event=corresponding_event[j], Boxes=b, baseline_Boxes=baseline_b))
        
        if random.random() < p_CreateDoubleBorder:
            border_size = 10
            sub_image = simulate_CreateDoubleBorder(sub_image, borderSize=border_size)
            for sub in event_list:
                sub.Boxes.add_padding((border_size*2, border_size*2, border_size*2, border_size*2))
        event_with_pil_list.append(eventWithPil(image=sub_image, events=event_list))
            
    return eventWithPilList(event_with_pil_list)
        


def vobsubpng_to_dataset(
        root_dataset_path: str | Path,
        path_to_vobsubpng_folder: str | Path,
        path_to_sub: str | Path, 
        train_test_split: float | None = None,
        image_save_path: str | Path | None = None,
        test_image_save_path: str | Path | None = None,
        dataset_txt: str | Path | None = None,
        test_dataset_txt: str | Path | None = None,
        multiline: bool = True,
        padding: tuple[int,int,int,int] | float = (1, 1, 1, 1),
        format: Literal['PaddleOCR'] = 'PaddleOCR',
        weight: int = 1,
        
) -> None:
    """
    Convert a folder of VobSub PNG subtitles into a structured text detection dataset.

    This function processes PNG images exported from VobSub subtitle streams (e.g., using `vobsub2png`),
    aligns them with textual subtitle events from an `.ass` or `.srt` file, and saves both the images
    and corresponding text bounding boxes in a dataset format suitable for OCR training.

    It internally calls `vobsubpng_to_eventWithPilList()` to associate each PNG with:
        - its parsed subtitle event(s),
        - the detected text bounding boxes (from alpha channel),
        - and the subtitle text itself.

    Args:
        root_dataset_path (str | Path): 
            Root path of the dataset folder where images and annotations will be saved.
        path_to_vobsubpng_folder (str | Path): 
            Path to the folder containing PNG subtitle images and their `index.json`.
        path_to_sub (str | Path): 
            Path to the subtitle text file (`.ass` or `.srt`).
        image_save_path (str | Path | None, optional): 
            Path where processed PNG images should be stored. 
            Defaults to `<root_dataset_path>/images/text`.
        dataset_txt (str | Path | None, optional): 
            Path to the dataset text annotation file.
            Defaults to `<root_dataset_path>/dataset.txt`.
        multiline (bool, optional): 
            If True, treat multiline subtitles as a single text block.
            If False, split them into separate text boxes per line.
        padding (tuple[int,int,int,int], optional): 
            Padding (left, top, right, bottom) to apply around detected bounding boxes.
        format (Literal['PaddleOCR'], optional): 
            Output format for the dataset annotations. Currently supports 'PaddleOCR' only.
        weight (int, optional): 
            The number of times a image should appear in the dataset. Defaults to `1`.

    Raises:
        FileNotFoundError: If any of the required paths or files are missing.
        ValueError: If parsing fails or events and images cannot be aligned properly.
        IndexError: If subtitle timing alignment fails between PNG and text events.

    Example:
        ```python
        vobsubpng_to_dataset(
            path_to_vobsubpng_folder='/path/to/vobsubpng',
            path_to_sub='/path/to/subs/video.ass',
            multiline=False,
            root_dataset_path='/path/to/dataset'
        )
        ```
    Notes:
        - The alpha channel of each PNG is used to detect text areas.
        - An `index.json` file generated alongside the PNGs is required for proper time alignment.
        - Output images and labels can be used directly to train OCR models such as PaddleOCR.
    """
    def write_metadata(
            dataset_path: str,
            multiline: bool,
            sub_name: str,
            format: str,
            n_text_images: int,
            metadata_name: str = 'dataset_metadata.txt',
        ) -> None:
        if not exists(join(dataset_path, metadata_name)):
            return None
        with open(join(dataset_path, metadata_name), encoding='utf-8', mode='a') as f:
            f.write("========================================\n")
            f.write(f'Added vobsub PNG Images from {sub_name}\n')
            f.write(f'Date: {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}\n')
            f.write(f'Multiline: {multiline}\n')
            f.write(f'Save format: {format}\n')
            f.write(f'Images added: {n_text_images} (text: {n_text_images}, no_text: 0)\n')
            f.write('========================================\n')
    if train_test_split is not None and (not isinstance(train_test_split, float) or train_test_split >1 or train_test_split <0):
        raise ValueError(f'train_test_split should be a float between 0 and 1 (here {train_test_split})')
    
    if image_save_path is None: 
        image_save_path = join(str(root_dataset_path), 'det_images', 'text') if train_test_split is None else join(str(root_dataset_path), 'det_images','train', 'text')
    if test_image_save_path is None:
        test_image_save_path = join(str(root_dataset_path), 'det_images','test' ,'text')
    if not exists(image_save_path):
        makedirs(image_save_path, exist_ok=True)
    if train_test_split is not None and not exists(test_image_save_path):
        makedirs(test_image_save_path, exist_ok=True)
    if dataset_txt is None: 
        dataset_txt = join(str(root_dataset_path), 'dataset.txt') if train_test_split is None else join(str(root_dataset_path),'train', 'detImages_train_text.txt')
    if test_dataset_txt is None:
        test_dataset_txt = join(str(root_dataset_path),'test', 'detImages_test_text.txt')
    if not isinstance(weight, int) or weight < 1:
        raise ValueError('weight should be a positive int')

    
    
    sub_name = Path(path_to_sub).stem
    eventWithPillist = vobsubpng_to_eventWithPilList(
        path_to_sub=path_to_sub,
        path_to_vobsubpng_folder=path_to_vobsubpng_folder,
        multiline=multiline,
        padding=padding
    )

    for event in eventWithPillist:
        is_test = False if train_test_split is None else random.random()>train_test_split

        temp_dataset_txt = test_dataset_txt if is_test else dataset_txt
        temp_image_save_path = test_image_save_path if is_test else image_save_path
        for i in range(1, weight+1):
            image_num = f'_{i}' if weight > 1 else ''
            image_name = f'{sub_name}_sVOB_t{event.events[0].Event.start.total_seconds()}{image_num}.png'
            event.image.save(join(str(temp_image_save_path), image_name))

            event_dataset_image = dataset_image(
                image_path=join(temp_image_save_path, image_name),
                event_list=event.events
            )

            event_dataset_image.to_text(
                path=temp_dataset_txt,
                format=format
            )
    
    write_metadata(
        dataset_path=str(root_dataset_path),
        multiline=multiline,
        sub_name=sub_name,
        format=format,
        n_text_images=len(eventWithPillist),
    )

