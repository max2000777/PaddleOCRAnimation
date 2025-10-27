import subprocess
from os.path import exists, dirname, abspath, join
import importlib.resources
from PIL import Image
from pathlib import Path
import logging
import json
from platform import system
from .DocumentPlus import DocumentPlus, split_dialogue
from .RendererClean import Box
from ..Video import eventWithPil, FrameToBoxEvent, eventWithPilList

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
        print("Erreur de commande :", e.stderr)

def vobsubpng_to_dataset(
        path_to_vobsubpng_folder: str | Path,
        path_to_sub: str | Path, 
        multiline: bool = True,
        padding: tuple[int,int,int,int] = (0,0,0,0)
)->eventWithPilList:
    """
    Build a dataset linking VobSub PNG images to subtitle text and bounding boxes.

    This function associates each PNG subtitle image (exported via vobsub2png or similar)
    with its corresponding subtitle event parsed from an `.ass` or `.srt` file.
    It detects text regions on each PNG using the alpha channel and aligns them with
    the text lines from the subtitle file.

    Args:
        path_to_vobsubpng_folder (str | Path): Path to the folder containing VobSub PNGs and `index.json`.
        path_to_sub (str | Path): Path to the subtitle text file (`.ass` or `.srt`).
        multiline (bool, optional): Whether to treat multiline subtitles as a single block. Defaults to True.
        padding (tuple[int,int,int,int], optional): Padding (left, top, right, bottom) to expand bounding boxes.

    Returns:
        eventWithPilList: A list-like object where each element links:
            - the subtitle PNG (`PIL.Image`),
            - detected text boxes,
            - and the corresponding subtitle text lines.

    Raises:
        FileNotFoundError: If required files or folders are missing.
        ValueError: If parsing fails or detected text boxes don't match text lines.
        IndexError: If events cannot be aligned by timing.

    Notes:
        - The function assumes PNG transparency corresponds to text areas.
        - It expects an `index.json` with subtitle metadata and start times.
    """
    def detect_text_line_boxes(sub_image, multiline: bool = True):
        import numpy as np
        # we need to find the box  of the text, because the image is transparent it is relativly easy
        alpha = sub_image.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            return []  # there is not text on the image
        
        if multiline:
            # beacause multiline is allowed, no further modification need to be done
            return [bbox]

        cropped = alpha.crop(bbox)
        arr = np.array(cropped)
        binary = (arr > 0).astype(np.uint8)
        projection = binary.sum(axis=1)
        line_boxes = []
        in_line = False
        start = 0

        for y, val in enumerate(projection):
            if val > 0 and not in_line:
                in_line = True
                start = y
            elif val == 0 and in_line:
                in_line = False
                end = y
                line_boxes.append((start, end))

        if in_line:
            line_boxes.append((start, len(projection)))
        abs_boxes = []
        for (y1, y2) in line_boxes:
            # we want the boxes of each lines
            abs_y1 = bbox[1] + y1
            abs_y2 = bbox[1] + y2
            abs_boxes.append((bbox[0], abs_y1, bbox[2], abs_y2))
        return abs_boxes

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
        raise FileNotFoundError(f"The folder exists but the index that should come with it does not : \n{path_to_index.absolute()}")
    
    document = DocumentPlus.parse_file_plus(str(path_to_sub))
    if len(document.events) <1: 
        raise ValueError(f'The subfile was parsed but no event were detected')
    
    with open(path_to_index) as f:
        index = json.load(f)
    if 'subtitles' not in index:
        raise ValueError('Prasing of index.json successfull but "subtitles" not  in the json')
    
    index['subtitles'] = sorted(index['subtitles'], key=lambda x: x['start'])
    last_found = 0
    event_with_pil_list = []
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
            logger.warning(f'The size of the sub n{i} in the index ({sub['size']}) is not the same as the real size ({sub_image.size})')
        
        # Here we need to find the event corresponding to the image
        # the index is not always a good indicator beacause events are sorted by time
        # start timing are the best information we have
        # sadly the end timing is often not the same in the .idx and the .sub (or vobsub2png does not write the correct end timing is dont know)
        # so if two sub have the exact same start timing, they can be swaped
        # most of the time, this does not happen in sub/idx files 
        corresponding_event = None
        j=last_found
        for event in document.events[last_found:]:
            if round(event.start.total_seconds(), 2) == round(sub['start'],2):
                corresponding_event = event
                last_found=j+1
                break
            elif event.start.total_seconds() > sub['start']:
                # because events are sorted by default, this means the corresponding event cannot be found
                raise IndexError(f'the corresponding event for sub {i} cannot be found')
            j+=1
        boxes = detect_text_line_boxes(sub_image, multiline) # try to isolate the text 
        boxes.sort(key=lambda x: x[1], reverse=False) # we sort boxes by their top coord
        if multiline:
            corresponding_event = [corresponding_event]
        else:
            corresponding_event = split_dialogue(corresponding_event)
        if len(corresponding_event) != len(boxes):
            raise ValueError(f'The number of lines detected for the sub {i} ({len(boxes)} lines) is not the same as the number of lines in the text ({len(corresponding_event)} lines)')
        event_list = []
        for j, bbox in enumerate(boxes):
            w, h = sub_image.size
            b = Box(
                [max(bbox[0]-padding[0], 0), max(bbox[1]-padding[1], 0)], [min(bbox[2]+padding[2], w), max(bbox[1]-padding[1], 0)],
                [min(bbox[2]+padding[2], w), min(bbox[3]+padding[3], h)], [max(bbox[0]-padding[0], 0), min(bbox[3]+padding[3], h)]
            )
            event_list.append(FrameToBoxEvent(Event=corresponding_event[j], Boxes=b))
        event_with_pil_list.append(eventWithPil(image=sub_image, events=event_list))
            
    return eventWithPilList(event_with_pil_list)
        



if __name__ == "__main__":
    vobsubpng_to_dataset(
        path_to_sub='/home/maxim/code/SubProject/PaddleOCRAnime/examples/data/subs/A1_t00.ass',
        path_to_vobsubpng_folder='/home/maxim/code/SubProject/PaddleOCRAnime/dev/subcrest'
    )

