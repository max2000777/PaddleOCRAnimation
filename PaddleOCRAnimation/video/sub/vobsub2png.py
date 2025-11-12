import subprocess
from os.path import exists, dirname, abspath, join, relpath
from os import makedirs
import importlib.resources
from PIL import Image
from pathlib import Path
import logging
import json
from platform import system
from .DocumentPlus import DocumentPlus, split_dialogue
from .RendererClean import Box
from ..Video import eventWithPil, FrameToBoxEvent, eventWithPilList
from datetime import datetime
from ..classes import dataset_image
from typing import Literal
from ..utilis import detect_text_line_boxes

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

def vobsubpng_to_eventWithPilList(
        path_to_vobsubpng_folder: str | Path,
        path_to_sub: str | Path, 
        multiline: bool = True,
        padding: tuple[int,int,int,int] = (7,2,3,1)
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
        


def vobsubpng_to_dataset(
        root_dataset_path: str | Path,
        path_to_vobsubpng_folder: str | Path,
        path_to_sub: str | Path, 
        image_save_path: str | Path | None = None,
        dataset_txt: str | Path | None = None,
        multiline: bool = True,
        padding: tuple[int,int,int,int] = (1, 1, 1, 1),
        format: Literal['PaddleOCR'] = 'PaddleOCR',
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
    if image_save_path is None : 
        image_save_path = join(str(root_dataset_path), 'det_images', 'text')
    if not exists(image_save_path):
        makedirs(image_save_path, exist_ok=True)
    if dataset_txt is None: 
        dataset_txt = join(str(root_dataset_path), 'dataset.txt')
    
    sub_name = Path(path_to_sub).stem
    image_dataset_path = relpath(image_save_path, root_dataset_path)
    eventWithPillist = vobsubpng_to_eventWithPilList(
        path_to_sub=path_to_sub,
        path_to_vobsubpng_folder=path_to_vobsubpng_folder,
        multiline=multiline,
        padding=padding
    )

    for event in eventWithPillist:
        image_name = f'{sub_name}_sVOB_t{event.events[0].Event.start.total_seconds()}.png'
        event.image.save(join(str(image_save_path), image_name))

        event_dataset_image = dataset_image(
            image_path=join(image_dataset_path, image_name),
            event_list=event.events
        )

        event_dataset_image.to_text(
            path=dataset_txt,
            format=format
        )
    
    write_metadata(
        dataset_path=str(root_dataset_path),
        multiline=multiline,
        sub_name=sub_name,
        format=format,
        n_text_images=len(eventWithPillist),
    )

