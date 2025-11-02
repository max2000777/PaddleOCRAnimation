from PIL import Image
from typing import TypedDict, Optional, Union
from .sub.RendererClean import Box
from ass import line
import re
from json import dumps
from dataclasses import dataclass
from os.path import isabs, join, exists, abspath, dirname, relpath, basename, splitext
from os import makedirs
from pprint import pformat
import json
import re
from pathlib import Path 
from typing import Literal
import os

class SubTrackInfo(TypedDict):
    """
    Représente les métadonnées d'une piste de sous-titres extraite d'un fichier MKV.

    Ce dictionnaire est généré par la fonction `recup_infos_MKV`, qui s'appuie sur
    `ffprobe` pour détecter les flux de type "subtitle" présents dans un conteneur MKV.

    Attributes:
        index (int) : Index brut de la piste tel que retourné par ffprobe (utilisé par ffmpeg).

        id_sub (int) : Identifiant relatif de la piste de sous-titres, réinitialisé à 0 pour la
            première piste détectée, 1 pour la deuxième, etc. Sert à référencer les
            pistes dans le reste du code (par exemple pour l'affichage).

        langage (str) : Langue déclarée de la piste (ex. "eng", "fre", etc.),
            ou "inconnu" si absente.

        title (str) : Titre descriptif éventuel de la piste (ex. "Forcés", "SDH", etc.),
            ou "non nommé" si non spécifié dans les métadonnées.

        codec (Optional[str]) : Nom du codec utilisé pour la piste de sous-titres
            (ex "ass", "srt", "subrip").
            Peut être `None` si l'information est manquante ou inaccessible.

        nb_frames (Union[str, int]) : Nombre de sous-titres (ou d'événements) estimés dans
            cette piste.
            Cette information est issue des métadonnées (tag `NUMBER_OF_FRAMES`) et peut
            parfois être une chaîne de caractères ou absente.

        fps (float) : Fréquence estimée des sous-titres (nombre de sous-titres par seconde),
            calculée comme `nb_frames / durée`. La fiabilité dépend de la disponibilité
            de ces deux informations.

        is_extarnal (bool) : Si le sous titre est dans le MKV ou juste dans le même dossier
    """
    index: int
    id_sub: int
    langage: str
    title: str
    codec: Optional[str]
    nb_frames: Union[str, int]
    fps: float
    is_extarnal: bool


class MKVInfo(TypedDict):
    durée: float
    sous_titres: list[SubTrackInfo]


@dataclass
class FrameToBoxEvent:
    Event: line.Dialogue
    Boxes: Box

    def to_transcription(self):
        text = re.sub(r'\{[^}]*\}', '', self.Event.text)

        return {
            'transcription': text,
            'points': self.Boxes.full_box
        }


@dataclass
class FrameToBoxResult:
    ImagePath: str
    Events: list[FrameToBoxEvent]

    def to_dect_dataset(self) -> str:
        event_textes = [
            dumps(event.to_transcription(), ensure_ascii=False) for event in self.Events
        ]
        chemin_normalise = self.ImagePath.replace("\\", "/")
        return f"{chemin_normalise}   [{', '.join(event_textes)}]"
    
    def to_rec_dataset(self, dataset_path: str | None = None, image_save_path: str | None = None) -> str:
        """
        Génère un dataset de reconnaissance de texte à partir des zones détectées dans l'image,
        en enregistrant chaque zone recadrée comme une image et en produisant un texte listant
        le chemin de chaque crop et son texte associé.

        Args:
            dataset_path (str | None, optional): Chemin vers la racine du dataset. Utilisé pour
                résoudre les chemins relatifs et produire des chemins relatifs dans le fichier texte.
                Si None, les chemins relatifs sont interprétés depuis le répertoire courant.
            image_save_path (str | None, optional): Dossier où sauvegarder les images recadrées.
                Si None, les crops sont enregistrés dans un sous-dossier du dataset ou à côté de l'image.

        Returns:
            str: Contenu formaté du dataset texte, avec une ligne par crop sous la forme
                "<chemin_image>\t<texte>".
        """
        def normalize_box(box: Box) -> tuple[int, int, int, int]:
            full = box.full_box
            left = min(p[0] for p in full)
            top = min(p[1] for p in full)
            right = max(p[0] for p in full)
            bottom = max(p[1] for p in full)
            return (left, top, right, bottom)

        if isabs(self.ImagePath):
            base_image_path = self.ImagePath
        else:
            if dataset_path is not None:
                base_image_path = join(dataset_path, self.ImagePath)
            else:
                base_image_path = self.ImagePath

        if not exists(base_image_path):
            raise FileNotFoundError(f"L'image n'existe pas : {abspath(base_image_path)}")

        base_img = Image.open(base_image_path)

        if image_save_path is None:
            if dataset_path is None:
                save_dir = dirname(base_image_path)
            else:
                save_dir = join(dataset_path, "images")
        else:
            save_dir = image_save_path

        return_texte = ""
        for i, event in enumerate(self.Events):
            crop_box = normalize_box(event.Boxes)
            crop = base_img.crop(crop_box)

            save_name = f"{splitext(basename(base_image_path))[0]}_{i}.png"
            final_crop_path = join(save_dir, save_name)

            makedirs(dirname(final_crop_path), exist_ok=True)
            crop.save(final_crop_path)

            if dataset_path:
                rel_crop_path = relpath(final_crop_path, dataset_path)
            else:
                rel_crop_path = final_crop_path

            clean_text = re.sub(r'\{[^}]*\}', '', event.Event.text)

            return_texte += f"{rel_crop_path}\t{clean_text}\n"

        return return_texte

@dataclass
class eventWithPil:
    image: Image.Image
    events: list[FrameToBoxEvent]

    def to_pil(self, show_boxes: bool = True)-> Image.Image:
        """
        Converts the current event (subtitle) and its bounding boxes into a composite PIL image.

        Args:
            show_boxes (bool, optional): If True, overlays the subtitle bounding boxes on the image.
                                        Defaults to True.

        Returns:
            Image.Image: The rendered image with optional subtitle box overlays.
        """
        size = self.image.size
        base=self.image
        if not show_boxes:
            return base
        for event in self.events:
            event_pil = event.Boxes.to_pil(size)
            base = Image.alpha_composite(base, event_pil)
        return base
    
    def add_padding(self, padding: tuple[int, int, int, int]):
        """Add or remove transparent padding around the subtitle image and update event bounding boxes.

        This method adjusts the current RGBA subtitle image by adding or removing padding 
        on each side, and updates all associated event bounding boxes accordingly.

        The padding is defined as (left, top, right, bottom):
        - Positive values add transparent space around the image.
        - Negative values crop the image, removing pixels from the corresponding sides.

        When cropping occurs, any subtitle event (i.e., a line from an ASS or SRT file) 
        whose bounding box falls completely outside the visible image area is removed. 
        Events partially affected by cropping are clamped so that their coordinates remain 
        within the new image boundaries.

        Args:
            padding (tuple[int, int, int, int]): Padding values (left, top, right, bottom). 
                Positive values expand the image, negative values crop it.

        Raises:
            ValueError: 
                - If `padding` is not a tuple of four integers.
                - If the requested negative padding would crop more than the image size.
        """
        if (
            not isinstance(padding, tuple) 
            or not all([isinstance(a, int) for a in padding]) 
            or not len(padding)==4
        ):
            raise ValueError(f'padding should be a tuple with 4 int, here {padding}')
        previous_w, previous_h = self.image.size
        if (min(padding[0],0)+min(padding[2],0)+previous_w < 0) or (min(padding[1],0)+min(padding[3],0)+previous_h < 0):
            raise ValueError("Cannot crop more than the size of the image")

        if any([a < 0 for a in padding]):
            self.image=self.image.crop((-min(0,padding[0]), -min(0, padding[1]), previous_w+min(0, padding[2]), previous_h+min(0, padding[3])))
            previous_w, previous_h = self.image.size
        
        new_image = Image.new(
            mode="RGBA",
            size=(
                previous_w+max(padding[0],0)+max(padding[2],0),
                previous_h+max(padding[1],0)+max(padding[3],0)
            ),
            color=(0, 0, 0, 0)
        )
        new_image.paste(self.image, (max(padding[0],0), max(padding[1],0)), self.image)
        self.image = new_image

        newevents = []
        for event in self.events:
            event.Boxes.add_padding(padding=padding, image_size=new_image.size)
            if event.Boxes.full_box != [[0, 0], [0, 0], [0, 0], [0, 0]]:
                newevents.append(event)
        self.events = newevents
            
            



class eventWithPilList(list[eventWithPil]):
    """
    A specialized list of `eventWithPil` objects with convenient visualization and compositing methods.

    Each element in this list contains:
        - `image`: a PIL Image representing a frame.
        - `events`: a list of `FrameToBoxEvent` objects, each with text and bounding box information.

    Representation:
        The `__repr__` method returns a nested dictionary-like string for readability, showing:
            - The index of each `eventWithPil` in the list.
            - The size of its PIL image.
            - The text and bounding box of each event.
        Note: This is purely for human-readable output and does NOT reflect the actual internal structure,
              which remains a list of `eventWithPil` objects.

    Example:
        {
          '0': { '0': { 'Box': '[[881, 49], [1046, 49], [1046, 110], [881, 110]]',
                         'Text': '{\\an2\\pos(966.4,103.1)\\fnIwata Mincho Old Pro-Fate B\\fs50\\blur0.9}Preview'},
                 'PilImage Size': (1920, 1080)},
          '1': { '0': { 'Box': '[[984, 274], [1612, 274], [1612, 345], [984, 345]]',
                         'Text': '{\\an2\\pos(1298.67,336)\\fnIwata Mincho Old Pro-Fate B\\fs60\\c&HFF4DD5&\\blur0.9}Dis-nous au moins ton nom !'},
                 'PilImage Size': (1920, 1080)},
          ...
        }
    """
    def __repr__(self) -> str:
        return_str = {}
        for i, event in enumerate(self):
            event_dict = {"PilImage Size": event.image.size}
            for j, line in enumerate(event.events):
                event_dict[str(j)]= {"Text":line.Event.text, 'Box':str(line.Boxes.full_box)}
            return_str[str(i)] = event_dict
        return pformat(return_str, indent=2, width=100)

    def to_pil(self, show_boxes: bool = True) -> Image.Image:
        """
        Composites all images in the list into a single PIL image, optionally overlaying bounding boxes for each event.

        Args:
            show_boxes (bool, optional): Whether to overlay bounding boxes of events on the resulting image.
                                         Defaults to True.

        Raises:
            ValueError: If the images in the list do not all have the same size. The object was propably created without giving a size.

        Returns:
            Image.Image: A single PIL Image representing the composited frames with optional bounding boxes.
        """
        size = self[0].image.size
        base = Image.new(mode="RGBA", size=size)
        for event in self:
            if size != event.image.size:
                raise ValueError('The sizes of the images do not match, the list was most likely made without the size attribute')
            size = event.image.size

            base = Image.alpha_composite(base, event.image)
            if show_boxes:
                for line in event.events:
                    base = Image.alpha_composite(base, line.Boxes.to_pil(size))

        return base
    def add_padding(self, padding: tuple[int, int, int, int]):
        """Apply padding to all subtitle events in the list.

        This method iterates over each `eventWithPil` in the list and applies the 
        specified padding using each event's `add_padding` method. Positive padding 
        expands the image around each event, while negative padding crops it. 
        Any events that become fully outside their image after cropping are automatically 
        removed by the event-level method.

        Args:
            padding (tuple[int, int, int, int]): Padding values (left, top, right, bottom).
                Positive values add transparent space, negative values crop the image.

        Raises:
            ValueError: 
                - If `padding` is not a tuple of four integers.
                - If any of the individual event's padding operations would crop more than 
                  its image size (handled by the event's own `add_padding` method).
        """
        if (
            not isinstance(padding, tuple) 
            or not all([isinstance(a, int) for a in padding]) 
            or not len(padding)==4
        ):
            raise ValueError(f'padding should be a tuple with 4 int, here {padding}')
        l=[]
        for event in self:
            event.add_padding(padding=padding)
            if len(event.events)>0:
                l.append(event)
        self[:]= l


@dataclass
class dataset_image:
    """The represtation of one dataset entry
    """
    image_path: str
    event_list: list[FrameToBoxEvent]

    def to_paddleOCR(
            self, path: str | Path,
            remove_overrides: bool = True,
            prevent_karaoke: bool = True,
        ) -> None:
        """Append this entry to a PaddleOCR-compatible .txt dataset file.
        
        Each line has the format:
            <image_path>\t[{"transcription": "...", "points": [[x1,y1], [x2,y2], ...]}, ...]

        Args:
            path (str | Path): Path to the output .txt file.
            remove_overrides (bool): if `True` will remove overrides in the text (ex: `{\\i1}`)
        """

        dict_list = []
        for event in self.event_list:
            if event.Boxes.full_box == [[0, 0], [0, 0], [0, 0], [0, 0]]:
                continue

            text: str = event.Event.text
            if remove_overrides:
                l_text = len(text)
                text = re.sub(r'\{[=\\][^}]*\}', '', text)
                if prevent_karaoke and l_text > 15 and len(text) < 3:
                    # karaoke a often 99% overrides and 1 or 2 real letters
                    continue
            dict_list.append({"transcription": text, "points": event.Boxes.full_box})
        
        with open(path, mode='a', encoding='utf-8') as f:
            f.write(f"{self.image_path}\t{json.dumps(dict_list, ensure_ascii=False)}\n")


    def to_text(self, path: str | Path | None = None, 
                remove_overrides: bool = True, 
                format: Literal['PaddleOCR'] = 'PaddleOCR'):
        """Export this entry to a dataset text file in the specified format.

        Args:
            path (str | Path | None, optional): Output file path. Defaults to './dataset.txt'.
            format (Literal['PaddleOCR'], optional): Output format. Currently only 'PaddleOCR' is supported.
            remove_overrides (bool): if `True` will remove overrides in the text (ex: `{\\i1}`)

        Raises:
            ValueError: If the output file is not a .txt or if the format is unsupported.
        """
        if path is None: 
            path = os.path.join(os.getcwd(), 'dataset.txt')
        if not str(path).endswith('.txt'):
            raise ValueError(f'Path should be a path to a .txt file')
        if format == 'PaddleOCR':
            self.to_paddleOCR(path=path, remove_overrides=remove_overrides)
        else: 
            raise ValueError(f"Unsupported format: {format}")