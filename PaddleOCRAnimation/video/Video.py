from os.path import exists, dirname, abspath, join, relpath, splitext, basename, isabs
from shutil import which
from typing import TypedDict, Optional, Union, overload, Literal
import subprocess
from json import loads
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from datetime import timedelta
from os import makedirs, chdir, getcwd
from warnings import warn
from .sub import RendererClean
from .sub.DocumentPlus import DocumentPlus
from ass import line, data, Dialogue
from copy import deepcopy
import random
import matplotlib.font_manager as fm
from numpy import clip
from numpy.random import normal
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from json import dumps
from platform import system
from ctypes import cdll
# from s3fs.core import S3File
import re
import importlib.resources
from pathlib import Path
from pprint import pformat


try:
    system = system()
    if system == "Linux":
        cdll.LoadLibrary("libGL.so.1")
except OSError as e:
    print(
        f"{e}\nVous n'avez pas la librairie libGL d'installée veuillez "
        "l'installer en faisant (sur Ubuntu) :\nsudo apt install libgl1"
        )
else:
    import cv2


def style_transform(self: line.Style) -> line.Style:
    """Applique des transformations aléatoires sur les attributs d'un style de ligne.

    Cette fonction modifie aléatoirement certains paramètres visuels d'un objet `line.Style`
    pour créer de la diversité graphique :
        - Changement de police (en évitant les polices problématiques sous Windows)
        - Perturbation des couleurs (primaire et contour) via une distribution normale
        - Légère variation de la taille de police
        - Inversion d'alignement (haut ↔ bas) pour certains cas
        - Invertion de italique/gras

    Args:
        self (line.Style): Style de ligne d'origine à transformer.

    Returns:
        line.Style: Nouveau style modifié de manière aléatoire.
    """
    def change_color(color: data.Color, ecart_type: float = 80) -> data.Color:
        for col in ['r', 'g', 'b']:
            setattr(color, col, int(clip(normal(getattr(color, col), ecart_type), 0, 255)))
        return color

    mauvaises_polices = {  # Les polices qui, sur windows, ne donne pas du texte
        "Wingdings 2", "Webdings", "Wingdings", "MS Reference Specialty",
        "MT Extra", "MS Outlook", "Bookshelf Symbol 7", "Segoe MDL2 Assets",
        "Symbol", "Segoe Fluent Icons", "Wingdings 3"
    }
    self = deepcopy(self)
    if random.random() < 0.25:
        nom_polices = {
            fm.FontProperties(fname=font).get_name(): font
            for font in fm.findSystemFonts(fontpaths=None, fontext='ttf')
        }
        nouvelle_police = random.choice(
            [nom for nom in nom_polices if nom not in mauvaises_polices]
        )
        self.fontname = nouvelle_police

    if random.random() < 0.25:
        self.primary_color = change_color(self.primary_color)

    if random.random() < 0.15:
        self.outline_color = change_color(self.outline_color)

    if random.random() < 0.2:
        self.fontsize = normal(self.fontsize, 5)

    mapAlignement = {8: 2, 2: 8}
    if self.alignment in mapAlignement and random.random() < 0.1:
        self.alignment = mapAlignement[self.alignment]

    if random.random() < 0.1:
        self.bold = not self.bold

    if random.random() < 0.15:
        self.italic = not self.italic

    return self


def degrade_image(img: Image.Image) -> Image.Image:
    """Applique aléatoirement une ou plusieurs dégradations visuelles à une image.

    Cette fonction simule des artefacts réalistes pouvant survenir dans des images du monde réel,
    tels que le flou, le bruit, la compression JPEG, et le bruit "sel et poivre".

    Les effets sont appliqués avec des probabilités différentes :
        - Flou gaussien : 30%
        - Bruit gaussien : 15%
        - Compression JPEG : 20%
        - Sel et poivre : 10%

    Args:
        img (Image.Image): Image d'entrée (PIL), au format RGB ou RGBA.

    Returns:
        Image.Image: Nouvelle image dégradée, au même format (RGBA si l'entrée l'était).
    """
    def add_noise(img, mean=0, std=10):
        """Rajoute du bruit (grain) sur l'image
        """
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(mean, std, np_img.shape)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)  # Pour rester entre 0 et 255
        img = Image.fromarray(noisy_img)
        return img.convert('RGBA') if img.mode == 'RGBA' else img

    def jpeg_compress(img: Image.Image, quality=10):
        """Sauvegarde sur RAM en JPEG (avec compression) et réouvre cette sauvegarde
        """
        baseMode = img.mode
        if baseMode == 'RGBA':
            img = img.convert('RGB')
        buffer = BytesIO()  # Sauvegarde en RAM et non sur le disque
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGBA') if baseMode == 'RGBA' else Image.open(buffer)

    def salt_and_pepper(img, amount=0.003):
        """Rajoute des points blancs et noirs sur l'image
        """
        np_img = np.array(img)
        num_salt = np.ceil(amount * np_img.size * 0.5)
        num_pepper = np.ceil(amount * np_img.size * 0.5)

        # Salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_img.shape]
        np_img[tuple(coords)] = 255

        # Pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_img.shape]
        np_img[tuple(coords)] = 0
        img = Image.fromarray(np_img)
        return img.convert('RGBA') if img.mode == 'RGBA' else img

    if random.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))
    if random.random() < 0.15:
        img = add_noise(img, std=random.uniform(2, 12))
    if random.random() < 0.20:
        img = jpeg_compress(img, quality=random.randint(15, 36))
    if random.random() < 0.1:
        img = salt_and_pepper(img).convert('RGBA') if img.mode == 'RGBA' else salt_and_pepper(img)
    return img


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
    """
    index: int
    id_sub: int
    langage: str
    title: str
    codec: Optional[str]
    nb_frames: Union[str, int]
    fps: float


class MKVInfo(TypedDict):
    durée: float
    sous_titres: list[SubTrackInfo]


@dataclass
class FrameToBoxEvent:
    Event: line.Dialogue
    Boxes: RendererClean.Box

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
        def normalize_box(box: RendererClean.Box) -> tuple[int, int, int, int]:
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
        """Add transparent padding around the subtitle image and update all box coordinates.

        The padding is defined as (left, top, right, bottom). 
        A transparent border is added around the current RGBA image, 
        and each event's bounding box is shifted accordingly.

        Args:
            padding (tuple[int, int, int, int]): Padding values (left, top, right, bottom).

        Raises:
            ValueError: If padding is not a tuple of four integers.
        """
        if (
            not isinstance(padding, tuple) 
            or not all([isinstance(a, int) for a in padding]) 
            or not len(padding)==4
        ):
            raise ValueError(f'padding should be a tuple with 4 int, here {padding}')
        previous_w, previous_h = self.image.size
        new_image = Image.new(
            mode="RGBA",
            size=(
                previous_w+padding[0]+padding[2],
                previous_h+padding[1]+padding[3]
            ),
            color=(0, 0, 0, 0)
        )
        new_image.paste(self.image, (padding[0], padding[1]), self.image)
        self.image = new_image

        for event in self.events:
            event.Boxes.add_padding(padding=padding)



    




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

class Video:
    path: str
    duree: float
    taille: tuple[int, int]
    sous_titres: list[SubTrackInfo]
    extracted_sub_path: dict[int, str]
    attachement_path: str | None
    docs: dict[int, DocumentPlus]

    def __init__(self, path: str, taille: tuple[int, int],
                 duree: float, sous_titres: list[SubTrackInfo]):
        self.path: str = path
        self.taille: tuple[int, int] = taille
        self.duree: float = duree
        self.sous_titres: list[SubTrackInfo] = sous_titres
        self.extracted_sub_path: dict[int, str] = {}
        self.attachement_path: str | None = None
        self.docs: dict[int, DocumentPlus] = {}

    @staticmethod
    def recup_infos_MKV(cheminVersMKV: str):
        """
        Extrait des informations sur les pistes de sous-titres d'un fichier MKV via ffprobe.

        Args:
            cheminVersMKV (str): Chemin vers le fichier MKV à analyser.

        Raises:
            FileExistsError: Si le fichier spécifié n'existe pas.
            SystemError: Si ffprobe n'est pas installé ou absent du PATH.
            RuntimeError: Si une erreur se produit lors de l'exécution de ffprobe.

        Returns:
            dict: Un dictionnaire `MKVInfo` contenant :
                - 'durée' (float): Durée du fichier en secondes (0 si non disponible).
                - 'taille' (tuple[int]): la taille de la vidéo (largeur, hauteur).
                - 'sous_titres' (list[dict]): Liste de dictionnaires représentant chaque piste
                    de sous-titres, avec :
                    - 'index' (int): Index de la piste dans le conteneur.
                    - 'id_sub' (int): ID de piste de sous-titres relatif (0, 1, 2...).
                    - 'langage' (str): Langue déclarée (ou "inconnu").
                    - 'title' (str): Titre de la piste (ou "non nommé").
                    - 'codec' (str | None): Codec utilisé (ex: "ass", "srt").
                    - 'nb_frames' (str | int): Nombre de sous-titres détectés (souvent chaîne).
                    - 'fps' (float): Estimation des sous-titres/seconde
                    (peut être faux si nb_frames non fiable).
        """
        if which("ffprobe") is None:
            raise SystemError("ffprobe n'est pas installé ou n'est pas dans le PATH")
        if not exists(cheminVersMKV):
            raise FileExistsError(f"Le fichier MKV n'existe pas :\n{cheminVersMKV}")
        commande = [
                "ffprobe",
                "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                cheminVersMKV
            ]
        # comprint = " ".join(commande)
        result = subprocess.run(
            commande,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        if result.returncode != 0:
            raise RuntimeError(f"erreur ffprobe : {result.stderr}")

        info = loads(result.stdout)
        duration = float(info['format']['duration']) if 'duration' in info['format'] else 0

        pistes_sous_titres = []
        id_sub = 0
        SIZE = (0, 0)
        for stream in info['streams']:
            if (
                stream['codec_type'] == 'video'
                and stream.get("codec_name") != "mjpeg"
                and stream.get('avg_frame_rate') != '0/0' # some video stream can in fact be a image
            ):
                SIZE = (stream['width'], stream['height'])
            if stream['codec_type'] == 'subtitle':
                nb_frames = 0
                if 'tags' in stream:
                    for cle, valeur in stream['tags'].items():  # bug de 'NUMBER_OF_FRAMES-eng'
                        if cle.startswith('NUMBER_OF_FRAMES'):
                            nb_frames = valeur

                info_sous_titres = {
                    'index': stream.get('index'),
                    'id_sub': id_sub,
                    'langage': stream.get('tags', {}).get('language', 'inconnu'),
                    'title': stream.get('tags', {}).get('title', 'non nommé'),
                    'codec': stream.get('codec_name', None),
                    'nb_frames': nb_frames,
                    'fps': float(nb_frames)/float(duration) if duration is not None else 0
                }
                pistes_sous_titres.append(info_sous_titres)
                id_sub += 1

        return {
            'durée': duration,
            'taille': SIZE,
            'sous_titres': pistes_sous_titres
        }

    def extract_frame_as_pil(self, timestamp: int | timedelta) -> Image.Image:
        """
    Extrait une image (frame) de la vidéo (SANS LES SOUS TITRES) à un instant donné
    et la retourne sous forme d'objet `PIL.Image`.

    Cette méthode utilise OpenCV pour accéder à la vidéo et extraire la frame
    correspondant au timestamp spécifié (en secondes ou en objet timedelta).
    L'image extraite est convertie en format RGBA compatible PIL.

    Args:
        timestamp (int | timedelta):
            - Si int, temps en secondes depuis le début de la vidéo.
            - Si timedelta,  durée depuis le début de la vidéo.

    Returns:
        Image: L'image extraite à l'instant donné, au format PIL.Image (RGBA).
    """
        mkv_path = self.path
        cap = cv2.VideoCapture(mkv_path)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {mkv_path}")
        if isinstance(timestamp, timedelta):
            timestamp = int(timestamp.total_seconds())
        if not isinstance(timestamp, int):
            raise ValueError(
                f"timestamp doit être un int (secondes) ou un timedelta (ici {type(timestamp)})"
            )

        if self.duree < timestamp:
            raise ValueError(
                f"La vidéo ({self.duree:.0f}s) est plus courte que le timestamp ({timestamp:.0f})"
            )
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, timestampo = cap.read()

        cap.release()

        if not ret:
            raise ValueError(f"Impossible de lire la frame {timestamp}")

        frame_rgb = cv2.cvtColor(timestampo, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb, mode='RGB').convert("RGBA")

        return pil_image

    def dumpAtachement(self, dossier: str):
        """
        Extrait tous les fichiers attachés (polices, images, etc.)
        d'un fichier MKV à l'aide de ffmpeg.

        Args:
            dossier (str): Répertoire de destination où les fichiers extraits seront enregistrés.
        """
        vid_abspath= abspath(self.path)
        if which("ffmpeg") is None:
            raise SystemError("ffmpeg n'est pas installé ou n'est pas présent dans le PATH")
        makedirs(dossier, exist_ok=True)
        # Sauvegarde du répertoire courant
        original_cwd = getcwd()
        chdir(dossier)
        commande = [
                "ffmpeg",
                '-y',
                "-dump_attachment:t", "",
                # "" = dump tous les fichiers attachés en gardant leur nom
                "-i", vid_abspath,
                "-t", '0',
                "-f", 'null',
                'null'  # https://superuser.com/a/1828612
            ]
        resultat = subprocess.run(commande, capture_output=True, text=True)
        if resultat.returncode != 0:
            print(
                "Erreur lors de l'extraction des attachments avec ffmpeg :"
                f" \n commande : {' '.join(commande)}\n{resultat.stderr}"
            )
        chdir(original_cwd)
        self.attachement_path = abspath(dossier)

    def prase_file(self, piste: int):
        if piste + 1 > len(self.sous_titres):
            raise ValueError(
                f"La vidéo n'a que {len(self.sous_titres)}"
                f" pistes, impossible d'accéder à la {piste}ième"
            )
        if piste not in self.extracted_sub_path:
            raise ValueError(f"La piste {piste} n'a pas encore été extraite (voir extractSub)")
        with open(self.extracted_sub_path[piste], encoding='utf_8_sig') as f:
            doc = DocumentPlus.parse_file_plus(f)
        self.docs[piste] = doc

    def extractSub(self, piste: int, sortie: str,
                   prase_after: bool = True, codec: str | None = None) -> None:
        """
        Extrait une piste de sous-titres (ASS, SRT) depuis un fichier MKV en utilisant FFmpeg.

        Si le fichier de sortie existe déjà, il est automatiquement écrasé (option -y).
        Un nettoyage est appliqué au fichier extrait pour supprimer d'éventuels caractères null
        (`\\x00`) présents dans certaines pistes.
        Si le codec source est 'subrip' (SRT), une conversion automatique en format ASS est
        effectuée via FFmpeg.

        Args:
            cheminVersMKV (str): Chemin absolu ou relatif vers le fichier vidéo MKV.
            piste (int): Index de la piste de sous-titres à extraire (commence à 0).
            sortie (str): Chemin du fichier de sortie, sans extension. L'extension appropriée
                (.ass ou .srt) sera ajoutée automatiquement.
            codec (str, optional): Codec attendu pour la piste à extraire
                'ass' ou 'subrip'. Par défaut 'ass'.

        Raises:
            SystemError: Si FFmpeg n'est pas installé ou non détecté dans le PATH.
            FileExistsError: Si le fichier MKV source n'existe pas.
            ValueError: Si l'index de piste n'est pas un entier ou si le codec n'est ni 'ass'
                 ni 'subrip'.
            RuntimeError: Si la conversion du format SRT vers ASS échoue.

        Example:
            >>> extractSubFromMKV(
            ...     cheminVersMKV=r"C:\\vids\\anime.mkv",
            ...     piste=1,
            ...     sortie=r"C:\\subs\\anime_episode01",
            ...     codec='subrip'
            ... )
            # → Extrait la piste 1 (SRT), la nettoie, et crée deux fichiers :
            #   - anime_episode01.srt (intermédiaire nettoyé)
            #   - anime_episode01.ass (version convertie en ASS)
        """
        # TODO : mettre a jours documentation
        cheminVersMKV, sortie = abspath(self.path), abspath(sortie)
        if which("ffmpeg") is None:
            raise SystemError("ffmpeg n'est pas installé ou n'est pas présent dans le PATH")
        if not exists(cheminVersMKV):
            raise FileExistsError(f"Le fichier MKV n'existe pas :\n{cheminVersMKV}")
        if piste + 1 > len(self.sous_titres):
            raise ValueError(
                f"La vidéo n'a que {len(self.sous_titres)} sous titres, "
                f"il est donc impossible d'accéder à la {piste}ième"
            )
        if exists(sortie):
            warn(f"La sortie {sortie} existe déjà, elle va être écrasée", UserWarning)
        if not isinstance(piste, int):
            raise ValueError("La piste doit être un entier")
        if codec is None:
            for st in self.sous_titres:
                if st['id_sub'] == piste:
                    codec = st['codec']
        if codec != 'ass' and codec != 'subrip':
            raise ValueError(f"Le codec du sous titre n'est ni ass ni subrip (c'est {codec})")
        mapCodec = {'ass': '.ass', 'subrip': '.srt'}
        sortie += mapCodec[codec]
        makedirs(dirname(sortie), exist_ok=True)
        commande = [
            "ffmpeg",
            "-i", cheminVersMKV.replace('\\', '/'),
            "-map", f"0:s:{piste}",
            "-c:s", "copy",
            "-y",
            sortie.replace('\\', '/')
        ]
        resultat = subprocess.run(commande, capture_output=True, text=True)

        if resultat.returncode != 0:
            print(f'{" ".join(commande)} \n{resultat.stderr}')

        with open(sortie, "rb") as f:  # Problème avec OutlawStar, je ne sais pas pourquoi
            content = f.read().replace(b"\x00", b"")
        with open(sortie, "wb") as f:
            f.write(content)

        if codec == "subrip":
            commande = [
                "ffmpeg",
                "-y",
                "-i", sortie.replace('\\', '/'),
                sortie.replace(".srt", ".ass").replace('\\', '/')
            ]
            resultat = subprocess.run(commande, capture_output=True, text=True)
            sortie = sortie.replace(".srt", ".ass")
            if resultat.returncode != 0:
                raise RuntimeError(
                    f"Le changement de Srt à Ass a échoué \n{' '.join(commande)}\n{resultat.stderr}"
                )

        self.extracted_sub_path[piste] = sortie
        if prase_after:
            self.prase_file(piste=piste)

    def render_frame(
        self, timestamp: int | timedelta, piste: int = 0,
        dump_attachement: bool = False, attachement_path: str | None = None
    ) -> Image.Image:
        """
        Rend une image de la vidéo à un instant donné, avec les sous-titres incrustés (via libass).

        Cette méthode extrait une frame de la vidéo (sans sous-titres),
        puis rend les sous-titres de la piste spécifiée à l'instant donné et les superpose à
        l'image vidéo. Les polices nécessaires sont extraites si besoin.

        Args:
            timestamp (int | timedelta):
                - Si int, temps en secondes depuis le début de la vidéo.
                - Si timedelta, durée depuis le début de la vidéo.
            piste (int, optional): Index de la piste de sous-titres à utiliser (par défaut 0).
            dump_attachement (bool, optional): Si True, extrait les fichiers attachés
                (polices, images) du MKV avant rendu (par défaut False).
            attachement_path (str | None, optional): Répertoire où extraire les fichiers attachés.
                Si None, utilise le dossier courant.

        Raises:
            ValueError: Si le timestamp n'est pas un int ou un timedelta,
                ou si la piste demandée n'est pas extraite.

        Returns:
            PIL.Image.Image: Image RGBA contenant la frame vidéo avec les sous-titres rendus
                à l'instant donné.
        """
        if isinstance(timestamp, int):
            timestamp = timedelta(seconds=timestamp)
        if not isinstance(timestamp, timedelta):
            raise ValueError(f"timestamp doit être un int ou un timedelta (ici {type(timestamp)})")
        if len(self.extracted_sub_path) < piste+1:
            raise ValueError(
                f"La vidéo n'a que {len(self.extracted_sub_path)} pistes extraites,"
                f" impossible d'accéder à la numéro {piste}"
            )

        if dump_attachement:
            self.dumpAtachement(
                dossier=attachement_path if attachement_path is not None else ""
            )
        base = self.extract_frame_as_pil(timestamp)
        SIZE = base.size

        if piste not in self.extracted_sub_path:
            self.extractSub(piste=piste, sortie=f"piste-{piste}", prase_after=True)
        if piste not in self.docs:
            self.prase_file(piste=piste)

        doc = self.docs[piste]
        ctx = RendererClean.Context()
        if self.attachement_path is not None:
            ctx.fonts_dir = self.attachement_path.encode('utf-8')
        r = ctx.make_renderer()
        r.set_fonts(fontconfig_config="\0")
        r.set_all_sizes(SIZE)
        t = ctx.make_track()
        t.populate(doc)
        resultats_libass = r.render_frame(t, timestamp)

        for image in resultats_libass:
            base = Image.alpha_composite(base, image.to_pil(SIZE))
        return base
    
    def get_subtitle_boxes(self,  timestamp: int | timedelta, SIZE: tuple[int, int], renderer: RendererClean.Renderer,
        context: RendererClean.Context, piste: int,
        transform_sub: bool = False, multiline: bool = False,
        padding: tuple[int, int, int, int] = (7, 10, 0, 0)
        )-> eventWithPilList:
        """
        Extrait les boîtes englobantes des sous-titres présents à un instant donné.

        Cette méthode rend les sous-titres de la piste spécifiée à l'instant `timestamp` et
        retourne pour chaque événement (ligne de sous-titre) la boîte englobante correspondante,
        ainsi que l'image PIL du rendu des sous-titres. Elle gère les sous-titres multilignes,
        l'application optionnelle de transformations de style, et le padding autour des boîtes.

        Args:
            timestamp (int | timedelta): Instant de la vidéo pour lequel extraire les sous-titres.
            SIZE (tuple[int, int]): Taille de l'image (largeur, hauteur).
            renderer (RendererClean.Renderer): Objet de rendu libass.
            context (RendererClean.Context): Contexte de rendu libass.
            piste (int): Index de la piste de sous-titres à utiliser.
            transform_sub (bool, optional): Applique une transformation aléatoire au style du sous-titre (par défaut False).
            multiline (bool, optional): Si True, retourne une boîte par événement, sinon une boîte par ligne (par défaut False).
            padding (tuple[int, int, int, int], optional): Padding à appliquer autour des boîtes (par défaut (7, 10, 0, 0)).

        Returns:
            tuple:
                - Liste des événements avec leur boîte englobante.
                - Liste des images PIL correspondant au rendu des sous-titres.
        """
        def splitDialogue(dialogue: Dialogue) -> list[Dialogue]:
            """sépare un dialogue de plusieurs lignes en plusieurs dialogues d'une ligne
            """
            if len(getattr(dialogue, 'text', '').replace('\\N', '\\n').split('\\n')) >1:
                # le dialogue à un passage à la ligne, pour avoir une ligne par box un le coupe
                liste_textes = getattr(dialogue, 'text', '').replace('\\N', '\\n').split('\\n')
                l_dialogues = []
                for t in liste_textes:
                    dialogue_de_ligne = deepcopy(dialogue)
                    dialogue_de_ligne.text = t
                    l_dialogues.append(dialogue_de_ligne)
                return l_dialogues
            else: 
                return [dialogue]
        def trier_boxes_par_position(boxes: list[RendererClean.Box]) -> list[RendererClean.Box]:
            """Permet de trier une liste de boites, avec les plus hautes en premier
            en cas d'équalitées c'est la plus à droite qui est choisie
            """
            def position_cle(box: RendererClean.Box):
                y_min = min(point[1] for point in box.full_box)
                x_min = min(point[0] for point in box.full_box)
                return (y_min, x_min)
            return sorted(boxes, key=position_cle)
        if isinstance(timestamp, int):
            timestamp = timedelta(seconds=timestamp)
        doc = self.docs[piste]
        nb_event_in_frame, events_in_frame = doc.nb_event_dans_frame(timestamp, returnEvents=True)

        if transform_sub and nb_event_in_frame == 1 and events_in_frame[0].style == 'Default':
            # Il n'y a qu'un seul event dans la frame et il a le style par défaut,
            # on peu se permettre de le modifier
            doc = doc.doc_event_donne(events_in_frame[0])
            for i in range(len(doc.styles)):

                # GERER LES STYLES OVERIDES
                if doc.styles[i].name == 'Default':
                    doc.styles[i] = style_transform(doc.styles[i])
                    break

        returnliste = []
        for i, event in enumerate(events_in_frame):
            if nb_event_in_frame > 1:
                # On créé un doc avec un seul élément pour savoir exactement
                # quel évènement donne quelles boxes
                docCopie = doc.doc_event_donne(event)
            else:
                docCopie = doc

            t = context.make_track()
            t.populate(docCopie)
            resultats_libass = renderer.render_frame(t, timestamp)
            if SIZE is not None and SIZE != (0,0):
                biggest_h, biggest_w, smallest_dist_x, smallest_dist_y = 0, 0, 0, 0
            else: # No sizes so create small images
                biggest_h, biggest_w, smallest_dist_x, smallest_dist_y = resultats_libass.get_distances_list()
            PIL = resultats_libass.to_pil(SIZE)
            event_tuple=(PIL,[])
            # TODO : faire un dictonnaire de classe
            if not multiline:
                events_list = splitDialogue(event)
                boxes_list = trier_boxes_par_position(
                    resultats_libass.to_singleline_boxes(padding=padding, xy_offset=(smallest_dist_x, smallest_dist_y))
                )
                if len(events_list) != len(boxes_list):
                    raise ValueError(
                        f"there should be the same number of line than the number of boxes (here {len(events_list)} lines and {len(boxes_list)} boxes"
                    )
                for i in range(len(events_list)):
                    dict_event = {
                        "Event": events_list[i],
                        "Boxes": boxes_list[i]
                    }
                    event_tuple[1].append(FrameToBoxEvent(**dict_event))
            else:
                dict_event = {
                        "Event": event,
                        "Boxes": resultats_libass.to_box(padding=padding, xy_offset=(smallest_dist_x, smallest_dist_y))
                    }
                event_tuple[1].append(
                    FrameToBoxEvent(**dict_event)
                )
            returnliste.append(eventWithPil(image=event_tuple[0], events=event_tuple[1]))
        return eventWithPilList(returnliste)

    def extract_subtitle_boxes_from_frame(
        self,  timestamp: int | timedelta, renderer: RendererClean.Renderer,
        context: RendererClean.Context, piste: int = 0, 
        include_video_background: bool = True, draw_boxes: bool = False,
        annotate_boxes: bool = False, save_image: bool = True, SortieImage: str | None = None,
        transform_sub: bool = False, transform_image: bool = False, multiline: bool = False,
        padding: tuple[int, int, int, int] = (7, 10, 0, 0), dataset: str | None = None
    ) -> tuple[FrameToBoxResult, Image.Image]:
        """
        Extrait les boîtes englobantes des sous-titres à un instant donné d'une vidéo,
        avec option de rendu, d'annotation et de sauvegarde.

        Cette méthode :
            1. Extrait une frame vidéo au timestamp donné.
            2. Rend les sous-titres de la piste spécifiée.
            3. Calcule leurs boîtes englobantes.
            4. (Optionnel) dessine et/ou annote les boîtes sur l'image.
            5. (Optionnel) applique des transformations au style des sous-titres et/ou à l'image.
            6. (Optionnel) sauvegarde l'image finale sur disque.

        Args:
            timestamp (int | timedelta): Instant de la vidéo (en secondes ou timedelta).
            renderer (RendererClean.Renderer): Objet de rendu libass.
            context (RendererClean.Context): Contexte de rendu libass.
            piste (int, optional): Index de la piste de sous-titres à utiliser (par défaut 0).
            include_video_background (bool, optional): Si False, fond transparent (par défaut True).
            draw_boxes (bool, optional): Si True, dessine les boîtes englobantes (par défaut False).
            annotate_boxes (bool, optional): Si True, affiche le texte à côté des boîtes (par défaut False).
            save_image (bool, optional): Sauvegarde l'image finale (par défaut True).
            SortieImage (str | None, optional): Chemin de sauvegarde de l'image. Si None, nom par défaut.
            transform_sub (bool, optional): Applique une transformation de style aléatoire aux sous-titres.
            transform_image (bool, optional): Dégrade l'image de façon réaliste (bruit, flou...).
            multiline (bool, optional): Si True, une boîte par événement ; sinon, une boîte par ligne.
            padding (tuple[int, int, int, int], optional): Marges à appliquer aux boîtes (gauche, haut, droite, bas).
            dataset (str | None, optional): Si fourni, le chemin de l'image dans `FrameToBoxResult`
                sera relatif à ce dossier.

        Returns:
            tuple:
                - FrameToBoxResult: Métadonnées sur l'image et les boîtes extraites.
                - Image.Image: Image PIL finale (avec ou sans annotations).
        
        Raises:
            ValueError: Si le timestamp est invalide ou hors durée.
            ValueError: Si la piste n'est pas parsée (voir `Video.prase_file`).
        """
        if isinstance(timestamp, int):
            timestamp = timedelta(seconds=timestamp)
        if not isinstance(timestamp, timedelta):
            raise ValueError(f"timestamp doit être un int ou un timedelta (ici {type(timestamp)})")
        if piste not in self.docs or self.docs[piste] is None:
            raise ValueError(
                f"La piste {piste} n'est pas convertie en document (voir Video.prase_file)"
            )
        if timestamp > timedelta(seconds=self.duree):
            raise ValueError(
                f"La vidéo ne dure que {self.duree:.0f}, impossible"
                f" d'accéder au timestamp à la {timedelta.total_seconds():.0f}ième seconde"
                )

        if SortieImage is None:
            SortieImage = join(getcwd(), f"frame-{timestamp.total_seconds():.0f}.png")

        base = self.extract_frame_as_pil(timestamp)
        SIZE = base.size
        if not include_video_background:
            # pas le fond vidéo
            base = Image.new(mode='RGBA', size=SIZE, color=(0,0,0,0))
        
        ReturnEventsListe, PILImages = self.get_subtitle_boxes(
            timestamp=timestamp, SIZE=SIZE, renderer=renderer,
            context=context, multiline=multiline, padding=padding,
            piste=piste, transform_sub=transform_sub
        )

        for image in PILImages:
            # on rajoute les sous titres
            base = Image.alpha_composite(base, image)
        del PILImages
        
        if draw_boxes:
            for event in ReturnEventsListe:
                # On rajoute les box
                base = Image.alpha_composite(base, event.Boxes.to_pil(SIZE))
        
        if annotate_boxes: 
            draw = ImageDraw.Draw(base)
            with importlib.resources.path("OCRSub.libs.fonts", "arial.ttf") as font_path:
                font = ImageFont.truetype(font_path, 20)
            for event in ReturnEventsListe:
                # On rajoute les textes
                # A FINIR
                event_box = event.Boxes.full_box
                position=(event_box[0][0], event_box[0][1])
                texte=re.sub(r'\{[^}]*\}', '', event.Event.text.replace('\\N', '\n'))
                draw.multiline_text(position, texte, font=font, fill=(0, 0, 255), spacing=4)

        makedirs(dirname(abspath(SortieImage)), exist_ok=True)
        if transform_image:
            base = degrade_image(base)
        if save_image:
            base.save(abspath(SortieImage))

        dict_result = {
            # On ne retourne pas le abspath pour une meilleure reprobuctibilitée
            "ImagePath": (relpath(abspath(SortieImage), abspath(dataset))
                          if dataset is not None else SortieImage),
            "Events": ReturnEventsListe
        }
        return (FrameToBoxResult(**dict_result), base)

    @classmethod
    def make_video(cls, path_to_mkv: str| Path) -> 'Video':
        """_summary_

        Args:
            path_to_mkv (str): _description_

        Returns:
            Video: _description_
        """
        if isinstance(path_to_mkv, Path):
            path_to_mkv = str(path_to_mkv)
        if isinstance(path_to_mkv, str):
            if not exists(path_to_mkv):
                raise FileNotFoundError(f"Le fichier {path_to_mkv} n'existe pas")
            elif not path_to_mkv.lower().endswith('.mkv'):
                raise ValueError(
                    f"Le fichier {path_to_mkv} existe mais ce n'est pas un fichier mkv"
                    )

        ffprobe_results = cls.recup_infos_MKV(path_to_mkv)

        return cls(
            path=str(path_to_mkv),
            taille=ffprobe_results['taille'],
            duree=ffprobe_results['durée'],
            sous_titres=ffprobe_results['sous_titres']
        )

    @classmethod
    def copy_video(cls, path_to_mkv: str, taille: tuple[int, int], duree: float,
                   sous_titres: list[SubTrackInfo],
                   extracted_sub_path: dict[int, str] | None = None,
                   attachement_path: str | None = None,
                   docs: dict[int, DocumentPlus] | None = None
                   ) -> 'Video':
        episode = cls(
            path=path_to_mkv,
            taille=taille,
            duree=duree,
            sous_titres=sous_titres
        )
        if extracted_sub_path:
            episode.extracted_sub_path = extracted_sub_path
        if attachement_path:
            episode.attachement_path = attachement_path
        if docs:
            episode.docs = docs
        return episode
    
    def choose_sub_track(self, langage: str = 'fre',
                  forbidden_words: list[str] | None = None
                  ) -> tuple[int | None, str | None]:
        """
        Randomly selects the most suitable subtitle track for a given language.

        The function filters available subtitle tracks based on language, codec type,
        and optionally forbidden words in the title. Among valid candidates, it randomly 
        chooses one, giving preference to tracks with higher FPS values (indicating more completeness).

        Args:
            langage (str, optional): Target language code (e.g., `fre` for French). Defaults to `fre`.
            forbidden_words (list[str] | None, optional): Words to exclude from titles. Defaults to a preset list.

        Returns:
            tuple[int | None, str | None]: The selected subtitle track's ID and title, 
            or `(None, None)` if no suitable track is found.
        """
        def choisir_pondere(liste, alpha: float = 0.8):
            """permet de choisir une piste au hasard mais le fps joue un role
            """
            # éviter les fps nuls
            fps_list = [max(0.01, piste.get('fps', 0.01)) ** alpha for piste in liste]
            total = sum(fps_list)
            poids = [fps / total for fps in fps_list]
            piste = random.choices(liste, weights=poids, k=1)[0]
            return piste['id_sub'], piste['title']
        
        sous_titres = self.sous_titres
        mots_interdis = ['canada', 'forced', 'basque', 'sings'] if forbidden_words is None else forbidden_words
        bon_sous_titres = []
        for piste in sous_titres:
            if piste['langage'] == langage and piste['codec'] in ['ass', 'subrip']:
                bon_sous_titres.append(piste)

        if len(bon_sous_titres) == 1:
            return bon_sous_titres[0]['id_sub'], bon_sous_titres[0]['title']

        encore_mieux = []
        for piste in bon_sous_titres:
            # On a trop de pistes francaises, il faut faire un tris
            if (
                piste.get('fps', 0) and  # supprime les pistes avec très peu de subs (piste de DUB)
                not any(mot.lower() in piste.get('title', '').lower() for mot in mots_interdis)
                # supprime les pistes francaises mais pas vraiment
            ):
                encore_mieux.append(piste)

        if len(encore_mieux) == 1:
            # après filtrage on a qu'une seule piste donc on la renvoie
            return encore_mieux[0]['id_sub'], encore_mieux[0]['title']
        elif len(bon_sous_titres) != 0 and len(encore_mieux) == 0:
            # On a supprimé toutes les pistes, il faut choisir au hasard
            return choisir_pondere(bon_sous_titres)
        elif len(encore_mieux) > 1:
            # Malgré les filtrages on a plus de une piste, on choisis au hasard
            return choisir_pondere(encore_mieux)

        return None, None

