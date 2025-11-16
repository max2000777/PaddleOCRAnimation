from os.path import exists, dirname, abspath, join, relpath, basename
from shutil import which, copy
import subprocess
from json import loads
from PIL import Image, ImageDraw, ImageFont
from datetime import timedelta
from os import makedirs, chdir, getcwd, listdir
from warnings import warn
from .sub import RendererClean
from .sub.DocumentPlus import DocumentPlus
from ass import Dialogue
from copy import deepcopy
import random
from platform import system
from ctypes import cdll
import re
import importlib.resources
from pathlib import Path
import logging
from .iso_codes import iso_639_dict
from ..dataset.utilis.disturb import disturb_image, style_transform
from .classes import eventWithPilList, eventWithPil, FrameToBoxEvent, FrameToBoxResult, SubTrackInfo
from ..video.utilis import detect_text_line_boxes
from .sub.box import Box
from warnings import warn

logger = logging.getLogger(__name__)

try:
    system = system()
    if system == "Linux":
        cdll.LoadLibrary("libGL.so.1")
except OSError as e:
    raise ImportError(
        f"{e}\nVous n'avez pas la librairie libGL d'installée veuillez "
        "l'installer en faisant (sur Ubuntu) :\nsudo apt install libgl1"
        )
else:
    import cv2

class NoCorrectSubFound(Exception):
    """used to indicate that no correct sub has been found in a mkv
    """
    pass


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
        """Extracts metadata and subtitle information from an MKV video file.

        This method uses `ffprobe` to analyze the provided MKV file and retrieve 
        technical details such as video resolution, duration, and available subtitle tracks 
        (both embedded and external `.ass` or `.srt` files located in the same directory).

        Args:
            cheminVersMKV (str): Path to the MKV file to analyze.

        Raises:
            SystemError: If `ffprobe` is not installed or not found in the system PATH.
            ValueError: If the provided path does not point to an `.mkv` file.
            FileExistsError: If the specified MKV file does not exist.
            RuntimeError: If `ffprobe` fails to process the file.

        Returns:
            dict: A dictionary containing:
                - 'durée' (float): Duration of the video in seconds.
                - 'taille' (tuple): Video resolution as (width, height).
                - 'sous_titres' (list[dict]): List of subtitle tracks with fields:
                    * 'index' (int | None)
                    * 'id_sub' (int)
                    * 'langage' (str)
                    * 'title' (str)
                    * 'codec' (str)
                    * 'nb_frames' (int | None)
                    * 'fps' (float)
                    * 'is_extarnal' (bool)
        """
        if which("ffprobe") is None:
            raise SystemError("ffprobe n'est pas installé ou n'est pas dans le PATH")
        # if not cheminVersMKV.endswith('.mkv'):
        #     raise ValueError(f'Please provide a path to a mkv file')
        if isinstance(cheminVersMKV, Path):
            cheminVersMKV = str(cheminVersMKV)
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
                    'fps': float(nb_frames)/float(duration) if duration is not None else 0,
                    'is_extarnal':False

                }
                pistes_sous_titres.append(info_sous_titres)
                id_sub += 1
        
        files_in_dir = [
            file for file in listdir(dirname(cheminVersMKV)) if (file.endswith('.ass') or file.endswith('.srt')) and file.startswith(basename(cheminVersMKV)[:-4])
        ]
        for file in files_in_dir:
            # sometimes a sub can be next to the mkv file
            # media players detect thoses by default, we try to recreate that
            regex = re.findall(
                pattern=r"\.([a-z]{2})\.(srt|ass)$",
                string=file
            )
            if len(regex)==1 and len(regex[0])==2 and regex[0][0] in iso_639_dict:
                # a language was decected in the name of the external subfile
                lang = iso_639_dict[regex[0][0]]
            else:
                lang = 'inconnu'
            info_sous_titres = {
                    'index': None,
                    'id_sub': id_sub,
                    'langage': lang,
                    'title': file,
                    'codec': 'ass' if file.endswith('.ass') else 'subrip',
                    'nb_frames': None,
                    'fps': 500, # beacause fps is important, we give a high value
                    'is_extarnal':True
                }
            id_sub+=1
            pistes_sous_titres.append(info_sous_titres)

        return {
            'durée': duration,
            'taille': SIZE,
            'sous_titres': pistes_sous_titres
        }

    def extract_frame_as_pil(self, timestamp: float | timedelta) -> Image.Image:
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
            timestamp = float(timestamp.total_seconds())
        if not (isinstance(timestamp, float) or isinstance(timestamp, int)):
            raise ValueError(
                f"timestamp doit être un float (secondes) ou un timedelta (ici {type(timestamp)})"
            )

        if self.duree < timestamp:
            raise ValueError(
                f"La vidéo ({self.duree:.0f}s) est plus courte que le timestamp ({timestamp:.0f})"
            )
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, timestampo = cap.read()

        cap.release()

        if not ret:
            raise ValueError(f"Impossible de lire la frame {timestamp}")

        frame_rgb = cv2.cvtColor(timestampo, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb, mode='RGB').convert("RGBA")

        del cap, ret, timestampo

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
        resultat = subprocess.run(commande, capture_output=True, text=True, encoding='utf-8')
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

    def extractSub(self, piste: int, sortie: str | None = None,
                   prase_after: bool = True, codec: str | None = None) -> None:
        """Extracts a subtitle track (ASS or SRT) from the MKV video.

    This method uses `ffmpeg` to extract the specified subtitle track, 
    either embedded in the MKV file or as an external file in the same directory.  
    If the extractes file is a `.srt` file, it will be converted to a `.ass` file with ffmpeg.
    The extracted subtitle can optionally be  parsed afterward.

    Args:
        piste (int): Subtitle track ID to extract.
        sortie (str | None, optional): Output path for the extracted subtitle file. 
            Defaults to the MKV filename with the track index appended.
        prase_after (bool, optional): Whether to parse the extracted subtitle file after extraction. Defaults to True.
        codec (str | None, optional): Codec of the subtitle ('ass' or 'subrip'). 
            If None, it is inferred automatically.

    Raises:
        SystemError: If `ffmpeg` is not installed or not found in the PATH.
        FileExistsError: If the MKV file does not exist.
        ValueError: If the subtitle index is invalid or the codec is unsupported.
        RuntimeError: If the extraction or conversion process fails.
    """
        # TODO : mettre a jours documentation
        cheminVersMKV = abspath(self.path)
        if which("ffmpeg") is None:
            raise SystemError("ffmpeg n'est pas installé ou n'est pas présent dans le PATH")
        if not exists(cheminVersMKV):
            raise FileExistsError(f"Le fichier MKV n'existe pas :\n{cheminVersMKV}")
        if piste + 1 > len(self.sous_titres):
            raise ValueError(
                f"La vidéo n'a que {len(self.sous_titres)} sous titres, "
                f"il est donc impossible d'accéder à la {piste}ième"
            )
        is_external, title = False, 'No title'
        if not isinstance(piste, int):
            raise ValueError("La piste doit être un entier")
        for st in self.sous_titres:
            if st['id_sub'] == piste:
                codec = st['codec'] if codec is None else codec
                is_external = st.get('is_extarnal', False)
                title = st.get('title')
        if codec != 'ass' and codec != 'subrip':
            raise ValueError(f"Le codec du sous titre n'est ni ass ni subrip (c'est {codec})")
        mapCodec = {'ass': '.ass', 'subrip': '.srt'}
        if sortie is None:
            sortie = abspath(cheminVersMKV)[:-4]+'_'+str(piste)
        if exists(sortie):
            warn(f"La sortie {sortie} existe déjà, elle va être écrasée", UserWarning)
        sortie += mapCodec[codec]
        makedirs(dirname(sortie), exist_ok=True)
        if is_external:
            # the sub not in the mkv but in the same folder
            copy(
                join(dirname(cheminVersMKV), str(title)),
                sortie
            )
        else: 
            commande = [
                "ffmpeg",
                "-i", cheminVersMKV.replace('\\', '/'),
                "-map", f"0:s:{piste}",
                "-c:s", "copy",
                "-y",
                sortie.replace('\\', '/')
            ]
            resultat = subprocess.run(commande, capture_output=True, text=True, encoding="utf-8")

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
        self, timestamp: float | timedelta, piste: int = 0,
        dump_attachement: bool = False, attachement_path: str | None = None
    ) -> Image.Image:
        """
        Rend une image de la vidéo à un instant donné, avec les sous-titres incrustés (via libass).

        Cette méthode extrait une frame de la vidéo (sans sous-titres),
        puis rend les sous-titres de la piste spécifiée à l'instant donné et les superpose à
        l'image vidéo. Les polices nécessaires sont extraites si besoin.

        Args:
            timestamp (float | timedelta):
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
        if isinstance(timestamp, int) or isinstance(timestamp, float):
            timestamp = timedelta(seconds=timestamp)
        if not isinstance(timestamp, timedelta):
            raise ValueError(f"timestamp doit être un float ou un timedelta (ici {type(timestamp)})")
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
    
    def get_subtitle_boxes(
            self,  
            timestamp: float | timedelta, 
            renderer: RendererClean.Renderer,
            context: RendererClean.Context, piste: int,
            SIZE: tuple[int, int] | None = None,
            multiline: bool = False,
            padding: tuple[int, int, int, int] | tuple[float, float, float, float] = (1, 1, 1, 1),
            use_transparency: bool = True,
        )-> eventWithPilList:
        """
        Render subtitle events at a given timestamp and extract their text bounding boxes.

        This method uses libass to render subtitles from the specified track (`piste`) 
        into a transparent PIL image, then detects text bounding boxes for each subtitle 
        line or block.

        Args:
            timestamp (float | timedelta): Time position in seconds or timedelta.
            renderer (RendererClean.Renderer): The libass renderer used to draw subtitles.
            context (RendererClean.Context): Rendering context for the ASS document.
            piste (int): Subtitle track index in the MKV file.
            SIZE (tuple[int, int] | None, optional): Target render size (width, height). 
                Defaults to the video size.
            multiline (bool, optional): Whether to treat multiline subtitles as one box. 
                Defaults to False.
            padding (tuple[int, int, int, int], optional): Padding around detected boxes 
                (left, top, right, bottom). Defaults to (1, 1, 1, 1).
            use_transparency (bool, optional): Whether to refine box detection using 
                the alpha channel from the rendered image. Defaults to True.

        Returns:
            eventWithPilList: A list-like container where each element contains:
                - `image` (PIL.Image): The rendered subtitle frame.
                - `events` (list[FrameToBoxEvent]): Each containing the event and its box.

        Raises:
            ValueError: If the number of detected boxes does not match the number 
                of subtitle lines when `multiline=False`.

        Notes:
            - When `use_transparency=True`, bounding boxes are refined from the rendered 
            alpha mask instead of relying solely on libass geometry.
            - When multiple subtitle events overlap at the same timestamp, each is 
            processed independently.
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
        def trier_boxes_par_position(boxes: list[Box]) -> list[Box]:
            """Permet de trier une liste de boites, avec les plus hautes en premier
            en cas d'équalitées c'est la plus à droite qui est choisie
            """
            def position_cle(box: Box):
                y_min = min(point[1] for point in box.full_box)
                x_min = min(point[0] for point in box.full_box)
                return (y_min, x_min)
            return sorted(boxes, key=position_cle)

        if isinstance(timestamp, float) or isinstance(timestamp, int):
            timestamp = timedelta(seconds=timestamp)
        if SIZE is None:
            SIZE = self.taille
        doc = self.docs[piste]

        nb_event_in_frame, events_in_frame = doc.nb_event_dans_frame(timestamp, returnEvents=True)

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
                    resultats_libass.to_singleline_boxes(padding=padding if not use_transparency else (0, 0, 0, 0),
                                                         xy_offset=(smallest_dist_x, smallest_dist_y))
                )
                if len(events_list) != len(boxes_list):
                    vid_name = basename(self.path)
                    raise ValueError(
                        f"'{vid_name}', {timestamp} : there should be the same number of line than the number of boxes (here {len(events_list)} lines and {len(boxes_list)} boxes). "
                        "This is most likely due to libass automatic line break when the text is too long"
                    )
                
                if use_transparency:
                    boxes_list = detect_text_line_boxes(PIL, multiline=multiline, libass_box=boxes_list)
                    for i, box in enumerate(boxes_list):
                        w,h = PIL.size
                        box_w, box_h = box[2] - box[0], box[3]-box[1] 
                        if all([isinstance(b, int) for b in padding]):
                            boxes_list[i] = Box(
                                [max(box[0]-padding[0], 0), max(box[1]-padding[1], 0)], [min(box[2]+padding[2], w), max(box[1]-padding[1], 0)],
                                [min(box[2]+padding[2], w), min(box[3]+padding[3], h)], [max(box[0]-padding[0], 0), min(box[3]+padding[3], h)]
                            )
                        elif all([isinstance(b, float) for b in padding]) and all([0<=b<=1 for b in padding]):
                            boxes_list[i] = Box(
                                [max(int(box[0]-box_w * padding[0]), 0), max(int(box[1]-box_h*padding[1]), 0)], 
                                [min(int(box[2]+box_w *padding[2]), w), max(int(box[1]- box_h*padding[1]), 0)],
                                [min(int(box[2]+box_w *padding[2]), w), min(int(box[3]+box_h*padding[3]), h)], 
                                [max(int(box[0]-box_w *padding[0]), 0), min(int(box[3]+box_h*padding[3]), h)]
                            )
                        else:
                            raise ValueError('Padding should be all int or all float between 0 and 1')
                for i in range(len(events_list)):
                    dict_event = {
                        "Event": events_list[i],
                        "Boxes": boxes_list[i]
                    }
                    event_tuple[1].append(FrameToBoxEvent(**dict_event))
            else:
                if use_transparency:
                    box = detect_text_line_boxes(PIL, multiline=multiline)[0]
                    if len(box) != 1:
                        raise ValueError(f"The should be one box, there is only one event and it is multiline")
                    w,h = PIL.size
                    box_w, box_h = box[2] - box[0], box[3]-box[1] 
                    if all([isinstance(b, int) for b in padding]):
                        boxes_list[i] = Box(
                            [max(box[0]-padding[0], 0), max(box[1]-padding[1], 0)], [min(box[2]+padding[2], w), max(box[1]-padding[1], 0)],
                            [min(box[2]+padding[2], w), min(box[3]+padding[3], h)], [max(box[0]-padding[0], 0), min(box[3]+padding[3], h)]
                        )
                    elif all([isinstance(b, float) for b in padding]) and all([0<=b<=1 for b in padding]):
                        boxes_list[i] = Box(
                            [max(int(box[0]-box_w * padding[0]), 0), max(int(box[1]-box_h*padding[1]), 0)], 
                            [min(int(box[2]+box_w *padding[2]), w), max(int(box[1]- box_h*padding[1]), 0)],
                            [min(int(box[2]+box_w *padding[2]), w), min(int(box[3]+box_h*padding[3]), h)], 
                            [max(int(box[0]-box_w *padding[0]), 0), min(int(box[3]+box_h*padding[3]), h)]
                        )
                    else:
                        raise ValueError('Padding should be all int or all float between 0 and 1')
                else:
                    box = resultats_libass.to_box(padding=padding, xy_offset=(smallest_dist_x, smallest_dist_y))

                dict_event = {
                        "Event": event,
                        "Boxes": box
                    }
                event_tuple[1].append(
                    FrameToBoxEvent(**dict_event)
                )
            returnliste.append(eventWithPil(image=event_tuple[0], events=event_tuple[1]))
        return eventWithPilList(returnliste)

    def extract_subtitle_boxes_from_frame(
        self,  timestamp: float | timedelta, renderer: RendererClean.Renderer,
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
            timestamp (float | timedelta): Instant de la vidéo (en secondes ou timedelta).
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
        if isinstance(timestamp, float) or isinstance(timestamp, int):
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
            base = disturb_image(base)
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
                warn(
                    f"The file {path_to_mkv} exists but is not a mkv file"
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
    
    def copy(
            self, timing: float | None = None, 
            doc_id: int | list[int] | None = None
        ):
        if timing is None: 
            return deepcopy(self)
        
        if doc_id is None:
            doc_id = list(self.docs.keys())
        if isinstance(doc_id, int):
            doc_id= [doc_id]
        if not doc_id:
            raise ValueError(f"doc_id is empty")
        
        copy = Video(
            path=self.path,
            taille=self.taille,
            duree=self.duree,
            sous_titres=self.sous_titres.copy()
        )
        copy.attachement_path = self.attachement_path
        
        for doc in doc_id:
            if doc not in self.docs.keys():
                raise ValueError(f'The doc {doc} is not present in the vid docs')
            copy.docs[doc] = self.docs[doc].copy(timing=timing)
        
        return copy

    
    def choose_sub_track(self, langage: str = 'fre',
                  forbidden_words: list[str] | None = None
                  ) -> tuple[int, str]:
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

        raise NoCorrectSubFound(f"Cant find a sub of language {langage} in {self.path}")

