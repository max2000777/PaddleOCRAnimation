import json
from os.path import exists, join, abspath, dirname, basename, splitext
from random import shuffle
from PIL import Image as PILImage
from ..video.sub.RendererClean import Box
from os import makedirs, environ
from tqdm.auto import tqdm
import re
from pathlib import Path
import warnings
import random
from numpy.random import normal
from collections import Counter
environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
from paddlex.inference.pipelines.components.common import CropByPolys
from paddlex.inference.common.reader.image_reader import ReadImage
import numpy as np
from typing import Iterable, Mapping, Optional

class paddleDataset:
    def __init__(self, path: str, images: list[dict]):
        """
        Initialise le dataset.

        Args:
            path (str): Chemin du fichier source listant les images et annotations.
            images (list[dict]): Liste de dicts {'image_path': ..., 'annotations': ...}.
        """
        self.path:str = path
        self.images:list[dict] = images
        self.length:int = len(images)

        
        self.name_dict: dict[str, int] = {}
        for i, image in enumerate(images):
            if image.get('image_path', None) is None:
                raise ValueError(f'The {i} image of the dataset does not have a image_path')
            if image['image_path'] in self.name_dict.keys():
                raise ValueError(f"The image {image['image_path']} appears multiple times, this should not be possible")
            self.name_dict[image['image_path']] = i

    def __len__(self):
        """Retourne le nombre total d'entrées dans le dataset."""
        return len(self.images)

    def __getitem__(self, index: int | str):
        """
        Accède à une entrée par index.

        Args:
            index (int): Position de l'image à récupérer ou son nom.

        Returns:
            dict: Entrée correspondante avec 'image_path' et 'annotations'.

        Raises:
            IndexError: Si l'index est hors bornes.
        """
        if isinstance(index, str):
            if index not in self.name_dict.keys():
                raise ValueError(f'The image {index} is not in the dataset')
            index = self.name_dict[index]
        if index < 0 or index >= len(self):
            raise IndexError(f'Out of bound (len {len(self)})')
        return self.images[index]

    def verify_images(self) -> list[str]:
        """
        Vérifie l'existence des fichiers image référencés.

        Returns:
            list[str]: Liste des chemins d'images manquantes.
        """
        missing = []
        for image in self.images:
            if not exists(join(dirname(self.path), image['image_path'])):
                missing.append(image['image_path'])

        return missing
    
    def display_image(self, index: int | str) -> PILImage.Image:
        if isinstance(index, str):
            if not index in self.name_dict.keys():
                raise ValueError(f'The image name {index} is not in the dataset')
            index = self.name_dict[index]
        if index <0 or index > self.length-1:
            raise IndexError(f"Out of range (dataset is length {self.length})")
        image_path = self[index]['image_path']
        text_file_path = join(dirname(self.path), image_path)
        if not exists(text_file_path):
            raise FileNotFoundError(f"The image {text_file_path} was not found")

        return PILImage.open(text_file_path) 
    def save_dataframe(self, path: str, data):
        ... # should be defined by subclass
    def makeTrainTest(
        self, trainProp: float = 0.8,
        trainName: str = 'detTrain.txt', testName: str = 'detTest.txt'
    ):
        """
        Sépare le dataset en train et test et écrit deux fichiers.

        Args:
            trainProp (float, optional): Proportion d'exemples pour l'entraînement. Par défaut 0.8.
            trainName (str, optional): Nom du fichier d'entraînement généré. Par défaut 'train.txt'.
            testName (str, optional): Nom du fichier de test généré. Par défaut 'test.txt'.
        """


        missing = self.verify_images()

        real_images = [image for image in self.images if image['image_path'] not in missing]

        shuffle(real_images)

        split_index = int(len(real_images) * trainProp)
        train_data = real_images[:split_index]
        test_data = real_images[split_index:]

        base_dir = dirname(self.path)
        self.save_dataframe(join(base_dir, trainName), train_data)
        self.save_dataframe(join(base_dir, testName), test_data)

class detDataset(paddleDataset):
    """Gestion simple d'un dataset d'images pour détéction de texte avec paddle OCR."""

    @classmethod
    def make_dataset(cls, path: str, val_path: str | None = None):
        def load_file(path) -> list[dict]:
            images_temp=[]
            name_dict = {}
            with open(path, 'r', encoding='utf-8') as f:
                i=0
                for line in f:
                    
                    line = line.strip()
                    if not line:
                        continue
                    # séparation chemin et annotations
                    try:
                        img_path, ann_json = line.split('\t', 1)
                    except ValueError:
                        # ligne mal formée (pas de tab)
                        continue

                    # parse annotations JSON
                    try:
                        annotations = json.loads(ann_json)
                    except json.JSONDecodeError:
                        annotations = []
                    
                    if img_path in name_dict.keys():
                        warnings.warn(f'The image path {img_path} appears multiple times')
                        continue

                    images_temp.append({
                        'image_path': img_path,
                        'annotations': annotations
                    })
                    i+=1
                print(i)
            return images_temp
        
        def create_counters(images: list[dict], path_to_dataset: str):
            percentage_counter = Counter()
            w_counter = Counter()
            for image in images:
                if not exists(join(path_to_dataset,image['image_path'])):
                    continue
                with PILImage.open(join(path_to_dataset,image['image_path'])) as img:
                    w, h = img.size
                for transcription in image['annotations']:
                    tran_w = transcription['points'][1][0]-transcription['points'][0][0]
                    percentage_counter.update([int(tran_w*100/w)])
                    w_counter.update([tran_w])
            return percentage_counter, w_counter
                

        if not exists(path):
            raise FileNotFoundError(f"Le fichier {abspath(path)} n'existe pas")

        images = []
        images += load_file(path)

        if val_path is not None:
            if not exists(val_path):
                raise FileNotFoundError(f"Le fichier de validation {abspath(val_path)} n'existe pas")
            
            images += load_file(val_path)
        dataset = cls(path, images)
        dataset.per_counter, dataset.w_counter = create_counters(images=images, path_to_dataset=dirname(path))
        return dataset
    
    def renderImageWithBox(self, item: int | str, use_baseline_box:bool = False):
        item_dict = self[item]

        item_image = join(dirname(self.path), item_dict.get('image_path', None))
        if item_image is None:
            raise ValueError('The format is invalid, the item should have a \'image_path\' attribute')
        if not exists(item_image):
            raise FileNotFoundError(f'The image {item_image} does not exist')
        base = PILImage.open(item_image).convert('RGBA')
        SIZE = base.size

        item_annotations = item_dict.get('annotations', None)
        if item_annotations is None: 
            raise ValueError("item should have a 'annotations' item")
        if not isinstance(item_annotations, list):
            raise ValueError(f"annotations should be a list, here {type(item_annotations)}")
        
        for annotation in item_annotations:
            box = annotation.get('points', None) if not use_baseline_box else annotation.get('baseline_points', None)
            if box is None: 
                raise ValueError("every annotation should have a 'points' item")
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(el, list) and len(el)==2 for el in box):
                raise ValueError('The format of a box should be a list of for list, each containing 2 int')
            
            box = Box(box[0], box[1], box[2], box[3])
            base = PILImage.alpha_composite(base, box.to_pil(SIZE))

        return base
    
    def verify_dataset(self):
        def is_valid_box_structure(x: list[list]) -> bool:
            return (
                isinstance(x, list)
                and len(x) == 4
                and all(
                    isinstance(sub, list)
                    and len(sub) == 2
                    and all(isinstance(i, int) for i in sub)
                    for sub in x
                )
            )
        
        def is_valid_box_content(x: list[list], size: tuple[int, int]) -> bool:
            width, height = size
            for (px, py) in x:
                if not (0 <= px <= width and 0 <= py <= height):
                    return False
            return True

        missing_images = self.verify_images()
        if len(missing_images)>0 :
            #some images are missing
            raise FileNotFoundError(f"{len(missing_images)} images are missing")
        
        for i, line in enumerate(self):
            if 'annotations' not in line.keys() or 'image_path' not in line.keys():
                raise ValueError(f'each line should have the keys annotations, image_path. the {i} line doesnt')
            image = PILImage.open(join(dirname(self.path), line['image_path']))

            for y, annotation in enumerate(line['annotations']):
                if 'transcription' not in annotation.keys() or 'points' not in annotation.keys():
                    raise ValueError('Each annotation should have the keys transcription, points. '
                                     f'The {y} annotation of the {i} line doesnt')
                
                if not is_valid_box_structure(annotation['points']):
                    raise ValueError('Each box sould be a list of 4 list each with 2 int. '
                                     f'The {y} box of the {i} line is not')
                
                if not is_valid_box_content(annotation['points'], image.size):
                    raise ValueError(f'the box {line["image_path"]} is not standard format : {annotation["points"]}; img size : {image.size}')
    def replace(self, text_dict: dict[str, str]):
        for image in self:
            for annotation in image['annotations']:
                for replace in text_dict.keys():
                    annotation['transcription'] = annotation['transcription'].replace(replace, text_dict[replace])

    def to_rec_dataset(
            self, foldername: str | None = None,
            txt_name: str | None = None, 
            traintestsplit: float | None = None,
            # p_random_tilt: float = 0.08,
            # p_random_padding: float = 0.07,
            val_txt_name: str = 'recTest.txt',
            max_text_length: int = 150,
            min_text_length: int = 1,
            baseline_p: float = 0.7,
        )-> None:
        """
        Génère un dataset pour la reconnaissance de texte à partir des annotations existantes.

        Les zones de texte annotées sont recadrées et sauvegardées en images, et un fichier
        texte d'indexation est créé (format : `chemin_image<TAB>transcription`).
        Peut également séparer en ensembles d'entraînement et de validation.

        Args:
            foldername (str | None, optional): Nom du dossier où sauvegarder les crops.
                                            Si None, ils sont créés à la racine du dataset.
            txt_name (str | None, optional): Nom du fichier texte principal. Par défaut "rec.txt" 
                                            ou "rec_train.txt" si `traintestsplit` est défini.
            traintestsplit (float | None, optional): Proportion (0-1) des données pour l'entraînement.
                                                    Si None, aucun split n'est effectué.
            val_txt_name (str, optional): Nom du fichier texte de validation. Par défaut `recTrain.txt`.
            max_text_length (int, optional): TODO
            min_text_length (int, optional): TODO

        Raises:
            ValueError: Si `traintestsplit` est hors de [0, 1].
            ValueError: Si une annotation ne contient pas la clé 'points'.
    """
        def randomize_box(
                box: list[list[int]],
                img_shape: tuple[int, int, int] | None = None,
                p_random_padding: float = 0.35
            )-> list[list[int]]:
            new_box = [pt[:] for pt in box]  # copie “profonde” suffisante ici
            for point in new_box:
                for i in range(2):
                    if random.random() < p_random_padding:
                        point[i] += random.choices([3, 2, 1, -1, -2, -3], weights=[5, 10, 15, 15, 10, 5], k=1)[0]
            
            if img_shape is not None: 
                h, w, _ = img_shape
                new_box = [
                    [
                        max(0, new_box[0][0]),
                        max(0, new_box[0][1])
                    ],
                    [
                        min(w, new_box[1][0]),
                        max(0, new_box[1][1])
                    ],
                    [
                        min(w, new_box[2][0]),
                        min(h, new_box[2][1])
                    ],
                    [
                        max(0, new_box[3][0]),
                        min(h, new_box[3][1])
                    ]
                ]
            return new_box
        
        def simulate_tilt(
                box:list[list[int]], text:str, p_tilt: float=0.2, center_band: float = 0.08,
                max_pct: float = 0.20, min_pct: float = 0.03,
            )-> list[list[int]]:
            def descender_center_of_mass(
                text: str,
                hard_chars: Iterable[str],
                *,
                weights: Optional[Mapping[str, float]] = None,
                ignore_whitespace: bool = True,
                casefold: bool = True,
            ) -> float:
                """
                Centre de masse (mu) des 'descenders' dans le texte, sur l'axe horizontal normalisé [0, 1].

                - mu proche de 0   => descenders plutôt à gauche
                - mu proche de 1   => descenders plutôt à droite
                - mu ~ 0.5         => descenders plutôt au centre (côté peu déterminé)

                Retourne None s'il n'y a aucun hard_char dans le texte.
                """
                hard_set = set(hard_chars)
                wmap = weights or {}

                s = text.casefold() if casefold else text
                if ignore_whitespace:
                    s = "".join(ch for ch in s if not ch.isspace())

                n = len(s)
                if n == 0:
                    return 0

                sum_w = 0.0
                sum_wx = 0.0

                for i, ch in enumerate(s):
                    if ch not in hard_set:
                        continue
                    w = float(wmap.get(ch, 1.0))
                    x = (i + 0.5) / n  # position normalisée dans (0, 1)
                    sum_w += w
                    sum_wx += w * x

                return 0 if sum_w == 0.0 else (sum_wx / sum_w)
            hard_tilt_chars = ['y', 'g', 'j', 'p', 'q', 'ç']
            count_hard_tilt_chars = sum(1 for ch in text if ch in hard_tilt_chars)

            if count_hard_tilt_chars ==0:
                # no need to simulate tilt
                return box
            
            if count_hard_tilt_chars > 1:
                # TODO :  might be wrong if they are on the same side
                return box
            
            if random.random() > p_tilt:
                return box
            
            mu = descender_center_of_mass(text, ['y','g','j','p','q','ç','Ç'])
            
            d = mu - 0.5
            if abs(d) <= center_band:
                sign = random.choice([-1.0, 1.0])
                t = 0.0
            else:
                sign = 1.0 if d > 0 else -1.0
                t = (abs(d) - center_band) / (0.5 - center_band)

            mag = min_pct + (max_pct - min_pct) * (t ** 1.2)

            left_pct  = +mag if sign < 0 else -mag
            right_pct = -mag if sign < 0 else +mag

            box[2][1] += box[2][1]*right_pct
            box[3][1] += box[3][1]*left_pct

            return box

        images_not_exist = self.verify_images()
        if traintestsplit is not None and not (traintestsplit<=1 and traintestsplit>=0):
            raise ValueError(f"traintestsplit should be a float between 0 and 1 (here {traintestsplit})")
        if foldername is not None:
            makedirs(
                        join(dirname(self.path), foldername),
                        exist_ok=True
                    )
        rec_text_list=[]
        crop_maker = CropByPolys(det_box_type="quad")
        im_reader= ReadImage(format="BGR")

        existing_images = [image for image in self.images if image.get('image_path', None) not in images_not_exist]
        

        for image in tqdm(existing_images, desc="Images creation"):

            img = im_reader([join(dirname(self.path), image['image_path'])])[0]

            for i, annotation in enumerate(image.get('annotations', [])):
                if len(annotation['transcription']) > max_text_length or len(annotation['transcription']) < min_text_length:
                    continue

                if 'points' not in annotation:
                    raise ValueError("'points' not present in dict")
                
                if 'baseline_points' in annotation and random.random() < baseline_p:
                    points=[randomize_box(annotation['baseline_points'], img_shape = img.shape)]
                else: 
                    points = [randomize_box(simulate_tilt(annotation['points'], text=annotation['transcription']), img_shape = img.shape)]
                try:
                    crop = crop_maker(img, points)
                    crop = PILImage.fromarray(crop[0])
                except ValueError as e:
                    print(f'error for {basename(image["image_path"])}: {e}')
                    continue


                rel_path = join(
                    foldername,
                    f"{splitext(basename(image['image_path']))[0]}_{i}{splitext(basename(image['image_path']))[1]}"
                ) if foldername is not None else f"{splitext(basename(image['image_path']))[0]}_{i}{splitext(basename(image['image_path']))[1]}"

                text = annotation['transcription']
                text = re.sub(r'\{.*?\}', '', text)
                
                try:
                    crop.save(join(dirname(self.path), rel_path))
                except SystemError as e:
                    print(f'error for {basename(image["image_path"])}: {e}')
                    continue
                else:
                    rec_text_list.append(f"{Path(rel_path).as_posix()}\t{text}")
        if traintestsplit is None:
            with open(join(dirname(self.path), "rec.txt" if not txt_name else txt_name), 'w', encoding="utf-8") as f:
                f.write('\n'.join(rec_text_list))
        else: 
            shuffle(rec_text_list)
            with open(join(dirname(self.path), "recTrain.txt" if not txt_name else txt_name), 'w', encoding="utf-8") as f:
                f.write('\n'.join(rec_text_list[:int(traintestsplit*len(rec_text_list))]))

            with open(join(dirname(self.path),val_txt_name), 'w', encoding="utf-8") as f:
                f.write('\n'.join(rec_text_list[int(traintestsplit*len(rec_text_list)):]))

    def save_dataframe(self, path: str, data):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                transcriptions = [json.dumps(event, ensure_ascii=False) for event in item['annotations']]
                line = f"{item['image_path']}\t[{', '.join(transcriptions)}]\n"
                f.write(line)


