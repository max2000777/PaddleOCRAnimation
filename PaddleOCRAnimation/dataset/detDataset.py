import json
from os.path import exists, join, abspath, dirname, basename, splitext
from random import shuffle
from PIL import Image as PILImage
from .RendererClean import Box
from os import makedirs
from tqdm.auto import tqdm

class paddleDataset:
    def __init__(self, path: str, images: list[str]):
        """
        Initialise le dataset.

        Args:
            path (str): Chemin du fichier source listant les images et annotations.
            images (list[dict]): Liste de dicts {'image_path': ..., 'annotations': ...}.
        """
        self.path:str = path
        self.images:list = images
        self.length:int = len(images)

    def __len__(self):
        """Retourne le nombre total d'entrées dans le dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """
        Accède à une entrée par index.

        Args:
            index (int): Position de l'image à récupérer.

        Returns:
            dict: Entrée correspondante avec 'image_path' et 'annotations'.

        Raises:
            IndexError: Si l'index est hors bornes.
        """
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
    
    def display_image(self, index: int) -> PILImage.Image:
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
        trainName: str = 'train.txt', testName: str = 'test.txt'
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
        def load_file(path) -> list:
            images_temp=[]
            with open(path, 'r', encoding='utf-8') as f:
                i=0
                for line in f:
                    i+=1
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

                    images_temp.append({
                        'image_path': img_path,
                        'annotations': annotations
                    })
                print(i)
            return images_temp

        if not exists(path):
            raise FileNotFoundError(f"Le fichier {abspath(path)} n'existe pas")

        images = []
        images += load_file(path)

        if val_path is not None:
            if not exists(val_path):
                raise FileNotFoundError(f"Le fichier de validation {abspath(val_path)} n'existe pas")
            
            images += load_file(val_path)

        return cls(path, images)
    
    def renderImageWithBox(self, item: int):
        item_dict = self[item]

        item_image = join(dirname(self.path), item_dict.get('image_path', None))
        if item_image is None:
            raise ValueError('The format is invalid, the item should have a \'image_path\' attribute')
        if not exists(item_image):
            raise FileNotFoundError(f'The image {item_image} does not exist')
        base = PILImage.open(item_image)
        SIZE = base.size

        item_annotations = item_dict.get('annotations', None)
        if item_annotations is None: 
            raise ValueError("item should have a 'annotations' item")
        if not isinstance(item_annotations, list):
            raise ValueError(f"annotations should be a list, here {type(item_annotations)}")
        
        for annotation in item_annotations:
            box = annotation.get('points', None)
            if box is None: 
                raise ValueError("every annotation should have a 'points' item")
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(el, list) and len(el)==2 for el in box):
                raise ValueError('The format of a box should be a list of for list, each containing 2 int')
            
            box = Box(box[0], box[1], box[2], box[3])
            base = PILImage.alpha_composite(base, box.to_pil(SIZE))

        return base
    
    def to_rec_dataset(
            self, foldername: str | None = None,
            txt_name: str | None = None, 
            traintestsplit: float | None = None,
            val_txt_name: str = 'rec_val.txt'
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
            val_txt_name (str, optional): Nom du fichier texte de validation. Par défaut 'rec_val.txt'.

        Raises:
            ValueError: Si `traintestsplit` est hors de [0, 1].
            ValueError: Si une annotation ne contient pas la clé 'points'.
    """
        images_not_exist = self.verify_images()
        if traintestsplit is not None and not (traintestsplit<=1 and traintestsplit>=0):
            raise ValueError(f"traintestsplit should be a float between 0 and 1 (here {traintestsplit})")
        if foldername is not None:
            makedirs(
                        join(dirname(self.path), foldername),
                        exist_ok=True
                    )
        rec_text_list=[]
        for image in tqdm(self.images, desc="Images creation"):
            if image.get('image_path', None) is None or image.get('image_path', None) in images_not_exist:
                continue

            img = PILImage.open(join(dirname(self.path), image['image_path']))

            for i, annotation in enumerate(image.get('annotations', [])):
                def normalize_box(
                        box: list[list[int, int], list[int, int], list[int, int], list[int, int]]
                    ) -> tuple[int, int, int, int]:
                    left = min(p[0] for p in box)
                    top = min(p[1] for p in box)
                    right = max(p[0] for p in box)
                    bottom = max(p[1] for p in box)
                    return (left, top, right, bottom)
                if 'points' not in annotation:
                    raise ValueError("'points' not present in dict")
                
                crop = img.crop(normalize_box(annotation['points']))

                rel_path = join(
                    foldername,
                    f"{splitext(basename(image['image_path']))[0]}_{i}{splitext(basename(image['image_path']))[1]}"
                ) if foldername is not None else f"{splitext(basename(image['image_path']))[0]}_{i}{splitext(basename(image['image_path']))[1]}"

                rec_text_list.append(f"{rel_path}\t{annotation['transcription']}")

                crop.save(join(dirname(self.path), rel_path))
        if traintestsplit is None:
            with open(join(dirname(self.path), "rec.txt" if not txt_name else txt_name), 'w', encoding="utf-8") as f:
                f.write('\n'.join(rec_text_list))
        else: 
            shuffle(rec_text_list)
            with open(join(dirname(self.path), "rec_train.txt" if not txt_name else txt_name), 'w', encoding="utf-8") as f:
                f.write('\n'.join(rec_text_list[:int(traintestsplit*len(rec_text_list))]))

            with open(join(dirname(self.path),val_txt_name), 'w', encoding="utf-8") as f:
                f.write('\n'.join(rec_text_list[int(traintestsplit*len(rec_text_list)):]))

    def save_dataframe(self, path: str, data):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                transcriptions = [json.dumps(event, ensure_ascii=False) for event in item['annotations']]
                line = f"{item['image_path']}\t[{', '.join(transcriptions)}]\n"
                f.write(line)


