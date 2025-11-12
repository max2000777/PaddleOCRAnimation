
from os.path import exists, abspath,splitext
import json 
from .detDataset import paddleDataset
import logging

logger = logging.getLogger(__name__)


class recDataset(paddleDataset):
        """Gestion simple d'un dataset d'images pour reconnaissance de texte avec paddle OCR."""
        @classmethod
        def make_dataset(cls, path: str, val_path: str | None = None) :
            def load_file(path)-> list:
                with open(path, 'r', encoding='utf-8') as f:
                    images_temp = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            img_path, transcript = line.split('\t', 1)
                        except ValueError:
                            # ligne mal formée (pas de tab)
                            continue


                        images_temp.append({
                            'image_path': img_path,
                            'transcript': transcript
                        })
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
        
        def verify_dictionnary(self, dictionnaryPath: str, include_space: bool = True) -> set:
            if not exists(dictionnaryPath):
                raise FileNotFoundError(f"The dictionary was not found on disc {abspath(dictionnaryPath)}")
            if splitext(dictionnaryPath)[1] != ".txt":
                raise ValueError(f"The dictionary should be a .txt file (here {splitext(dictionnaryPath)[1]})")
             
            dict_chars = set(open(dictionnaryPath,'r', encoding="utf-8").read().splitlines())
            if include_space:
                dict_chars = dict_chars | set(' ')
            images_not_present = self.verify_images()

            dataset_chars = set()
            for image in self:
                if image['image_path'] in images_not_present:
                    continue

                image_chars = set(image.get('transcript', ''))
                dataset_chars = dataset_chars | image_chars
            
            chars_not_in_dict = dataset_chars - dict_chars

            return chars_not_in_dict
        
        def rescrict_length(self, min_length:int = 3, max_length: int = 45):
            """Permet d'enlever les images qui ont un texte trop court.

            Args:
                min_length (int, optional): la taille minimale d'une image acceptable. Par défaut 3.
            """
            new_image_list = []
            num_del_images = 0
            for image in self: 
                if len(image.get('transcript', '')) >= min_length and len(image.get('transcript', '')) <= max_length:
                    new_image_list.append(image)
                else:
                    num_del_images += 1
            
            print(f'removed {num_del_images} images (new length is {len(new_image_list)})')
            self.images = new_image_list
        
        def print_text_and_display_image(self, index: int | str):
            print(self[index]['transcript'])
            return self.display_image(index=index)

        def save_dataframe(self, path: str, data):
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    transcriptions = [json.dumps(event, ensure_ascii=False) for event in item['transcript']]
                    line = f"{item['image_path']}\t{item['transcript']}\n"
                    f.write(line)





             
