
from os.path import exists, abspath,splitext
import json 
from .detDataset import paddleDataset
import logging
from collections import Counter
from random import shuffle
from os.path import join
from os import getcwd

logger = logging.getLogger(__name__)


class recDataset(paddleDataset):
        """Gestion simple d'un dataset d'images pour reconnaissance de texte avec paddle OCR."""
        @classmethod
        def make_dataset(cls, path: str, val_path: str | None = None) :
            def load_file(path)-> tuple[list, Counter, Counter]:
                with open(path, 'r', encoding='utf-8') as f:
                    images_temp = []
                    c = Counter()
                    c_length = Counter()
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
                        c.update(transcript)
                        c_length[len(transcript)] += 1
                return images_temp, c, c_length
            if not exists(path):
                raise FileNotFoundError(f"Le fichier {abspath(path)} n'existe pas")
            
            images = []
            i, c, c_length = load_file(path)
            images +=  i

            if val_path is not None:
                if not exists(val_path):
                    raise FileNotFoundError(f"Le fichier de validation {abspath(val_path)} n'existe pas")
                i_v, c_v, c_length_v = load_file(val_path)
                images += i_v
                c += c_v
                c_length += c_length_v
            s = cls(path, images)
            s.counter = c
            s.counter_length = c_length
            return s
        
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

        def replace(self, replace_dict: dict[str, str]):
            """Replace substrings in each image transcript using a mapping.

            Iterates over ``self.images`` and applies all (old -> new) replacements
            from ``replace_dict`` to the ``"transcript"`` field of each item.

            Args:
                replace_dict (dict[str, str]): Mapping of substrings to replace
                    (keys) with their replacement values.
             """
            c = Counter()
            for img in self.images:
                t = img.get("transcript", "")
                for old, new in replace_dict.items():
                    t = t.replace(old, new)
                img["transcript"] = t
                c.update(t)
            self.counter = c


            
        def rescrict_length(self, min_length:int = 3, max_length: int = 45):
            """Permet d'enlever les images qui ont un texte trop court.

            Args:
                min_length (int, optional): la taille minimale d'une image acceptable. Par défaut 3.
            """
            new_image_list = []
            num_del_images = 0
            c = Counter()
            c_length = Counter()
            for image in self: 
                transcript = image.get('transcript', '')
                if len(transcript) >= min_length and len(transcript) <= max_length:
                    new_image_list.append(image)
                    c.update(transcript)
                    c_length[len(transcript)] += 1
                else:
                    num_del_images += 1
            
            print(f'removed {num_del_images} images (new length is {len(new_image_list)})')
            self.images = new_image_list
            self.counter = c
            self.counter_length = c_length
        
        def print_text_and_display_image(self, index: int | str):
            print(self[index]['transcript'])
            return self.display_image(index=index)
        
        def print_text_and_display_image(self, index: int | str):
            print(self[index]['transcript'])
            return self.display_image(index=index)

        def save_dataset(
                self, path: str | None = None, train_prop: float = 1,
                train_file_name: str = 'recTrain.txt', 
                test_file_name: str = 'recTest.txt',
                max_train_length: int = 100
            ):
            """Save the dataset to disk as train/test TSV files.

            Shuffles samples, then selects ``int(len(self.images) * train_prop)`` items
            for the train split and writes them to ``train_file_name``. A max transcript
            length filter (``max_train_length``) is applied **only to the train split**
            during writing, so the final number of written train samples (and thus the
            effective train/test proportion) may differ from ``train_prop``.
            If ``train_prop < 1``, the remaining samples are written to ``test_file_name``
            without length filtering. Each line is formatted as:
            ``<image_path>\\t<transcript>\\n``.

            Args:
                path: Output directory. If None, uses the current working directory.
                train_prop: Proportion of samples selected for the train split (0.0 to 1.0).
                train_file_name: Filename for the train split.
                test_file_name: Filename for the test split.
                max_train_length: Maximum transcript length allowed for train samples.
            """
            def write_data(path: str, data: list, max_length: int = 1000):
                with open(path, 'w', encoding='utf-8') as f:
                    for item in data:
                        if len(item['transcript']) <= max_length:
                            line = f"{item['image_path']}\t{item['transcript']}\n"
                            f.write(line)

            if not (0.0 <= train_prop <= 1.0):
                raise ValueError("train_prop must be in [0, 1].")
            path = getcwd() if not path else path
            l= self.images.copy()
            n_train = int(len(l) * train_prop)
            shuffle(l)
            write_data(join(path, train_file_name), l[:n_train], max_length=max_train_length)

            if train_prop != 1:
                write_data(join(path, test_file_name), l[n_train:])



            
            





             
