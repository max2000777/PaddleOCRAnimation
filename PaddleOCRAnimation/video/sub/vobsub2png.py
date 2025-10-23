import subprocess
from os.path import exists, dirname, abspath, join
import importlib.resources

from platform import system


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


if __name__ == "__main__":
    ep_name = f"/home/maxim/code/SubProject/OCRSub/examples/data/Subs/A1_t00_track3_[fre].idx"
    vobsub2png(
        idx_path=ep_name
    )
