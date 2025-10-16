from OCRSub.B2_Segmentation.Video import Video, SubTrackInfo
from os.path import abspath, join, splitext, basename, isdir
from os import makedirs, cpu_count, listdir, rmdir, walk
from OCRSub.B2_Segmentation.RendererClean import Context
from numpy import nonzero
from numpy.random import rand
from tqdm.auto import tqdm
from datetime import timedelta
from random import choices
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from sys import gettrace
logging.basicConfig(
    filename='trainingDetection.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def choisis_piste(sous_titres: list[SubTrackInfo], langage: str = 'fre',
                  mots_interdis: list[str] | None = None
                  ) -> tuple[int | None, str | None]:
    def choisir_pondere(liste, alpha: float = 0.8):
        """permet de choisir une piste au hasard mais le fps joue un role
        """
        # éviter les fps nuls
        fps_list = [max(0.01, piste.get('fps', 0.01)) ** alpha for piste in liste]
        total = sum(fps_list)
        poids = [fps / total for fps in fps_list]
        piste = choices(liste, weights=poids, k=1)[0]
        return piste['id_sub'], piste['title']

    mots_interdis = ['canada', 'forced', 'basque'] if mots_interdis is None else mots_interdis
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


def episode_to_dataset(path_to_mkv: str, attachement_path: str = "attachement",
                       sub_path: str = 'subs', piste: int | None = None,
                       proba_capture: float = 0.1, dossier_images: str = 'dataset/images',
                       dossier_dataset: str = 'dataset',
                       padding: tuple[int, int, int, int] = (5, 5, 1, 1)
                       ) -> list[str]:
    nom_episode = splitext(basename(path_to_mkv))[0]
    episode = Video.make_video(path_to_mkv)

    chemin_vers_attachement = abspath(
        join(attachement_path, nom_episode)
        )
    makedirs(chemin_vers_attachement, exist_ok=True)
    episode.dumpAtachement(chemin_vers_attachement)

    makedirs(sub_path, exist_ok=True)
    nom_piste = ''
    if piste is None:
        piste, nom_piste = choisis_piste(episode.sous_titres)
    if piste is None:
        # choisi_piste na pas trouvée de piste
        return []
    episode.extractSub(piste=piste, sortie=join(sub_path, nom_episode))

    ctx = Context()
    ctx.fonts_dir = chemin_vers_attachement.encode('utf-8')
    r = ctx.make_renderer()
    r.set_fonts(fontconfig_config="\0")
    r.set_all_sizes(episode.taille)

    # Pour chaque seconde on tire une pièce pour savoir si on la prend ou pas
    frames = nonzero(
        rand(int(episode.duree)) < proba_capture
    )[0] + 1
    frames = sorted(frames.tolist())

    makedirs(abspath(dossier_images), exist_ok=True)

    transctiptions = []
    for frame in tqdm(frames):
        result = episode.frame_to_box(
            timedelta(seconds=frame),
            context=ctx,
            renderer=r,
            padding=padding,
            piste=piste,
            SortieImage=join(dossier_images, f"{nom_episode}_{nom_piste}_{frame}.png"),
            dataset=dossier_dataset,
            transform_image=True,
            transform_sub=True
        )
        transctiptions.append(result.to_dect_dataset())
    return transctiptions


def process_frame_multi(args):
    # On importe ici pour éviter les problèmes de pickling
    from OCRSub.B2_Segmentation.Video import Video
    from OCRSub.B2_Segmentation.RendererClean import Context
    from os.path import basename, splitext

    (path, taille, duree, sous_titres, extracted_sub_path, attachement_path,
     docs, frame, padding, sortie, dossier_dataset, transform_image, transform_sub,
     piste) = args

    nom_episode = splitext(basename(path))[0]
    episode = Video.copy_video(
        path_to_mkv=path,
        taille=taille,
        duree=duree,
        sous_titres=sous_titres,
        extracted_sub_path=extracted_sub_path,
        attachement_path=attachement_path,
        docs=docs
    )
    try:
        ctx = Context()
        ctx.fonts_dir = attachement_path.encode('utf-8')
        r = ctx.make_renderer()
        r.set_fonts(fontconfig_config="\0")
        r.set_all_sizes(episode.taille)

        # On suppose que les sous-titres sont déjà extraits et parsés
        result = episode.frame_to_box(
            frame,
            context=ctx,
            renderer=r,
            padding=padding,
            SortieImage=sortie,
            dataset=dossier_dataset,
            transform_image=transform_image,
            transform_sub=transform_sub,
            piste=piste
        )
        return result.to_dect_dataset()
    except Exception as e:
        if gettrace() is not None:
            raise  # si on est dans le débuggeur je veux stopper
        logging.error(
            f"Erreur lors du traitement de la frame {frame} de l'épisode '{nom_episode}': {e}"
        )
        return None


def episode_to_dataset_multi(path_to_mkv: str, attachement_path: str = "attachement",
                             sub_path: str = 'subs', piste: int | None = None,
                             proba_capture: float = 0.1, dossier_images: str = 'dataset/images',
                             dossier_dataset: str = 'dataset',
                             padding: tuple[int, int, int, int] = (5, 5, 1, 1)
                             ) -> list[str]:

    nom_episode = splitext(basename(path_to_mkv))[0]
    episode = Video.make_video(path_to_mkv)

    chemin_vers_attachement = abspath(join(attachement_path, nom_episode))
    makedirs(chemin_vers_attachement, exist_ok=True)
    episode.dumpAtachement(chemin_vers_attachement)

    makedirs(sub_path, exist_ok=True)
    nom_piste = ''
    if piste is None:
        piste, nom_piste = choisis_piste(episode.sous_titres)
    if piste is None:
        return []
    episode.extractSub(piste=piste, sortie=join(sub_path, nom_episode))

    frames = nonzero(rand(int(episode.duree)) < proba_capture)[0] + 1
    frames = sorted(frames.tolist())

    makedirs(abspath(dossier_images), exist_ok=True)

    args_list = [
        (
            getattr(episode, "path", None),
            getattr(episode, "taille", None),
            getattr(episode, "duree", None),
            getattr(episode, "sous_titres", None),
            getattr(episode, "extracted_sub_path", None),
            getattr(episode, "attachement_path", None),
            getattr(episode, "docs", None),
            timedelta(seconds=frame),
            padding,
            join(dossier_images, f"{nom_episode}_{nom_piste}_{frame}.png"),
            dossier_dataset,
            True,
            True,
            piste
        )
        for frame in frames
    ]

    transcriptions = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(process_frame_multi, args) for args in args_list]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{nom_episode} {nom_piste}",
            leave=False
        ):
            result = f.result()
            if result is not None:
                transcriptions.append(result)

    if not listdir(chemin_vers_attachement):
        # le dossier attachement est vide, pas besoin de le garder
        rmdir(chemin_vers_attachement)

    return transcriptions


def dossier_to_dataset(
        chemin_vers_dossier: str,
        attachement_path: str = 'attachement',
        sub_path: str = 'subs',
        proba_capture: float = 0.1,
        dossier_images: str = 'dataset/images',
        dossier_dataset: str = 'dataset',
        padding: tuple[int, int, int, int] = (5, 5, 1, 1)
):
    if not isdir(chemin_vers_dossier):
        raise FileExistsError(f"Le dossier n'existe pas\n {abspath(chemin_vers_dossier)}")
    fichiers_mkv = []
    for racine, dossiers, fichiers in walk(chemin_vers_dossier):
        for fichier in fichiers:
            if fichier.lower().endswith(".mkv"):
                chemin_complet = join(racine, fichier)
                fichiers_mkv.append(chemin_complet)

    dataset_transcription = []
    for fichier in tqdm(fichiers_mkv, desc="Traitement des fichiers MKV"):
        transcriptions = episode_to_dataset_multi(
            path_to_mkv=fichier,
            attachement_path=attachement_path,
            sub_path=sub_path,
            proba_capture=proba_capture,
            dossier_images=dossier_images,
            dossier_dataset=dossier_dataset,
            padding=padding,
        )
        dataset_transcription += transcriptions

    with open(join(abspath(dossier_dataset), 'dataset.txt'), "w", encoding="utf-8") as f:
        for ligne in dataset_transcription:
            if isinstance(ligne, str):
                f.write(ligne + "\n")


if __name__ == '__main__':
    episode_to_dataset(
        path_to_mkv=r"D:\Téléchargement\Dataser\Mobile Suit Gundam - Pack - GundamGuy\Turn A Gundam"
                    r"\Turn A Gundam (1999) - 45 FANSUB VOSTFR BDrip 1080p FLAC x265-GundamGuy.mkv",
    )
    # dossier_to_dataset(
    #     chemin_vers_dossier=r"D:\Téléchargement\Dataser\Monogatari (BD 1080p)",
    #     proba_capture=0.02,
    # )
