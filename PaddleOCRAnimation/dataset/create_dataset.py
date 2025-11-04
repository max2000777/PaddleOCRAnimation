from tqdm.auto import tqdm
import logging
from pathlib import Path
from .utilis.create_dataset_utilis import video_to_dataset, redirect_c_stdout_to_logger, dataset_metadata_after, dataset_metadata_before
import os
from typing import Literal
from datetime import datetime
import gc

logger = logging.getLogger(__name__)


def path_check(
    main_mkv_path: str | Path,
    dataset_path: str | Path | None = None,
    image_save_path: str | Path | None = None,
    no_text_image_save_path: str | Path | None = None,
    attachement_path: str | Path | None = None,
    extracted_sub_path: str | Path | None = None,
    separate_text_and_no_text_images: bool = True,
) -> tuple[str, str, str, str, str, str]:
    """
    Validates and initializes all necessary directories for dataset generation.

    This function checks that the root MKV directory exists and then prepares
    a standardized folder structure for dataset creation. It ensures that paths
    for images, subtitles, and attachments are properly defined and created.

    If some paths are not explicitly provided, they are automatically generated
    relative to a default `dataset/` directory in the current working directory.

    Args:
        main_mkv_path (str | Path):
            Path to the root directory containing MKV videos (and possibly subdirectories).
        dataset_path (str | Path | None, optional):
            Root directory for the dataset. Defaults to `./dataset` if not specified.
        image_save_path (str | Path | None, optional):
            Directory where generated images (with or without subtitles) will be stored.
            Defaults to `<dataset_path>/images`.
        no_text_image_save_path (str | Path | None, optional):
            Directory for images without subtitles (background-only).
            Automatically derived from `image_save_path` if not specified.
        attachement_path (str | Path | None, optional):
            Directory where extracted video attachments (e.g., fonts) will be stored.
            Defaults to `<dataset_path>/attachements`.
        extracted_sub_path (str | Path | None, optional):
            Directory where extracted subtitle files will be stored.
            Defaults to `<dataset_path>/extracted_subs`.
        separate_text_and_no_text_images (bool, optional):
            If True, creates separate folders for `text/` and `no_text/` images
            inside the images root directory. If False, both are saved in the same folder.
            Defaults to True.

    Raises:
        FileNotFoundError:
            If `main_mkv_path` does not exist.
        ValueError:
            If `main_mkv_path` exists but is not a directory.

    Returns:
        tuple[str, str, str, str, str, str]:
            A tuple of absolute string paths in the following order:
            `(main_mkv_path, image_save_path, no_text_image_save_path, dataset_path, attachement_path, extracted_sub_path)`.

    Side Effects:
        - Creates all necessary directories if they do not already exist.
        - Standardizes and returns all paths as strings.
    """
    main_mkv_path = Path(main_mkv_path)
    if not main_mkv_path.exists():
        raise FileNotFoundError(f"{main_mkv_path.absolute()} does not exist")
    if not main_mkv_path.is_dir():
        raise ValueError(f"{main_mkv_path.absolute()} is not a directory")

    if dataset_path is None:
        dataset_path = Path(os.getcwd()) / "dataset"
    else:
        dataset_path = Path(dataset_path)

    images_root = Path(image_save_path) if image_save_path else dataset_path / "images"

    if separate_text_and_no_text_images:
        image_save_path = images_root / "text"
        no_text_image_save_path = images_root / "no_text"
    else:
        image_save_path = images_root
        no_text_image_save_path = images_root

    extracted_sub_path = Path(extracted_sub_path) if extracted_sub_path else dataset_path / "extracted_subs"
    attachement_path = Path(attachement_path) if attachement_path else dataset_path / "attachements"

    for path in [dataset_path, images_root, image_save_path, no_text_image_save_path, extracted_sub_path, attachement_path]:
        os.makedirs(path, exist_ok=True)

    return (
        str(main_mkv_path),
        str(image_save_path),
        str(no_text_image_save_path),
        str(dataset_path),
        str(attachement_path),
        str(extracted_sub_path),
    )


def create_ocr_dataset(
        main_mkv_path: str | Path, 
        image_save_path: str | Path | None = None,
        no_text_image_save_path: str | Path | None = None,
        dataset_path: str | Path | None = None,
        save_format: Literal['PaddleOCR'] = 'PaddleOCR',
        extracted_sub_path: str | Path | None = None,
        preferd_sub_language: str = 'fre', 
        attachement_path: str | Path | None = None,
        multiline: bool = False,
        p_timing: float = 0.005
    ) -> None:
    """
    Automatically generates a complete OCR training dataset from a collection of MKV videos.

    This function scans a directory of `.mkv` video files, extracts subtitles and
    video attachments (e.g., fonts), renders frames at selected time positions,
    and produces paired image–text annotations suitable for OCR training.

    Each sampled frame is processed as follows:
      - If at least one subtitle is active at the chosen timestamp,
        the image (with subtitles rendered on top) is stored in the **text** folder.
      - If no subtitle is active, the plain video frame is stored in the **no_text** folder.
      - Subtitle text and bounding boxes are automatically written to a dataset file
        (`dataset.txt`) in the specified OCR format (e.g., PaddleOCR).

    The resulting dataset can directly be used to train or fine-tune OCR models.

    Args:
        main_mkv_path (str | Path):
            Path to the root directory containing one or more MKV video files.
            Subdirectories are also scanned recursively.
        image_save_path (str | Path | None, optional):
            Directory where images **with subtitles** will be saved.
            Defaults to `<dataset_path>/images/text` if not specified.
        no_text_image_save_path (str | Path | None, optional):
            Directory where images **without subtitles** will be saved.
            Defaults to `<dataset_path>/images/no_text` if not specified.
        dataset_path (str | Path | None, optional):
            Root directory of the dataset. It will contain the `dataset.txt` annotation file, and by default
            the `images/` folder, and subfolders for attachments and extracted subtitles.
            Defaults to `./dataset`.
        save_format (Literal['PaddleOCR'], optional):
            Format of the annotation text file. Currently, only `'PaddleOCR'` is supported.
            Defaults to `'PaddleOCR'`.
        extracted_sub_path (str | Path | None, optional):
            Directory where extracted subtitle files will be saved.
            Defaults to `<dataset_path>/extracted_subs`.
        preferd_sub_language (str, optional):
            Preferred subtitle language code (e.g., `'fre'` for French, `'eng'` for English).
            The most appropriate subtitle track is automatically selected if available.
            Defaults to `'fre'`.
        attachement_path (str | Path | None, optional):
            Directory where extracted attachments (such as embedded font files) will be stored.
            Defaults to `<dataset_path>/attachements`.
        multiline (bool, optional):
            If one event (detection box) can contain multiple text lines. If `False`, one box 
            will contain maximum one line. Default to `False`.
        p_timing (float, optional):
            Probability parameter controlling the number of selected timings over one
            video duration. Roughly corresponds to a sampling ratio.
            Defaults to `0.005`.

    Workflow:
        1. **Directory setup:** Uses `path_check()` to validate and/or create all
           necessary directories (dataset, images, subtitles, attachments).
        2. **Video scanning:** Recursively finds all `.mkv` files in `main_mkv_path`.
        3. **Subtitle extraction:** Chooses and extracts the most suitable subtitle track
           based on `preferd_sub_language`.
        4. **Frame sampling:** Randomly selects timestamps from each video.
        5. **Rendering and dataset writing:**
           - Generates frames with and without subtitles.
           - Writes OCR annotations (text and bounding boxes) to `dataset.txt`.
        6. **Logging and cleanup:** All progress and warnings are written to
           `create_ocr_dataset.log`. C library output (from `libass`) is redirected
           to the logger to avoid terminal spam.

    Outputs:
        The following structure is created by default under the dataset directory:
        ```
        dataset/
        ├── dataset.txt               # OCR annotations
        ├── images/
        │   ├── text/                 # Frames with subtitles
        │   └── no_text/              # Frames without subtitles
        ├── attachements/             # Extracted font and other MKV attachments
        └── extracted_subs/           # Extracted subtitle files
        ```

    Side Effects:
        - Creates multiple folders and files on disk.
        - Logs all activity to `create_ocr_dataset.log`.
        - Redirects `stdout` and `stderr` temporarily to capture C library output.

    Example:
        >>> create_ocr_dataset(
        ...     main_mkv_path="/mnt/d/videos/anime/",
        ...     preferd_sub_language="eng"
        ... )

    Notes:
        - This function handles *all* MKV files found recursively in `main_mkv_path`.
        - Images are automatically categorized into `text/` or `no_text/` based on
          whether subtitles are active at the selected timestamp.
        - Each video’s subtitle fonts are extracted and reused for accurate rendering.
        - Advanced data augmentation (image distortions, style changes, etc.) occurs
          internally during frame generation.
        - Metatata are written in `<dataset_path>/dataset_metadata.txt`
    """
    main_mkv_path, image_save_path, no_text_image_save_path, dataset_path, attachement_path, extracted_sub_path = path_check(
        main_mkv_path=main_mkv_path, 
        image_save_path=image_save_path, 
        no_text_image_save_path= no_text_image_save_path,
        dataset_path=dataset_path, 
        extracted_sub_path = extracted_sub_path, 
        attachement_path = attachement_path
    )

    logging.basicConfig(
        filename='create_ocr_dataset.log', level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

    mkv_files = list(Path(main_mkv_path).rglob("*.mkv"))
    logger.info(f'starting, found {len(mkv_files)} mkv in "{main_mkv_path}"')

    fd_stdout = os.dup(1) # libass is a c library, 
    fd_stderr = os.dup(2) # we need to redirect sdout
    w_fd = redirect_c_stdout_to_logger()

    dataset_metadata_before(dataset_path=dataset_path, main_mkv_path=main_mkv_path,
                            preferd_sub_language=preferd_sub_language, save_format=save_format,
                            multiline=multiline)
    
    start_time = datetime.now()

    n_text_image, n_no_text_image, skipped = 0, 0, 0
    for mkv in tqdm(mkv_files):
        logger.debug(f'starting, {mkv}')
        text, no_text = video_to_dataset(
            video_path=mkv,
            extracted_sub_path=extracted_sub_path,
            preferd_sub_language=preferd_sub_language,
            no_text_image_save_path=no_text_image_save_path,
            dataset_path=dataset_path,
            root_mkv_path=str(main_mkv_path),
            attachement_path=attachement_path,
            image_save_path=image_save_path,
            save_format=save_format,
            multiline=multiline,
            p_timing=p_timing
        )
        n_text_image += text
        n_no_text_image += no_text
        if text == 0 and no_text == 0:
            skipped +=1
        
        gc.collect() # trying to prevent memory leak

    os.dup2(fd_stdout, 1)
    os.dup2(fd_stderr, 2)
    os.close(w_fd)

    dataset_metadata_after(n_video_found=len(mkv_files), n_video_skiped=skipped,
                           n_images_text=n_text_image, n_images_no_text=n_no_text_image,
                           dataset_text_path=os.path.join(dataset_path, 'dataset.txt'),
                           start_time=start_time)
    
    logger.info(f"Process done, {n_text_image} images with text, {n_no_text_image} images without.")
    print(n_text_image, n_no_text_image)

if __name__ == '__main__':
    create_ocr_dataset("/mnt/c/Téléchargement/Banner of the stars")