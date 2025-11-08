from pathlib import Path
from ...video.Video import Video, NoCorrectSubFound
import os
import logging
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import rmtree
from datetime import timedelta, datetime
from typing import cast, Literal
from .disturb import disturb_image, style_transform, disturb_text
from ...video.sub.RendererClean import Context
from PIL import Image
from ...video.classes import dataset_image
from .events_to_dataset import small_images_to_dataset, big_images_to_dataset
import sys
import threading


logger = logging.getLogger(__name__)

def dataset_metadata_before(
        dataset_path: str,
        main_mkv_path: str,
        preferd_sub_language: str,
        save_format: str,
        multiline: bool 

    ) -> None:
    """metadata logging before processing
    """
    with open(os.path.join(dataset_path, 'dataset_metadata.txt'), mode='a', encoding='utf-8') as f:
        f.write("\n========================================\nDataset generation summary\n")
        f.write(f'Date: {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}\n')
        f.write(f'Main MKV path: {main_mkv_path}\n')
        f.write(f'Sub language: {preferd_sub_language}\n')
        f.write(f'Multiline: {multiline}\n')
        f.write(f'Save format: {save_format}\n')

def dataset_metadata_after(
        n_video_found: int,
        n_video_skiped: int,
        n_images_text: int,
        n_images_no_text: int,
        dataset_text_path: str,
        start_time: datetime    
) -> None:
    """metadata logging after processing
    """
    def format_duration(td: timedelta) -> str:
        seconds = int(td.total_seconds())
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    metadata_path = os.path.join(os.path.dirname(dataset_text_path), 'dataset_metadata.txt')
    total_time= datetime.now() - start_time
    with open(metadata_path, mode='a', encoding='utf-8') as f:
        f.write('----------------------------------------\n')
        f.write(f'Time elapsed: {format_duration(total_time)}\n')
        f.write(f'Videos found: {n_video_found}\n')
        f.write(f'Processed: {n_video_found-n_video_skiped}\n')
        f.write(f'Images generated: {n_images_text+n_images_no_text} (text: {n_images_text}, no_text: {n_images_no_text})\n')
        f.write(f'Output dataset: {dataset_text_path}\n')
        f.write('========================================\n')


def redirect_c_stdout_to_logger(
        ass_log_level: Literal['INFO', 'DEBUG', 'WARNING', 'CRITICAL', 'ERROR']  | None = 'DEBUG'
) -> int:
    """Redirects all C-level and Python stdout/stderr output to the Python logger.

    This function creates a pipe, redirects the process's standard output and error 
    streams (file descriptors 1 and 2) to it, and spawns a background thread that 
    continuously reads from the pipe and sends each line to the logger.

    Args:
        ass_log_level (Literal | None): The log level of the libass related outs.

    Note:
        The redirection remains active until stdout and stderr are manually restored.
        To revert to the original state, duplicate and save the original file 
        descriptors before calling this function, then restore them later with os.dup2().

    Returns:
        int: The write-end file descriptor of the pipe used for redirection.
    """
    r_fd, w_fd = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(w_fd, 1)
    os.dup2(w_fd, 2)

    ass_log_method = getattr(logger, ass_log_level.lower(), logger.debug) if ass_log_level else None

    def reader():
        with os.fdopen(r_fd) as r:
            for line in r:
                if line.startswith('[ass]'):
                    if ass_log_method:
                        ass_log_method(line.strip())
                elif line.strip() != '': 
                    logger.info(line.strip())

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    return w_fd 

def choose_and_extract_sub(
        vid:Video, main_path: str,
        extracted_sub_path: str | Path |None = None,
        preferd_sub_language: str = 'fre'
    ) -> tuple[int, str]:
    """randomly select a subtitle track of the desired language.
    Track with higher fps have higher chance of getting selected.
    """

    video_path= Path(vid.path)
    selected_sub_id, selected_sub_name = vid.choose_sub_track(langage=preferd_sub_language)
    logger.debug(f'chose sub {selected_sub_name} with id {selected_sub_id} for {video_path.name}')
    extracted_sub_path = os.path.join(os.getcwd(),'extracted_subs') if extracted_sub_path is None else extracted_sub_path
    extracted_sub_path = os.path.join(
        extracted_sub_path,
        os.path.relpath(os.path.abspath(os.path.dirname(video_path)), os.path.abspath(main_path)),
        f'{video_path.stem}_{selected_sub_id}_{selected_sub_name}'
    )
    vid.extractSub(
        piste=selected_sub_id,
        sortie=extracted_sub_path,
    )
    return selected_sub_id, selected_sub_name

def dump_attachement(
        vid: Video, main_path: str, attachement_path: str | Path | None = None
    ) -> None:
    video_path= Path(vid.path)
    attachement_path = os.path.join(os.getcwd(),'attachements') if attachement_path is None else attachement_path
    attachement_path = os.path.join(
        attachement_path,
        os.path.relpath(os.path.abspath(os.path.dirname(video_path)), os.path.abspath(main_path)),
        video_path.stem
    )
    vid.dumpAtachement(dossier=str(attachement_path))
    

def select_timings(duration_sec:float, p:float=0.01, precision:float=0.1) -> list[float]:
    """select random timings of a video
    Note:
    - binomial mean = n*p so here the a average number of picked timing is duration/precision * p
    - With duration_sec = 1500, p = 0.01 and precision = 0.1, mean = 150
    """
    import numpy as np
    n = int(duration_sec/precision)
    k = np.random.binomial(n, p)
    timings = np.random.uniform(0, duration_sec, size=k)
    timings = (np.round(np.sort(timings), 3)).tolist()
    logger.debug(f'Selected {len(timings)} timings')
    return timings

def timing_to_dataset(
        timing_sec:float, vid:Video, selected_sub_id: int,
        no_text_image_save_path: str | Path, dataset_path: str | Path,
        image_save_path: str | Path, multiline: bool = False,
    )-> list[dataset_image]:
    """
    Generates one or more dataset images for a specific video timestamp.

    This function extracts a background frame from the video at `timing_sec`
    and overlays the active subtitles (if any) using `libass`. Depending on random
    probabilities, it may also:
      - Produce transparent images containing only subtitles.
      - Generate individual images per subtitle event.
      - Apply visual transformations such as cropping, blurring, padding, or 
        subtitle style modifications (font, size, color, etc.).

    Args:
        timing_sec (float): Timestamp (in seconds) within the video.
        vid (Video): Video object containing subtitle and attachment data.
        selected_sub_id (int): ID of the subtitle track being processed.
        no_text_image_save_path (str | Path): Directory for frames without text.
        dataset_path (str | Path): Root dataset directory (used for relative paths).
        image_save_path (str | Path): Directory for frames with subtitles.
        multiline (bool, optional): Whether multiline subtitle rendering is allowed (a sub is an event). Defaults to `False`, a line is an event.

    Returns:
        list[dataset_image]: A list of generated dataset images, including the
        main composite image and optional variant images.

    Notes:
        - Each image may be visually or textually augmented with a certain probability.
        - The function is typically called internally by `video_to_dataset()` and is
          not meant for direct use.
    """
    vid_name = Path(vid.path).stem
    n_event_in_frame, event_section = vid.docs[selected_sub_id].nb_event_dans_frame(timedelta(seconds=timing_sec), returnEvents=True)

    
    background = vid.extract_frame_as_pil(timing_sec)
    if n_event_in_frame == 0:
        # there is no sub active in that timing, not much to do
        #TODO  : ajouter texte japonais non considéré comme texte ?
        background = disturb_image(img=background)
        image_name = os.path.join(no_text_image_save_path,f'{vid_name}_s{selected_sub_id}_t{timing_sec}.png')
        background.save(image_name)
        return [
            dataset_image(
                image_path=os.path.relpath(image_name, dataset_path),
                event_list=[])
        ]
    
    ctx = Context()
    ctx.fonts_dir = str(vid.attachement_path).encode('utf-8')
    r = ctx.make_renderer()
    r.set_fonts(fontconfig_config="\0")
    r.set_all_sizes(background.size)

    if n_event_in_frame == 1 and event_section[0].style == 'Default':
        for i, style in enumerate(vid.docs[selected_sub_id].styles):
            if style.name == 'Default':
                vid.docs[selected_sub_id].styles[i] = style_transform(style=style)
        vid.docs[selected_sub_id].events = disturb_text(event_list=vid.docs[selected_sub_id].events, timestamp=timing_sec)
    
    events_with_pil = vid.get_subtitle_boxes(timestamp=timing_sec, renderer=r, context=ctx, piste=selected_sub_id, multiline = multiline)

    return_event_list = []
    
    for event in events_with_pil: 
        background = Image.alpha_composite(background, event.image)
        return_event_list+=event.events
    
    background, events_with_pil = disturb_image(background, events_with_pil) # disturb image after composite to try to disturb subs too

    # crop can remove all events in the frame
    image_name = os.path.join(image_save_path if len(events_with_pil)>0 else no_text_image_save_path,f'{vid_name}_s{selected_sub_id}_t{timing_sec}.png')
    background.save(image_name)

    return_dataset_image_list = [
        dataset_image(
            image_path=os.path.relpath(image_name, dataset_path),
            event_list=return_event_list
    )]

    return_dataset_image_list += small_images_to_dataset(
        timestamp=timing_sec, video=vid, r=r,
        dataset_path=str(dataset_path), image_save_path=str(image_save_path),
        ctx=ctx, multiline=multiline, sub_id=selected_sub_id
    )    

    return_dataset_image_list += big_images_to_dataset(
        events_with_pil=events_with_pil, dataset_path=str(dataset_path), image_save_path= str(image_save_path),
        vid_name= vid_name, sub_id=selected_sub_id, timestamp= timing_sec
    )

    return return_dataset_image_list

def after_video_to_dataset_cleanup(
        vid: Video,
        attachement_path: str
) -> None:
    """detete vid attachement folder if empty and all the folders 
    between vid attachement folder and root attachement folder
    """
    vid.attachement_path = cast(str, vid.attachement_path) # remove pylance error
    if not os.listdir(vid.attachement_path) and os.path.isdir(vid.attachement_path):
        # no attachements were extracted, no need to leave a empty folder
        rmtree(vid.attachement_path)

    root_dir = os.path.abspath(attachement_path)
    current_dir = os.path.abspath(vid.attachement_path)

    while os.path.abspath(current_dir).startswith(root_dir):
        if (
            os.path.isdir(current_dir) and not os.listdir(current_dir)
            and not os.path.abspath(current_dir) == root_dir
        ):
            rmtree(current_dir)
        else:
            break

        if os.path.abspath(current_dir)==root_dir:
            break

        current_dir = os.path.abspath(current_dir)
    

def video_to_dataset(
        video_path: Path, root_mkv_path: str,
        image_save_path: str | Path,
        extracted_sub_path: str | Path,
        attachement_path: str | Path,
        no_text_image_save_path: str | Path,
        dataset_path: str | Path,
        save_format: Literal['PaddleOCR'] = 'PaddleOCR',
        preferd_sub_language: str = 'fre', p_timing: float = 0.005,
        multiline: bool = False
) -> tuple[int, int]:
    """
    Extracts subtitle-based training data from an MKV video and saves it as a dataset.

    This function processes a single `.mkv` video by:
      1. Selecting and extracting the most suitable subtitle track (e.g., French).
      2. Extracting video attachments (such as font files embedded in the MKV).
      3. Sampling multiple time positions ("timings") across the video duration.
      4. Rendering screenshots at each timing — with and without subtitles — using `cv2` for
         background frames and `libass` for subtitle rendering.
      5. For each generated image, the subtitle text and bounding boxes are saved as dataset
         entries in `dataset.txt`, following the `PaddleOCR` text recognition format.

    This is typically used to generate OCR training data where subtitles serve as labeled text
    over video frames.

    Args:
        video_path (Path):
            Path to the input MKV video file.
        root_mkv_path (str):
            Root directory containing the MKV files (and possibly subdirectories of MKV files).
        image_save_path (str | Path):
            Root path for images rendered **with subtitles**.
        extracted_sub_path (str | Path):
            Directory where the extracted subtitle file will be saved.
        attachement_path (str | Path):
            Root directory where extracted attachments (e.g., fonts) will be stored.
        no_text_image_save_path (str | Path):
            Root path for images rendered **without subtitles** (clean background).
        dataset_path (str | Path):
            Directory containing the dataset output. This directory will include
            `dataset.txt` (containing OCR annotations) and possibly subfolders for images/attachments.
        save_format (Literal['PaddleOCR'], optional):
            The format used for dataset text entries.
            Defaults to `'PaddleOCR'`. Each image line follows the PaddleOCR dataset format.
        preferd_sub_language (str, optional):
            Preferred subtitle language code (e.g., `'fre'` for French, `'eng'` for English).
            Defaults to `'fre'`.
        p_timing (float, optional):
            Probability parameter controlling the number of selected timings over the
            video duration. Roughly corresponds to a sampling ratio.
            Defaults to `0.005`.
        multiline (bool, optional):
            If one event (detection box) can contain multiple text lines. If `False`, one box 
            will contain maximum one line. Default to `False`.

    Raises:
        FileNotFoundError:
            If the specified MKV video file does not exist.
        ValueError:
            If the provided file is not an MKV file.
        NoCorrectSubFound:
            If no suitable subtitle track matching the preferred language is found.
            (Handled internally — the video is skipped with a warning.)

    Side Effects:
        - Creates multiple image files in `image_save_path` and `no_text_image_save_path`.
        - Creates or appends to a text dataset file at `{dataset_path}/dataset.txt`.
        - Extracts subtitle and attachment files into `extracted_sub_path` and `attachement_path`.
        - Uses multithreading (via ThreadPoolExecutor) to parallelize frame extraction.

    Notes:
        - Each timing is processed concurrently in a separate thread.
        - Writing to `dataset.txt` is performed as results are gathered.
        - The function returns `None`, but logs detailed progress and errors using `logger`.

    Example:
        >>> video_to_dataset(
        ...     video_path=Path("movies/example.mkv"),
        ...     root_mkv_path="/datasets/mkv/",
        ...     image_save_path="dataset/images/text",
        ...     extracted_sub_path="dataset/subs",
        ...     attachement_path="dataset/attachments",
        ...     no_text_image_save_path="dataset/images/no_text",
        ...     dataset_path="dataset/",
        ...     preferd_sub_language="eng"
        ... )

    Returns:
        tuple[int, int]: The number of images created, (images with text, images without text)
    """
    logger.debug(f'starting {os.path.basename(video_path)}')
    if not video_path.exists():
        raise FileNotFoundError(f'video {video_path.absolute()} was not found')
    if not str(video_path).endswith('.mkv'):
        raise ValueError(f'{video_path} is not a mkv file')
    try:
        vid = Video.make_video(video_path)
    except RuntimeError:
        logger.error(f"Error opening video {os.path.basename(video_path)}, this video will be skipped.")
        return 0, 0
    try:
        selected_sub_id, selected_sub_name =choose_and_extract_sub(vid=vid, main_path=root_mkv_path,
                            extracted_sub_path=extracted_sub_path, 
                            preferd_sub_language=preferd_sub_language)
    except NoCorrectSubFound:
        logger.warning(f'Cannot find a good sub track for {os.path.dirname(video_path)}, this video will be skiped')
        return 0, 0
    
    dump_attachement(vid=vid, main_path=root_mkv_path, attachement_path=attachement_path)

    timings = select_timings(
        duration_sec=vid.duree,
        p=p_timing
    )

    dataset_file = os.path.join(dataset_path, 'dataset.txt')

    executor = ThreadPoolExecutor(max_workers=50)

    
    retults = [
        executor.submit(
            timing_to_dataset,
            timing_sec = timing,
            vid = vid.copy(timing=timing, doc_id=selected_sub_id),
            selected_sub_id= selected_sub_id,
            image_save_path= image_save_path,
            no_text_image_save_path=no_text_image_save_path,
            dataset_path = dataset_path,
            multiline=multiline
        ) for timing in timings]
    
    n_text_image, n_no_text_image = 0, 0
    for t in tqdm(as_completed(retults), total= len(timings), desc=os.path.basename(video_path), leave=False):
        try:
            timing_results= t.result()
            for image in timing_results:
                image.to_text(path=dataset_file, format= save_format)
                if image.event_list:
                    n_text_image += 1
                else:
                    n_no_text_image += 1
                del image
            del timing_results
        except Exception as e:
            logger.error(f'{Path(video_path).stem} : Error during result opening {e}')
    
    after_video_to_dataset_cleanup(
        vid=vid,
        attachement_path=str(attachement_path)
    )

    logger.debug(f"Finished video {video_path}, {n_text_image} images with text and {n_no_text_image} without.")
    return n_text_image, n_no_text_image



    