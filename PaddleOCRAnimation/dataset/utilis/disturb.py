from PIL import Image, ImageFilter
from ...video.classes import eventWithPilList, eventWithPil
from typing import overload
import random
import numpy as np
from io import BytesIO
from ass import line, data, section
from copy import deepcopy
import matplotlib.font_manager as fm
from datetime import timedelta
import re
import logging
from os.path import exists, dirname, join
import pandas as pd
from string import digits

logger = logging.getLogger(__name__)

def replace_random_space(s: str, replacement: str, chance_of_first: float = 0) -> str:
    if random.random() < chance_of_first:
        return replacement.strip()+" "+s
    space_positions = [i for i, ch in enumerate(s) if ch == " "]
    if not space_positions:
        return s
    idx = random.choice(space_positions)
    return s[:idx] + replacement + s[idx+1:]

def disturb_eventWithPil(events: eventWithPil, p_padding:float = 0.15,
                             mean_band_size_perc:float = 0.2,
                             p_pixelize: float = 0.25,
                             p_change_rez: float = 0.15,
                            ) -> eventWithPil:
    """
    Applies random visual disturbances to an event image to increase dataset variability.

    This function can:
      - Add random transparent padding around the image (per side) with probability `p_padding`.
      - Pixelize the image (reduce quality without resizing) with probability `p_pixelize`.
      - Randomly rescale the image and corresponding bounding boxes with probability `p_change_rez`.

    Args:
        events (eventWithPil): The event containing the image, text, and boxes.
        p_padding (float, optional): Probability of adding transparent padding on each side. Defaults to 0.15.
        mean_band_size_perc (float, optional): Mean relative padding size per side. Defaults to 0.2.
        p_pixelize (float, optional): Probability to pixelize the image (without changing its resolution). Defaults to 0.25.
        p_change_rez (float, optional): Probability to randomly change the image resolution and resize boxes accordingly. Defaults to 0.15.

    Returns:
        eventWithPil: The modified event, possibly padded, pixelized, or resized.
    """
    def add_transparent_padding(
            events: eventWithPil,
            p_padding:float,
            mean_band_size_perc:float
        ) -> eventWithPil:
        perc: list[float] = [0, 0, 0, 0]

        for i in range(0, len(perc), 1): 
            if random.random() < p_padding:
                perc[i] = abs(random.gauss(mean_band_size_perc, 0.15))
        
        if perc == [0, 0, 0, 0]:
            return events
        
        im_w, im_h = events.image.size
        padding = (int(im_w*perc[0]), int(im_h*perc[1]), int(im_w*perc[2]), int(im_h*perc[3]))
        events.add_padding(padding=padding)
        return events
    
    def change_rez(
            events: eventWithPil,
            p: float = 0.1
        ) -> eventWithPil:
        if random.random() > p:
            return events
        
        ratio = max(0.3, random.gauss(mu= 0.65, sigma=0.12))
        w, h = events.image.size
        w, h = int(w*ratio), int(h * ratio)
        events.image = events.image.resize(size=(w, h))
        for i, event in enumerate(events.events):
            box = event.Boxes
            box.resize(scale=ratio)
            events.events[i].Boxes = box
        
        return events

    
    events = add_transparent_padding(events=events, p_padding=p_padding, 
                                     mean_band_size_perc=mean_band_size_perc)
    if random.random() < p_pixelize:
        events.image = pixelate_image(events.image)
    
    events = change_rez(events=events, p=p_change_rez)
    
    return events



@overload
def crop_image(
    image: Image.Image, event_list: eventWithPilList,
    height_cut_ratio: float = 0.72, width_cut_ratio: float = 0.01,
    reverse:bool = False
    ) -> tuple[Image.Image, eventWithPilList]:
    ...
@overload
def crop_image(
    image: Image.Image, event_list: None = None,
    height_cut_ratio: float = 0.72, width_cut_ratio: float = 0.01,
    reverse:bool = False
) -> Image.Image:
    ...
def crop_image(
        image: Image.Image, event_list: eventWithPilList | None = None,
        height_cut_ratio: float = 0.72, width_cut_ratio: float = 0.015,
        reverse:bool = False, p_cut_h: float = 0.9, p_cut_w: float = 0.8,
    ):
    """
    Randomly crops an image to simulate the frame cropping performed during OCR preprocessing.

    By default, the crop removes a large part of the top of the image in order to keep the
    lower area where subtitles usually appear. If `reverse=True`, the behavior is inverted and
    the crop keeps the upper part instead.

    When an `event_list` is provided, subtitle bounding boxes are inspected to compute a valid
    crop limit: the crop is reduced if needed so that relevant subtitle boxes remain inside the
    final image. Boxes located in the opposite half of the image are ignored when determining
    this limit.

    Args:
        image (Image.Image):
            Input image to crop.
        event_list (eventWithPilList | None, optional):
            Subtitle events with bounding boxes. When provided, their coordinates are shifted to
            match the cropped image, and they are also used to prevent cropping through relevant
            subtitle regions. Defaults to `None`.
        height_cut_ratio (float, optional):
            Mean proportion of image height removed vertically. Defaults to `0.72`.
        width_cut_ratio (float, optional):
            Mean proportion of image width removed on both horizontal sides. Defaults to `0.015`.
        reverse (bool, optional):
            If `False`, keeps the bottom part of the image. If `True`, keeps the top part.
            Defaults to `False`.
        p_cut_h (float, optional):
            Probability of applying a vertical crop. Defaults to `0.9`.
        p_cut_w (float, optional):
            Probability of applying a horizontal crop. Defaults to `0.8`.

    Returns:
        Image.Image | tuple[Image.Image, eventWithPilList]:
            The cropped image. If `event_list` is provided, returns the cropped image together
            with the updated event list.
    """
    im_w, im_h = image.size
    min_h, min_w, max_h, max_w = im_h, im_w, 0, 0
    if event_list is not None:
        for event_bloc in event_list:
            for event in event_bloc.events:
                box = event.Boxes.full_box
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                e_min_w, e_max_w = min(xs), max(xs)
                e_min_h, e_max_h = min(ys), max(ys)
                
                if reverse and (e_min_h > im_h//2 or e_max_h > im_h//2):
                    # The sub is in the bottom part of the image (or right in the middle), we only care about the top part
                    continue
                elif not reverse and (e_min_h < im_h//2 or e_max_h < im_h//2):
                    continue
                min_h, min_w, max_h, max_w = min(e_min_h, min_h), min(e_min_w, min_w), max(max_h, e_max_h), max(max_w, e_max_w)

    cut_top = abs(int(random.gauss(height_cut_ratio, 0.08)* im_h)) if random.random() < p_cut_h else 0
    cut_sides = abs(int(random.gauss(width_cut_ratio , 0.035)* im_w)) if random.random() < p_cut_w else 0
    cut_bottom = 0

    if reverse :
        # we want the top the the image
        cut_bottom, cut_top = cut_top, cut_bottom
    
    cut_left, cut_top, cut_right, cut_bottom = min(min_w, cut_sides), min(min_h, cut_top), max(im_w-cut_sides, max_w), max(max_h, im_h-cut_bottom)
    im = image.crop((cut_left, cut_top, cut_right, cut_bottom))

    if event_list is not None: 
        event_list.add_padding((-cut_left, -cut_top, -(im_w-cut_right), -(im_h-cut_bottom)))
        
        return im, event_list
    
    return im

@overload
def add_black_band(
        img: Image.Image, event_list: eventWithPilList,
        mean_band_size_perc: float = 0.15, p: float=0.35
    ) -> tuple[Image.Image, eventWithPilList]:
    ...
@overload
def add_black_band(
        img: Image.Image, event_list: None = None,
        mean_band_size_perc: float = 0.15, p: float=0.35
    ) -> Image.Image:
    ...
def add_black_band(
        img: Image.Image, event_list: eventWithPilList | None = None,
        mean_band_size_perc: float = 0.15, p: float=0.35
    ):
    """Randomly adds black bands on image borders to improve model robustness.

    Args:
        img (Image.Image): Input image.
        event_list (eventWithPilList | None, optional): 
            List of text events with bounding boxes to update after padding. Defaults to `None`.
        mean_band_size_perc (float, optional): 
            Mean band thickness as a percentage of image size. Defaults to `0.15`.
        p (float, optional): 
            Probability of adding a black band on each side. Defaults to `0.35`.

    Returns:
        Image.Image | tuple[Image.Image, eventWithPilList]: 
            The augmented image, and the updated event list if provided.
    """
    im_w, im_h = img.size
    padding = [0, 0, 0, 0]
    for i in range(4):
        if random.random() < p:
            s = im_w if i in [0, 2] else im_h
            pad = abs(int(random.gauss(mean_band_size_perc, 0.10)* s))
            padding[i] = pad
    
    if sum(padding) == 0:
        return img if event_list is None else (img, event_list)
    new_img= Image.new(img.mode, (im_w+padding[0] +padding[2], im_h +padding[1]+padding[3]), (0, 0, 0))
    new_img.paste(img, (padding[0], padding[1]))

    if event_list is not None: 
        event_list.add_padding(padding=(padding[0], padding[1], padding[2], padding[3]))
        return new_img, event_list
    
    return new_img


def add_noise(img, mean: float =0, std: float =10):
    """Rajoute du bruit (grain) sur l'image
    """
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, np_img.shape)
    noisy_img = np_img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)  # Pour rester entre 0 et 255
    img = Image.fromarray(noisy_img)
    return img.convert('RGBA') if img.mode == 'RGBA' else img

@overload
def pixelate_image(
        img:Image.Image,
        event_list:eventWithPilList,
        factor: float = 0.7,
    ) -> tuple[Image.Image, eventWithPilList]:
    ...
@overload
def pixelate_image(
        img:Image.Image,
        event_list: None = None,
        factor: float = 0.7,
    ) -> Image.Image:
    ...
def pixelate_image(
        img:Image.Image,
        event_list:eventWithPilList | None = None,
        factor: float = 0.7,
    ) -> Image.Image | tuple[Image.Image, eventWithPilList]:
    """Applies a pixelation effect to an image and optional event overlays.

    Args:
        img (Image.Image): Input image to pixelate.
        event_list (eventWithPilList | None, optional):
            Optional event list whose images are pixelated with the same factor.
            Defaults to None.
        factor (float, optional):
            Downscaling factor used before upscaling back to the original size.
            Lower values produce stronger pixelation. Clamped to the [0.05, 0.9] range.
            Defaults to 0.7.

    Returns:
        Image.Image | tuple[Image.Image, eventWithPilList]:
            The pixelated image, and the updated event list if provided.
    """

    def downsize_upsize(img: Image.Image, factor: float)->Image.Image:
        small = img.resize(
            (int(width * factor), int(height * factor)),
            resample=2 # Resampling.BILINEAR
        )

        pixelated = small.resize(
            (width, height),
            resample=0 # Resampling.NEAREST
        )
        return pixelated
    width, height = img.size

    factor = max(0.05, min(0.9, factor))


    pixelated = downsize_upsize(img, factor=factor)
    logger.debug(f"Pixaleted image with ratio {factor}")

    if event_list is None:
        return pixelated
    
    for i, event in enumerate(event_list):
        event_list[i].image = downsize_upsize(event.image, factor=factor)
    return pixelated, event_list

def jpeg_compress(img: Image.Image, quality:int =10):
    """Sauvegarde sur RAM en JPEG (avec compression) et réouvre cette sauvegarde
    """
    quality = min(max(1, quality), 95)
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

@overload
def change_rez_image(
    img:Image.Image, event_list: None =None,
    ratio: float = 0.7,
)-> Image.Image:
    ...
@overload
def change_rez_image(
    img:Image.Image, event_list: eventWithPilList,
    ratio: float = 0.7,
)-> tuple[Image.Image, eventWithPilList]:
    ...
def change_rez_image(
        img:Image.Image, event_list: eventWithPilList | None =None,
        ratio: float = 0.7,
    ) -> tuple[Image.Image, eventWithPilList] | Image.Image:
    """Resizes an image and its associated events by a given scale factor.

    Args:
        img (Image.Image): Input image to resize.
        event_list (eventWithPilList | None, optional):
            Optional event list whose images and boxes are resized with the same factor.
            Defaults to None.
        ratio (float, optional):
            Resize factor applied to width and height. Clamped to the [0.3, 1.0] range.
            Defaults to 0.7.

    Returns:
        Image.Image | tuple[Image.Image, eventWithPilList]:
            The resized image, and the updated event list if provided.
    """
    w, h = img.size
    ratio = min(max(ratio, 0.3), 1)
    w, h = int(w*ratio), int(h * ratio)
    img = img.resize(size=(w, h))

    if event_list is not None:
        for i, event in enumerate(event_list):
            w, h = event.image.size
            w, h = int(w*ratio), int(h*ratio)
            event_list[i].image = event.image.resize(size=(w, h))

            for unique_event in event.events:
                unique_event.Boxes.resize(scale=ratio)
                if unique_event.baseline_Boxes is not None:
                    unique_event.baseline_Boxes.resize(scale=ratio)
        
        return img, event_list
    
    return img


@overload
def disturb_image(img: Image.Image, event_list: eventWithPilList) -> tuple[Image.Image, eventWithPilList]:
    ...
@overload
def disturb_image(img: Image.Image, event_list: None = None) -> Image.Image:
    ...
def disturb_image(img: Image.Image, event_list: eventWithPilList | None = None):
    """
    Randomly applies a set of visual distortions to simulate real-world noise in text detection datasets.  
    This includes random cropping, blurring, noise addition, JPEG compression, and salt-and-pepper artifacts.  
    If an event list with bounding boxes is provided, it is adjusted accordingly after cropping.

    Args:
        img (Image.Image): Input image to be distorted.  
        event_list (eventWithPilList | None, optional):  
            List of bounding boxes or events to adjust after cropping. Defaults to `None`.

    Returns:
        Image.Image | tuple[Image.Image, eventWithPilList]:  
            The distorted image, and the updated event list if provided.
    """
    def linear_value_from_resolution(
        resolution: float,
        value_low_res: float,
        value_high_res: float,
        low_res: float = 480,
        high_res: float = 1080,
    ) -> float:
        """
        Interpole/extrapole linéairement une valeur en fonction de la résolution.
        """
        if low_res == high_res:
            raise ValueError("low_res et high_res doivent être différents.")

        a = (value_high_res - value_low_res) / (high_res - low_res)
        b = value_low_res - a * low_res
        return a * resolution + b


    def get_mu_sigma_from_resolution(
        resolution: float,
        mu_low_res: float,
        mu_high_res: float,
        sigma_low_res: float,
        sigma_high_res: float,
        low_res: float = 480,
        high_res: float = 1080,
    ) -> tuple[float, float]:
        """
        Renvoie (mu, sigma) interpolés/extrapolés selon la résolution.
        """
        mu = linear_value_from_resolution(
            resolution=resolution,
            value_low_res=mu_low_res,
            value_high_res=mu_high_res,
            low_res=low_res,
            high_res=high_res,
        )

        sigma = linear_value_from_resolution(
            resolution=resolution,
            value_low_res=sigma_low_res,
            value_high_res=sigma_high_res,
            low_res=low_res,
            high_res=high_res,
        )

        return mu, sigma
    original_w, original_h = img.size
    transforms_applied = []

    hard_quality_degradation = random.choices(
        population=['GaussianBlur', 'jpeg_compress', 'pixelate_image', 'change_rez_image', 'None'],
        weights=   [            10,              15,               20,                 20,     40], k=1
    )[0]

    if random.random() < 0.15:
        if event_list is None:
            img=crop_image(image=img)
        else:
            img, event_list = crop_image(image=img, event_list=event_list)
        transforms_applied.append("crop_image")

    elif random.random() < 0.15:
        if event_list is None:
            img=crop_image(image=img, reverse=True)
        else:
            img, event_list = crop_image(image=img, event_list=event_list, reverse=True)
        transforms_applied.append("crop_image_reverse")

    if random.random() < 0.15:
        img = add_noise(img, std=random.uniform(2, 12))
        transforms_applied.append("add_noise")

    if random.random() < 0.1:
        img = salt_and_pepper(img).convert('RGBA') if img.mode == 'RGBA' else salt_and_pepper(img)
        transforms_applied.append("salt_and_pepper")

    if random.random()<0.20:
        if event_list is None:
            img=add_black_band(img=img)
        else:
            img, event_list = add_black_band(img=img, event_list=event_list)
        transforms_applied.append("add_black_band")

    if hard_quality_degradation == "GaussianBlur":
        mu, sigma = get_mu_sigma_from_resolution(resolution=original_h, mu_high_res=2, mu_low_res=0.6,
                                                 sigma_high_res=0.25, sigma_low_res=0.1)
        radius=random.gauss(mu= mu, sigma=sigma)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        transforms_applied.append("GaussianBlur")

    elif hard_quality_degradation == 'jpeg_compress':
        mu, sigma = get_mu_sigma_from_resolution(resolution=original_h, mu_high_res=30, mu_low_res=85,
                                                 sigma_high_res=8, sigma_low_res=3)
        quality = int(random.gauss(mu= mu, sigma=sigma))
        img = jpeg_compress(img, quality=quality)
        transforms_applied.append("jpeg_compress")

    elif hard_quality_degradation == 'pixelate_image':
        mu, sigma = get_mu_sigma_from_resolution(resolution=original_h, mu_high_res=0.65, mu_low_res=0.95,
                                                 sigma_high_res=0.1, sigma_low_res=0.04)
        factor = random.gauss(mu= mu, sigma=sigma)
        if event_list is None:
            img = pixelate_image(img=img, factor=factor)
        else:
            img, event_list = pixelate_image(img=img, event_list=event_list, factor=factor)
        transforms_applied.append("pixelate_image")
    
    elif hard_quality_degradation == 'change_rez_image':
        mu, sigma = get_mu_sigma_from_resolution(resolution=original_h, mu_high_res=0.65, mu_low_res=0.95,
                                                 sigma_high_res=0.02, sigma_low_res=0.8)
        ratio = random.gauss(mu=mu, sigma=sigma)
        if event_list is None:
            img=change_rez_image(img=img, ratio=ratio)
        else:
            img, event_list = change_rez_image(img=img, event_list=event_list, ratio=ratio)
        transforms_applied.append("change_rez_image")

    if event_list is None:
        return img
    return img, event_list



def style_transform(style: line.Style) -> line.Style:
    """Applique des transformations aléatoires sur les attributs d'un style de ligne.

    Cette fonction modifie aléatoirement certains paramètres visuels d'un objet `line.Style`
    pour créer de la diversité graphique :
        - Changement de police (en évitant les polices problématiques sous Windows)
        - Perturbation des couleurs (primaire et contour) via une distribution normale
        - Légère variation de la taille de police
        - Inversion d'alignement (haut ↔ bas) pour certains cas
        - Invertion de italique/gras

    Args:
        style (line.Style): Style de ligne d'origine à transformer.

    Returns:
        line.Style: Nouveau style modifié de manière aléatoire.
    """
    def change_color(color: data.Color, ecart_type: float = 80) -> data.Color:
        for col in ['r', 'g', 'b']:
            setattr(color, col, int(np.clip(np.random.normal(getattr(color, col), ecart_type), 0, 255)))
        return color

    mauvaises_polices = {  # Les polices qui, sur windows, ne donne pas du texte
        "Wingdings 2", "Webdings", "Wingdings", "MS Reference Specialty",
        "MT Extra", "MS Outlook", "Bookshelf Symbol 7", "Segoe MDL2 Assets",
        "Symbol", "Segoe Fluent Icons", "Wingdings 3"
    }
    style = deepcopy(style)
    if random.random() < 0.60:
        nom_polices = {
            fm.FontProperties(fname=font).get_name(): font
            for font in fm.findSystemFonts(fontpaths=None, fontext='ttf')
        }
        nouvelle_police = random.choice(
            [nom for nom in nom_polices if nom not in mauvaises_polices]
        )
        style.fontname = nouvelle_police

    if random.random() < 0.25:
        style.primary_color = change_color(style.primary_color)

    if random.random() < 0.15:
        style.outline_color = change_color(style.outline_color)

    if random.random() < 0.2:
        style.fontsize = np.random.normal(style.fontsize, 5)

    mapAlignement = {8: 2, 2: 8}
    if style.alignment in mapAlignement and random.random() < 0.1:
        style.alignment = mapAlignement[style.alignment]

    if random.random() < 0.1:
        style.bold = not style.bold

    if random.random() < 0.35:
        style.italic = not style.italic

    return style


def disturb_text(
        event_list: section.EventsSection,
        p_three_dots_before: float = 0.04,
        p_three_dots_after: float = 0.07,
        timestamp: float | timedelta | None = None
        ) -> section.EventsSection:
    """Applies simple random text perturbations to subtitle dialogue events (in place).

    Current perturbations:
      - With probability p=0.6, replaces the event text by a single randomly chosen word.
      - Optionally adds ellipses ("..."/"…") before/after the text, and sometimes a final "."
        (ellipsis insertion is applied only if the event is active at `timestamp`, when provided).

    Args:
        event_list: EventsSection containing Dialogue events.
        p_three_dots_before: Probability to prepend ellipses.
        p_three_dots_after: Probability to append ellipses.
        timestamp: If given (seconds or timedelta), only events active at this time may receive
            ellipses (word reduction is currently unconditional).

    Returns:
        The same EventsSection object, modified in place.
    """
    def add_three_dots(
            event: line.Dialogue,
            p_three_dots_before: float = 0.2,
            p_three_dots_after:float = 0.40,
            p_for_dots_after: float = 0,
            p_two_dots_after:float = 0,
            p_point_after: float = 0.25,
            timestamp: timedelta | None = None
    ) -> line.Dialogue:
        if not timestamp or (event.start <= timestamp <= event.end):
            text = re.sub(r'\{.*?\}', '', event.text.strip())
            if random.random() < p_three_dots_before and not text.startswith('...') and not text.startswith('…'):
                text = '...'+text
                event.text = text

            
            if random.random() < p_three_dots_after and not text.endswith(("...", "…", "!", "?", ",")):
                if text.endswith('.'):
                    text = text+'..'
                    event.text = text
                    logger.debug(f'Added three dots to text, new text : {text}')
                else:
                    text = text+'...'
                    logger.debug(f'Added three dots to text, new text : {text}')
                    event.text = text
            elif random.random() < p_two_dots_after and not text.endswith(("...", "…", "!", "?", ",", "..")):
                if text.endswith('.'):
                    text = text+'.'
                    event.text = text
                    logger.debug(f'Added two dots to text, new text : {text}')
                else:
                    text = text+'..'
                    logger.debug(f'Added two dots to text, new text : {text}')
                    event.text = text
            elif random.random() < p_for_dots_after and not text.endswith(("...", "…", "!", "?", '.', ",")):
                text = text+'....'
                event.text = text
                logger.debug(f'Added four dots to text, new text : {text}')
            elif random.random() < p_point_after and not text.endswith(("...", "…", "!", "?", '.', ",")):
                text = text+'.'
                event.text = text
        return event
    
    def add_one_spe_char_word(
            event:line.Dialogue, p: float = 0.8,
            char_list: list[str] = ['â', 'ö', 'ï', 'î', 'ô', 'ë','ê', 'û', 'ü', 'à', 'é', 'ç', 'è'],
            p_maj:float = 0.4
        ) -> line.Dialogue:
        
        if random.random() > p :
            return event
        capital = random.random()<p_maj
        path = dirname(__file__)
        logger.debug(path)
        path = join(path, 'wordfreq.parquet')
        if not exists(path):
            raise FileNotFoundError('The file wordfreq.parquet does not exists, it is probably an import error')
        # TODO : global cache
        df = pd.read_parquet(path)
        spe_char = random.choices(
            char_list, weights=[11, 3, 11, 7, 8, 10, 5, 7, 4, 8, 0, 5, 2] if not capital else [10, 2, 10, 10, 10, 10, 10, 10, 2, 10, 15, 10, 10]
            )[0]
        df = df[df['ortho'].str.contains(spe_char, na=False)][['ortho','freqfilms2']]
        if df.empty:
            return event
        mot = df.sample(n=1, weights='freqfilms2', replace=True)['ortho'].iloc[0]

        if capital:
            mot = mot.replace(spe_char, spe_char.capitalize())
        
        event.text = replace_random_space(s=event.text, replacement=' '+mot.strip()+' ', chance_of_first=0.3)
        logger.debug(f"added spe_char_word {mot} in {event.text}")
        return event
    
    def add_any_spe_char(
            event: line.Dialogue, p: float = 0.15,
    ) -> line.Dialogue:
        if random.random() >p:
            return event
        dic = [
            '_', '.', '°', '*', '[', ']',
            '-', '–', '—', 
            "'", ':', '=', 'Œ', 'Ç', 'È', 'Ô', 'ô', '€', 'À', 'Û',
            'Â', '"', ',', '’', '…', 'â', '%', 'ç', '?', '!', ';', '(',
            '+', 'Ë', '<', 'Î', 'Ï', '&', '@', 'œ', 'ü', '^'
            '«', '»', 'æ', 'µ', '$', '#', ')', 'ï', '²', 'ß'
        ]

        spe_char = random.choice(dic)
        event.text = replace_random_space(s=event.text, replacement=' '+spe_char+' ')
        return event
    
    def add_prio_spe_char(
            event: line.Dialogue, p: float = 0.2,
    ):
        if random.random()>p:
            return event
        
        rom_num = random.choices(
                    [' I ', ' II ', ' III ', ' IV' , ' V ', ' VI ', ' VIII ', ' XI ', ' XII '], 
            weights=[   10,     20,      17,     13,    10,      8,        6,      1,       1], k=1)[0]
        prio_list = [' “', '” ', ' W', ' K', ' Y',' F', '_','" ',' "', ' : ', '...', '... ',  rom_num]
        w=          [  10,   10,    6,    6,    5,   6,  10,  10,  10,    10,    20,     20,       15]
        spe_char= random.choices(prio_list, weights=w,k=1)[0]
        event.text = replace_random_space(s=event.text, replacement=spe_char)
        logger.debug(f'added prio spe char {spe_char} in {event.text}')
        return event

    def add_numbers(
            event: line.Dialogue, p: float = 0.3,
            p_float: float = 0.3, p_text: float = 0.4
    ) -> line.Dialogue:
        """train model to reco numbers"""
        if random.random()>p:
            return event
        n_number = random.choices([1, 2, 3, 4], weights=[10, 17, 20, 10], k=1)[0]
        digit = digits 
        code = "".join(random.choice(digit) for _ in range(n_number))

        if random.random() < p_float:
            sep = random.choice([',', '.'])
            n_number = random.choices([1, 2, 3, 4], weights=[20, 25, 7, 3], k=1)[0]
            code = code+sep+"".join(random.choice(digit) for _ in range(n_number))
        
        if random.random()< p_text:
            t= random.choices(['€', '$', '%', 'h', 's', '¥'], weights=[20, 10, 35, 20, 15, 10], k=1)[0]
            code +=t
        event.text = replace_random_space(s=event.text, replacement=' '+code+' ')
        logger.debug(f'added {code} to {event.text}')
        return event
    
    def capitalize_sentence(
            event: line.Dialogue, 
            w_no_change: int = 30,
            w_all_sentence: int = 13,
            w_firt_letter: int = 20,
            w_entire_word: int = 13,
            w_random_letters: int = 5,
            
        ) -> line.Dialogue:   
        def uppercase_random_letters(word: str, p: float = 0.33) -> str:
            chars = list(word)
            for i, c in enumerate(chars):
                if c.isalpha() and random.random() < p:
                    chars[i] = c.upper()
            return ''.join(chars)
        words = event.text.split()
        if len(words) <= 1:
            return event
        idx = random.randrange(1, len(words))

        change = random.choices(
            population=['no_change', 'all_sentence', 'firt_letter', 'entire_word', 'random_letters'],
            weights=   [w_no_change, w_all_sentence, w_firt_letter, w_entire_word, w_random_letters], k=1
        )[0]

        if change == 'no_change':
            return event
        elif change == 'all_sentence':
            words = [word.upper() for word in words]
        elif change == 'firt_letter':
            words[idx] = words[idx].capitalize()
        elif change == 'entire_word':
            words[idx] = words[idx].upper()
        elif change == 'random_letters':
            for i, word in enumerate(words):
                words[i] = uppercase_random_letters(word)

        event.text = " ".join(words)
        return event


    
    def keep_one_word(event: line.Dialogue, p: float = 0.1) -> line.Dialogue:
        """replace the event text by one word
        """
        if random.random() < p:
            text = re.sub(r'\{.*?\}', '', event.text)
            word_list = text.replace("\n", " ").split(" ")
            word = random.choice(word_list)
            event.text = word
        return event

    if isinstance(timestamp, float) or isinstance(timestamp, int):
        timestamp = timedelta(seconds=timestamp)

    for i, event in enumerate(event_list):
        event_list[i]= add_one_spe_char_word(
            event
        )
        event_list[i]= capitalize_sentence(
            event
        )
        event_list[i] = add_numbers(
            event
        )
        event_list[i] = add_any_spe_char(
            event
        )
        event_list[i] = add_prio_spe_char(
            event
        )
        event_list[i] = keep_one_word(
            event
        )
        event_list[i] = add_three_dots(
            event,
            p_three_dots_after=p_three_dots_after,
            p_three_dots_before=p_three_dots_before,
            timestamp=timestamp
        )
    
    return event_list
    
