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

logger = logging.getLogger(__name__)


def disturb_eventWithPil(events: eventWithPil, p:float = 0.15,
                             mean_band_size_perc:float = 0.2
                            ) -> eventWithPil:
    """
    Randomly adds transparent padding around an event image.

    With probability `p` for each side (left, top, right, bottom),
    a random padding width is sampled from a Gaussian distribution
    centered at `mean_band_size_perc` of the corresponding image dimension.
    The padding is applied via `event.add_padding()`.

    Args:
        events (eventWithPil): The event containing the image, text, and boxes.
        p (float, optional): Probability of adding padding on each side. Defaults to `0.15`.
        mean_band_size_perc (float, optional): Mean relative padding size per side. Defaults to `0.2`.

    Returns:
        eventWithPil: The modified event, potentially with added padding.
    """
    perc: list[float] = [0, 0, 0, 0]

    for i in range(0, len(perc), 1): 
        if random.random() < p:
               perc[i] = abs(random.gauss(mean_band_size_perc, 0.15))
    
    if perc == [0, 0, 0, 0]:
        return events
    
    im_w, im_h = events.image.size
    padding = (int(im_w*perc[0]), int(im_h*perc[1]), int(im_w*perc[2]), int(im_h*perc[3]))
    events.add_padding(padding=padding)
  
    return events



@overload
def crop_image(
    image: Image.Image, event_list: eventWithPilList,
    height_cut_ratio: float = 0.65, width_cut_ratio: float = 0.01,
    reverse:bool = False
    ) -> tuple[Image.Image, eventWithPilList]:
    ...
@overload
def crop_image(
    image: Image.Image, event_list: None = None,
    height_cut_ratio: float = 0.65, width_cut_ratio: float = 0.01,
    reverse:bool = False
) -> Image.Image:
    ...
def crop_image(
        image: Image.Image, event_list: eventWithPilList | None = None,
        height_cut_ratio: float = 0.65, width_cut_ratio: float = 0.01,
        reverse:bool = False
    ):
    """
    Randomly crops an image to simulate partial frame cuts that occur during OCR processing.
    The crop mainly removes content from the top of the image (since subtitles are usually at the bottom).
    If an event list with bounding boxes is provided, their coordinates are adjusted accordingly,
    and boxes falling outside the cropped area are removed.

    Args:
        image (Image.Image): Input image to crop.
        event_list (eventWithPilList | None, optional): 
            List of bounding boxes or events to adjust after cropping. Defaults to `None`.
        height_cut_ratio (float, optional): 
            Average proportion of image height to crop from the top. Defaults to `0.65`.
        width_cut_ratio (float, optional): 
            Average proportion of image width to crop from each side. Defaults to `0.01`.
        reverse (bool, optional):
            If `true`, everithing is reversed meaning we get the top part of the image instead of the bottom part. Defaults to `False`.

    Returns:
        Image.Image | tuple[Image.Image, eventWithPilList]: 
            The cropped image, and the updated event list if provided.
    """
    im_w, im_h = image.size
    cut_top = abs(int(random.gauss(height_cut_ratio, 0.05)* im_h))
    cut_sides = abs(int(random.gauss(width_cut_ratio , 0.035)* im_w))
    cut_bottom = 0

    if reverse :
        # we want the top the the image
        cut_bottom, cut_top = cut_top, cut_bottom

    im = image.crop((cut_sides, cut_top, im_w-cut_sides, im_h-cut_bottom))

    if event_list is not None: 
        event_list.add_padding((-cut_sides, -cut_top, -cut_sides, -cut_bottom))
        
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

def jpeg_compress(img: Image.Image, quality:int =10):
    """Sauvegarde sur RAM en JPEG (avec compression) et réouvre cette sauvegarde
    """
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
    if random.random() < 0.15:
        if event_list is None:
            img=crop_image(image=img)
        else:
            img, event_list = crop_image(image=img, event_list=event_list)
    elif random.random() < 0.15:
        if event_list is None:
            img=crop_image(image=img, reverse=True)
        else:
            img, event_list = crop_image(image=img, event_list=event_list, reverse=True)
    if random.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))
    if random.random() < 0.15:
        img = add_noise(img, std=random.uniform(2, 12))
    if random.random() < 0.20:
        img = jpeg_compress(img, quality=random.randint(15, 36))
    if random.random() < 0.1:
        img = salt_and_pepper(img).convert('RGBA') if img.mode == 'RGBA' else salt_and_pepper(img)
    if random.random()<0.20:
        if event_list is None:
            img=add_black_band(img=img)
        else:
            img, event_list = add_black_band(img=img, event_list=event_list)
    

    
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
    if random.random() < 0.30:
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

    if random.random() < 0.15:
        style.italic = not style.italic

    return style


def disturb_text(
        event_list: section.EventsSection,
        p_three_dots_before: float = 0.04,
        p_three_dots_after: float = 0.07,
        timestamp: float | timedelta | None = None
        ) -> section.EventsSection:
    """Randomly applies text disturbances to subtitle events to increase dataset robustness.

    Currently, it may add ellipses ("...") before or after dialogue lines with given
    probabilities. Future versions may include other modifications (e.g., typos,
    truncations, casing changes).

    Args:
        event_list (section.EventsSection): List of dialogue events to modify.
        p_three_dots_before (float, optional): Probability of adding "..." before text. Defaults to 0.04.
        p_three_dots_after (float, optional): Probability of adding "..." after text. Defaults to 0.07.
        timestamp (float | timedelta | None, optional): Optional time filter; only events active at this time are modified. Defaults to None.

    Returns:
        section.EventsSection: The modified EventsSection (same object, changed in place).
    """
    def add_three_dots(
            event: line.Dialogue,
            p_three_dots_before: float = 0.1,
            p_three_dots_after:float = 0.2,
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
            elif random.random() < p_point_after and not text.endswith(("...", "…", "!", "?", '.', ",")):
                text = text+'.'
                event.text = text
        return event 
    if isinstance(timestamp, float) or isinstance(timestamp, int):
        timestamp = timedelta(seconds=timestamp)
    for i, event in enumerate(event_list):
        event_list[i] = add_three_dots(
            event,
            p_three_dots_after=p_three_dots_after,
            p_three_dots_before=p_three_dots_before,
            timestamp=timestamp
        )
    
    return event_list
    
