import ctypes
import ctypes.util

from collections import Counter

from datetime import timedelta

from PIL import Image as PILIMAGE
from PIL import ImageDraw

from ass import Document

from functools import cached_property

from os.path import join, dirname, abspath

from platform import system

import numpy as np

from typing import Iterator, Tuple, ClassVar

import importlib.resources
import importlib.resources._legacy


system = system()
fileRoot = dirname(dirname(abspath(__file__)))
if system == "Windows":
    with importlib.resources.as_file(importlib.resources.files("PaddleOCRAnimation.libs.Windows")/"libass-9.dll") as lib_path:
        _libass = ctypes.cdll.LoadLibrary(lib_path)
    _libc = ctypes.cdll.msvcrt
elif system == "Linux":
    for lib in ["libgraphite2.so.3", "libfribidi.so.0", "libharfbuzz.so.0", "libm.so.6",
                "libfontconfig.so.1", "libfreetype.so.6", "libc.so.6", "libz.so.1", "libbz2.so.1.0",
                "libpng16.so.16", "libbrotlidec.so.1", "libbrotlicommon.so.1"]:
        try:
            with importlib.resources.as_file(importlib.resources.files("PaddleOCRAnimation.libs.linux") / lib) as lib_path:
                ctypes.cdll.LoadLibrary(lib_path)
        except OSError as e:
            raise RuntimeError(f"Échec du chargement de {lib} : {e}")
    with importlib.resources.as_file(importlib.resources.files("PaddleOCRAnimation.libs.linux")/ "libass.so.9.4.1") as lib_path:
                _libass = ctypes.cdll.LoadLibrary(lib_path)

    _libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("c"))
else:
    raise RuntimeError(f"OS non supporté : {system}")


if _libass is None:
    raise ImportError("Librairie Libass non trouvée")


if _libc is None:
    raise ImportError("Librairie C non trouvée (pas instalée ou pas dans le PATH)")

class Box:
    """
    Représente une boîte englobante (rectangle ou polygone) associée à
    un élément rendu (texte, outline, etc.).
    Permet de manipuler et de dessiner la boîte sur une image PIL.
    """
    def __init__(self, haut_gauche, haut_droit, bas_droit, bas_gauche, event_type=None):
        for coin in [haut_gauche, haut_droit, bas_droit, bas_gauche]:
            for element in coin:
                if not isinstance(element, int) or element < 0:
                    raise ValueError(f"Boxes coin should be a positive int (here {element})")
        self.haut_gauche: list[int] = haut_gauche
        self.haut_droit: list[int] = haut_droit
        self.bas_droit: list[int] = bas_droit
        self.bas_gauche: list[int] = bas_gauche
        self.full_box: list[list[int]] = [
            self.haut_gauche,
            self.haut_droit,
            self.bas_droit,
            self.bas_gauche
            ]
        self.event_type: int | None = event_type
    
    def __repr__(self):
        return f'{self.full_box}'

    def get_bounding_box(self):
        """Retourne le plus petit rectangle aligné avec les axes qui contient entièrement la boîte
        """
        xs = [point[0] for point in self.full_box]
        ys = [point[1] for point in self.full_box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return x_min, y_min, x_max, y_max  # gauche, haut, droite, bas
    
    def add_padding(self, padding: tuple[int, int, int, int]):
        """Shift the box coordinates according to padding or cropping applied to the image.

        The padding is defined as (left, top, right, bottom). 
        Positive values indicate that padding is added to the image, 
        which moves the box rightward or downward.
        Negative values indicate that a region is cropped (removed) from the image, 
        which moves the box leftward or upward.

        The box coordinates are clamped to remain non-negative (>= 0) after the shift.

        Args:
            padding (tuple[int, int, int, int]): Padding values (left, top, right, bottom). 
                Can include negative values to represent cropping.

        Raises:
            ValueError: If `padding` is not a tuple of four integers.
    """
        if (
            not isinstance(padding, tuple) 
            or not all([isinstance(a, int) for a in padding]) 
            or not len(padding)==4
        ):
            raise ValueError(f'padding should be a tuple with 4 int, here {padding}')
        
        padding_left, padding_top, _, _ = padding

        self.haut_gauche = [max(self.haut_gauche[0]+padding_left, 0), max(self.haut_gauche[1]+padding_top, 0)]
        self.haut_droit = [max(self.haut_droit[0]+padding_left, 0), max(self.haut_droit[1]+padding_top, 0)]
        self.bas_droit = [max(self.bas_droit[0]+padding_left, 0), max(self.bas_droit[1]+padding_top, 0)]
        self.bas_gauche = [max(self.bas_gauche[0]+padding_left, 0), max(self.bas_gauche[1]+padding_top, 0)]

        self.full_box = [
            self.haut_gauche,
            self.haut_droit,
            self.bas_droit,
            self.bas_gauche
        ]


    def to_pil(self, size,
               border_color: int | tuple[int] = (255, 0, 0, 255),
               border_width=3, use_type: bool = True) -> PILIMAGE.Image:
        """
        Génère une image PIL contenant la boîte (polygone) représentée par cet objet Box.

        Args:
            size (tuple[int, int]): Taille de l'image de sortie (largeur, hauteur).
            border_color (int | tuple[int], optional): Couleur du contour. Peut être un entier
                (0 à 3 pour un mapping couleur prédéfini)
                ou un tuple RGBA (4 entiers entre 0 et 255). Par défaut (255, 0, 0, 255).
            border_width (int, optional): Largeur du contour en pixels. Par défaut 3.
            use_type (bool, optional): Si True et que event_type est défini, utilise event_type
                pour déterminer la couleur du contour. Par défaut True.

        Raises:
            ValueError: Si border_color est un int mais n'est pas dans [0, 1, 2, 3].
            ValueError: Si border_color n'est pas un tuple RGBA valide.

        Returns:
            PIL.Image: Image PIL RGBA contenant la boîte dessinée avec le contour spécifié.
        """
        img = PILIMAGE.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        if use_type and self.event_type is not None:
            border_color = self.event_type
        if isinstance(border_color, int):
            if border_color not in [0, 1, 2, 3]:
                raise ValueError(
                    f"Si border_color est un int, il doit être dans 0, 1, 2, 3. Ici {border_color}"
                )
            colorMap = {
                0: (0, 0, 255, 255),
                1: (255, 0, 0, 255),
                2: (0, 255, 0, 255),
                3: (255, 255, 0, 255)
            }
            border_color = colorMap[border_color]
        if (
            not isinstance(border_color, tuple)
            or len(border_color) != 4
            or not all(0 <= col <= 255 for col in border_color)
           ):
            raise ValueError(
                "border_color doit être un tuple de taille 4 remplis"
                f" de int entre 0 et 255 (RGBA) ici {border_color}"
            )

        draw.polygon(self.full_box, outline=border_color)

        # Épaissir le contour si nécessaire
        if border_width > 1:
            for i in range(len(self.full_box)):
                draw.line([self.full_box[i], self.full_box[(i+1) % len(self.full_box)]],
                          fill=border_color, width=border_width)

        return img


class Image(ctypes.Structure):
    """
    Structure représentant une image Libass (bitmap d'un élément rendu).
    Contient les informations de position, couleur, type et accès pixel à pixel.
    """
    TYPE_CHARACTER: ClassVar[int] = 0
    TYPE_OUTLINE: ClassVar[int] = 1
    TYPE_SHADOW: ClassVar[int] = 2

    @cached_property
    def bitmapNumpy(self):
        h, _, stride = self.h, self.w, self.stride
        buffer = ctypes.string_at(self.bitmap, h * stride)

        # Convertir en tableau NumPy
        arr = np.frombuffer(buffer, dtype=np.uint8)

        # Redimensionner en (hauteur, stride)
        arr = arr.reshape((h, stride))
        image_np = arr[:, :self.w]
        return image_np

    @property
    def rgba(self) -> Tuple[int, int, int, int]:
        color = self.color

        a = color & 0xff
        color >>= 8

        b = color & 0xff
        color >>= 8

        g = color & 0xff
        color >>= 8

        r = color & 0xff

        return (r, g, b, a)

    @property
    def typeText(self) -> str:
        type = self.type
        typeDict = {
            self.TYPE_CHARACTER: "TYPE_CHARACTER",
            self.TYPE_OUTLINE: "TYPE_OUTLINE",
            self.TYPE_SHADOW: "TYPE_SHADOW"
        }

        return typeDict[type]

    def __getitem__(self, loc) -> int:
        x, y = loc
        # Numpy fait la convertion (hauteur, largeur) et non (largeur, hauteur)
        return self.bitmapNumpy[y, x]

    def to_pil(
            self, SIZE: tuple[int, int] | None = None
        ) -> PILIMAGE.Image:
        """
        Convertit l'image Libass courante en une image PIL RGBA.

        Args:
            SIZE (tuple[int, int]): Taille de l'image de sortie (largeur, hauteur).

        Returns:
            PIL.Image: Image PIL RGBA contenant le rendu de l'image à la position
                et couleur spécifiées.
        """
        # TODO : gérer les multilignes 
        if SIZE == (0, 0):
            SIZE = None
        width, height = self.w, self.h
        r, g, b, a = self.rgba
        dist_x, dist_y = (self.dst_x, self.dst_y) if SIZE is not None else (0, 0)

        if SIZE is not None and SIZE !=(0, 0):
            im = PILIMAGE.new("RGBA", (SIZE[0], SIZE[1]))
        else: # we just want the image with the padding
            im = PILIMAGE.new("RGBA", (width, height))
        pixels = im.load()

        for y in range(height):
            for x in range(width):
                alpha_bitmap = int(self[x, y])

                alpha = alpha_bitmap*(256-a)//256
                if alpha > 0:
                    pixels[x+dist_x, y+dist_y] = (r, g, b, alpha)
                else:
                    pixels[x, y] = (0, 0, 0, 0)

        return im

    def to_box(
            self, 
            padding: tuple[int, int, int, int] = (0, 0, 0, 0),
            xy_offset: tuple[int, int] = (0,0)
        ) -> Box:
        """
        Calcule la boîte englobante (bounding box) de l'image courante.
        Cette fonction détermine les coordonnées exactes de la boîte couvrant
        entièrement l'image (ou le calque) associée à l'événement courant.
        Elle permet également d'appliquer un décalage (`xy_offset`) ou un
        agrandissement (`padding`) pour ajuster la position et la taille de la box.

        Args:
            padding (tuple[int, int, int, int]): 
                Marges supplémentaires à appliquer sur chaque bord de la boîte 
                sous la forme (gauche, haut, droite, bas).  
                Par défaut, aucun padding n'est appliqué `(0, 0, 0, 0)`.
            
            xy_offset (tuple[int, int]): 
                Décalage (x, y) à soustraire aux coordonnées de la boîte.  
                Cela permet de repositionner la box sur une image plus petite
                (par exemple lors du recadrage d’un rendu local d’événement).  
                Par défaut `(0, 0)`.

        Returns:
            Box: 
                Objet `Box` représentant la boîte englobante de l'image, avec 
                les coordonnées ajustées et le type d'image associé 
                (texte, contour, etc.).
        """
        x1 = self.dst_x - xy_offset[0]
        x2 = x1 + self.w
        y1 = self.dst_y - xy_offset[1]
        y2 = y1 + self.h

        box = Box(
            [max(x1 - padding[0], 0), max(y1 - padding[1], 0)],
            [max(x2 + padding[2], 0), max(y1 - padding[1], 0)],
            [max(x2 + padding[2], 0), max(y2 + padding[3], 0)],
            [max(x1 - padding[0], 0), max(y2 + padding[3], 0)],
            event_type=self.type
        )
        return box


Image._fields_ = [
    ("w", ctypes.c_int),
    ("h", ctypes.c_int),
    ("stride", ctypes.c_int),
    ("bitmap", ctypes.POINTER(ctypes.c_char)),
    ("color", ctypes.c_uint32),
    ("dst_x", ctypes.c_int),
    ("dst_y", ctypes.c_int),
    ("next_ptr", ctypes.POINTER(Image)),
    ("type", ctypes.c_int)
]


class ImageSequence(object):
    """
    Séquence chaînée d'objets Image générés par Libass pour un frame donné.
    Permet d'itérer, d'accéder par index, de composer en image PIL
        ou de calculer la boîte englobante globale.
    """
    def __init__(self, renderer, head_ptr):
        self.renderer = renderer
        self.head_ptr = head_ptr

    def __iter__(self) -> Iterator[Image]:
        cur = self.head_ptr
        while cur:
            yield cur.contents
            cur = cur.contents.next_ptr

    def __getitem__(self, index: int) -> Image:
        cur = self.head_ptr
        i = 0
        while cur:
            if i == index:
                return cur.contents
            cur = cur.contents.next_ptr
            i += 1
        raise IndexError('Index out of range')

    def __len__(self) -> int:
        i = 0
        for image in self:
            i += 1
        return i
    
    def get_distances_list(self) -> tuple[int, int, int, int]:
        images_h, images_w, images_dst_x, images_dst_y = [], [], [], []
        for image in self:
            images_h.append(image.h)
            images_w.append(image.w)
            images_dst_x.append(image.dst_x)
            images_dst_y.append(image.dst_y)
        smallest_dist_x = min(images_dst_x)
        smallest_dist_y = min(images_dst_y)
        biggest_h, biggest_w = -1, -1
        for i in range(len(images_h)):
            biggest_h, biggest_w = max(images_dst_y[i]+images_h[i], biggest_h), max(images_dst_x[i]+images_w[i], biggest_w)
        return (biggest_h, biggest_w, smallest_dist_x, smallest_dist_y)

    def to_pil(self, size: Tuple[int, int]) -> PILIMAGE.Image:
        """Convertit un ensemble d'images (par exemple des couches de rendu d'un événement ASS)
        en une image PIL unique, prête à être affichée ou sauvegardée.

        Cette fonction fusionne les différentes couches (textes, contours, etc.)
        associées à un événement de sous-titre pour produire une image RGBA finale.

        Args:
            size (Tuple[int, int]): Taille de l'image finale (largeur, hauteur).
                - Si `size` est `None` ou `(0, 0)`, l'image résultante correspond uniquement 
                à la taille minimale nécessaire pour contenir l'événement (bounding box locale).
                - Si `size` est spécifiée, elle doit correspondre à la taille complète de la 
                vidéo d'origine afin d'assurer un positionnement correct des sous-titres.

        Returns:
            PILIMAGE.Image: Une image PIL en mode RGBA représentant le rendu complet 
            de l'événement de sous-titre.
        """
        # TODO : refaire documentation 
        # base = PILIMAGE.new("RGBA", size, (255, 255, 255, 0))
        # Problème potentiel de tris ? Si les outlines sont après le texte ?
        # Pourtant FFMPEG ne fait pas le tris non plus
        # Normalement c'est bon (voir Libas render_text)
        biggest_h, biggest_w, smallest_dist_x, smallest_dist_y = self.get_distances_list()
        
        if size is not None and size != (0,0):
            base = PILIMAGE.new("RGBA", size)
            temp = base.copy()
            smallest_dist_x, smallest_dist_y = 0,0
        else: 
            base = PILIMAGE.new("RGBA", (biggest_w-smallest_dist_x, biggest_h-smallest_dist_y))
            temp = base.copy()

        for i, image in enumerate(self):
            # TODO : if multinile
            pil_image = image.to_pil(size)
            if size is not None and size != (0,0):
                temptwo = pil_image   
            else:
                temptwo = temp.copy()
                temptwo.paste(pil_image, (image.dst_x-smallest_dist_x, image.dst_y-smallest_dist_y))          
            base = PILIMAGE.alpha_composite(
                base, temptwo
            )
        return base

    def to_box(
            self, padding: tuple[int, int, int, int] = (0, 0, 0, 0),
            xy_offset: tuple[int, int] = (0,0)
        ) -> Box:
        """
        Calcule la boîte englobante (bounding box) de toutes les images de la séquence.
        Args:
            padding (tuple[int, int, int, int]): permet d'agrendir la boite
                (droite, haute, gauche, bas), par défaut aucun padding.

        Returns:
            Box: Objet Box représentant la boîte englobante de la séquence,
                 avec un type correspondant au type d'image le plus bas (texte, outline, etc.).
        """
        x1 = float('inf')
        x2 = 0
        y1 = float('inf')
        y2 = 0

        type = 2
        for image in self:
            fullBox = image.to_box(padding=(0, 0, 0, 0), xy_offset=xy_offset).full_box
            type = image.type if image.type < type else type

            x1 = min(x1, fullBox[0][0], fullBox[3][0])

            x2 = max(x2, fullBox[1][0], fullBox[2][0])

            y1 = min(y1, fullBox[0][1], fullBox[1][1])

            y2 = max(y2, fullBox[2][1], fullBox[3][1])
        return Box(
            [max(x1 - padding[0], 0), max(y1 - padding[1], 0)], 
            [max(x2 + padding[2], 0), max(y1 - padding[1], 0)],
            [max(x2 + padding[2], 0), max(y2 + padding[3], 0)], 
            [max(x1 - padding[0], 0), max(y2 + padding[3], 0)],
            event_type=type
        )

    def to_singleline_boxes(
            self, padding: tuple[int, int, int, int] = (0, 0, 0, 0),
            xy_offset: tuple[int, int] = (0,0)
        ) -> list[Box]:
        """
        Calcule les boites de toutes les images de la séquences, une boite par ligne.
        Args:
            padding (tuple[int, int, int, int]): permet d'agrendir la boite
                (droite, haute, gauche, bas), par défaut aucun padding.

        Returns:
            list[Box]: Une liste de boite (toute de types character).
        """
        box_characters = []
        for image in self:
            box = image.to_box(padding=padding, xy_offset=xy_offset)
            if box.event_type == image.TYPE_CHARACTER:
                box_characters.append(box)
        # Les boxes outlines et shadow sont bien trop larges
        # les boxes characters sont un peu trop petites mais restent les meilleures
        return box_characters


def _make_libass_setter(name, types):
    fun = _libass[name]
    fun.argtypes = [ctypes.c_void_p] + types

    def setter(self, v):
        if len(types) == 1:
            fun(ctypes.byref(self), v)
        else:
            fun(ctypes.byref(self), *v)
        self._internal_fields[name] = v

    return setter


def _make_libass_property(name, types):
    def getter(self):
        return self._internal_fields.get(name)

    return property(getter, _make_libass_setter(name, types))


class Context(ctypes.Structure):
    def __new__(self) -> 'Context':
        return _libass.ass_library_init().contents

    def __init__(self):
        self._internal_fields = {}

        self._style_overrides_buffers = []

        if not ctypes.byref(self):
            raise RuntimeError("could not initialize libass")

        self.extract_fonts = False
        self.style_overrides = []

    def __del__(self):
        _libass.ass_library_done(ctypes.byref(self))

    fonts_dir = _make_libass_property("ass_set_fonts_dir", [
        ctypes.c_char_p
    ])
    extract_fonts = _make_libass_property("ass_set_extract_fonts", [
        ctypes.c_int
    ])

    @property
    def style_overrides(self):
        return [buf.value for buf in self._style_overrides_buffers]

    @style_overrides.setter
    def style_overrides(self, xs):
        self._style_overrides_buffers = [ctypes.create_string_buffer(x)
                                         for x in xs]

        if self._style_overrides_buffers:
            ptr = (ctypes.c_char_p * len(self._style_overrides_buffers))(*[
                ctypes.addressof(buf)
                for buf in self._style_overrides_buffers
            ])
        else:
            ptr = ctypes.POINTER(ctypes.c_char_p)()

        _libass.ass_set_style_overrides(
            ctypes.byref(self),
            ptr)

    def make_renderer(self) -> 'Renderer':
        """ Make a renderer instance for rendering tracks. """
        renderer = _libass.ass_renderer_init(ctypes.byref(self)).contents
        renderer._after_init(self)
        return renderer

    def parse_to_track(self, data, codepage="UTF-8"):
        """ Parse ASS data to a track. """
        return _libass.ass_read_memory(ctypes.byref(self), data, len(data),
                                       codepage.encode("utf-8")).contents

    def make_track(self) -> 'Track':
        track = _libass.ass_new_track(ctypes.byref(self)).contents
        track._after_init(self)
        return track


class Renderer(ctypes.Structure):
    SHAPING_SIMPLE = 0
    SHAPING_COMPLEX = 1

    HINTING_NONE = 0
    HINTING_LIGHT = 1
    HINTING_NORMAL = 2
    HINTING_NATIVE = 3

    def _after_init(self, ctx):
        self._ctx = ctx
        self._fonts_set = False
        self._internal_fields = {}

        self.frame_size = (640, 480)
        self.storage_size = (640, 480)
        self.margins = (0, 0, 0, 0)
        self.use_margins = True
        self.font_scale = 1
        self.line_spacing = 0
        self.pixel_aspect = 1.0

    def __del__(self):
        _libass.ass_renderer_done(ctypes.byref(self))

    frame_size = _make_libass_property("ass_set_frame_size", [
        ctypes.c_int,
        ctypes.c_int
    ])
    storage_size = _make_libass_property("ass_set_storage_size", [
        ctypes.c_int,
        ctypes.c_int
    ])
    shaper = _make_libass_property("ass_set_shaper", [
        ctypes.c_int
    ])
    margins = _make_libass_property("ass_set_margins", [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ])
    use_margins = _make_libass_property("ass_set_use_margins", [
        ctypes.c_int
    ])
    pixel_aspect = _make_libass_property("ass_set_pixel_aspect", [
        ctypes.c_double
    ])
    aspect_ratio = _make_libass_property("ass_set_aspect_ratio", [
        ctypes.c_double,
        ctypes.c_double
    ])
    font_scale = _make_libass_property("ass_set_font_scale", [
        ctypes.c_double
    ])
    hinting = _make_libass_property("ass_set_hinting", [
        ctypes.c_int
    ])
    line_spacing = _make_libass_property("ass_set_line_spacing", [
        ctypes.c_double
    ])
    line_position = _make_libass_property("ass_set_line_position", [
        ctypes.c_double
    ])

    def set_fonts(self, default_font=None, default_family=None,
                  fontconfig_config=None, update_fontconfig=None):
        fc = fontconfig_config is not None

        if update_fontconfig is None:
            update_fontconfig = fontconfig_config is not None

        if default_font is not None:
            default_font = default_font.encode("utf-8")

        if default_family is not None:
            default_family = default_family.encode("utf-8")

        _libass.ass_set_fonts(ctypes.byref(self), default_font, default_family,
                              fc, fontconfig_config.encode("utf-8") or "",
                              update_fontconfig)
        self._fonts_set = True

    def update_fonts(self):
        if not self._fonts_set:
            raise RuntimeError("set_fonts before updating them")
        _libass.ass_fonts_update(ctypes.byref(self))

    set_cache_limits = _make_libass_setter("ass_set_cache_limits", [
        ctypes.c_int,
        ctypes.c_int
    ])

    @staticmethod
    def timedelta_to_ms(td):
        return int(td.total_seconds()) * 1000 + td.microseconds // 1000

    def render_frame(self, track, now) -> ImageSequence:
        if not self._fonts_set:
            raise RuntimeError("set_fonts before rendering")
        head = _libass.ass_render_frame(ctypes.byref(self),
                                        ctypes.byref(track),
                                        Renderer.timedelta_to_ms(now),
                                        ctypes.POINTER(ctypes.c_int)())
        return ImageSequence(self, head)

    def set_all_sizes(self, size):
        self.frame_size = size
        self.storage_size = size
        self.pixel_aspect = 1.0


class Style(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("fontname", ctypes.c_char_p),
        ("fontsize", ctypes.c_double),
        ("primary_color", ctypes.c_uint32),
        ("secondary_color", ctypes.c_uint32),
        ("outline_color", ctypes.c_uint32),
        ("back_color", ctypes.c_uint32),
        ("bold", ctypes.c_int),
        ("italic", ctypes.c_int),
        ("underline", ctypes.c_int),
        ("strike_out", ctypes.c_int),
        ("scale_x", ctypes.c_double),
        ("scale_y", ctypes.c_double),
        ("spacing", ctypes.c_double),
        ("angle", ctypes.c_double),
        ("border_style", ctypes.c_int),
        ("outline", ctypes.c_double),
        ("shadow", ctypes.c_double),
        ("alignment", ctypes.c_int),
        ("margin_l", ctypes.c_int),
        ("margin_r", ctypes.c_int),
        ("margin_v", ctypes.c_int),
        ("encoding", ctypes.c_int),
        ("treat_fontname_as_pattern", ctypes.c_int),
        ("blur", ctypes.c_double),
        ("justify", ctypes.c_double)
    ]

    @staticmethod
    def numpad_align(val):
        v = (val - 1) // 3
        if v != 0:
            v = 3 - v
        res = ((val - 1) % 3) + 1
        res += v * 4
        return res

    def _after_init(self, track):
        self._track = track

    def populate(self, style):
        self.name = style.name.encode("utf-8")
        self.fontname = style.fontname.encode("utf-8")
        self.fontsize = style.fontsize
        self.primary_color = style.primary_color.to_int()
        self.secondary_color = style.secondary_color.to_int()
        self.outline_color = style.outline_color.to_int()
        self.back_color = style.back_color.to_int()
        self.bold = style.bold
        self.italic = style.italic
        self.underline = style.underline
        self.strike_out = style.strike_out
        self.scale_x = style.scale_x / 100.0
        self.scale_y = style.scale_y / 100.0
        self.spacing = style.spacing
        self.angle = style.angle
        self.border_style = style.border_style
        self.outline = style.outline
        self.shadow = style.shadow
        self.alignment = Style.numpad_align(style.alignment)
        self.margin_l = style.margin_l
        self.margin_r = style.margin_r
        self.margin_v = style.margin_v
        self.encoding = style.encoding


class Event(ctypes.Structure):
    _fields_ = [
        ("start_ms", ctypes.c_longlong),
        ("duration_ms", ctypes.c_longlong),
        ("read_order", ctypes.c_int),
        ("layer", ctypes.c_int),
        ("style_id", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("margin_l", ctypes.c_int),
        ("margin_r", ctypes.c_int),
        ("margin_v", ctypes.c_int),
        ("effect", ctypes.c_char_p),
        ("text", ctypes.c_char_p),
        ("render_priv", ctypes.c_void_p)
    ]

    def _after_init(self, track):
        self._track = track

    @property
    def start(self):
        return timedelta(milliseconds=self.start_ms)

    @start.setter
    def start(self, td):
        self.start_ms = Renderer.timedelta_to_ms(td)

    @property
    def duration(self):
        return timedelta(milliseconds=self.duration_ms)

    @duration.setter
    def duration(self, td):
        self.duration_ms = Renderer.timedelta_to_ms(td)

    @property
    def style(self):
        return self._track.styles[self.style_id].name

    @style.setter
    def style(self, v):
        # NOTE: linear time every time we want to add a style
        idStart = 0
        if self._track._stylesCounter[b'Default'] == 2:
            # Permer d'éviter le tyle Default ajouté FAIRE DEFAULT STYLE
            idStart = 1
        for i, style in enumerate(self._track.styles[idStart:], start=idStart):
            if style.name == v.encode("utf-8"):
                self.style_id = i
                return

        raise ValueError(f"style {v} not found")

    def populate(self, event):
        self.start = event.start
        self.duration = event.end - event.start
        self.layer = event.layer
        self.style = event.style
        self.name = event.name.encode("utf-8")
        self.margin_l = event.margin_l
        self.margin_r = event.margin_r
        self.margin_v = event.margin_v
        self.effect = event.effect.encode("utf-8")
        self.text = event.text.encode("utf-8")


class Track(ctypes.Structure):
    TYPE_UNKNOWN = 0
    TYPE_ASS = 1
    TYPE_SSA = 2

    _fields_ = [
        ("n_styles", ctypes.c_int),
        ("max_styles", ctypes.c_int),
        ("n_events", ctypes.c_int),
        ("max_events", ctypes.c_int),
        ("styles_arr", ctypes.POINTER(Style)),
        ("events_arr", ctypes.POINTER(Event)),
        ("style_format", ctypes.c_char_p),
        ("event_format", ctypes.c_char_p),
        ("track_type", ctypes.c_int),
        ("play_res_x", ctypes.c_int),
        ("play_res_y", ctypes.c_int),
        ("timer", ctypes.c_double),
        ("wrap_style", ctypes.c_int),
        ("scaled_border_and_shadow", ctypes.c_int),
        ("kerning", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("ycbcr_matrix", ctypes.c_int),
        ("default_style", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("library", ctypes.POINTER(Context)),
        ("parser_priv", ctypes.c_void_p),
        ("layout_res_x", ctypes.c_int),  # manquant à la base
        ("layout_res_y", ctypes.c_int)   # idem
    ]

    def _after_init(self, ctx):
        self._ctx = ctx

    @property
    def styles(self):
        if self.n_styles == 0:
            return []
        return ctypes.cast(self.styles_arr,
                           ctypes.POINTER(Style * self.n_styles)).contents

    @property
    def events(self):
        if self.n_events == 0:
            return []
        return ctypes.cast(self.events_arr,
                           ctypes.POINTER(Event * self.n_events)).contents

    def make_style(self):
        style = self.styles_arr[_libass.ass_alloc_style(ctypes.byref(self))]
        style._after_init(self)
        return style

    def make_event(self) -> Event:
        event = self.events_arr[_libass.ass_alloc_event(ctypes.byref(self))]
        event._after_init(self)
        return event

    def __del__(self):
        # XXX: we can't use ass_free_track because it assumes we've allocated
        #      our strings in the heap (wat), so we just free them with libc.
        _libc.free(self.styles_arr)
        _libc.free(self.events_arr)
        _libc.free(ctypes.byref(self))

    def populate(self, doc: Document):
        """ Convert an ASS document to a track. """
        self.type = Track.TYPE_ASS

        self.play_res_x = doc.fields.get('PlayResX', 0)
        self.play_res_y = doc.fields.get('PlayResY', 0)
        # les fichiers str convertis en ass n'ont pas de wrap_style
        self.wrap_style = doc.sections["Script Info"].get("WrapStyle", 0)
        self.scaled_border_and_shadow = (
            doc.fields.get("ScaledBorderAndShadow", '').lower() == "yes"
            )

        self.style_format = ", ".join(
            doc.SECTIONS[doc.STYLE_ASS_HEADER].field_order
        ).encode("utf-8")
        self.event_format = ", ".join(
            doc.SECTIONS[doc.EVENTS_HEADER].field_order
        ).encode("utf-8")

        for d_style in doc.styles:
            style = self.make_style()
            style.populate(d_style)

        self._stylesCounter = Counter([style.name for style in self.styles])

        for d_event in doc.events:
            if d_event.TYPE != "Dialogue":
                continue
            event = self.make_event()
            event.populate(d_event)


_libc.free.argtypes = [ctypes.c_void_p]

_libass.ass_library_init.restype = ctypes.POINTER(Context)

_libass.ass_library_done.argtypes = [ctypes.POINTER(Context)]

_libass.ass_renderer_init.argtypes = [ctypes.POINTER(Context)]
_libass.ass_renderer_init.restype = ctypes.POINTER(Renderer)

_libass.ass_renderer_done.argtypes = [ctypes.POINTER(Renderer)]

_libass.ass_new_track.argtypes = [ctypes.POINTER(Context)]
_libass.ass_new_track.restype = ctypes.POINTER(Track)

_libass.ass_set_style_overrides.argtypes = [
    ctypes.POINTER(Context),
    ctypes.POINTER(ctypes.c_char_p)
]
_libass.ass_set_fonts.argtypes = [
    ctypes.POINTER(Renderer),
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int
]
_libass.ass_fonts_update.argtypes = [ctypes.POINTER(Renderer)]

_libass.ass_render_frame.argtypes = [
    ctypes.POINTER(Renderer),
    ctypes.POINTER(Track),
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_int)
]
_libass.ass_render_frame.restype = ctypes.POINTER(Image)

_libass.ass_read_memory.argtypes = [
    ctypes.POINTER(Context),
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_char_p
]
_libass.ass_read_memory.restype = ctypes.POINTER(Track)

_libass.ass_alloc_style.argtypes = [ctypes.POINTER(Track)]
_libass.ass_alloc_style.restype = ctypes.c_int

_libass.ass_alloc_event.argtypes = [ctypes.POINTER(Track)]
_libass.ass_alloc_event.restype = ctypes.c_int
