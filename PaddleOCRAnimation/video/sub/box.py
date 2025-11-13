import PIL.Image as PILIMAGE
from PIL import ImageDraw
from random import randint


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
    
    def add_padding(self, padding: tuple[int, int, int, int],
                    image_size: tuple[int, int] | None = None):
        """Shift and optionally clamp the box coordinates when padding or cropping 
        is applied to the subtitle image.

        The padding is defined as (left, top, right, bottom):  
        - Positive values add transparent padding around the image, shifting the box 
        rightward or downward.  
        - Negative values indicate cropping (pixels removed from edges), shifting 
        the box leftward or upward.  

        When `image_size` is provided, the method also ensures that the box 
        coordinates remain within the image boundaries. Boxes that fall completely 
        outside after cropping are invalidated (set to [[0, 0], ...]).

        Args:
            padding (tuple[int, int, int, int]): Padding values (left, top, right, bottom). 
                Positive for expansion, negative for cropping.
            image_size (tuple[int, int] | None, optional): 
                The (width, height) of the image after padding/cropping. 
                If provided, coordinates are clamped to the image boundaries.

        Raises:
            ValueError: 
                - If `padding` is not a tuple of four integers.
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

        if image_size is not None:
            if (padding[2]<0): 
                self.bas_droit[0] = min(self.bas_droit[0], image_size[0])
                self.haut_droit[0] = self.bas_droit[0]
            if (padding[3]<0):
                self.bas_droit[1] = min(self.bas_droit[1], image_size[1])
                self.bas_gauche[1] = self.bas_droit[1]
        self.full_box = [
            self.haut_gauche,
            self.haut_droit,
            self.bas_droit,
            self.bas_gauche
        ]
        if self.bas_droit[0] <= self.bas_gauche[0]:
            # The box is considered non existant 
            self.full_box = [[0, 0], [0, 0], [0, 0], [0, 0]]
        if self.bas_droit[1] <= self.haut_droit[1]:
                self.full_box = [[0, 0], [0, 0], [0, 0], [0, 0]]

    def resize(self, scale: float):
        """Resizes the box coordinates by a given scale factor.

        Args:
            scale (float): Factor by which to scale the box coordinates.
        """
        def rescale_point(pt, scale):
            return [int(pt[0] * scale), int(pt[1] * scale)]

        self.haut_gauche = rescale_point(self.haut_gauche, scale=scale)
        self.haut_droit  = rescale_point(self.haut_droit, scale=scale)
        self.bas_droit   = rescale_point(self.bas_droit, scale=scale)
        self.bas_gauche  = rescale_point(self.bas_gauche, scale=scale)
        self.full_box = [self.haut_gauche, self.haut_droit, self.bas_droit, self.bas_gauche]

    def to_pil(
            self, size: tuple[int, int],
            show_only_borders: bool = False,
            color: tuple[int, int, int, int] | None = None,
        ) -> PILIMAGE.Image:
        """Render the box as a PIL RGBA image, either filled or outlined.

        If `show_only_borders` is False, the box is drawn as a semi-transparent
        highlight. Otherwise, only the border is drawn.

        Args:
            size (tuple[int, int]): Output image size as (width, height).
            show_only_borders (bool, optional): If `True`, only the box outline
                is drawn. If `False`, the box area is filled. Defaults to `False`.
            color (tuple[int, int, int, int] | None, optional): RGBA color to use
                for the box. If None, a random color is generated. When
                `show_only_borders` is `True`, the color is fully opaque; otherwise
                a semi-transparent alpha (≈100) is used.

        Raises:
            ValueError: If `color` is not a 4-tuple of integers in [0, 255].

        Returns:
            PIL.Image: An RGBA image of the box (filled or outlined).
        """
        img = PILIMAGE.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        color = color or (randint(1, 255), randint(1, 255), randint(1, 255), 100 if not show_only_borders else 255)
        if (
            not isinstance(color, tuple)
            or len(color) != 4
            or not all(0 <= col <= 255 for col in color)
           ):
            raise ValueError(
                "border_color doit être un tuple de taille 4 remplis"
                f" de int entre 0 et 255 (RGBA) ici {color}"
            )
        if show_only_borders:
            fill_color, outline_color = (0, 0, 0, 0), color
        else:
            fill_color, outline_color = color, (0, 0, 0, 0)

        draw.polygon(self.full_box, fill=fill_color, outline=outline_color)
        return img