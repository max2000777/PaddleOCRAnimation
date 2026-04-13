from PIL import Image
from .sub.box import Box
import numpy as np
import math

def detect_text_line_xyxy(
        sub_image: Image.Image, 
        multiline: bool = True, 
        threshold_percent:float = 0.01,
        libass_box: list[Box] | None = None
    ) -> list[tuple[int, int, int, int]]:
        """
        Detects bounding boxes for text lines in a transparent subtitle image.

        This function analyzes the alpha channel of a subtitle image (typically rendered by libass)
        to estimate the bounding boxes of each text line. It supports both heuristic detection based
        on pixel projection and guided splitting using precomputed libass bounding boxes.

        Args:
            sub_image (Image.Image): Transparent image of a single subtitle event (one or more lines).
            multiline (bool, optional): If True, treat the subtitle as a single multi-line block.
                If False, detect and separate individual text lines. Defaults to True.
            threshold_percent (float, optional): Threshold ratio (relative to the maximum horizontal
                projection) used to detect line boundaries when `libass_box` is not provided.
                Defaults to 0.01.
            libass_box (list[Box] | None, optional): Optional list of bounding boxes from libass
                to refine or guide the line splitting. Defaults to `None`.

        Returns:
            list[tuple[int, int, int, int]]: List of bounding boxes (x1, y1, x2, y2) for each
            detected text line in absolute image coordinates.
        """
        #the image should be the transpatent image of one event (one sub) with one or more line

        import numpy as np
        # we need to find the box  of the text, because the image is transparent it is relativly easy
        alpha = sub_image.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            return []  # there is not text on the image
        
        if multiline:
            # beacause multiline is allowed, no further modification needs to be done
            return [bbox]

        cropped = alpha.crop(bbox)
        arr = np.array(cropped)
        binary = (arr > 0).astype(np.uint8)
        projection = binary.sum(axis=1)
        threshold = np.max(projection) * threshold_percent
        smooth = np.convolve(projection, np.ones(5)/5, mode='same')
        line_boxes = []
        in_line = False
        start = 0

        if libass_box:
            # We already have libass boxes, we can use them to split lines 
            # we asume that all the boxes are for the same event
            boxes_y_mean = []
            for box in libass_box:
                boxes_y_mean.append((box.haut_droit[1]+box.bas_droit[1])//2)
            boxes_y_mean = sorted(boxes_y_mean)
            cut_y = 0 # we asume that the boxes are sorted by h
            for y1, y2 in zip(boxes_y_mean, boxes_y_mean[1:]):
                # we know that we are between boxes, we just need to find the cutoff
                search_top = max(0, y1 - bbox[1] - 10)
                search_bottom = min(smooth.shape[0], y2 - bbox[1] + 10)
                local_smooth = smooth[search_top:search_bottom]

                if max(local_smooth)>1e-6:
                    local_smooth = local_smooth/max(local_smooth)

                y_range = np.arange(search_top, search_bottom)
                mid = (search_top + search_bottom) // 2
                distance_to_mid = np.abs(y_range - mid)

                
                score = 0.8 * local_smooth + 0.2 * (distance_to_mid / distance_to_mid.max())
                last_cut = cut_y
                cut_y = y_range[np.argmin(score)]

                box_projecton = projection[last_cut:cut_y]

                line_boxes.append((
                    int(last_cut+np.where(box_projecton > 0)[0][0]), # first id not null 
                    int(last_cut + np.where(box_projecton > 0)[0][-1]) # last id not null
                ))
            
            # last line
            box_projecton = projection[cut_y:bbox[3]-bbox[1]]
            line_boxes.append((
                int(cut_y+np.where(box_projecton > 0)[0][0]),
                int(cut_y+np.where(box_projecton > 0)[0][-1])
            ))

        else:
            for y, val in enumerate(projection):
                if val > threshold  and not in_line:
                    in_line = True
                    start = y
                elif val <=threshold and in_line:
                    in_line = False
                    end = y
                    line_boxes.append((start, end))
    
            if in_line:
                line_boxes.append((start, len(projection)))

        abs_boxes = []
        for (y1, y2) in line_boxes:
            line_region = binary[y1:y2, :]
            h = line_region.shape[0]
            core_top = int(h * 0.10) # above line can be annoying
            core_bottom = int(h * 0.95)
            core = line_region[core_top:core_bottom, :]

            x_proj = core.sum(axis=0)
            x_indices = np.where(x_proj > 0)[0]

            if len(x_indices) == 0:
                continue  # empty line

            x1_local, x2_local = x_indices[0], x_indices[-1]

            abs_y1 = int(bbox[1] + y1)
            abs_y2 = int(bbox[1] + y2)
            abs_x1 = int(bbox[0] + x1_local)
            abs_x2 = int(bbox[0] + x2_local)

            abs_boxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
        return abs_boxes


def estimate_baseline(
        y_bottom: np.ndarray,
        W: int,
        H: int,
    ) -> tuple[int, float]:
    """
    Estimate the text baseline from a per-column bottom profile.

    The baseline is taken as the mode of y_bottom over non-empty columns. A second value,
    left_cross_ratio in [0, 1], measures how much (super-linearly weighted) below-baseline
    mass lies on the left half of the image (useful to detect where descenders are).

    Notes:
        Works best for a single text line. Multiple lines may create multiple peaks.

    Args:
        y_bottom: Array of shape (W,) giving the bottom-most ink pixel y for each column,
            or -1 for empty columns (crop-local coordinates).
        W: Width of the crop/mask in pixels.
        H: Height of the crop/mask in pixels.

    Returns:
        (baseline_y, left_cross_ratio)
        baseline_y is crop-local. left_cross_ratio≈0.5 means roughly balanced; >0.5 means
        more descender mass on the left side.
    """
    yb = y_bottom[y_bottom >= 0]
    if yb.size == 0:
        return 0, 0.5 

    counts = np.bincount(yb, minlength=H)
    baseline_y = int(np.argmax(counts))

    deltas = np.maximum(0, y_bottom - baseline_y).astype(np.float32)
    e = deltas ** 1.5

    sum_left = float(e[:W // 2].sum())
    sum_right = float(e[W // 2:].sum())

    if (sum_left + sum_right) == 0.0:
        return baseline_y, 0.5

    left_cross_ratio = sum_left / (sum_left + sum_right)
    return baseline_y, left_cross_ratio


def _solve_other_side_y(
        y_bottom: np.ndarray,
        baseline_y: int,
        anchor_x: int,
        other_x: int,
        margin_px: int = 1,
        edge_ignore: int = 0,
    ) -> int:
    """
    Solve the opposite bottom-corner y so a slanted bottom edge does not intersect the text.

    The bottom edge is the segment from (anchor_x, baseline_y) to (other_x, y_other). This
    function returns the smallest y_other such that, for all considered columns x, the line
    stays below the ink profile: y_line(x) >= y_bottom[x] + margin_px.

    Args:
        y_bottom: Bottom-most ink profile of shape (W,), crop-local coords (-1 for empty).
        baseline_y: Anchor y (crop-local), typically the estimated baseline.
        anchor_x: X position of the anchored corner (e.g., 0 or W-1).
        other_x: X position of the opposite corner (e.g., W-1 or 0).
        margin_px: Safety margin (in pixels) kept below the ink.
        edge_ignore: Number of columns ignored at both left and right edges to reduce noise.

    Returns:
        y_other (int): Crop-local y coordinate for the opposite bottom corner.
    """
    W = y_bottom.shape[0]
    m = 0.0

    for x in range(edge_ignore, W - edge_ignore):
        yb = int(y_bottom[x])
        if yb < 0:
            continue

        dist = abs(x - anchor_x)
        if dist == 0:
            continue

        need = (yb + margin_px - baseline_y) / dist
        if need > m:
            m = need

    y_other = baseline_y + m * abs(other_x - anchor_x)

    return int(math.ceil(y_other))


def adjust_box_to_baseline(
        pilimage: Image.Image,
        box: Box,
        t_high: float = 0.6,
        margin_px: int = 1,
        edge_ignore: int = 10,
        max_extra_ratio: float =0.25,
    )->Box:
    """
    Adjust a subtitle bounding box by fitting a slanted bottom edge aligned to the baseline.

    Workflow:
      - Crop the image by 'box', threshold the alpha channel to build a text mask.
      - Estimate the baseline and whether descenders are mostly on the left or right.
      - Anchor the bottom corner on the side with fewer descenders at the baseline.
      - Solve the opposite bottom corner so the bottom edge stays below the text mask.
      - If the required extra height is too large, return the original box.

    Notes:
        Designed for images containing a single subtitle line. Strong shadows/glow can bias
        the mask unless t_high is sufficiently high.

    Args:
        pilimage: Source RGBA image (or convertible to RGBA).
        box: Input crop box as three points [[xL, yT], [xR, y?], [x?, yB]] where the crop is
            (left=xL, top=yT, right=xR, bottom=yB) in source image coordinates.
        t_high: Relative alpha threshold in [0, 1] used to build a "solid ink" mask.
        margin_px: Vertical margin added under the ink when fitting the bottom edge.
        edge_ignore: Number of columns ignored near the crop borders when fitting the line.
        max_extra_ratio: Reject the slanted result if added height exceeds this fraction of H.

    Returns:
        list: A 4-point quadrilateral [[xL, yT], [xR, yT], [xR, yR], [xL, yL]] in source image
        coordinates. If rejected or ambiguous, returns the original 'box'.
    """
    f_box= box.full_box
    xL, yT = f_box[0]
    xR = f_box[1][0]
    yB = f_box[2][1]

    event_img = pilimage.crop((xL, yT, xR, yB))
    if event_img.mode != "RGBA":
        event_img = event_img.convert("RGBA")

    alpha = np.array(event_img.split()[-1], dtype=np.uint8)
    amax = int(alpha.max())
    if amax == 0:
        return box

    thr = t_high * (amax / 255.0)
    mask = (alpha.astype(np.float32) / 255.0) >= thr

    H, W = mask.shape

    y_bottom = np.full(W, -1, dtype=np.int32)
    for x in range(W):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            y_bottom[x] = int(ys[-1])

    baseline_y, left_cross_ratio = estimate_baseline(y_bottom, W, H)

    if 0.4 < left_cross_ratio < 0.6:
        return box

    if left_cross_ratio > 0.5:
        yR_local = baseline_y
        yL_local = _solve_other_side_y(
            y_bottom=y_bottom,
            baseline_y=baseline_y+3,
            anchor_x=W - 1,
            other_x=0,
            margin_px=margin_px,
            edge_ignore=edge_ignore,
        )
    else:
        yL_local = baseline_y
        yR_local = _solve_other_side_y(
            y_bottom=y_bottom,
            baseline_y=baseline_y+3,
            anchor_x=0,
            other_x=W - 1,
            margin_px=margin_px,
            edge_ignore=edge_ignore,
        )

    yL = yT + yL_local
    yR = yT + yR_local
    extra_left = yL_local - baseline_y
    extra_right = yR_local - baseline_y
    extra = max(extra_left, extra_right)
    if extra > max_extra_ratio * H:
        return box

    _, h = pilimage.size
    yR= min(yR, h)
    yL=min(yL, h)
    return Box([xL, yT], [xR, yT], [xR, yR], [xL, yL])
    

