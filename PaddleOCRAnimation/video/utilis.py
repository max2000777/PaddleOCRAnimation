from PIL import Image
from .sub.box import Box

def detect_text_line_boxes(
        sub_image: Image.Image, 
        multiline: bool = True, 
        threshold_percent:float = 0.01,
        libass_box: list[Box] | None = None
    ) -> list[tuple[int, int, int, int]]:
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
                    int(last_cut+np.where(last_cut+box_projecton > 0)[0][-1]) # last id not null
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
            x_proj = line_region.sum(axis=0)
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