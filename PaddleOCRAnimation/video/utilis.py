def detect_text_line_boxes(sub_image, multiline: bool = True, threshold_percent:float = 0.01):
        #the image should be the transpatent image of one event (one sub) with one or more line

        import numpy as np
        # we need to find the box  of the text, because the image is transparent it is relativly easy
        alpha = sub_image.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            return []  # there is not text on the image
        
        if multiline:
            # beacause multiline is allowed, no further modification need to be done
            return [bbox]

        cropped = alpha.crop(bbox)
        arr = np.array(cropped)
        binary = (arr > 0).astype(np.uint8)
        projection = binary.sum(axis=1)
        threshold = np.max(projection) * threshold_percent
        line_boxes = []
        in_line = False
        start = 0

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