from ...video.Video import Video
from ...video.sub.RendererClean import Renderer, Context
import random
from pathlib import Path
from os.path import join, relpath
from ...video.classes import dataset_image, eventWithPilList
from .disturb import disturb_eventWithPil
import logging 

logger = logging.getLogger(__name__)

def small_images_to_dataset(
        timestamp: float,
        video: Video, r: Renderer,
        dataset_path: str, image_save_path: str,
        ctx: Context, multiline: bool = True,
        sub_id: int = 0,
        p: float = 0.15
) -> list[dataset_image]:
    vid_name = Path(video.path).stem
    if random.random()>p:
        return []
    
    events_with_pil = video.get_subtitle_boxes(timestamp=timestamp, renderer=r, context=ctx, piste=sub_id, multiline = multiline, SIZE=(0,0))

    return_list = []
    for i, event in enumerate(events_with_pil):
        save_path = join(image_save_path, f'{vid_name}_s{sub_id}_t{timestamp}_e{i}.png')
        event = disturb_eventWithPil(event)
        event.image.save(save_path)
        return_list.append(dataset_image(
            image_path=relpath(save_path, dataset_path),
            event_list=event.events
        ))
    return return_list

def big_images_to_dataset(
        events_with_pil:eventWithPilList,
        dataset_path: str, image_save_path: str,
        vid_name: str, sub_id: int, timestamp: float,
        p:float = 0.1
    ) -> list[dataset_image]:
    if random.random() > 1 - (1 - p)**(len(events_with_pil) * 1.3):
        return []
    
    save_path = join(image_save_path, f'{vid_name}_s{sub_id}_t{timestamp}_full.png')

    return_list = []
    for event in events_with_pil:
        return_list += event.events
    
    events_with_pil.to_pil(show_boxes=False).save(save_path)

    return [dataset_image(image_path=relpath(save_path, dataset_path), event_list=return_list)]



    
