from pathlib import Path
from _io import TextIOWrapper
import re


def srt_to_ass_lines(srt_dict_list:list[dict])->list[str]:
    """Convert a list of SRT subtitle entries into ASS-formatted lines.

    This function takes subtitle data (start time, end time, and text) parsed from an SRT file
    and generates corresponding lines that can be written to an ASS (Advanced SubStation Alpha) file.
    Basic text formatting (bold, italic, underline) and line breaks are converted to ASS syntax.

    Args:
        srt_dict_list (list[dict]): A list of subtitle entries, each containing
            'start' (float, in seconds), 'end' (float, in seconds), and 'text' (str).

    Returns:
        list[str]: A list of lines forming the body of an ASS subtitle file.
    """
    def seconds_to_str(seconds:float)->str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.3f}"
    
    def srt_text_to_ass_text(text:str)->str:
        """try to adapt the text to a ass format
        """
        changes = [
            (r'<b>',r'{\b1}'), (r'<\b>', r'{\b0}'), (r'<i>',r'{\i1}'), (r'<\b>',r'{\b0}'),
            (r'<u>',r'{\u1}'), (r'<\u>',r'{\u0}'), ('\n', r'\N')
        ]
        for before, after in changes:
            text.replace(before, after)
        return text

    lines = [
        '[V4+ Styles]', 
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
        'Style: Default,Arial,20,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,1,2,10,10,10,1', # this is the default style for a ass file
        '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text'
    ]

    for event in srt_dict_list:
        if any([key not in event for key in ['start', 'end', 'text']]):
            raise ValueError(f'Every dict in the list should have a "start", "end" and "text" key')
        
        start= seconds_to_str(seconds=event['start'])
        end = seconds_to_str(seconds=event['end'])
        lines.append(
            f"Dialogue: 100,{start},{end},Default,,0,0,0,,{srt_text_to_ass_text(event['text'])}"
        )
        secs_str, _, csecs = str(start).partition(".")
        hours, mins, secs = map(int, secs_str.split(":"))

        r = hours * 60 * 60 + mins * 60 + secs + int(csecs) * 1e-2
    return lines

def parse_str_file(file: str | Path | TextIOWrapper)->list[dict]:
    """Parse an SRT file and extract subtitle entries as structured data.

    This function reads a SubRip (.srt) subtitle file and extracts each subtitle’s
    start time, end time, and text content into a list of dictionaries.

    Args:
        file (str | Path | TextIOWrapper): Path or file object of the SRT file.

    Raises:
        ValueError: If the input is not a valid file path or text stream.

    Returns:
        list[dict]: A list of dictionaries with keys:
            - 'start' (float): start time in seconds
            - 'end' (float): end time in seconds
            - 'text' (str): subtitle text
    """
    if isinstance(file, str) or isinstance(file, Path):
        with open(file, encoding="utf-8") as f:
            srt_content = f.read()
    elif isinstance(file, TextIOWrapper):
        srt_content = file.read()
    else:
        raise ValueError(f"file should be a str path, a Path or a TextIOWrapper (here {type(file)})")

    pattern = re.compile( # a srt event should always look like this
        r'(?m)^\d+\r?\n' # the id of the event
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\r?\n' # the timing of the event
        r'((?:.+(?:\r?\n)?)+?)' # the text of the event
        r'(?=\r?\n\d+\r?\n|$)'
    )
    matches = pattern.findall(srt_content)

    st = []
    for start, end, text in matches:
        patern_time = re.compile(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
        )
        start_match= patern_time.findall(start)
        end_match = patern_time.findall(end)

        sub_dict = {
            'start':int(start_match[0][0])*3600 +int(start_match[0][1])*60 + int(start_match[0][2]) + int(start_match[0][3])/1000,
            'end':int(end_match[0][0])*3600 +int(end_match[0][1])*60 + int(end_match[0][2]) + int(end_match[0][3])/1000,
            'text':text.strip()
        }
        st.append(sub_dict)
    return st