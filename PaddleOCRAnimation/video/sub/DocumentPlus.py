import ass
from datetime import timedelta
from typing import Union, overload, Literal
from _io import TextIOWrapper
from . import RendererClean
from .RendererClean import Event
from os.path import exists, abspath
from copy import deepcopy
from pathlib import Path
from .DocumentSrt import srt_to_ass_lines, parse_str_file

def split_dialogue(dialogue:ass.line.Dialogue) -> list[ass.line.Dialogue]:
    """split event beacause some OCR apps require singleline event.
    """
    from copy import deepcopy
    text = dialogue.text.replace(r"\N", r'\n')
    lines = text.split(r'\n')
    event_list = []
    for line in lines:
        event = deepcopy(dialogue)
        event.text = line
        event_list.append(event)
    return event_list


class DocumentPlus(ass.Document):
    """
    Extension de la classe ass.Document pour la manipulation avancée de sous-titres ASS.

    Cette classe ajoute des méthodes utilitaires pour :
      - Trier les événements (sous-titres) selon différents critères.
      - Compter le nombre d'événements actifs à un instant donné.
      - Extraire un document ne contenant qu'un événement précis actif à une frame donnée.
      - Parser un fichier ASS avec tri automatique des événements.
    """
    @property
    def styles(self) -> ass.section.StylesSection:
        """Permet d'avoir la rétrocompatibilitée avec les vieux fichiers ass"""
        # Priorité à V4+ Styles si présent et non vide, sinon V4 Styles
        styles_ass = self.sections.get(self.STYLE_ASS_HEADER)
        styles_ssa = self.sections.get(self.STYLE_SSA_HEADER)
        if styles_ass and getattr(styles_ass, "_lines", None):
            return styles_ass
        elif styles_ssa:
            return styles_ssa
        else:
            return styles_ass  # fallback

    def sort_events(self, key='start', reverse=False) -> 'DocumentPlus':
        """Trie les événements (dialogues) dans la section Events.

        Args:
            key (str, optional): Clé de tri ou fonction. Par défaut 'start'.
            reverse (bool, optional): Tri décroissant si True. Par défaut False.
        """
        if not hasattr(self.events, "_lines"):
            raise AttributeError("La section Events ne contient pas de lignes triables.")

        if callable(key):
            self.events._lines.sort(key=key, reverse=reverse)
        else:
            self.events._lines.sort(key=lambda e: getattr(e, key), reverse=reverse)

        return self
    @overload
    def nb_event_dans_frame(self, frame: timedelta, returnEvents: Literal[False] = False
                            ) -> int:
        ...
    @overload
    def nb_event_dans_frame(self, frame: timedelta, returnEvents: Literal[True]
                            ) -> tuple[int, ass.section.EventsSection]:
        ...
    def nb_event_dans_frame(self, frame: timedelta, returnEvents: bool = False
                            ) -> Union[int, tuple[int, ass.section.EventsSection]]:
        """
        Compte le nombre d'événements (events) actifs à un instant donné (frame).

        Args:
            frame (timedelta): L'instant (temps) pour lequel on souhaite
                compter les événements actifs.
            returnEvents (bool, optional):
                Si `True`, retourne également la liste des événements actifs à cet instant
                    sous forme d'une `EventsSection`.
                Si `False` (par défaut), retourne seulement le nombre d'événements.

        Returns:
            int: Nombre d'événements actifs à l'instant donné si `returnEvents=False`.
                Un tuple contenant le nombre d'événements actifs et la section
                Events correspondante si `returnEvents=True`.
        """
        nb = 0
        Events = ass.section.EventsSection('Events')
        temps_avant = timedelta(seconds=0)
        for event in self.events:
            if temps_avant > event.start:
                raise ValueError("Les évènements du document doivent êtres triés")
            temps_avant = event.start
            if event.start <= frame <= event.end:
                nb += 1
                Events.append(event)
            elif event.start > frame:
                break
        Events = sorted(Events, key=lambda e: (e.layer))
        if returnEvents:
            return nb, Events
        return nb
    
    def copy(self, timing:float | timedelta | None = None)-> 'DocumentPlus':
        """
        Create a copy of the current DocumentPlus instance.

        If `timing` is not provided, all events are copied.  
        If `timing` is provided, only events active at that specific time are copied.

        Args:
            timing (float | timedelta | None, optional): A time in seconds to filter events. Defaults to None.

        Returns:
            DocumentPlus: A new DocumentPlus instance with copied info, styles, and filtered events.
        """
        copy = DocumentPlus()
        copy.info.set_data(deepcopy(self.info))
        copy.styles.set_data(deepcopy(self.styles))

        if timing is None: 
            copy.events.set_data(deepcopy(self.events))
            return copy
        elif isinstance(timing, float) or isinstance(timing, int):
            timing= timedelta(seconds=timing)
        
        found_events= []
        for event in self.events:
            if event.start <= timing <= event.end:
                found_events.append(event)
            elif event.end > timing:
                # we assume that the document IS SORTED
                break
        copy.events.set_data(deepcopy(found_events))
        return copy


    def doc_event_precis(self, frame: timedelta, event_id: int = 0) -> 'DocumentPlus':
        """
        Extrait un document ne contenant qu'un événement précis actif à un instant donné.

        Args:
            frame (timedelta): L'instant (temps) pour lequel on souhaite extraire l'événement.
            event_id (int, optional):
                L'indice (0 pour le premier, 1 pour le second, etc.) de l'événement actif
                à cet instant.
                Par défaut 0 (le premier événement trouvé).

        Returns:
            DocumentPlus: Une copie du document ne contenant que l'événement sélectionné avec
                un seul style (celui de l'évènement).

        Raises:
            ValueError: Si aucun événement n'est actif à cet instant, ou si l'event_id
                demandé n'existe pas.
        """
        i = -1
        event_precis = None
        for event in self.events:
            if event.start < frame and frame < event.end:
                i += 1
                if i == event_id:
                    event_precis = event
                    break
            elif event.start > frame:
                break

        if i == -1 or not event_precis:
            raise ValueError("La frame n'a aucun event")
        elif i != event_id:
            raise ValueError(
                f"La frame ne contient pas de {event_id}ᵉ event (seulement {i + 1}"
                f"trouvés, donc IdMax est {i})."
                )

        docCopie = DocumentPlus()
        docCopie.info.set_data(self.info)

        if event_precis.style == "Default":
            default_styles = [s for s in self.styles if s.name == "Default"]
            if len(default_styles) == 1:
                style_de_levent = default_styles[0]
            elif len(default_styles) >= 2:
                style_de_levent = default_styles[1]
            else:
                raise ValueError("Aucun style 'Default' trouvé.")
        else:
            style_de_levent = next((s for s in self.styles if s.name == event_precis.style), None)
            if style_de_levent is None:
                raise ValueError(f"Style '{event_precis.style}' introuvable.")

        docCopie.styles.set_data([style_de_levent])
        docCopie.events.set_data([event_precis])

        return docCopie

    def doc_event_donne(self, event: Event) -> 'DocumentPlus':
        # TODO : trouver a quoi sert cette fonction
        docCopie = DocumentPlus()
        docCopie.info.set_data(self.info)

        if event.style == "Default":
            default_styles = [s for s in self.styles if s.name == "Default"]
            if len(default_styles) == 1:
                style_de_levent = default_styles[0]
            elif len(default_styles) >= 2:
                style_de_levent = default_styles[1]
            else:
                raise ValueError("Aucun style 'Default' trouvé.")
        else:
            style_de_levent = next((s for s in self.styles if s.name == event.style), None)
            if style_de_levent is None:
                raise ValueError(f"Style '{event.style}' introuvable.")

        docCopie.styles.set_data([style_de_levent])
        docCopie.events.set_data([event])

        return docCopie

    def event_to_pil(
        self, frame: int | timedelta, size: tuple[int, int],
        event_id: int = 0, fonts_dir: str | None = None
    )->RendererClean.ImageSequence:
        # TODO : changer nom et faire documentation
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError(
                "size doit être un tuple de taille 2 (ex: (1900, 1080))"
            )
        if fonts_dir is not None and not exists(fonts_dir):
            raise FileExistsError(
                f"Le dossier {abspath(fonts_dir)} n'existe pas"
            )
        if isinstance(frame, int):
            frame = timedelta(seconds=frame)
        elif not isinstance(frame, timedelta):
            raise ValueError(
                f"frame doit être un int ou un timedelta (idi {type(frame)})"
            )
        doc_precis = self.doc_event_precis(frame, event_id)
        
        ctx = RendererClean.Context()
        if fonts_dir is not None:
            ctx.fonts_dir = fonts_dir
        r = ctx.make_renderer()
        r.set_fonts(fontconfig_config="\0")
        r.set_all_sizes(size)
        t = ctx.make_track()
        t.populate(doc_precis)
        resultats_libass = r.render_frame(t, frame)
        return resultats_libass

    @classmethod
    def parse_file_plus(cls, f: TextIOWrapper | Path | str, sort: bool = True) -> 'DocumentPlus':
        """
        Parse an ASS or SRT subtitle file and return a DocumentPlus object.

        This method extends `parse_file` by supporting both ASS and SRT files.
        If an SRT file is provided, it is first converted to ASS format before parsing.
        Optionally, the parsed events can be automatically sorted by start time.

        Args:
            f (TextIOWrapper | Path | str): File object or path to an ASS or SRT file.
            sort (bool, optional): If True (default), events are sorted by start time.

        Returns:
            DocumentPlus: A parsed document containing subtitle events and styles.
        """
        if isinstance(f, TextIOWrapper):
            doc = cls.parse_file(f)
        elif (isinstance(f, Path) or isinstance(f, str)) and str(f).endswith('.srt'):
            # the file is a .srt file, not a .ass file
            # things need to be done to simulate the ass format
            srt_list = parse_str_file(f)
            srt_list = srt_to_ass_lines(srt_list)
            doc = cls.parse_file(srt_list)
        elif ((isinstance(f, Path) and f.exists()) or (isinstance(f, str) and Path(f).exists())) and (str(f).endswith('.ass')):
            with open(f,  encoding='utf_8_sig') as file:
                doc = cls.parse_file(file)
        else:
            raise ValueError(f'f should be a TextIOWrapper or a valid path to a .srt or .ass file')
        
        if sort:
            doc = doc.sort_events()
        return doc

