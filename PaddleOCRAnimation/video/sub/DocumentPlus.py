import ass
from datetime import timedelta
from typing import Union
from _io import TextIOWrapper
from OCRSub.B2_Segmentation import RendererClean
from OCRSub.B2_Segmentation.RendererClean import Event
from os.path import exists, abspath

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
    def styles(self):
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

        if i == -1:
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
        ctx.fonts_dir = b"fontMonogatari"
        r = ctx.make_renderer()
        r.set_fonts(fontconfig_config="\0")
        r.set_all_sizes(size)
        t = ctx.make_track()
        t.populate(doc_precis)
        resultats_libass = r.render_frame(t, frame)
        return resultats_libass

    @classmethod
    def parse_file_plus(cls, f: TextIOWrapper, sort: bool = True) -> 'DocumentPlus':
        """
        Parse un fichier ASS et retourne un objet DocumentPlus,
        avec possibilité de trier automatiquement les événements.

        Args:
            f (file-like object): Fichier ASS ouvert en lecture (texte).
            sort (bool, optional): Si `True` (par défaut), trie les
                événements par temps de début après le parsing.

        Returns:
            DocumentPlus: Un objet DocumentPlus contenant les données du fichier ASS,
                éventuellement triées.
        """
        doc = cls.parse_file(f)
        if sort:
            doc = doc.sort_events()
        return doc


if __name__ == "__main__":
    import ass

    with open("B2_Segmentation/TestsLibass/testST.ass", encoding='utf_8_sig') as f:
        doc = DocumentPlus.parse_file_plus(f)
    doc.doc_event_precis(frame=timedelta(minutes=24, seconds=38), event_id=4)
