"""
Une lecture à voix haute, phrase par phrase.

Avant, « Lire le texte copié » fabriquait tout le son puis le jouait : on
attendait la synthèse entière avant d'entendre la première syllabe, d'où un
plafond de 2 000 caractères, et une seule phrase ratée faisait tout échouer.

Ici le texte est déjà découpé en phrases (`synthese.decouper_en_phrases`) et
deux activités avancent en parallèle :

- la **synthèse**, une phrase à la fois, confiée à qui possède le modèle
  (dans l'app : l'inference-worker, en priorité basse). Chaque phrase est une
  tâche à part : une dictée passe donc devant entre deux phrases, au lieu
  d'attendre la fin d'un long texte ;
- la **lecture**, sur un thread à nous (`speech-play`), qui joue les phrases
  dans l'ordre à mesure qu'elles arrivent.

La synthèse ne prend que quelques phrases d'avance (`AVANCE_MAX`) : inutile
de fabriquer vingt minutes de son si l'utilisateur arrête au bout de dix
secondes, et le worker reste disponible pour autre chose.

Une phrase que le modèle n'arrive pas à dire est sautée et la lecture
continue — sauf si c'est la première : là, c'est très probablement le modèle
ou la voix qui manque, et s'acharner sur les phrases suivantes ne ferait que
répéter l'échec.

Ce module ne connaît ni rumps, ni MLX, ni sounddevice : tout lui est passé en
fonctions, ce qui le rend testable tel quel.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Sequence


# Nombre de phrases prêtes d'avance, au plus. À 3, la synthèse a de la marge
# sur la lecture sans jamais accaparer le worker.
AVANCE_MAX = 3


@dataclass
class LectureCallbacks:
    """Ce que la lecture signale. Appelé depuis ses threads, jamais depuis
    le main thread : à l'appelant de repasser sur le sien s'il le faut."""

    # (numéro de la phrase qui commence à jouer, à partir de 1 ; total)
    on_progress: Callable[[int, int], None]
    # Fin, quelle qu'en soit la cause. `erreur` est None si tout s'est bien
    # passé ou si l'utilisateur a arrêté ; sinon un message à lui montrer.
    on_done: Callable[[str | None], None]


class Lecture:
    """Une lecture en cours. Jetable : une instance par texte lu."""

    def __init__(
        self,
        phrases: Sequence[str],
        synthesize: Callable[[str], Any],
        submit: Callable[[Callable[[], None]], Any],
        start_playback: Callable[[Any], None],
        wait_playback: Callable[[], None],
        stop_playback: Callable[[], None],
        callbacks: LectureCallbacks,
    ) -> None:
        self._phrases = list(phrases)
        # Texte → son. Appelé là où `submit` exécute les tâches.
        self._synthesize = synthesize
        # Confie une tâche au propriétaire du modèle, sans attendre.
        self._submit = submit
        # Lance un son et rend la main tout de suite.
        self._start_playback = start_playback
        # Attend la fin du son lancé (ou son arrêt).
        self._wait_playback = wait_playback
        # Coupe le son en cours ; doit débloquer `wait_playback`.
        self._stop_playback = stop_playback
        self._cb = callbacks

        self._lock = threading.Lock()
        # Départ d'un son et arrêt s'excluent : sans ce verrou, un arrêt
        # tombé pile entre deux phrases couperait « rien », puis la phrase
        # suivante partirait quand même — en plein dans le micro de la
        # dictée qui a demandé le silence.
        self._play_lock = threading.Lock()
        self._cancelled = threading.Event()
        # (index, son ou None si la phrase a été sautée, message d'erreur)
        self._ready: "queue.Queue[tuple[int, Any, str | None] | None]" = queue.Queue()
        self._next = 0  # prochaine phrase à synthétiser
        self._played = 0  # phrases déjà retirées de la file par la lecture
        self._in_flight = False  # une synthèse est en file ou en cours
        self._thread: threading.Thread | None = None

    # ---- Contrôle ----

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def start(self) -> None:
        """Lance la lecture. Rend la main tout de suite."""
        self._thread = threading.Thread(
            target=self._run, name="speech-play", daemon=True
        )
        self._thread.start()
        self._feed()

    def cancel(self) -> None:
        """Arrête tout : le son en cours, et la synthèse des phrases à venir.

        Sans danger depuis n'importe quel thread, et plusieurs fois. La fin
        est annoncée par `on_done`, comme pour une lecture qui va au bout.
        """
        if self._cancelled.is_set():
            return
        self._cancelled.set()
        with self._play_lock:
            try:
                self._stop_playback()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        # Débloque le thread de lecture s'il attend une phrase.
        self._ready.put(None)

    # ---- Synthèse (chez le propriétaire du modèle) ----

    def _feed(self) -> None:
        """Met la phrase suivante en synthèse, s'il y a lieu.

        Une seule tâche à la fois, et pas plus d'`AVANCE_MAX` phrases
        d'avance sur la lecture. Appelé après chaque synthèse finie et après
        chaque phrase jouée : c'est ce qui fait avancer la chaîne.
        """
        with self._lock:
            if (
                self._cancelled.is_set()
                or self._in_flight
                or self._next >= len(self._phrases)
                or self._next - self._played >= AVANCE_MAX
            ):
                return
            index = self._next
            self._in_flight = True
        try:
            self._submit(lambda: self._synthesize_one(index))
        except Exception as exc:  # noqa: BLE001
            # Le worker refuse la tâche (app en train de quitter) : on rend
            # la phrase « ratée » pour que la lecture se termine proprement.
            traceback.print_exc()
            self._deliver(index, None, str(exc))

    def _synthesize_one(self, index: int) -> None:
        audio: Any = None
        erreur: str | None = None
        if not self._cancelled.is_set():
            try:
                audio = self._synthesize(self._phrases[index])
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                erreur = str(exc) or exc.__class__.__name__
        self._deliver(index, audio, erreur)

    def _deliver(self, index: int, audio: Any, erreur: str | None) -> None:
        with self._lock:
            self._in_flight = False
            self._next = index + 1
        self._ready.put((index, audio, erreur))
        self._feed()

    # ---- Lecture (thread speech-play) ----

    def _run(self) -> None:
        total = len(self._phrases)
        sautees = 0
        erreur_finale: str | None = None
        try:
            for attendu in range(total):
                item = self._ready.get()
                if item is None or self._cancelled.is_set():
                    break
                index, audio, erreur = item
                with self._lock:
                    self._played = index + 1
                if erreur is not None or audio is None:
                    if attendu == 0:
                        # Rien n'a encore été dit : le modèle ou la voix
                        # manque. On s'arrête là, avec le vrai message.
                        erreur_finale = erreur or "Le modèle n'a produit aucun son."
                        self._cancelled.set()
                        break
                    sautees += 1
                    self._feed()
                    continue
                self._feed()
                self._cb.on_progress(index + 1, total)
                with self._play_lock:
                    if self._cancelled.is_set():
                        break
                    self._start_playback(audio)
                self._wait_playback()
                if self._cancelled.is_set():
                    break
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            erreur_finale = str(exc) or exc.__class__.__name__
            self._cancelled.set()

        if erreur_finale is None and sautees and not self._cancelled.is_set():
            erreur_finale = (
                f"{sautees} phrases sur {total} n'ont pas pu être lues "
                f"et ont été sautées."
                if sautees > 1
                else f"1 phrase sur {total} n'a pas pu être lue et a été sautée."
            )
        self._cb.on_done(erreur_finale)
