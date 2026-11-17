"""Tests de lecture.py — la lecture à voix haute phrase par phrase.

De vrais threads, mais ni modèle ni son : la « synthèse » rend une chaîne, la
« lecture » la note. On vérifie l'ordre, l'avance bornée, l'arrêt, et ce qui
arrive quand une phrase échoue.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import pytest

from lecture import AVANCE_MAX, Lecture, LectureCallbacks


class Banc:
    """Tout ce dont une `Lecture` a besoin, en version observable."""

    def __init__(self, phrases: list[str], inline: bool = True) -> None:
        self.joues: list[str] = []
        self.progres: list[tuple[int, int]] = []
        self.synthetises: list[str] = []
        self.fin = threading.Event()
        self.erreur: str | None = "pas fini"
        self.echecs: set[str] = set()
        self.arrets = 0
        # Posé par un test pour retenir la lecture au milieu d'une phrase.
        self.blocage: threading.Event | None = None
        self.en_lecture = threading.Event()
        self._inline = inline
        self.taches: list[Callable[[], None]] = []
        self.lecture = Lecture(
            phrases=phrases,
            synthesize=self._synthesize,
            submit=self._submit,
            start_playback=self._start,
            wait_playback=self._wait,
            stop_playback=self._stop,
            callbacks=LectureCallbacks(
                on_progress=lambda i, n: self.progres.append((i, n)),
                on_done=self._done,
            ),
        )

    def _synthesize(self, phrase: str) -> str:
        self.synthetises.append(phrase)
        if phrase in self.echecs:
            raise RuntimeError(f"illisible : {phrase}")
        return f"son<{phrase}>"

    def _submit(self, task: Callable[[], None]) -> None:
        if self._inline:
            task()
        else:
            self.taches.append(task)

    def _start(self, audio: Any) -> None:
        self.joues.append(audio)

    def _wait(self) -> None:
        self.en_lecture.set()
        if self.blocage is not None:
            self.blocage.wait(5)

    def _stop(self) -> None:
        self.arrets += 1
        if self.blocage is not None:
            self.blocage.set()

    def _done(self, erreur: str | None) -> None:
        self.erreur = erreur
        self.fin.set()

    def attendre(self) -> None:
        assert self.fin.wait(5), "la lecture ne s'est jamais terminée"


def test_sentences_are_played_in_order_then_done_without_error():
    banc = Banc(["Un.", "Deux.", "Trois.", "Quatre.", "Cinq."])
    banc.lecture.start()
    banc.attendre()
    assert banc.joues == [f"son<{p}>" for p in ["Un.", "Deux.", "Trois.", "Quatre.", "Cinq."]]
    assert banc.progres == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert banc.erreur is None


def test_synthesis_stays_only_a_few_sentences_ahead():
    """Tant que rien n'est joué, on ne fabrique pas tout le texte : le worker
    reste libre, et un arrêt précoce ne gâche pas vingt minutes de calcul."""
    phrases = [f"Phrase {i}." for i in range(10)]
    banc = Banc(phrases)
    banc.blocage = threading.Event()  # la 1re phrase reste « en train de jouer »
    banc.lecture.start()
    assert banc.en_lecture.wait(5)
    time.sleep(0.05)
    # 1 phrase en cours de lecture + AVANCE_MAX prêtes derrière, pas plus.
    assert len(banc.synthetises) == 1 + AVANCE_MAX
    banc.blocage.set()
    banc.blocage = None
    banc.attendre()
    assert len(banc.joues) == 10


def test_each_sentence_is_a_separate_task_submitted_one_at_a_time():
    """C'est ce qui laisse une dictée passer entre deux phrases."""
    banc = Banc(["Un.", "Deux.", "Trois."], inline=False)
    banc.lecture.start()
    assert len(banc.taches) == 1
    banc.taches.pop(0)()  # le worker exécute la 1re : la 2e est alors soumise
    assert len(banc.taches) == 1
    while banc.taches:
        banc.taches.pop(0)()
        time.sleep(0.02)
    banc.attendre()
    assert banc.synthetises == ["Un.", "Deux.", "Trois."]


def test_cancel_stops_playback_and_further_synthesis():
    phrases = [f"Phrase {i}." for i in range(10)]
    banc = Banc(phrases)
    banc.blocage = threading.Event()
    banc.lecture.start()
    assert banc.en_lecture.wait(5)

    banc.lecture.cancel()
    banc.attendre()

    assert banc.lecture.cancelled
    assert banc.arrets == 1
    assert banc.joues == ["son<Phrase 0.>"]
    assert len(banc.synthetises) <= 1 + AVANCE_MAX
    assert banc.erreur is None  # un arrêt voulu n'est pas une erreur


def test_cancel_twice_is_harmless():
    banc = Banc(["Un."])
    banc.blocage = threading.Event()
    banc.lecture.start()
    assert banc.en_lecture.wait(5)
    banc.lecture.cancel()
    banc.lecture.cancel()
    banc.attendre()
    assert banc.arrets == 1


def test_an_unreadable_sentence_is_skipped_and_reported():
    banc = Banc(["Un.", "Deux.", "Trois."])
    banc.echecs = {"Deux."}
    banc.lecture.start()
    banc.attendre()
    assert banc.joues == ["son<Un.>", "son<Trois.>"]
    assert banc.erreur == "1 phrase sur 3 n'a pas pu être lue et a été sautée."


def test_first_sentence_failure_aborts_with_the_real_message():
    """Rien n'a été dit : c'est le modèle ou la voix qui manque. On ne
    s'acharne pas sur la suite, et on montre la vraie cause."""
    banc = Banc(["Un.", "Deux.", "Trois.", "Quatre.", "Cinq.", "Six."])
    banc.echecs = {"Un."}
    banc.lecture.start()
    banc.attendre()
    assert banc.joues == []
    assert banc.erreur == "illisible : Un."
    assert len(banc.synthetises) <= 1 + AVANCE_MAX


def test_a_refused_submission_ends_the_reading_cleanly():
    """L'inference-worker refuse les tâches quand l'app quitte."""

    def refuse(_task):  # noqa: ANN001, ANN202
        raise RuntimeError("worker arrêté")

    fin = threading.Event()
    erreurs: list[str | None] = []
    lecture = Lecture(
        phrases=["Un.", "Deux."],
        synthesize=lambda p: p,
        submit=refuse,
        start_playback=lambda a: None,
        wait_playback=lambda: None,
        stop_playback=lambda: None,
        callbacks=LectureCallbacks(
            on_progress=lambda i, n: None,
            on_done=lambda e: (erreurs.append(e), fin.set()),
        ),
    )
    lecture.start()
    assert fin.wait(5)
    assert erreurs == ["worker arrêté"]


@pytest.mark.parametrize("phrases", [[], ["Seule."]])
def test_degenerate_texts_still_finish(phrases: list[str]):
    banc = Banc(phrases)
    banc.lecture.start()
    banc.attendre()
    assert banc.joues == [f"son<{p}>" for p in phrases]
    assert banc.erreur is None


def test_no_sentence_starts_once_cancel_has_returned():
    """Arrêt tombé pile entre deux phrases : la suivante ne doit PAS partir.
    C'est le cas d'une dictée qui démarre — le micro est déjà ouvert."""
    banc = Banc([f"Phrase {i}." for i in range(6)])
    fin_de_phrase = threading.Event()
    reprise = threading.Event()

    def wait() -> None:
        # La 1re phrase « finit de jouer », puis la lecture est retenue ici,
        # juste avant d'enchaîner sur la 2e.
        fin_de_phrase.set()
        reprise.wait(5)

    banc.lecture._wait_playback = wait
    banc.lecture.start()
    assert fin_de_phrase.wait(5)

    banc.lecture.cancel()
    reprise.set()
    banc.attendre()

    assert banc.joues == ["son<Phrase 0.>"]
