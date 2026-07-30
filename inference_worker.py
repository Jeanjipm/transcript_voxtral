"""
Le thread unique propriétaire du modèle MLX.

Toute inférence — dictée, préchargement, bloc de fichier — passe par une
file de priorité servie par un seul thread. Deux bénéfices, gratuits :

1. **Exclusion mutuelle par construction.** Un seul thread touche le
   modèle, donc plus besoin d'un verrou autour de lui. Ça supprime au
   passage deux courses réelles de l'ancienne architecture : le swap de
   `self.transcriber` par le hot-reload de config pendant qu'un thread
   transcrivait, et le `_ensure_loaded()` en check-then-act qui pouvait
   faire charger 5 Go deux fois.

2. **Priorité.** Une dictée (priorité 0) passe devant un bloc de fichier
   déjà en file (priorité 2). L'utilisateur qui dicte pendant la
   transcription d'un enregistrement d'une heure attend au pire la fin du
   bloc en cours, pas la fin du job.

Pourquoi un compteur `seq` dans la clé de tri : `PriorityQueue` compare les
éléments entre eux, et deux tâches de priorité égale feraient comparer les
objets `_QueuedTask`, ce qui lève `TypeError`. Le compteur garantit un ordre
total et donne le FIFO à priorité égale.
"""

from __future__ import annotations

import itertools
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


# Priorités. Plus petit = servi plus tôt.
PRIORITY_DICTATION = 0
PRIORITY_MODEL = 1  # préchargement, changement de modèle, déchargement
PRIORITY_FILE = 2

# Attente entre deux tours de boucle quand la file est vide. Uniquement là
# pour que le thread remarque `_stopping` sans dépendre d'un réveil.
_POLL_TIMEOUT_S = 0.25


class TaskHandle:
    """Poignée rendue par `submit()` — permet d'attendre le résultat.

    Utilisable depuis n'importe quel thread SAUF le worker lui-même (une
    tâche qui attendrait sa propre poignée se bloquerait indéfiniment).
    """

    __slots__ = ("_done", "_result", "_error")

    def __init__(self) -> None:
        self._done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None

    def _complete(self, result: Any, error: BaseException | None) -> None:
        self._result = result
        self._error = error
        self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Attend la fin de la tâche. True si terminée, False si timeout."""
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> Any:
        """Attend puis retourne le résultat, ou relève l'exception de la tâche.

        Lève `TimeoutError` si la tâche n'est pas finie dans le délai.
        """
        if not self._done.wait(timeout):
            raise TimeoutError("tâche d'inférence non terminée dans le délai")
        if self._error is not None:
            raise self._error
        return self._result


@dataclass(order=False)
class _QueuedTask:
    """Une tâche en file. `fn` est exécutée sur le thread du worker."""

    fn: Callable[[], Any]
    handle: TaskHandle
    label: str = ""
    # Sentinelle d'arrêt : le worker sort de sa boucle en la voyant.
    is_shutdown: bool = field(default=False)


class InferenceWorker:
    """File de priorité + thread unique qui exécute toutes les inférences.

    Usage :
        worker = InferenceWorker()
        worker.start()
        handle = worker.submit(lambda: transcriber.transcribe(p))
        texte = handle.result(timeout=120)
        worker.shutdown()
    """

    def __init__(self, name: str = "inference-worker") -> None:
        self._name = name
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._counter = itertools.count()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._lock = threading.Lock()

    # ---- Cycle de vie ----

    def start(self) -> None:
        """Démarre le thread. Idempotent."""
        with self._lock:
            if self._thread is not None:
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run, daemon=True, name=self._name
            )
            self._thread.start()

    def shutdown(self, wait: bool = False, timeout: float = 2.0) -> None:
        """Demande l'arrêt du worker.

        `wait=False` par défaut : appelé depuis `quit_app`, on ne veut
        surtout pas bloquer le main thread sur une inférence MLX en cours.
        Les tâches encore en file sont abandonnées, et leurs poignées
        libérées avec une erreur pour ne pas laisser un appelant attendre
        pour toujours.
        """
        with self._lock:
            if self._thread is None:
                return
            thread = self._thread
            self._stopping = True

        # Priorité la plus haute : on ne veut pas attendre la fin d'un job
        # fichier de 40 minutes avant de voir la sentinelle.
        self._queue.put(
            (-1, next(self._counter), _QueuedTask(
                fn=lambda: None, handle=TaskHandle(), label="shutdown",
                is_shutdown=True,
            ))
        )

        if wait:
            thread.join(timeout=timeout)

        with self._lock:
            self._thread = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ---- Soumission ----

    def submit(
        self,
        fn: Callable[[], Any],
        priority: int = PRIORITY_DICTATION,
        label: str = "",
    ) -> TaskHandle:
        """Met `fn` en file et retourne une poignée. Ne bloque jamais.

        `fn` sera exécutée sur le thread du worker, donc peut librement
        toucher le modèle MLX. Elle ne doit PAS toucher à Cocoa / rumps
        (cf. le contrat de threads dans dictation_controller.py).
        """
        handle = TaskHandle()
        if self._stopping:
            handle._complete(None, RuntimeError("worker d'inférence arrêté"))
            return handle
        task = _QueuedTask(fn=fn, handle=handle, label=label)
        self._queue.put((priority, next(self._counter), task))
        return handle

    @property
    def pending(self) -> int:
        """Nombre de tâches en attente. Indicatif (peut bouger aussitôt lu)."""
        return self._queue.qsize()

    # ---- Boucle du worker ----

    def _run(self) -> None:
        while True:
            try:
                _priority, _seq, task = self._queue.get(timeout=_POLL_TIMEOUT_S)
            except queue.Empty:
                if self._stopping:
                    break
                continue

            if task.is_shutdown:
                break

            # Une tâche qui lève ne doit JAMAIS tuer le worker : ce thread
            # est unique, l'abattre rendrait toute transcription impossible
            # jusqu'au redémarrage. On enregistre l'erreur sur la poignée et
            # on continue.
            try:
                result = task.fn()
            except BaseException as exc:  # noqa: BLE001
                traceback.print_exc()
                task.handle._complete(None, exc)
            else:
                task.handle._complete(result, None)

        self._drain_abandoned()

    def _drain_abandoned(self) -> None:
        """Libère les poignées des tâches jamais exécutées.

        Sans ça, un appelant bloqué dans `handle.result()` attendrait
        indéfiniment après l'arrêt du worker.
        """
        while True:
            try:
                _priority, _seq, task = self._queue.get_nowait()
            except queue.Empty:
                return
            if not task.is_shutdown:
                task.handle._complete(
                    None, RuntimeError("worker d'inférence arrêté avant exécution")
                )
