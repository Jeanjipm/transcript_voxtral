"""Tests d'inference_worker.py — priorité, exclusion mutuelle, robustesse."""

from __future__ import annotations

import threading
import time

import pytest

from inference_worker import (
    PRIORITY_DICTATION,
    PRIORITY_FILE,
    PRIORITY_MODEL,
    InferenceWorker,
)


@pytest.fixture
def worker():
    w = InferenceWorker(name="test-inference")
    w.start()
    yield w
    w.shutdown(wait=True)


# ---- Cycle de vie ----


def test_start_is_idempotent(worker: InferenceWorker):
    """Un 2e start() ne doit pas créer un 2e thread — sinon deux threads
    toucheraient le modèle, ce que toute cette classe existe pour empêcher."""
    thread = worker._thread
    worker.start()
    assert worker._thread is thread


def test_submit_runs_the_function(worker: InferenceWorker):
    handle = worker.submit(lambda: 6 * 7)
    assert handle.result(timeout=2.0) == 42


def test_result_reraises_task_exception(worker: InferenceWorker):
    def boom() -> None:
        raise ValueError("modèle introuvable")

    handle = worker.submit(boom)
    with pytest.raises(ValueError, match="modèle introuvable"):
        handle.result(timeout=2.0)


def test_worker_survives_a_raising_task(worker: InferenceWorker):
    """Le worker est unique : une tâche qui lève ne doit pas l'abattre, sinon
    plus aucune transcription n'est possible jusqu'au redémarrage."""
    worker.submit(lambda: 1 / 0).wait(timeout=2.0)

    handle = worker.submit(lambda: "toujours vivant")
    assert handle.result(timeout=2.0) == "toujours vivant"
    assert worker.is_running is True


def test_result_raises_timeout_when_not_done(worker: InferenceWorker):
    gate = threading.Event()
    handle = worker.submit(gate.wait)
    with pytest.raises(TimeoutError):
        handle.result(timeout=0.05)
    gate.set()


def test_submit_after_shutdown_returns_failed_handle():
    """Une soumission tardive ne doit pas rester en attente pour toujours."""
    w = InferenceWorker(name="test-late")
    w.start()
    w.shutdown(wait=True)

    handle = w.submit(lambda: "trop tard")
    assert handle.done is True
    with pytest.raises(RuntimeError, match="arrêté"):
        handle.result(timeout=1.0)


def test_shutdown_releases_pending_handles():
    """Les tâches jamais exécutées doivent voir leur poignée libérée, sinon
    un appelant bloqué dans result() attend indéfiniment."""
    w = InferenceWorker(name="test-drain")
    w.start()

    gate = threading.Event()
    blocking = w.submit(gate.wait, priority=PRIORITY_DICTATION)
    # Mise en file derrière la tâche bloquante : ne sera jamais exécutée.
    abandoned = w.submit(lambda: "jamais", priority=PRIORITY_FILE)

    w.shutdown(wait=False)
    gate.set()
    blocking.wait(timeout=2.0)

    assert abandoned.wait(timeout=2.0) is True
    with pytest.raises(RuntimeError, match="arrêté"):
        abandoned.result(timeout=1.0)


def test_shutdown_is_idempotent():
    w = InferenceWorker(name="test-double-shutdown")
    w.start()
    w.shutdown(wait=True)
    w.shutdown(wait=True)  # ne doit pas lever


# ---- Priorité ----


def test_dictation_jumps_ahead_of_queued_file_blocks():
    """LE comportement clé : une dictée passe devant des blocs de fichier
    déjà en file, pour que dicter pendant un job d'une heure reste fluide."""
    w = InferenceWorker(name="test-priority")
    order: list[str] = []
    gate = threading.Event()

    w.start()
    # Occupe le worker pour que tout s'empile derrière.
    w.submit(gate.wait, priority=PRIORITY_MODEL, label="bloqueur")

    for i in range(3):
        w.submit(lambda i=i: order.append(f"fichier{i}"), priority=PRIORITY_FILE)
    w.submit(lambda: order.append("dictee"), priority=PRIORITY_DICTATION)

    gate.set()
    # Attend que tout soit drainé.
    w.submit(lambda: order.append("fin"), priority=PRIORITY_FILE).result(timeout=3.0)
    w.shutdown(wait=True)

    assert order[0] == "dictee", f"la dictée doit passer en tête, ordre={order}"
    assert order[-1] == "fin"


def test_equal_priorities_keep_fifo_order():
    """À priorité égale, l'ordre de soumission est préservé."""
    w = InferenceWorker(name="test-fifo")
    order: list[int] = []
    gate = threading.Event()

    w.start()
    w.submit(gate.wait, priority=PRIORITY_MODEL)
    for i in range(5):
        w.submit(lambda i=i: order.append(i), priority=PRIORITY_FILE)

    gate.set()
    w.submit(lambda: None, priority=PRIORITY_FILE).result(timeout=3.0)
    w.shutdown(wait=True)

    assert order == [0, 1, 2, 3, 4]


def test_equal_priorities_do_not_raise_typeerror():
    """PriorityQueue compare les éléments : sans le compteur seq, deux tâches
    de priorité égale feraient comparer les objets tâche et lèveraient
    TypeError. Ce test verrouille le départage."""
    w = InferenceWorker(name="test-tiebreak")
    w.start()
    handles = [
        w.submit(lambda i=i: i, priority=PRIORITY_FILE) for i in range(20)
    ]
    for i, h in enumerate(handles):
        assert h.result(timeout=3.0) == i
    w.shutdown(wait=True)


# ---- Exclusion mutuelle ----


def test_only_one_thread_ever_runs_a_task():
    """Détecteur de réentrance : prouve qu'aucune tâche ne tourne en parallèle
    d'une autre. C'est la garantie qui remplace un verrou autour du modèle."""
    w = InferenceWorker(name="test-mutex")
    concurrent = 0
    max_concurrent = 0
    guard = threading.Lock()

    def task() -> None:
        nonlocal concurrent, max_concurrent
        with guard:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.01)
        with guard:
            concurrent -= 1

    w.start()
    handles = [w.submit(task, priority=PRIORITY_FILE) for _ in range(12)]
    for h in handles:
        h.result(timeout=5.0)
    w.shutdown(wait=True)

    assert max_concurrent == 1


def test_all_tasks_run_on_the_same_thread():
    """Corollaire : le modèle MLX n'est jamais touché depuis deux threads."""
    w = InferenceWorker(name="test-one-thread")
    thread_ids: set[int] = set()

    w.start()
    handles = [
        w.submit(lambda: thread_ids.add(threading.get_ident())) for _ in range(8)
    ]
    for h in handles:
        h.result(timeout=3.0)
    w.shutdown(wait=True)

    assert len(thread_ids) == 1
