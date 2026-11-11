"""Tests de synthese.py — le moteur texte → son, sans MLX ni ffmpeg réels.

On vérifie ce que le module DÉCIDE : la commande ffmpeg exacte (mono, 64 kb/s,
ralenti, volume normalisé), l'écriture atomique du MP3, le refus d'une voix
inconnue, la liste des voix lue dans le cache. La qualité du son, elle, se
juge à l'oreille (--essai de vocaliser_cahier.py).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import synthese
from synthese import (
    LIBELLES_VOIX,
    MAX_CHARS,
    SynthesisError,
    Synthetiseur,
    libelle_voix,
    tronquer,
    vers_mp3,
    voix_disponibles,
    voix_francaises,
)


# ---- Voix ----


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Un snapshot HF factice avec quatre empreintes de voix."""
    dossier = tmp_path / "snap" / synthese.VOICE_DIR
    dossier.mkdir(parents=True)
    for nom in ("fr_male", "casual_female", "fr_female", "de_male"):
        (dossier / f"{nom}.safetensors").write_bytes(b"x")
    (dossier / "notes.txt").write_text("pas une voix")
    monkeypatch.setattr(
        synthese.hf_offline, "resolve_local_path", lambda repo: str(tmp_path / "snap")
    )
    return tmp_path / "snap"


def test_voix_disponibles_reads_voice_files_sorted(snapshot: Path):
    assert voix_disponibles("org/tts") == [
        "casual_female", "de_male", "fr_female", "fr_male",
    ]


def test_voix_francaises_keeps_fr_prefix_only(snapshot: Path):
    assert voix_francaises("org/tts") == ["fr_female", "fr_male"]


def test_voix_disponibles_falls_back_when_model_absent(monkeypatch: pytest.MonkeyPatch):
    """Modèle pas encore téléchargé : `resolve_local_path` rend le repo tel
    quel, et mlx_audio n'est pas importable ici → nos libellés servent de
    liste."""
    monkeypatch.setattr(synthese.hf_offline, "resolve_local_path", lambda repo: repo)
    assert voix_disponibles("org/tts") == [code for code, _ in LIBELLES_VOIX]


def test_libelles_put_french_voices_first():
    assert [code for code, _ in LIBELLES_VOIX[:2]] == ["fr_female", "fr_male"]
    assert libelle_voix("fr_male") == "Français — homme"
    assert libelle_voix("inconnue") == "inconnue"


# ---- lire ----


def _fake_model(
    chunks: list[np.ndarray],
    voices=("fr_female", "fr_male"),
    token_counts: list[int] | None = None,
):
    """`token_counts` : nombre de trames rendu à chaque appel de generate
    (pour simuler une lecture partie en boucle)."""
    appels: list[dict] = []

    def generate(text: str, voice: str, max_tokens: int = 4096):
        appels.append({"text": text, "voice": voice, "max_tokens": max_tokens})
        trames = token_counts[len(appels) - 1] if token_counts else len(chunks)
        for c in chunks:
            yield SimpleNamespace(audio=c, sample_rate=24_000, token_count=trames)

    return SimpleNamespace(
        generate=generate,
        appels=appels,
        sample_rate=24_000,
        _voice_embedding_files={v: Path(f"{v}.safetensors") for v in voices},
    )


def test_lire_concatenates_generator_chunks():
    synth = Synthetiseur("org/tts")
    synth._model = _fake_model([np.ones(3, dtype=np.float32), np.zeros(2, dtype=np.float32)])
    audio = synth.lire("Bonjour.", "fr_female")
    assert audio.dtype == np.float32
    assert audio.tolist() == [1, 1, 1, 0, 0]


def test_lire_refuses_unknown_voice_before_generating():
    """mlx-audio lit avec une voix par défaut quand le nom est inconnu : pour
    des MP3 destinés à des enfants, on préfère une erreur nette."""
    synth = Synthetiseur("org/tts")
    synth._model = _fake_model([np.ones(1, dtype=np.float32)])
    with pytest.raises(SynthesisError, match="fr_female, fr_male"):
        synth.lire("Bonjour.", "en_robot")


def test_lire_refuses_empty_text_without_loading(monkeypatch: pytest.MonkeyPatch):
    synth = Synthetiseur("org/tts")
    monkeypatch.setattr(synth, "_ensure_loaded", lambda: pytest.fail("ne doit pas charger"))
    with pytest.raises(SynthesisError):
        synth.lire("   ", "fr_female")


def test_lire_rejects_silent_generation():
    synth = Synthetiseur("org/tts")
    synth._model = _fake_model([np.zeros(0, dtype=np.float32)])
    with pytest.raises(SynthesisError, match="aucun son"):
        synth.lire("Bonjour.", "fr_male")


def test_budget_tokens_grows_with_text_and_is_capped():
    court = synthese.budget_tokens("Bonjour.")
    long = synthese.budget_tokens("x" * 300)
    assert court < long < synthese.MAX_TOKENS_PLAFOND
    assert synthese.budget_tokens("x" * 5000) == synthese.MAX_TOKENS_PLAFOND
    # Une phrase de 38 caractères mesurée à 6,8 s d'audio (85 trames) doit
    # tenir largement dans son budget.
    assert synthese.budget_tokens("75 plus 9 : par quelle dizaine je passe ?") > 85 * 1.5


def test_lire_passes_a_text_sized_max_tokens():
    """Sans plafond, une lecture partie en boucle déroulait 4 096 trames et
    le décodage audio tentait d'allouer 32 Go (processus tué)."""
    synth = Synthetiseur("org/tts")
    synth._model = _fake_model([np.ones(2, dtype=np.float32)])
    synth.lire("Bonjour.", "fr_female")
    assert synth._model.appels[0]["max_tokens"] == synthese.budget_tokens("Bonjour.")


def test_lire_retries_a_runaway_generation_then_succeeds(capsys):
    synth = Synthetiseur("org/tts")
    budget = synthese.budget_tokens("Bonjour.")
    synth._model = _fake_model(
        [np.ones(2, dtype=np.float32)], token_counts=[budget, budget, 10]
    )
    audio = synth.lire("Bonjour.", "fr_female")
    assert len(synth._model.appels) == 3
    assert audio.tolist() == [1, 1]
    assert "partie en boucle" in capsys.readouterr().out


def test_lire_retries_an_implausibly_long_reading(capsys):
    """« oiseau » lu en 6 s de charabia : la génération s'arrête d'elle-même
    (le plafond ne voit rien) mais la durée trahit le problème."""
    synth = Synthetiseur("org/tts")
    charabia = np.zeros(int(6.2 * 24_000), dtype=np.float32)
    normal = np.zeros(int(0.6 * 24_000), dtype=np.float32)
    appels = {"n": 0}

    def generate(text, voice, max_tokens=4096):
        appels["n"] += 1
        yield SimpleNamespace(audio=charabia if appels["n"] == 1 else normal, token_count=10)

    synth._model = SimpleNamespace(
        generate=generate, sample_rate=24_000,
        _voice_embedding_files={"fr_female": Path("x")},
    )
    audio = synth.lire("oiseau", "fr_female")
    assert appels["n"] == 2
    assert len(audio) == len(normal)
    assert "trop longue pour le texte" in capsys.readouterr().out


def test_duree_max_plausible_accepts_normal_readings():
    # Mesuré : 75 caractères lus en 4,9 à 6,7 s ; 38 caractères en 6,8 s au plus lent.
    assert synthese.duree_max_plausible("x" * 75) > 7
    assert synthese.duree_max_plausible("x" * 38) > 7
    # Un mot seul se lit en 0,4 à 0,7 s : 1,9 s de charabia (« heure »),
    # 3,8 s (« arbre ») et 6 s (« oiseau ») doivent être rejetés.
    assert 0.7 < synthese.duree_max_plausible("heure") < 1.9
    # Les chiffres se lisent comme des mots : « 54 plus. égale 60 6, 4, 5 »
    # a été lue en 6,8 s sans être du charabia.
    assert synthese.duree_max_plausible("54 plus. égale 60 6, 4, 5") > 6.8
    assert synthese.duree_max_plausible("arbre") < 3.8
    assert synthese.duree_max_plausible("oiseau") < 6


def test_lire_gives_up_after_three_runaway_generations():
    synth = Synthetiseur("org/tts")
    budget = synthese.budget_tokens("Bonjour.")
    synth._model = _fake_model([np.ones(2, dtype=np.float32)], token_counts=[budget] * 5)
    with pytest.raises(SynthesisError, match="n'arrive pas à finir"):
        synth.lire("Bonjour.", "fr_female")
    assert len(synth._model.appels) == synthese.TENTATIVES


# ---- Décodage du son par tranches (mémoire bornée) ----


ECH_PAR_TRAME = 4  # le vrai décodeur en rend 1 920 ; 4 suffisent ici


@pytest.fixture
def faux_mlx(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Un `mlx.core` minimal : numpy fait le travail, `eval` note ses appels."""
    import sys

    evalues: list = []
    core = SimpleNamespace(
        eval=lambda *tableaux: evalues.extend(tableaux),
        concatenate=np.concatenate,
        evalues=evalues,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


def _faux_decodeur(contexte_utile: int):
    """Décodeur factice qui se comporte comme le vrai : chaque trame rend
    ECH_PAR_TRAME échantillons portant sa valeur, mais les `contexte_utile`
    premières trames d'un appel sortent fausses (−1), faute de passé — sauf si
    l'appel commence au tout début de la phrase (première valeur 0)."""
    appels: list[int] = []

    def decode(codes: np.ndarray) -> np.ndarray:
        trames = codes[0, :, 0].astype(np.float32)
        appels.append(len(trames))
        onde = np.repeat(trames, ECH_PAR_TRAME)
        if trames[0] != 0:
            onde[: contexte_utile * ECH_PAR_TRAME] = -1
        return onde[None, :]

    return decode, appels


def _codes(n: int) -> np.ndarray:
    """(1, n, 37) : la première colonne numérote les trames."""
    codes = np.zeros((1, n, 37), dtype=np.int64)
    codes[0, :, 0] = np.arange(n)
    return codes


def test_decoder_par_tranches_short_audio_is_one_untouched_call():
    """Jusqu'à une tranche, rien ne change : un seul appel, mêmes codes, et
    MLX n'est même pas importé (le test n'en fournit pas)."""
    decode, appels = _faux_decodeur(contexte_utile=16)
    codes = _codes(synthese.TRANCHE_TRAMES)
    onde = synthese.decoder_par_tranches(decode, codes)
    assert appels == [synthese.TRANCHE_TRAMES]
    assert onde.shape == (1, synthese.TRANCHE_TRAMES * ECH_PAR_TRAME)


def test_decoder_par_tranches_long_audio_matches_one_block(faux_mlx: SimpleNamespace):
    """Le plantage des générations en série : 4 096 trames décodées d'un bloc demandaient
    un tableau de 32 Gio. Par tranches, aucun appel ne dépasse une tranche plus
    son contexte, et le son recollé est celui du décodage d'un bloc."""
    decode, appels = _faux_decodeur(contexte_utile=synthese.CONTEXTE_TRAMES)
    codes = _codes(4096)
    onde = synthese.decoder_par_tranches(decode, codes)
    attendu = np.repeat(np.arange(4096, dtype=np.float32), ECH_PAR_TRAME)[None, :]
    assert np.array_equal(onde, attendu)
    assert max(appels) <= synthese.TRANCHE_TRAMES + synthese.CONTEXTE_TRAMES
    assert len(appels) == -(-4096 // synthese.TRANCHE_TRAMES)
    # Chaque tranche est calculée tout de suite, pas toutes à la fin.
    assert len(faux_mlx.evalues) == len(appels)


def test_decoder_par_tranches_needs_its_context(faux_mlx: SimpleNamespace):
    """Sans contexte, le début de chaque tranche est faux : c'est bien le
    contexte jeté qui rend les raccords propres."""
    decode, _ = _faux_decodeur(contexte_utile=8)
    onde = synthese.decoder_par_tranches(decode, _codes(100), tranche=40, contexte=0)
    assert (onde == -1).any()
    decode, _ = _faux_decodeur(contexte_utile=8)
    onde = synthese.decoder_par_tranches(decode, _codes(100), tranche=40, contexte=8)
    assert not (onde == -1).any()
    assert onde.shape == (1, 100 * ECH_PAR_TRAME)


def test_appliquer_decodage_par_tranches_wraps_the_model_decoder(faux_mlx: SimpleNamespace):
    decode, appels = _faux_decodeur(contexte_utile=synthese.CONTEXTE_TRAMES)
    model = SimpleNamespace(audio_tokenizer=SimpleNamespace(decode=decode))
    assert synthese.appliquer_decodage_par_tranches(model) is True
    onde = model.audio_tokenizer.decode(_codes(600))
    assert onde.shape == (1, 600 * ECH_PAR_TRAME)
    assert len(appels) == 3 and max(appels) <= synthese.TRANCHE_TRAMES + synthese.CONTEXTE_TRAMES


def test_appliquer_decodage_par_tranches_without_decoder_does_nothing():
    assert synthese.appliquer_decodage_par_tranches(SimpleNamespace()) is False


class _TableauMlx:
    """Se fait passer pour un mx.array : numpy sait le convertir."""

    def __init__(self, valeurs: list[float]) -> None:
        self._valeurs = np.asarray(valeurs, dtype=np.float32)

    def __array__(self, dtype=None, copy=None):
        return self._valeurs if dtype is None else self._valeurs.astype(dtype)


_TableauMlx.__module__ = "mlx.core"


def test_vers_numpy_evaluates_an_mlx_array_before_converting(faux_mlx: SimpleNamespace):
    """Le calcul est déclenché explicitement : une allocation refusée par
    Metal devient une exception Python au lieu de tuer le processus pendant
    la conversion numpy."""
    tableau = _TableauMlx([0.5, -0.5])
    assert synthese._vers_numpy(tableau).tolist() == [0.5, -0.5]
    assert faux_mlx.evalues == [tableau]


def test_lire_turns_a_metal_allocation_failure_into_a_clean_error():
    synth = Synthetiseur("org/tts")

    def generate(text, voice, max_tokens=4096):
        raise RuntimeError(
            "[metal::malloc] Attempting to allocate 34359738368 bytes which is "
            "greater than the maximum allowed buffer size."
        )
        yield  # pragma: no cover — fait de `generate` un générateur

    synth._model = SimpleNamespace(
        generate=generate, sample_rate=24_000,
        _voice_embedding_files={"fr_female": Path("x")},
    )
    with pytest.raises(SynthesisError, match="Mémoire insuffisante"):
        synth.lire("Bonjour.", "fr_female")


def test_lire_lets_other_runtime_errors_through():
    synth = Synthetiseur("org/tts")

    def generate(text, voice, max_tokens=4096):
        raise RuntimeError("No audio frames generated")
        yield  # pragma: no cover

    synth._model = SimpleNamespace(
        generate=generate, sample_rate=24_000,
        _voice_embedding_files={"fr_female": Path("x")},
    )
    with pytest.raises(RuntimeError, match="No audio frames"):
        synth.lire("Bonjour.", "fr_female")


def test_module_lire_reuses_shared_instance_per_repo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(synthese, "_partage", None)
    a = synthese.synthetiseur("org/a")
    assert synthese.synthetiseur("org/a") is a
    b = synthese.synthetiseur("org/b")
    assert b is not a and b.repo == "org/b"


# ---- tronquer ----


def test_tronquer_keeps_short_text():
    assert tronquer("Bonjour tout le monde.") == ("Bonjour tout le monde.", False)


def test_tronquer_cuts_long_text_on_a_word_boundary():
    texte = " ".join(["mot"] * 1000)
    coupe, tronque = tronquer(texte)
    assert tronque is True
    assert len(coupe) <= MAX_CHARS
    assert not coupe.endswith("mo")
    assert coupe.endswith("mot")


# ---- vers_mp3 ----


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> dict:
    """ffmpeg simulé : note la commande et l'entrée, écrit le fichier demandé."""
    seen: dict = {}

    def fake_run(cmd, input=None, capture_output=True, timeout=None, check=False):  # noqa: A002
        seen["cmd"] = cmd
        seen["input"] = input
        if seen.get("fail"):
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom: codec")
        Path(cmd[-1]).write_bytes(b"ID3mp3")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(synthese, "chemin_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(synthese.subprocess, "run", fake_run)
    return seen


def test_vers_mp3_builds_the_expected_ffmpeg_command(tmp_path: Path, fake_ffmpeg: dict):
    audio = np.linspace(-0.5, 0.5, 240, dtype=np.float32)
    sortie = tmp_path / "audio" / "abc.mp3"
    vers_mp3(audio, sortie, sample_rate=24_000)

    cmd = fake_ffmpeg["cmd"]
    assert cmd[0] == "/fake/ffmpeg"
    # Entrée : PCM float32 mono 24 kHz sur stdin, pas de WAV temporaire.
    assert cmd[cmd.index("-f") + 1] == "f32le"
    assert cmd[cmd.index("-ar") + 1] == "24000"
    assert cmd[cmd.index("-i") + 1] == "pipe:0"
    assert len(fake_ffmpeg["input"]) == 240 * 4
    # Sortie : mono, 64 kb/s, ralenti 0,92 puis volume normalisé.
    assert cmd.count("-ac") == 2 and cmd[cmd.index("-ac") + 1] == "1"
    positions_ar = [i for i, x in enumerate(cmd) if x == "-ar"]
    assert len(positions_ar) == 2 and cmd[positions_ar[1] + 1] == "24000"  # sortie
    assert cmd[cmd.index("-b:a") + 1] == "64k"
    assert cmd[cmd.index("-af") + 1] == "atempo=0.92,loudnorm=I=-16:TP=-1.5:LRA=11"
    assert cmd[-1].endswith(".mp3.tmp")


def test_vers_mp3_writes_atomically(tmp_path: Path, fake_ffmpeg: dict):
    sortie = tmp_path / "x.mp3"
    vers_mp3(np.zeros(10, dtype=np.float32), sortie)
    assert sortie.read_bytes() == b"ID3mp3"
    assert not sortie.with_name("x.mp3.tmp").exists()


def test_vers_mp3_failure_raises_and_leaves_no_file(tmp_path: Path, fake_ffmpeg: dict):
    fake_ffmpeg["fail"] = True
    sortie = tmp_path / "x.mp3"
    with pytest.raises(SynthesisError, match="codec"):
        vers_mp3(np.zeros(10, dtype=np.float32), sortie)
    assert not sortie.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_vers_mp3_without_ffmpeg_tells_how_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(synthese, "chemin_ffmpeg", lambda: None)
    with pytest.raises(SynthesisError, match="brew install ffmpeg"):
        vers_mp3(np.zeros(10, dtype=np.float32), tmp_path / "x.mp3")


def test_vers_mp3_timeout_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(synthese, "chemin_ffmpeg", lambda: "/fake/ffmpeg")

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(synthese.subprocess, "run", slow)
    with pytest.raises(SynthesisError, match="ffmpeg a échoué"):
        vers_mp3(np.zeros(10, dtype=np.float32), tmp_path / "x.mp3")


def test_chemin_ffmpeg_looks_in_homebrew_when_path_is_minimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Une app lancée par LaunchAgent n'a pas /opt/homebrew/bin dans son PATH."""
    monkeypatch.setattr(synthese.shutil, "which", lambda name: None)
    binaire = tmp_path / "ffmpeg"
    binaire.write_bytes(b"")
    monkeypatch.setattr(synthese, "FFMPEG_CANDIDATES", (str(binaire),))
    assert synthese.chemin_ffmpeg() == str(binaire)
    monkeypatch.setattr(synthese, "FFMPEG_CANDIDATES", (str(tmp_path / "absent"),))
    assert synthese.chemin_ffmpeg() is None


# ---- Correctif des débuts de phrase (mlx-audio < 0.5.8) ----


@pytest.mark.parametrize(
    "version,attendu",
    [((0, 4, 6), True), ((0, 4, 2), True), ((0, 5, 7), True), ((0, 5, 8), False),
     ((0, 6, 4), False), ((1, 0, 0), False), (None, False)],
)
def test_correctif_debuts_necessaire_before_0_5_8(version, attendu):
    assert synthese.correctif_debuts_necessaire(version) is attendu


def test_appliquer_correctif_binds_the_corrected_generate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(synthese, "_version_mlx_audio", lambda: (0, 4, 6))
    model = SimpleNamespace(generate="original")
    assert synthese.appliquer_correctif_debuts(model) is True
    assert model.generate.__func__ is synthese._generate_corrige
    assert model.generate.__self__ is model


def test_appliquer_correctif_leaves_recent_versions_alone(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(synthese, "_version_mlx_audio", lambda: (0, 5, 8))
    model = SimpleNamespace(generate="original")
    assert synthese.appliquer_correctif_debuts(model) is False
    assert model.generate == "original"


def test_corrected_generate_has_no_extra_audio_token_step():
    """Le bug : un décodage supplémentaire avec le jeton [AUDIO] juste après le
    préremplissage. La copie embarquée ne doit pas le contenir, et doit garder
    la signature publique de mlx-audio."""
    import inspect

    source = inspect.getsource(synthese._generate_corrige)
    assert "audio_tok_emb" not in source
    assert "decode_one_frame" in source  # le reste de la boucle est bien là
    params = list(inspect.signature(synthese._generate_corrige).parameters)
    assert params[:3] == ["self", "text", "voice"] and "max_tokens" in params


def test_synthetiseur_applies_the_fix_when_loading(monkeypatch: pytest.MonkeyPatch, tmp_path):
    decode_origine = lambda codes: codes  # noqa: E731
    fake_model = SimpleNamespace(
        sample_rate=24_000, generate="original",
        audio_tokenizer=SimpleNamespace(decode=decode_origine),
    )
    fake_utils = SimpleNamespace(load=lambda source: fake_model)
    import sys
    monkeypatch.setitem(sys.modules, "mlx_audio", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", fake_utils)
    monkeypatch.setattr(synthese.hf_offline, "resolve_local_path", lambda repo: repo)
    applied: list = []
    monkeypatch.setattr(
        synthese, "appliquer_correctif_debuts", lambda m: applied.append(m) or True
    )
    synth = Synthetiseur("org/tts")
    synth.preload()
    assert applied == [fake_model]
    assert synth.correctif_debuts is True and synth.is_loaded
    # Le décodeur audio est lui aussi remplacé par sa version à mémoire bornée.
    assert fake_model.audio_tokenizer.decode is not decode_origine

