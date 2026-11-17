#!/bin/bash
# Banc d'essai « voix » : prépare deux environnements à part, puis mesure.
#
#   bash tools/bench_voix.sh            # tout : lecture, dictée en continu, mains libres
#   bash tools/bench_voix.sh tts        # une seule partie (tts, stt ou vad)
#
# Rien n'est installé dans l'app : Pocket TTS et Kyutai STT vont chacun dans
# leur dossier sous ~/.voxtral/bench/ (leurs bibliothèques sont incompatibles
# entre elles et avec celles de l'app). Pour tout effacer ensuite :
#   rm -rf ~/.voxtral/bench
# Relançable : ce qui est déjà installé n'est pas refait.

set -u
cd "$(dirname "$0")/.."

APP_PY="$HOME/.voxtral/venv/bin/python"
BENCH="$HOME/.voxtral/bench"

if [ ! -x "$APP_PY" ]; then
    echo "Voxtral Dictée n'est pas installé (introuvable : $APP_PY)." >&2
    exit 1
fi
mkdir -p "$BENCH"

# ---- Pocket TTS : même Python que l'app -------------------------------
if [ ! -x "$BENCH/tts/bin/pocket-tts" ]; then
    echo "[voix] Installation de Pocket TTS (quelques minutes, ~500 Mo)…"
    rm -rf "$BENCH/tts"
    "$APP_PY" -m venv "$BENCH/tts" \
        && "$BENCH/tts/bin/pip" install --quiet --upgrade pip \
        && "$BENCH/tts/bin/pip" install --quiet pocket-tts \
        || { echo "[voix] Pocket TTS n'a pas pu être installé : cette partie sera marquée en échec." >&2; rm -rf "$BENCH/tts"; }
fi

# ---- Kyutai STT : il lui faut Python 3.12 -----------------------------
# (sa dépendance sentencepiece 0.2.0 n'a pas de version prête pour 3.13)
if [ ! -x "$BENCH/stt/bin/python" ] || ! "$BENCH/stt/bin/python" -c "import moshi_mlx" 2>/dev/null; then
    PY312="$(command -v python3.12 || true)"
    if [ -z "$PY312" ] && command -v brew >/dev/null 2>&1; then
        echo "[voix] Installation de Python 3.12 (Homebrew)…"
        brew install --quiet python@3.12
        PY312="$(brew --prefix python@3.12 2>/dev/null)/bin/python3.12"
    fi
    if [ -n "$PY312" ] && [ -x "$PY312" ]; then
        echo "[voix] Installation de Kyutai STT (quelques minutes)…"
        rm -rf "$BENCH/stt"
        "$PY312" -m venv "$BENCH/stt" \
            && "$BENCH/stt/bin/pip" install --quiet --upgrade pip \
            && "$BENCH/stt/bin/pip" install --quiet "moshi_mlx==0.2.12" \
            || { echo "[voix] Kyutai STT n'a pas pu être installé : cette partie sera marquée en échec." >&2; rm -rf "$BENCH/stt"; }
    else
        echo "[voix] Python 3.12 introuvable : la partie « dictée en continu » sera marquée en échec." >&2
    fi
fi

# ---- Mesures ----------------------------------------------------------
# Le banc a besoin du réseau la première fois (modèles : ~220 Mo et ~670 Mo
# pour Pocket TTS, ~2,4 Go pour Kyutai STT, ~35 Mo pour Silero et Smart Turn).
"$APP_PY" tools/bench_voix.py "$@"
