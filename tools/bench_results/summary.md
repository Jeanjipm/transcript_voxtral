## Synthèse par modèle

| Modèle | Disque | WER synthé | WER réel | Temps réel (× à chaud) | 1re transcription 10 s | Chargement | Pic RAM | Horodatages | Hors-ligne |
|---|---|---|---|---|---|---|---|---|---|
| Voxtral Mini 3B 8-bit (dictée actuelle) | 5.4 Go | 3.9 % | 2.8 % | 10× (min 6×) | 1.0 s | 1.5 s | 5.3 Go | non | oui |
| Voxtral Mini 3B 4-bit (catalogue) | 3.2 Go | 3.3 % | 3.3 % | 12× (min 7×) | 0.8 s | 7.1 s | 5.8 Go | non | oui |
| Whisper large-v3-turbo (fichiers actuels) | 1.6 Go | 3.1 % | 3.1 % | 43× (min 17×) | 0.3 s | 0.7 s | 1.8 Go | phrase | oui |
| Whisper large-v3 (traduction) | 3.1 Go | 1.7 % | 3.1 % | 14× (min 7×) | 0.6 s | 1.0 s | 3.5 Go | phrase | oui |
| Voxtral Mini 4B Realtime 4-bit (Apache 2.0) | 3.1 Go | 2.6 % | 2.5 % | 3× (min 3×) | 2.4 s | 1.4 s | 3.5 Go | non | oui |
| Qwen3-ASR 1.7B 8-bit (Apache 2.0) | 2.5 Go | 2.3 % | 7.3 % | 25× (min 18×) | 0.5 s | 1.3 s | 3.0 Go | non | oui |
| Qwen3-ASR 1.7B 4-bit (Apache 2.0) | 1.6 Go | 3.8 % | 8.0 % | 37× (min 26×) | 0.2 s | 1.3 s | 2.2 Go | non | oui |
| Qwen3-ASR 4-bit + aligneur 0.6B (horodatages) | 1.6 Go | 3.8 % | 8.0 % | 30× (min 20×) | 0.3 s | 1.4 s | 3.2 Go | mot | oui |
| Parakeet TDT 0.6B v3 (CC BY 4.0, fp32) | 2.5 Go | 4.5 % | 19.6 % | 154× (min 109×) | 0.3 s | 1.3 s | 2.7 Go | phrase | oui |
| Parakeet TDT 0.6B v3 converti en bf16 | 2.5 Go | 4.5 % | 18.6 % | 204× (min 137×) | 0.1 s | 1.3 s | 2.7 Go | phrase | oui |
| Parakeet TDT 0.6B v3 bf16, blocs de 60 s | 2.5 Go | 4.5 % | 25.9 % | 205× (min 140×) | 0.1 s | 1.3 s | 2.7 Go | phrase | oui |
| Parakeet TDT 0.6B v3 bf16, blocs de 20 s | 2.5 Go | — | 2.5 % | 181× (min 174×) | 0.0 s | 1.4 s | 2.7 Go | phrase | non testé |

## Détail par fichier

| Modèle | Fichier | Durée | WER | Erreurs / mots | À chaud | À froid | Temps réel |
|---|---|---|---|---|---|---|---|
| voxtral-mini-3b-8bit | reel_entretien_5min | 300 s | — | 0 / 0 | 26.51 s | 26.70 s | 11× |
| voxtral-mini-3b-8bit | reel_formation_ia | 122 s | 3.4 % | 13 / 385 | 12.31 s | 12.48 s | 10× |
| voxtral-mini-3b-8bit | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 6.73 s | 6.78 s | 13× |
| voxtral-mini-3b-8bit | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 1.43 s | 1.49 s | 14× |
| voxtral-mini-3b-8bit | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.76 s | 0.81 s | 9× |
| voxtral-mini-3b-8bit | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 1.09 s | 1.15 s | 8× |
| voxtral-mini-3b-8bit | synth_court_dictee | 5 s | 17.6 % | 3 / 17 | 0.78 s | 1.28 s | 6× |
| voxtral-mini-3b-8bit | synth_court_noms | 6 s | 5.0 % | 1 / 20 | 0.97 s | 1.22 s | 6× |
| voxtral-mini-3b-8bit | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.81 s | 0.87 s | 6× |
| voxtral-mini-3b-8bit | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 0.89 s | 0.95 s | 7× |
| voxtral-mini-3b-8bit | synth_long_dictee_pro | 48 s | 3.3 % | 4 / 123 | 4.32 s | 4.37 s | 11× |
| voxtral-mini-3b-8bit | synth_long_tourisme | 54 s | 0.7 % | 1 / 150 | 4.45 s | 4.51 s | 12× |
| voxtral-mini-3b-8bit | synth_moyen_audrey | 15 s | 6.5 % | 3 / 46 | 1.66 s | 1.70 s | 9× |
| voxtral-mini-3b-8bit | synth_moyen_mail | 21 s | 1.8 % | 1 / 57 | 1.83 s | 1.90 s | 12× |
| voxtral-mini-3b-4bit | reel_entretien_5min | 300 s | — | 0 / 0 | 20.62 s | 20.84 s | 15× |
| voxtral-mini-3b-4bit | reel_formation_ia | 122 s | 4.9 % | 19 / 385 | 9.57 s | 9.23 s | 13× |
| voxtral-mini-3b-4bit | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 5.24 s | 5.20 s | 16× |
| voxtral-mini-3b-4bit | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 1.13 s | 1.19 s | 18× |
| voxtral-mini-3b-4bit | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.64 s | 0.67 s | 10× |
| voxtral-mini-3b-4bit | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 0.85 s | 0.89 s | 10× |
| voxtral-mini-3b-4bit | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.65 s | 0.70 s | 7× |
| voxtral-mini-3b-4bit | synth_court_noms | 6 s | 5.0 % | 1 / 20 | 0.78 s | 0.83 s | 8× |
| voxtral-mini-3b-4bit | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.68 s | 0.71 s | 8× |
| voxtral-mini-3b-4bit | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 0.72 s | 0.79 s | 9× |
| voxtral-mini-3b-4bit | synth_long_dictee_pro | 48 s | 14.6 % | 18 / 123 | 3.22 s | 3.24 s | 15× |
| voxtral-mini-3b-4bit | synth_long_tourisme | 54 s | 0.7 % | 1 / 150 | 3.71 s | 3.29 s | 15× |
| voxtral-mini-3b-4bit | synth_moyen_audrey | 15 s | 6.5 % | 3 / 46 | 1.22 s | 1.31 s | 12× |
| voxtral-mini-3b-4bit | synth_moyen_mail | 21 s | 1.8 % | 1 / 57 | 1.38 s | 1.43 s | 16× |
| whisper-large-v3-turbo | reel_entretien_5min | 300 s | — | 0 / 0 | 5.06 s | 5.11 s | 59× |
| whisper-large-v3-turbo | reel_formation_ia | 122 s | 4.2 % | 16 / 385 | 2.34 s | 2.37 s | 52× |
| whisper-large-v3-turbo | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 5.08 s | 5.51 s | 17× |
| whisper-large-v3-turbo | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 0.31 s | 0.42 s | 66× |
| whisper-large-v3-turbo | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.19 s | 0.30 s | 34× |
| whisper-large-v3-turbo | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 0.21 s | 0.32 s | 39× |
| whisper-large-v3-turbo | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.20 s | 0.31 s | 24× |
| whisper-large-v3-turbo | synth_court_noms | 6 s | 5.0 % | 1 / 20 | 0.23 s | 0.33 s | 26× |
| whisper-large-v3-turbo | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.20 s | 0.31 s | 25× |
| whisper-large-v3-turbo | synth_court_sigles | 6 s | 18.2 % | 4 / 22 | 0.21 s | 0.32 s | 29× |
| whisper-large-v3-turbo | synth_long_dictee_pro | 48 s | 2.4 % | 3 / 123 | 0.77 s | 0.87 s | 62× |
| whisper-large-v3-turbo | synth_long_tourisme | 54 s | 1.3 % | 2 / 150 | 0.91 s | 0.98 s | 60× |
| whisper-large-v3-turbo | synth_moyen_audrey | 15 s | 0.0 % | 0 / 46 | 0.30 s | 0.41 s | 50× |
| whisper-large-v3-turbo | synth_moyen_mail | 21 s | 0.0 % | 0 / 57 | 0.35 s | 0.46 s | 61× |
| whisper-large-v3 | reel_entretien_5min | 300 s | — | 0 / 0 | 45.29 s | 45.68 s | 7× |
| whisper-large-v3 | reel_formation_ia | 122 s | 2.9 % | 11 / 385 | 7.78 s | 7.90 s | 16× |
| whisper-large-v3 | reel_ma_voix | 86 s | 1.9 % | 4 / 214 | 4.85 s | 4.98 s | 18× |
| whisper-large-v3 | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 1.00 s | 1.10 s | 20× |
| whisper-large-v3 | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.43 s | 0.53 s | 15× |
| whisper-large-v3 | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 0.54 s | 0.65 s | 16× |
| whisper-large-v3 | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.57 s | 0.72 s | 8× |
| whisper-large-v3 | synth_court_noms | 6 s | 10.0 % | 2 / 20 | 0.62 s | 0.72 s | 10× |
| whisper-large-v3 | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.49 s | 0.59 s | 10× |
| whisper-large-v3 | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 0.58 s | 0.66 s | 11× |
| whisper-large-v3 | synth_long_dictee_pro | 48 s | 2.4 % | 3 / 123 | 2.99 s | 3.14 s | 16× |
| whisper-large-v3 | synth_long_tourisme | 54 s | 0.0 % | 0 / 150 | 3.17 s | 3.29 s | 17× |
| whisper-large-v3 | synth_moyen_audrey | 15 s | 0.0 % | 0 / 46 | 0.96 s | 1.07 s | 16× |
| whisper-large-v3 | synth_moyen_mail | 21 s | 0.0 % | 0 / 57 | 1.29 s | 1.40 s | 17× |
| voxtral-rt-4b-4bit | reel_entretien_5min | 300 s | — | 0 / 0 | 101.23 s | 102.66 s | 3× |
| voxtral-rt-4b-4bit | reel_formation_ia | 122 s | 2.3 % | 9 / 385 | 40.21 s | 40.32 s | 3× |
| voxtral-rt-4b-4bit | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 28.39 s | 28.45 s | 3× |
| voxtral-rt-4b-4bit | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 6.87 s | 6.89 s | 3× |
| voxtral-rt-4b-4bit | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 2.49 s | 2.48 s | 3× |
| voxtral-rt-4b-4bit | synth_court_chiffres | 8 s | 0.0 % | 0 / 22 | 3.03 s | 3.06 s | 3× |
| voxtral-rt-4b-4bit | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 1.89 s | 2.42 s | 3× |
| voxtral-rt-4b-4bit | synth_court_noms | 6 s | 20.0 % | 4 / 20 | 2.27 s | 2.28 s | 3× |
| voxtral-rt-4b-4bit | synth_court_question | 5 s | 0.0 % | 0 / 18 | 2.01 s | 2.03 s | 3× |
| voxtral-rt-4b-4bit | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 2.37 s | 2.39 s | 3× |
| voxtral-rt-4b-4bit | synth_long_dictee_pro | 48 s | 2.4 % | 3 / 123 | 15.76 s | 15.90 s | 3× |
| voxtral-rt-4b-4bit | synth_long_tourisme | 54 s | 0.0 % | 0 / 150 | 17.90 s | 18.06 s | 3× |
| voxtral-rt-4b-4bit | synth_moyen_audrey | 15 s | 0.0 % | 0 / 46 | 5.23 s | 5.23 s | 3× |
| voxtral-rt-4b-4bit | synth_moyen_mail | 21 s | 3.5 % | 2 / 57 | 7.25 s | 7.26 s | 3× |
| qwen3-asr-1.7b-8bit | reel_entretien_5min | 300 s | — | 0 / 0 | 13.18 s | 13.43 s | 23× |
| qwen3-asr-1.7b-8bit | reel_formation_ia | 122 s | 3.4 % | 13 / 385 | 5.27 s | 5.30 s | 23× |
| qwen3-asr-1.7b-8bit | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 3.02 s | 3.04 s | 28× |
| qwen3-asr-1.7b-8bit | reel_rene_char | 20 s | 18.2 % | 8 / 44 | 0.51 s | 0.54 s | 40× |
| qwen3-asr-1.7b-8bit | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.24 s | 0.27 s | 28× |
| qwen3-asr-1.7b-8bit | synth_court_chiffres | 8 s | 0.0 % | 0 / 22 | 0.38 s | 0.41 s | 22× |
| qwen3-asr-1.7b-8bit | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.27 s | 1.05 s | 18× |
| qwen3-asr-1.7b-8bit | synth_court_noms | 6 s | 20.0 % | 4 / 20 | 0.34 s | 0.37 s | 18× |
| qwen3-asr-1.7b-8bit | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.26 s | 0.29 s | 20× |
| qwen3-asr-1.7b-8bit | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 0.29 s | 0.32 s | 22× |
| qwen3-asr-1.7b-8bit | synth_long_dictee_pro | 48 s | 0.8 % | 1 / 123 | 1.80 s | 1.83 s | 26× |
| qwen3-asr-1.7b-8bit | synth_long_tourisme | 54 s | 2.0 % | 3 / 150 | 2.02 s | 2.05 s | 27× |
| qwen3-asr-1.7b-8bit | synth_moyen_audrey | 15 s | 0.0 % | 0 / 46 | 0.63 s | 0.66 s | 24× |
| qwen3-asr-1.7b-8bit | synth_moyen_mail | 21 s | 0.0 % | 0 / 57 | 0.81 s | 0.84 s | 27× |
| qwen3-asr-1.7b-4bit | reel_entretien_5min | 300 s | — | 0 / 0 | 9.65 s | 9.71 s | 31× |
| qwen3-asr-1.7b-4bit | reel_formation_ia | 122 s | 3.1 % | 12 / 385 | 3.55 s | 3.57 s | 34× |
| qwen3-asr-1.7b-4bit | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 2.01 s | 2.04 s | 43× |
| qwen3-asr-1.7b-4bit | reel_rene_char | 20 s | 20.5 % | 9 / 44 | 0.35 s | 0.38 s | 58× |
| qwen3-asr-1.7b-4bit | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.17 s | 0.20 s | 40× |
| qwen3-asr-1.7b-4bit | synth_court_chiffres | 8 s | 9.1 % | 2 / 22 | 0.26 s | 0.29 s | 32× |
| qwen3-asr-1.7b-4bit | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.18 s | 0.21 s | 26× |
| qwen3-asr-1.7b-4bit | synth_court_noms | 6 s | 20.0 % | 4 / 20 | 0.23 s | 0.25 s | 27× |
| qwen3-asr-1.7b-4bit | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.18 s | 0.21 s | 29× |
| qwen3-asr-1.7b-4bit | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 0.20 s | 0.23 s | 32× |
| qwen3-asr-1.7b-4bit | synth_long_dictee_pro | 48 s | 1.6 % | 2 / 123 | 1.17 s | 1.19 s | 41× |
| qwen3-asr-1.7b-4bit | synth_long_tourisme | 54 s | 1.3 % | 2 / 150 | 1.32 s | 1.35 s | 41× |
| qwen3-asr-1.7b-4bit | synth_moyen_audrey | 15 s | 2.2 % | 1 / 46 | 0.41 s | 0.44 s | 37× |
| qwen3-asr-1.7b-4bit | synth_moyen_mail | 21 s | 3.5 % | 2 / 57 | 0.52 s | 0.55 s | 41× |
| qwen3-asr-1.7b-4bit+aligner | reel_entretien_5min | 300 s | — | 0 / 0 | 10.88 s | 10.93 s | 28× |
| qwen3-asr-1.7b-4bit+aligner | reel_formation_ia | 122 s | 3.1 % | 12 / 385 | 3.96 s | 4.00 s | 31× |
| qwen3-asr-1.7b-4bit+aligner | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 2.29 s | 2.32 s | 37× |
| qwen3-asr-1.7b-4bit+aligner | reel_rene_char | 20 s | 20.5 % | 9 / 44 | 0.44 s | 0.47 s | 46× |
| qwen3-asr-1.7b-4bit+aligner | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.22 s | 0.25 s | 30× |
| qwen3-asr-1.7b-4bit+aligner | synth_court_chiffres | 8 s | 9.1 % | 2 / 22 | 0.31 s | 0.35 s | 27× |
| qwen3-asr-1.7b-4bit+aligner | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.24 s | 0.49 s | 20× |
| qwen3-asr-1.7b-4bit+aligner | synth_court_noms | 6 s | 20.0 % | 4 / 20 | 0.30 s | 0.31 s | 20× |
| qwen3-asr-1.7b-4bit+aligner | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.23 s | 0.26 s | 23× |
| qwen3-asr-1.7b-4bit+aligner | synth_court_sigles | 6 s | 0.0 % | 0 / 22 | 0.25 s | 0.28 s | 25× |
| qwen3-asr-1.7b-4bit+aligner | synth_long_dictee_pro | 48 s | 1.6 % | 2 / 123 | 1.32 s | 1.35 s | 36× |
| qwen3-asr-1.7b-4bit+aligner | synth_long_tourisme | 54 s | 1.3 % | 2 / 150 | 1.50 s | 1.52 s | 36× |
| qwen3-asr-1.7b-4bit+aligner | synth_moyen_audrey | 15 s | 2.2 % | 1 / 46 | 0.48 s | 0.51 s | 31× |
| qwen3-asr-1.7b-4bit+aligner | synth_moyen_mail | 21 s | 3.5 % | 2 / 57 | 0.62 s | 0.65 s | 35× |
| parakeet-tdt-0.6b-v3 | reel_entretien_5min | 300 s | — | 0 / 0 | 2.23 s | 2.33 s | 135× |
| parakeet-tdt-0.6b-v3 | reel_formation_ia | 122 s | 53.8 % | 207 / 385 | 0.68 s | 0.71 s | 179× |
| parakeet-tdt-0.6b-v3 | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 0.45 s | 0.47 s | 192× |
| parakeet-tdt-0.6b-v3 | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 0.11 s | 0.13 s | 184× |
| parakeet-tdt-0.6b-v3 | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.05 s | 0.07 s | 146× |
| parakeet-tdt-0.6b-v3 | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 0.06 s | 0.08 s | 145× |
| parakeet-tdt-0.6b-v3 | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.04 s | 1.39 s | 112× |
| parakeet-tdt-0.6b-v3 | synth_court_noms | 6 s | 15.0 % | 3 / 20 | 0.06 s | 0.07 s | 109× |
| parakeet-tdt-0.6b-v3 | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.04 s | 0.06 s | 114× |
| parakeet-tdt-0.6b-v3 | synth_court_sigles | 6 s | 18.2 % | 4 / 22 | 0.05 s | 0.07 s | 129× |
| parakeet-tdt-0.6b-v3 | synth_long_dictee_pro | 48 s | 2.4 % | 3 / 123 | 0.25 s | 0.27 s | 189× |
| parakeet-tdt-0.6b-v3 | synth_long_tourisme | 54 s | 0.0 % | 0 / 150 | 0.27 s | 0.30 s | 200× |
| parakeet-tdt-0.6b-v3 | synth_moyen_audrey | 15 s | 4.3 % | 2 / 46 | 0.10 s | 0.12 s | 158× |
| parakeet-tdt-0.6b-v3 | synth_moyen_mail | 21 s | 0.0 % | 0 / 57 | 0.13 s | 0.15 s | 166× |
| parakeet-tdt-0.6b-v3-bf16 | reel_entretien_5min | 300 s | — | 0 / 0 | 1.41 s | 2.53 s | 213× |
| parakeet-tdt-0.6b-v3-bf16 | reel_formation_ia | 122 s | 50.6 % | 195 / 385 | 0.48 s | 0.51 s | 252× |
| parakeet-tdt-0.6b-v3-bf16 | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 0.32 s | 0.34 s | 267× |
| parakeet-tdt-0.6b-v3-bf16 | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 0.09 s | 0.17 s | 227× |
| parakeet-tdt-0.6b-v3-bf16 | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.04 s | 0.06 s | 180× |
| parakeet-tdt-0.6b-v3-bf16 | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 0.05 s | 0.07 s | 175× |
| parakeet-tdt-0.6b-v3-bf16 | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.03 s | 0.09 s | 148× |
| parakeet-tdt-0.6b-v3-bf16 | synth_court_noms | 6 s | 15.0 % | 3 / 20 | 0.04 s | 0.06 s | 141× |
| parakeet-tdt-0.6b-v3-bf16 | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.04 s | 0.06 s | 137× |
| parakeet-tdt-0.6b-v3-bf16 | synth_court_sigles | 6 s | 18.2 % | 4 / 22 | 0.04 s | 0.06 s | 161× |
| parakeet-tdt-0.6b-v3-bf16 | synth_long_dictee_pro | 48 s | 2.4 % | 3 / 123 | 0.18 s | 0.20 s | 261× |
| parakeet-tdt-0.6b-v3-bf16 | synth_long_tourisme | 54 s | 0.0 % | 0 / 150 | 0.19 s | 0.21 s | 283× |
| parakeet-tdt-0.6b-v3-bf16 | synth_moyen_audrey | 15 s | 4.3 % | 2 / 46 | 0.07 s | 0.10 s | 203× |
| parakeet-tdt-0.6b-v3-bf16 | synth_moyen_mail | 21 s | 0.0 % | 0 / 57 | 0.10 s | 0.12 s | 210× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | reel_entretien_5min | 300 s | — | 0 / 0 | 1.24 s | 1.28 s | 242× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | reel_formation_ia | 122 s | 73.2 % | 282 / 385 | 0.47 s | 0.49 s | 261× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | reel_ma_voix | 86 s | 0.0 % | 0 / 214 | 0.34 s | 0.37 s | 250× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 0.09 s | 0.11 s | 234× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_court_audrey | 7 s | 0.0 % | 0 / 16 | 0.04 s | 0.06 s | 180× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_court_chiffres | 8 s | 4.5 % | 1 / 22 | 0.05 s | 0.07 s | 174× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_court_dictee | 5 s | 0.0 % | 0 / 17 | 0.03 s | 0.05 s | 146× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_court_noms | 6 s | 15.0 % | 3 / 20 | 0.04 s | 0.06 s | 140× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_court_question | 5 s | 0.0 % | 0 / 18 | 0.04 s | 0.06 s | 141× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_court_sigles | 6 s | 18.2 % | 4 / 22 | 0.04 s | 0.06 s | 163× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_long_dictee_pro | 48 s | 2.4 % | 3 / 123 | 0.18 s | 0.20 s | 259× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_long_tourisme | 54 s | 0.0 % | 0 / 150 | 0.19 s | 0.22 s | 283× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_moyen_audrey | 15 s | 4.3 % | 2 / 46 | 0.08 s | 0.09 s | 199× |
| parakeet-tdt-0.6b-v3-bf16-blocs60 | synth_moyen_mail | 21 s | 0.0 % | 0 / 57 | 0.11 s | 0.13 s | 202× |
| parakeet-tdt-0.6b-v3-bf16-blocs20 | reel_entretien_5min | 300 s | — | 0 / 0 | 1.67 s | 1.72 s | 180× |
| parakeet-tdt-0.6b-v3-bf16-blocs20 | reel_formation_ia | 122 s | 2.6 % | 10 / 385 | 0.69 s | 0.74 s | 179× |
| parakeet-tdt-0.6b-v3-bf16-blocs20 | reel_ma_voix | 86 s | 0.5 % | 1 / 214 | 0.44 s | 0.48 s | 193× |
| parakeet-tdt-0.6b-v3-bf16-blocs20 | reel_rene_char | 20 s | 4.5 % | 2 / 44 | 0.12 s | 0.25 s | 174× |

## Textes produits (phrases synthétiques seulement)

### Voxtral Mini 3B 8-bit (dictée actuelle)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, salle B, avec un budget de 3500 euros hors taxe.
- **synth_court_dictee** (17.6 %) : Je peux te rappeler demain matin à 9h pour finaliser le devis du client de Chambéry.
- **synth_court_noms** (5.0 %) : Jean-Jérôme Danton rencontre l'Office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Ovrin-sur-Mérel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension ? Ou pas?
- **synth_court_sigles** (0.0 %) : Le PDF de la Direction des Finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (3.3 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point, l'état d'avancement du projet. Le lot 1 est livré, le lot 2 est en recette depuis lundi et le lot 3 démarre le 15 janvier. Deuxième point, les risques. Le prestataire signale un retard possible de trois semaines sur la reprise des données liées à la qualité des fichiers sources. Décision : une lecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point : le budget L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10h, en visioconférence.
- **synth_long_tourisme** (0.7 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portés surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement : les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (6.5 %) : Pour la commande du marché, il nous faut 3 kg de pommes, 2 bouquets de basilic, un litre d'huile d'olive, si ce et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (1.8 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur ? Et prévenir les douze participants. Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Gérôme

### Voxtral Mini 3B 4-bit (catalogue)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, salle B, avec un budget de 3500 euros hors taxe.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry ?
- **synth_court_noms** (5.0 %) : Jean-Jérôme Danton rencontre l'Office de tourisme de Brumeval, les Sorbiers et la mairie de Saint-Ovrin-sur-Mérel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension ? Ou pas ?
- **synth_court_sigles** (0.0 %) : Le PDF de la Direction des Finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (14.6 %) : Présent : la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire.

1. L'état d'avancement du projet.
   - Le lot 1 est livré, le lot 2 est en recette depuis lundi et le lot 3 démarre le 15 janvier.

2. Les risques.
   - Le prestataire signale un retard possible de 3 semaines sur la reprise des données, lié à la qualité des fichiers sources.
   - Décision : une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive.

3. Le budget.
   - L'enveloppe initiale de 200 000 € est consommée à 65 %, ce qui est conforme au calendrier.

Prochaine réunion le 8 février à 10h, en visioconférence.
- **synth_long_tourisme** (0.7 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portés surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement : les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (6.5 %) : Pour la commande du marché, il nous faut 3 kg de pommes, 2 bouquets de basilic, un litre d'huile d'olive, si ce et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (1.8 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur ? Et prévenir les 12 participants. Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Gérôme

### Whisper large-v3-turbo (fichiers actuels)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, salle B, avec un budget de 3500 euros hors taxe.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry ?
- **synth_court_noms** (5.0 %) : Jean-Jérôme Danton rencontre l'office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Ovrin-sur-Mérel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas ?
- **synth_court_sigles** (18.2 %) : Le PDF de la Direction des Finances sur la TV a été envoyé à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (2.4 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point, l'état d'avancement du projet. Le lot 1 est livré, le lot 2 est en recette depuis lundi, et le lot 3 démarre le 15 janvier. Deuxième point, les risques. Le prestataire signale un retard possible de trois semaines sur la reprise des données, liée à la qualité des fichiers sources. Décision, une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point, le budget. L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10 heures, en visioconférence.
- **synth_long_tourisme** (1.3 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentèse, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portés surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement, les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (0.0 %) : Pour la commande du marché, il nous faut 3 kilos de pommes, 2 bouquets de basilic, 1 litre d'huile d'olive, 6 œufs et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (0.0 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants ? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme.

### Whisper large-v3 (traduction)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, salle B, avec un budget de 3500 euros hors taxe.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry ?
- **synth_court_noms** (10.0 %) : Jean-Jérôme Danton rencontre l'office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Auvrin-sur-Merrel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas ?
- **synth_court_sigles** (0.0 %) : Le PDF de la Direction des Finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (2.4 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point, l'état d'avancement du projet. Le lot 1 est livré, le lot 2 est en recette depuis lundi, et le lot 3 démarre le 15 janvier. Deuxième point, les risques. Le prestataire signale un retard possible de trois semaines sur la reprise des données, liées à la qualité des fichiers sources. Décision, une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point, le budget. L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10h, en visioconférence.
- **synth_long_tourisme** (0.0 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement, les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (0.0 %) : Pour la commande du marché, il nous faut 3 kilos de pommes, 2 bouquets de basilic, 1 litre d'huile d'olive, 6 œufs et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (0.0 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants ? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme.

### Voxtral Mini 4B Realtime 4-bit (Apache 2.0)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (0.0 %) : La réunion est fixée au 12 mars 2026 à 14h30, salle B, avec un budget de 3 500 euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry?
- **synth_court_noms** (20.0 %) : Jean-Jérôme dont on rencontre l'Office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Auverin-sur-Mérel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas?
- **synth_court_sigles** (0.0 %) : Le PDF de la Direction des Finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (2.4 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point, l'état d'avancement du projet. Le lot 1 est livré, le lot 2 est en recette depuis lundi, et le lot 3 démarre le 15 janvier. Deuxième point, les risques. Le prestataire signale un retard possible de trois semaines sur la reprise des données, liées à la qualité des fichiers sources. Décision, une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point, le budget. L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10h, en visioconférence..
- **synth_long_tourisme** (0.0 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement, les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années..
- **synth_moyen_audrey** (0.0 %) : Pour la commande du marché, il nous faut 3 kilos de pommes, 2 bouquets de basilic, 1 litre d'huile d'olive, 6 oeufs et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine..
- **synth_moyen_mail** (3.5 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage? C'est avec le vidéoprojecteur, et prévenir les douze participants? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme..

### Qwen3-ASR 1.7B 8-bit (Apache 2.0)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (0.0 %) : La réunion est fixée au douze mars deux mille vingt-six à quatorze heures trente, salle B, avec un budget de trois mille cinq cents euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à neuf heures pour finaliser le devis du client de Chambéry?
- **synth_court_noms** (20.0 %) : Jean Jérôme, dont on rencontre l'office de tourisme de Brumeval, les Sorbiers et la mairie de Saint-Ouvrain-sur-Merel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas ?
- **synth_court_sigles** (0.0 %) : Le PDF de la direction des finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (0.8 %) : Compte rendu de la réunion du comité de pilotage présent : la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point : l'état d'avancement du projet. Le lot 1 est livré, le lot 2 est en recette depuis lundi, et le lot 3 démarre le 15 janvier. Deuxième point : les risques. Le prestataire signale un retard possible de trois semaines sur la reprise des données, lié à la qualité des fichiers sources. Décision : une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archives. Troisième point : le budget. L'enveloppe initiale de 200 000 euros est consommée à 65 %, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10 heures, en visioconférence.
- **synth_long_tourisme** (2.0 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculé d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double : simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 10,8 % des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement. Les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (0.0 %) : Pour la commande du marché, il nous faut trois kilos de pommes, deux bouquets de basilic, un litre d'huile d'olive, six œufs et une baguette de tradition. Le total ne doit pas dépasser quarante euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (0.0 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants ? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme.

### Qwen3-ASR 1.7B 4-bit (Apache 2.0)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (9.1 %) : La réunion est fixée au 12 mars 2026 à 14 heures 30, Salbey, avec un budget de 3 500 euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à neuf heures pour finaliser le devis du client de Chambéry?
- **synth_court_noms** (20.0 %) : Jean Jérôme, dont on rencontre l'office de tourisme de Brumeval, les Sorbiers et la mairie de Saint-Auvrin-sur-Merel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas?
- **synth_court_sigles** (0.0 %) : Le PDF de la direction des finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (1.6 %) : Compte rendu de la réunion du comité de pilotage présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point, l'état d'avancement du projet, le lot 1 est livré, le lot 2 est en recette depuis lundi et le lot 3 démarre le 15 janvier. Deuxième point, les risques, le prestataire signale un retard possible de trois semaines sur la reprise des données, lié à la qualité des fichiers source. Décision, une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archives. Troisième point, le budget, l'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10 heures, en visioconférence.
- **synth_long_tourisme** (1.3 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double : simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 10,8% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement. Les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (2.2 %) : Pour la commande du marché, il nous faut trois kilos de pommes, deux bouquets de basilic, un litre d'huile d'olive, six euros et une baguette de tradition. Le total ne doit pas dépasser quarante euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (3.5 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Nancy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants. Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Gérome.

### Qwen3-ASR 4-bit + aligneur 0.6B (horodatages)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (9.1 %) : La réunion est fixée au 12 mars 2026 à 14 heures 30, Salbey, avec un budget de 3 500 euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à neuf heures pour finaliser le devis du client de Chambéry?
- **synth_court_noms** (20.0 %) : Jean Jérôme, dont on rencontre l'office de tourisme de Brumeval, les Sorbiers et la mairie de Saint-Auvrin-sur-Merel.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas?
- **synth_court_sigles** (0.0 %) : Le PDF de la direction des finances sur la TVA est à envoyer à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (1.6 %) : Compte rendu de la réunion du comité de pilotage présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point, l'état d'avancement du projet, le lot 1 est livré, le lot 2 est en recette depuis lundi et le lot 3 démarre le 15 janvier. Deuxième point, les risques, le prestataire signale un retard possible de trois semaines sur la reprise des données, lié à la qualité des fichiers source. Décision, une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archives. Troisième point, le budget, l'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10 heures, en visioconférence.
- **synth_long_tourisme** (1.3 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double : simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 10,8% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement. Les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (2.2 %) : Pour la commande du marché, il nous faut trois kilos de pommes, deux bouquets de basilic, un litre d'huile d'olive, six euros et une baguette de tradition. Le total ne doit pas dépasser quarante euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (3.5 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Nancy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants. Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Gérome.

### Parakeet TDT 0.6B v3 (CC BY 4.0, fp32)

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, Sal B, avec un budget de 3500 euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry ?
- **synth_court_noms** (15.0 %) : Jean-Jérôme dont on rencontre l'office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Auvrain-sur-Mérelle.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas ?
- **synth_court_sigles** (18.2 %) : Le PDF de la direction des finances sur la TV a été envoyé à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (2.4 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point. L'état d'avancement du projet. Le lot 1 est livré. Le lot 2 est en recette depuis lundi. Et le lot 3 démarre le 15 janvier. Deuxième point. Les risques. Le prestataire signale un retard possible de 3 semaines sur la reprise des données, liée à la qualité des fichiers sources. Décision. Une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point : le budget. L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10h, en visioconférence.
- **synth_long_tourisme** (0.0 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement, les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (4.3 %) : Pour la commande du marché, il nous faut 3 kg de pommes, 2 bouquets de basilic, 1 litre d'huile d'olive, 6 sœurs et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (0.0 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants ? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme.

### Parakeet TDT 0.6B v3 converti en bf16

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, Sal B, avec un budget de 3500 euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry ?
- **synth_court_noms** (15.0 %) : Jean-Jérôme dont on rencontre l'office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Auvrain-sur-Mérelle.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas ?
- **synth_court_sigles** (18.2 %) : Le PDF de la direction des finances sur la TV a été envoyé à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (2.4 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point. L'état d'avancement du projet. Le lot 1 est livré. Le lot 2 est en recette depuis lundi. Et le lot 3 démarre le 15 janvier. Deuxième point. Les risques. Le prestataire signale un retard possible de 3 semaines sur la reprise des données, liée à la qualité des fichiers sources. Décision. Une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point : le budget. L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10h, en visioconférence.
- **synth_long_tourisme** (0.0 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement, les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (4.3 %) : Pour la commande du marché, il nous faut 3 kg de pommes, 2 bouquets de basilic, 1 litre d'huile d'olive, 6 sœurs et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (0.0 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants ? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme.

### Parakeet TDT 0.6B v3 bf16, blocs de 60 s

- **synth_court_audrey** (0.0 %) : Bonjour, je voudrais réserver deux places pour le concert de samedi soir, si possible côté balcon.
- **synth_court_chiffres** (4.5 %) : La réunion est fixée au 12 mars 2026 à 14h30, Sal B, avec un budget de 3500 euros hors taxes.
- **synth_court_dictee** (0.0 %) : Peux-tu me rappeler demain matin à 9h pour finaliser le devis du client de Chambéry ?
- **synth_court_noms** (15.0 %) : Jean-Jérôme dont on rencontre l'office de tourisme de Brumeval-les-Sorbiers et la mairie de Saint-Auvrain-sur-Mérelle.
- **synth_court_question** (0.0 %) : Est-ce que l'outil reconnaît bien les questions, les virgules et les points de suspension, ou pas ?
- **synth_court_sigles** (18.2 %) : Le PDF de la direction des finances sur la TV a été envoyé à l'URSSAF et à la CCI avant vendredi.
- **synth_long_dictee_pro** (2.4 %) : Compte rendu de la réunion du comité de pilotage. Présent, la directrice générale, le responsable informatique, deux représentants des utilisateurs et le prestataire. Premier point. L'état d'avancement du projet. Le lot 1 est livré. Le lot 2 est en recette depuis lundi. Et le lot 3 démarre le 15 janvier. Deuxième point. Les risques. Le prestataire signale un retard possible de 3 semaines sur la reprise des données, liée à la qualité des fichiers sources. Décision. Une relecture manuelle des 500 fiches les plus anciennes sera confiée au service archive. Troisième point : le budget. L'enveloppe initiale de 200 000 euros est consommée à 65%, ce qui est conforme au calendrier. Prochaine réunion le 8 février à 10h, en visioconférence.
- **synth_long_tourisme** (0.0 %) : Le tourisme de montagne traverse une période de transformation profonde. Les stations de moyenne altitude voient leur enneigement reculer d'année en année, et les offices de tourisme doivent réinventer leur offre pour attirer des visiteurs toute l'année. La randonnée, le vélo électrique, les activités en famille et le patrimoine local deviennent des arguments aussi importants que le ski. Dans la vallée de la Tarentaise, plusieurs communes ont choisi de mutualiser leurs moyens pour financer une plateforme de réservation commune. L'objectif est double, simplifier le parcours du client et récupérer des données fiables sur la fréquentation. Les premiers résultats montrent une hausse de 18% des séjours hors saison, portée surtout par une clientèle de proximité venue de Lyon, de Genève et de Grenoble. Reste la question du financement, les budgets communaux sont contraints, et les partenaires privés attendent des garanties avant de s'engager sur plusieurs années.
- **synth_moyen_audrey** (4.3 %) : Pour la commande du marché, il nous faut 3 kg de pommes, 2 bouquets de basilic, 1 litre d'huile d'olive, 6 sœurs et une baguette de tradition. Le total ne doit pas dépasser 40 euros, sinon on reporte le fromage à la semaine prochaine.
- **synth_moyen_mail** (0.0 %) : Bonjour Claire, merci pour ton retour rapide. Je te confirme que la formation sur l'intelligence artificielle aura lieu les 3 et 4 décembre à Annecy. Peux-tu réserver la salle du deuxième étage, celle avec le vidéoprojecteur, et prévenir les douze participants ? Je t'envoie le programme détaillé en fin de semaine. Bonne journée, Jean-Jérôme.

### Parakeet TDT 0.6B v3 bf16, blocs de 20 s

