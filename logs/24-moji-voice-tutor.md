# Moji: real-time voice English practice partner
**Dates:** 2026-09-26 (all 10 commits; HANDOFF last updated 2026-09-26) · **Status:** Phase 1 (live voice practice) and smart memory are built and in daily use. Phase 3 (English tutor feedback) is next and not started. · **Stack:** Python (FastAPI, stdlib sqlite3, google-genai) · TypeScript, React, Vite · SVG face and Web Audio

## Goal
Moji is a voice conversation partner for practising spoken English on a phone or PC. The longer-term goal is for every conversation to teach the user something concrete: natural phrasings, new vocabulary, and one small goal for next time.

## Architecture
- **Voice model:** Gemini Live (`gemini-3.8-live`, as named in the project's code and HANDOFF). Audio streams between the browser and Gemini over WebSocket. Transcription is enabled. Context compression runs from 25k down to 8k tokens.
- **Auth:** The backend mints a single-use ephemeral token for each call with `mint_token()`. The whole Live session config is locked into the token server-side. The client can only change `sessionResumption`. The token lives 2 h, so it outlasts a long call. Resuming a session does not count as a use.
- **Backend (`backend/app/`):** FastAPI with stdlib sqlite3. It is not on the audio path. It mints tokens, classifies facial expressions, and stores calls and memories.
- **Memory layer:** After each call, `gemini-3.8-flash` turns the transcript into facts of at most 15 words. Facts are merged, updated and expired rather than appended. Up to about 300 tokens of them are injected into the next call's context. The `memories` table is capped at 60 rows. `mistake` memories are stored but hidden from the live model. Transcripts are kept for 7 days. Summaries and median latency are kept per call.
- **Expression classifier:** `gemini-3.5-flash-lite` with structured output, choosing one of 34 emotions. Given an empty sentence, it returns Moji's reaction to what the user said.
- **Frontend (`frontend/src/`):** React and TypeScript with Vite. The mic feeds a speech gate, which sends audio to Gemini only while the user is speaking. Moji's audio plays back gapless at 24 kHz. Lip sync uses HeadAudio viseme detection.
- **Avatar:** An SVG emoji-style face, independent of the AI layer. Face expressions change phrase by phrase. Moji's streamed transcript is split into sentences and clauses, and each gets its own face.
- **End-to-end flow:**
  1. The client calls `POST /api/calls`. The backend summarizes any abandoned call, creates the new call, and mints a token with memory baked in.
  2. The browser opens a WebSocket to Gemini Live with that token.
  3. Mic audio is gated and streamed. Moji's audio and streamed transcript come back. The face follows the transcript.
  4. The transcript is saved after every exchange (`POST /api/calls/{id}/transcript`).
  5. On end (`POST /api/calls/{id}/end`), memory consolidation runs in the background.

## What works today
- Real-time voice calls with Moji. Typed input also works.
- An animated emoji face with 36 expressions, changing phrase by phrase, plus viseme lip sync.
- 21 colour themes, 30 voices, and a Normal or Slow pace. Slow is requested in the instructions, because Gemini has no speaking-rate setting.
- Cross-session memory, saved calls, and a latency overlay.
- A self tile showing a mic waveform, with an optional camera.
- Calls longer than 10 minutes, using session resumption.
- Phone testing over the local network, with HTTPS.

## Next
Phase 3 is planned, not built. The goal is an English tutor that gives feedback after each call.
- After the call, show 3 to 5 things the user said, each with a more natural native phrasing.
- Add pronunciation and fluency notes where possible.
- List new vocabulary from the call.
- Set one small goal for the next call.
- Offer an optional coach mode with light in-conversation corrections. The rule is not to interrupt the conversation to correct grammar by default.
- Open design questions, to be settled before coding:
  - Where feedback appears: a post-call screen, or a history of past calls.
  - Whether Moji follows up on the last call's goal.
  - Whether coach mode is on or off by default.
  - Whether progress is tracked over time, for example whether mistakes decrease.
  - How pronunciation can be judged from audio. The Live API returns transcripts, not pronunciation scores, so this needs research.

## Key files
- `backend/app/main.py`: FastAPI endpoints for calls, transcripts, ending a call, voices, and expression classification.
- `backend/app/live.py`: Live session config, model name, voices, pace settings, and `mint_token()`.
- `backend/app/persona.py`: Moji's system instruction and the `PACES` settings.
- `backend/app/expression.py`: the 34 emotions and `classify()`.
- `backend/app/memory.py`: post-call fact extraction, merge and expiry, and `context_block()` for injection into the next call.
- `backend/app/db.py`: SQLite schema for the `calls` and `memories` tables.
- `frontend/src/call/useCall.ts`: orchestrates a call: mic, transcript, faces, latency stats, and ending on page hide.
- `frontend/src/call/speechGate.ts`: sends mic audio only while the user speaks, with an adaptive threshold and a 320 ms pre-roll. It sends digital silence until the reply starts.
- `frontend/src/call/phrases.ts`: splits Moji's streamed transcript into phrases for per-phrase faces.
- `frontend/src/voice/types.ts`: the `VoiceBackend` interface, a seam for swapping in another voice model.
- `frontend/src/voice/geminiLive.ts`: the Gemini Live implementation, with resumption on `goAway` or a dropped connection.
- `frontend/src/audio/mic.ts` and `mic.worklet.js`: 16 kHz mic capture in 512-sample chunks.
- `frontend/src/audio/player.ts`: gapless 24 kHz playback, barge-in flush, and lip sync.
- `frontend/src/face/`: the SVG face, its expression presets, motion, visemes, and themes.
