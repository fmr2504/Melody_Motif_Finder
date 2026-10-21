# 🎵 Melody Motif Finder

**A Deterministic Finite Automaton (DFA) Melodic Contour & Motif Search Engine**  
*Built with Python 3.10+ and Streamlit (Phases 1 to 5 Complete)*

---

## 📖 Overview

**Melody Motif Finder** is a pattern-matching search engine that locates musical motifs across a validated library of public-domain melodies. Instead of requiring exact pitches or keys, the engine encodes melodies into **Parsons-style melodic contours** over the alphabet $\{U, D, R\}$ and searches every song in linear time using a **Knuth-Morris-Pratt (KMP) Deterministic Finite Automaton (DFA)**.

Because contour matching is transposition-invariant while MIDI pitches are preserved alongside the search, the system automatically distinguishes between:
- **Exact Repeats** ($0$ semitones shift)
- **Transposed Repeats** ($\pm k$ semitones shift)
- **Pure Contour Matches** (when searching via contour codes such as `c: RUU`)

It supports three input modalities:
1. **Typed Scientific Pitch Notes** (e.g., `E4 E4 F4 G4`)
2. **Direct Contour Queries** (e.g., `c: RUU` or `c: DDD`)
3. **Recorded, Uploaded, or Synthesized Audio** (converted to floating-point MIDI pitches via fundamental frequency tracking, median smoothing, and dead-zone segmentation)

---

## ✨ Key Features (Phases 1–5)

| Phase | Module / Area | Key Capabilities |
| :--- | :--- | :--- |
| **Phase 1** | `encoding` / `dfa` / `render` | Scientific pitch parsing (`C4` = MIDI $60$), contour encoding over $\{U, D, R\}$ with optional `min_step` dead-zone, KMP-DFA construction (`build_dfa`) and execution (`run_dfa` with `restart_after_match=True`), and aligned ASCII trace tables (`CELL = 6`). |
| **Phase 2** | `library` (`songs.json`) | Built-in validated collection of **10 public-domain melodies**. Automatically verifies on startup that every song's stored contour matches `encode_contour(midis)`. |
| **Phase 3** | `search` | Builds the query DFA **once per query** and runs it across all songs, ranking hits by match count and calculating semitone transposition shifts (`round(song_midis[start] - query.midis[0])`). |
| **Phase 4** | `app` (Streamlit UI) | Interactive web interface with a sidebar song browser, short-query warning ($< 6$ notes), dual-line contour charts highlighting matched notes, aligned DFA state traces, and closest-attempt diagnostics for non-matching songs. |
| **Phase 5** | `audio` | Full audio-to-contour pipeline (`load_audio` $\rightarrow$ `extract_pitch` $\rightarrow$ per-voiced-run median `smooth` $\rightarrow$ `segment_notes` $\rightarrow$ `audio_to_query`), an **Editable Contour Safety Net**, and a built-in **Sine-Wave Melody Synthesizer** (with optional $0.3$-semitone vibrato). |

---

## 🔤 How Melodic Contour Encoding Works

Given a sequence of $N$ MIDI note values $m_0, m_1, \dots, m_{N-1}$ and a dead-zone threshold $\text{min\_step} \ge 0$, the encoder produces a string of $N - 1$ symbols over the alphabet $\{U, D, R\}$:

$$\text{symbol}_i = \begin{cases} \text{R (Repeat }\rightarrow\text{)} & \text{if } |m_i - m_{i-1}| \le \text{min\_step} \\ \text{U (Up }\nearrow\text{)} & \text{if } m_i - m_{i-1} > \text{min\_step} \\ \text{D (Down }\searrow\text{)} & \text{if } m_i - m_{i-1} < -\text{min\_step} \end{cases}$$

- **Library & Typed Queries:** Use $\text{min\_step} = 0.0$ (exact semitone comparison).
- **Hummed / Audio Queries:** Use $\text{min\_step} = 0.75$ semitones by default so natural human pitch drift or vibrato on a repeated note still encodes cleanly as `R`.
- **Match Span:** A DFA match starting at contour index $s$ with pattern length $L$ spans song notes $s$ through $s + L$ inclusive ($L + 1$ notes).

---

## 🏗️ System Architecture

Dependencies point strictly downward so core algorithms remain decoupled from the UI and audio layers:

```text
   Typed Notes          Contour (c: RUU)        Audio (Record / Upload / Synth)
        │                      │                              │
        └──────────────────────┼──────────────────────────────┘
                               ▼
                         ┌───────────┐
                         │  app.py   │  (Streamlit UI - Phases 4 & 5)
                         └─────┬─────┘
         ┌───────────────┬─────┴─────────┬───────────────┐
         ▼               ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │ audio.py  │   │ search.py │   │library.py │   │ render.py │
   │ (Phase 5) │   │(Phases1,3)│   │ (Phase 2) │   │ (Phase 1) │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │               │
         └───────────────►┌─────────────┐◄───────────────┘
                          │ encoding.py │  (Notes <-> MIDI <-> Contour)
                          └──────┬──────┘
                                 ▼
                          ┌─────────────┐
                          │   dfa.py    │  (build_dfa / run_dfa)
                          └─────────────┘
```

*(Note: In the standalone prototype, all 7 modules are cleanly organized and labeled inside `app.py` for zero-configuration execution, and can be split into the package structure at any time.)*

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python 3.10 or newer**
- `pip` (Python package manager)

### 2. Create a Virtual Environment
Open a terminal in your project folder and run:

**Windows (PowerShell / CMD):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install streamlit numpy pandas scipy librosa pytest
```
*(Note: If `librosa` or `scipy` is unavailable on a restricted machine, `app.py` automatically falls back to a built-in pure-NumPy pitch tracker and WAV reader so all 5 phases continue to work seamlessly.)*

### 4. Run the Application
```bash
streamlit run app.py
```
Your browser will automatically open to **`http://localhost:8501`**.

---

## 🎮 Usage Guide

### 1. Typed Notes & Contour Search (Tab 1)
- **Search by Notes:** Enter space-separated scientific pitch notes such as:
  ```text
  E4 E4 F4 G4
  ```
  Click **Search Library**. You will see **Ode to Joy (opening)** match twice:
  - **Match #1 (Notes 1–4):** `E4 E4 F4 G4` $\rightarrow$ *Exact repeat ($0$ semitones shift)*
  - **Match #2 (Notes 9–12):** `C4 C4 D4 E4` $\rightarrow$ *Transposed repeat ($4$ semitones lower, shift $= -4$)*
- **Search by Contour Only:** Prefix your query with `c:` to match contour shapes without reference pitches:
  ```text
  c: DDD
  ```
  Because `restart_after_match=True` is active in the DFA, overlapping matches (such as contour indices `4` and `5` in *Ode to Joy*) are automatically detected.

### 2. Audio Input & Synthesizer (Tab 2)
- **Microphone / File Upload:** Record yourself humming or singing at least **6 distinct notes** (sing *"ta ta ta"* with brief gaps between repeated notes) or upload a `.wav` clip ($\le 20\text{ s}$).
- **Built-in Test Synthesizer:** Select a preset melody (e.g., *Ode to Joy* or *Twinkle Twinkle*), optionally enable the **$0.3$-semitone vibrato** checkbox, and click **Synthesize & Analyze Clip** to test the Phase 5 pitch extraction pipeline without a microphone.
- **Editable Contour Safety Net:** If a pitch tracker octave jump causes a single-symbol error in the extracted contour, edit the `U/D/R` string directly in the **Detected Contour Code** box and click **Search with Audio Query**.

### 3. Automated Acceptance Checks (Tab 3)
- Switch to the **Architecture & Acceptance Checks** tab and click **Run All Phase 1–5 Acceptance Checks** to execute the built-in verification suite covering scale contours, dead-zone encoding, library integrity, exact/transposed matches, overlapping DFA transitions, and synthetic sine-wave audio tests.

---

## 🎼 Included Public-Domain Song Library

All 10 melodies are validated on load against `encode_contour`:
1. **Twinkle Twinkle (opening)** (`twinkle`)
2. **Ode to Joy (opening)** (`ode-to-joy`)
3. **Frère Jacques (opening)** (`frere-jacques`)
4. **Mary Had a Little Lamb** (`mary-lamb`)
5. **Row Row Row Your Boat** (`row-your-boat`)
6. **Jingle Bells (chorus)** (`jingle-bells`)
7. **London Bridge Is Falling Down** (`london-bridge`)
8. **Amazing Grace (opening)** (`amazing-grace`)
9. **Auld Lang Syne (opening)** (`auld-lang-syne`)
10. **Brahms' Lullaby (opening)** (`brahms-lullaby`)

---

## ⚠️ Known Limitations & Design Notes

- **Minimum Query Length ($6$ Notes):** A melody of $N$ notes yields $3^{N-1}$ possible contours ($4$ notes = only $27$ contours). Typed queries shorter than $6$ notes display a false-positive warning, and audio queries strictly require at least $6$ detected notes.
- **Legato Singing on Repeated Pitches:** Singing smooth, connected vowels (*"la la la"*) on the same pitch without a brief silence gap may merge consecutive notes into one segment. Articulate notes clearly (*"ta ta ta"*) so `segment_notes` detects the unvoiced boundary.
- **Exact DFA Matching:** A single contour error prevents an exact state reach to $q_m$. The closest-attempt diagnostic view (for misses) and the editable audio contour box provide immediate workarounds.
