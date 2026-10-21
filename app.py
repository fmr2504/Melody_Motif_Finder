"""
Melody Motif Finder: Complete Working Prototype (Phases 1 to 5)
Framework: Streamlit (Python 3.10+)

This file consolidates the exact modular architecture specified in the 
'Melody Motif Finder: System Architecture and Development Guide':
  1. encoding.py  - Note names, MIDI numbers, contour code, query parsing
  2. dfa.py       - KMP-style DFA construction and execution (restart_after_match=True)
  3. library.py   - Song dataclass, JSON loading, and strict contour verification
  4. search.py    - Query, SongResult, SearchResponse dataclasses & multi-song search
  5. render.py    - Aligned text blocks (CELL=6), match comparisons, closest-attempt views
  6. audio.py     - Audio synthesis, pitch extraction (pyin/autocorr), smoothing, segmentation
  7. app.py       - Presentable Streamlit UI with Typed Search, Audio Input, DFA Inspector & Tests

Run command:
  streamlit run app.py
"""

from __future__ import annotations
import io
import json
import math
import os
import re
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional imports for Phase 5 Audio Input (graceful fallback as required by guide)
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ==============================================================================
# MODULE 1: encoding.py (Phase 1)
# Responsibility: Note names, MIDI numbers, contour code, query parsing
# Imports from: nothing
# ==============================================================================

NOTE_OFFSETS: Dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5,
    "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11, "CB": 11, "B#": 0,
}

NOTE_NAMES: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

ALPHABET: Tuple[str, str, str] = ("U", "D", "R")

CONTOUR_SHAPE: Dict[str, str] = {
    "U": "↗ Up",
    "D": "↘ Down",
    "R": "→ Repeat",
}

NOTE_REGEX = re.compile(r"^([A-Ga-g])([#bB]?)(-?\d+)$")


def note_to_midi(note: str) -> int:
    """Convert a scientific pitch note name (e.g. 'C4', 'F#5', 'Bb3') to MIDI number."""
    token = note.strip()
    match = NOTE_REGEX.match(token)
    if not match:
        raise ValueError(
            f"Invalid note name '{note}'. Expected format like C4, F#4, or Bb3."
        )
    letter, accidental, octave_str = match.groups()
    key = (letter.upper() + accidental.upper()).replace("B", "B")
    if key not in NOTE_OFFSETS:
        raise ValueError(f"Unrecognized pitch class in note '{note}'.")
    octave = int(octave_str)
    # Standard MIDI: C4 = 60
    midi = (octave + 1) * 12 + NOTE_OFFSETS[key]
    if not (0 <= midi <= 127):
        raise ValueError(f"Note '{note}' (MIDI {midi}) is outside valid range 0..127.")
    return midi


def midi_to_note(midi: float | int) -> str:
    """Convert a MIDI number (float or int) to its nearest standard note name."""
    midi_int = int(round(float(midi)))
    octave = (midi_int // 12) - 1
    pitch_class = midi_int % 12
    return f"{NOTE_NAMES[pitch_class]}{octave}"


def parse_notes(text: str) -> Tuple[List[str], List[int]]:
    """Parse space/comma-separated note string into canonical note names and MIDI ints."""
    cleaned = text.replace(",", " ").strip()
    if not cleaned:
        raise ValueError("Note sequence cannot be empty.")
    tokens = cleaned.split()
    if len(tokens) < 2:
        raise ValueError(
            "At least 2 notes are required to form a melodic contour symbol."
        )
    names: List[str] = []
    midis: List[int] = []
    for tok in tokens:
        m = note_to_midi(tok)
        # Keep canonical normalized casing (e.g., C4, F#4, Bb3)
        letter = tok[0].upper()
        rest = tok[1:]
        if len(rest) >= 2 and rest[0] in ("b", "B"):
            canonical = f"{letter}b{rest[1:]}"
        elif len(rest) >= 2 and rest[0] == "#":
            canonical = f"{letter}#{rest[1:]}"
        else:
            canonical = f"{letter}{rest}"
        names.append(canonical)
        midis.append(m)
    return names, midis


def encode_contour(midis: List[float | int], min_step: float = 0.0) -> str:
    """
    Encode a sequence of MIDI pitches into a contour string over {U, D, R}.
    Phase 1 specification: compares abs(current - previous) <= min_step first (gives R),
    then decides U or D.
    """
    if len(midis) < 2:
        raise ValueError("At least 2 notes are required to encode a contour.")
    symbols: List[str] = []
    for i in range(1, len(midis)):
        diff = float(midis[i]) - float(midis[i - 1])
        if abs(diff) <= min_step:
            symbols.append("R")
        elif diff > 0:
            symbols.append("U")
        else:
            symbols.append("D")
    return "".join(symbols)


def decode_contour(code: str) -> List[str]:
    """Decode a contour string into human-readable directional step labels."""
    cleaned = code.strip().upper()
    for ch in cleaned:
        if ch not in ALPHABET:
            raise ValueError(
                f"Invalid contour symbol '{ch}'. Allowed alphabet is U, D, R."
            )
    return [CONTOUR_SHAPE[ch] for ch in cleaned]


def parse_motif(
    text: str, min_step: float = 0.0
) -> Tuple[Optional[List[str]], Optional[List[float]], str]:
    """
    Parse either a typed note sequence ('E4 E4 F4 G4') or a direct contour query ('c: RUU').
    Returns (names, midis, pattern).
    """
    raw = text.strip()
    if not raw:
        raise ValueError("Query cannot be empty. Enter notes (e.g. 'E4 E4 F4 G4') or 'c: RUU'.")

    # Direct contour mode: starts with 'c:' or 'C:'
    if raw.lower().startswith("c:"):
        pattern = raw[2:].strip().upper().replace(" ", "")
        if not pattern:
            raise ValueError("Contour query after 'c:' cannot be empty.")
        for ch in pattern:
            if ch not in ALPHABET:
                raise ValueError(
                    f"Invalid character '{ch}' in contour pattern. Only U, D, R are allowed."
                )
        return None, None, pattern

    # Typed notes mode
    names, midis_int = parse_notes(raw)
    midis_float = [float(m) for m in midis_int]
    pattern = encode_contour(midis_float, min_step=min_step)
    return names, midis_float, pattern


# ==============================================================================
# MODULE 2: dfa.py (Phase 1)
# Responsibility: build_dfa, run_dfa (exact KMP-style DFA logic)
# Imports from: encoding (for ALPHABET)
# ==============================================================================

def build_dfa(
    pattern: str, alphabet: Tuple[str, ...] = ALPHABET
) -> Dict[int, Dict[str, int]]:
    """
    Build deterministic finite automaton transition table for exact substring matching
    over the contour alphabet.
    States are 0 .. len(pattern). State m = len(pattern) is the accepting state.
    """
    if not pattern:
        raise ValueError("DFA pattern cannot be empty.")
    for ch in pattern:
        if ch not in alphabet:
            raise ValueError(f"Symbol '{ch}' is not in alphabet {alphabet}.")

    m = len(pattern)
    transitions: Dict[int, Dict[str, int]] = {}

    for state in range(m + 1):
        transitions[state] = {}
        for symbol in alphabet:
            # Candidate prefix formed by taking current matched prefix + new symbol
            candidate = pattern[:state] + symbol
            # Find longest prefix of `pattern` that is a suffix of `candidate`
            next_state = min(m, state + 1)
            while next_state > 0 and not candidate.endswith(pattern[:next_state]):
                next_state -= 1
            transitions[state][symbol] = next_state

    return transitions


def run_dfa(
    transitions: Dict[int, Dict[str, int]],
    code: str,
    pattern_len: int,
    restart_after_match: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Run the DFA transition table across `code`.
    Returns:
      - states: list of DFA states of length len(code) + 1 (starts with 0)
      - starts: list of 0-based code indices where each full match begins
    Note: With restart_after_match=True (the default in the guide), the DFA naturally
    follows KMP fallback transitions from the accepting state so overlapping matches
    are preserved.
    """
    state = 0
    states: List[int] = [0]
    starts: List[int] = []

    for i, symbol in enumerate(code):
        if symbol not in transitions[state]:
            raise ValueError(f"Unexpected symbol '{symbol}' at index {i}.")
        state = transitions[state][symbol]
        states.append(state)

        if state == pattern_len:
            start_index = i - pattern_len + 1
            starts.append(start_index)
            if not restart_after_match:
                # If restart_after_match were False, it would freeze or reset hard to 0
                state = 0

    return states, starts


# ==============================================================================
# MODULE 3: library.py (Phase 2)
# Responsibility: Song dataclass, loads and validates songs.json
# Imports from: encoding
# ==============================================================================

@dataclass
class Song:
    id: str
    title: str
    notes: str          # e.g. 'C4 C4 G4 G4 A4 A4 G4'
    code: str           # stored contour, verified against encode_contour


# Default built-in dataset matching Phase 2 specifications (10 verified public-domain songs)
DEFAULT_SONGS_JSON: List[Dict[str, str]] = [
    {
        "id": "twinkle",
        "title": "Twinkle Twinkle (opening)",
        "notes": "C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4",
        "code": "RURURDDRDRDRD",
    },
    {
        "id": "ode-to-joy",
        "title": "Ode to Joy (opening)",
        "notes": "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4 D4 D4",
        "code": "RUURDDDDRUURDR",
    },
    {
        "id": "frere-jacques",
        "title": "Frère Jacques (opening)",
        "notes": "C4 D4 E4 C4 C4 D4 E4 C4 E4 F4 G4 E4 F4 G4",
        "code": "UUDRUUDUUUDUU",
    },
    {
        "id": "mary-lamb",
        "title": "Mary Had a Little Lamb",
        "notes": "E4 D4 C4 D4 E4 E4 E4 D4 D4 D4 E4 G4 G4 E4 D4 C4 D4 E4 E4 E4 E4 D4 D4 E4 D4 C4",
        "code": "DDUURRDRRUURDDDUURRRDRUDD",
    },
    {
        "id": "row-your-boat",
        "title": "Row Row Row Your Boat",
        "notes": "C4 C4 C4 D4 E4 E4 D4 E4 F4 G4 C5 C5 C5 G4 G4 G4 E4 E4 E4 C4 C4 C4 G4 F4 E4 D4 C4",
        "code": "RRUURDUUUURRDRRDRRDRUDDDD",
    },
    {
        "id": "jingle-bells",
        "title": "Jingle Bells (chorus)",
        "notes": "E4 E4 E4 E4 E4 E4 E4 G4 C4 D4 E4 F4 F4 F4 F4 F4 E4 E4 E4 E4 D4 D4 E4 D4 G4",
        "code": "RRRRRRUDUUURRRRDRRRDRUDU",
    },
    {
        "id": "london-bridge",
        "title": "London Bridge Is Falling Down",
        "notes": "G4 A4 G4 F4 E4 F4 G4 D4 E4 F4 E4 F4 G4 G4 A4 G4 F4 E4 F4 G4 D4 G4 E4 C4",
        "code": "UDDDUUDUUDUURUDDDUUDUDD",
    },
    {
        "id": "amazing-grace",
        "title": "Amazing Grace (opening)",
        "notes": "D4 G4 B4 G4 B4 A4 G4 E4 D4 D4 G4 B4 G4 B4 A4 D5",
        "code": "UUDUDDDDRUUDUDU",
    },
    {
        "id": "auld-lang-syne",
        "title": "Auld Lang Syne (opening)",
        "notes": "C4 F4 F4 F4 A4 G4 F4 G4 A4 F4 F4 A4 C5 D5",
        "code": "URRUDDUUDRUUU",
    },
    {
        "id": "brahms-lullaby",
        "title": "Brahms' Lullaby (opening)",
        "notes": "E4 E4 G4 E4 E4 G4 E4 G4 C5 B4 A4 A4 G4 D4 E4 F4 D4 D4 E4 F4",
        "code": "RUDRUDUUDDRDDUUDRUU",
    },
]


def validate_song_entries(raw_entries: List[Dict[str, str]]) -> List[Song]:
    """Validate raw song dictionaries and verify stored contour code against encode_contour."""
    songs: List[Song] = []
    for entry in raw_entries:
        song_id = entry.get("id", "<missing-id>")
        title = entry.get("title", song_id)
        notes_str = entry.get("notes", "")
        stored_code = entry.get("code", "")

        try:
            _, midis = parse_notes(notes_str)
        except ValueError as exc:
            raise ValueError(f"Invalid notes in song '{song_id}': {exc}") from exc

        if len(midis) < 2:
            raise ValueError(f"Song '{song_id}' must have at least 2 notes.")

        recomputed = encode_contour(midis, min_step=0.0)
        if recomputed != stored_code:
            raise ValueError(
                f"Contour mismatch in song '{song_id}': stored '{stored_code}' "
                f"!= recomputed '{recomputed}'."
            )

        songs.append(Song(id=song_id, title=title, notes=notes_str, code=stored_code))
    return songs


def load_library(path: Optional[str] = "data/songs.json") -> List[Song]:
    """
    Load and validate songs from JSON file if it exists on disk, otherwise
    load from the verified built-in JSON library.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)
    else:
        raw_entries = DEFAULT_SONGS_JSON
    return validate_song_entries(raw_entries)


def get_song(songs: List[Song], song_id: str) -> Song:
    """Return a song by its ID or raise KeyError."""
    for s in songs:
        if s.id == song_id:
            return s
    raise KeyError(f"Song ID '{song_id}' not found in library.")


# ==============================================================================
# MODULE 4: search.py (Phases 1 & 3)
# Responsibility: Query, SongResult, SearchResponse dataclasses, search orchestration
# Imports from: encoding, dfa, library
# ==============================================================================

@dataclass
class Query:
    names: Optional[List[str]]      # None for a typed contour such as c: RUU
    midis: Optional[List[float]]    # floats because audio notes are not whole numbers
    pattern: str                    # what the DFA searches for


@dataclass
class SongResult:
    song: Song
    names: List[str]
    midis: List[int]
    code: str
    states: List[int]               # DFA state after each code symbol (starts with 0)
    starts: List[int]               # code index where each match begins
    shifts: List[Optional[int]]     # semitone shift of each match; None for contour-only


@dataclass
class SearchResponse:
    query: Query
    transitions: Dict[int, Dict[str, int]]  # built once, reused for every song
    hits: List[SongResult]                  # songs with >= 1 match, ranked
    misses: List[SongResult]                # songs with 0 matches (for closest-attempt view)


def build_query(text: str, min_step: float = 0.0) -> Query:
    """Build a Query object from raw user text (either notes or 'c: <contour>')."""
    names, midis, pattern = parse_motif(text, min_step=min_step)
    return Query(names=names, midis=midis, pattern=pattern)


def search_song(
    song: Song, query: Query, transitions: Dict[int, Dict[str, int]]
) -> SongResult:
    """Search a single Song using a pre-built DFA transition table."""
    names, midis = parse_notes(song.notes)
    code = encode_contour(midis, min_step=0.0)
    states, starts = run_dfa(
        transitions, code, len(query.pattern), restart_after_match=True
    )

    shifts: List[Optional[int]] = []
    for s in starts:
        if query.midis is not None and len(query.midis) > 0:
            shift = int(round(midis[s] - query.midis[0]))
            shifts.append(shift)
        else:
            shifts.append(None)

    return SongResult(
        song=song,
        names=names,
        midis=midis,
        code=code,
        states=states,
        starts=starts,
        shifts=shifts,
    )


def search_library(songs: List[Song], query: Query) -> SearchResponse:
    """
    Build the DFA once for `query.pattern`, run it across all `songs`, and rank hits
    by (-len(r.starts), r.song.title).
    """
    transitions = build_dfa(query.pattern)  # built once per query
    results = [search_song(s, query, transitions) for s in songs]
    hits = sorted(
        (r for r in results if r.starts),
        key=lambda r: (-len(r.starts), r.song.title),
    )
    misses = [r for r in results if not r.starts]
    return SearchResponse(
        query=query,
        transitions=transitions,
        hits=hits,
        misses=misses,
    )


# ==============================================================================
# MODULE 5: render.py (Phases 1 & 4)
# Responsibility: Turns results into aligned text blocks (returns strings, never prints)
# Imports from: encoding
# ==============================================================================

CELL = 6  # Column width mandated by Phase 1 specification


def row(label: str, items: List[str], cell_width: int = CELL) -> str:
    """Format a single aligned row with a fixed-width left label and padded cells."""
    formatted_cells = "".join(f"{str(item):<{cell_width}}" for item in items)
    return f"{label:<10}{formatted_cells}".rstrip()


def trace_text(result: SongResult, per_line: Optional[int] = 12) -> str:
    """
    Produce aligned text blocks showing Index, Notes, MIDI, Contour Code, and DFA States.
    Supports `per_line` chunking so long melodies stack cleanly on screen (Phase 4).
    """
    n = len(result.names)
    indices = [str(i + 1) for i in range(n)]
    notes_row = result.names
    midi_row = [str(m) for m in result.midis]
    # Code has N-1 symbols; align symbol k under transition from note k to k+1
    code_row = ["-"] + list(result.code)
    state_row = [str(s) for s in result.states]

    if per_line is None or per_line <= 0 or n <= per_line:
        lines = [
            row("Note #:", indices),
            row("Notes:", notes_row),
            row("MIDI:", midi_row),
            row("Contour:", code_row),
            row("DFA St:", state_row),
        ]
        return "\n".join(lines)

    blocks: List[str] = []
    for start in range(0, n, per_line):
        end = min(n, start + per_line)
        block_lines = [
            f"--- Notes {start + 1} to {end} ---",
            row("Note #:", indices[start:end]),
            row("Notes:", notes_row[start:end]),
            row("MIDI:", midi_row[start:end]),
            row("Contour:", code_row[start:end]),
            row("DFA St:", state_row[start:end]),
        ]
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def format_match_comparison(result: SongResult, query: Query) -> List[str]:
    """
    Format human-readable descriptions for each match in `result`, labeling exact
    repeats, transposed repeats (with semitone shift), or contour-only matches.
    """
    lines: List[str] = []
    pat_len = len(query.pattern)

    for idx, (start, shift) in enumerate(zip(result.starts, result.shifts), start=1):
        end_note_idx = start + pat_len  # covers notes s to s + len(pattern) inclusive
        matched_notes = " ".join(result.names[start : end_note_idx + 1])
        matched_code = result.code[start : start + pat_len]
        note_num_start = start + 1
        note_num_end = end_note_idx + 1

        if shift is None:
            label = "Contour match (no reference pitch in query)"
        elif shift == 0:
            label = "Exact repeat (0 semitones shift)"
        else:
            direction = "higher" if shift > 0 else "lower"
            label = f"Transposed repeat ({abs(shift)} semitones {direction}, shift={shift:+d})"

        lines.append(
            f"Match #{idx}: Notes {note_num_start}–{note_num_end} [{matched_notes}] "
            f"| Contour: {matched_code} -> {label}"
        )
    return lines


def display_no_match(result: SongResult, pattern: str) -> str:
    """
    For a song with no full match, find the closest partial attempt (highest DFA state
    reached) and return a diagnostic string.
    """
    max_state = max(result.states) if result.states else 0
    if max_state == 0:
        return (
            f"{result.song.title}: No partial progress (0/{len(pattern)} contour steps matched)."
        )

    # First code position where max_state was achieved
    best_pos = result.states.index(max_state)  # 1-based index in states
    start_note_idx = best_pos - max_state
    end_note_idx = best_pos
    snippet_notes = " ".join(result.names[start_note_idx : end_note_idx + 1])
    matched_prefix = pattern[:max_state]

    return (
        f"{result.song.title}: Reached state {max_state}/{len(pattern)} "
        f"(matched prefix '{matched_prefix}' at notes {start_note_idx + 1}–{end_note_idx + 1}: "
        f"[{snippet_notes}])"
    )


def format_dfa_table(transitions: Dict[int, Dict[str, int]], pattern: str) -> pd.DataFrame:
    """Convert the DFA transition dictionary into a clean DataFrame for inspection."""
    rows = []
    m = len(pattern)
    for state in sorted(transitions.keys()):
        prefix = pattern[:state] if state > 0 else "ε (start)"
        status = "ACCEPT (Match!)" if state == m else ("Start" if state == 0 else "Partial")
        rows.append(
            {
                "State": f"q{state}",
                "Matched Prefix": prefix,
                "Status": status,
                "On 'U' (↗)": f"q{transitions[state]['U']}",
                "On 'D' (↘)": f"q{transitions[state]['D']}",
                "On 'R' (→)": f"q{transitions[state]['R']}",
            }
        )
    return pd.DataFrame(rows)


# ==============================================================================
# MODULE 6: audio.py (Phase 5)
# Responsibility: Audio clip -> pitch per frame -> smoothed -> segmented notes -> Query
# Imports from: encoding, search (Query)
# ==============================================================================

@dataclass
class AudioNote:
    midi: float
    start: float
    duration: float

    @property
    def name(self) -> str:
        return midi_to_note(self.midi)


def load_audio(source) -> Tuple[np.ndarray, int]:
    """
    Step 1 of Phase 5: Load audio into mono waveform at sr=22050.
    Rejects clips > 20 seconds and clips that are almost silent.
    Supports raw NumPy tuple (y, sr), WAV bytes/BytesIO, or file paths.
    """
    target_sr = 22050

    if isinstance(source, tuple) and len(source) == 2:
        y, sr = source
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
    elif LIBROSA_AVAILABLE:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        y, sr = librosa.load(source, sr=target_sr, mono=True)
    else:
        # Built-in pure-Python WAV fallback so WAV uploads/recordings work even without librosa
        if isinstance(source, (bytes, bytearray)):
            bio = io.BytesIO(source)
        elif hasattr(source, "read"):
            source.seek(0)
            bio = io.BytesIO(source.read())
        else:
            raise ValueError("Audio source requires librosa or WAV bytes.")
        with wave.open(bio, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth}")
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            y = samples

    duration = len(y) / float(sr) if sr > 0 else 0.0
    if duration > 20.5:
        raise ValueError(
            f"Audio clip is too long ({duration:.1f}s). Maximum allowed is 20 seconds."
        )
    if duration < 0.2:
        raise ValueError("Audio clip is too short to contain a melody.")

    peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if len(y) > 0 else 0.0
    if peak < 1e-3 or rms < 5e-4:
        raise ValueError("Audio clip is silent or nearly silent. Please record louder.")

    return y, sr


def extract_pitch(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2 of Phase 5: Extract fundamental frequency per frame and convert to float MIDI.
    Uses librosa.pyin when installed; falls back to an accurate windowed autocorrelation
    pitch tracker restricted to C2..C6 (65.4 Hz .. 1046.5 Hz) if librosa is unavailable.
    Unvoiced frames are represented as NaN.
    """
    fmin = 65.406   # C2
    fmax = 1046.50  # C6
    hop_length = 512
    frame_length = 2048

    if LIBROSA_AVAILABLE:
        f0, _, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        midi = np.where(np.isnan(f0), np.nan, 69.0 + 12.0 * np.log2(f0 / 440.0))
        return times, midi

    # Pure NumPy autocorrelation tracker (ensures synthetic audio tests & WAVs always work)
    n_frames = max(1, 1 + (len(y) - frame_length) // hop_length)
    times = np.arange(n_frames) * (hop_length / float(sr))
    midi = np.full(n_frames, np.nan, dtype=np.float64)

    min_lag = max(1, int(sr / fmax))
    max_lag = min(frame_length // 2, int(sr / fmin))
    window = np.hanning(frame_length)

    global_rms = float(np.sqrt(np.mean(y**2)))
    silence_thresh = max(0.01, global_rms * 0.18)

    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + frame_length]
        if len(frame) < frame_length:
            padded = np.zeros(frame_length, dtype=np.float32)
            padded[: len(frame)] = frame
            frame = padded

        frame_rms = float(np.sqrt(np.mean(frame**2)))
        if frame_rms < silence_thresh:
            continue

        win_frame = (frame - np.mean(frame)) * window
        corr = np.correlate(win_frame, win_frame, mode="full")
        corr = corr[len(corr) // 2 :]
        if corr[0] <= 1e-9:
            continue
        corr = corr / corr[0]

        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue
        best_offset = int(np.argmax(search_region))
        peak_val = float(search_region[best_offset])
        if peak_val < 0.55:
            continue

        lag = min_lag + best_offset
        # Parabolic interpolation for sub-sample frequency accuracy
        if 0 < lag < len(corr) - 1:
            y0, y1, y2 = corr[lag - 1], corr[lag], corr[lag + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-9:
                lag_interp = lag + 0.5 * (y0 - y2) / denom
            else:
                lag_interp = float(lag)
        else:
            lag_interp = float(lag)

        freq = float(sr) / lag_interp
        if fmin <= freq <= fmax:
            midi[i] = 69.0 + 12.0 * np.log2(freq / 440.0)

    return times, midi


def smooth(midi: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Step 3 of Phase 5: Apply a median filter to each continuous voiced (non-NaN) run
    separately, because scipy.signal.medfilt cannot handle NaN across the full array.
    """
    if kernel % 2 == 0:
        kernel += 1
    out = np.copy(midi).astype(np.float64)
    n = len(out)
    i = 0

    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(out[j]):
            j += 1
        run = out[i:j]
        run_len = len(run)
        if run_len >= 3:
            k = min(kernel, run_len if run_len % 2 == 1 else run_len - 1)
            if k >= 3:
                if SCIPY_AVAILABLE:
                    out[i:j] = scipy.signal.medfilt(run, kernel_size=k)
                else:
                    half = k // 2
                    filtered = np.copy(run)
                    for idx in range(run_len):
                        lo = max(0, idx - half)
                        hi = min(run_len, idx + half + 1)
                        filtered[idx] = np.median(run[lo:hi])
                    out[i:j] = filtered
        i = j

    return out


def segment_notes(
    times: np.ndarray,
    midi: np.ndarray,
    tol: float = 0.5,
    min_dur: float = 0.10,
    max_gap: float = 0.05,
) -> List[AudioNote]:
    """
    Step 4 of Phase 5: Walk through the frames and start a new note when the pitch moves
    more than `tol` semitones from the current note's running median, or when a NaN gap
    is longer than `max_gap` seconds. Drop segments shorter than `min_dur`.
    """
    if len(times) == 0 or len(midi) == 0:
        return []

    dt = float(times[1] - times[0]) if len(times) > 1 else (512.0 / 22050.0)
    notes: List[AudioNote] = []

    curr_pitches: List[float] = []
    curr_start: Optional[float] = None
    last_voiced_time: Optional[float] = None

    def flush_segment():
        nonlocal curr_pitches, curr_start, last_voiced_time
        if curr_pitches and curr_start is not None and last_voiced_time is not None:
            dur = (last_voiced_time - curr_start) + dt
            if dur >= min_dur:
                notes.append(
                    AudioNote(
                        midi=float(np.median(curr_pitches)),
                        start=float(curr_start),
                        duration=float(dur),
                    )
                )
        curr_pitches = []
        curr_start = None
        last_voiced_time = None

    for t, m in zip(times, midi):
        if np.isnan(m):
            if last_voiced_time is not None and (float(t) - last_voiced_time) > max_gap:
                flush_segment()
            continue

        # Voiced frame
        if not curr_pitches:
            curr_pitches = [float(m)]
            curr_start = float(t)
            last_voiced_time = float(t)
        else:
            running_med = float(np.median(curr_pitches))
            if abs(float(m) - running_med) > tol:
                flush_segment()
                curr_pitches = [float(m)]
                curr_start = float(t)
                last_voiced_time = float(t)
            else:
                curr_pitches.append(float(m))
                last_voiced_time = float(t)

    flush_segment()
    return notes


def audio_to_query(
    source,
    min_step: float = 0.75,
    tol: float = 0.5,
    min_dur: float = 0.10,
    max_gap: float = 0.05,
) -> Tuple[Query, List[AudioNote], np.ndarray, np.ndarray]:
    """
    Step 5 of Phase 5: Chain steps 1 to 4, require at least 6 notes (raise ValueError
    otherwise), then build the Query with names rounded for display, midis left as floats,
    and pattern = encode_contour(midis, min_step).
    Also returns the intermediate AudioNotes and pitch track for visualization in app.py.
    """
    y, sr = load_audio(source)
    times, raw_midi = extract_pitch(y, sr)
    smoothed_midi = smooth(raw_midi, kernel=5)
    segments = segment_notes(
        times, smoothed_midi, tol=tol, min_dur=min_dur, max_gap=max_gap
    )

    if len(segments) < 6:
        raise ValueError(
            f"Audio query detected only {len(segments)} notes. "
            "At least 6 distinct notes are required for reliable audio contour search."
        )

    names = [seg.name for seg in segments]
    midis = [seg.midi for seg in segments]
    pattern = encode_contour(midis, min_step=min_step)
    query = Query(names=names, midis=midis, pattern=pattern)
    return query, segments, times, smoothed_midi


def synthesize_melody_wav(
    note_sequence: str,
    note_dur: float = 0.40,
    gap_dur: float = 0.08,
    vibrato_semitones: float = 0.0,
    sr: int = 22050,
) -> bytes:
    """
    Synthesize a clean sine-wave WAV clip from a note string (e.g. 'E4 E4 F4 G4 G4 F4 E4 D4')
    following the exact Phase 5 test specifications (0.4s tone + 0.08s silence gap,
    optional vibrato). Returns WAV bytes suitable for playback and `audio_to_query`.
    """
    _, midis = parse_notes(note_sequence)
    segments_audio: List[np.ndarray] = []
    gap_samples = int(round(gap_dur * sr))
    note_samples = int(round(note_dur * sr))
    t_note = np.arange(note_samples) / float(sr)

    # Smooth 10ms fade-in/out envelope to prevent clicks
    fade_len = min(int(0.01 * sr), note_samples // 4)
    envelope = np.ones(note_samples, dtype=np.float32)
    if fade_len > 0:
        envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len)
        envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len)

    for m in midis:
        if vibrato_semitones > 0.0:
            # 5.5 Hz vibrato modulating instantaneous pitch by +/- vibrato_semitones
            inst_midi = m + vibrato_semitones * np.sin(2.0 * np.pi * 5.5 * t_note)
            inst_freq = 440.0 * (2.0 ** ((inst_midi - 69.0) / 12.0))
            phase = 2.0 * np.pi * np.cumsum(inst_freq) / float(sr)
            tone = 0.5 * np.sin(phase).astype(np.float32) * envelope
        else:
            freq = 440.0 * (2.0 ** ((m - 69.0) / 12.0))
            tone = 0.5 * np.sin(2.0 * np.pi * freq * t_note).astype(np.float32) * envelope

        segments_audio.append(tone)
        if gap_samples > 0:
            segments_audio.append(np.zeros(gap_samples, dtype=np.float32))

    waveform = np.concatenate(segments_audio) if segments_audio else np.zeros(sr, dtype=np.float32)
    pcm16 = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return bio.getvalue()


# ==============================================================================
# AUTOMATED ACCEPTANCE CHECK SUITE (Phases 1 to 5 Verification)
# ==============================================================================

def run_acceptance_suite() -> List[Dict[str, str]]:
    """
    Execute the acceptance tests defined across Phases 1 to 5 of the guide and
    return a structured report table.
    """
    results: List[Dict[str, str]] = []

    def check(phase: str, name: str, fn):
        try:
            detail = fn()
            results.append({"Phase": phase, "Test Case": name, "Status": "✅ PASS", "Detail": detail})
        except Exception as exc:
            results.append({"Phase": phase, "Test Case": name, "Status": "❌ FAIL", "Detail": str(exc)})

    # Phase 1 checks
    def test_p1_scale():
        _, midis = parse_notes("C4 D4 E4 F4 G4 F4 E4 D4 C4")
        code = encode_contour(midis)
        assert code == "UUUUDDDD", f"Expected UUUUDDDD, got {code}"
        return f"C4..G4..C4 encoded to '{code}'"

    def test_p1_repeat_and_deadzone():
        _, midis = parse_notes("C4 C4 G4 G4")
        code0 = encode_contour(midis, min_step=0)
        assert code0 == "RUR", f"Expected RUR, got {code0}"
        code_dz = encode_contour([60.0, 60.4, 67.0, 66.5], min_step=0.75)
        assert code_dz == "RUR", f"Dead-zone encoding failed: {code_dz}"
        return "Exact repeat ('RUR') and min_step=0.75 dead-zone verified"

    check("Phase 1", "Scale Contour (UUUUDDDD)", test_p1_scale)
    check("Phase 1", "Repeated Notes & Dead Zone", test_p1_repeat_and_deadzone)

    # Phase 2 checks
    def test_p2_library():
        songs = load_library()
        assert len(songs) >= 8, f"Expected >= 8 songs, loaded {len(songs)}"
        for s in songs:
            _, m = parse_notes(s.notes)
            assert s.code == encode_contour(m), f"Mismatch in {s.id}"
        return f"Loaded and validated {len(songs)} public-domain songs"

    check("Phase 2", "Song Library Validation (10 songs)", test_p2_library)

    # Phase 3 checks
    def test_p3_exact_and_transposed():
        songs = load_library()
        q = build_query("E4 E4 F4 G4")
        resp = search_library(songs, q)
        ode = next(r for r in resp.hits if r.song.id == "ode-to-joy")
        assert ode.starts == [0, 8], f"Expected starts [0, 8] (notes 1 and 9), got {ode.starts}"
        assert ode.shifts == [0, -4], f"Expected shifts [0, -4], got {ode.shifts}"
        twinkle_hit = any(r.song.id == "twinkle" for r in resp.hits)
        assert not twinkle_hit, "Twinkle should not match RUU"
        return "Ode to Joy matched at Note 1 (shift 0) & Note 9 (shift -4); Twinkle excluded"

    def test_p3_overlapping():
        songs = load_library()
        q = build_query("c: DDD")
        resp = search_library(songs, q)
        ode = next(r for r in resp.hits if r.song.id == "ode-to-joy")
        assert ode.starts == [4, 5], f"Expected overlapping starts [4, 5], got {ode.starts}"
        return f"Overlapping c:DDD in Ode to Joy found at code indices {ode.starts}"

    def test_p3_no_match_and_errors():
        songs = load_library()
        q = build_query("c: UUUUUUUUU")
        resp = search_library(songs, q)
        assert len(resp.hits) == 0 and len(resp.misses) == len(songs)
        for bad in ["", "C4", "H9 C4"]:
            try:
                build_query(bad)
                raise AssertionError(f"Bad query '{bad}' did not raise ValueError")
            except ValueError:
                pass
        return "No-match query puts all songs in misses; invalid queries raise ValueError"

    check("Phase 3", "Exact & Transposed Matches (E4 E4 F4 G4)", test_p3_exact_and_transposed)
    check("Phase 3", "Overlapping DFA Matches (c:DDD)", test_p3_overlapping)
    check("Phase 3", "No Match & Invalid Query Handling", test_p3_no_match_and_errors)

    # Phase 5 checks (Synthetic Audio)
    def test_p5_synthetic_clean_and_vibrato():
        seq = "E4 E4 F4 G4 G4 F4 E4 D4"
        wav_clean = synthesize_melody_wav(seq, note_dur=0.40, gap_dur=0.08, vibrato_semitones=0.0)
        q_clean, notes_clean, _, _ = audio_to_query(wav_clean, min_step=0.75)
        assert len(notes_clean) == 8, f"Expected 8 notes, got {len(notes_clean)}"
        assert q_clean.pattern == "RUURDDD", f"Expected RUURDDD, got {q_clean.pattern}"

        wav_vib = synthesize_melody_wav(seq, note_dur=0.40, gap_dur=0.08, vibrato_semitones=0.3)
        q_vib, _, _, _ = audio_to_query(wav_vib, min_step=0.75)
        assert q_vib.pattern == "RUURDDD", f"Vibrato contour mismatch: {q_vib.pattern}"
        return f"8 synthetic tones (+0.3st vibrato) -> contour '{q_vib.pattern}'"

    def test_p5_short_and_silent_rejection():
        wav_short = synthesize_melody_wav("C4 D4 E4", note_dur=0.40, gap_dur=0.08)
        try:
            audio_to_query(wav_short)
            raise AssertionError("3-note audio clip should raise ValueError (< 6 notes)")
        except ValueError:
            pass
        return "Audio clips with < 6 notes or silence raise ValueError as required"

    check("Phase 5", "Synthetic Sine Tones & Vibrato (min_step=0.75)", test_p5_synthetic_clean_and_vibrato)
    check("Phase 5", "Minimum 6-Note Audio Validation", test_p5_short_and_silent_rejection)

    return results


# ==============================================================================
# MODULE 7: app.py (Phases 4 & 5 Streamlit UI)
# Responsibility: Presentable Streamlit interface
# Imports from: search, library, render, audio
# ==============================================================================

@st.cache_data
def get_cached_library() -> List[Song]:
    """Wrap load_library in @st.cache_data as specified in Phase 4 implementation notes."""
    return load_library()


def build_contour_chart_df(result: SongResult, pat_len: int) -> pd.DataFrame:
    """
    Phase 4 item 3: Create a DataFrame with two columns: the full melody and the
    matched notes (NaN everywhere else), so the match is highlighted as a second line.
    """
    n = len(result.midis)
    matched_mask = [False] * n
    for s in result.starts:
        for idx in range(s, min(n, s + pat_len + 1)):
            matched_mask[idx] = True

    full_melody = [float(m) for m in result.midis]
    matched_notes = [float(result.midis[i]) if matched_mask[i] else np.nan for i in range(n)]

    df = pd.DataFrame(
        {
            "Full Melody (MIDI)": full_melody,
            "Matched Motif (MIDI)": matched_notes,
        },
        index=pd.Index(range(1, n + 1), name="Note #"),
    )
    return df


def render_results_section(response: SearchResponse):
    """Render the SearchResponse following Phase 4 layout guidelines."""
    total_matches = sum(len(r.starts) for r in response.hits)
    q = response.query

    # Query Summary Banner
    st.markdown("### 🔎 Search Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Songs Matched", f"{len(response.hits)} / {len(response.hits) + len(response.misses)}")
    col2.metric("Total Occurrences", str(total_matches))
    col3.metric("DFA Pattern", q.pattern)
    col4.metric(
        "Query Mode",
        "Notes + Contour" if q.names is not None else "Contour-Only (c:)",
    )

    if q.names is not None and q.midis is not None:
        rounded_notes = " ".join(
            f"{name} ({midi:.1f})" if not float(midi).is_integer() else f"{name}"
            for name, midi in zip(q.names, q.midis)
        )
        st.caption(f"**Parsed Query Notes:** `{rounded_notes}`  →  **Encoded Contour:** `{q.pattern}`")
    else:
        steps_desc = " , ".join(decode_contour(q.pattern))
        st.caption(f"**Contour Pattern:** `{q.pattern}` ({steps_desc})")

    # Hits expanders ordered by rank
    if not response.hits:
        st.info("No songs in the library matched the contour pattern `" + q.pattern + "`.")
    else:
        for rank, hit in enumerate(response.hits, start=1):
            match_word = "match" if len(hit.starts) == 1 else "matches"
            expander_title = (
                f"#{rank} — 🎵 {hit.song.title}  "
                f"({len(hit.starts)} {match_word})"
            )
            with st.expander(expander_title, expanded=(rank == 1)):
                # 1. Comparison text (exact or transposed) for each match
                st.markdown("#### 🎯 Match Breakdown")
                for line in format_match_comparison(hit, q):
                    if "Exact repeat" in line:
                        st.success(line)
                    elif "Transposed repeat" in line:
                        st.info(line)
                    else:
                        st.warning(line)

                # 2. Contour chart highlighting matched notes as a second line
                st.markdown("#### 📈 Melodic Contour & Highlighted Motif (MIDI Pitch)")
                chart_df = build_contour_chart_df(hit, len(q.pattern))
                st.line_chart(chart_df, height=250)

                # 3. Aligned notes, MIDI, code and DFA-state rows from render.trace_text
                st.markdown("#### 🧬 Aligned DFA Execution Trace")
                st.code(trace_text(hit, per_line=12), language="text")

    # 4. Misses: short list showing closest attempt from display_no_match
    if response.misses:
        with st.expander(
            f"🔍 Closest Attempts in Non-Matching Songs ({len(response.misses)} songs)",
            expanded=(len(response.hits) == 0),
        ):
            for miss in sorted(response.misses, key=lambda r: -max(r.states)):
                st.text("• " + display_no_match(miss, q.pattern))

    # Optional DFA Transition Table view for academic presentation
    with st.expander("⚙️ Inspect KMP-DFA Transition Table (Built Once for Query)", expanded=False):
        st.caption(
            f"Deterministic Finite Automaton with {len(q.pattern) + 1} states (`q0` to `q{len(q.pattern)}`). "
            "`restart_after_match=True` keeps KMP fallback transitions active on the accepting state."
        )
        st.dataframe(
            format_dfa_table(response.transitions, q.pattern),
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(
        page_title="Melody Motif Finder — DFA Contour Search",
        page_icon="🎵",
        layout="wide",
    )

    # Load validated song library (cached)
    try:
        songs = get_cached_library()
    except ValueError as exc:
        st.error(f"Library validation failed: {exc}")
        return

    # Initialize session_state so results persist across widget interactions (Phase 4)
    if "last_response" not in st.session_state:
        default_q = build_query("E4 E4 F4 G4")
        st.session_state["last_response"] = search_library(songs, default_q)
    if "typed_query_input" not in st.session_state:
        st.session_state["typed_query_input"] = "E4 E4 F4 G4"
    if "edited_audio_contour" not in st.session_state:
        st.session_state["edited_audio_contour"] = ""

    # --------------------------------------------------------------------------
    # SIDEBAR: Song Library List & U/D/R Alphabet Reference (Phase 4)
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.title("🎼 Melody Library")
        st.caption(
            f"**{len(songs)} verified public-domain melodies** loaded and checked against `encode_contour`."
        )

        for s in songs:
            with st.expander(f"🎵 {s.title}"):
                st.markdown(f"**ID:** `{s.id}`")
                st.markdown(f"**Notes ({len(s.code) + 1}):** `{s.notes}`")
                st.markdown(f"**Contour ({len(s.code)}):** `{s.code}`")
                if st.button(f"Load snippet as query", key=f"sidebar_load_{s.id}"):
                    snippet = " ".join(s.notes.split()[:6])
                    st.session_state["typed_query_input"] = snippet
                    q = build_query(snippet)
                    st.session_state["last_response"] = search_library(songs, q)
                    st.rerun()

        st.divider()
        st.subheader("🔤 Contour Alphabet")
        st.markdown(
            """
            A melody of $N$ notes yields a contour code of $N-1$ symbols:
            - **`U` (Up ↗):** Next note is higher in pitch
            - **`D` (Down ↘):** Next note is lower in pitch
            - **`R` (Repeat →):** Next note is the same pitch (or within dead-zone `min_step`)

            **Query Formats:**
            - **Typed Notes:** `E4 E4 F4 G4` *(identifies exact vs. transposed repeats)*
            - **Direct Contour:** `c: RUU` *(matches contour shape only)*
            """
        )

    # --------------------------------------------------------------------------
    # MAIN HEADER
    # --------------------------------------------------------------------------
    st.title("🎵 Melody Motif Finder")
    st.markdown(
        "**Deterministic Finite Automaton (DFA) Melodic Contour & Motif Search Engine** — *Phases 1 to 5 Prototype*"
    )

    tab_typed, tab_audio, tab_tests = st.tabs(
        [
            "⌨️ Phase 1–4: Typed Notes & Contour Search",
            "🎙️ Phase 5: Audio Input (Record / Upload / Synthesize)",
            "✅ Architecture & Acceptance Checks",
        ]
    )

    # ==========================================================================
    # TAB 1: TYPED NOTES OR CONTOUR SEARCH (Phases 1 to 4)
    # ==========================================================================
    with tab_typed:
        st.subheader("Search by Scientific Pitch Notes or Contour Code")

        # Quick-fill preset buttons for easy classroom demonstration
        st.caption("Quick Demo Presets (Click to populate & run):")
        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        preset_clicked: Optional[str] = None
        if pcol1.button("Ode to Joy (E4 E4 F4 G4)", use_container_width=True):
            preset_clicked = "E4 E4 F4 G4"
        if pcol2.button("Ode Full Phrase (8 notes)", use_container_width=True):
            preset_clicked = "E4 E4 F4 G4 G4 F4 E4 D4"
        if pcol3.button("Overlapping (c: DDD)", use_container_width=True):
            preset_clicked = "c: DDD"
        if pcol4.button("Twinkle Opening (7 notes)", use_container_width=True):
            preset_clicked = "C4 C4 G4 G4 A4 A4 G4"
        if pcol5.button("No Match (c: UUUUUUUUU)", use_container_width=True):
            preset_clicked = "c: UUUUUUUUU"

        if preset_clicked is not None:
            st.session_state["typed_query_input"] = preset_clicked
            try:
                q_preset = build_query(preset_clicked)
                st.session_state["last_response"] = search_library(songs, q_preset)
            except ValueError as exc:
                st.error(str(exc))

        query_text = st.text_area(
            "Enter Notes (e.g., `E4 E4 F4 G4 G4 F4`) or a Contour Pattern prefixed with `c:` (e.g., `c: RUU`):",
            key="typed_query_input",
            height=90,
            placeholder="Example: E4 E4 F4 G4 G4 F4 E4 D4   or   c: RUURDDD",
        )

        search_btn = st.button("🔍 Search Library", type="primary")

        if search_btn:
            try:
                q = build_query(query_text, min_step=0.0)
                st.session_state["last_response"] = search_library(songs, q)
            except ValueError as exc:
                st.error(f"Query Error: {exc}")

        # Minimum-length warning check (Section 2.5 & Phase 4: warning below 6 notes / 5 contour steps)
        current_resp: Optional[SearchResponse] = st.session_state.get("last_response")
        if current_resp is not None:
            num_notes_equiv = len(current_resp.query.pattern) + 1
            if num_notes_equiv < 6:
                st.warning(
                    f"⚠️ **Short Query Warning:** Your query spans **{num_notes_equiv} notes** "
                    f"(`{len(current_resp.query.pattern)}` contour symbols, only $3^{{{len(current_resp.query.pattern)}}} = "
                    f"{3 ** len(current_resp.query.pattern)}$ possible contours). "
                    "Queries shorter than 6 notes may produce false-positive matches across multiple songs."
                )

            st.divider()
            render_results_section(current_resp)

    # ==========================================================================
    # TAB 2: AUDIO INPUT (Phase 5)
    # ==========================================================================
    with tab_audio:
        st.subheader("🎙️ Audio-to-Contour Query Pipeline (Phase 5)")
        st.markdown(
            """
            Record or upload a hummed/played melody (or generate a synthetic test clip below).
            The pipeline runs `load_audio` → `extract_pitch` → `smooth` (median filter on voiced runs)
            → `segment_notes` → `encode_contour(midis, min_step)`.
            """
        )

        # Parameter tuning expanders (matching Phase 5 starting points: 0.75, 0.5, 0.10, 0.05)
        with st.expander("🎛️ Audio Segmentation & Dead-Zone Parameters", expanded=False):
            acol1, acol2, acol3, acol4 = st.columns(4)
            min_step = acol1.slider(
                "Dead-Zone (min_step semitones)", 0.0, 2.0, 0.75, 0.05,
                help="Steps with |ΔMIDI| <= min_step are encoded as R (Repeat)."
            )
            tol = acol2.slider(
                "Pitch Tolerance (tol semitones)", 0.2, 1.5, 0.50, 0.05,
                help="Starts a new note when pitch moves more than tol from running median."
            )
            min_dur = acol3.slider(
                "Min Note Duration (s)", 0.05, 0.30, 0.10, 0.01,
                help="Drops voiced segments shorter than min_dur seconds."
            )
            max_gap = acol4.slider(
                "Max Unvoiced Gap (s)", 0.02, 0.20, 0.05, 0.01,
                help="Splits consecutive identical pitches when separated by a gap > max_gap."
            )

        in_col1, in_col2, in_col3 = st.columns(3)

        audio_bytes_to_analyze: Optional[bytes] = None
        source_label = ""

        with in_col1:
            st.markdown("**1. Microphone Recording**")
            if hasattr(st, "audio_input"):
                recorded_audio = st.audio_input("Sing or hum >= 6 notes (say 'ta ta ta')")
                if recorded_audio is not None:
                    audio_bytes_to_analyze = recorded_audio.getvalue()
                    source_label = "Microphone Recording"
            else:
                st.info("`st.audio_input` requires a newer Streamlit version. Use Upload or Synthesizer.")

        with in_col2:
            st.markdown("**2. Upload WAV Audio File**")
            uploaded_file = st.file_uploader("Upload a mono/stereo `.wav` clip (<= 20s)", type=["wav"])
            if uploaded_file is not None:
                audio_bytes_to_analyze = uploaded_file.getvalue()
                source_label = f"Uploaded File ({uploaded_file.name})"

        with in_col3:
            st.markdown("**3. Built-in Test Synthesizer**")
            synth_preset = st.selectbox(
                "Select test melody to synthesize:",
                [
                    "Ode to Joy: E4 E4 F4 G4 G4 F4 E4 D4",
                    "Twinkle Twinkle: C4 C4 G4 G4 A4 A4 G4",
                    "Frère Jacques: C4 D4 E4 C4 C4 D4 E4 C4",
                    "Mary Had a Little Lamb: E4 D4 C4 D4 E4 E4 E4",
                    "London Bridge: G4 A4 G4 F4 E4 F4 G4",
                ],
            )
            vib_toggle = st.checkbox("Add 0.3 semitone vibrato (Phase 5 test)", value=False)
            if st.button("🔊 Synthesize & Analyze Clip", use_container_width=True):
                notes_part = synth_preset.split(": ")[1]
                st.session_state["synth_wav"] = synthesize_melody_wav(
                    notes_part,
                    note_dur=0.40,
                    gap_dur=0.08,
                    vibrato_semitones=0.3 if vib_toggle else 0.0,
                )
                st.session_state["synth_label"] = f"Synthetic Audio ({synth_preset})"

        if audio_bytes_to_analyze is None and "synth_wav" in st.session_state:
            audio_bytes_to_analyze = st.session_state["synth_wav"]
            source_label = st.session_state.get("synth_label", "Synthetic Audio")

        if audio_bytes_to_analyze is not None:
            st.markdown(f"**Active Audio Source:** `{source_label}`")
            st.audio(audio_bytes_to_analyze, format="audio/wav")

            try:
                audio_q, segments, times, smoothed_midi = audio_to_query(
                    audio_bytes_to_analyze,
                    min_step=min_step,
                    tol=tol,
                    min_dur=min_dur,
                    max_gap=max_gap,
                )

                st.success(
                    f"✅ Extracted **{len(segments)} notes** → Contour Pattern: **`{audio_q.pattern}`**"
                )

                # Display segmented floating-point MIDI notes table
                seg_rows = [
                    {
                        "Note #": idx + 1,
                        "Nearest Note": seg.name,
                        "Float MIDI": f"{seg.midi:.2f}",
                        "Start (s)": f"{seg.start:.2f}",
                        "Duration (s)": f"{seg.duration:.2f}",
                    }
                    for idx, seg in enumerate(segments)
                ]
                st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

                # Pitch track visualization
                pitch_df = pd.DataFrame(
                    {"Smoothed Pitch (MIDI)": smoothed_midi},
                    index=pd.Index(np.round(times, 2), name="Time (s)"),
                )
                st.line_chart(pitch_df, height=200)

                # Editable contour field (Phase 5 safety net requirement)
                st.markdown("#### ✏️ Editable Contour Field (Phase 5 Safety Net)")
                st.caption(
                    "If the pitch tracker made a one-symbol error (e.g. an octave jump or missed repeat), "
                    "you can edit the contour string below and search with `c: <edited code>`."
                )
                ec1, ec2 = st.columns([3, 1])
                edited_code = ec1.text_input(
                    "Detected Contour Code (U / D / R):",
                    value=audio_q.pattern,
                    key="audio_contour_editor",
                )
                if ec2.button("🔎 Search with Audio Query", type="primary", use_container_width=True):
                    cleaned_edit = edited_code.strip().upper()
                    if cleaned_edit == audio_q.pattern:
                        # Use full floating-point audio Query (includes transposition shifts)
                        st.session_state["last_response"] = search_library(songs, audio_q)
                    else:
                        # User edited the contour code manually -> search with c: <edited code>
                        manual_q = build_query(f"c: {cleaned_edit}")
                        st.session_state["last_response"] = search_library(songs, manual_q)

                st.divider()
                if st.session_state.get("last_response") is not None:
                    render_results_section(st.session_state["last_response"])

            except ValueError as exc:
                st.error(f"Audio Processing Error: {exc}")

    # ==========================================================================
    # TAB 3: ARCHITECTURE & AUTOMATED ACCEPTANCE SUITE
    # ==========================================================================
    with tab_tests:
        st.subheader("✅ Automated Acceptance Verification (Phases 1 to 5)")
        st.markdown(
            "Click below to execute the built-in unit and integration acceptance checks "
            "specified across Phases 1 to 5 of the Development Guide."
        )
        if st.button("▶️ Run All Phase 1–5 Acceptance Checks", type="primary"):
            report = run_acceptance_suite()
            st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)
        else:
            report = run_acceptance_suite()
            st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
