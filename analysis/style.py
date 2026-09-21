"""Shared run labels and colors for analysis charts."""

RUNS = [
    ("20260920-163007", "MLP 384→64→GELU, mean-pool 5 s"),
    ("20260921-082751", "Linear, mean-pool 5 s"),
    ("20260921-084027", "Linear, tail 400 ms"),
    ("20260921-084831", "Linear, EMA 400 ms"),
    ("20260921-085616", "Linear, tail 1 s"),
    ("20260921-085931", "Linear, tail 200 ms"),
    ("20260921-093516", "MLP 384→64→GELU, tail 400 ms"),
]

TARGET_RUN = "Linear, tail 400 ms"
"""The pause head we are keeping for serving and language breakdowns."""

LANG_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "tr": "Turkish",
    "zh": "Chinese",
}

COLORS = {
    "MLP 384→64→GELU, mean-pool 5 s": "#E69F00",
    "Linear, mean-pool 5 s": "#4D4D4D",
    "Linear, tail 400 ms": "#0072B2",
    "Linear, EMA 400 ms": "#CC79A7",
    "Linear, tail 1 s": "#009E73",
    "Linear, tail 200 ms": "#6A3D9A",
    "MLP 384→64→GELU, tail 400 ms": "#D55E00",
}
