# `turn_ui` package

Thin FastAPI app that serves the microphone page. It does not load PyTorch.

The browser talks to `turn-runtime-serve` for `/heads` and `/ws`. This
process injects `RUNTIME_URL` into `index.html`; the page can override it.

## Structure

```text
turn_ui/
├── app.py                 Application factory (`create_app`) and Uvicorn entrypoint (`run`)
├── config/
│   └── logging_setup.py   `LOG_LEVEL` console logging
├── index.html             Capture UI and scrolling state chart
├── routers/
│   └── page.py            GET /, GET /health
└── settings/
    └── settings.py        HOST, PORT, RUNTIME_URL
```

## Configuration

`RUNTIME_URL` (default `http://127.0.0.1:8766`) is the turn runtime origin.
`HOST` / `PORT` default to `127.0.0.1:8765`. See the repo-root
[`.env.example`](../../../../.env.example).
