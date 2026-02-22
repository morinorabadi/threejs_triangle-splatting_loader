live demo at [here](https://morinorabadi.ir/triangels/)

# Triangle/TSGD Web Viewer


Browser viewer for triangle splatting models with:
- `TSGD` primary support
- backward-compatible `TGSP` loading
- `Mesh` and experimental `Splat` render modes

## Run

```bash
cd web-viewer
pnpm install
pnpm run dev
```

Open `http://localhost:8001`.

## Load a model

Use the file picker to load `.tgsp`.

## Convert `.pth` to `.tgsp`

From repo download "export_tgsp.py" and run:

```bash
python3 export_tgsp.py /path/to/model.pth /path/to/model.tgsp
```


## Render modes

- `Splat (Experimental)`: default shader-based path; falls back to Mesh if not supported.
- `Mesh`: stable and fast to validate correctness.


## under development
still under development have some opacity issue
