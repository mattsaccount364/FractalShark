# Resumable zoom movies

Choose **Save → Export Zoom Movie Command…** in the GUI. Naming the export
`name.ini` creates `name.movie/movie.ini` and copies a command such as:

```powershell
& '.\Release\FractalSharkCli.exe' --movie '.\name.movie'
```

New jobs render from zoom one to the captured center and target zoom. Export
uses the default palette and preserves the current dimensions, antialiasing,
and final iteration limit. The CLI prepares each needed reference type at the
deepest remaining view that uses it.

## Configuration

A hand-written INI can be supplied with `--movie-config FILE` when creating
an empty job. Edit the INI before starting. Once rendering starts, the job
stores the encoder and render settings in its profile.

```ini
[movie]
center-x=-0.743643887037151
center-y=0.131825904205330
zoom=1e30
zoom-per-frame=1.01
iterations=1000000
width=1920
height=1080
video-profile=hevc-high
video-fps=60
video-gop-frames=1000
```

Use enough coordinate digits for the target zoom. The required keys are
`center-x`, `center-y`, `zoom`, `zoom-per-frame`, and `iterations`.
`wide-iterations` and `la-slope-ratio` control the iteration ramp.
`render-algorithm` and `perturbation-alg` default to Auto.

| Profile | Output |
| --- | --- |
| `hevc-high` | HEVC, P5, constant QP 16, BT.709 limited-range NV12 |
| `hevc-low` | HEVC, P5, constant QP 22, BT.709 limited-range NV12 |
| `hevc-rgb-lossless` | HEVC, P4, lossless 4:4:4 with full-range RGB signaling |
| `h264-high` | H.264 High, P5, variable bitrate with quality 18, NV12 |
| `h264-low` | H.264 High, P1, variable bitrate with quality 30, NV12 |
| `png-only` | Numbered RGB16 PNGs |

Set `video-preset` to override P1–P7, `video-qp` for lossy HEVC, or
`video-cq` for H.264. The lossless preset rounds RGB16 to RGB8 and uses
NVIDIA's documented lossless tuning on the resulting sample values.
NV12 profiles require even dimensions. The CLI checks the chosen codec,
format and size before rendering. Video profiles use the NVIDIA driver
on Windows or Linux with NVENC API 13.1 support. The `--no-gpu` flag
selects CPU fractal rendering. Video encoding still uses NVENC.

## Pause and resume

The renderer and encoder overlap with up to four surfaces in flight.
The encoder writes directly to `movie.ts.partial` and publishes
`movie.ts` at completion. Each checkpoint flushes the stream and records
the durable byte offset and next frame. Checkpoints occur every 32 completed
frames or one second. IDRs occur at frame zero, every `video-gop-frames`,
and on the first frame after resume. Checkpoints are independent of IDRs.

Press Ctrl+C or hold Ctrl+Alt for three seconds to pause gracefully. Rerun
the same command to resume. On restart, the CLI truncates bytes beyond the
last durable checkpoint and resumes encoding at the recorded frame. Wall
time accumulates across runs and excludes downtime. The PNG-only profile
checkpoints files in order and stores them under `frames/` in numbered
folders.

Jobs created by the earlier external-encoder pipeline use a different
format. Create a new job folder for native NVENC.
