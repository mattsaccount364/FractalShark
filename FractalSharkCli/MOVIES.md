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
zoom-per-frame=1.02
iterations=1000000
width=1920
height=1080
video-profile=hevc-high
video-fps=30
video-gop-frames=1000
video-output=/home/user/name.movie/movie.ts
```

Use enough coordinate digits for the target zoom. The required keys are
`center-x`, `center-y`, `zoom`, and `iterations`. When omitted,
`zoom-per-frame` defaults to 1.02 and `video-fps` defaults to 30.
GUI exports preserve the full center and round the zoom scale to at most 39
significant digits. `wide-iterations` is the cap at 1× zoom. In GPU Auto the
iteration ramp steepens at 1e9× zoom, or at 1e34× in CPU Auto. The default
`la-slope-ratio=8` sets the ratio of the slopes after and before the switch.
`render-algorithm` and `perturbation-alg` default to Auto.

| Profile | Output |
| --- | --- |
| `hevc-high` | HEVC High tier, P5, NV12, VBR quality 16, up to 180 Mb/s |
| `hevc-low` | HEVC Main tier, P5, NV12, VBR quality 22, up to 50 Mb/s |
| `hevc-rgb-lossless` | HEVC, P4, lossless 4:4:4 with full-range RGB signaling |
| `h264-high` | H.264 High, P5, variable bitrate with quality 18, NV12 |
| `h264-low` | H.264 High, P1, variable bitrate with quality 30, NV12 |
| `png-only` | Numbered RGB16 PNGs |

Generated INIs list the choices beside each setting and include the active HEVC
tuning values. Explicit settings override profile defaults. Comment out tuning
lines to use a different profile's defaults.

- `video-preset` selects P1–P7. `video-cq` sets the lossy quality target from 1 to 51,
  with fractional values allowed for HEVC at 1/256 precision.
- Lossy HEVC `video-max-bitrate-mbps` requests a peak in whole Mb/s, capped by the
  tier and level. `video-average-bitrate-mbps` defaults to two-thirds of that effective
  peak and cannot exceed it. Both are encoder rate-control targets.
- Lossy HEVC `video-vbv-buffer-ms` sets a buffer size from 1 to 1,000 ms of peak bitrate.
  `video-vbv-initial-ms` sets initial fill up to that size. The defaults are 250 ms
  and 100 ms. Quantization can rise to meet the bitrate budget.

NV12 uses BT.709 limited range. The lossless preset rounds
RGB16 to RGB8 and uses NVIDIA's lossless tuning on the resulting sample values.
NV12 profiles require even dimensions. The CLI checks the chosen codec,
format and size before rendering. Video profiles use the NVIDIA driver
on Windows or Linux with NVENC API 13.1 support. The `--no-gpu` flag
selects CPU fractal rendering. Video encoding still uses NVENC.

## Pause and resume

The renderer and encoder overlap with up to four surfaces in flight.
`video-output` defaults to the absolute path to `movie.ts` beside the job INI.
Set it to an absolute `.ts` path in an existing directory before starting.
A custom destination uses a unique staging file in that directory and records
its name in `movie-state.ini` in the job folder. At completion, publication gives
the staged file its final name without replacing an existing file. Each checkpoint
flushes the stream and atomically records the durable byte offset and frame count.
Checkpoints occur every 32 completed
frames or one second. IDRs occur at frame zero, every `video-gop-frames`,
and on the first frame after resume. Checkpoints are independent of IDRs.

Open `movie-state.ini` to see `checkpointed-frames` out of `total-frames`,
`percent-complete`, `video-bytes`, and `active-time-ms`. The percentage
appears once the CLI knows the total and reaches `100.00` when every frame
is durable. Final publication follows the last checkpoint.

Press Ctrl+C or hold Ctrl+Alt for three seconds to pause gracefully. Rerun
the same command to resume. On restart, the CLI truncates bytes beyond the
last durable checkpoint and resumes encoding at the recorded frame. Wall
time accumulates across runs and excludes downtime. The PNG-only profile
checkpoints files in order and stores them under `frames/` in numbered
folders.

Create a new job folder when moving from an earlier movie state format.
