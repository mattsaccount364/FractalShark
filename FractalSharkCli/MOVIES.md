# Resumable zoom movies

Movie mode keeps the center fixed, renders the deepest frame first, and works
outward. Playback runs in the opposite order. It uses FractalShark's existing
native reference-orbit and LA autosave and loader, with **unchanged flush and
shutdown semantics**. A crash in the existing persistence window can require
recalculating a reference. Incomplete reference calculations are not checkpointed.

Create a configuration file (decimal coordinate values can contain millions of
digits; they are not passed through the operating system command line):

```ini
[movie]
center-x=-0.743643887037151
center-y=0.131825904205330
zoom=1e100
zoom-per-frame=1.01
iterations=1000000
width=1920
height=1080
render-algorithm=AutoSelect
perturbation-alg=Auto
wide-iterations=8192
la-slope-ratio=8
video-chunk-frames=1000
video-fps=60
```

The coordinates above illustrate syntax; choose a target with enough accurate
digits for the desired depth. `zoom` must be at least one, and
`zoom-per-frame` must exceed one. The required fields are `center-x`, `center-y`,
`zoom`, `zoom-per-frame`, and `iterations`. Other render keys are `antialiasing`,
`palette-map`, and `palette-depth`, with the same meanings as the normal CLI.
Dimensions default to 1024 by 768. `render-algorithm` defaults to `AutoSelect`.
Paths in the INI are relative to that file; palette paths need no surrounding
quotes even when they contain spaces.

```powershell
# Create the job and archive groups continuously.
.\Release\FractalSharkCli.exe --movie H:\movies\target --movie-config movie.ini --ffmpeg C:\FFMPEG\bin\ffmpeg.exe

# Resume the same job. The saved FFmpeg path is reused.
.\Release\FractalSharkCli.exe --movie H:\movies\target

# Override the executable location on resume.
.\Release\FractalSharkCli.exe --movie H:\movies\target --ffmpeg C:\other\ffmpeg.exe
```

Linux uses the same options, for example `--ffmpeg /usr/bin/ffmpeg`.
The executable must include `libaom-av1`. An exact RGB round trip is checked before
the first expensive render. FFprobe is not required.

Omitting `--ffmpeg` on creation produces PNGs only. Supplying it later compacts
existing groups, including a finished job. An archive-enabled job requires its
encoder on subsequent runs; a missing executable stops the job before more frames
are rendered. `--no-gpu` and `--commit-cap-bytes` are runtime options. Put render
options in the INI rather than mixing them with `--movie`.

Press Ctrl+C to request an orderly pause. The current render/reference calculation
and pending PNG saves finish; an unfinished video encode may be cancelled. Run the
same command to resume. An OS lock prevents two processes writing one job. The job
snapshots its configuration and palette, so the original inputs are not needed on
resume. Rendering settings and the resolved CPU/GPU profile remain fixed.

## Frames and iteration limits

Frame zero has zoom one, at the target center. Frame indices increase toward the
exact target zoom. Adjacent frames use the requested multiplier except for a
possibly shorter first interval. Names have 20 digits and PNG directories hold at
most 10,000 frames. The scheduler computes each view from its index and does not
retain interactive navigation history or a list of all views.

Iteration limits grow linearly with logarithmic depth up to full LA v2 engagement,
then continue linearly with the slope multiplied by `la-slope-ratio` (default 8).
For AutoSelect, the transition follows the existing GPU/CPU algorithm tiers.
Forced algorithms that always or never use full LA have a single linear segment.
`wide-iterations` defaults to 8192, capped at `iterations`. Limits reach the exact
configured target at the final playback frame. This curve is traversed backward
while rendering.

The full target is stored once in the movie input. Progress files store indices,
not per-frame coordinate strings. Existing native reference metadata still stores
its own reference coordinates in its existing format.

## Lossless video chunks

Chunks cover fixed playback ranges (0–999, 1000–1999, etc.). Each is encoded in
ascending playback order, although deeper chunks are produced first. The highest
range can be shorter. One million frames normally produce 1,000 movie chunks.

PNG output remains RGB16. **Video archives are RGB8:** each 16-bit channel is
rounded using `(value + 128) / 257`, with no gamma conversion or dithering. AV1 is
pixel-lossless relative to those RGB8 samples and uses inter-frame prediction.
Deleting archived PNGs gives up their extra 16-bit precision, as intended for this
mode. Existing CLI and GUI PNG output retain their usual precision.

Each Matroska chunk has exactly one initial keyframe. Subsequent frames can
depend on earlier frames in that chunk, and no dependency crosses chunks. Encoder
settings use libaom High profile, full-range identity RGB, lossless mode, CPU-used
4, disabled alternate-reference frames/lookahead, and a fixed keyframe interval
equal to the actual chunk length.

Every encoded chunk is decoded and checked for exact RGB8 content, frame count,
dimensions, and one initial keyframe. Only after durable movie publication and
registry publication are its PNGs deleted. Interrupted deletion is safe to retry.
Unregistered movies are regenerated from retained PNGs rather than trusted.
Frame/chunk publication has movie-specific durability operations; native
reference flush behavior is untouched.

The pipeline allows one batch encoding, one batch collecting, and at most four
pending PNG saves. Slow encoding applies backpressure. Failed encoding preserves
PNGs and stops further accumulation. Logs report actual input/output bytes and
time; compression depends on the scene and zoom rate and is not guaranteed to
beat PNG for every chunk.

`movie.ffconcat` lists committed chunks in playback order. After completion:

```text
ffmpeg -f concat -safe 0 -i movie.ffconcat -c copy movie.mkv
```

Keep `movie.ini`, `profile.txt`, progress files, chunk completion records, and the
native reference files with the job. Frame PNGs for committed chunks are removed
automatically. To change render settings, create a new job directory.

## Tests

The normal `FractalSharkTest --filter 'Movie*'` suite covers scheduling, recovery,
hashes, PNG conversion, and render/resume. Set `FRACTALSHARK_TEST_FFMPEG` to an
absolute executable path to enable the 1,000-frame codec integration case:

```powershell
$env:FRACTALSHARK_TEST_FFMPEG = 'C:\FFMPEG\bin\ffmpeg.exe'
.\Debug\FractalSharkTest.exe --filter 'Movie*'
```
