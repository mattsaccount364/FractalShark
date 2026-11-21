# Resumable zoom movies

New movies keep the center fixed and render in playback order, from zoom one
inward to the exact deep target. Before rendering, the renderer prepares one
reference at the deepest remaining frame for each required orbit type, starting
with the deepest view. Auto's two ordinary float modes share a reference. The
normal cache reuses prepared references within their ranges. On resume, the
native orbit loader can reuse references saved during an earlier run. A crash
before an orbit is saved can require recalculating that reference.

## Export from the GUI

Choose **Save → Export Zoom Movie Command…** in the Windows or Linux GUI. Choose
an INI name and select an FFmpeg executable that includes `hevc_nvenc`. Choosing
`name.ini` creates a new `name.movie` folder containing `movie.ini` and the saved
encoder path in `ffmpeg.txt`; no sibling INI is written. Existing job folders
cannot be overwritten by exporting. The GUI
locates the neighboring CLI executable, or asks for its location if necessary.
After saving, it copies the command and displays it for review. Paste it into
PowerShell on Windows or a POSIX shell on Linux to start rendering.

The current view becomes the final playback frame. The export preserves its
full coordinate precision, dimensions, antialiasing, and final iteration limit.
It uses Auto render/reference algorithms and the CLI's default palette settings.
The INI defaults to `1.01` zoom per frame, `60` fps, and `1000` frames per chunk;
edit `movie.ini` inside the folder before starting to change these settings.
The generated command contains only the executable and `--movie JOBDIR`, for
both the first run and resumption. Rerun the command to resume the same job;
changing settings after starting requires a new job directory. The generated INI
also contains commented lossless/HEVC/H.264 encoder examples.

## Create a configuration manually

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
render-order=wide-to-deep
video-encoder-args=-c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 16 -profile:v main -pix_fmt nv12 -colorspace bt709 -color_range tv
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
The executable must include `hevc_nvenc`, with a supported NVIDIA GPU and driver.
Preflight checks encoder support at the requested resolution and hard-link support before
rendering. There is no software encoder fallback. FFprobe is not required.

Omitting `--ffmpeg` on creation produces PNGs only. Supplying it later compacts
existing groups, including a finished job. An archive-enabled job requires its
encoder on subsequent runs; a missing executable stops the job before more frames
are rendered. `--no-gpu` controls fractal rendering; archiving still requires NVENC.
`--commit-cap-bytes` is a runtime option. Put render
options in the INI rather than mixing them with `--movie`.

Press Ctrl+C or hold Ctrl+Alt for three seconds to request an orderly pause. The
Ctrl+Alt shortcut reads desktop key state on Windows and X11. On headless
Linux, use Ctrl+C or SIGTERM. The current render/reference calculation and
pending PNG saves finish, and an unfinished video encode may be cancelled. Run the
same command to resume. An interrupted encoder restarts its chunk from retained
PNGs, while completed frames and registered chunks are reused. An OS lock prevents two processes writing one job. The job
snapshots its configuration and palette, so the original inputs are not needed on
resume. Rendering settings and the resolved CPU/GPU profile remain fixed.
The `movie.lock` file normally remains after exit; the OS handle, rather than
the presence of that file, determines whether the job is locked.
Resume logs report rendered and remaining frames across all frame directories,
the next frame index, and the number of ready archive chunks. Saved chunks are
queued without waiting for their encoding before rendering the next missing frame.

## Active wall time and rendering pipeline

The CLI reports session and cumulative active wall time on completion, pause, or
failure. It includes setup, reference preparation, rendering, PNG saves, encoding,
assembly, waits, and cleanup. Time between runs is excluded. The CRC32-protected
`time.txt` is atomically replaced every second and at orderly exit. A forced kill
restores the last complete heartbeat. `progress.txt` uses the same replacement
protocol for the saved frame frontier. On resume, the latest valid numbered
record is migrated and its old names are retired after the replacement is durable.
Jobs created before timing was available
start tracking at their next launch; earlier active time cannot be reconstructed.

Movie frames use the existing four renderer slots. Up to eight render/save requests
may be outstanding. Frame state is prepared in order, GPU execution and PNG saves
overlap, and checkpoints advance only through contiguous successful PNG saves.
Each save owns its frame's iterations and captured coloring settings. Headless
movie work does not enter presentation queues. Logs distinguish submitted frames,
saved frames, and archive waits.

## Frames and iteration limits

For new jobs, frame zero is zoom one and the last frame is the exact deep target.
Indices increase in playback order. Adjacent frames use the requested multiplier
except for a possibly shorter final interval. Names have ten digits, covering the supported
`INT_MAX` frame-count limit. Every frame folder belongs to one video chunk:

```text
frames/chunk_0000000000/frame_0000000000.png
frames/chunk_0000000000/frame_0000000001.png
...
frames/chunk_0000000001/frame_0000001000.png
chunks/chunk_0000000000.mkv
chunks/chunk_0000000001.mkv
```

With the default chunk size, the first folder holds the first 1,000 rendered
frames. Only the last chunk may be shorter. Folder numbers increase with rendering;
there is no separate 10,000-frame bucket or divisibility restriction. The scheduler
computes views directly from indices. Progress records store the next ascending
render index, and the CLI logs each frame before rendering it.

Iteration limits grow linearly with logarithmic depth up to full LA v2 engagement,
then continue linearly with the slope multiplied by `la-slope-ratio` (default 8).
For AutoSelect, the transition follows the existing GPU/CPU algorithm tiers.
Forced algorithms that always or never use full LA have a single linear segment.
`wide-iterations` defaults to 8192, capped at `iterations`. Limits reach the exact
configured target at the final playback frame. New movies traverse this curve toward increasing iteration limits; legacy jobs keep their saved direction.

The full target is stored once in the movie input. Progress files store indices,
not per-frame coordinate strings. Existing native reference metadata still stores
its own reference coordinates in its existing format.

## Video chunks and assembly

Chunks cover ascending render ranges (0�999, 1000�1999, etc.). New jobs encode
frames in playback order, and their assembly manifest lists ascending chunks.
One million frames normally produce 1,000 movies in the single `chunks/` folder.

PNG output is RGB16. Before encoding, each channel is rounded to 8 bits using
`(value + 128) / 257`, with no gamma conversion or dithering. The default HEVC
NVENC profile converts those samples to BT.709 limited-range NV12 (4:2:0) and
uses P5 with QP 16. The optional lossless profile preserves the RGB8 samples in
full-range RGB using RExt and P4. Each chunk begins with a keyframe and uses no
B-frames or lookahead. NV12 can round odd encoded dimensions up to even values.
Use even dimensions or select RGB lossless to preserve an odd frame size.
Lossless tuning is documented in [NVIDIA's programming guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html#encoder-tuning-info-and-preset-configurations);
[FFmpeg maps `-tune lossless` directly to that mode](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/nvenc_hevc.c).
No dependency crosses chunks. Deleting source PNGs gives up their extra 16-bit
precision; normal CLI and GUI PNG output retain their usual precision.

FractalShark relies on the selected NVENC settings and validates the archive
without hashing or decoding it a second time. FFmpeg must exit successfully,
consume the expected input dimensions/count, report the expected encoded count in its final progress
block, and produce a nonempty movie. Only after durable movie publication and
registry publication is its entire frame folder deleted. Interrupted deletion is
safe to retry. The empty `frames/` directory is removed after completion.
Unregistered movies are regenerated from retained PNGs rather than trusted.
Frame/chunk publication has movie-specific durability operations; native
reference flush behavior is untouched.

FFmpeg reads each chunk as a numbered PNG sequence and converts its pixels
directly to planar RGB8 for NVENC. New jobs read their numbered PNG sequence directly from their frame folder.
That folder gives image2 a real end-of-sequence boundary: FFmpeg's input prefetch cannot reach a neighboring chunk being deleted
by another worker. Hard links share the existing image data; no PNG data is copied
or decoded by C++. Older deep-first jobs retain their reverse hard-link input sequence and
require hard-link support. Temporary legacy input links are removed after use or
on the next resume following a hard termination. Only progress and diagnostic metadata crosses the
process pipes. Up to four archive workers encode chunks concurrently. The shared decoder budget is
logical processor count minus one, with a minimum of one; fewer processors also
reduce the worker count. Fixed reservations divide that budget across workers,
and each decoder is additionally capped by its chunk length. Filter threads use
the worker's reservation. Encoding runs on NVIDIA NVENC hardware. PNG decoding,
RGB conversion remain in FFmpeg on the CPU.
FFmpeg runs at idle priority in a kill-on-close job
on Windows and at nice level 19 in an owned process group on Linux. Cancellation
is checked even while the process produces no output. A hard-killed Windows
renderer also kills its encoder through the job object.

The pipeline allows up to four batches archiving, one batch collecting, and at most four
pending PNG saves. On resume, an inherited backlog of ready chunks may also be
queued. Rendering can collect one new chunk alongside that backlog; it waits for
archive capacity only after that new chunk fills, until the chunk starts encoding.
Any completed worker can free capacity; completion does not wait for the oldest
chunk. Completion records and PNG deletion stay on the render thread, and the
assembly manifest remains in playback order. The wait log names the active worker
count, worker limit, queued chunks, and remaining render count.
Failed encoding stops new rendering and checkpoints successful pending PNG saves
before reporting the error. Cancellation joins every worker before releasing the
job lock. Unarchived PNGs remain available for resumption.
With an encoder enabled, completion is reported after the final movie is published.
Logs report actual input/output bytes and encoding and archive-wait
time; compression depends on the scene and zoom rate and is not guaranteed to
beat PNG for every chunk.

`movie.ffconcat` lists committed chunks in playback order while archiving. Once
all chunks are ready, FFmpeg stream copies each into `movie.ts.partial`. The
append is flushed and recorded in `assembly.txt` before its source MKV and
completion record are removed. Disk use during assembly is the remaining chunks,
the committed transport stream, and at most one in-progress chunk's output.
The final rename publishes **`movie.ts`** as one playable file.

After an interruption, rerun the original command. The job trims any unfinished
tail and resumes at the first uncommitted chunk. Assembly uses FFmpeg without
starting the renderer or NVENC when all chunks are already archived. A completed
movie can be reopened by the CLI without FFmpeg. Keep `assembly.txt` with the job
so the CLI can recognize the committed output. This assembly path supports
durations below the MPEG-TS timestamp wrap (about 26.5 hours). Longer movies
retain their MKV chunks and manifest for a separate join.

Keep `movie.ini`, `profile.txt`, `progress.txt`, `time.txt`, and the native
reference files with the job. To change render settings, create a new job
directory. Existing deep-first jobs retain their
saved direction, indices, and completed work; no frame/chunk remapping is needed.
New jobs record wide-to-deep order explicitly. The immediately preceding HEVC
metadata is upgraded automatically under the job lock, retaining completed movies,
PNG files, and frame progress. Only small metadata uses CRC32; image data is never
checksummed by the application during encoding. Publishing the upgraded profile
last makes an interrupted upgrade safe to retry. Old AV1 jobs remain unsupported.

## Encoder arguments and examples

`video-encoder-args` is a whitespace-separated list of option/value pairs; double
quotes group a value. It is passed as arguments, without shell expansion. The
selected codec must be `hevc_nvenc` or `h264_nvenc`. Supported options cover preset,
tune, profile, pixel format, rate control (`-rc`, `-qp`, `-cq`, `-b:v`, `-maxrate:v`,
`-bufsize:v`, `-multipass`), AQ, level, and color metadata. Input/output paths,
filters, frame limits, GOP boundaries, progress and process settings remain owned
by FractalShark. Encoder settings are fixed for the job so chunks stay compatible.
An omitted key defaults to HEVC QP 16 when starting a job. Legacy snapshots
without the key resume with their original RGB lossless settings.

Generated INIs include HEVC lossless, HEVC high/low quality, and H.264 high/low
quality. HEVC High uses QP 16 by default and HEVC Low uses QP 22 for smaller
files. Both use P5 and NV12. H.264 High uses P5/CQ18 and H.264 Low uses P1/CQ30,
both with VBR. H.264 NVENC works at 3840x2160 on the tested
RTX 5090; that driver reports a 4096-pixel H.264 width limit. Choose compatible
movie dimensions before starting; the INI encoder arguments do not resize frames.
The capability probe tests the actual selected settings and dimensions.

For comparison, these full commands transcode an existing RGB8 movie. The H.264
commands resize to width 3840 while preserving aspect ratio:

```text
ffmpeg -i input.mkv -c:v hevc_nvenc -preset p4 -tune lossless -profile:v rext -pix_fmt gbrp -colorspace rgb -color_range pc hevc-lossless.mkv
ffmpeg -i input.mkv -vf "scale=iw:ih:in_range=pc:out_range=tv:out_color_matrix=bt709,format=nv12" -c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 16 -profile:v main -pix_fmt nv12 -colorspace bt709 -color_range tv hevc-high.mkv
ffmpeg -i input.mkv -vf "scale=iw:ih:in_range=pc:out_range=tv:out_color_matrix=bt709,format=nv12" -c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 22 -profile:v main -pix_fmt nv12 -colorspace bt709 -color_range tv hevc-low.mkv
ffmpeg -i input.mkv -vf "scale=3840:-2:in_range=pc:out_range=tv:out_color_matrix=bt709,format=nv12" -c:v h264_nvenc -preset p5 -tune hq -rc vbr -cq 18 -b:v 0 -profile:v high -colorspace bt709 -color_range tv h264-high.mkv
ffmpeg -i input.mkv -vf "scale=3840:-2:in_range=pc:out_range=tv:out_color_matrix=bt709,format=nv12" -c:v h264_nvenc -preset p1 -tune hq -rc vbr -cq 30 -b:v 0 -profile:v high -colorspace bt709 -color_range tv h264-low.mkv
```

Every chunk has a fresh encoder session and identical stream settings. The GOP
length is the configured chunk size (minimum two), even for the partial last chunk.
Assembly with `-c copy` remuxes compressed packets without decoding or
re-encoding. It still reads and writes the compressed bytes.

## Tests

The normal `FractalSharkTest --filter 'Movie*'` suite covers scheduling, recovery,
metadata checksums, PNG conversion, and render/resume. Set `FRACTALSHARK_TEST_FFMPEG` to an
absolute executable path to enable the existing NVENC integration cases on an
NVIDIA-equipped machine. Those cases require `--use-gpu`:

```powershell
$env:FRACTALSHARK_TEST_FFMPEG = 'C:\FFMPEG\bin\ffmpeg.exe'
.\Debug\FractalSharkTest.exe --use-gpu --filter 'Movie*'
```

For fast resume-scheduler validation with FFmpeg, exclude the separate codec case:

```powershell
.\Debug\FractalSharkTest.exe --use-gpu --filter 'Movie*' --exclude 'MovieVideo_1000FrameLosslessSingleKeyframe'
```

The `MovieResume_*` regression cases use pre-generated 256-by-256 PNGs and at most
eight actual tiny renders per rendering case. Per-chunk encoder gates prove
concurrent starts, out-of-order completion, bounded rendering overlap, cancellation,
and resumption after failure. Synthetic PNGs also check zoom-in chunk assembly. Production-size frame indices and directory boundaries use metadata only.
