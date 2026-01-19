# Introduction

## What is FractalShark?

FractalShark is a high-performance Mandelbrot set renderer focused on extreme zoom depths and experimental GPU-accelerated numerical methods on NVIDIA hardware.

## Why FractalShark?

Because fractals and sharks are both cool, so why not? And it's relatively unique, so hopefully, it can be Google'd.

## What is the Mandelbrot Set?

The Mandelbrot set is defined as the set of complex numbers that do not diverge to infinity when iterated according to \( Z_{n+1} \leftarrow Z_n^2 + c \), with \( Z_0 \) initialized to 0. It's possible to visualize the set by assigning a color depending on the number of iterations necessary to determine whether the point was in the set. In FractalShark, points are left black to denote membership, and colored if they escape (diverge) to infinity after some number of iterations. Membership is necessarily approximate in practice: infinite iterations are required in the general case, but a sufficiently large fixed iteration limit is "good enough" for visualization.

## Why do we need another Mandelbrot Set renderer? Thousands of others already exist!

FractalShark includes several innovations relative to most other Mandelbrot renderers. These are primarily engineering and implementation experiments rather than new mathematical results.

### 1. Experimental GPU-accelerated reference orbit computation

As of December 2025 (version 0.5), FractalShark includes an **experimental GPU-accelerated reference orbit implementation**. To my knowledge, this implementation is unique in the context of trying to accelerate Mandelbrot rendering.  FractalShark implements a full high-precision floating point pipeline on the GPU, including high-precision multiply, add, and subtract operations. The multiply implementation uses the Number Theoretic Transform for efficient at operation at very high preicsion.  These operations exploit GPU parallelism to accelerate arithmetic that is traditionally CPU-bound.

At a precision of \(10^{16384}\) using 32-bit limbs (≈ 100,000+ decimal digits), this GPU reference orbit implementation outperforms the existing multithreaded MPIR + AVX-2 CPU reference orbit by approximately **10× on an RTX 4090**.  The only built-in View that shows a clear benefit to the GPU-accelerated approach is View #30, which uses 16384 32-bit limbs internally.  My new RTX 5090 is slightly slower than the 4090, at ~9x faster than the existing FractalShark multithreaded CPU-based reference orbit calculator.  

This feature is still experimental and under active development.  To try it, you'll need an RTX 2xxx series or newer, e.g. RTX 3xxx/4xxx/5xxx should all work with recent drivers.  Then, start FractalShark, manually choose the HDRx32 LAv2 kernel, and under Perturbation choose GPU (experimental).  Then try e.g. View #5 to see if it renders.  If it does, then it's working.

### 2. Multiple CUDA Mandelbrot rendering strategies

FractalShark includes numerous **distinct CUDA implementations** of Mandelbrot rendering. These demonstrate different architectural and algorithmic strategies for mapping the problem to GPUs.

Several of these approaches appear to be undocumented or unpublished elsewhere, based on attempts to find prior examples. They are included both for performance experimentation and as reference implementations for different CUDA design patterns.

### 3. CUDA-based linear approximation implementations

FractalShark includes **two CUDA implementations of linear approximation**, a relatively recent technique for achieving high-performance deep-zoom rendering.

No existing CUDA implementations of this technique appear to be publicly available. These implementations were ported from **FractalZoomer**, and ultimately originated from work by **Claude** and **Zhuoran**. FractalShark adapts and extends these ideas within a CUDA-centric architecture.

### 4. Multithreaded high-performance CPU reference orbit calculation

FractalShark includes a **multithreaded CPU reference orbit implementation** designed to maximize performance on modern CPUs.

Using three threads total—two dedicated to squaring and one coordinating—has empirically provided the best performance on an AMD 5950X. Reference orbit computation is the dominant bottleneck at extreme magnifications, making this optimization particularly important.

This implementation outperforms all other known CPU-based reference orbit implementations when run on modern multi-core CPUs with AVX-2 support. Additional optimizations are possible but remain unexplored.

This subsystem also includes a **custom memory allocator** supporting an optional *perturbed perturbation* mode, in which intermediate-resolution reference orbit values are cached and reused across successive zooms.

### 5. Custom “2×32 + exponent” numeric type

FractalShark implements a custom **“2×32” floating-point type**, optionally combined with linear approximation and a *float + exponent* representation.

This type uses a pair of 32-bit floating-point values plus a shared exponent, providing an effective ~48-bit mantissa without using native 64-bit floating-point arithmetic. The result is substantially higher performance on consumer GPUs with only a modest loss of precision.

This implementation is **CUDA-only**; on CPUs, native 64-bit floating-point arithmetic is generally preferable.

### 6. Reference orbit compression and on-the-fly decompression

FractalShark supports **reference orbit compression**, currently a work in progress.

This idea was first implemented in Imagina for saving and loading reference orbits. FractalShark extends Zhuoran’s approach to **runtime per-pixel rendering**, decompressing reference orbit segments on demand during rendering.

For high-period locations (e.g. period 600,000,000), this can reduce memory usage by multiple gigabytes, often making the difference between a render being feasible or impossible.

## CUDA? What are the system requirements?

- **AVX-2 CPU**: used for reference orbit/MPIR library.
- **CUDA-capable NVIDIA 900-series or newer** (~2016)
- **CUDA-capable NVIDIA RTX 2xxx series or newer** for GPU-accelerated reference orbit.
- **Try updating your NVIDIA driver** if you get a "cuda error 35" when you run it.
- **Windows**

Basic Linux compatibility.  Tested with Wine using this configuration:
- Wine version: 9.0
- Ubuntu 24.04 LTS
- No nvidia GPU, CPU-only rendering
- Command line: wine FractalShark.exe -safemode

## Where do I download it?

- Downloadable from here: [https://github.com/mattsaccount364/FractalShark](https://github.com/mattsaccount364/FractalShark)

## More docs?
See FractalShark\Notes.  It's a major WIP and is mostly AI slop currently but I'd like to make it better.  It builds fine on Windows with Miktex installed.

## What else do I need to know?

- FractalShark is an experimental research project, not a polished end-user application. Expect bugs, dead code, rough edges, and missing abstractions.  Other implementations, like FractalZoomer, have much more polish, and Imagina will probably beat it on performance in the long term because Zhuoran is definitely better at coming up with optimized algorithms than I am.
- FractalShark is designed exclusively for use with Nvidia GPUs on Windows. I have an RTX 4090 and it's optimized for operation on that, but I believe it'll work fine on anything since the GTX 9xx series from ~2016.
- FractalShark only supports the Mandelbrot set, not other variants/derivatives.
- FractalShark offers very good performance if you have an Nvidia GPU, especially a 4090.
- FractalShark includes most algorithmic enhancements I'm aware of: linear approximation, high-precision MPIR perturbation for the reference orbit calculation, high-dynamic range float/float exp, many CUDA implementations, periodicity detection for reference calculation, and so forth.
- The source code is available under the GPLv3 license, so you can look at it if you want.

## How do I use it?

- Download `FractalShark.exe` from GitHub.
- Run it. If you get a blank screen or error message and believe you meet the system requirements then let me know and I'll speculate about the problem.
- Right-click to get a pop-up menu. Some of the options are buggy and will just crash/misbehave but most of the basic things should be fine.
- Left-click/drag to zoom in on a box. Alternatively, use hot keys: `z` to zoom in at mouse cursor, `shift+Z` to zoom out a bit, `b` to go back, `-` or `=` to increase/decrease iterations.

## Can I use CPU-only rendering and try it without an NVIDIA card?

If you get a CUDA error when you launch, as you might if you don't have an NVIDIA card or an updated driver, and if FractalShark is still responsive, you may be able to work around the problem using CPU-only rendering.

To try the workaround: assuming you have a black screen and an error message, dismiss the error, right-click on the black screen, select "Choose Render Algorithm" and pick one of the ones near the top that includes the word "CPU." Then right-click again, and select "Recalculate, Reuse Reference" and see if you get an image.

If you use this workaround, you may as well use Imagina. It offers superior CPU performance.

Note that the GPU-accelerated reference orbit really doesn't pay off until you're at truly ridiculous depths, e.g. 4096 limbs or greater, which is 10K+ digits.  At shallower depths it'll be slower than CPU, so don't get too hung up if your card is a little older.

## What features will FractalShark never have?

- Support for non-Mandelbrot fractals. Not worth the engineering complexity relative to the payoff.
- Support for AMD cards unless I buy one
- High-quality CPU support. Zhuoran's Imagina easily has that covered - he did a great job. FractalShark does have two CPU-based linear approximation implementations, but it's not very well optimized, and is intended primarily for testing/debugging. Runtime reference compression also is supported in CPU-only mode.

## What future work is there?

- Minibrot detection/location. Imagina has this I know, not sure who else has it.
- TBD - depends on motivation.

## What bugs does FractalShark have?

- I've not tried running this on a machine without an NVIDIA card in a long time, and the CPU-only workaround described above may not really work.
- The "autozoom" feature is busted and should be replaced. Don't use it.
- Load location/save location have weird nuances - future work. The "current position" will copy your location to the clipboard.
- Some CUDA kernels are busted.
- If it's slow and taking a long time, there's no way to abort it. Use task manager and terminate it that way if you don't want to wait. Try holding the CTRL key and maybe it'll work depending on what's being slow, but it probably won't.
- Too many to list - don't expect much.

## What if it just exits suddenly with no indication of what went wrong? What if I hit some other random bug and it's annoying me enough to complain about it?

If FractalShark suddenly exits, that implies it crashed. It's buggy and kludgy. With some luck, there should be a "core.dmp" file created in the same directory as the executable. If you're motivated, feel free to create a new "issue" on GitHub and upload the file as an attachment, and with some luck, I (the author) can sort out what I screwed up.

If you hit any other bug/wrong behavior, you can also create an "issue" and describe the problem. The URL to create the "issue" and notify me about it is: [https://github.com/mattsaccount364/FractalShark/issues](https://github.com/mattsaccount364/FractalShark/issues)

## What other projects are used for inspiration?

Many.

- The perturbation inspiration is from Kalles Fraktaler. It certainly helped clarify some things. And Claude's blog was a great help. [Kalles Fraktaler](https://mathr.co.uk/kf/kf.html)
- [Deep Zoom Theory and Practice](https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html)
- [Deep Zoom Theory and Practice Again](https://mathr.co.uk/blog/2022-02-21_deep_zoom_theory_and_practice_again.html)

- The reference orbit periodicity detection implementation is from Imagina. [Imagina](https://github.com/5E-324/Imagina)

- The Float/Exp and linear/bilinear approximation implementations are from FractalZoomer. FractalZoomer is written in Java, but it was easy enough to port to C++ and "templatize." [FractalZoomer](https://sourceforge.net/projects/fractalzoomer/)

- FractalShark uses several third-party libraries: MPIR, and WPngImage ([WPngImage](https://github.com/WarpRules/WPngImage)). I also found a CUDA QuadDouble and DblDbl implementations somewhere and have example kernels demonstrating their use.

- FractalShark's reference orbit compression is novel code, but based on the approach Zhuoran described here: [Reference Compression](https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142). Claude posted a simple easy-to-understand sample here, which FractalShark's implementation is loosely based on: [Fractal Bits](https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression)

- This implementation would probably not have happened without the work of these folks so thank you!

## Closing

If you're bored and want to try yet another Mandelbrot set renderer, give it a go. You can download releases from the GitHub page. Your mileage may vary. If you don't meet the system requirements, it will likely fail noisily rather than degrade gracefully. In terms of time investment, I'll probably only spend a couple of hours on weekends fussing with it so don't expect dramatic rewrites.

## History

- **2001**: The first version of FractalShark I did as a freshman in college. I added a basic network/distributed strategy to speed it up. It split the work across machines. This feature is now broken and won't be revived.
- **2008**: I did a distributed version that ran on the University of Wisconsin's "Condor" platform. That was pretty cool, and allowed for fast creation of zoom movies.
- **2017**: I resurrected it, and added high precision, but no perturbation or anything else. At that point, it was theoretically capable of rendering very deep images, but it was so slow as to be largely useless.
- **2023-Q2 2024-Q2**: I bought this new video card, and wanted to play with it, so ended up learning about CUDA and all these clever algorithmic approaches you all have found out to speed this up. So here we are.
- **2024-Q3-2025**: Worked almost exclusively on high-precision GPU-accelerated reference orbit implementation.

## Build instructions

The most authoritative formal instructions are available via the github build.yml file, since FractalShark is now hooked up to github's CI/CD "Actions" mechanism. You can find this formal build script at .github/workflows/build.yml.  Look at build results to see if it's passing.  So far, only one person has ever pushed anything to this repository; thus, I consider myself free to break the main branch.  If anyone else actually wants to work on it, then I'll stop doing that :)

Here are some more hand-wavy instructions that basically describe the idea:

1. `git clone https://github.com/mattsaccount364/FractalShark.git`
2. `cd FractalShark`
3. `git clone https://github.com/BrianGladman/mpir.git`
4. Open `mpir\msvc\vs22\mpir.sln`
5. Rebuild everything. Note, you probably want YASM installed at: `C:\Program Files\yasm\*`. This may take a little while.
6. The distributed version of FractalShark uses the static lib_mpir_skylake_avx incarnation of MPIR.
7. Install Nvidia Nsight Compute SDK. The distributed version of FractalShark uses 12.2
8. At this point, you should be able to build all the FractalShark projects. The one called FractalShark is the most interesting. FractalTray basically works. The rest are effectively dead code.

Have fun, hopefully :)