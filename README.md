# Introduction

## What is FractalShark?

A Mandelbrot Set renderer optimized for use with NVIDIA GPUs.

## Why FractalShark?

Because fractals and sharks are both cool, so why not? And it's relatively unique, so hopefully, it can be Google'd.

## What is the Mandelbrot Set?

The Mandelbrot set is defined as the set of complex numbers that do not diverge to infinity when iterated according to \( Z_{n+1} \leftarrow Z_n^2 + c \), with \( Z_0 \) initialized to 0. It's possible to visualize the set by assigning a color depending on the number of iterations necessary to determine whether the point was in the set. In FractalShark, points are left black to denote membership, and colored if they escape (diverge) to infinity after some number of iterations. Points determined to be in the set are necessarily approximate. Infinite iterations are necessary in the general case, but the software can simply use a large fixed number because it's "good enough."

## Why do we need another Mandelbrot Set renderer? Thousands of others already exist!

FractalShark includes a few innovations relative to most other renderers, most related to engineering rather than fundamentals.

First, as of December 2025, FractalShark includes an experimental GPU-accelerated reference orbit implementation.  At present, this is the only known implementation of such a strategy.  FractalShark implements the number theoretic transform (NTT) with associated high-precision multiply, add, and subtract support.  It uses the GPU's parallelism to speed up these operations relative to what's possible on the CPU.  At 10^16384 32-bit limbs (~100K digits), FractalShark beats its existing multithreaded MPIR/AVX-2 reference orbit implementation by 10x on an RTX 4090.  More information on the implementation

Second, FractalShark includes numerous NVIDIA CUDA implementations, demonstrating distinct strategies for rendering the Mandelbrot. These are cool in their own right, because I was not able to find existing examples of some of these approaches.

Third, FractalShark includes two different CUDA-based linear approximation implementations. Linear approximation is a relatively new technique for getting high performance visualizations, and no existing CUDA implementations appear to exist. These implementations are ported from [FractalZoomer](https://github.com/hrkalona/Fractal-Zoomer), and themselves originated from [Claude](https://mathr.co.uk/web/) and [Zhuoran](https://github.com/5E-324/Imagina).

Fourth, FractalShark includes a multi-threaded reference orbit calculation for a better performance. Using three threads total, with two for squaring and one for coordinating, appears to offer superior performance, at least on an AMD-5950X. Calculating the reference orbit is the single-largest bottleneck to high magnification renders, so it's important to spend some time on that. This implementation beats all other known examples in performance assuming you have a modern multi-core CPU capable of executing AVX-2 instructions. More optimizations are possible here that are as-yet unexplored. This implementation also uses a custom memory allocator for an optional "perturbed perturbation" implementation, wherein intermediate resolution reference orbit entries are stored and re-used for subsequent zooms.

Another neat FractalShark-specific feature is an implementation of a "2x32" type combined with optional linear approximation and "float+exp." This implementation supports a pair of 32-bit floating point numbers + an exponent, to provide a combined ~48-bit mantissa without using the native 64-bit type. The benefit is significant performance improvements on consumer video cards, with nearly the same precision. This implementation is CUDA-only. On a CPU, you might as well use the native 64-bit type.

Finally, FractalShark supports reference orbit compression. This feature remains a work in progress, but it promises to reduce end-to-end memory (RAM) requirements, especially on high-period locations. Imagina first implemented reference compression for saving/loading reference orbits. FractalShark extends Zhuoran's idea to runtime/per-pixel rendering, by decompressing the reference orbit on the fly as needed. An example period-600,000,000 spot can see a memory reduction of multiple gigabytes, which can make the difference between being able to render it and not being able to render it.

## CUDA? What are the system requirements?

- **AVX-2 CPU**: used for reference orbit/MPIR library.
- **CUDA-capable NVIDIA 900-series or newer** (~2016)
- **CUDA-capable NVIDIA RTX 2xxx series or newer** for GPU-accelerated reference orbit.
- **Try updating your NVIDIA driver** if you get a "cuda error 35" when you run it.
- **Windows**

## Where do I download it?

- Downloadable from here: [https://github.com/mattsaccount364/FractalShark](https://github.com/mattsaccount364/FractalShark)

## More docs?
See FractalShark\Notes.  It's a major WIP and is mostly AI slop currently but I'd like to make it better.  It builds fine on Windows with Miktex installed.

## What else do I need to know?

- FractalShark is a work in progress, has many bugs, lots of messed-up code, and is basically just a huge blob of hacks. Other implementations, like FractalZoomer, have much more polish, and Imagina will probably beat it on performance in the long term because Zhuoran is definitely better at coming up with optimized algorithms than I am.
- If you look at the source code, much of it is dead. FractalShark is the main project and is the live part for the time being.
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

- Support for non-Mandelbrot fractals. Too much dinking around.
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

If you're bored and want to try yet another Mandelbrot set renderer, give it a go. You can download releases from the GitHub page. YMMV, caveat emptor, don't bother running it unless you meet the system requirements because it'll probably just spew some opaque error and exit. In terms of time investment, I'll probably only spend a couple of hours on weekends fussing with it so don't expect dramatic rewrites.

## History

- **2001**: The first version of FractalShark I did as a freshman in college. I added a basic network/distributed strategy to speed it up. It split the work across machines. This feature is now broken and won't be revived.
- **2008**: I did a distributed version that ran on the University of Wisconsin's "Condor" platform. That was pretty cool, and allowed for fast creation of zoom movies.
- **2017**: I resurrected it, and added high precision, but no perturbation or anything else. At that point, it was theoretically capable of rendering very deep images, but it was so slow as to be largely useless.
- **2023-2024q2**: I bought this new video card, and wanted to play with it, so ended up learning about CUDA and all these clever algorithmic approaches you all have found out to speed this up. So here we are.
- **2024q3-2025**: Worked almost exclusively on high-precision GPU-accelerated reference orbit implementation.

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