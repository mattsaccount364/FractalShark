# Introduction

## 2025-9-1 News

The full reference orbit works with CUDA, though without periodicity detection or high precision to "float exp" conversion.  That's future work but I'm not worried about it.  To be clear, this is not hooked up end-to-end with FractalShark itself, it's only working in a standalone test environment.  But the results are promising and prove it works.

This initial implementation relies on Karatsuba for the multiplies/squaring and then follows those with the high-precision adds/subtracts.  Initial results suggest a ~12x perf improvement relative to single-threaded CPU only, when comparing an overclocked 5950X vs an RTX 4090 with CUDA.  I'm happy with that, but not completely.

The main performance problem is this Karatsuba implementation.  Getting decent performance out of Karatsuba obviously requires recursion, and that gets costly on the GPU.  This implementation recurses several levels, which avoids costly local memory spill, but bites us because of register pressure.  The high register pressure limits the parallelism we can achieve.  The nice thing about Karatsuba for me is that it's not that hard to understand conceptually, so it was a great initial target for someone who doesn't know what they're doing.

Now that it's working, and I have a better sense of what's going on, I'm going to try a full NTT-based high-precision multiply approach.  The idea here is to rely on the number theoretic transform, similar to FFT, and parallelize the high precision multiply that way.

With this commit, we have a working host-based (CPU-only) approach to NTT high-precision multiply that supports power-of-2 mantissa sizes and should scale effectively to CUDA but that's TBD.  It will be at least several months more work at my current rate (a few hours on the weekends) to achieve a first-cut CUDA implementation.

### NTT-based high-precision multiply (magic prime 2^64 - 2^32 + 1)

AI-generated slop follows in this subsection.  It looks accurate.

I'm experimenting with an NTT implementation over the 64-bit "magic" prime p = 2^64 - 2^32 + 1. This prime is NTT-friendly: it admits 2^32-th roots of unity, so power-of-two transform sizes are straightforward, and it enables fast modular reduction on 128-bit products using the identity 2^64 ≡ 2^32 - 1 (mod p).

High-level plan
- Represent big mantissas as base-2 limbs (currently 32-bit limbs are convenient on GPU/CPU). Choose N = next power of two ≥ 2·L (L = limb count) for the convolution length.
- Forward NTT(A), NTT(B) mod p, pointwise multiply, inverse NTT, multiply by N^{-1} mod p, then perform carry propagation back to the chosen limb base.
- Use iterative radix-2 Cooley–Tukey with an explicit bit-reversal permutation (DIT). Twiddles (powers of a primitive root) are precomputed and cached.
- Butterflies and pointwise products operate in Montgomery form; 128-bit products are reduced via Montgomery multiplication (R = 2^64). A direct pseudo-Mersenne fold (lo + (hi << 32) − hi) exists but isn’t used on the hot path.

Notes and guardrails
- Single-prime NTT is attractive here because p fits in 64 bits and gives ample dynamic range; if/when larger bases or tighter bounds are desired, a multi-prime CRT variant is the next step.
- Power-of-two sizes only: that matches the current host prototype and simplifies CUDA mapping.
- GPU mapping: TBD.
- Carry fix-up remains outside the NTT and is done in base-2^k with linear-time passes; lazy (deferred) carries may help throughput.

Why this might beat Karatsuba on GPU
- Avoids deep recursion and its register pressure; most work is regular butterflies, which parallelize and schedule well.
- Pointwise multiplies dominate cost but are simple 64×64→128 with fast reduction; memory access is structured and coalesced.

If the CUDA path pans out, the NTT route should scale better across precisions while keeping occupancy higher than the recursive Karatsuba path.

References (NTT / GPU big-int)
- CGBN: CUDA Big-Num with Cooperative Groups — https://github.com/NVlabs/CGBN
- Number-theoretic transform — https://en.wikipedia.org/wiki/Number-theoretic_transform

## What is FractalShark?

A Mandelbrot Set renderer optimized for use with NVIDIA GPUs.

## Why FractalShark?

Because fractals and sharks are both cool, so why not? And it's relatively unique, so hopefully, it can be Google'd.

## What is the Mandelbrot Set?

The Mandelbrot set is defined as the set of complex numbers that do not diverge to infinity when iterated according to \( Z_{n+1} \leftarrow Z_n^2 + c \), with \( Z_0 \) initialized to 0. It's possible to visualize the set by assigning a color depending on the number of iterations necessary to determine whether the point was in the set. In FractalShark, points are left black to denote membership, and colored if they escape (diverge) to infinity after some number of iterations. Points determined to be in the set are necessarily approximate. Infinite iterations are necessary in the general case, but the software can simply use a large fixed number because it's "good enough."

## Why do we need another Mandelbrot Set renderer? Thousands of others already exist!

FractalShark includes a few innovations relative to most other renderers, most related to engineering rather than fundamentals.

First, FractalShark includes numerous NVIDIA CUDA implementations, demonstrating distinct strategies for rendering the Mandelbrot. These are cool in their own right, because I was not able to find existing examples of some of these approaches.

Second, and more importantly, FractalShark includes two different CUDA-based linear approximation implementations. Linear approximation is a relatively new technique for getting high performance visualizations, and no existing CUDA implementations appear to exist. These implementations are ported from [FractalZoomer](https://github.com/hrkalona/Fractal-Zoomer), and themselves originated from [Claude](https://mathr.co.uk/web/) and [Zhuoran](https://github.com/5E-324/Imagina).

Third, FractalShark includes a multi-threaded reference orbit calculation for a little better performance. Using three threads total, with two for squaring and one for coordinating, appears to offer superior performance, at least on an AMD-5950X. Calculating the reference orbit is the single-largest bottleneck to high magnification renders, so it's important to spend some time on that. This implementation beats all other known examples in performance assuming you have a modern multi-core CPU capable of executing AVX-2 instructions. More optimizations are possible here that are as-yet unexplored. This implementation also uses a custom memory allocator for an optional "perturbed perturbation" implementation, wherein intermediate resolution reference orbit entries are stored and re-used for subsequent zooms.

Another neat FractalShark-specific feature is an implementation of a "2x32" type combined with optional linear approximation and "float+exp." This implementation supports a pair of 32-bit floating point numbers + an exponent, to provide a combined ~48-bit mantissa without using the native 64-bit type. The benefit is significant performance improvements on consumer video cards, with nearly the same precision. This implementation is CUDA-only. On a CPU, you might as well use the native 64-bit type.

Finally, FractalShark supports reference orbit compression. This feature remains a work in progress, but it promises to reduce end-to-end memory (RAM) requirements, especially on high-period locations. Imagina first implemented reference compression for saving/loading reference orbits. FractalShark extends Zhuoran's idea to runtime/per-pixel rendering, by decompressing the reference orbit on the fly as needed. An example period-600,000,000 spot can see a memory reduction of multiple gigabytes, which can make the difference between being able to render it and not being able to render it.

## CUDA? What are the system requirements?

- **AVX-2 CPU**: used for reference orbit/MPIR library
- **CUDA-capable NVIDIA 900-series or newer** (~2016)
- **Try updating your NVIDIA driver** if you get a "cuda error 35" when you run it.
- **Windows**

## Where do I download it?

- Downloadable from here: [https://github.com/mattsaccount364/FractalShark](https://github.com/mattsaccount364/FractalShark)

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

## Where are the most interesting parts of the implementation in the code?

- All the CUDA kernels are in `render_gpu.cu`. The two most interesting are probably: `mandel_1xHDR_float_perturb_bla` and `mandel_1xHDR_float_perturb_lav2` which are the ones I've spent the most time on lately. For better performance at low zoom levels, you could look at `mandel_1x_float_perturb` which leaves out linear approximation and just does straight perturbation up to ~10e30, which corresponds with the 32-bit float exponent range.
- One fun thing you can try is running with LAv2 + "LA only". This approach actually works pretty well once Linear Approximation kicks in at deeper zooms - it gives you an idea of what the actual image should look like but is very fast, since it does no perturbation. The images it produces are not precise, and often leave out the fine detail - however, it's fun to play with when zooming in on a specific point.
- The `mandel_1x_float` is the classic 32-bit float mandelbrot and is screaming fast on a GPU. This one is optimized with fused multiply-add for fun even though it's kind of useless because you can barely zoom in before you get pixellation.
- The most interesting reference orbit calculation is at `AddPerturbationReferencePointMT3`. It includes a "bad" calculation which is used for the "scaled" CUDA kernels. The multithreaded approach handily beats the single-threaded implementation on my CPU in all scenarios.
- There are CPU renderers, but they were mostly to learn/debug more easily, and aren't optimized heavily. They're much easier to understand and reason about though.

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

## 2025-6-15 News

This page actually gets traffic occasionally, so I just wanted to post a short update.  Since last August, I've been working on a CUDA-based, high-precision reference orbit implementation.  The objective is to beat FractalShark's existing multithreaded reference-orbit performance at higher digit counts, at least if you have a decent card.  Scroll down to "[2025-6-15](#2025-6-15)" for the latest information on this subject.

Still fussing with it, with some delays because of vacation etc.  Having some issues with the optimized "add" implementation that does the 5-way add/subtract.  It's a fun project, but has ended up more complex than I'd expected.  The reference implementation is almost working the way I want.

Worst case I could dump it and fallback to a series of regular A+B adds/subtracts but I'm pretty determined to make the optimized approach work.  TBD if the performance actually pays off.  (Yes, I can hear you saying the Mandelbrot multiplies/squares dominate the cost, but it's bugging me and fun to play with).

## 2025-4-26

This high-precision arithmetic project is a lot of fun even if it's pretty ameteur-hour -- I know I'm leaving a lot of perf on the table yet.

Here's a brief update.  I'm happy enough with Karatsuba multiply now.  I've got 3x parallel multiplies working.  For Mandelbrot, that corresponds to x^2, y^2 and x*y.  Rather than doing an optimized squaring implementation, I'm just jamming everything into the same Karatsuba implementation, so that all the synchronization is re-used.

A few weeks ago I had CUDA floating point add working.  That's much easier of course, though carry propagation is interesting.  I tried a parallel prefix sum but the performance was a bit underwhelming in the average case, which is what I care about.  I instead implemented a more naive strategy that has better average case perf and linear worst-case performance, which I think I'm fine with for Mandelbrot.  I'm not using warp-level primitives and haven't hooked up shared memory on that, so it's horrid performance compared to simply doing it on the CPU but as a percentage of the total it's minor, because large multiplies are so costly.  I'm not that worried about Add at this point.

I'm currently hooking up a 5-way combined add/subtract that does A - B + C and D+ E in parallel to produce two outputs.  These inputs corresponds to X^2 - Y^2 + C_x and 2*X*Y + C_y.

Strategy-wise, the idea is to complete this multi-way add operator, and then we should be able to do a reference orbit using alternating 3-way multiply and 5-way add calls in CUDA.  It also needs periodicity detection and high-precision to float+exp conversion, which shouldn't be bad.  Maybe in a few months I'll have something working end-to-end.

I was also speculating about trying Schönhage-Strassen CUDA multiply, but that's a ways out.

## 2025-3

It's been a month so here's an update before I go on vacation.  I'm still focusing on multiply performance and correctness.  I have a pretty aggressive test framework set up now and am evaluating it with various number sizes and hardware allocations.  Supporting weird lengths makes it easier to apply more levels of Karatsuba recursion.

I've added optional debug-specific logic that calculates checksums of each intermediate step and outputs those as well.  The host calculates the same intermediate checksums using my reference CPU implementation and comparison of the two happens in the test framework.  This approach is a pretty handy way to debug this nightmarish stuff because it just compares these checksums and immediately identifies where the first discrepency arises in the CUDA calculation.  The discrepency points right at the bad chunk of code.  Getting this checksumming strategy to work reliably was a real pain but it's a lot easier to debug than just getting a result that says "wrong answer."

For additional validation, it's initializing all dynamically-allocated CUDA global memory with a 0xCDCDCDCD pattern, so if the implementation misses a byte, or overwrites something incorrectly, the checksum immediately captures it and makes it clear where the problem occurred.  This is not default CUDA behavior so I just put in a memset.  This approach also helps ensure that I have clear definitions for how many digits are being processed at each point in the calculation, since CUDA doesn't have nice std::vector or related containers.

One annoying thing I hit is slow compilation times.  It's using templates aggressively, so the kernel it spits out is optimized for a specific length of number.  That's OK in principle because we can just produce a series of kernels for different precisions but the downside is compiling a bunch of them takes quite a while and produces large code sizes.  It may make more sense to introduce more runtime variables and rely less on templates here but as it is this endeavor is mostly an academic exercise anyway, and I'm not expecting this thing to replace the existing CPU-based reference orbit calculations we have in the general sense.  But it'd be cool to get high performance in some meaningful range of scenarios anyway, hehe.

I'll probably try moving to 3x parallel multiplies soon as a step toward a reference orbit, because I want to check that this thing can still compile effectively with that change.  This kernel already requires a fair number of registers in order to perform (avoid spilling registers to memory) and that's a bit of a concern because if 3x parallel multiplies pushes it over the edge, performance will suffer.  There are various things I could do to decrease register usage of course, but all this stuff takes time.  Once 3x multiplies works, then adding the additions/subtracts for a reference orbit should be OK.  Those would take place after the multiplies so should have no adverse effect on register use.

After that I still have to deal with periodicity and truncating the high-precision values to float/exp, and all that will take more time.  This part may actually be rather costly perf-wise if I'm not careful because the naive approach is to serialize it with the rest of the calculation but that's a waste of hardware.

In a nutshell, this is a fun project and I'm having a blast, but it ended up bigger than I anticipated given how far I'm from the actual goal.  I'll keep grinding away at and we'll see where it can go.

Current best result (5950X MPIR vs RTX 4090), 50000 sequential multiplies of 7776-limb numbers, 128 blocks, 96 threads/block, uses shared memory and 162 registers (max 255):

Host iter time: 26051 ms
GPU iter time: 2757 ms

Edited from earlier, fussing with perf-related parameters.  I'm very happy with how it's looking now. 

## 2025-2 - What's going on with this native CUDA reference orbit calculation?

It's been a month so here's an update before I go on vacation.  I'm still focusing on multiply performance and correctness.  I have a pretty aggressive test framework set up now and am evaluating it with various number sizes and hardware allocations.  Supporting weird lengths makes it easier to apply more levels of Karatsuba recursion.

I've added optional debug-specific logic that calculates checksums of each intermediate step and outputs those as well.  The host calculates the same intermediate checksums using my reference CPU implementation and comparison of the two happens in the test framework.  This approach is a pretty handy way to debug this nightmarish stuff because it just compares these checksums and immediately identifies where the first discrepency arises in the CUDA calculation.  The discrepency points right at the bad chunk of code.  Getting this checksumming strategy to work reliably was a real pain but it's a lot easier to debug than just getting a result that says "wrong answer."

For additional validation, it's initializing all dynamically-allocated CUDA global memory with a 0xCDCDCDCD pattern, so if the implementation misses a byte, or overwrites something incorrectly, the checksum immediately captures it and makes it clear where the problem occurred.  This is not default CUDA behavior so I just put in a memset.  This approach also helps ensure that I have clear definitions for how many digits are being processed at each point in the calculation, since CUDA doesn't have nice std::vector or related containers.

One annoying thing I hit is slow compilation times.  It's using templates aggressively, so the kernel it spits out is optimized for a specific length of number.  That's OK in principle because we can just produce a series of kernels for different precisions but the downside is compiling a bunch of them takes quite a while and produces large code sizes.  It may make more sense to introduce more runtime variables and rely less on templates here but as it is this endeavor is mostly an academic exercise anyway, and I'm not expecting this thing to replace the existing CPU-based reference orbit calculations we have in the general sense.  But it'd be cool to get high performance in some meaningful range of scenarios anyway, hehe.

I'll probably try moving to 3x parallel multiplies soon as a step toward a reference orbit, because I want to check that this thing can still compile effectively with that change.  This kernel already requires a fair number of registers in order to perform (avoid spilling registers to memory) and that's a bit of a concern because if 3x parallel multiplies pushes it over the edge, performance will suffer.  There are various things I could do to decrease register usage of course, but all this stuff takes time.  Once 3x multiplies works, then adding the additions/subtracts for a reference orbit should be OK.  Those would take place after the multiplies so should have no adverse effect on register use.

After that I still have to deal with periodicity and truncating the high-precision values to float/exp, and all that will take more time.  This part may actually be rather costly perf-wise if I'm not careful because the naive approach is to serialize it with the rest of the calculation but that's a waste of hardware.

In a nutshell, this is a fun project and I'm having a blast, but it ended up bigger than I anticipated given how far I'm from the actual goal.  I'll keep grinding away at and we'll see where it can go.

## 2025-1 - What's going on with this native CUDA reference orbit calculation?

The repository ("TestCuda") has a new Karatsuba, high-precision, floating point multiply implementation working on my GPU and results are showing the CPU take 3-4x longer (MPIR/AVX2) than the GPU on sequential multiplies of random numbers.  That's a key point - sequential multiplies, so the result is applicable to e.g. a reference orbit calculation.  I'm really happy with this result, because there's leftover hardware on the GPU that could be used to run a couple of these in parallel (e.g. two squares and a multiplication or three squares for Mandelbrot).

Getting high-throughput high-precision on-GPU is already more-or-less solved:  Nvidia already provides a library for it (see related work below).  But getting sequential to work decently at sizes that are still interesting (e.g. ones we might actually try on the Mandelbrot) is not as widely investigated, which is why I've been dwelling on this for a while and still have only mediocre results :p

A variety of caveats currently:
- I'm comparing a 5950X vs RTX 4090, which clearly affects the relative numbers.
- The approach requires CUDA cooperative groups, which I think is RTX 2xxx and later, so fairly recent cards.
- The size I'm getting the best result at currently is relatively large: 8192 32-bit limbs, which is pretty big.  It still beats the CPU down to 2048 limbs though (CPU takes 1.6x longer here) and at that size there is a bunch of unused hardware on the GPU so it should be possible to do the three multiplies for Mandelbrot in parallel.

Anyhow, I wanted to post it because this is a pretty complex investigation and I expect to spend some more time on it because it's been fun to look at.

Here are some bullet points on the approach:
- Uses CUDA cooperative groups, so we should be able to do a reference orbit with a single kernel invocation.
- 32-bit limbs, 128-bit intermediate results (2x64 integers) because of intermediate carries.  Really it's just 64 bits + plus a few more.  This approach is likely not super-efficient but it's where we're at.
- Stores the input mantissa in the chip "shared memory".  For 8192 limbs, that's 8192 * 4 bytes * 2 numbers to multiply = 64KB, and then stores 2 * 16KB extra for an intermediate Karatsuba result, for 96KB total.  This piece is negotiable and I could bring down/eliminate the shared memory requirement depending on how things progress.
- One GPU thread per input limb, or two output limbs per thread.  It does a full 16384 limb output in this example and then truncates/shifts it.
- The GPU floating point format has a separate integer exponent, which is overkill for Mandelbrot but I figured I'd keep it for now because it's not a performance problem.  It also keeps a sign separately.
- The application I'm using to test this has a bunch of cases to verify that it's producing correct results.  It generates pseudo-random numbers with many 0xFFF... limbs, zero-limbs, and related, to force carries/borrows.  The test program compares all results against my own Karatsuba CPU-based implementation I can use as a reference, and more importantly, the MPIR implementation (mpf_mul) for correctness.
- I've got MPIR to GPU and GPU to MPIR conversion capability, so it's easy to translate formats as needed.
- The GPU implementation gets its best performance when it doesn't recurse, and instead switches straight to convolutions on the sub problems.  Recursing is still getting me slightly worse performance and I think I know why but haven't worked out how to fix it.

Here is some related work:
A Study of High Performance Multiple Precision Arithmetic on Graphics Processing Units
Niall Emmart
https://scholarworks.umass.edu/server/api/core/bitstreams/3cd92dff-0d11-4976-8556-2dc34d0abb27/content

- This recent dissertation on the subject is the most relevant.  I've not read the whole thing because I wanted to learn more about it all myself first, but it's clear to me this topic is quite deep.

Missing a trick: Karatsuba variations
Michael Scott
https://eprint.iacr.org/2015/1247

MFFT: A GPU Accelerated Highly Efficient Mixed-Precision Large-Scale FFT Framework
https://dl.acm.org/doi/full/10.1145/3605148

Karatsuba Multiplication in GMP:
https://gmplib.org/manual/Karatsuba-Multiplication

Karatsuba:
https://en.wikipedia.org/wiki/Karatsuba_algorithm

Toom Cook:
https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication

Schönhage-Strassen algorithm
https://en.wikipedia.org/wiki/Sch%C3%B6nhage%E2%80%93Strassen_algorithm

CGBN: CGBN: CUDA Accelerated Multiple Precision Arithmetic (Big Num) using Cooperative Groups
https://github.com/NVlabs/CGBN

All the code is here / GPL etc but it's just an experiment and not integrated with the application:
https://github.com/mattsaccount364/FractalShark

It'll all end up in FractalShark eventually assuming I can get something end-to-end working, and at this point I believe I can, but this is very slow-going yet.  Anyway, it's a fun area and I'll probably continuing dinking with it, so there probably won't be much new on FractalShark proper until I can get this behemoth under better control.  And we'll see if it works out - lots of remaining details to resolve and it may not work decently when combined into a full reference orbit.  Overall though I'm really happy with where it's at.  Happy to answer questions etc.

## Closing

If you're bored and want to try yet another Mandelbrot set renderer, give it a go. You can download releases from the GitHub page. YMMV, caveat emptor, don't bother running it unless you meet the system requirements because it'll probably just spew some opaque error and exit. In terms of time investment, I'll probably only spend a couple of hours on weekends fussing with it so don't expect dramatic rewrites.

## History

- **2001**: The first version of FractalShark I did as a freshman in college. I added a basic network/distributed strategy to speed it up. It split the work across machines. This feature is now broken and won't be revived.
- **2008**: I did a distributed version that ran on the University of Wisconsin's "Condor" platform. That was pretty cool, and allowed for fast creation of zoom movies.
- **2017**: I resurrected it, and added high precision, but no perturbation or anything else. At that point, it was theoretically capable of rendering very deep images, but it was so slow as to be largely useless.
- **2023-2024**: I bought this new video card, and wanted to play with it, so ended up learning about CUDA and all these clever algorithmic approaches you all have found out to speed this up. So here we are.

## Build instructions

1. `git clone https://github.com/mattsaccount364/FractalShark.git`
2. `cd FractalShark`
3. `git clone https://github.com/BrianGladman/mpir.git`
4. Open `mpir\msvc\vs22\mpir.sln`
5. Rebuild everything. Note, you probably want YASM installed at: `C:\Program Files\yasm\*`. This may take a little while.
6. The distributed version of FractalShark uses the static lib_mpir_skylake_avx incarnation of MPIR.
7. Install Nvidia Nsight Compute SDK. The distributed version of FractalShark uses 12.2
8. At this point, you should be able to build all the FractalShark projects. The one called FractalShark is the most interesting. FractalTray basically works. The rest are effectively dead code.

Have fun, hopefully :)