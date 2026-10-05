#pragma once

#include "PngCompressionKernels.cuh"

#include <cuda_runtime.h>

#include <limits>

namespace FractalShark::Png::Detail::Huffman {

// Boundary package merge constructs optimal length-limited codes in a single lane.
// Sorting, node chains, and the explicit traversal stack all reside in device workspace.
inline constexpr unsigned LiteralSymbols = Format::LiteralLengthSymbols;
inline constexpr unsigned DistanceSymbols = Format::DistanceSymbols;
inline constexpr unsigned TableSymbols = LiteralSymbols + DistanceSymbols;
inline constexpr unsigned MaximumBits = Format::MaximumCodeBits;
// Two boundary chains per level, each at most MaximumBits nodes deep, plus allocation
// slack for replacing boundaries. Unreachable nodes are recycled when the pool fills.
inline constexpr unsigned NodeCapacity = 2 * MaximumBits * (MaximumBits + 1);
inline constexpr unsigned NoNode = std::numeric_limits<unsigned>::max();

static_assert(NodeCapacity >= TableSymbols); // The free list also holds the combined lengths.

struct Node {
    uint64_t m_Weight;
    unsigned m_Count;  // Number of sorted leaves preceding this boundary.
    unsigned m_Tail;   // Lower-level boundary chain for a package, or NoNode for a leaf.
    unsigned m_Marked; // Reachability scratch used only when recycling the node pool.
};

struct Leaf {
    uint64_t m_Weight;
    unsigned m_Symbol;
};

struct Frame {
    unsigned m_Level;
    unsigned m_Stage; // 0 selects a boundary; 1 and 2 resume after the two child advances.
};

// These arrays live exclusively in cudaMalloc workspace, never in a CUDA local object.
struct Workspace {
    Node m_Nodes[NodeCapacity];
    unsigned m_Free[NodeCapacity];
    Leaf m_Leaves[LiteralSymbols];
    Leaf m_Merge[LiteralSymbols];
    unsigned m_Heads0[MaximumBits];
    unsigned m_Heads1[MaximumBits];
    Frame m_Frames[MaximumBits];
    unsigned m_Counts[MaximumBits + 1];
    unsigned m_NextCodes[MaximumBits + 1];
    unsigned m_FreeCount;
    unsigned m_NextFree;
};

struct Region {
    unsigned m_Frequencies[TableSymbols];
    unsigned m_Lengths[TableSymbols];
    unsigned m_Codes[TableSymbols];
    CodeLengthRun m_Runs[TableSymbols];
    unsigned m_HeaderFrequencies[Format::CodeLengthSymbols];
    unsigned m_HeaderLengths[Format::CodeLengthSymbols];
    unsigned m_HeaderCodes[Format::CodeLengthSymbols];
    Workspace m_Workspace;
    uint64_t m_ExtraBits; // Match length/distance extra bits, excluding Huffman codes.
    unsigned m_LiteralCount;
    unsigned m_DistanceCount;
    unsigned m_HeaderCount;
    unsigned m_RunCount;
    unsigned m_Bytes;      // Complete candidate size, including the trailing alignment block.
    unsigned m_UseDynamic; // Published to the warp after the lane-zero cost comparison.
};

__device__ inline unsigned
CreateNode(Workspace *workspace, uint64_t weight, unsigned count, unsigned tail, unsigned maximumBits)
{
    if (workspace->m_NextFree == workspace->m_FreeCount) {
        // Only chains reachable from the two current boundaries at each level are live.
        // Mark those chains before rebuilding the free list; no device allocation or recursion.
        for (unsigned index = 0; index < NodeCapacity; ++index) {
            workspace->m_Nodes[index].m_Marked = 0;
        }
        for (unsigned level = 0; level < maximumBits; ++level) {
            for (unsigned head = 0; head < 2; ++head) {
                unsigned node = head == 0 ? workspace->m_Heads0[level] : workspace->m_Heads1[level];
                while (node != NoNode && workspace->m_Nodes[node].m_Marked == 0) {
                    workspace->m_Nodes[node].m_Marked = 1;
                    node = workspace->m_Nodes[node].m_Tail;
                }
            }
        }
        workspace->m_FreeCount = 0;
        for (unsigned index = 0; index < NodeCapacity; ++index) {
            if (workspace->m_Nodes[index].m_Marked == 0) {
                workspace->m_Free[workspace->m_FreeCount++] = index;
            }
        }
        workspace->m_NextFree = 0;
    }
    const unsigned index = workspace->m_Free[workspace->m_NextFree++];
    workspace->m_Nodes[index].m_Weight = weight;
    workspace->m_Nodes[index].m_Count = count;
    workspace->m_Nodes[index].m_Tail = tail;
    return index;
}

__device__ inline void
BoundaryPackage(Workspace *workspace, unsigned present, unsigned maximumBits, unsigned number)
{
    // Advancing a level selects the next leaf or a package made from two lower boundaries.
    // Frames replace recursion, and m_Stage ensures both lower advances run before returning.
    unsigned depth = 1;
    workspace->m_Frames[0] = Frame{maximumBits - 1, 0};
    while (depth != 0) {
        Frame *frame = workspace->m_Frames + depth - 1;
        const unsigned level = frame->m_Level;
        if (frame->m_Stage == 0) {
            const unsigned previous = workspace->m_Heads1[level];
            const unsigned count = workspace->m_Nodes[previous].m_Count;
            if (level == 0) {
                if (count < present) {
                    workspace->m_Heads0[level] = previous;
                    workspace->m_Heads1[level] = CreateNode(
                        workspace, workspace->m_Leaves[count].m_Weight, count + 1, NoNode, maximumBits);
                }
                --depth;
                continue;
            }
            const uint64_t sum = workspace->m_Nodes[workspace->m_Heads0[level - 1]].m_Weight +
                                 workspace->m_Nodes[workspace->m_Heads1[level - 1]].m_Weight;
            workspace->m_Heads0[level] = previous;
            if (count < present && sum > workspace->m_Leaves[count].m_Weight) {
                workspace->m_Heads1[level] = CreateNode(workspace,
                                                        workspace->m_Leaves[count].m_Weight,
                                                        count + 1,
                                                        workspace->m_Nodes[previous].m_Tail,
                                                        maximumBits);
                --depth;
                continue;
            }
            workspace->m_Heads1[level] =
                CreateNode(workspace, sum, count, workspace->m_Heads1[level - 1], maximumBits);
            if (number + 1 >= 2 * present - 2) {
                --depth;
                continue;
            }
            frame->m_Stage = 1;
            workspace->m_Frames[depth++] = Frame{level - 1, 0};
        } else if (frame->m_Stage == 1) {
            frame->m_Stage = 2;
            workspace->m_Frames[depth++] = Frame{level - 1, 0};
        } else {
            --depth;
        }
    }
}

__device__ inline void
BuildLengths(const unsigned *frequencies,
             unsigned symbols,
             unsigned maximumBits,
             Workspace *workspace,
             unsigned *lengths)
{
    unsigned present = 0;
    for (unsigned symbol = 0; symbol < symbols; ++symbol) {
        lengths[symbol] = 0;
        if (frequencies[symbol] != 0) {
            workspace->m_Leaves[present++] = Leaf{frequencies[symbol], symbol};
        }
    }
    // Two one-bit codes also support decoders that reject a single-symbol tree.
    if (present < 2) {
        const unsigned symbol = present == 0 ? 0 : workspace->m_Leaves[0].m_Symbol;
        lengths[symbol] = 1;
        lengths[symbol == 0 ? 1 : 0] = 1;
        return;
    }
    // Leaves enter in symbol order. Stable merging by weight preserves symbol order on
    // frequency ties, matching the CPU oracle's deterministic package-merge input.
    Leaf *source = workspace->m_Leaves;
    Leaf *destination = workspace->m_Merge;
    for (unsigned width = 1; width < present; width *= 2) {
        for (unsigned start = 0; start < present; start += 2 * width) {
            const unsigned middle = start + width < present ? start + width : present;
            const unsigned end = start + 2 * width < present ? start + 2 * width : present;
            unsigned left = start;
            unsigned right = middle;
            for (unsigned index = start; index < end; ++index) {
                if (left < middle && (right == end || source[left].m_Weight <= source[right].m_Weight)) {
                    destination[index] = source[left++];
                } else {
                    destination[index] = source[right++];
                }
            }
        }
        Leaf *previous = source;
        source = destination;
        destination = previous;
    }
    if (source != workspace->m_Leaves) {
        for (unsigned index = 0; index < present; ++index) {
            workspace->m_Leaves[index] = source[index];
        }
    }
    workspace->m_FreeCount = NodeCapacity;
    workspace->m_NextFree = 0;
    for (unsigned index = 0; index < NodeCapacity; ++index) {
        workspace->m_Free[index] = index;
    }
    // Seed directly: the first two allocations cannot exhaust the node pool.
    CreateNode(workspace, workspace->m_Leaves[0].m_Weight, 1, NoNode, maximumBits);
    CreateNode(workspace, workspace->m_Leaves[1].m_Weight, 2, NoNode, maximumBits);
    for (unsigned level = 0; level < maximumBits; ++level) {
        workspace->m_Heads0[level] = 0;
        workspace->m_Heads1[level] = 1;
    }
    for (unsigned number = 2; number < 2 * present - 2; ++number) {
        BoundaryPackage(workspace, present, maximumBits, number);
    }
    // A complete binary code needs 2*present-2 boundary selections (two were seeded).
    // Each selected boundary increments the lengths of its prefix of sorted leaves.
    unsigned node = workspace->m_Heads1[maximumBits - 1];
    while (node != NoNode) {
        for (unsigned index = 0; index < workspace->m_Nodes[node].m_Count; ++index) {
            ++lengths[workspace->m_Leaves[index].m_Symbol];
        }
        node = workspace->m_Nodes[node].m_Tail;
    }
}

__device__ inline void
BuildCodes(const unsigned *lengths, unsigned symbols, Workspace *workspace, unsigned *codes)
{
    // Canonical codes are assigned by length, then symbol. Reverse each code because
    // BitWriter emits low bits first while DEFLATE Huffman codes are specified MSB first.
    for (unsigned bits = 0; bits <= MaximumBits; ++bits) {
        workspace->m_Counts[bits] = 0;
        workspace->m_NextCodes[bits] = 0;
    }
    for (unsigned symbol = 0; symbol < symbols; ++symbol) {
        if (lengths[symbol] != 0) {
            ++workspace->m_Counts[lengths[symbol]];
        }
    }
    unsigned code = 0;
    for (unsigned bits = 1; bits <= MaximumBits; ++bits) {
        code = (code + workspace->m_Counts[bits - 1]) << 1;
        workspace->m_NextCodes[bits] = code;
    }
    for (unsigned symbol = 0; symbol < symbols; ++symbol) {
        const unsigned bits = lengths[symbol];
        codes[symbol] =
            bits == 0 ? 0 : __brev(workspace->m_NextCodes[bits]++) >> (Format::WordBits - bits);
    }
}

__device__ inline unsigned
EncodeRuns(const unsigned *lengths, unsigned count, CodeLengthRun *runs)
{
    // Emit a nonzero length once before repeating it. Zero runs have their own short
    // and long symbols, and may cross the literal/distance alphabet boundary.
    unsigned output = 0;
    unsigned position = 0;
    while (position < count) {
        const unsigned value = lengths[position];
        unsigned repeated = 1;
        while (position + repeated < count && lengths[position + repeated] == value) {
            ++repeated;
        }
        position += repeated;
        if (value != 0) {
            runs[output++] = CodeLengthRun{value, 0, 0};
            --repeated;
        }
        while (repeated != 0) {
            if (value == 0 && repeated >= Format::RepeatZeroLongMinimum) {
                const unsigned consumed =
                    repeated < Format::RepeatZeroLongMaximum ? repeated : Format::RepeatZeroLongMaximum;
                runs[output++] = CodeLengthRun{Format::RepeatZeroLongSymbol,
                                               consumed - Format::RepeatZeroLongMinimum,
                                               Format::RepeatZeroLongExtraBits};
                repeated -= consumed;
            } else if (value == 0 && repeated >= Format::RepeatZeroShortMinimum) {
                const unsigned consumed = repeated < Format::RepeatZeroShortMaximum
                                              ? repeated
                                              : Format::RepeatZeroShortMaximum;
                runs[output++] = CodeLengthRun{Format::RepeatZeroShortSymbol,
                                               consumed - Format::RepeatZeroShortMinimum,
                                               Format::RepeatZeroShortExtraBits};
                repeated -= consumed;
            } else if (value != 0 && repeated >= Format::RepeatPreviousMinimum) {
                const unsigned consumed =
                    repeated < Format::RepeatPreviousMaximum ? repeated : Format::RepeatPreviousMaximum;
                runs[output++] = CodeLengthRun{Format::RepeatPreviousSymbol,
                                               consumed - Format::RepeatPreviousMinimum,
                                               Format::RepeatPreviousExtraBits};
                repeated -= consumed;
            } else {
                runs[output++] = CodeLengthRun{value, 0, 0};
                --repeated;
            }
        }
    }
    return output;
}

__device__ inline unsigned
HeaderOrder(unsigned index)
{
    // RFC 1951's wire permutation: 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15.
    // A scalar switch keeps the mandated ordering explicit without a CUDA local array.
    switch (index) {
        case 0:
            return Format::RepeatPreviousSymbol;
        case 1:
            return Format::RepeatZeroShortSymbol;
        case 2:
            return Format::RepeatZeroLongSymbol;
        case 3:
            return 0;
        case 4:
            return 8;
        case 5:
            return 7;
        case 6:
            return 9;
        case 7:
            return 6;
        case 8:
            return 10;
        case 9:
            return 5;
        case 10:
            return 11;
        case 11:
            return 4;
        case 12:
            return 12;
        case 13:
            return 3;
        case 14:
            return 13;
        case 15:
            return 2;
        case 16:
            return 14;
        case 17:
            return 1;
        default:
            return Format::MaximumCodeBits;
    }
}

__device__ inline void
BuildRegion(Region *region, uint64_t baselineBytes)
{
    BuildLengths(
        region->m_Frequencies, LiteralSymbols, MaximumBits, &region->m_Workspace, region->m_Lengths);
    BuildLengths(region->m_Frequencies + LiteralSymbols,
                 DistanceSymbols,
                 MaximumBits,
                 &region->m_Workspace,
                 region->m_Lengths + LiteralSymbols);
    BuildCodes(region->m_Lengths, LiteralSymbols, &region->m_Workspace, region->m_Codes);
    BuildCodes(region->m_Lengths + LiteralSymbols,
               DistanceSymbols,
               &region->m_Workspace,
               region->m_Codes + LiteralSymbols);
    uint64_t dataBits = region->m_ExtraBits;
    for (unsigned symbol = 0; symbol < TableSymbols; ++symbol) {
        dataBits += static_cast<uint64_t>(region->m_Frequencies[symbol]) * region->m_Lengths[symbol];
    }
    region->m_UseDynamic = 0;
    // dataBits already includes the end-of-block frequency. Ignore header runs to get
    // a lower bound: if even that cannot win, skip header construction and parser replay.
    if (Format::CompressedRegionBytes(dataBits + Format::MinimumDynamicHeaderBits) >= baselineBytes) {
        return;
    }
    region->m_LiteralCount = LiteralSymbols;
    while (region->m_LiteralCount > Format::MinimumLiteralCount &&
           region->m_Lengths[region->m_LiteralCount - 1] == 0) {
        --region->m_LiteralCount;
    }
    region->m_DistanceCount = DistanceSymbols;
    while (region->m_DistanceCount > Format::MinimumDistanceCount &&
           region->m_Lengths[LiteralSymbols + region->m_DistanceCount - 1] == 0) {
        --region->m_DistanceCount;
    }
    // The code-length alphabets form one sequence, including runs across their boundary.
    // Both trees are finished, so their free-list scratch may temporarily hold this
    // sequence. EncodeRuns consumes it before the header tree reinitializes the workspace.
    unsigned *sequence = region->m_Workspace.m_Free;
    for (unsigned symbol = 0; symbol < region->m_LiteralCount; ++symbol) {
        sequence[symbol] = region->m_Lengths[symbol];
    }
    for (unsigned symbol = 0; symbol < region->m_DistanceCount; ++symbol) {
        sequence[region->m_LiteralCount + symbol] = region->m_Lengths[LiteralSymbols + symbol];
    }
    region->m_RunCount =
        EncodeRuns(sequence, region->m_LiteralCount + region->m_DistanceCount, region->m_Runs);
    for (unsigned symbol = 0; symbol < Format::CodeLengthSymbols; ++symbol) {
        region->m_HeaderFrequencies[symbol] = 0;
    }
    for (unsigned index = 0; index < region->m_RunCount; ++index) {
        ++region->m_HeaderFrequencies[region->m_Runs[index].m_Symbol];
    }
    BuildLengths(region->m_HeaderFrequencies,
                 Format::CodeLengthSymbols,
                 Format::MaximumHeaderCodeBits,
                 &region->m_Workspace,
                 region->m_HeaderLengths);
    BuildCodes(
        region->m_HeaderLengths, Format::CodeLengthSymbols, &region->m_Workspace, region->m_HeaderCodes);
    region->m_HeaderCount = Format::CodeLengthSymbols;
    while (region->m_HeaderCount > Format::MinimumHeaderCount &&
           region->m_HeaderLengths[HeaderOrder(region->m_HeaderCount - 1)] == 0) {
        --region->m_HeaderCount;
    }
    uint64_t bits =
        Format::DynamicHeaderPrefixBits + Format::HeaderLengthBits * region->m_HeaderCount + dataBits;
    for (unsigned index = 0; index < region->m_RunCount; ++index) {
        const CodeLengthRun run = region->m_Runs[index];
        bits += region->m_HeaderLengths[run.m_Symbol] + run.m_ExtraBits;
    }
    region->m_Bytes = static_cast<unsigned>(Format::CompressedRegionBytes(bits));
    // Dynamic must strictly improve the fixed/stored candidate; ties retain its bytes.
    region->m_UseDynamic = region->m_Bytes < baselineBytes;
}

template <class Writer>
__device__ inline void
WriteHeader(Writer &writer, const Region *region)
{
    writer.BlockHeader(Format::BlockType::Dynamic, false);
    writer.Write(region->m_LiteralCount - Format::MinimumLiteralCount, Format::LiteralCountBits);
    writer.Write(region->m_DistanceCount - Format::MinimumDistanceCount, Format::DistanceCountBits);
    writer.Write(region->m_HeaderCount - Format::MinimumHeaderCount, Format::HeaderCountBits);
    for (unsigned index = 0; index < region->m_HeaderCount; ++index) {
        writer.Write(region->m_HeaderLengths[HeaderOrder(index)], Format::HeaderLengthBits);
    }
    for (unsigned index = 0; index < region->m_RunCount; ++index) {
        const CodeLengthRun run = region->m_Runs[index];
        writer.Write(region->m_HeaderCodes[run.m_Symbol], region->m_HeaderLengths[run.m_Symbol]);
        writer.Write(run.m_Extra, run.m_ExtraBits);
    }
}

} // namespace FractalShark::Png::Detail::Huffman
