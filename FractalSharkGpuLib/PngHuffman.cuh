#pragma once

#include "PngCompressionKernels.cuh"

#include <cuda_runtime.h>

namespace FractalShark::Png::Detail::Huffman {

inline constexpr unsigned LiteralSymbols = 286;
inline constexpr unsigned DistanceSymbols = 30;
inline constexpr unsigned TableSymbols = LiteralSymbols + DistanceSymbols;
inline constexpr unsigned MaximumBits = 15;
inline constexpr unsigned NodeCapacity = 2 * MaximumBits * (MaximumBits + 1);
inline constexpr unsigned NoNode = 0xffffffffu;

struct Node {
    uint64_t m_Weight;
    unsigned m_Count;
    unsigned m_Tail;
    unsigned m_Marked;
};

struct Leaf {
    uint64_t m_Weight;
    unsigned m_Symbol;
};

struct Frame {
    unsigned m_Level;
    unsigned m_Stage;
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
    unsigned m_HeaderFrequencies[19];
    unsigned m_HeaderLengths[19];
    unsigned m_HeaderCodes[19];
    Workspace m_Workspace;
    uint64_t m_ExtraBits;
    unsigned m_LiteralCount;
    unsigned m_DistanceCount;
    unsigned m_HeaderCount;
    unsigned m_RunCount;
    unsigned m_Bytes;
    unsigned m_UseDynamic;
};

__device__ inline unsigned
CreateNode(Workspace *workspace, uint64_t weight, unsigned count, unsigned tail, unsigned maximumBits)
{
    if (workspace->m_NextFree == workspace->m_FreeCount) {
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
        codes[symbol] = bits == 0 ? 0 : __brev(workspace->m_NextCodes[bits]++) >> (32 - bits);
    }
}

__device__ inline unsigned
EncodeRuns(const unsigned *lengths, unsigned count, CodeLengthRun *runs)
{
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
            if (value == 0 && repeated >= 11) {
                const unsigned consumed = repeated < 138 ? repeated : 138;
                runs[output++] = CodeLengthRun{18, consumed - 11, 7};
                repeated -= consumed;
            } else if (value == 0 && repeated >= 3) {
                const unsigned consumed = repeated < 10 ? repeated : 10;
                runs[output++] = CodeLengthRun{17, consumed - 3, 3};
                repeated -= consumed;
            } else if (value != 0 && repeated >= 3) {
                const unsigned consumed = repeated < 6 ? repeated : 6;
                runs[output++] = CodeLengthRun{16, consumed - 3, 2};
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
    switch (index) {
        case 0:
            return 16;
        case 1:
            return 17;
        case 2:
            return 18;
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
            return 15;
    }
}

__device__ inline void
BuildRegion(Region *region, uint64_t baselineBytes)
{
    BuildLengths(region->m_Frequencies, LiteralSymbols, 15, &region->m_Workspace, region->m_Lengths);
    BuildLengths(region->m_Frequencies + LiteralSymbols,
                 DistanceSymbols,
                 15,
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
    // Even a minimum header cannot win: skip header construction and replay.
    if ((dataBits + 29 + 3 + 7) / 8 + 4 >= baselineBytes) {
        return;
    }
    region->m_LiteralCount = LiteralSymbols;
    while (region->m_LiteralCount > 257 && region->m_Lengths[region->m_LiteralCount - 1] == 0) {
        --region->m_LiteralCount;
    }
    region->m_DistanceCount = DistanceSymbols;
    while (region->m_DistanceCount > 1 &&
           region->m_Lengths[LiteralSymbols + region->m_DistanceCount - 1] == 0) {
        --region->m_DistanceCount;
    }
    // The code-length alphabets form one sequence, including runs across their boundary.
    unsigned *sequence = region->m_Workspace.m_Free;
    for (unsigned symbol = 0; symbol < region->m_LiteralCount; ++symbol) {
        sequence[symbol] = region->m_Lengths[symbol];
    }
    for (unsigned symbol = 0; symbol < region->m_DistanceCount; ++symbol) {
        sequence[region->m_LiteralCount + symbol] = region->m_Lengths[LiteralSymbols + symbol];
    }
    region->m_RunCount =
        EncodeRuns(sequence, region->m_LiteralCount + region->m_DistanceCount, region->m_Runs);
    for (unsigned symbol = 0; symbol < 19; ++symbol) {
        region->m_HeaderFrequencies[symbol] = 0;
    }
    for (unsigned index = 0; index < region->m_RunCount; ++index) {
        ++region->m_HeaderFrequencies[region->m_Runs[index].m_Symbol];
    }
    BuildLengths(region->m_HeaderFrequencies, 19, 7, &region->m_Workspace, region->m_HeaderLengths);
    BuildCodes(region->m_HeaderLengths, 19, &region->m_Workspace, region->m_HeaderCodes);
    region->m_HeaderCount = 19;
    while (region->m_HeaderCount > 4 &&
           region->m_HeaderLengths[HeaderOrder(region->m_HeaderCount - 1)] == 0) {
        --region->m_HeaderCount;
    }
    uint64_t bits = 17 + 3 * region->m_HeaderCount + dataBits;
    for (unsigned index = 0; index < region->m_RunCount; ++index) {
        const CodeLengthRun run = region->m_Runs[index];
        bits += region->m_HeaderLengths[run.m_Symbol] + run.m_ExtraBits;
    }
    region->m_Bytes = static_cast<unsigned>((bits + 3 + 7) / 8 + 4);
    region->m_UseDynamic = region->m_Bytes < baselineBytes;
}

template <class Writer>
__device__ inline void
WriteHeader(Writer &writer, const Region *region)
{
    writer.Write(4, 3); // BFINAL=0, BTYPE=10.
    writer.Write(region->m_LiteralCount - 257, 5);
    writer.Write(region->m_DistanceCount - 1, 5);
    writer.Write(region->m_HeaderCount - 4, 4);
    for (unsigned index = 0; index < region->m_HeaderCount; ++index) {
        writer.Write(region->m_HeaderLengths[HeaderOrder(index)], 3);
    }
    for (unsigned index = 0; index < region->m_RunCount; ++index) {
        const CodeLengthRun run = region->m_Runs[index];
        writer.Write(region->m_HeaderCodes[run.m_Symbol], region->m_HeaderLengths[run.m_Symbol]);
        writer.Write(run.m_Extra, run.m_ExtraBits);
    }
}

} // namespace FractalShark::Png::Detail::Huffman
