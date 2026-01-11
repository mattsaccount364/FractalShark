#include "..\DbgHeap.h"
#include "include/llist.h"
#include "include/HeapPanic.h"

#include <exception>
#include <type_traits>

static inline void
mark_in_bin(node_t *n, uint64_t idx)
{
    n->in_bin = idx;
    ++n->in_bin_gen;
}

static inline void
mark_not_in_bin(node_t *n)
{
    n->in_bin = node_t::NotInBin;
    ++n->in_bin_gen;
}


bool
has_cycle(node_t *head)
{
    node_t *slow = head;
    node_t *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast)
            return true;
    }
    return false;
}

static constexpr uintptr_t kPoisonNext = 0xDEADDEADDEADDEADull;
static constexpr uintptr_t kPoisonPrev = 0xBAADF00DBAADF00Dull;

static inline void
poison_links(node_t *n)
{
    n->next = reinterpret_cast<node_t *>(kPoisonNext);
    n->prev = reinterpret_cast<node_t *>(kPoisonPrev);
}

static inline bool
is_poison(node_t *p)
{
    auto v = reinterpret_cast<uintptr_t>(p);
    return v == kPoisonNext || v == kPoisonPrev;
}

static void
validate_bin(bin_t *bin)
{
    node_t *head = bin->head;

    // Empty list is valid
    if (!head)
        return;

    // Head must have prev == NULL (strong invariant)
    if (head->prev != nullptr) {
        HeapPanic("Bin head prev is not null");
    }

    // Detect cycles early (and avoid infinite loops below)
    if (has_cycle(head)) {
        HeapPanic("Cycle detected in bin");
    }

    // Basic link invariants + self-loop checks
    // Also add a hard cap so you can't hang even without has_cycle
    constexpr int kMaxSteps = 1 << 20; // large, but finite
    node_t *cur = head;
    node_t *prev = nullptr;

    for (int steps = 0; cur != nullptr; ++steps) {
        if (steps > kMaxSteps)
            HeapPanic("Max steps exceeded in bin validation");

        if (cur == cur->next)
            HeapPanic("Self-link detected in bin");
        if (cur == cur->prev)
            HeapPanic("Self-link detected in bin");

        if (cur->prev != prev)
            HeapPanic("Invalid node linkage in bin");

        if (cur->next) {
            if (cur->next->prev != cur)
                HeapPanic("Invalid node linkage in bin");
        }

        prev = cur;
        cur = cur->next;
    }
}

// Guarded pointer set that trips on self-linking immediately
static inline void
set_next(node_t *a, node_t *b)
{
    if (a == b)
        HeapPanic("Self-link detected in bin");
    a->next = b;
}
static inline void
set_prev(node_t *a, node_t *b)
{
    if (a == b)
        HeapPanic("Self-link detected in bin");
    a->prev = b;
}

static inline void
assert_not_linked(node_t *n, const char *msg)
{
    HEAP_ASSERT(n->in_bin == node_t::NotInBin, msg);
    HEAP_ASSERT(n->next == nullptr, "node unexpectedly has next");
    HEAP_ASSERT(n->prev == nullptr, "node unexpectedly has prev");
}

void
add_node(bin_t *bin, node_t *node, uint64_t in_bin)
{
    validate_bin(bin);

    HEAP_ASSERT(node != nullptr, "add_node: node null");
    HEAP_ASSERT(node->hole == 1, "add_node: node not free");
    HEAP_ASSERT(in_bin < HEAP_BIN_COUNT, "add_node: invalid bin index");

    // Must not already be in any bin.
    assert_not_linked(node, "add_node: node already linked");

    node->next = nullptr;
    node->prev = nullptr;

    if (bin->head == nullptr) {
        bin->head = node;

        // MUST stamp even on empty-list insertion.
        mark_in_bin(node, in_bin);

        validate_bin(bin);
        return;
    }

    node_t *current = bin->head;
    node_t *previous = nullptr;

    while (current && current->size <= node->size) {
        previous = current;
        current = current->next;
    }

    if (!current) {
        previous->next = node;
        node->prev = previous;
    } else if (previous) {
        node->next = current;
        node->prev = previous;
        previous->next = node;
        current->prev = node;
    } else {
        node->next = bin->head;
        bin->head->prev = node;
        bin->head = node;
    }

    mark_in_bin(node, in_bin);
    validate_bin(bin);
}



void
remove_node(bin_t *bin, node_t *node, uint64_t in_bin)
{
    validate_bin(bin);

    if (!bin || !node)
        return;

    HEAP_ASSERT(node->hole == 1, "remove_node: removing non-free node");
    HEAP_ASSERT(node->in_bin == in_bin, "remove_node: wrong in_bin for this node");

    if (!bin->head)
        HeapPanic("remove_node: empty bin");

    if (bin->head == node) {
        bin->head = node->next;
        if (bin->head)
            bin->head->prev = nullptr;

        node->next = nullptr;
        node->prev = nullptr;

        mark_not_in_bin(node);
        validate_bin(bin);
        return;
    }

    node_t *p = node->prev;
    node_t *n = node->next;

    HEAP_ASSERT(p != nullptr, "remove_node: not head but prev==null (corruption)");

    p->next = n;
    if (n)
        n->prev = p;

    node->next = nullptr;
    node->prev = nullptr;

    mark_not_in_bin(node);
    validate_bin(bin);
}


node_t *
get_best_fit(bin_t *bin, size_t size) {
    validate_bin(bin);

    if (bin->head == NULL) return NULL; // empty list!

    node_t *temp = bin->head;

    while (temp != NULL) {
        if (temp->size >= size) {
            return temp; // found a fit!
        }
        temp = temp->next;
    }
    return NULL; // no fit!
}

node_t *
get_last_node(bin_t *bin) {
    node_t *temp = bin->head;

    validate_bin(bin);

    while (temp->next != NULL) {
        temp = temp->next;
    }
    return temp;
}

