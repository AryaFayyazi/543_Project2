// btree.h
#ifndef BTREE_H
#define BTREE_H

#include <stddef.h>
#include <stdint.h>

typedef int64_t BTKey;
typedef void*   BTPayload;

// For node visit statistics.
typedef struct {
    long node_visits;
} BTStats;

typedef struct BTreeNode BTreeNode;

typedef struct {
    BTreeNode *root;
    int        t;   // minimum degree (B-tree parameter)
} BTree;

BTree*  bt_create(int t);
void    bt_free(BTree *tree);

// Insert key → payload. (No duplicates handling; last insert "wins")
void    bt_insert(BTree *tree, BTKey k, BTPayload v);

// Search for key; returns payload or NULL if not found.
// If stats != NULL, it accumulates node visits.
BTPayload bt_search(BTree *tree, BTKey k, BTStats *stats);

// Range scan: call callback(k, v, arg) for all keys in [lo, hi].
typedef void (*BTRangeCallback)(BTKey k, BTPayload v, void *arg);
void    bt_range_search(BTree *tree, BTKey lo, BTKey hi,
                        BTRangeCallback cb, void *arg, BTStats *stats);

// For stats: approximate number of keys in tree.
size_t  bt_count_keys(BTree *tree);

#endif // BTREE_H
