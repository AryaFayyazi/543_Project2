// btree.c
#include "btree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

struct BTreeNode {
    int       nkeys;
    BTKey    *keys;
    BTPayload *values;
    BTreeNode **children;
    int       leaf;
};

static BTreeNode* bt_new_node(int t, int leaf) {
    BTreeNode *node = (BTreeNode*)malloc(sizeof(BTreeNode));
    node->nkeys = 0;
    node->leaf = leaf;
    node->keys = (BTKey*)malloc(sizeof(BTKey) * (2*t - 1));
    node->values = (BTPayload*)malloc(sizeof(BTPayload) * (2*t - 1));
    node->children = (BTreeNode**)malloc(sizeof(BTreeNode*) * (2*t));
    for (int i = 0; i < 2*t; i++) node->children[i] = NULL;
    return node;
}

static void bt_free_node(BTreeNode *node, int t) {
    if (!node) return;
    if (!node->leaf) {
        for (int i = 0; i <= node->nkeys; i++)
            bt_free_node(node->children[i], t);
    }
    free(node->keys);
    free(node->values);
    free(node->children);
    free(node);
}

BTree* bt_create(int t) {
    BTree *tree = (BTree*)malloc(sizeof(BTree));
    tree->t = t;
    tree->root = bt_new_node(t, 1);
    return tree;
}

void bt_free(BTree *tree) {
    if (!tree) return;
    bt_free_node(tree->root, tree->t);
    free(tree);
}

static BTPayload bt_search_node(BTreeNode *node, BTKey k, BTStats *stats, int t) {
    if (stats) stats->node_visits++;

    int i = 0;
    while (i < node->nkeys && k > node->keys[i]) i++;

    if (i < node->nkeys && k == node->keys[i]) {
        return node->values[i];
    }

    if (node->leaf) {
        return NULL;
    } else {
        return bt_search_node(node->children[i], k, stats, t);
    }
}

BTPayload bt_search(BTree *tree, BTKey k, BTStats *stats) {
    if (!tree || !tree->root) return NULL;
    return bt_search_node(tree->root, k, stats, tree->t);
}

// Split child y of node x at index i.
static void bt_split_child(BTree *tree, BTreeNode *x, int i) {
    int t = tree->t;
    BTreeNode *y = x->children[i];
    BTreeNode *z = bt_new_node(t, y->leaf);
    z->nkeys = t - 1;

    // Copy upper half of y to z
    for (int j = 0; j < t-1; j++) {
        z->keys[j] = y->keys[j + t];
        z->values[j] = y->values[j + t];
    }

    // Copy children
    if (!y->leaf) {
        for (int j = 0; j < t; j++) {
            z->children[j] = y->children[j + t];
        }
    }

    y->nkeys = t - 1;

    // Shift children of x
    for (int j = x->nkeys; j >= i+1; j--) {
        x->children[j+1] = x->children[j];
    }
    x->children[i+1] = z;

    // Shift keys of x
    for (int j = x->nkeys - 1; j >= i; j--) {
        x->keys[j+1] = x->keys[j];
        x->values[j+1] = x->values[j];
    }

    // Move middle key from y to x
    x->keys[i] = y->keys[t-1];
    x->values[i] = y->values[t-1];
    x->nkeys++;
}

static void bt_insert_nonfull(BTree *tree, BTreeNode *x, BTKey k, BTPayload v) {
    int i = x->nkeys - 1;

    if (x->leaf) {
        // Find position to insert
        while (i >= 0 && k < x->keys[i]) {
            x->keys[i+1] = x->keys[i];
            x->values[i+1] = x->values[i];
            i--;
        }
        // Overwrite if equal (simple “update” semantics)
        if (i >= 0 && x->keys[i] == k) {
            x->values[i] = v;
            return;
        }
        x->keys[i+1] = k;
        x->values[i+1] = v;
        x->nkeys++;
    } else {
        // Find child to descend
        while (i >= 0 && k < x->keys[i]) i--;
        i++;
        if (x->children[i]->nkeys == 2*tree->t - 1) {
            bt_split_child(tree, x, i);
            if (k > x->keys[i]) i++;
        }
        bt_insert_nonfull(tree, x->children[i], k, v);
    }
}

void bt_insert(BTree *tree, BTKey k, BTPayload v) {
    BTreeNode *r = tree->root;
    int t = tree->t;
    if (r->nkeys == 2*t - 1) {
        BTreeNode *s = bt_new_node(t, 0);
        s->children[0] = r;
        tree->root = s;
        bt_split_child(tree, s, 0);
        bt_insert_nonfull(tree, s, k, v);
    } else {
        bt_insert_nonfull(tree, r, k, v);
    }
}

// Range search helper.
static void bt_range_node(BTreeNode *node, BTKey lo, BTKey hi,
                          BTRangeCallback cb, void *arg, BTStats *stats, int t) {
    if (!node) return;
    if (stats) stats->node_visits++;

    int i;
    for (i = 0; i < node->nkeys; i++) {
        if (!node->leaf) {
            if (lo <= node->keys[i])
                bt_range_node(node->children[i], lo, hi, cb, arg, stats, t);
        }
        if (node->keys[i] >= lo && node->keys[i] <= hi) {
            cb(node->keys[i], node->values[i], arg);
        }
        if (node->keys[i] > hi) {
            if (!node->leaf)
                bt_range_node(node->children[i], lo, hi, cb, arg, stats, t);
            return;
        }
    }
    if (!node->leaf) {
        bt_range_node(node->children[i], lo, hi, cb, arg, stats, t);
    }
}

void bt_range_search(BTree *tree, BTKey lo, BTKey hi,
                     BTRangeCallback cb, void *arg, BTStats *stats) {
    if (!tree || !tree->root) return;
    bt_range_node(tree->root, lo, hi, cb, arg, stats, tree->t);
}

static size_t bt_count_keys_node(BTreeNode *node) {
    if (!node) return 0;
    size_t res = node->nkeys;
    if (!node->leaf) {
        for (int i = 0; i <= node->nkeys; i++)
            res += bt_count_keys_node(node->children[i]);
    }
    return res;
}

size_t bt_count_keys(BTree *tree) {
    if (!tree || !tree->root) return 0;
    return bt_count_keys_node(tree->root);
}
