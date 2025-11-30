// hctree.c
#include "hctree.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

HCIndex* hc_create(int64_t max_key, int btree_degree, HCParams params) {
    HCIndex *idx = (HCIndex*)malloc(sizeof(HCIndex));
    idx->hot  = bt_create(btree_degree);
    idx->cold = bt_create(btree_degree);

    idx->max_key   = max_key;
    idx->hit_score = (double*)calloc((size_t)(max_key + 1), sizeof(double));

    idx->params = params;
    memset(&idx->stats, 0, sizeof(HCStats));

    // NEW: init adaptation state
    idx->last_q_for_adapt = 0;
    idx->last_cost        = 0.0;
    idx->last_D           = params.sampling_rate;

    return idx;
}

void hc_free(HCIndex *idx) {
    if (!idx) return;
    bt_free(idx->hot);
    bt_free(idx->cold);
    free(idx->hit_score);
    free(idx);
}

void hc_insert(HCIndex *idx, BTKey k, BTPayload v) {
    // For this project, we assume 0 <= k <= max_key.
    if (k < 0 || k > idx->max_key) {
        fprintf(stderr,
                "hc_insert: key %" PRId64 " out of range [0, %" PRId64 "]\n",
                (int64_t)k, (int64_t)idx->max_key);
        return;
    }
    bt_insert(idx->cold, k, v);
}

// --- NEW: ML-style adaptation of sampling rate D ------------------------

// Adapt sampling_rate (D) based on observed cost = node_visits / query.
static void hc_maybe_adapt_sampling(HCIndex *idx) {
    if (!idx->params.adapt_sampling)
        return;

    const long MIN_DELTA_Q = 5000; // adjust every 5k queries
    long q = idx->stats.queries;
    if (q - idx->last_q_for_adapt < MIN_DELTA_Q)
        return;

    double total_nodes = (double)idx->stats.hot_node_visits +
                         (double)idx->stats.cold_node_visits;
    double cost = (q > 0) ? (total_nodes / (double)q) : 0.0;

    double D = idx->params.sampling_rate;

    if (idx->last_q_for_adapt == 0) {
        // FIRST adaptation: always move D a bit.
        // Use hot-hit fraction to choose direction.
        double hot_frac = (double)idx->stats.hot_hits / (double)(q ? q : 1);
        double target_hot = 0.6;  // heuristic target hot-hit ratio

        if (hot_frac < target_hot) {
            // Hot tier under-utilized → increase D
            D += 0.05;
        } else {
            // Hot tier maybe too big / too aggressive → decrease D
            D -= 0.05;
        }
    } else {
        // SUBSEQUENT adaptations: look at how cost changed since last time.
        double dC = cost - idx->last_cost;
        double dD = D - idx->last_D;

        if (fabs(dD) < 1e-9) {
            // If we somehow didn't move last time, nudge based on hot_frac.
            double hot_frac = (double)idx->stats.hot_hits / (double)(q ? q : 1);
            double target_hot = 0.6;
            if (hot_frac < target_hot) D += 0.05;
            else                       D -= 0.05;
        } else {
            // If increasing D last time decreased cost → keep that direction.
            // If it increased cost → reverse direction.
            if (dC * dD < 0.0) {
                D += 0.05 * ((dD > 0.0) ? 1.0 : -1.0);
            } else if (dC * dD > 0.0) {
                D -= 0.05 * ((dD > 0.0) ? 1.0 : -1.0);
            }
            // If dC ≈ 0, leave D unchanged.
        }
    }

    // Clamp D to [0, 1]
    if (D < 0.0) D = 0.0;
    if (D > 1.0) D = 1.0;

    idx->last_D           = idx->params.sampling_rate; // old D
    idx->last_cost        = cost;
    idx->last_q_for_adapt = q;
    idx->params.sampling_rate = D;                     // new D
}
// --- Sampling-based promotion ------------------------------------------

static void maybe_promote(HCIndex *idx, BTKey k) {
    if (!idx->params.inclusive) {
        // We only implement inclusive mode in this standalone version.
        return;
    }

    // 1) Sampling-based promotion: with probability (1 - D) skip.
    double D = idx->params.sampling_rate;
    if (D < 0.0) D = 0.0;
    if (D > 1.0) D = 1.0;
    double u = (double)rand() / ((double)RAND_MAX + 1.0);
    if (u > D) {
        return; // sampled out, no promotion this time
    }

    // 2) Capacity check: keep hot index under max_hot_fraction of keyspace.
    size_t total_keys = (size_t)(idx->max_key + 1);
    size_t hot_keys   = bt_count_keys(idx->hot);
    size_t cold_keys  = bt_count_keys(idx->cold);
    (void)cold_keys; // not used in inclusive mode

    double max_hot = idx->params.max_hot_fraction * (double)total_keys;
    if ((double)hot_keys >= max_hot) {
        return; // hot index already at capacity
    }

    // 3) If key already in hot, nothing to do.
    BTStats s = {0};
    BTPayload existing = bt_search(idx->hot, k, &s);
    if (existing != NULL) return;

    // 4) Key must exist in cold; fetch payload.
    BTStats s2 = {0};
    BTPayload v = bt_search(idx->cold, k, &s2);
    if (v == NULL) return; // not found; nothing to promote

    bt_insert(idx->hot, k, v);
}

// Point lookup: hot first, then cold.
BTPayload hc_search(HCIndex *idx, BTKey k) {
    idx->stats.queries++;

    // NEW: occasionally adapt sampling rate based on stats so far.
    hc_maybe_adapt_sampling(idx);

    BTStats hot_s = {0};
    BTPayload v = bt_search(idx->hot, k, &hot_s);
    idx->stats.hot_node_visits += hot_s.node_visits;

    if (v != NULL) {
        idx->stats.hot_hits++;
        if (k >= 0 && k <= idx->max_key) {
            double old = idx->hit_score[k];
            idx->hit_score[k] = idx->params.decay_alpha * old + 1.0;
            // We don't re-promote; already hot.
        }
        return v;
    }

    BTStats cold_s = {0};
    v = bt_search(idx->cold, k, &cold_s);
    idx->stats.cold_node_visits += cold_s.node_visits;

    if (v != NULL) {
        idx->stats.cold_hits++;
        if (k >= 0 && k <= idx->max_key) {
            double old = idx->hit_score[k];
            double new_score = idx->params.decay_alpha * old + 1.0;
            idx->hit_score[k] = new_score;
            if (new_score >= idx->params.hot_threshold)
                maybe_promote(idx, k);
        }
        return v;
    } else {
        idx->stats.not_found++;
        return NULL;
    }
}

// Helper for deduped range scan: simple callback wrapper
typedef struct {
    BTRangeCallback user_cb;
    void           *user_arg;
    int64_t        *seen;     // seen[key] = 1 if already emitted (size max_key+1)
    HCIndex        *idx;
} HCRangeCtx;

static void hc_range_cb_hot(BTKey k, BTPayload v, void *arg) {
    HCRangeCtx *ctx = (HCRangeCtx*)arg;
    if (k < 0 || k > ctx->idx->max_key) return;
    if (!ctx->seen[k]) {
        ctx->seen[k] = 1;
        ctx->user_cb(k, v, ctx->user_arg);
    }
}

static void hc_range_cb_cold(BTKey k, BTPayload v, void *arg) {
    HCRangeCtx *ctx = (HCRangeCtx*)arg;
    if (k < 0 || k > ctx->idx->max_key) return;
    if (!ctx->seen[k]) {
        ctx->seen[k] = 1;
        ctx->user_cb(k, v, ctx->user_arg);
    }
}

void hc_range_search(HCIndex *idx, BTKey lo, BTKey hi,
                     BTRangeCallback cb, void *arg) {
    HCRangeCtx ctx;
    ctx.user_cb = cb;
    ctx.user_arg = arg;
    ctx.idx = idx;
    ctx.seen = (int64_t*)calloc((size_t)(idx->max_key + 1), sizeof(int64_t));

    BTStats hot_s = {0}, cold_s = {0};
    bt_range_search(idx->hot, lo, hi, hc_range_cb_hot, &ctx, &hot_s);
    bt_range_search(idx->cold, lo, hi, hc_range_cb_cold, &ctx, &cold_s);

    idx->stats.hot_node_visits  += hot_s.node_visits;
    idx->stats.cold_node_visits += cold_s.node_visits;

    free(ctx.seen);
}

HCStats hc_get_stats(HCIndex *idx) {
    HCStats s = idx->stats;

    s.hot_keys  = bt_count_keys(idx->hot);
    s.cold_keys = bt_count_keys(idx->cold);

    // NEW: expose final ML-learned sampling rate
    s.final_sampling_rate = idx->params.sampling_rate;

    return s;
}

