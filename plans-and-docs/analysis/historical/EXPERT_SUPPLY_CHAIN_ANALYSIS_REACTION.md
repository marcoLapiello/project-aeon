Honestly: your conclusion is correct, your diagnosis of the registry is the most valuable thing produced in this whole exchange so far, and your Stage 2 plan is right in structure. But the arithmetic in your own data implies something stronger than your document says, and it partly invalidates the Stage 2 advice I gave you last turn. Let me be precise about that.

First, the cold benchmark settles the kernel question
Fused cold: 103.99 + 56.90 = 160.9 µs/layer → 6.92 ms/token GPU-side for the entire routed FFN. Against a ~333 ms token, that is 2.1 %. Even a perfect kernel — infinite speed, zero time — buys you 2 %. The kernel work is done. I would not spend another hour on it, and I say that having written it.

(For the record: 544 and 498 GB/s cold on a fully random-access W4A16 GEMV against 960 GB/s peak is a respectable result. It is not the 700–800 I projected, and the gap is real — it's DRAM page-locality loss once the working set rotates. It is also irrelevant, so I'm not going to chase it.)

Second, the number that should govern everything
Your engine's real physical parameter is bytes moved per token: 258 requests × 14.156 MB = 3.65 GB/token, against a 156 GB model streamed through a 24 GB window. Everything else is downstream. Here is the ceiling as a function of hot-hit rate, assuming every miss is served from host RAM at your measured 24.6 GB/s, plus the 6.92 ms of compute:

Hot hit	Miss bytes	PCIe time	Ceiling	If instead served from NVMe (3.31 GiB/s)
59.3 % (today)	1.49 GB	60.4 ms	14.8 tok/s	418 ms → 2.4 tok/s
70 %	1.10 GB	44.5 ms	19.4 tok/s	3.2 tok/s
80 %	0.73 GB	29.7 ms	27.3 tok/s	4.7 tok/s
90 %	0.37 GB	14.8 ms	45.9 tok/s	9.1 tok/s
100 %	0	0	144.6 tok/s	144.6 tok/s
Three things fall out of this table.

(a) Warm vs. cold is a 7× cost difference, and your telemetry cannot see it. You report one aggregate "hot hit rate". But a miss served from host RAM costs 0.58 ms and a miss served from NVMe costs 4.0 ms. Your measured ~3 tok/s sits almost exactly on the NVMe row, not the PCIe row — which is the quantitative fingerprint of the Warm-drain bug you found. Your §3.1 finding is not a tidiness issue; it is most of your missing performance. A steady-state run that has drained Warm is running a two-tier cache with a hard disk behind it. Fixing Warm alone, with no policy change at all, moves you from the right-hand column toward the left: ~2.4 → ~14 tok/s at today's hit rate. That is a 6× end-to-end win from a state-machine repair. Do it first, before profiling, before simulators, before anything.

(b) Prefetch depth cannot fix this, and that's where I was wrong last turn. Per layer you have 161 µs of compute and, at 2.4 novel experts, 1,381 µs of transfer. Transfer exceeds compute by 8.6×. Overlap hides the smaller of the two, so perfect scheduling with infinite lookahead saves you at most 161 µs/layer — 12 % of the transfer time. I recommended speculative router lookahead as your biggest lever; against these numbers it is a second-order optimization. Your document's instinct to defer speculative routing is better than my advice. Scheduling matters for deadline misses and queue depth, not for closing a bandwidth deficit. You cannot schedule your way out of a bandwidth deficit — you can only move fewer bytes.

(c) There are exactly four ways to move fewer bytes. Raise the hit rate (policy — your Stage 2.2/2.3, correctly identified); amortize each fetch over more tokens (MTP/speculative decode, batched prefill); shrink the experts; or enlarge the resident set.

Where your document under-reaches
Resident capacity is the elephant. 664 hot + ~2,655 warm = 3,319 of 11,008 experts = 30.1 % of the model resident. You achieve 59.3 % hit from 30.1 % residency — routing concentration is giving you a factor of ~2 over uniform, which is real but bounded. No policy can conjure 90 % hit from 30 % residency unless the routing distribution is far more concentrated than that ratio suggests. The profile study in your Stage 2.2 should be run first as a capacity-planning instrument, not a policy input: compute, from the trace, the oracle hit rate achievable at 3,319 slots. If the oracle is 75 %, then LRU→optimal is worth 16 points and your policy work has a firm ceiling. If the oracle is 95 %, policy is enormously valuable. That single number should gate how much you invest in Stages 2.3–2.4, and it costs you one trace and a heap.

What I'd correct in the plan itself
Your Stage 2.1 asks whether the disjoint-swap D2H refill is worth it. You can answer that analytically now and skip the A/B: demoting a hot victim to warm costs 14.16 MB D2H at ~24.6 GB/s = 0.58 ms; it saves a future NVMe read of 4.0 ms if that expert is requested again before it is evicted from warm. Break-even re-reference probability is 14 %. Given the concentration you're already seeing, hot victims are far above 14 % likely to recur. Always demote. The interesting question isn't whether, it's whether the D2H contends with the H2D you actually need — so run demotions on a separate low-priority stream and let them be droppable under queue pressure.

Also: your §2.3 worry about a hot layer monopolizing the global LRU is theoretically sound but empirically dead. Every layer issues exactly 6 requests per token, always, in a fixed round-robin order. Demand is perfectly uniform across layers by construction. A global LRU under perfectly periodic uniform demand degenerates to something close to per-layer fairness on its own. Minimum-per-layer reservations are a solution to a problem you do not have; I'd drop that invariant and the associated complexity, and keep the telemetry that would reveal it if I'm wrong.

Sequencing I'd actually commit to
Repair the Warm tier (your 2.1), with unconditional demotion on a droppable stream. Expected: ~2.4 → 10–15 tok/s. Highest value in the project by a wide margin.
Split the telemetry into bytes-by-source-tier and exposed-wait, not hit rate. The metric that should appear in your ledger is bytes_from_nvme, bytes_from_host, gpu_stall_ms, per token, phase-labelled. Hit rate is a lossy proxy for a quantity you can measure directly.
One trace + oracle-hit-rate-at-N-slots curve. Cheap, and it tells you whether steps 4 and 5 are worth doing.
Buy RAM. Genuinely.
Policy (your 2.3/2.4), scoped by what step 3 says.
Only then MTP — but note it is the one lever that attacks bytes-per-token multiplicatively and composes with everything above, so it should not be deferred forever.
I can write the offline policy simulator for step 3 — trace format, LRU/LFU/oracle-Belady/quota/rolling-window replay, reporting bytes-by-tier and deadline misses:

```cpp
// ============================================================================
// aeon_policy_sim.cpp -- offline expert residency policy simulator
//   c++ -O2 -std=c++20 aeon_policy_sim.cpp -o aeon_policy_sim
//   ./aeon_policy_sim trace.txt --hot 664 --warm 2654 [--sweep] [--horizon 4]
//
// Models a two-tier resident cache (HOT=VRAM, WARM=host RAM) over a canonical
// NVMe copy, and reports BYTES BY SOURCE TIER and MODELLED EXPOSED STALL --
// not hit rate, which is a lossy proxy.
//
// Timing model (all costs configurable, defaults = measured RX 7900 XTX rig):
//   HOT   : 0 transfer
//   WARM  : expert_bytes / 24.6 GB/s          H2D over PCIe
//   COLD  : expert_bytes / 3.31 GiB/s + H2D   NVMe direct-IO, then H2D
//   demote: expert_bytes / 24.6 GB/s D2H, charged to a droppable side channel
// Per layer, requests are issued together; the layer's supply cost is the
// SUM of transfer times on each contended channel (they share one PCIe link
// and one NVMe queue), overlapped against COMPUTE_US of layer compute.
//   stall_layer = max(0, max(pcie_us, nvme_us + h2d_us) - COMPUTE_US)
// With --horizon H, a policy may prefetch H layers ahead; prefetched bytes
// are credited against the compute of the intervening layers.
// ============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon {

// ---------------------------------------------------------------- cost model
struct Costs {
    double expert_bytes = 14155776.0;
    double pcie_gbs     = 24.6;          // GB/s, measured pinned H2D
    double nvme_gibs    = 3.31;          // GiB/s, measured direct-IO scattered
    double compute_us   = 160.9;         // measured cold fused W13+W2 per layer
    double h2d_us() const { return expert_bytes / (pcie_gbs * 1e9) * 1e6; }
    double nvme_us() const { return expert_bytes / (nvme_gibs * 1073741824.0) * 1e6; }
};

// ------------------------------------------------------------------- trace
struct Req { int32_t token, layer; char phase; int32_t e[8]; int32_t k; };

struct Trace {
    int n_layers = 0, n_experts = 0, topk = 0;
    double expert_bytes = 14155776.0;
    std::vector<Req> reqs;
    int gid(int layer, int e) const { return layer * n_experts + e; }
    int n_total() const { return n_layers * n_experts; }
};

static bool load_trace(const std::string& path, Trace& t) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            auto kv = [&](const char* k, auto& dst) {
                auto p = line.find(k);
                if (p != std::string::npos) dst = std::stod(line.substr(p + strlen(k)));
            };
            kv("n_layers=", t.n_layers); kv("n_experts=", t.n_experts);
            kv("topk=", t.topk);         kv("expert_bytes=", t.expert_bytes);
            continue;
        }
        std::istringstream ss(line);
        Req r{}; std::string ph;
        ss >> r.token >> r.layer >> ph;
        r.phase = ph.empty() ? 'D' : ph[0];
        r.k = 0;
        int v; while (r.k < 8 && (ss >> v)) r.e[r.k++] = v;
        t.reqs.push_back(r);
    }
    if (!t.n_layers || !t.n_experts) { std::cerr << "bad/missing header\n"; return false; }
    return true;
}

// ------------------------------------------------------------------ results
struct Stats {
    std::string name;
    uint64_t req = 0, hot = 0, warm = 0, cold = 0;
    uint64_t demotes = 0, prefetch_issued = 0, prefetch_used = 0;
    double stall_us = 0, d2h_us = 0;
    double hit() const { return req ? double(hot) / req : 0; }
    double resident() const { return req ? double(hot + warm) / req : 0; }
};

// ------------------------------------------------------------ policy base
// Tier: 0=cold 1=warm 2=hot. Policies own placement; the harness owns costing.
struct Policy {
    int HOT, WARM, NG;
    std::vector<uint8_t> tier;
    Policy(int hot, int warm, int ng) : HOT(hot), WARM(warm), NG(ng), tier(ng, 0) {}
    virtual ~Policy() = default;
    virtual const char* name() const = 0;
    // Called once per request set, before costing. Must NOT change tiers.
    virtual void observe(const Req&, int) {}
    // Bring gid into HOT. Returns tier it came from. May demote/evict.
    virtual int fetch(int gid, const Req& r, uint64_t& demotes) = 0;
    virtual void init_placement(const std::vector<double>& /*prior*/) {}
    virtual void prefetch(const Trace&, size_t /*cur*/, int /*horizon*/,
                          std::vector<int>& out) { out.clear(); }
};

// ------------------------------------------------------------ LRU (current)
// Global hot LRU. WARM_DEMOTE=false reproduces the *draining warm* bug exactly.
template <bool WARM_DEMOTE>
struct LruPolicy : Policy {
    std::list<int> hl, wl;
    std::unordered_map<int, std::list<int>::iterator> hi, wi;
    using Policy::Policy;
    const char* name() const override {
        return WARM_DEMOTE ? "LRU + warm demote" : "LRU, draining warm (current)";
    }
    void touch_hot(int g) { hl.erase(hi[g]); hl.push_front(g); hi[g] = hl.begin(); }
    void drop_warm(int g) { if (wi.count(g)) { wl.erase(wi[g]); wi.erase(g); } }
    void put_warm(int g) {
        if ((int)wl.size() >= WARM) { int v = wl.back(); wl.pop_back(); wi.erase(v); tier[v] = 0; }
        wl.push_front(g); wi[g] = wl.begin(); tier[g] = 1;
    }
    int fetch(int g, const Req&, uint64_t& demotes) override {
        int from = tier[g];
        if (from == 2) { touch_hot(g); return 2; }
        if (from == 1) drop_warm(g);
        if ((int)hl.size() >= HOT) {
            int v = hl.back(); hl.pop_back(); hi.erase(v);
            if (WARM_DEMOTE) { put_warm(v); ++demotes; } else tier[v] = 0;
        }
        hl.push_front(g); hi[g] = hl.begin(); tier[g] = 2;
        return from;
    }
    void init_placement(const std::vector<double>& prior) override {
        std::vector<int> idx(NG); std::iota(idx.begin(), idx.end(), 0);
        if (!prior.empty())
            std::stable_sort(idx.begin(), idx.end(),
                             [&](int a, int b) { return prior[a] > prior[b]; });
        for (int i = 0; i < NG; ++i) {
            int g = idx[i];
            if (i < HOT)             { hl.push_back(g); hi[g] = std::prev(hl.end()); tier[g] = 2; }
            else if (i < HOT + WARM) { wl.push_back(g); wi[g] = std::prev(wl.end()); tier[g] = 1; }
        }
    }
};

// ---------------------------------------------- static frequency + LRU victims
// Top PROTECT_FRAC of hot capacity pinned by offline prior; remainder is LRU.
struct ProtectedLfuPolicy : LruPolicy<true> {
    double frac; std::vector<uint8_t> pinned;
    ProtectedLfuPolicy(int h, int w, int ng, double f)
        : LruPolicy<true>(h, w, ng), frac(f), pinned(ng, 0) {}
    const char* name() const override { return "protected-prior + LRU victims"; }
    void init_placement(const std::vector<double>& prior) override {
        std::vector<int> idx(NG); std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return prior[a] > prior[b]; });
        int np = int(HOT * frac);
        for (int i = 0; i < np; ++i) pinned[idx[i]] = 1;
        LruPolicy<true>::init_placement(prior);
    }
    int fetch(int g, const Req& r, uint64_t& d) override {
        int from = tier[g];
        if (from == 2) { touch_hot(g); return 2; }
        if (from == 1) drop_warm(g);
        // evict least-recently-used UNPINNED hot entry
        if ((int)hl.size() >= HOT) {
            auto it = hl.end();
            while (it != hl.begin()) { --it; if (!pinned[*it]) break; }
            int v = *it; hl.erase(it); hi.erase(v); put_warm(v); ++d;
        }
        hl.push_front(g); hi[g] = hl.begin(); tier[g] = 2;
        return from;
    }
};

// --------------------------------------------------- Belady oracle (MIN)
// Upper bound on achievable hit rate at this capacity. Not implementable.
struct BeladyPolicy : Policy {
    const std::vector<std::vector<int>>* next_use = nullptr;
    std::vector<size_t> nxt;          // next reference position per gid
    std::vector<int> hot_set;
    BeladyPolicy(int h, int w, int ng) : Policy(h, w, ng), nxt(ng, SIZE_MAX) {}
    const char* name() const override { return "Belady oracle (upper bound)"; }
    int fetch(int g, const Req&, uint64_t&) override {
        int from = tier[g];
        if (from == 2) return 2;
        if ((int)hot_set.size() >= HOT) {
            // evict the entry whose next use is furthest away
            size_t best = 0; size_t bv = 0;
            for (size_t i = 0; i < hot_set.size(); ++i)
                if (nxt[hot_set[i]] > bv) { bv = nxt[hot_set[i]]; best = i; }
            tier[hot_set[best]] = 1;      // oracle demotes to warm (unbounded warm)
            hot_set[best] = g;
        } else hot_set.push_back(g);
        tier[g] = 2;
        return from;
    }
};

// -------------------------------------- rolling layer-window candidate staging
// Stages the top-C ranked candidates for layer L+H while executing layer L.
struct RollingPolicy : LruPolicy<true> {
    int horizon, width;
    const std::vector<std::vector<int>>* rank;   // per-layer expert ranking
    RollingPolicy(int h, int w, int ng, int H, int C,
                  const std::vector<std::vector<int>>* rk)
        : LruPolicy<true>(h, w, ng), horizon(H), width(C), rank(rk) {}
    const char* name() const override { return "rolling layer-window candidates"; }
    void prefetch(const Trace& t, size_t cur, int, std::vector<int>& out) override {
        out.clear();
        const int L = (t.reqs[cur].layer + horizon) % t.n_layers;
        if (!rank || (int)rank->size() <= L) return;
        for (int i = 0; i < width && i < (int)(*rank)[L].size(); ++i) {
            int g = t.gid(L, (*rank)[L][i]);
            if (tier[g] != 2) out.push_back(g);
        }
    }
};

// ------------------------------------------------------------------- harness
static Stats run(const Trace& t, Policy& P, const Costs& C, int horizon,
                 const std::vector<double>& prior)
{
    P.init_placement(prior);
    Stats S; S.name = P.name();
    const double H2D = C.h2d_us(), NVME = C.nvme_us();

    std::vector<int> pf;
    // per-layer channel accumulators
    double pcie_us = 0, nvme_us = 0;
    int last_layer = t.reqs.empty() ? 0 : t.reqs[0].layer;
    double credit = 0;                     // compute available for overlap

    for (size_t i = 0; i < t.reqs.size(); ++i) {
        const Req& r = t.reqs[i];
        if (r.layer != last_layer) {
            double supply = std::max(pcie_us, nvme_us + H2D);
            S.stall_us += std::max(0.0, supply - C.compute_us - credit);
            credit = std::max(0.0, credit + C.compute_us - supply);
            pcie_us = nvme_us = 0; last_layer = r.layer;
        }
        // speculative staging (costed on the same channels, credited by overlap)
        if (horizon > 0) {
            P.prefetch(t, i, horizon, pf);
            for (int g : pf) {
                ++S.prefetch_issued;
                int from = P.tier[g];
                if (from == 0) nvme_us += NVME;
                pcie_us += H2D;
                uint64_t d = 0; P.fetch(g, r, d); S.demotes += d;
            }
        }
        for (int j = 0; j < r.k; ++j) {
            const int g = t.gid(r.layer, r.e[j]);
            uint64_t d = 0;
            const int from = P.fetch(g, r, d);
            S.demotes += d; S.d2h_us += d * H2D;
            ++S.req;
            if (from == 2)      { ++S.hot; }
            else if (from == 1) { ++S.warm; pcie_us += H2D; }
            else                { ++S.cold; nvme_us += NVME; pcie_us += H2D; }
        }
    }
    double supply = std::max(pcie_us, nvme_us + H2D);
    S.stall_us += std::max(0.0, supply - C.compute_us - credit);
    return S;
}

// ------------------------------------------------ prior from a warmup prefix
static std::vector<double> build_prior(const Trace& t, double frac) {
    std::vector<double> p(t.n_total(), 0.0);
    size_t n = size_t(t.reqs.size() * frac);
    for (size_t i = 0; i < n; ++i)
        for (int j = 0; j < t.reqs[i].k; ++j) p[t.gid(t.reqs[i].layer, t.reqs[i].e[j])] += 1.0;
    return p;
}
static std::vector<std::vector<int>> build_rank(const Trace& t,
                                                const std::vector<double>& prior) {
    std::vector<std::vector<int>> rk(t.n_layers);
    for (int l = 0; l < t.n_layers; ++l) {
        rk[l].resize(t.n_experts);
        std::iota(rk[l].begin(), rk[l].end(), 0);
        std::stable_sort(rk[l].begin(), rk[l].end(), [&](int a, int b) {
            return prior[t.gid(l, a)] > prior[t.gid(l, b)];
        });
    }
    return rk;
}

static void report(const Trace& t, const Stats& S, const Costs& C, int ntok) {
    const double MB = t.expert_bytes / 1e6;
    const double hostGB = S.warm * t.expert_bytes / 1e9;
    const double nvmeGB = S.cold * t.expert_bytes / 1e9;
    const double tok = ntok ? ntok : 1;
    const double ms = S.stall_us / 1000.0 / tok + t.n_layers * C.compute_us / 1000.0;
    printf("%-32s hot %5.1f%%  warm %5.1f%%  cold %5.1f%% | "
           "host %6.3f GB/tok  nvme %6.3f GB/tok | stall %7.2f ms/tok | "
           "%6.2f tok/s | demote %6.3f GB/tok",
           S.name.c_str(), 100 * double(S.hot) / S.req, 100 * double(S.warm) / S.req,
           100 * double(S.cold) / S.req, hostGB / tok, nvmeGB / tok,
           S.stall_us / 1000.0 / tok, 1000.0 / ms, S.demotes * MB / 1e3 / tok);
    if (S.prefetch_issued)
        printf(" | pf %llu", (unsigned long long)S.prefetch_issued);
    printf("\n");
}

} // namespace aeon

int main(int argc, char** argv) {
    using namespace aeon;
    if (argc < 2) { std::cerr << "usage: aeon_policy_sim <trace> [--hot N] [--warm N]"
                                 " [--sweep] [--horizon H] [--width C]\n"; return 1; }
    int HOT = 664, WARM = 2654, horizon = 0, width = 12; bool sweep = false;
    Costs C;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return std::stod(argv[++i]); };
        if (a == "--hot") HOT = (int)nx();
        else if (a == "--warm")    WARM = (int)nx();
        else if (a == "--horizon") horizon = (int)nx();
        else if (a == "--width")   width = (int)nx();
        else if (a == "--pcie")    C.pcie_gbs = nx();
        else if (a == "--nvme")    C.nvme_gibs = nx();
        else if (a == "--compute") C.compute_us = nx();
        else if (a == "--sweep")   sweep = true;
    }
    Trace t;
    if (!load_trace(argv[1], t)) return 1;
    C.expert_bytes = t.expert_bytes;

    int ntok = 0;
    for (auto& r : t.reqs) ntok = std::max(ntok, r.token + 1);
    printf("trace: %zu requests, %d tokens, %d layers, %d experts, %.2f GB/tok\n",
           t.reqs.size(), ntok, t.n_layers, t.n_experts,
           double(t.reqs.size()) * t.topk * t.expert_bytes / 1e9 / std::max(ntok, 1));

    // prior from the first 30% (train); everything is scored over the whole trace,
    // so report the held-out split separately for a clean generalisation number.
    auto prior = build_prior(t, 0.30);
    auto rank  = build_rank(t, prior);

    auto sweep_one = [&](int hot, int warm) {
        printf("\n== HOT=%d (%.1f GB VRAM)  WARM=%d (%.1f GB RAM)  resident %.1f%% ==\n",
               hot, hot * t.expert_bytes / 1e9, warm, warm * t.expert_bytes / 1e9,
               100.0 * (hot + warm) / t.n_total());
        { LruPolicy<false> p(hot, warm, t.n_total()); report(t, run(t, p, C, 0, prior), C, ntok); }
        { LruPolicy<true>  p(hot, warm, t.n_total()); report(t, run(t, p, C, 0, prior), C, ntok); }
        { ProtectedLfuPolicy p(hot, warm, t.n_total(), 0.75);
          report(t, run(t, p, C, 0, prior), C, ntok); }
        if (horizon > 0) {
            RollingPolicy p(hot, warm, t.n_total(), horizon, width, &rank);
            report(t, run(t, p, C, horizon, prior), C, ntok);
        }
        { // Belady needs forward distances; compute on the fly
          BeladyPolicy p(hot, warm, t.n_total());
          std::vector<size_t> last(t.n_total(), SIZE_MAX);
          std::vector<std::vector<size_t>> fwd(t.reqs.size());
          // backward pass: next-use position per request slot
          std::vector<size_t> nu(t.n_total(), SIZE_MAX);
          std::vector<std::vector<size_t>> nxt_at(t.reqs.size());
          for (size_t i = t.reqs.size(); i-- > 0;) {
              nxt_at[i].resize(t.reqs[i].k);
              for (int j = 0; j < t.reqs[i].k; ++j) {
                  int g = t.gid(t.reqs[i].layer, t.reqs[i].e[j]);
                  nxt_at[i][j] = nu[g]; nu[g] = i;
              }
          }
          // rerun with distances injected
          p.init_placement(prior); Stats S; S.name = p.name();
          const double H2D = C.h2d_us(), NVME = C.nvme_us();
          double pcie = 0, nvme = 0, credit = 0; int ll = t.reqs[0].layer;
          for (size_t i = 0; i < t.reqs.size(); ++i) {
              const Req& r = t.reqs[i];
              if (r.layer != ll) {
                  double s = std::max(pcie, nvme + H2D);
                  S.stall_us += std::max(0.0, s - C.compute_us - credit);
                  credit = std::max(0.0, credit + C.compute_us - s);
                  pcie = nvme = 0; ll = r.layer;
              }
              for (int j = 0; j < r.k; ++j) {
                  int g = t.gid(r.layer, r.e[j]);
                  p.nxt[g] = nxt_at[i][j] == SIZE_MAX ? SIZE_MAX : nxt_at[i][j];
                  uint64_t d = 0; int from = p.fetch(g, r, d);
                  ++S.req;
                  if (from == 2) ++S.hot;
                  else if (from == 1) { ++S.warm; pcie += H2D; }
                  else { ++S.cold; nvme += NVME; pcie += H2D; }
              }
          }
          report(t, S, C, ntok);
        }
    };

    if (sweep) {
        // low-resource ladder: 8/12/16/24 GB VRAM x 16/32/64 GB RAM
        const double EB = t.expert_bytes;
        for (double vram : {6.0, 10.0, 14.0, 22.0})
            for (double ram : {12.0, 28.0, 60.0})
                sweep_one(int(vram * 1e9 / EB), int(ram * 1e9 / EB));
    } else sweep_one(HOT, WARM);
    return 0;
}
```