#pragma once

// -----------------------------------------------------------------------------
// The resident-state record — what a live conversation's layer state already covers.
//
// The engine keeps a model state resident across turns; this is the *description*
// of that state, so the next turn can decide whether it genuinely extends the
// state or has to replay. It holds no tensors and knows no model: it is a list of
// the token ids the state was fed, in order, plus the computation key the state
// was built under. The binding that owns the device state feeds this record at the
// same sites it feeds the graph, so the two cannot drift.
//
// All-or-nothing. There is no longest-common-prefix reuse: the state has a ring
// and a recurrent partial with no rewind, so a prefix that diverges anywhere is a
// cold replay. The record's job is to make that decision cheap and provable.
//
// `feed` throws on a position gap rather than growing a sparse record, because a
// gap means a feed site was skipped and the record would silently describe a state
// the engine does not hold.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aeon::session {

enum class ReuseVerdict {
    Cold,        // no resident state to build on
    Reused,      // the prompt extends the resident state by at least one token
    KeyChanged,  // a non-token input changed; the state was built for another computation
    Diverged,    // the prompt's tokens differ from the resident ones before the boundary
    NoTail,      // the prompt does not extend the record; nothing new to prefill
    StaleState,  // the record and the device state are not the same generation
    Disabled,    // reuse was not requested for this request
};

// What one turn's reuse decision was, and why.
struct ReuseDecision {
    ReuseVerdict verdict{ReuseVerdict::Cold};

    // The tokens the resident state covers, i.e. the number of leading prompt
    // tokens that need not be prefilled again. Zero unless `Reused`.
    uint32_t start{0};

    // The first index at which the prompt differed from the record. Diagnostic
    // only — nothing acts on it — so it is populated for `Diverged` and left zero
    // otherwise.
    uint32_t diverged_at{0};
};

class PrefixRecord {
public:
    void clear() noexcept {
        ids_.clear();
        key_.clear();
    }

    // Start a record for a state being built from scratch under `computation_key`.
    void begin(std::string computation_key) {
        clear();
        key_ = std::move(computation_key);
    }

    // Record one fed token at `position`. The position must be exactly the current
    // size, so a skipped feed site (or a double feed) is caught here rather than
    // discovered later as a subtly wrong reuse.
    void feed(uint32_t position, uint32_t token_id) {
        if (position != ids_.size()) {
            throw std::logic_error(
                "PrefixRecord::feed: position " + std::to_string(position) +
                " is not the next expected position " + std::to_string(ids_.size()) +
                " — a feed site was skipped or repeated");
        }
        ids_.push_back(token_id);
    }

    uint32_t size() const noexcept { return static_cast<uint32_t>(ids_.size()); }

    // Pure. Decide whether `prompt` can be served by keeping the resident state.
    //
    // Order is load-bearing: an empty record is always `Cold` (nothing to keep); a
    // key change is checked before the tokens so a re-rendered template under a
    // different input is reported as such; a prompt that does not exceed the record
    // is `NoTail` (there would be no new token to sample); and only then is the
    // token prefix compared.
    ReuseDecision plan(const std::vector<uint32_t>& prompt,
                       const std::string& computation_key) const {
        ReuseDecision decision;
        if (ids_.empty()) {
            decision.verdict = ReuseVerdict::Cold;
            return decision;
        }
        // Exact string compare, not a hash: a collision here would silently serve
        // state built for a different computation, and the key is short.
        if (key_ != computation_key) {
            decision.verdict = ReuseVerdict::KeyChanged;
            return decision;
        }
        if (ids_.size() >= prompt.size()) {
            decision.verdict = ReuseVerdict::NoTail;
            return decision;
        }
        for (size_t index = 0; index < ids_.size(); ++index) {
            if (ids_[index] != prompt[index]) {
                decision.verdict = ReuseVerdict::Diverged;
                decision.diverged_at = static_cast<uint32_t>(index);
                return decision;
            }
        }
        decision.verdict = ReuseVerdict::Reused;
        decision.start = static_cast<uint32_t>(ids_.size());
        return decision;
    }

    static const char* verdict_name(ReuseVerdict verdict) {
        switch (verdict) {
            case ReuseVerdict::Cold: return "cold";
            case ReuseVerdict::Reused: return "reused";
            case ReuseVerdict::KeyChanged: return "key_changed";
            case ReuseVerdict::Diverged: return "diverged";
            case ReuseVerdict::NoTail: return "no_tail";
            case ReuseVerdict::StaleState: return "stale_state";
            case ReuseVerdict::Disabled: return "disabled";
        }
        return "cold";
    }

private:
    std::vector<uint32_t> ids_;
    std::string key_;
};

}  // namespace aeon::session
