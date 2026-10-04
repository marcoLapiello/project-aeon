// -----------------------------------------------------------------------------
// Gate — the prefix record: does this prompt extend the resident state?
//
// The engine keeps a model state across turns; the record is the only description
// of what that state covers, and a wrong verdict silently serves stale state or
// needlessly replays. So this gate pins every branch of `PrefixRecord::plan` and
// the feed discipline that keeps the record honest.
//
// Pure and CPU-only: no device, no model. The model-backed gate
// (`test_v4_prefix_reuse`) is what proves the record describes the *device* state;
// this proves the verdict logic is right in isolation.
//
// The record is deliberately all-or-nothing and checks the key before the tokens,
// so the cases below are the contract, not examples: empty, extension, equal,
// shorter, a mismatch at the first / a middle / the last recorded id, a key change
// under identical ids, and the gap guards.
// -----------------------------------------------------------------------------

#include "infrastructure/session/prefix_record.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using aeon::session::PrefixRecord;
using aeon::session::ReuseDecision;
using aeon::session::ReuseVerdict;

namespace {

uint32_t checks = 0;
uint32_t failures = 0;

void check(const char* label, bool ok, const std::string& detail = "") {
    std::printf("  %-58s %-34s %s\n", label, detail.c_str(), ok ? "PASS" : "FAIL");
    ++checks;
    if (!ok) ++failures;
}

PrefixRecord record_of(const std::vector<uint32_t>& ids, const std::string& key) {
    PrefixRecord record;
    record.begin(key);
    for (size_t index = 0; index < ids.size(); ++index) {
        record.feed(static_cast<uint32_t>(index), ids[index]);
    }
    return record;
}

bool feed_throws(PrefixRecord& record, uint32_t position, uint32_t token) {
    try {
        record.feed(position, token);
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("  the prefix record: is the resident state still the prompt's prefix?\n");
    std::printf("================================================================================\n");

    const std::string key = "chat|drop=true|low|bos|tools=none|fmt=none";
    const std::vector<uint32_t> resident = {5, 6, 7, 8};

    // --- the branches of `plan` ---------------------------------------------

    std::printf("\n[plan]\n");
    {
        PrefixRecord empty;
        const ReuseDecision decision = empty.plan(resident, key);
        check("empty record is Cold", decision.verdict == ReuseVerdict::Cold &&
                                          decision.start == 0,
              PrefixRecord::verdict_name(decision.verdict));
    }
    {
        const PrefixRecord record = record_of(resident, key);
        const std::vector<uint32_t> extended = {5, 6, 7, 8, 9, 10};
        const ReuseDecision decision = record.plan(extended, key);
        check("exact extension is Reused", decision.verdict == ReuseVerdict::Reused,
              PrefixRecord::verdict_name(decision.verdict));
        check("Reused start == record size", decision.start == resident.size(),
              std::to_string(decision.start) + " == " + std::to_string(resident.size()));
    }
    {
        const PrefixRecord record = record_of(resident, key);
        const ReuseDecision decision = record.plan(resident, key);
        check("equal prompt is NoTail", decision.verdict == ReuseVerdict::NoTail,
              PrefixRecord::verdict_name(decision.verdict));
        check("NoTail start == 0", decision.start == 0, std::to_string(decision.start));
    }
    {
        const PrefixRecord record = record_of(resident, key);
        const std::vector<uint32_t> shorter = {5, 6, 7};
        const ReuseDecision decision = record.plan(shorter, key);
        check("shorter prompt is NoTail", decision.verdict == ReuseVerdict::NoTail,
              PrefixRecord::verdict_name(decision.verdict));
    }
    {
        const PrefixRecord record = record_of(resident, key);
        const std::vector<uint32_t> changed = {99, 6, 7, 8, 9};
        const ReuseDecision decision = record.plan(changed, key);
        check("mismatch at 0 is Diverged", decision.verdict == ReuseVerdict::Diverged,
              PrefixRecord::verdict_name(decision.verdict));
        check("Diverged at 0 reports diverged_at 0", decision.diverged_at == 0,
              std::to_string(decision.diverged_at));
    }
    {
        const PrefixRecord record = record_of(resident, key);
        const std::vector<uint32_t> changed = {5, 6, 99, 8, 9};
        const ReuseDecision decision = record.plan(changed, key);
        check("mismatch in the middle is Diverged", decision.verdict == ReuseVerdict::Diverged,
              PrefixRecord::verdict_name(decision.verdict));
        check("Diverged reports the first differing index", decision.diverged_at == 2,
              std::to_string(decision.diverged_at));
    }
    {
        const PrefixRecord record = record_of(resident, key);
        const std::vector<uint32_t> changed = {5, 6, 7, 99, 9};
        const ReuseDecision decision = record.plan(changed, key);
        check("mismatch at the last recorded id is Diverged",
              decision.verdict == ReuseVerdict::Diverged,
              PrefixRecord::verdict_name(decision.verdict));
        check("Diverged at the last id reports index 3", decision.diverged_at == 3,
              std::to_string(decision.diverged_at));
    }
    {
        // Identical ids, different key: the key is checked first, so this is a key
        // change and not a reuse.
        const PrefixRecord record = record_of(resident, key);
        const std::vector<uint32_t> extended = {5, 6, 7, 8, 9};
        const ReuseDecision decision = record.plan(extended, "thinking|drop=false|high|bos|tools=none|fmt=none");
        check("identical ids under a changed key is KeyChanged",
              decision.verdict == ReuseVerdict::KeyChanged,
              PrefixRecord::verdict_name(decision.verdict));
        check("KeyChanged start == 0", decision.start == 0, std::to_string(decision.start));
    }
    {
        // A key match with a diverging prompt is reported as a token divergence,
        // which is the only place the two mechanisms are distinguishable.
        PrefixRecord record;
        record.begin(key);
        record.feed(0, 5);
        const std::vector<uint32_t> extended = {7, 6};
        const ReuseDecision decision = record.plan(extended, key);
        check("key match + token divergence is Diverged",
              decision.verdict == ReuseVerdict::Diverged,
              PrefixRecord::verdict_name(decision.verdict));
    }

    // --- the feed discipline ------------------------------------------------

    std::printf("\n[feed]\n");
    {
        PrefixRecord record;
        record.begin(key);
        record.feed(0, 5);
        record.feed(1, 6);
        check("feed grows size", record.size() == 2, std::to_string(record.size()));
        check("a skipped position throws", feed_throws(record, 3, 8),
              "gap 2");
        check("a double feed throws", feed_throws(record, 1, 7), "repeat 1");
        check("out-of-order (behind) throws", feed_throws(record, 0, 5), "behind 0");
        check("the record is unchanged by a refused feed", record.size() == 2,
              std::to_string(record.size()));
    }

    // --- clear / begin ------------------------------------------------------

    std::printf("\n[lifecycle]\n");
    {
        PrefixRecord record = record_of(resident, key);
        record.clear();
        check("clear empties the record", record.size() == 0, std::to_string(record.size()));
        const ReuseDecision decision = record.plan(resident, key);
        check("a cleared record plans Cold", decision.verdict == ReuseVerdict::Cold,
              PrefixRecord::verdict_name(decision.verdict));
    }
    {
        PrefixRecord record = record_of(resident, key);
        record.begin("another-key");
        check("begin empties the ids", record.size() == 0, std::to_string(record.size()));
        const ReuseDecision decision = record.plan(resident, "another-key");
        check("begin resets to Cold under the new key", decision.verdict == ReuseVerdict::Cold,
              PrefixRecord::verdict_name(decision.verdict));
    }

    // --- verdict names ------------------------------------------------------

    std::printf("\n[names]\n");
    {
        const ReuseVerdict all[] = {
            ReuseVerdict::Cold,    ReuseVerdict::Reused,  ReuseVerdict::KeyChanged,
            ReuseVerdict::Diverged, ReuseVerdict::NoTail, ReuseVerdict::StaleState,
            ReuseVerdict::Disabled,
        };
        bool all_named = true;
        for (ReuseVerdict verdict : all) {
            const char* name = PrefixRecord::verdict_name(verdict);
            if (name == nullptr || name[0] == '\0') all_named = false;
        }
        check("every verdict has a non-empty name", all_named, "7 verdicts");
    }

    std::printf("\n================================================================================\n");
    std::printf("  %u checks, %u failures\n", checks, failures);
    std::printf("================================================================================\n");
    return failures == 0 ? 0 : 1;
}
