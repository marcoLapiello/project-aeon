// -----------------------------------------------------------------------------
// Gate — the DSV4 stream decoder.
//
// The batch path (decode the whole reply, strip thinking with a string search) is
// the oracle: the streamed deltas must reassemble to exactly the same text, split
// at the thinking marker, with nothing lost to a code point that straddles a token
// boundary and nothing emitted that is not valid UTF-8.
//
// Each fixed token sequence is fed one token at a time — the only granularity a
// push has — and the concatenation of the two channels is checked against the
// batch decode.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/text/dsv4_stream_decoder.hpp"
#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using aeon::session::Channel;
using aeon::session::TextDelta;
using aeon::text::Dsv4StreamDecoder;
using aeon::text::Dsv4Tokenizer;

namespace {

// Reimplemented from the definition (not shared with the decoder) so the check is
// independent: the text after the last thinking marker.
std::string strip_marker(const std::string& text, const std::string& marker) {
    if (marker.empty()) return text;
    const size_t position = text.rfind(marker);
    return position == std::string::npos ? text : text.substr(position + marker.size());
}

bool is_valid_utf8(const std::string& text) {
    size_t index = 0;
    while (index < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        size_t length = 0;
        if (lead < 0x80) length = 1;
        else if (lead >= 0xC2 && lead <= 0xDF) length = 2;
        else if (lead >= 0xE0 && lead <= 0xEF) length = 3;
        else if (lead >= 0xF0 && lead <= 0xF4) length = 4;
        else return false;
        if (index + length > text.size()) return false;
        for (size_t k = 1; k < length; ++k) {
            const unsigned char byte = static_cast<unsigned char>(text[index + k]);
            if (byte < 0x80 || byte > 0xBF) return false;
        }
        index += length;
    }
    return true;
}

struct Streamed {
    std::string reasoning;
    std::string content;
    bool channel_went_backwards{false};
    bool all_valid_utf8{true};
    size_t delta_count{0};
};

Streamed stream(const Dsv4Tokenizer& tokenizer, const std::vector<uint32_t>& ids,
                bool thinking) {
    Dsv4StreamDecoder decoder(tokenizer, thinking);
    std::vector<TextDelta> deltas;
    for (const uint32_t id : ids) decoder.push(id, deltas);
    decoder.finish(deltas);

    Streamed out;
    bool seen_content = false;
    for (const TextDelta& delta : deltas) {
        ++out.delta_count;
        if (!is_valid_utf8(delta.text)) out.all_valid_utf8 = false;
        if (delta.channel == Channel::Content) seen_content = true;
        if (delta.channel == Channel::Reasoning) {
            if (seen_content) out.channel_went_backwards = true;
            out.reasoning += delta.text;
        } else {
            out.content += delta.text;
        }
    }
    return out;
}

// The decoded text of `ids` with a trailing EOS removed, special tokens kept (so
// the thinking marker survives).
std::string batch_decode(const Dsv4Tokenizer& tokenizer, std::vector<uint32_t> ids) {
    if (!ids.empty() && ids.back() == tokenizer.eos_token_id()) ids.pop_back();
    return tokenizer.decode(ids);
}

uint32_t checks = 0;
uint32_t failures = 0;

void expect(bool ok, const std::string& label) {
    ++checks;
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << label << "\n";
    if (!ok) ++failures;
}

void check_sequence(const Dsv4Tokenizer& tokenizer, const std::string& name,
                    const std::vector<uint32_t>& ids, bool thinking) {
    const std::string marker = tokenizer.decode({tokenizer.thinking_end_token_id()});
    const std::string full = batch_decode(tokenizer, ids);
    const bool has_marker =
        !marker.empty() &&
        std::find(ids.begin(), ids.end(), tokenizer.thinking_end_token_id()) != ids.end();

    const Streamed result = stream(tokenizer, ids, thinking);

    expect(result.all_valid_utf8, name + ": every delta is valid UTF-8");
    expect(!result.channel_went_backwards, name + ": channel never returns to reasoning");

    if (has_marker) {
        expect(result.reasoning + marker + result.content == full,
               name + ": reasoning + marker + content == batch decode");
        expect(result.content == strip_marker(full, marker),
               name + ": content == strip_thinking(batch decode)");
    } else if (thinking) {
        expect(result.reasoning == full && result.content.empty(),
               name + ": thinking with no marker is all reasoning");
    } else {
        expect(result.content == full && result.reasoning.empty(),
               name + ": chat with no marker is all content");
    }
}

}  // namespace

int main() {
    try {
        Dsv4Tokenizer tokenizer;
        tokenizer.load("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/tokenizer.aeon");

        std::cout << "[Test] DSV4 stream decoder\n";

        const uint32_t marker = tokenizer.thinking_end_token_id();
        const uint32_t eos = tokenizer.eos_token_id();

        // ASCII, no marker, thinking mode.
        const std::vector<uint32_t> ascii = tokenizer.encode("Hello, world!");
        check_sequence(tokenizer, "ascii", ascii, true);

        // CJK: multi-byte code points inside and across tokens.
        check_sequence(tokenizer, "cjk", tokenizer.encode("你好世界，这是一个测试。"), true);

        // Emoji, which a byte-level BPE may split across several tokens.
        check_sequence(tokenizer, "emoji", tokenizer.encode("😀🎉🚀"), true);

        // A marker in the middle, thinking mode: reasoning before, content after.
        {
            std::vector<uint32_t> ids = tokenizer.encode("let me think");
            ids.push_back(marker);
            for (const uint32_t id : tokenizer.encode("the answer is 4")) ids.push_back(id);
            check_sequence(tokenizer, "marker-middle", ids, true);
        }

        // The same with a trailing EOS: the EOS must be dropped from the output.
        {
            std::vector<uint32_t> ids = tokenizer.encode("reasoning here");
            ids.push_back(marker);
            for (const uint32_t id : tokenizer.encode("visible")) ids.push_back(id);
            ids.push_back(eos);
            check_sequence(tokenizer, "marker-eos", ids, true);
        }

        // No EOS, no marker (thinking off): everything is content.
        check_sequence(tokenizer, "chat-no-marker", tokenizer.encode("no thinking here"),
                       false);

        // A UTF-8 code point split by byte-level tokens: feed the emoji's own
        // encoded ids one at a time; the holdback must reassemble it.
        {
            const std::vector<uint32_t> ids = tokenizer.encode("café — 😀");
            check_sequence(tokenizer, "multibyte", ids, false);
        }

        std::cout << "[Test] " << (checks - failures) << "/" << checks << " checks passed"
                  << std::endl;
        if (failures != 0) return 1;
        std::cout << "[PASS] DSV4 stream decoder" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] DSV4 stream decoder: " << error.what() << std::endl;
        return 1;
    }
}
