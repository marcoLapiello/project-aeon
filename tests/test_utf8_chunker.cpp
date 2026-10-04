// -----------------------------------------------------------------------------
// Gate — the UTF-8 holdback.
//
// A byte-level BPE can end a token inside a multi-byte code point, so a streamed
// delta must not emit a partial sequence. This gate splits a valid string at every
// byte boundary and requires the concatenated output to reproduce the input
// exactly; it feeds an emoji one byte at a time; and it checks that malformed
// input (a lone continuation, an overlong form, an invalid lead, a truncated tail)
// comes out as valid UTF-8 with U+FFFD where the bytes were unusable.
//
// The validator here is written independently of the chunker, so "the output is
// valid UTF-8" is a real check and not the chunker agreeing with itself.
// -----------------------------------------------------------------------------

#include "infrastructure/text/utf8_chunker.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

const char* const kReplacement = "\xEF\xBF\xBD";

// Independent UTF-8 validator: maximal-length sequences, no overlong forms, no
// surrogates, no value past U+10FFFF. Shares no code with `Utf8Chunker`.
bool is_valid_utf8(std::string_view text) {
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
            bool ok = byte >= 0x80 && byte <= 0xBF;
            if (ok && k == 1) {
                if (lead == 0xE0) ok = byte >= 0xA0;
                else if (lead == 0xED) ok = byte <= 0x9F;
                else if (lead == 0xF0) ok = byte >= 0x90;
                else if (lead == 0xF4) ok = byte <= 0x8F;
            }
            if (!ok) return false;
        }
        index += length;
    }
    return true;
}

// Push `input` in pieces cut at every offset in `cuts`, then flush; return the
// concatenation.
std::string push_in_pieces(std::string_view input, const std::vector<size_t>& cuts) {
    aeon::text::Utf8Chunker chunker;
    std::string output;
    size_t start = 0;
    for (const size_t cut : cuts) {
        output += chunker.push(input.substr(start, cut - start));
        start = cut;
    }
    output += chunker.push(input.substr(start));
    output += chunker.flush();
    return output;
}

}  // namespace

int main() {
    // A string with one code point of each width: 'A' (1), 'é' (2), '€' (3),
    // '😀' (4).
    const std::string mixed = "A\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80";

    // Split at every byte boundary: the concatenation must equal the input, and
    // every piece must be valid UTF-8.
    for (size_t cut = 0; cut <= mixed.size(); ++cut) {
        aeon::text::Utf8Chunker chunker;
        const std::string first = chunker.push(mixed.substr(0, cut));
        const std::string second = chunker.push(mixed.substr(cut));
        const std::string flushed = chunker.flush();
        assert(is_valid_utf8(first));
        assert(is_valid_utf8(second));
        assert(is_valid_utf8(flushed));
        assert(first + second + flushed == mixed);
    }

    // The emoji fed one byte per push: nothing is emitted until the fourth byte,
    // then the whole code point comes out at once.
    {
        aeon::text::Utf8Chunker chunker;
        const std::string emoji = "\xF0\x9F\x98\x80";
        std::string output;
        output += chunker.push(emoji.substr(0, 1));
        output += chunker.push(emoji.substr(1, 1));
        output += chunker.push(emoji.substr(2, 1));
        output += chunker.push(emoji.substr(3, 1));
        output += chunker.flush();
        assert(output == emoji);
        assert(is_valid_utf8(output));
    }

    // A lead byte alone is held, then completed by the next push.
    {
        aeon::text::Utf8Chunker chunker;
        assert(chunker.push("\xE2") == "");
        assert(chunker.push("\x82\xAC") == "\xE2\x82\xAC");
        assert(chunker.flush() == "");
    }

    // A lone continuation byte is a replacement.
    {
        aeon::text::Utf8Chunker chunker;
        const std::string output = chunker.push("\x80") + chunker.flush();
        assert(output == kReplacement);
        assert(is_valid_utf8(output));
    }

    // An invalid lead byte is a replacement.
    {
        aeon::text::Utf8Chunker chunker;
        const std::string output = chunker.push("\xF5") + chunker.flush();
        assert(output == kReplacement);
        assert(is_valid_utf8(output));
    }

    // A truncated tail at flush becomes a single replacement.
    {
        aeon::text::Utf8Chunker chunker;
        assert(chunker.push("\xE2\x82") == "");
        assert(chunker.flush() == kReplacement);
        assert(chunker.flush() == "");  // idempotent once released
    }

    // An overlong form is rejected, not decoded: the output is replacement(s) and
    // never the overlong bytes.
    {
        aeon::text::Utf8Chunker chunker;
        const std::string output = chunker.push("\xC0\xAF") + chunker.flush();
        assert(output != "\xC0\xAF");
        assert(is_valid_utf8(output));
        assert(output.find("\xC0") == std::string::npos);
    }

    // A valid continuation that follows an invalid lead is still re-examined: the
    // two-byte prefix overlong form yields two replacements, not one.
    {
        aeon::text::Utf8Chunker chunker;
        const std::string output = chunker.push("\xE0\x80") + chunker.flush();
        assert(is_valid_utf8(output));
        assert(output == std::string(kReplacement) + kReplacement);
    }

    // A sequence whose second byte is invalid is not held: it is recognised and
    // replaced at once, and the following ASCII byte still comes through.
    {
        aeon::text::Utf8Chunker chunker;
        const std::string output = chunker.push("\xE2\x28") + chunker.push("A") + chunker.flush();
        assert(output == std::string(kReplacement) + "(" + "A");
    }

    // Well-formed input split every which way always reproduces itself.
    {
        const std::vector<std::string> samples = {
            "", "hello", mixed, "\xF0\x9F\x98\x80\xF0\x9F\x98\x81",
            "a\xC3\xA9" "b\xE2\x82\xAC" "c\xF0\x9F\x98\x80" "d"};
        for (const std::string& sample : samples) {
            for (size_t cut = 0; cut <= sample.size(); ++cut) {
                assert(push_in_pieces(sample, {cut}) == sample);
            }
        }
    }

    std::cout << "[PASS] UTF-8 holdback" << std::endl;
    return 0;
}
