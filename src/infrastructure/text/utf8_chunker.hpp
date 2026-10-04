#pragma once

// -----------------------------------------------------------------------------
// UTF-8 holdback.
//
// A streamed reply arrives as tokens, and the byte-level BPE can end a token
// *inside* a multi-byte code point: the last one to three bytes of a UTF-8
// sequence can be the first token of the next delta. A delta that ends mid-code-
// point is not valid UTF-8, and wrapping it in a JSON string would emit bytes no
// decoder accepts.
//
// `push` therefore emits only the longest prefix of *complete* code points and
// holds an incomplete tail until the bytes that finish it arrive. `flush` releases
// the held tail at the end of the reply, turning the truncated sequence into
// U+FFFD. Invalid bytes (a stray continuation, an overlong form, a surrogate, or a
// value past U+10FFFF) are emitted as U+FFFD the moment they are recognised, so
// everything returned is valid UTF-8 — and therefore valid inside a JSON string.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <string>
#include <string_view>

namespace aeon::text {

class Utf8Chunker {
public:
    // The longest prefix of `bytes` made of complete code points, after any bytes
    // held from a previous push. An incomplete tail (1 to 3 bytes) is held, not
    // emitted.
    std::string push(std::string_view bytes) {
        std::string output;
        buffer_.append(held_);
        held_.clear();
        buffer_.append(bytes.data(), bytes.size());

        size_t index = 0;
        while (index < buffer_.size()) {
            const unsigned char lead = static_cast<unsigned char>(buffer_[index]);
            if (lead < 0x80) {
                output.push_back(static_cast<char>(lead));
                ++index;
                continue;
            }
            const size_t length = sequence_length(lead);
            if (length == 0) {
                append_replacement(output);
                ++index;
                continue;
            }
            bool valid = true;
            bool complete = true;
            for (size_t k = 1; k < length; ++k) {
                if (index + k >= buffer_.size()) {
                    complete = false;
                    break;
                }
                if (!continuation_ok(lead, k,
                                     static_cast<unsigned char>(buffer_[index + k]))) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                // The maximal invalid subpart is the lead byte alone; emit one
                // replacement and resync from the byte that failed.
                append_replacement(output);
                ++index;
                continue;
            }
            if (!complete) {
                held_.assign(buffer_, index, std::string::npos);
                break;
            }
            output.append(buffer_, index, length);
            index += length;
        }

        buffer_.clear();
        return output;
    }

    // The held tail, if any, as a single U+FFFD. Only a valid but truncated prefix
    // is ever held, so one replacement is the whole of it.
    std::string flush() {
        std::string output;
        if (!held_.empty()) {
            append_replacement(output);
            held_.clear();
        }
        buffer_.clear();
        return output;
    }

private:
    static size_t sequence_length(unsigned char lead) {
        if (lead >= 0xC2 && lead <= 0xDF) return 2;
        if (lead >= 0xE0 && lead <= 0xEF) return 3;
        if (lead >= 0xF0 && lead <= 0xF4) return 4;
        return 0;  // 0x80-0xBF stray continuation, 0xC0/0xC1 overlong, 0xF5-0xFF
    }

    // Whether the `k`-th byte of a sequence led by `lead` is allowed, rejecting the
    // overlong (E0/F0), surrogate (ED) and out-of-range (F4) sub-ranges.
    static bool continuation_ok(unsigned char lead, size_t k, unsigned char byte) {
        if (k > 1) return byte >= 0x80 && byte <= 0xBF;
        switch (lead) {
            case 0xE0: return byte >= 0xA0 && byte <= 0xBF;
            case 0xED: return byte >= 0x80 && byte <= 0x9F;
            case 0xF0: return byte >= 0x90 && byte <= 0xBF;
            case 0xF4: return byte >= 0x80 && byte <= 0x8F;
            default:   return byte >= 0x80 && byte <= 0xBF;
        }
    }

    static void append_replacement(std::string& output) {
        output += "\xEF\xBF\xBD";  // U+FFFD
    }

    std::string held_;
    std::string buffer_;
};

} // namespace aeon::text
