#pragma once

// -----------------------------------------------------------------------------
// DSV4 stream decoder — token ids in, decoded text deltas out.
//
// The batch path decodes the whole reply at the end and strips thinking with a
// string search. Streaming cannot wait for the end, so this does the same split
// per token: everything before the thinking-end token is `Reasoning`, everything
// after is `Content`, and the marker itself is a structural token that is emitted
// nowhere. Two `Utf8Chunker`s, one per channel, hold a partial code point that
// straddles a token boundary and are flushed on the channel switch and at the end.
//
// Model knowledge (which token marks the switch) lives here, in G4, so the serving
// layer sees only channels and never a token id.
// -----------------------------------------------------------------------------

#include "architecture/deepseek_v4/text/dsv4_tokenizer.hpp"
#include "infrastructure/session/conversation.hpp"
#include "infrastructure/text/utf8_chunker.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace aeon::text {

class Dsv4StreamDecoder {
public:
    Dsv4StreamDecoder(const Dsv4Tokenizer& tokenizer, bool thinking)
        : tokenizer_(tokenizer),
          channel_(thinking ? session::Channel::Reasoning : session::Channel::Content) {}

    void push(uint32_t token_id, std::vector<session::TextDelta>& out) {
        if (token_id == tokenizer_.eos_token_id()) return;
        if (token_id == tokenizer_.thinking_end_token_id()) {
            // Release whatever the reasoning channel held, then move on: the marker
            // is structure, not text.
            emit(chunker().flush(), out);
            channel_ = session::Channel::Content;
            return;
        }
        const std::string text = tokenizer_.decode({token_id});
        emit(chunker().push(text), out);
    }

    void finish(std::vector<session::TextDelta>& out) {
        emit(chunker().flush(), out);
    }

private:
    Utf8Chunker& chunker() {
        return channel_ == session::Channel::Reasoning ? reasoning_chunker_ : content_chunker_;
    }

    void emit(std::string text, std::vector<session::TextDelta>& out) {
        if (text.empty()) return;
        out.push_back(session::TextDelta{channel_, std::move(text)});
    }

    const Dsv4Tokenizer& tokenizer_;
    session::Channel channel_;
    Utf8Chunker reasoning_chunker_;
    Utf8Chunker content_chunker_;
};

}  // namespace aeon::text
