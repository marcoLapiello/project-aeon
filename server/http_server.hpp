#pragma once

// -----------------------------------------------------------------------------
// HttpServer — the G5 transport.
//
// A thin shell over the vendored HTTP library: it parses a request with the OpenAI
// codec, submits a job to the conversation service, and serialises the job's event
// stream back out (chunked for `stream = true`, one object otherwise). It names no
// model or engine type — only the neutral seam and the service — so it links no
// HIP runtime, which is the build-level proof of R8.
// -----------------------------------------------------------------------------

#include "server/conversation_service.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace aeon::server {

struct HttpOptions {
    std::string host{"127.0.0.1"};
    int port{8080};
    size_t max_queue{16};
};

class HttpServer {
public:
    HttpServer(ConversationService& service, HttpOptions options);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Binds and starts listening on its own thread. When `options.port` is 0 the
    // kernel picks a port; `port()` reports the bound one.
    void start();

    int port() const;

    // Stops listening and joins the listen thread.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aeon::server
