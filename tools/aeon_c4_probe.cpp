// -----------------------------------------------------------------------------
// C4 probe — can the NVMe write straight into VRAM?
//
// ## The question
//
// Today an expert payload crosses PCIe **twice**: NVMe → pinned host RAM (the
// staging arena) → VRAM (an `hipMemcpyAsync`). C4 asks whether the SSD can DMA
// directly into VRAM, which would delete the pinned staging (3.44 GiB of
// non-reclaimable host memory, the open host-pressure item) and the second hop.
//
// The three preconditions are all green on this machine: a 32 GiB BAR aperture
// (`resource0`), `CONFIG_PCI_P2PDMA=y`, and `amdgpu.pcie_p2p=Y`. What is **not**
// verified — and what this probe decides — is the fourth precondition: whether the
// NVMe block layer will accept a **VRAM-backed user buffer as a read destination**.
// The GPU-to-GPU P2P matrix being all-YES does not answer it: that path uses the
// GPU's own copy engine, whereas this needs the *NVMe controller* to write into the
// GPU's BAR, which is a different code path (`get_user_pages` → `dma_map_sgtable`
// on a dma-buf page).
//
// ## The protocol, and why throughput alone would not settle it
//
//   [1] `hipMalloc` a region, export it as a dma-buf, `mmap` it.
//   [2] Prove the mapping really aliases VRAM (write on one side, read on the other).
//   [3] Measure the CPU-visible BAR bandwidth (the ceiling for any bounce).
//   [4] The decisive test: `pread(O_DIRECT)` straight into the mapped pointer.
//   [5] A **bounce control**: the same read into pinned host RAM, then a CPU copy
//       into the same mapping. If the direct read costs what the control costs, the
//       kernel bounced it and C4 buys nothing even though it "works".
//
// Step [5] is the part that matters. On this machine the drive does ~6.3 GB/s and
// the BAR ~12 GB/s, so a bounced direct-read would still be disk-bound and would
// *look* fast — the throughput alone cannot distinguish the two. The control makes
// the bounce explicit: it does by hand exactly what the kernel would do silently.
//
// ## The reference, for comparison
//
//   [6] The current production leg: `pread(O_DIRECT)` into pinned host RAM, then
//       `hipMemcpyAsync` H2D. This is what C4 would replace, and its cost is the
//       number C4 has to beat.
//
// ## Verdicts
//
//   NOT_EXPORTABLE  VRAM could not be exported as a dma-buf.            C4 dead.
//   NOT_SUPPORTED   `pread` into the mapping failed.                    C4 dead.
//   CORRUPT         `pread` succeeded but the bytes are wrong.          C4 unsafe.
//   BOUNCES         content correct, cost ≈ the bounce control.         C4 useless.
//   DIRECT          content correct, cost ≈ the disk bound.             C4 works.
//
// Usage: aeon_c4_probe [file] [megabytes]
//   Defaults: the swizzled expert container, 256 MiB.
// -----------------------------------------------------------------------------

#define _GNU_SOURCE 1

#include "platform/rdna3/device.hpp"

#include "infrastructure/io/direct_io_reader.hpp"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kSector = 4096;
constexpr int kReps = 5;

double ms_since(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check_hip(hipError_t error, const char* what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "c4_probe: %s: %s\n", what, hipGetErrorString(error));
        std::exit(1);
    }
}

double gib_per_s(size_t bytes, double milliseconds) {
    return milliseconds <= 0.0 ? 0.0
        : (static_cast<double>(bytes) / 1073741824.0) / (milliseconds / 1000.0);
}

// Count differing bytes between two buffers, and report the first.
size_t first_mismatch(const uint8_t* a, const uint8_t* b, size_t n, size_t* index) {
    size_t count = 0;
    *index = SIZE_MAX;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (*index == SIZE_MAX) *index = i;
            ++count;
        }
    }
    return count;
}

// Check a BAR mapping without one PCIe transaction per byte: copy it into host
// RAM in bulk, then compare there. A byte-at-a-time loop over a 256 MiB mapping
// is ~268M individual PCIe reads and appears to hang.
bool bar_is_all(const uint8_t* bar, size_t bytes, uint8_t value, size_t* first_bad) {
    std::vector<uint8_t> host(bytes);
    std::memcpy(host.data(), bar, bytes);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (size_t i = 0; i < bytes; ++i) {
        if (host[i] != value) {
            if (first_bad != nullptr) *first_bad = i;
            return false;
        }
    }
    return true;
}

// `pread` one aligned span, retrying nothing. Returns bytes read, or -1.
ssize_t pread_exact(int fd, void* dst, size_t bytes, off_t offset) {
    ssize_t got = ::pread(fd, dst, bytes, offset);
    return got;
}

} // namespace

int main(int argc, char** argv) {
    const std::string file =
        (argc > 1 && argv[1][0] != '\0')
            ? argv[1]
            : std::string("models/DeepSeek-V4-Flash-0731-INT4-W4A16-Aeon/"
                          "model_experts_swizzled.aeon");
    size_t bytes = (argc > 2 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10))
                             : 256ULL) * 1024ULL * 1024ULL;
    bytes = (bytes + kSector - 1) / kSector * kSector;

    std::printf(
        "==============================================================================\n"
        "  C4 probe — can the NVMe write directly into VRAM?\n"
        "==============================================================================\n"
        "  file  : %s\n"
        "  span  : %zu MiB (%zu bytes, %zu-aligned)\n"
        "  reps  : %d (distinct file offsets, min reported)\n\n",
        file.c_str(), bytes / (1024 * 1024), bytes, kSector, kReps);

    struct stat st{};
    if (::stat(file.c_str(), &st) != 0) {
        std::fprintf(stderr, "c4_probe: stat(%s): %s\n", file.c_str(), std::strerror(errno));
        return 1;
    }
    if (static_cast<size_t>(st.st_size) < bytes * kReps) {
        std::fprintf(stderr, "c4_probe: file is smaller than %zu spans of %zu bytes\n",
                     static_cast<size_t>(kReps), bytes);
        return 1;
    }

    // ---- [0] device -----------------------------------------------------------
    aeon::core::select_compute_device(true);
    (void)::hsa_init();

    // ---- [1] VRAM + dma-buf export + mmap -------------------------------------
    void* vram = nullptr;
    check_hip(hipMalloc(&vram, bytes), "hipMalloc(vram)");

    int dmabuf_fd = -1;
    uint64_t dmabuf_offset = 0;
    const char* export_how = "hipMemGetHandleForAddressRange";

    // HIP's own export is the one documented for `hipMalloc` pointers.
    {
        int fd = -1;
        const hipError_t e = hipMemGetHandleForAddressRange(
            &fd, reinterpret_cast<hipDeviceptr_t>(vram), bytes,
            hipMemRangeHandleTypeDmaBufFd, 0);
        if (e == hipSuccess && fd >= 0) {
            dmabuf_fd = fd;
        }
    }
    // Fallback: the HSA runtime's portable export.
    if (dmabuf_fd < 0) {
        export_how = "hsa_amd_portable_export_dmabuf";
        const hsa_status_t s =
            hsa_amd_portable_export_dmabuf(vram, bytes, &dmabuf_fd, &dmabuf_offset);
        if (s != HSA_STATUS_SUCCESS) {
            std::printf("  [1] dma-buf export               : FAILED (hip + hsa; hsa_status=%d)\n",
                        static_cast<int>(s));
            std::printf("\n  verdict: NOT_EXPORTABLE — VRAM cannot be exposed as a dma-buf here.\n"
                        "           C4 is impossible on this platform.\n");
            return 0;
        }
    }
    std::printf("  [1] dma-buf export               : ok  (fd=%d, offset=%llu, via %s)\n",
                dmabuf_fd, static_cast<unsigned long long>(dmabuf_offset), export_how);

    void* map = ::mmap(nullptr, bytes + dmabuf_offset, PROT_READ | PROT_WRITE, MAP_SHARED,
                       dmabuf_fd, 0);
    if (map == MAP_FAILED) {
        std::printf("  [1] mmap the dma-buf             : FAILED (%s)\n", std::strerror(errno));
        std::printf("\n  verdict: NOT_EXPORTABLE — the dma-buf cannot be mapped user-side.\n");
        return 0;
    }
    uint8_t* bar = static_cast<uint8_t*>(map) + dmabuf_offset;
    const bool aligned = (reinterpret_cast<uintptr_t>(bar) % kSector) == 0;
    std::printf("  [1] mmap the dma-buf             : ok  (ptr=%p, %zu-aligned: %s)\n",
                static_cast<void*>(bar), kSector, aligned ? "yes" : "NO");
    if (!aligned) {
        std::printf("\n  verdict: NOT_SUPPORTED — an unaligned mapping cannot be an O_DIRECT target.\n");
        return 0;
    }

    // ---- [2] the mapping aliases VRAM ----------------------------------------
    // GPU writes 0xA5 everywhere; the CPU must see it through `bar`.
    check_hip(hipMemset(vram, 0xA5, bytes), "hipMemset");
    check_hip(hipDeviceSynchronize(), "sync after memset");
    {
        size_t at = 0;
        const bool ok = bar_is_all(bar, bytes, 0xA5, &at);
        std::printf("  [2] GPU write -> CPU read        : %s\n",
                    ok ? "ok (mapping aliases VRAM)"
                       : ("MISMATCH at byte " + std::to_string(at)).c_str());
        if (!ok) {
            std::printf("\n  verdict: mapping is not a VRAM alias; the rest is meaningless.\n");
            return 0;
        }
    }
    // And the reverse: CPU writes, the GPU must read it back.
    {
        std::memset(bar, 0x3C, bytes);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::vector<uint8_t> back(bytes);
        check_hip(hipMemcpy(back.data(), vram, bytes, hipMemcpyDeviceToHost), "verify D2H");
        std::printf("  [2] CPU write -> GPU read        : %s\n",
                    back[0] == 0x3C && back[bytes / 2] == 0x3C && back[bytes - 1] == 0x3C
                        ? "ok"
                        : "MISMATCH");
    }

    // ---- [3] BAR bandwidth ----------------------------------------------------
    // Report both directions: a write-combined mapping posts writes but stalls
    // reads, and the asymmetry is the point (a CPU-mediated bounce would have to
    // pay whichever direction is slower).
    {
        std::vector<uint8_t> tmp(bytes);
        std::memset(tmp.data(), 0x11, bytes);
        double w = 0.0, r = 0.0;
        for (int i = 0; i < 3; ++i) {
            auto t0 = Clock::now();
            std::memcpy(bar, tmp.data(), bytes);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            w = std::max(w, gib_per_s(bytes, ms_since(t0)));
            t0 = Clock::now();
            std::memcpy(tmp.data(), bar, bytes);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            r = std::max(r, gib_per_s(bytes, ms_since(t0)));
        }
        std::printf("  [3] BAR bandwidth (CPU-visible)  : write %.2f GiB/s, read %.2f GiB/s\n", w, r);
    }

    // ---- [4]/[5]/[6] the reads ------------------------------------------------
    const int fd = ::open(file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        std::printf("  [-] open(O_DIRECT)              : FAILED (%s)\n", std::strerror(errno));
        return 1;
    }

    std::vector<uint8_t> pinned;
    pinned.resize(bytes);
    void* pinned_ptr = nullptr;
    if (hipHostMalloc(&pinned_ptr, bytes, hipHostMallocPortable) != hipSuccess) {
        std::printf("  [-] hipHostMalloc               : FAILED\n");
        return 1;
    }

    // [6] reference: the current production read leg (disk -> pinned host).
    double t_pinned = 1e18;
    for (int r = 0; r < kReps; ++r) {
        const off_t off = static_cast<off_t>(r) * bytes;
        const auto t0 = Clock::now();
        if (pread_exact(fd, pinned_ptr, bytes, off) < 0) {
            std::printf("  [-] pread into pinned host       : FAILED (%s)\n", std::strerror(errno));
            return 1;
        }
        t_pinned = std::min(t_pinned, ms_since(t0));
    }
    const double disk_rate = gib_per_s(bytes, t_pinned);
    std::printf("  [6] pread -> pinned host (ref)   : %7.2f ms  %.2f GiB/s  (disk ceiling)\n",
                t_pinned, disk_rate);

    // [5] bounce control: the reference read, then a CPU copy into the same mapping.
    double t_bounce = 1e18;
    for (int r = 0; r < kReps; ++r) {
        const off_t off = static_cast<off_t>(r) * bytes;
        const auto t0 = Clock::now();
        if (pread_exact(fd, pinned_ptr, bytes, off) < 0) { return 1; }
        std::memcpy(bar, pinned_ptr, bytes);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        t_bounce = std::min(t_bounce, ms_since(t0));
    }
    std::printf("  [5] pread -> host -> BAR copy    : %7.2f ms  %.2f GiB/s  (bounce control)\n",
                t_bounce, gib_per_s(bytes, t_bounce));

    // [4] the decisive test: O_DIRECT straight into the mapped VRAM.
    double t_direct = 1e18;
    ssize_t direct_ret = -1;
    int direct_errno = 0;
    for (int r = 0; r < kReps; ++r) {
        const off_t off = static_cast<off_t>(r) * bytes;
        errno = 0;
        const auto t0 = Clock::now();
        const ssize_t got = pread_exact(fd, bar, bytes, off);
        const double took = ms_since(t0);
        direct_ret = got;
        direct_errno = errno;
        if (got < 0) break;
        t_direct = std::min(t_direct, took);
    }

    std::printf("  [4] pread -> VRAM (the target)   : ");
    if (direct_ret < 0) {
        std::printf("FAILED (%s)\n", std::strerror(direct_errno));
    } else if (static_cast<size_t>(direct_ret) != bytes) {
        std::printf("PARTIAL (%zd of %zu)\n", direct_ret, bytes);
    } else {
        std::printf("%7.2f ms  %.2f GiB/s\n", t_direct, gib_per_s(bytes, t_direct));
    }

    // [4a] control: the same O_DIRECT read into a plain anonymous host mapping.
    // This must succeed; if it does, the [4] failure is specific to the VRAM
    // mapping and not a generic property of `pread(O_DIRECT)` here.
    {
        void* anon = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (anon == MAP_FAILED) {
            std::printf("  [4a] control: pread -> anon host  : mmap FAILED (%s)\n",
                        std::strerror(errno));
        } else {
            errno = 0;
            const ssize_t got = pread_exact(fd, anon, bytes, 0);
            std::printf("  [4a] control: pread -> anon host  : %s\n",
                        got == static_cast<ssize_t>(bytes)
                            ? "ok (O_DIRECT itself works here)"
                            : (std::string("FAILED (") + std::strerror(errno) + ")").c_str());
            ::munmap(anon, bytes);
        }
    }

    // [4b] control: a buffered read into the same VRAM mapping. A buffered read may
    // be accepted (it copies through the page cache) while O_DIRECT is refused —
    // which distinguishes "the kernel rejects P2P pages" from "the kernel rejects
    // VRAM as an O_DIRECT target specifically".
    {
        const int bfd = ::open(file.c_str(), O_RDONLY);
        if (bfd >= 0) {
            errno = 0;
            const ssize_t got = ::pread(bfd, bar, bytes, 0);
            std::printf("  [4b]     buffered pread -> VRAM    : %s\n",
                        got == static_cast<ssize_t>(bytes)
                            ? "ok (buffered path accepts VRAM)"
                            : (std::string("FAILED (") + std::strerror(errno) + ")").c_str());
            ::close(bfd);
        }
    }

    // [4c] the production mechanism itself: `io_uring` O_DIRECT into VRAM, through
    // the same `DirectIOReader` the engine uses. `pread` and `io_uring` both pin
    // user pages, so the outcome is expected to match [4] — but this is the exact
    // path C4 would have to run on, so it is measured rather than inferred.
    {
        const int ufd = ::open(file.c_str(), O_RDONLY | O_DIRECT);
        if (ufd >= 0) {
            try {
                aeon::io::DirectIOReader reader(/*queue_depth=*/8, /*force_async=*/true,
                                                /*sector_size=*/kSector);
                reader.submit_read(ufd, bar, bytes, 0, 0xCA4);
                reader.submit_pending_reads();
                const auto completion = reader.wait_for_completion();
                std::printf("  [4c] io_uring O_DIRECT -> VRAM      : %s (cqe.res=%d)\n",
                            completion.result < 0 ? "REJECTED" : "ok", completion.result);
            } catch (const std::exception& error) {
                std::printf("  [4c] io_uring O_DIRECT -> VRAM      : exception: %s\n", error.what());
            }
            ::close(ufd);
        }
    }

    // ---- [7] content check, if it read at all ---------------------------------
    bool corrupt = false;
    if (direct_ret == static_cast<ssize_t>(bytes)) {
        hipError_t e = hipMemcpy(pinned.data(), vram, bytes, hipMemcpyDeviceToHost);
        if (e != hipSuccess) {
            std::printf("  [7] read back VRAM               : FAILED (%s)\n", hipGetErrorString(e));
            corrupt = true;
        } else {
            // The file's first bytes are the container header; compare against a
            // plain buffered read of the same span.
            std::vector<uint8_t> expect(bytes);
            const int bfd = ::open(file.c_str(), O_RDONLY);
            const ssize_t want = bfd >= 0 ? ::pread(bfd, expect.data(), bytes, 0) : -1;
            if (bfd >= 0) ::close(bfd);
            if (want != static_cast<ssize_t>(bytes)) {
                std::printf("  [7] content check                : skipped (reference read short)\n");
            } else {
                size_t at = SIZE_MAX;
                const size_t bad = first_mismatch(expect.data(), pinned.data(), bytes, &at);
                corrupt = bad != 0;
                std::printf("  [7] content check                : %s\n",
                            corrupt ? (std::string("MISMATCH — ") + std::to_string(bad) +
                                       " bytes, first at " + std::to_string(at)).c_str()
                                    : "ok (bytes identical to a buffered read)");
            }
        }
    }

    // ---- verdict --------------------------------------------------------------
    std::printf("\n");
    if (direct_ret < 0 || static_cast<size_t>(direct_ret) != bytes) {
        std::printf("  verdict: NOT_SUPPORTED — the kernel refuses a VRAM buffer as an O_DIRECT\n"
                    "           read destination. C4 is dead on this platform.\n");
    } else if (corrupt) {
        std::printf("  verdict: CORRUPT — the read lands in VRAM but the bytes are wrong.\n"
                    "           C4 is unsafe without a driver fix.\n");
    } else {
        // The discriminator: is the direct read at the disk bound, or at the bounce
        // cost? A 15% margin is well outside run-to-run noise at these sizes.
        const double ratio = t_direct / t_bounce;
        std::printf("  direct %.2f ms | disk-bound %.2f ms | bounce %.2f ms | direct/bounce %.2f\n",
                    t_direct, t_pinned, t_bounce, ratio);
        if (ratio <= 0.85) {
            std::printf("  verdict: DIRECT — the NVMe wrote into VRAM without a host bounce.\n"
                        "           C4 is real: the pinned staging and the second PCIe hop can go.\n");
        } else {
            std::printf("  verdict: BOUNCES — the read succeeded but cost what the hand-written\n"
                        "           bounce costs, so the kernel copied through host memory.\n"
                        "           C4 buys nothing over the current leg.\n");
        }
    }

    ::close(fd);
    (void)hipHostFree(pinned_ptr);
    ::munmap(map, bytes + dmabuf_offset);
    if (dmabuf_fd >= 0) ::close(dmabuf_fd);
    (void)hipFree(vram);
    (void)::hsa_shut_down();
    return 0;
}
