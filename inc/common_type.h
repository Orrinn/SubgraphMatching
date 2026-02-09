#pragma once

#include <stdint.h>
#include <vector>
#include <bitset>
#include <chrono>
#include "config.h"

#include <sstream>
#include <fmt/format.h>
#include <fmt/ranges.h>

#if DEBUG == 1 && SPDLOG == 1
#define LOG_OUTPUT
#include <spdlog/spdlog.h>
#endif

#define INVALID ((uint32_t)-1)
#define EIGEN_INDEX

// #define LARGE_QUERY

#ifdef LARGE_QUERY

#define MAX_QUERY_VERTEX (320)

#else

#define MAX_QUERY_VERTEX (64)

#endif

#define CHUNK_NUM (MAX_QUERY_VERTEX / 64)


// #define BITMASK ((MAX_QUERY_VERTEX == 64) ? ~0ULL : ((1ULL << MAX_QUERY_VERTEX) - 1)) // 0b111111....111

typedef uint32_t VertexID;
typedef uint32_t LabelID;

typedef std::pair<VertexID, VertexID> PartialOrder;
typedef std::vector<PartialOrder> Restriction;

typedef std::vector<VertexID> PreviousNeb; // ! NOTE:  prevNeb use VERTEX as idx, NOT the depth

typedef std::vector<VertexID> FollowingNeb; // ! NOTE:  prevNeb use VERTEX as idx, NOT the depth

typedef std::vector<VertexID> Order;

// typedef std::bitset<MAX_QUERY_VERTEX> Mask;
// typedef boost::dynamic_bitset<> Mask;

class Mask{
    uint64_t _mask[CHUNK_NUM] = {0};
public:
    void inline set(size_t idx) {
        _mask[idx / 64] |= (1ULL << (idx % 64));
    }

    void inline set(){
        for (size_t i = 0; i < CHUNK_NUM; ++i) {
            _mask[i] = ~0ULL; // set all bits to 1
        }
    }

    void inline reset(size_t idx) {
        _mask[idx / 64] &= ~(1ULL << (idx % 64));
    }

    void inline reset(){
        for (size_t i = 0; i < CHUNK_NUM; ++i) {
            _mask[i] = 0;
        }
    }

    bool inline test(size_t idx) const {
        return (_mask[idx / 64] & (1ULL << (idx % 64))) != 0;
    }

    bool inline any() const {
        for (size_t i = 0; i < CHUNK_NUM; i++) {
            if (_mask[i] != 0) return true;
        }
        return false;
    }

    size_t inline count() const {
        size_t total = 0;
        for (auto m : _mask) {
            total += __builtin_popcountll(m); // GCC/Clang
        }
        return total;
    }

    bool inline msb_index(uint32_t &index) const {
        for (int i = (CHUNK_NUM) - 1; i >= 0; --i) {
            uint64_t chunk = _mask[i];
            if (chunk != 0) {
                index = i * 64 + (63 - __builtin_clzll(chunk));
                return true;
            }
        }
        return false;
    }

    Mask operator|(const Mask &other) const {
        Mask result;
        for (size_t i = 0; i < CHUNK_NUM; i++) {
            result._mask[i] = _mask[i] | other._mask[i];
        }
        return result;
    }

    Mask &operator|=(const Mask &other) {
        for (size_t i = 0; i < CHUNK_NUM; i++) {
            _mask[i] |= other._mask[i];
        }
        return *this;
    }
    
    Mask operator&(const Mask &other) const {
        Mask result;
        for (size_t i = 0; i < CHUNK_NUM; i++) {
            result._mask[i] = _mask[i] & other._mask[i];
        }
        return result;
    }

    bool operator[](size_t idx) const {
        return (_mask[idx / 64] >> (idx % 64)) & 1ULL;
    }

};

std::string extractFileName(const std::string& filePath){
    
    size_t lastSlash = filePath.find_last_of("/\\");

    std::string fileName = filePath.substr(lastSlash + 1);
    
    size_t lastDot = fileName.find_last_of('.');
    if (lastDot != std::string::npos) {
        fileName = fileName.substr(0, lastDot);
    }

    return fileName;
}


class Profiler{
public:
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    static bool constexpr useProfiling = true;

    static Profiler& getInst(){
        static thread_local Profiler inst;
        return inst;
    }

    typedef struct _ps{
        uint32_t cnt{0};
        size_t mem_overhead{0}; // mem in Byte
        double mem_overhead_KB{0};
        std::string info(){
            return fmt::format("prune cnt {}, mem_overhead {}", cnt, mem_overhead);
        }
        void inline reset(){
            cnt=0;
            mem_overhead=0;
            mem_overhead_KB=0.0;
        }
    }PruneStatistics;

    PruneStatistics daf_fp;
    PruneStatistics gup_fp;
    PruneStatistics gup_cd;
    PruneStatistics bice_cd;
    PruneStatistics veq_ap;
    PruneStatistics bice_ap;

    uint32_t sample_index{13};
    size_t sample_inter = ((1 << sample_index));

    size_t total_intersection{0};
    double prune_init_time{0.0};
    size_t filter_hit_count[300];
    size_t conflict_count[300]; // per layer;
    size_t total_iter_count[300]; // per layer;
    size_t pruned_count[300]; // per layer;
    size_t pruned_iteration[300];
    std::vector<std::chrono::high_resolution_clock::time_point> timestamps; 
    std::chrono::high_resolution_clock::time_point start_time;

    void reset(){
        total_intersection = 0;
        prune_init_time = 0.0;
        daf_fp.reset();
        gup_fp.reset();
        gup_cd.reset();
        bice_cd.reset();
        veq_ap.reset();
        bice_ap.reset();
        memset(filter_hit_count, 0, sizeof(size_t) * 300);
        memset(total_iter_count, 0, sizeof(size_t) * 300);
        memset(conflict_count, 0, sizeof(size_t) * 300);
        memset(pruned_count, 0, sizeof(size_t) * 300);
        memset(pruned_iteration, 0, sizeof(size_t) * 300);
        timestamps.clear();
    }

    std::string info(){
        std::ostringstream oss;
        oss << fmt::format("DAF Failing Prune: {}\n", daf_fp.info());
        oss << fmt::format("GuP Failing Prune: {}\n", gup_fp.info());
        oss << fmt::format("GuP Conflict Detection: {}\n", gup_cd.info());
        oss << fmt::format("BICE Conflict Detection: {}\n", bice_cd.info());
        oss << fmt::format("VEQ Automorphism Prune: {}\n", veq_ap.info());
        oss << fmt::format("BICE Automorphism Prune: {}\n", bice_ap.info());
        oss << fmt::format("Total Intersection: {}\n", total_intersection);
        return oss.str();
    }

private:
    Profiler() = default;
};

