#pragma once
#include "graph.h"
#include <algorithm>
#include <vector>
#include <unordered_map>

#include "common_ops.h"
#include "common_type.h"
#include "api.h"
#include <limits>
#include <type_traits>
#include <variant>
#include <map>
#include <unordered_set>

#ifdef LARGE_QUERY

inline uint64_t get_block(const Mask& x, size_t block) {
    uint64_t chunk = 0;
    for (size_t j = 0; j < 64; ++j) {
        if (x[block * 64 + j])
            chunk |= (1ULL << j);
    }
    return chunk;
}

bool inline GetMSB_Index(const Mask& x, uint32_t &index) {
    // static_assert(MAX_QUERY_VERTEX == 64);
    static_assert(MAX_QUERY_VERTEX == 320);

    constexpr size_t chunk_size = 64; // Number of bits in a uint64_t
    constexpr size_t num_chunks = MAX_QUERY_VERTEX / chunk_size;

    for (int i = num_chunks - 1; i >= 0; --i) {
        uint64_t bits = get_block(x, i);
        if (bits != 0) {
            index = i * 64 + (63 - __builtin_clzll(bits));
            return true;
        }
    }

    return false;
    // uint64_t bits = x.to_ullong();
    
    // if(bits == 0) return false;
    // else{
    //     index = 63 - __builtin_clzll(bits);
    //     return true;
    // }
}

#else

bool inline GetMSB_Index(const Mask& x, uint32_t &index) {
    // static_assert(MAX_QUERY_VERTEX == 64);
    // uint64_t bits = x.to_ullong();
    
    return x.msb_index(index);
}

#endif

class EmptyFailingPrune{
public:
    constexpr EmptyFailingPrune() noexcept = default;
    ~EmptyFailingPrune() noexcept = default;

    // empty methods. this will be automatically eliminated when compiling
    constexpr void Init(const Query &query, const Order& order) const noexcept {}
    
    constexpr void ExtendIndex(uint32_t cur_depth, VertexID cur_v) const noexcept {}
    constexpr void ReduceIndex(uint32_t cur_depth, VertexID cur_v) const noexcept {}

    constexpr bool PruneCheck(uint32_t cur_depth) const noexcept {return false;}

    constexpr void SuccessMatch(uint32_t cur_depth, VertexID cur_v) const noexcept {}
    constexpr bool NoCandidatesConflictCheck(uint32_t cnt, uint32_t cur_depth) const noexcept {return false;}
    constexpr void InjectiveConflict(uint32_t cur_depth, VertexID conflict_u_depth) const noexcept {}
};


class DAFFailingPrune{
public:
    std::vector<Mask> _fail_mask;
    std::vector<Mask> _anscestor;
public:

    void Init(const Query &query, const Order& order) {
        int qvcnt = query.getVertexCnt();
        PreviousNeb *prevNeb = new PreviousNeb[qvcnt];
        GetPreviousNeb(query, order, prevNeb);

        _fail_mask.resize(qvcnt);
        _anscestor.resize(qvcnt);

        uint32_t qv_deps[qvcnt];
        for(int i=0; i<qvcnt; i++)
            qv_deps[order[i]] = i;

        for(int i=0; i<qvcnt; i++){
            _anscestor[i].set(i);
            for(VertexID upre: prevNeb[order[i]])
                _anscestor[i] |= _anscestor[qv_deps[upre]];
        }
        delete[] prevNeb;

        if constexpr (Profiler::useProfiling){
            size_t mem = 0;
            mem += (sizeof(Mask) * qvcnt);
            mem += (sizeof(Mask) * qvcnt);
            Profiler::getInst().daf_fp.mem_overhead = mem;
            Profiler::getInst().daf_fp.mem_overhead_KB = ((double)mem) / (1024.0); 
        }
    }
    
    constexpr void ExtendIndex(uint32_t cur_depth, VertexID cur_v) const noexcept {}
    constexpr void ReduceIndex(uint32_t cur_depth, VertexID cur_v) const noexcept {}

    bool inline PruneCheck(uint32_t cur_depth) {
        if(cur_depth != 0) [[likely]] {
            if (!_fail_mask[cur_depth].test(cur_depth)) {
                _fail_mask[cur_depth - 1] = _fail_mask[cur_depth];
                if constexpr (Profiler::useProfiling){
                    Profiler::getInst().daf_fp.cnt++;
                }
#ifdef LOG_OUTPUT
        spdlog::trace("Prune by failing_set, depth: {}, _fail_mask[{}]: {}", 
            cur_depth, cur_depth,
            [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<_fail_mask.size(); i++) vec.push_back(_fail_mask[cur_depth - 1][i]);
                return fmt::format("{}", fmt::join(vec, ","));
            }()
        );
#endif
                return true;
            }
            else _fail_mask[cur_depth - 1] |= _fail_mask[cur_depth];
        }
        return false;
    }

    void inline SuccessMatch(uint32_t cur_depth, VertexID cur_v) {
        _fail_mask[cur_depth].set();
        _fail_mask[cur_depth - 1] |= _fail_mask[cur_depth];
    }

    bool inline NoCandidatesConflictCheck(uint32_t cnt, uint32_t cur_depth) {
        if(cnt == 0){
            _fail_mask[cur_depth - 1] = _anscestor[cur_depth];
            return true;
        }
        else _fail_mask[cur_depth - 1].reset();
        return false;
    }

    bool inline NoCandidatesConflictCheckCache(uint32_t cnt, uint32_t cur_depth, uint32_t conf_dep) {
        if(cnt == 0){
            _fail_mask[cur_depth - 1] = _anscestor[conf_dep];
            return true;
        }
        else _fail_mask[cur_depth - 1].reset();
        return false;
    }

    void inline InjectiveConflict(uint32_t cur_depth, VertexID conflict_u_depth) {
        _fail_mask[cur_depth] = _anscestor[cur_depth];
        _fail_mask[cur_depth] |= _anscestor[conflict_u_depth];
        _fail_mask[cur_depth - 1] |= _fail_mask[cur_depth];
    }
};

class GUPFailingPrune{
public:
    
    // search-node optimization
    typedef struct{
        size_t search_id{0}; // age 
        uint32_t minimal_sup_len{0}; // u
        Mask _fail_u_mask;

        std::string debug_info(uint32_t qvcnt){
            std::vector<int> m;
            for(int i=0; i<qvcnt; i++) m.push_back(_fail_u_mask[i]);
            if(search_id == numeric_limits<size_t>::max())
                return fmt::format("search_id MAX, sup_len {}, mask {}", minimal_sup_len, fmt::join(m,""));
            else
                return fmt::format("search_id {}, sup_len {}, mask {}", search_id, minimal_sup_len, fmt::join(m,""));
        };
    }NoGood_V;

    struct MaskState{
        bool _useMask{true}; // err{}
        Mask _mask;

        MaskState(): _useMask(true){
            _mask.set();
        };

        void inline setUse(){
            _useMask = true;
            _mask.set();
        };
        void inline setNoUse(){
            _useMask = false;
            _mask.reset();
        };

        std::string debug_info(uint32_t qvcnt){
            std::vector<int> m;
            for(int i=0; i<qvcnt; i++) m.push_back(_mask[i]);
            if(_useMask)
                return fmt::format("use_mask \x1b[1;32m{}\x1b[0m, mask {}", _useMask, fmt::join(m,""));
            else 
                return fmt::format("use_mask \x1b[1;31m{}\x1b[0m, mask {}", _useMask, fmt::join(m,""));
        };
    };

    // ! use depth as index
    Mask* _bounding; 
    Mask** _bounding_bak; 
    NoGood_V **_nogood_v{nullptr}; // nogood_v[u_depth][v_idx] -> &nogood
    uint32_t *_candidates_v_tot{nullptr};
    uint32_t * _anc{nullptr};  
    uint32_t *_qv_depth{nullptr};
    bool _is_conflict{false};
    MaskState* _deadend_mask{nullptr};  // [u_depth]
    MaskState _inter_result;
    std::vector<uint32_t> *_fnebs_dep{nullptr};

    uint32_t qvcnt{0};

    GUPFailingPrune() = default;
    ~GUPFailingPrune(){
        for(int i=0; i<qvcnt; i++){
            delete[] _nogood_v[i];
            delete[] _bounding_bak[i];
        }
        delete[] _nogood_v;
        delete[] _deadend_mask;
        delete[] _candidates_v_tot;
        delete[] _bounding;
        delete[] _bounding_bak;
        
        delete[] _anc;
        delete[] _qv_depth;
        delete[] _fnebs_dep;
    }

    
    void Init(const Query &query, const Order& order, const Config& cfg) {

        qvcnt = query.getVertexCnt();

        _qv_depth = new uint32_t[qvcnt];
        for(int i=0; i<order.size(); i++)
            _qv_depth[order[i]] = i;
        
        _candidates_v_tot = new uint32_t[qvcnt];
        for(int i=0; i<qvcnt; i++)
            _candidates_v_tot[_qv_depth[i]] = cfg.can->candidates_count[i];

        _nogood_v = new NoGood_V*[qvcnt];
        _deadend_mask = new MaskState[qvcnt];
        for(int i=0; i<qvcnt; i++){
            _nogood_v[i] = new NoGood_V[_candidates_v_tot[i]];
            _deadend_mask[i].setUse();
            for(int j=0; j<_candidates_v_tot[i]; j++){
                _nogood_v[i][j]._fail_u_mask.reset();
            }
        }
        _inter_result.setNoUse();
        
        _anc = new uint32_t[qvcnt];
        for(int i=0; i<qvcnt; i++){
            _anc[i] = 1;
        }

        _bounding = new Mask[qvcnt];
        _bounding_bak = new Mask* [qvcnt];
        for(int i=0; i<qvcnt; i++){
            _bounding_bak[_qv_depth[i]] = new Mask[qvcnt];
            _bounding[_qv_depth[i]].reset();
        }

        for(int i=0; i<qvcnt; i++){
            for(int j=0; j<qvcnt; j++)
                _bounding_bak[i][j] = _bounding[j];
        }
        

        FollowingNeb fneb[qvcnt];
        GetFollowingNeb(query, order, fneb);

        _fnebs_dep = new std::vector<uint32_t>[qvcnt];
        for(int i=0; i<qvcnt; i++){
            for(auto uf: fneb[order[i]]){
                _fnebs_dep[i].push_back(_qv_depth[uf]);
            }
            std::sort(_fnebs_dep[i].begin(), _fnebs_dep[i].end());
        }
#ifdef LOG_OUTPUT
        for(int i=0; i<qvcnt; i++){
            spdlog::trace("Init: depth {}, boudings {}", 
                i, 
                [i, this](){
                    std::vector<int> v;
                    for(int idx=0; idx<this->qvcnt; idx++) v.push_back(this->_bounding[i][idx]);
                    return fmt::format("{}", fmt::join(v, ""));
                }());
        }
#endif
        if constexpr (Profiler::useProfiling){
            size_t mem = 0;
            // _rev_guards
            mem += (sizeof(Mask) * qvcnt); // _bounding
            mem += (sizeof(Mask) * qvcnt * qvcnt); // _bounding_bak
            // _nogood_v
            for(int i=0; i<qvcnt; i++)
                mem += (sizeof(NoGood_V) * _candidates_v_tot[i]);
            mem += (sizeof(uint32_t) * qvcnt); // _candidates_v_tot
            mem += (sizeof(uint32_t) * qvcnt); // _anc
            mem += (sizeof(uint32_t) * qvcnt); // _qv_depth
            mem += (sizeof(MaskState) * qvcnt); // _deadend_mask
            mem += sizeof(MaskState); //_inter_result;
            // fnebs_dep
            for(int i=0; i<qvcnt; i++)
                mem += (sizeof(uint32_t) * _fnebs_dep[i].size());

            Profiler::getInst().gup_fp.mem_overhead = mem;
            Profiler::getInst().gup_fp.mem_overhead_KB = ((double)mem) / (1024.0); 
        }
    }

    void inline UpdateBounding(uint32_t cur_depth, uint32_t bound_depth){
        _bounding[cur_depth].set(bound_depth);
    }
    
    void inline ExtendIndex(uint32_t cur_depth, VertexID cur_u, VertexID cur_v) {
        _anc[cur_depth]++;
        _inter_result.setNoUse();

        if(cur_depth < qvcnt - 1){
            _deadend_mask[cur_depth+1].setUse();
        }

    }

    constexpr void ReduceIndex(uint32_t cur_depth, VertexID cur_v) const noexcept {}

    void inline BackPruneCheck(uint32_t cur_depth, VertexID cur_u, uint32_t* pos, const uint32_t* tot) {}
    void inline ForwardPruneCheck(uint32_t cur_depth, VertexID cur_u, uint32_t* pos, const uint32_t* tot) {} // reservation and nogood prune

    bool inline Nogood_V_Check(uint32_t cur_depth, uint32_t v_idx, uint32_t v){
        if(_anc[_nogood_v[cur_depth][v_idx].minimal_sup_len] <= _nogood_v[cur_depth][v_idx].search_id){
            _inter_result._useMask = true;
            _inter_result._mask = _nogood_v[cur_depth][v_idx]._fail_u_mask;
            _inter_result._mask.set(cur_depth);
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;31mPruned (NoGood_V)\x1b[0m: depth {}, v {}, NoGood {{{}}}", cur_depth, v, _nogood_v[cur_depth][v_idx].debug_info(qvcnt));
#endif
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().gup_fp.cnt++;
            }
            return true;
        }
        return false;
    }

    void inline SuccessMatch(uint32_t cur_depth, VertexID cur_u, VertexID cur_v) {
        _inter_result.setNoUse();
    }
    
    bool inline NoCandidatesConflictCheck(uint32_t cnt, uint32_t cur_depth, VertexID cur_u) {
        if(cnt == 0){
            _inter_result._mask = _bounding[cur_depth];
            _inter_result._useMask = true;
            return true;
        }
        return false;
    }

    void inline InjectiveConflict(uint32_t cur_depth, VertexID cur_u, VertexID conflict_u_dep, uint32_t conflict_vidx) {
        _inter_result._mask.reset();
        _inter_result._mask.set(cur_depth);
        _inter_result._mask.set(conflict_u_dep);
        _inter_result._useMask = true;
    }

    bool inline Nogood_V_Update(uint32_t cur_depth, uint32_t v_idx, uint32_t v, uint32_t kcoreValue){

        if(_inter_result._useMask){
            auto tmp = _inter_result._mask;
            tmp.reset(cur_depth);
            uint32_t dep_last = 0;
            if(GetMSB_Index(tmp, dep_last)){
                _nogood_v[cur_depth][v_idx].minimal_sup_len = dep_last;
                _nogood_v[cur_depth][v_idx].search_id = _anc[dep_last];
                _nogood_v[cur_depth][v_idx]._fail_u_mask = tmp;
            }
            else{
                _nogood_v[cur_depth][v_idx].minimal_sup_len = 0;
                _nogood_v[cur_depth][v_idx].search_id = std::numeric_limits<size_t>::max();
                // _nogood_v[cur_depth][v_idx]._fail_u_mask = 0;
                _nogood_v[cur_depth][v_idx]._fail_u_mask.reset();
            }
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;33mUpdate NoGood_V\x1b[0m : depth {}, v {}, NoGood {{{}}}", cur_depth, v, _nogood_v[cur_depth][v_idx].debug_info(qvcnt));
#endif

            MaskStateUpdate(_inter_result._mask, _deadend_mask[cur_depth]);

            if(_inter_result._mask.test(cur_depth)){

            }
            else{
                // backjump
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;31mbackjump\x1b[0m");
#endif
                // pos[cur_depth] = tot[cur_depth];
                if constexpr (Profiler::useProfiling){
                    Profiler::getInst().gup_fp.cnt++;
                }
                return true;
            }
            

        }
        else{
            _deadend_mask[cur_depth].setNoUse();
        }
        return false;
    }

    void inline MaskStateUpdate(const Mask& mask, MaskState& ms){
        if(ms._useMask){
            uint32_t _ms_idx, _mask_idx;
            if(GetMSB_Index(mask, _mask_idx)){
                assert(GetMSB_Index(ms._mask, _ms_idx) == true);
                if(_mask_idx < _ms_idx) ms._mask = mask;
                else if(_mask_idx == _ms_idx) ms._mask |= mask;
            }
            else{ // mask == 0
                ms.setNoUse();
            }
        }
    }

    void inline Update_SkipEdge(){}

    void inline FinalReturn(uint32_t depth){
        if(_deadend_mask[depth]._useMask){
            if(_deadend_mask[depth]._mask[depth]){
                _deadend_mask[depth]._mask.reset(depth);
                _deadend_mask[depth]._mask |= _bounding[depth];
            }
            _inter_result._useMask = true;
            _inter_result._mask = _deadend_mask[depth]._mask;
        }
        else{
            _inter_result.setNoUse();
        }
    }

    void inline BoundingsBackUp(uint32_t depth, VertexID cur_u){
        for(auto uf_dep: _fnebs_dep[depth]){
            _bounding_bak[depth][uf_dep] = _bounding[uf_dep];
        }
    }

    void inline BoundingsRecover(uint32_t depth, VertexID cur_u){
        for(auto uf_dep: _fnebs_dep[depth]){
            _bounding[uf_dep] = _bounding_bak[depth][uf_dep];
        }
    }
    
};

class NoConflictDetection{
public:
    constexpr void Init(const Graph &data, const Query& query, const Order& order, const Config &cfg) const noexcept {}

    constexpr bool matchCheck(uint32_t depth, uint32_t v_idx, bool *vis, uint32_t *reverse_embeddings_deps) const noexcept {return false;}
};

// ! use depth as index
class GUPConflictDetection{
public:
    typedef struct _rs{
        bool _has{false};
        VertexID* _guard{nullptr};
        uint32_t _r{0};

        _rs(uint32_t r, VertexID *rev){
            _r = r;
            _guard = new VertexID[_r];
            _has = true;
            memcpy(_guard, rev, sizeof(VertexID) * r);
            std::sort(_guard, _guard+_r);
        }

        _rs():_has(false), _guard(nullptr), _r(0){}

        ~_rs(){
            if(_guard)
                delete[] _guard;
        };

        _rs(const _rs& other) {
            _r = other._r;
            _has = other._has;
            _guard = new VertexID[_r];
            std::copy(other._guard, other._guard + _r, _guard);
            std::sort(_guard, _guard+_r);
        }

        _rs& operator=(const _rs& rhs) {
            if (this != &rhs) {
                if(_guard)
                    delete[] _guard;
                _r = rhs._r;
                _has = rhs._has;
                _guard = new VertexID[_r];
                std::copy(rhs._guard, rhs._guard + _r, _guard);
            }
            return *this;
        }

        std::string debug_info(){
            if(_has){
                std::vector<uint32_t> v(_guard, _guard + _r);
                return fmt::format("{}", fmt::format("{}", fmt::join(v, ",")));
            }
            else
                return fmt::format("None");
        }
    }ReservationGuard;

    typedef struct{
        uint32_t freq{0};
        bool is_non_trivial{false};
    }CoverInfo;

    ReservationGuard **_rev_guards{nullptr}; // [depth][v_idx]
    

    Mask *former_u_deps{nullptr}; // [depth] -> former_u_depth_set. [1] -> 1000
    Mask *matchable_data_v{nullptr}; //[data_vertex] -> the query vertex depth mask that can matched with v
    CoverInfo *coverInfo{nullptr}; // [data_vertex] -> cover info
    std::unordered_set<VertexID> chosen;
    std::vector<VertexID> rev_candidates;
    std::vector<std::pair<VertexID, VertexID>> Gr;
    Mask _inter_res;


    uint32_t *_qv_depth{nullptr};
    uint32_t qvcnt{0};
    uint32_t dvcnt{0};
    uint32_t max_candidates_cnt{0};
    uint32_t _r{3};

    std::vector<uint32_t> *fneb_dep;  // [dep] -> follow_neb_dep

    ~GUPConflictDetection(){
        delete[] fneb_dep;
        delete[] _qv_depth;
        delete[] coverInfo;
        delete[] matchable_data_v;
        delete[] former_u_deps;
        for(int i=0; i<qvcnt; i++)
            delete[] _rev_guards[i];
        delete[] _rev_guards;
    }

    inline bool is_acceptable_as_rev(VertexID vf, uint32_t u_dep){
        return (matchable_data_v[vf] & former_u_deps[u_dep]).any();
    }

    inline uint32_t domsize(uint32_t u_dep, const Mask& ms){
        return (former_u_deps[u_dep] & ms).count();
    }

    inline uint32_t domSizeWith(uint32_t u_dep, VertexID vf, const Mask& ms){
        return ((ms | matchable_data_v[vf]) & former_u_deps[u_dep]).count();
    }

    inline bool matchCheck(uint32_t depth, uint32_t v_idx, bool *vis, uint32_t *reverse_embeddings_deps){

        if(_rev_guards[depth][v_idx]._has){
            Mask k;
            k.set(depth);
            for(int i=0; i<_rev_guards[depth][v_idx]._r; i++){
                VertexID v_guard = _rev_guards[depth][v_idx]._guard[i];
                if(vis[v_guard] == false) return false;
                uint32_t u_dep_ = reverse_embeddings_deps[v_guard];
                k.set(u_dep_);
            }
            _inter_res = k;
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().gup_cd.cnt++;
            }
            return true;
        }
        else{
            return false;
        }
    }

    inline Mask getCheckResult(){
        return _inter_res;
    }

    void compute_rev(const Config &cfg, const Order &order, uint32_t u_depth, VertexID v_idx, ReservationGuard &reserv_guard){
        
        VertexID u = order[u_depth];
        VertexID v = cfg.can->candidates[u][v_idx];

        std::vector<std::pair<VertexID, ReservationGuard>> all_reserv_guards;
        
        for(auto fneb_dep_it = fneb_dep[u_depth].rbegin(); fneb_dep_it != fneb_dep[u_depth].rend(); fneb_dep_it++){
#if DEBUG
            for(int i=0; i<dvcnt; i++){
                assert(coverInfo[i].freq == 0 && coverInfo[i].is_non_trivial == false);
            }
#endif
            Gr.clear();
            chosen.clear();
            rev_candidates.clear();

            Mask dom;
            dom.reset();
            uint32_t uf_dep = *fneb_dep_it;
            VertexID uf = order[uf_dep];
            uint32_t vf_nebs_cnt = 0;
            // const VertexID* vf_nebs = cfg.can->edge_matrix[u][uf]->getNeb_V(v, vf_nebs_cnt);
            const uint32_t* vf_nebs_indices = cfg.can->edge_matrix[u][uf]->getNeb(v_idx, vf_nebs_cnt);
            for(int _vf_indices_idx=0; _vf_indices_idx<vf_nebs_cnt; _vf_indices_idx++){
                uint32_t vf_idx = vf_nebs_indices[_vf_indices_idx];
                VertexID vf = cfg.can->candidates[uf][vf_idx];
                bool is_vf_matchable = is_acceptable_as_rev(vf, u_depth);
                ReservationGuard resv_tmp(1, &vf); // trivial rev guard
                if(_rev_guards[uf_dep][vf_idx]._has)
                    resv_tmp = _rev_guards[uf_dep][vf_idx];
                
                for(int w_idx=0; w_idx<resv_tmp._r; w_idx++){
                    VertexID w = resv_tmp._guard[w_idx];
                    if(w == v) continue;
                    if(chosen.find(vf) != chosen.end()) break;

                    bool is_w_matchable = is_acceptable_as_rev(w, u_depth);
                    if (is_vf_matchable && ( vf == w || is_w_matchable == false)){
                        chosen.insert(vf);
                        dom |= matchable_data_v[vf];
                    }
                    else if(is_vf_matchable == false && is_w_matchable){
                        chosen.insert(w);
                        dom |= matchable_data_v[w];
                    }

                    if(
                        (is_vf_matchable == false && is_w_matchable == false) ||
                        chosen.size() > _r ||
                        chosen.size() > domsize(u_depth, dom)
                    ){
                        chosen.clear();
                        reserv_guard._has = false;
                        return;
                    }
                }
            }

#ifdef LOG_OUTPUT
            spdlog::trace("Update {} {} by {}, chosen {{{}}}",
                u_depth, v_idx, uf_dep, [this](){
                    std::vector<uint32_t> vec;
                    for(auto v: this->chosen) vec.push_back(v);
                    std::sort(vec.begin(), vec.end());
                    return fmt::format("{}", fmt::join(vec, ","));
                }()
            );
#endif

            for(int _vf_indices_idx=0; _vf_indices_idx<vf_nebs_cnt; _vf_indices_idx++){
                uint32_t vf_idx = vf_nebs_indices[_vf_indices_idx];
                VertexID vf = cfg.can->candidates[uf][vf_idx];
                if(_rev_guards[uf_dep][vf_idx]._has){
                    if(chosen.find(vf) == chosen.end()){
                        for(int w_idx = 0; w_idx < _rev_guards[uf_dep][vf_idx]._r; w_idx++){
                            VertexID w = _rev_guards[uf_dep][vf_idx]._guard[w_idx];
                            if(w == v) continue;
                            if(chosen.find(w) == chosen.end()){
                                Gr.push_back(std::make_pair(vf, w));
                            }
                        }
                    }
                }
            }

#ifdef LOG_OUTPUT
            spdlog::trace("Update {} {} by {}, Gr Build [ {} ]",
                u_depth, v_idx, uf_dep, [this](){
                    std::ostringstream oss;
                    std::vector<std::pair<VertexID, VertexID>> vec;
                    for(auto v: this->Gr) vec.push_back(v);
                    std::sort(vec.begin(), vec.end());
                    for(auto &p: vec) oss << fmt::format("({},{}), ", p.first, p.second);
                    return fmt::format("{}", oss.str());
                }()
            );
#endif

            for(auto _e: Gr){
                VertexID vf = _e.first;
                VertexID w = _e.second;
                if(coverInfo[vf].freq == 0)
                    rev_candidates.push_back(vf);
                if(coverInfo[w].freq == 0)
                    rev_candidates.push_back(w);
                
                coverInfo[vf].freq += 1;
                coverInfo[w].freq +=1;
                coverInfo[w].is_non_trivial = true;
            }

#ifdef LOG_OUTPUT
            spdlog::trace("Update {} {} by {}, rev_candidates [{}]",
                u_depth, v_idx, uf_dep, fmt::format("{}", fmt::join(rev_candidates, ","))
            );
            for(auto _t_id: rev_candidates)
                spdlog::trace("Update {} {} by {}, coverInfo[{}]: freq {}, {}",
                    u_depth, v_idx, uf_dep, _t_id, coverInfo[_t_id].freq, coverInfo[_t_id].is_non_trivial
                );
#endif

            while(chosen.size() < _r){

                auto it = std::partition(rev_candidates.begin(), rev_candidates.end(), 
                    [&](int _v) {
                        return coverInfo[_v].freq > 0 && chosen.size() + 1 <= domSizeWith(u_depth, _v, dom);
                });

                uint32_t valid_rev_can_num = std::distance(rev_candidates.begin(), it);
                if(valid_rev_can_num == 0) break;

                uint32_t _max_rev_idx=0;
                uint32_t _max_freq = 0;
                for(int i=0; i<valid_rev_can_num; i++){
                    if(coverInfo[rev_candidates[i]].freq >= _max_freq){
                        _max_rev_idx = i;
                        _max_freq = coverInfo[rev_candidates[i]].freq;
                    }
                }
                VertexID chosed_rev = rev_candidates[_max_rev_idx];
                chosen.insert(chosed_rev);

                for (auto it = Gr.begin(); it != Gr.end(); ) {
                    VertexID vf = it->first;
                    VertexID w = it->second;

                    if (vf == chosed_rev || w == chosed_rev) {
                        coverInfo[vf].freq -= 1;
                        coverInfo[w].freq -= 1;
                        it = Gr.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

#ifdef LOG_OUTPUT
            spdlog::trace("Update {} {} by {}, Gr [ {} ]",
                u_depth, v_idx, uf_dep, [this](){
                    std::ostringstream oss;
                    std::vector<std::pair<VertexID, VertexID>> vec;
                    for(auto v: this->Gr) vec.push_back(v);
                    std::sort(vec.begin(), vec.end());
                    for(auto &p: vec) oss << fmt::format("({},{}), ", p.first, p.second);
                    return fmt::format("{}", oss.str());
                }()
            );
#endif
            for(auto v_: rev_candidates){
                coverInfo[v_].freq = 0;
                coverInfo[v_].is_non_trivial = false;
            }


            if(Gr.empty() && chosen.size()){
                VertexID _tmp_arr[chosen.size()];
                int _tmp_cnt = 0;
                for(auto v: chosen) _tmp_arr[_tmp_cnt++] = v;

                all_reserv_guards.push_back(std::make_pair(uf_dep, ReservationGuard(_tmp_cnt, _tmp_arr)));
            }
        }

        sort(
            all_reserv_guards.begin(),
            all_reserv_guards.end(),
            [this](
                const std::pair<VertexID, ReservationGuard> &a,
                const std::pair<VertexID, ReservationGuard> &b
            ){
                if(a.second._r != b.second._r)
                    return a.second._r < b.second._r;
                else
                    return a.first > b.first;
                return true;
            }
        );

#ifdef LOG_OUTPUT
        for(int idx=0; idx<all_reserv_guards.size(); idx++){
            spdlog::trace("Guards Update {} {}, guards[{}]:{{{}:[{}]}}",
                u_depth, v_idx, idx, all_reserv_guards[idx].first, all_reserv_guards[idx].second.debug_info()
            );
        }
#endif

        if(all_reserv_guards.size()){
            const auto &final_resv = all_reserv_guards[0].second;
            reserv_guard = final_resv;
        }
    }


    void Init(const Graph &data, const Query& query, const Order& order, const Config &cfg){
        dvcnt = data.getVertexCnt();
        qvcnt = query.getVertexCnt();
        for(int i=0; i<qvcnt; i++)
            max_candidates_cnt = std::max(max_candidates_cnt, cfg.can->candidates_count[i]);

#ifdef LOG_OUTPUT
        for(int i=0; i<qvcnt; i++){
            spdlog::trace("Candidates at depth {}, cnt {}: [{}]",
                i, cfg.can->candidates_count[order[i]], [&](){
                    std::ostringstream oss;
                    std::vector<VertexID> vec;
                    for(int id=0; id < cfg.can->candidates_count[order[i]]; id++)
                        vec.push_back(cfg.can->candidates[order[i]][id]);
                    for(auto &p: vec) oss << fmt::format("{},", p);
                    return fmt::format("{}", oss.str());
                }()
            );
        }
#endif
        
        _qv_depth = new uint32_t[qvcnt];
        _r = cfg.gupConflictDetectionSize;
        for(int i=0; i<qvcnt; i++)
            _qv_depth[order[i]] = i;

        FollowingNeb fneb[qvcnt];
        for(int i=0; i<qvcnt; i++) fneb[i].clear();
        GetFollowingNeb(query, order, fneb);

        former_u_deps = new Mask[qvcnt];
        former_u_deps[0].reset();
        for(int i=1; i<qvcnt; i++){
            former_u_deps[i].reset();
            former_u_deps[i].set(i-1);
            former_u_deps[i] |= former_u_deps[i-1];
        }
        
        fneb_dep = new std::vector<uint32_t>[qvcnt];
        for(int i=0; i<qvcnt; i++){
            fneb_dep[i].clear();
            VertexID u = order[i];
            for(auto u_follow: fneb[u]){
                fneb_dep[i].push_back(_qv_depth[u_follow]);
            }
            std::sort(fneb_dep[i].begin(), fneb_dep[i].end());
        }

        coverInfo = new CoverInfo[dvcnt];
        chosen.reserve(qvcnt * 2);

        rev_candidates.clear();
        Gr.clear();

        matchable_data_v = new Mask[dvcnt];
        for(int i=0; i<dvcnt; i++)
            matchable_data_v[i].reset();

        for(int i=0; i<qvcnt; i++){
            VertexID u = order[i];
            for(int j=0; j<cfg.can->candidates_count[u]; j++){
                VertexID v = cfg.can->candidates[u][j];
                matchable_data_v[v].set(i);
            }
        }

        _rev_guards = new ReservationGuard*[qvcnt];
        for(int i=0; i<qvcnt; i++)
            _rev_guards[i] = new ReservationGuard[max_candidates_cnt];

        for(int i=qvcnt-1; i>=0; i--){
            for(int j=0; j<cfg.can->candidates_count[order[i]]; j++){
                compute_rev(cfg, order, i, j, _rev_guards[i][j]);
#ifdef LOG_OUTPUT
                spdlog::trace("Guard status {} {} {{{}}}",
                    i, j, _rev_guards[i][j].debug_info()
                );
#endif
            }
        }
        
        if constexpr (Profiler::useProfiling){
            size_t mem = 0;
            // _rev_guards
            mem += (sizeof(ReservationGuard) * qvcnt * max_candidates_cnt);
            for(int i=qvcnt-1; i>=0; i--){
                for(int j=0; j<cfg.can->candidates_count[order[i]]; j++){
                    if(_rev_guards[i][j]._has)
                        mem += (_rev_guards[i][j]._r * sizeof(VertexID));
                }
            }
            Profiler::getInst().gup_cd.mem_overhead = mem;
            Profiler::getInst().gup_cd.mem_overhead_KB = ((double)mem) / (1024.0); 
        }
    }
};

uint32_t strhash(std::vector<std::pair<VertexID, VertexID>> nebs, uint32_t base, uint32_t mod){
    uint32_t hash_value = 0;
    uint32_t power = 1;
    for(auto p: nebs){
        hash_value = (hash_value + p.first*power)% mod;
        power = (power * base) % mod;

        hash_value = (hash_value + p.second*power)% mod;
        power = (power * base) % mod;
    }
    return hash_value;
}

void hashBase(const Query &query, const Config& cfg, const Order &order, std::vector<std::vector<uint32_t>> &vec_index, std::vector<std::vector<VertexID>> &vec_set){

    uint32_t mod1 = 1e9 + 7, mod2 = 1e9+9;
    uint32_t base = 131;
    uint32_t vec_cnt = 0;

    vec_index.clear();
    vec_index.resize(query.getVertexCnt());

    uint32_t max_depth = query.getVertexCnt();
    for(uint32_t dep=0; dep<max_depth; dep++){
        VertexID u = order[dep];
        uint32_t uneb_cnt = 0;
        const VertexID* unebs = query.getNeb(u, uneb_cnt);
        std::map<uint64_t, std::vector<VertexID>> maps;
        maps.clear();
        vec_index[dep].resize(cfg.can->candidates_count[u]);
        std::fill(vec_index[dep].begin(), vec_index[dep].end(), INVALID);
        for(int vidx = 0; vidx < cfg.can->candidates_count[u]; vidx++){
            std::vector<std::pair<VertexID, VertexID>> neb_list;
            // VertexID v = cfg.can->candidates[u][vidx];
            
            neb_list.clear();
            
            for(int i=0; i<uneb_cnt; i++){
                VertexID u_neb = unebs[i];
                Edges* e = cfg.can->edge_matrix[u][u_neb];
                uint32_t vneb_cnt = 0;
                const VertexID* vnebs = e->getNeb_V(vidx, vneb_cnt);
                for(int vneb_id = 0; vneb_id < vneb_cnt; vneb_id++){
                    VertexID vneb = vnebs[vneb_id];
                    neb_list.push_back(std::make_pair(u_neb, vneb));
                }
            }

            uint32_t hash1 = strhash(neb_list, base, mod1);
            uint32_t hash2 = strhash(neb_list, base, mod2);
            uint64_t hash = ((uint64_t)hash1)<<32|hash2;

            if (maps.find(hash) == maps.end()) {
                maps[hash] = std::vector<VertexID>();
            }

            maps[hash].push_back(vidx);
        }

        for(auto p:maps){
            vector<VertexID> tmp_vec;
            tmp_vec.clear();
            for(auto vidx: p.second){
                tmp_vec.push_back(cfg.can->candidates[u][vidx]);
                vec_index[dep][vidx] = vec_cnt;
            }
            std::sort(tmp_vec.begin(), tmp_vec.end());
            vec_set.push_back(tmp_vec);
            vec_cnt++;
        }
    }

}

class BICEAutomorphismPrune{
public:

    std::vector<std::vector<uint32_t>> cell_index;
    std::vector<std::vector<VertexID>> cell; // [idx] -> cell;
    std::vector<std::vector<uint32_t>> cell_index_by_depth; // [u] -> vec{index}
    std::vector<bool> vis;
    uint32_t *matching_cell{nullptr};
    bool *tmp_vis{nullptr};

    std::vector<VertexID> debug;

    uint32_t qvcnt{0};
    uint32_t dvcnt{0};

    ~BICEAutomorphismPrune(){
        delete[] matching_cell;
        delete[] tmp_vis;
    }

    void Init(const Graph &data, const Query &query, const Config& cfg, const Order &order){

        qvcnt = query.getVertexCnt();
        dvcnt = data.getVertexCnt();

        // ComputeStaticEqulCell(query, cfg, order, cell_index, cell);
        hashBase(query, cfg, order, cell_index, cell);

        cell_index_by_depth.clear();
        for(int i=0; i<qvcnt; i++){
            std::vector<uint32_t> vec;
            vec.clear();
            for(int j=0; j<cfg.can->candidates_count[order[i]]; j++){
                vec.push_back(cell_index[i][j]);
            }
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
            cell_index_by_depth.push_back(vec);
            // std::cout<<fmt::format("index_by_u[{}]: {}\n", i, fmt::join(cell_index_by_depth[i], " "));
        }

        vis.resize(cell.size(), false);

        matching_cell = new uint32_t [qvcnt];
        memset(matching_cell, INVALID, sizeof(uint32_t) * qvcnt);

        tmp_vis = new bool [dvcnt];
        memset(tmp_vis, 0, sizeof(bool) * dvcnt);


        if constexpr (Profiler::useProfiling){
            size_t mem = 0;
            mem += (sizeof(bool) * dvcnt); // tmp_vis;
            mem += (sizeof(uint32_t) * qvcnt); // matching_cell
            mem += (sizeof(bool) * cell.size()); // matching_cell
            for(auto v: cell){
                mem += (v.size() * sizeof(uint32_t)); // cell
            }
            for(auto v: cell_index){
                mem += (v.size() * sizeof(uint32_t)); // cell_index
            }    
            for(auto v: cell_index){
                mem += (v.size() * sizeof(uint32_t)); // cell_index_by_depth
            }      

            Profiler::getInst().bice_ap.mem_overhead = mem;
            Profiler::getInst().bice_ap.mem_overhead_KB = ((double)mem) / (1024.0); 
        }
    }

    size_t calculateAns(uint32_t depth){
        if(depth == qvcnt){
            // std::cout << fmt::format("{}\n", fmt::join(debug, ","));
            return 1;
        }

        size_t ans = 0;
        for(auto v: cell[matching_cell[depth]]){
            if(tmp_vis[v]) continue;
            tmp_vis[v] = true;
            // debug.push_back(v);
            ans += calculateAns(depth+1);
            tmp_vis[v] = false;
            // debug.pop_back();
        }
        return ans;
    }

    void inline SuccessMatch(size_t &ans){
        memset(tmp_vis, 0, sizeof(bool) * dvcnt);
        debug.clear();
        ans = calculateAns(0);
    }

    bool inline PruneCheck(uint32_t depth, uint32_t v_idx){
        return vis[cell_index[depth][v_idx]];
    }

    void inline ExtendIndex(uint32_t depth, uint32_t v_idx, VertexID v){
        uint32_t c_idx = cell_index[depth][v_idx];
        // if(v == cell[c_idx][0])
        vis[c_idx] = true;
        matching_cell[depth] = c_idx;
#ifdef LOG_OUTPUT
        spdlog::trace("extend {}, setting vis[cell[{}]]:{}", v, c_idx, vis[c_idx]);
#endif
    }

    void inline ReducedIndex(uint32_t depth, uint32_t v_idx, VertexID v){
        uint32_t c_idx = cell_index[depth][v_idx];
        if(v == *cell[c_idx].rbegin())
            vis[c_idx] = false;
#ifdef LOG_OUTPUT
        spdlog::trace("Reduce {}: cell[{}].last:{}, setting vis[cell[{}]]:{}", v, c_idx, *cell[c_idx].rbegin(), c_idx, vis[c_idx]);
#endif
    }

    void inline PruneUpdate(uint32_t depth, uint32_t v_idx, VertexID v){
        uint32_t c_idx = cell_index[depth][v_idx];
        if(v == *cell[c_idx].rbegin())
            vis[c_idx] = false;
#ifdef LOG_OUTPUT
        spdlog::trace("PruneUpdate {}: cell[{}].last:{}, setting vis[cell[{}]]:{}", v, c_idx, *cell[c_idx].rbegin(), c_idx, vis[c_idx]);
#endif
    }

};


class BICEConflictDetection{
public:


    class Bipartite{
    public:
        const uint32_t NIL = 0;
        const uint32_t INF = std::numeric_limits<uint32_t>::max();

        uint32_t q_cnt{0};
        uint32_t d_cnt{0};
        uint32_t *distance{nullptr};
        VertexID *q_pair{nullptr};
        VertexID *d_pair{nullptr};

        Bipartite(uint32_t qvcnt, uint32_t dvcnt){
            q_cnt = qvcnt;
            d_cnt = dvcnt;
            q_pair = new VertexID[q_cnt+1];
            memset(q_pair, NIL, sizeof(VertexID) * (q_cnt+1));
            d_pair = new VertexID[d_cnt+1];
            memset(d_pair, NIL, sizeof(VertexID) * (d_cnt+1));

            distance = new uint32_t[q_cnt + 1];
            std::fill(distance, distance + q_cnt + 1, INF);
        };

        ~Bipartite(){
            delete[] q_pair;
            delete[] d_pair;
            delete[] distance;
        }

        uint32_t inline qv2idx(uint32_t qv){return qv+1;}
        uint32_t inline dv2idx(uint32_t dv){return dv+1;}
        uint32_t inline idx2qv(uint32_t idx){return idx-1;}
        uint32_t inline idx2dv(uint32_t idx){return idx-1;}

        void get_noninjective_connect(VertexID u, VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth, std::vector<VertexID> &conn_q){
            std::queue<uint32_t> queue;
            queue.push(qv2idx(u));
            conn_q.push_back(u);
            while (!queue.empty())
            {
                uint32_t q_idx = queue.front();
                queue.pop();

                if(distance[q_idx] < distance[NIL]){
                    if(q_idx != NIL && cur_depth != INVALID && qv_depth[idx2qv(q_idx)] <= cur_depth) continue;
                    for(uint32_t dv_idx=0; dv_idx<edges_size[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)]; dv_idx++){
                        VertexID dv = edges[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)][dv_idx];
                        if(distance[d_pair[dv2idx(dv)]] == INF){
                            distance[d_pair[dv2idx(dv)]] = distance[q_idx]+1;
                            queue.push(d_pair[dv2idx(dv)]);
                        }
                    }
                }
            }
        }

        bool bfs(VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth){
            std::queue<uint32_t> queue;
            for(int i=0; i<q_cnt; i++){
                if(q_pair[qv2idx(i)] == NIL){
                    distance[qv2idx(i)] = 0;
                    queue.push(qv2idx(i));
                }
                else
                    distance[qv2idx(i)] = INF;
            }
            distance[NIL] = INF;
            while (!queue.empty())
            {
                uint32_t q_idx = queue.front();
                queue.pop();

                if(distance[q_idx] < distance[NIL]){
                    if(q_idx != NIL && cur_depth != INVALID && qv_depth[idx2qv(q_idx)] <= cur_depth) continue;
                    for(uint32_t dv_idx=0; dv_idx<edges_size[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)]; dv_idx++){
                        VertexID dv = edges[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)][dv_idx];
                        if(distance[d_pair[dv2idx(dv)]] == INF){
                            distance[d_pair[dv2idx(dv)]] = distance[q_idx]+1;
                            queue.push(d_pair[dv2idx(dv)]);
                        }
                    }
                }
            }
            return distance[NIL] != INF;
        }

        bool dfs(uint32_t q_idx, VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth){
            if(q_idx != NIL){
                if(cur_depth != INVALID && qv_depth[idx2qv(q_idx)] <= cur_depth) return false;
                for(uint32_t dv_idx=0; dv_idx<edges_size[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)]; dv_idx++){
                    VertexID dv = edges[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)][dv_idx];
                    if(distance[d_pair[dv2idx(dv)]] == distance[q_idx] + 1){
                        if(dfs(d_pair[dv2idx(dv)], edges, edges_size, stack_size, qv_depth, cur_depth)){
                            q_pair[q_idx] = dv2idx(dv);
                            d_pair[dv2idx(dv)] = q_idx;
                            return true;
                        }
                    }
                }
                distance[q_idx] = INF;
                return false;
            }   
            return true;
        }


        uint32_t FindMaxMatch(VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth){
            uint32_t ans=0;
            for(int i=0; i<q_cnt; i++){
                if(q_pair[qv2idx(i)] != NIL) ans++;
            }

            while (bfs(edges, edges_size, stack_size, qv_depth, cur_depth))
            {
                for(int qv=0; qv<q_cnt; qv++){
                    if(q_pair[qv2idx(qv)] == NIL && dfs(qv2idx(qv), edges, edges_size, stack_size, qv_depth, cur_depth))
                        ans++;
                }
            }
            return ans;
        }

        void get_noninjective_connect(VertexID u, VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth, BICEAutomorphismPrune *ap, std::vector<VertexID> &conn_q){

            uint32_t vis[q_cnt];
            memset(vis, 0, sizeof(uint32_t)*q_cnt);
        
            std::queue<VertexID> queue;
            queue.push(qv2idx(u));

            conn_q.clear();
            vis[u] = 1;

            while (!queue.empty())
            {
                uint32_t q_idx = queue.front();
                queue.pop();

                conn_q.push_back(idx2qv(q_idx));

                if(q_idx != NIL && cur_depth != INVALID && qv_depth[idx2qv(q_idx)] <= cur_depth){
                    uint32_t cidx = ap->matching_cell[qv_depth[idx2qv(q_idx)]];
                    for(VertexID dv: ap->cell[cidx]){
                        if(d_pair[dv2idx(dv)] != NIL){
                            VertexID qnext_id = d_pair[dv2idx(dv)];
                            if(vis[idx2qv(qnext_id)] == 0){
                                queue.push(qnext_id);
                                vis[idx2qv(qnext_id)] = 1;
                            }
                        }
                    }
                }
                else{
                    for(uint32_t dv_idx=0; dv_idx<edges_size[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)]; dv_idx++){
                        VertexID dv = edges[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)][dv_idx];

                        if(d_pair[dv2idx(dv)] != NIL){
                            VertexID qnext_id = d_pair[dv2idx(dv)];
                            if(vis[idx2qv(qnext_id)] == 0){
                                queue.push(qnext_id);
                                vis[idx2qv(qnext_id)] = 1;
                            }
                        }
                    }
                }
                
            }
        }

        bool bfs(VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth, BICEAutomorphismPrune *ap){
            std::queue<uint32_t> queue;
            for(int i=0; i<q_cnt; i++){
                if(q_pair[qv2idx(i)] == NIL){
                    distance[qv2idx(i)] = 0;
                    queue.push(qv2idx(i));
                }
                else
                    distance[qv2idx(i)] = INF;
            }
            distance[NIL] = INF;
            while (!queue.empty())
            {
                uint32_t q_idx = queue.front();
                queue.pop();

                if(distance[q_idx] < distance[NIL]){
                    if(q_idx != NIL && cur_depth != INVALID && qv_depth[idx2qv(q_idx)] <= cur_depth){
                        uint32_t cidx = ap->matching_cell[qv_depth[idx2qv(q_idx)]];
                        for(VertexID dv: ap->cell[cidx]){
                            if(distance[d_pair[dv2idx(dv)]] == INF){
                                distance[d_pair[dv2idx(dv)]] = distance[q_idx]+1;
                                queue.push(d_pair[dv2idx(dv)]);
                            }
                        }
                    }
                    else{
                        for(uint32_t dv_idx=0; dv_idx<edges_size[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)]; dv_idx++){
                            VertexID dv = edges[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)][dv_idx];
                            if(distance[d_pair[dv2idx(dv)]] == INF){
                                distance[d_pair[dv2idx(dv)]] = distance[q_idx]+1;
                                queue.push(d_pair[dv2idx(dv)]);
                            }
                        }
                    }
                }
            }
            return distance[NIL] != INF;
        }

        bool dfs(uint32_t q_idx, VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth, BICEAutomorphismPrune *ap){
            if(q_idx != NIL){
                if(cur_depth != INVALID && qv_depth[idx2qv(q_idx)] <= cur_depth){
                    uint32_t cidx = ap->matching_cell[qv_depth[idx2qv(q_idx)]];
                    for(VertexID dv: ap->cell[cidx]){
                        if(distance[d_pair[dv2idx(dv)]] == distance[q_idx] + 1){
                            if(dfs(d_pair[dv2idx(dv)], edges, edges_size, stack_size, qv_depth, cur_depth, ap)){
                                q_pair[q_idx] = dv2idx(dv);
                                d_pair[dv2idx(dv)] = q_idx;
                                return true;
                            }
                        }
                    }
                }
                else{
                    for(uint32_t dv_idx=0; dv_idx<edges_size[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)]; dv_idx++){
                        VertexID dv = edges[stack_size[idx2qv(q_idx)]-1][idx2qv(q_idx)][dv_idx];
                        if(distance[d_pair[dv2idx(dv)]] == distance[q_idx] + 1){
                            if(dfs(d_pair[dv2idx(dv)], edges, edges_size, stack_size, qv_depth, cur_depth, ap)){
                                q_pair[q_idx] = dv2idx(dv);
                                d_pair[dv2idx(dv)] = q_idx;
                                return true;
                            }
                        }
                    }
                }
                distance[q_idx] = INF;
                return false;
            }   
            return true;
        }


        uint32_t FindMaxMatch(VertexID ***edges, uint32_t **edges_size, uint32_t *stack_size, const uint32_t *qv_depth, uint32_t cur_depth, BICEAutomorphismPrune *ap){
            uint32_t ans=0;
            for(int i=0; i<q_cnt; i++){
                if(q_pair[qv2idx(i)] != NIL) ans++;
            }

            while (bfs(edges, edges_size, stack_size, qv_depth, cur_depth, ap))
            {
                for(int qv=0; qv<q_cnt; qv++){
                    if(q_pair[qv2idx(qv)] == NIL && dfs(qv2idx(qv), edges, edges_size, stack_size, qv_depth, cur_depth, ap))
                        ans++;
                }
            }
            return ans;
        }
    };

    uint32_t qvcnt{0};
    uint32_t max_candidates_cnt{0};
    uint32_t max_q_dgree{0};

    Bipartite* bp{nullptr};
    FollowingNeb *fnebs{nullptr};

    uint32_t *qv_depth{nullptr};

    VertexID ***edges_stack{nullptr};   // [depth][q][idx] -> [v]
    uint32_t **edges_size_stack{nullptr};  // [depth][q] -> size
    uint32_t *stack_size{nullptr};        // [q] -> [stack_size]

    VertexID ** q_pair_bak{nullptr}; // [depth] -> q_pair

    ~BICEConflictDetection(){

        for (int dep = 0; dep < max_q_dgree + 1; dep++) {
            for (int u = 0; u < qvcnt; u++) {
                delete[] edges_stack[dep][u];
            }
            delete[] edges_stack[dep];       
            delete[] edges_size_stack[dep];  
        }

        delete[] qv_depth;

        for(int dep=0; dep<qvcnt; dep++)
            delete[] q_pair_bak[dep];

        delete[] q_pair_bak;
        delete[] edges_stack;                
        delete[] edges_size_stack;           

        delete[] stack_size;                 
        delete bp;
        delete[] fnebs;
    }

    void Init(const Graph& data, const Query& query, const Order& order, const Config& cfg){
        qvcnt = query.getVertexCnt();
        max_q_dgree = query.getMaxDgree();

        fnebs = new FollowingNeb[qvcnt];
        GetFollowingNeb(query, order, fnebs);

        qv_depth = new uint32_t[qvcnt];
        for(int i=0; i<qvcnt; i++)
            qv_depth[order[i]] = i;

        bp = new Bipartite(qvcnt, data.getVertexCnt());

        for(int i=0; i<qvcnt; i++)
            max_candidates_cnt = std::max(max_candidates_cnt, cfg.can->candidates_count[i]);
        
        edges_stack = new VertexID **[max_q_dgree + 1];
        edges_size_stack = new uint32_t *[max_q_dgree + 1];
        stack_size = new uint32_t[qvcnt];
        memset(stack_size, 0, sizeof(uint32_t)*qvcnt);
        for(int dep=0; dep<max_q_dgree + 1; dep++){
            edges_size_stack[dep] = new uint32_t[qvcnt];
            memset(edges_size_stack[dep], 0, sizeof(uint32_t)*qvcnt);
            edges_stack[dep] = new VertexID *[qvcnt];
            for(int u=0; u<qvcnt; u++)
                edges_stack[dep][u] = new VertexID[max_candidates_cnt];
        }

        for(int u=0; u<qvcnt; u++){
            memcpy(edges_stack[0][u], cfg.can->candidates[u], sizeof(VertexID) * cfg.can->candidates_count[u]);
            edges_size_stack[0][u] = cfg.can->candidates_count[u];
            stack_size[u]++;
        }
        
        bp->FindMaxMatch(edges_stack, edges_size_stack, stack_size, qv_depth, INVALID);

        q_pair_bak = new VertexID*[qvcnt];
        for(int i=0; i<qvcnt; i++)
            q_pair_bak[i] = new VertexID[qvcnt+1];
            
        if constexpr (Profiler::useProfiling){
            size_t mem = 0;
            // bp
            mem += (sizeof(uint32_t) * (bp->q_cnt+1)); 
            mem += (sizeof(VertexID) * (bp->q_cnt+1)); 
            mem += (sizeof(VertexID) * (bp->d_cnt+1));
            // fnebs
            for(int i=0; i<qvcnt; i++)
                mem += (sizeof(VertexID) * fnebs[i].size());
            mem += (sizeof(VertexID) * (max_q_dgree + 1) * qvcnt * max_candidates_cnt); //edges_stack
            mem += (sizeof(uint32_t) * (max_q_dgree + 1) * qvcnt); // edges_size_stack
            mem += (sizeof(uint32_t) * qvcnt); // stack_size;
            mem += (sizeof(VertexID) * qvcnt * (qvcnt+1)); //q_pair_bak
            mem += (sizeof(uint32_t) * qvcnt); // qv_depth
            Profiler::getInst().bice_cd.mem_overhead = mem;
            Profiler::getInst().bice_cd.mem_overhead_KB = ((double)mem) / (1024.0); 
        }

#ifdef LOG_OUTPUT
        spdlog::trace("p_pair: {}, d_pair: {}", 
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->q_cnt+1; i++){
                    // vec.push_back(bp->q_pair[i]);
                    if(bp->q_pair[i] != bp->NIL)
                    vec.push_back(fmt::format("q[{}]={}", i-1, bp->q_pair[i]));
                } 
                return fmt::format("{}", fmt::join(vec, ","));
            }(),
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->d_cnt+1; i++){
                    // vec.push_back(bp->d_pair[i]);
                    if(bp->d_pair[i] != bp->NIL)
                        vec.push_back(fmt::format("d[{}]={}", i-1, bp->d_pair[i]));
                }
                return fmt::format("{}", fmt::join(vec, ","));
            }()
        );
#endif 
    }

    bool inline IsConflict(VertexID u, VertexID v, uint32_t v_idx, Edges *** edge_matrix){

        for(auto uf: fnebs[u]){
            uint32_t ufp_id = uf+1;
            if(bp->q_pair[ufp_id] != bp->NIL){
                bp->d_pair[bp->q_pair[ufp_id]] = bp->NIL;
                bp->q_pair[ufp_id] = bp->NIL;
            }

            uint32_t neb_cnt;
            const VertexID *nebs = edge_matrix[u][uf]->getNeb_V(v_idx, neb_cnt);

            uint32_t stack_top = stack_size[uf];

            Intersection(edges_stack[stack_top-1][uf], edges_size_stack[stack_top-1][uf], nebs, neb_cnt, edges_stack[stack_top][uf], edges_size_stack[stack_top][uf]);
            stack_size[uf]++;
        }

        uint32_t up_id = u+1;
        uint32_t vp_id = v+1;

        if(bp->q_pair[up_id] != bp->NIL)
            bp->d_pair[bp->q_pair[up_id]] = bp->NIL;
        bp->q_pair[up_id] = vp_id;
        if(bp->d_pair[vp_id] != bp->NIL)
            bp->q_pair[bp->d_pair[vp_id]] = bp->NIL;
        bp->d_pair[vp_id] = up_id;

#ifdef LOG_OUTPUT
        spdlog::trace("u {} v {} p_pair: {}, d_pair: {}", u, v,
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->q_cnt+1; i++){
                    // vec.push_back(bp->q_pair[i]);
                    if(bp->q_pair[i] != bp->NIL)
                    vec.push_back(fmt::format("q[{}]={}", i-1, bp->q_pair[i]));
                } 
                return fmt::format("{}", fmt::join(vec, ","));
            }(),
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->d_cnt+1; i++){
                    // vec.push_back(bp->d_pair[i]);
                    if(bp->d_pair[i] != bp->NIL)
                        vec.push_back(fmt::format("d[{}]={}", i-1, bp->d_pair[i]));
                }
                return fmt::format("{}", fmt::join(vec, ","));
            }()
        );
#endif 

#ifdef LOG_OUTPUT
        for(int i=0; i<qvcnt; i++){
            if(bp->q_pair[i+1] != bp->NIL){
                assert(bp->d_pair[bp->q_pair[i+1]] == i+1);
            }
            else{
                for(int j=0; j<bp->d_cnt; j++)
                    assert(bp->d_pair[j+1] != i+1);
            }
        }
#endif 

        if(bp->FindMaxMatch(edges_stack, edges_size_stack, stack_size, qv_depth, qv_depth[u]) != qvcnt){
#ifdef LOG_OUTPUT
            spdlog::trace("prune by conflict at u {}, bipartite: {}", u, bp->FindMaxMatch(edges_stack, edges_size_stack, stack_size, qv_depth, qv_depth[u]));
#endif  
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().bice_cd.cnt++;
            }
            return true;
        }

        return false;
    }

    bool inline IsConflict(VertexID u, VertexID v, uint32_t v_idx, Edges *** edge_matrix, BICEAutomorphismPrune *ap){

        for(auto uf: fnebs[u]){
            uint32_t ufp_id = uf+1;
            if(bp->q_pair[ufp_id] != bp->NIL){
                bp->d_pair[bp->q_pair[ufp_id]] = bp->NIL;
                bp->q_pair[ufp_id] = bp->NIL;
            }

            uint32_t neb_cnt;
            const VertexID *nebs = edge_matrix[u][uf]->getNeb_V(v_idx, neb_cnt);

            uint32_t stack_top = stack_size[uf];

            Intersection(edges_stack[stack_top-1][uf], edges_size_stack[stack_top-1][uf], nebs, neb_cnt, edges_stack[stack_top][uf], edges_size_stack[stack_top][uf]);
            stack_size[uf]++;
        }

        uint32_t up_id = u+1;
        uint32_t vp_id = v+1;

        if(bp->q_pair[up_id] != bp->NIL)
            bp->d_pair[bp->q_pair[up_id]] = bp->NIL;
        bp->q_pair[up_id] = vp_id;
        if(bp->d_pair[vp_id] != bp->NIL)
            bp->q_pair[bp->d_pair[vp_id]] = bp->NIL;
        bp->d_pair[vp_id] = up_id;

#ifdef LOG_OUTPUT
        spdlog::trace("u {} v {} p_pair: {}, d_pair: {}", u, v,
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->q_cnt+1; i++){
                    // vec.push_back(bp->q_pair[i]);
                    if(bp->q_pair[i] != bp->NIL)
                    vec.push_back(fmt::format("q[{}]={}", i-1, bp->q_pair[i]));
                } 
                return fmt::format("{}", fmt::join(vec, ","));
            }(),
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->d_cnt+1; i++){
                    // vec.push_back(bp->d_pair[i]);
                    if(bp->d_pair[i] != bp->NIL)
                        vec.push_back(fmt::format("d[{}]={}", i-1, bp->d_pair[i]));
                }
                return fmt::format("{}", fmt::join(vec, ","));
            }()
        );
#endif 

#ifdef LOG_OUTPUT
        for(int i=0; i<qvcnt; i++){
            if(bp->q_pair[i+1] != bp->NIL){
                assert(bp->d_pair[bp->q_pair[i+1]] == i+1);
            }
            else{
                for(int j=0; j<bp->d_cnt; j++)
                    assert(bp->d_pair[j+1] != i+1);
            }
        }
#endif 

        if(bp->FindMaxMatch(edges_stack, edges_size_stack, stack_size, qv_depth, qv_depth[u], ap) != qvcnt){
#ifdef LOG_OUTPUT
            spdlog::trace("prune by conflict at u {}, bipartite: {}", u, bp->FindMaxMatch(edges_stack, edges_size_stack, stack_size, qv_depth, qv_depth[u], ap));
#endif  
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().bice_cd.cnt++;
            }
            return true;
        }

        return false;
    }

    void inline Q_PairBakeup(uint32_t depth){
        memcpy(q_pair_bak[depth], bp->q_pair, sizeof(VertexID) * (qvcnt+1));
    }

    void inline PruneUpdate(VertexID u, uint32_t depth){
        if(bp->q_pair[u+1] != bp->NIL)
            bp->d_pair[bp->q_pair[u+1]] = bp->NIL;
        

        for(auto uf: fnebs[u]){
            stack_size[uf]--;
            if(bp->q_pair[uf+1] != bp->NIL)
                bp->d_pair[bp->q_pair[uf+1]] = bp->NIL;
        }

        // memset(bp->d_pair, bp->NIL, sizeof(VertexID) * (bp->d_cnt + 1));

        // memcpy(bp->q_pair, q_pair_bak[depth], sizeof(VertexID) * (qvcnt+1));
        for(int i=0; i<qvcnt; i++){
            if(q_pair_bak[depth][i+1] != bp->q_pair[i+1]){
                bp->d_pair[bp->q_pair[i+1]] = bp->NIL;
                bp->q_pair[i+1] = q_pair_bak[depth][i+1];
            }
        }

        for(int i=0; i<qvcnt; i++){
            if(bp->q_pair[i+1] != bp->NIL)
                bp->d_pair[bp->q_pair[i+1]] = i+1;
        }

#ifdef LOG_OUTPUT
        spdlog::trace("u {} p_pair: {}, d_pair: {}", u,
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->q_cnt+1; i++){
                    // vec.push_back(bp->q_pair[i]);
                    if(bp->q_pair[i] != bp->NIL)
                    vec.push_back(fmt::format("q[{}]={}", i-1, bp->q_pair[i]));
                } 
                return fmt::format("{}", fmt::join(vec, ","));
            }(),
            [&](){
                std::vector<string> vec;
                for(int i=0; i<bp->d_cnt+1; i++){
                    // vec.push_back(bp->d_pair[i]);
                    if(bp->d_pair[i] != bp->NIL)
                        vec.push_back(fmt::format("d[{}]={}", i-1, bp->d_pair[i]));
                }
                return fmt::format("{}", fmt::join(vec, ","));
            }()
        );
#endif 

#ifdef LOG_OUTPUT
        for(int i=0; i<qvcnt; i++){
            if(bp->q_pair[i+1] != bp->NIL){
                assert(bp->d_pair[bp->q_pair[i+1]] == i+1);
            }
            else{
                for(int j=0; j<bp->d_cnt; j++)
                    assert(bp->d_pair[j+1] != i+1);
            }
        }
#endif 

    }
};


class VEQAutomorphismPrune{
public:

    typedef struct _cl{
        VertexID *_data{nullptr};
        uint32_t _size{0};

        void inline Reset(){
            _size = 0;
        }

        void inline Init(uint32_t cap){
            _data = new VertexID[cap];   // storage in ascending order
            memset(_data, 0, sizeof(uint32_t) * cap);
            _size = cap;
        }

        _cl(){
            _data = nullptr;
            _size = 0;
        }

        ~_cl(){
            delete[] _data;
            _size = 0;
        }

        _cl(const _cl &other) = delete;

        _cl& operator=(const _cl &other) {
            if (this != &other) {
                _size = other._size;
                std::copy(other._data, other._data + _size, _data);
            }
            return *this;
        }

        std::string debug_info() const{
            std::vector<uint32_t> vec;
            vec.clear();
            for(int i=0; i<_size; i++) vec.push_back(_data[i]);
            return fmt::format("parr: {:#x}, size: {}, arr: {{{}}}", reinterpret_cast<uintptr_t>(_data), _size, fmt::join(vec, ","));
        }

        void inline UnionWith(const _cl& rhs, uint32_t *&buffer){
            uint32_t tmp;
            Union(_data, _size, rhs._data, rhs._size, buffer, tmp);
            _size = tmp;
            std::swap(_data, buffer);
        }

        void inline IntersectWith(const _cl& rhs, uint32_t *&buffer){
            uint32_t tmp;
            Intersection(_data, _size, rhs._data, rhs._size, buffer, tmp);
            _size = tmp;
            std::swap(_data, buffer);
        }

        void inline SubtractWith(const _cl& rhs, uint32_t *&buffer){
            uint32_t tmp;
            Subtraction(this->_data, this->_size, rhs._data, rhs._size, buffer, tmp);
            _size = tmp;
            std::swap(_data, buffer);
        }
        
    }Cell;

    uint32_t **Tm{nullptr}; // [u_dep][v_idx] -> embeddings cnt rooted at [u_dep][v_idx]
    uint32_t *Tm_tot{nullptr}; //[u_dep] -> total cnt at dep;
    Cell *delta_m{nullptr};  // [u_dep] -> Cell of delta at matching m
    // Cell *pi_negative_m{nullptr}; // [depth] -> negative cell

    Cell *pi_m{nullptr};        // [idx] -> cell
    uint32_t **pi_m_index{nullptr};  // [depth][v_idx] -> pi_m_idx
    uint32_t *pi_m_cnt{nullptr};
    VertexID *matched_vidx{nullptr};

    Cell *pi{nullptr};            // [idx] -> cell
    uint32_t **pi_index{nullptr};      // [depth][v_idx]
    VertexID ** eq_m{nullptr};

    VertexID *_buffer{nullptr};

    uint32_t qvcnt{0};
    uint32_t max_euqal_neb{0};
    uint32_t max_dgree{0};
    uint32_t max_candidates{0};

    std::vector<std::vector<uint32_t>> ancestor_dep;

    ~VEQAutomorphismPrune(){
        
        delete[] Tm_tot; //[u_dep] -> total cnt at dep;
        delete[] delta_m;  // [u_dep] -> Cell of delta at matching m
        // delete[] pi_negative_m; // [depth] -> negative cell
        delete[] pi_m;        // [idx] -> cell
        delete[] pi;            // [idx] -> cell
        delete[] _buffer;
        delete[] pi_m_cnt;
        delete[] matched_vidx;

        for(int i=0; i<qvcnt; i++){
            delete[] pi_m_index[i];
            delete[] pi_index[i];
            delete[] Tm[i];
            delete[] eq_m[i];
        }

        delete[] pi_m_index;  // [depth][v_idx] -> pi_m_idx
        delete[] pi_index;      // [depth][v_idx]
        delete[] eq_m;
        delete[] Tm; // [u_dep][v_idx] -> embeddings cnt rooted at [u_dep][v_idx]
    }

    void Init(const Graph &data, const Query &query, const Config& cfg, const Order &order){
        qvcnt = query.getVertexCnt();
        max_dgree = data.getMaxDgree();

        max_candidates = 0;
        for(int i=0; i<qvcnt; i++)
            max_candidates = std::max(cfg.can->candidates_count[i], max_candidates);

        std::vector<std::vector<uint32_t>> vec_index;
        std::vector<std::vector<uint32_t>> vec_set;  // []-> the v_idx in euqal set
        // ComputeStaticEqulCell(query, cfg, order, vec_index, vec_set);
        hashBase(query, cfg, order, vec_index, vec_set);
        for(auto v: vec_set) 
            max_euqal_neb = std::max((uint32_t)v.size(), max_euqal_neb);

        // std::cout << "max_euqal_neb: " << max_euqal_neb << "\n";
        
        // pi_negative_m = new Cell[qvcnt];
        delta_m = new Cell[qvcnt];
        Tm = new uint32_t*[qvcnt];
        pi_m_index = new uint32_t*[qvcnt];
        eq_m = new VertexID* [qvcnt];
        pi_index = new uint32_t*[qvcnt];

        for(int i=0; i<qvcnt; i++){
            // pi_negative_m[i].Init(max_euqal_neb * max_dgree);
            delta_m[i].Init(max_euqal_neb * max_dgree);
            Tm[i] = new uint32_t [max_candidates];
            pi_m_index[i] = new uint32_t[max_candidates];
            eq_m[i] = new uint32_t[max_candidates];
            pi_index[i] = new uint32_t[max_candidates];

            memset(Tm[i], 0, sizeof(uint32_t)*max_candidates);
            memset(pi_m_index[i], 0, sizeof(uint32_t)*max_candidates);
            memset(eq_m[i], 0, sizeof(uint32_t)*max_candidates);
        }

        pi_m_cnt = new uint32_t[qvcnt];
        memset(pi_m_cnt, 0, sizeof(uint32_t) * qvcnt);

        Tm_tot = new uint32_t[qvcnt];
        memset(Tm_tot, 0, sizeof(uint32_t)*qvcnt);



        uint32_t pi_m_tot = max_dgree * qvcnt + max_candidates;
        pi_m = new Cell[pi_m_tot];
        for(int i=0; i<pi_m_tot; i++){
            pi_m[i].Init(max_euqal_neb * max_dgree);
        }

        matched_vidx = new VertexID[pi_m_tot];
        memset(matched_vidx, INVALID, sizeof(VertexID)*pi_m_tot);

        pi = new Cell[vec_set.size()];
        for(int i=0; i<vec_set.size(); i++){
            pi[i].Init(vec_set[i].size());
            std::sort(vec_set[i].begin(), vec_set[i].end());
            memcpy(pi[i]._data, vec_set[i].data(), vec_set[i].size() * sizeof(uint32_t));
        }

        for(int i=0; i<qvcnt; i++)
            for(int j=0; j<cfg.can->candidates_count[order[i]]; j++)
                pi_index[i][j] = vec_index[i][j];
        
        uint32_t qv_deps[qvcnt];
        for(int i=0; i<qvcnt; i++)
            qv_deps[order[i]] = i;

        PreviousNeb prevNeb[qvcnt];
        GetPreviousNeb(query, order, prevNeb);

        std::vector<Mask> _anscestor;
        _anscestor.resize(qvcnt);

        ancestor_dep.clear();
        ancestor_dep.resize(qvcnt);
        for(int i=0; i<qvcnt; i++){
            _anscestor[i].set(i);
            for(VertexID upre: prevNeb[order[i]])
                _anscestor[i] |= _anscestor[qv_deps[upre]];
        }

        for(int i=0; i<qvcnt; i++){
            for(int j=0; j<qvcnt; j++){
                if(_anscestor[i].test(j)) ancestor_dep[i].push_back(j);
            }
            std::sort(ancestor_dep[i].begin(), ancestor_dep[i].end());
        }

        _buffer = new VertexID[max_euqal_neb * max_dgree];
        memset(_buffer, 0, sizeof(VertexID)*max_euqal_neb * max_dgree);

        if constexpr (Profiler::useProfiling){
            size_t mem = 0;
            mem += (sizeof(uint32_t) * qvcnt * max_candidates); //Tm
            mem += (sizeof(uint32_t) * qvcnt); //Tm_tot
            mem += CellMemCalculate(delta_m, qvcnt); // delta_m
            mem += CellMemCalculate(pi_m, pi_m_tot); // pi_m
            mem += (sizeof(uint32_t) * qvcnt * max_candidates); // pi_m_index
            mem += (sizeof(uint32_t) * qvcnt); //pi_m_cnt
            mem += (sizeof(VertexID) * pi_m_tot); // matched_vidx;
            mem += CellMemCalculate(pi, vec_set.size()); // pi
            mem += (sizeof(uint32_t) * qvcnt * max_candidates);// pi_index
            mem += (sizeof(VertexID) * qvcnt * max_candidates);// eq_m
            mem += (sizeof(VertexID) * max_euqal_neb * max_dgree); //_buffer
            Profiler::getInst().veq_ap.mem_overhead = mem;
            Profiler::getInst().veq_ap.mem_overhead_KB = ((double)mem) / (1024.0); 
        }
    }

    size_t inline CellMemCalculate(Cell *cell, uint32_t len){
        size_t mem=0;
        for(int i=0; i<len; i++){
            mem += (cell[i]._size * sizeof(VertexID));
        }
        return mem;
    }

    void inline ExtendIndex(uint32_t cur_depth, VertexID cur_u, VertexID cur_v, VertexID cur_v_idx, VertexID* embedding, uint32_t* idx_embedding, const Order& order){
        
        pi_m_index[cur_depth][cur_v_idx] = pi_m_cnt[cur_depth]++;
        pi_m[pi_m_index[cur_depth][cur_v_idx]] = pi[pi_index[cur_depth][cur_v_idx]];
        matched_vidx[pi_m_index[cur_depth][cur_v_idx]] = cur_v_idx;
    
        delta_m[cur_depth]._size = 0;
        
        auto* pi_uv = &pi[pi_index[cur_depth][cur_v_idx]];

        // for(auto anc_dep: ancestor_dep[cur_depth]){
        for(int anc_dep = 0; anc_dep < cur_depth; anc_dep++){
            VertexID ua = order[anc_dep];
            VertexID va = embedding[ua];
            uint32_t va_idx = idx_embedding[ua];
            bool is_va_in_topi = false;
            bool is_intersec_null = true;
            auto *pi_uava = &pi[pi_index[anc_dep][va_idx]];
#ifdef LOG_OUTPUT
            spdlog::trace("anc: ({},{}), pi_uv: {}, pi_anc: {}", ua, va, pi_uv->debug_info(), pi_uava->debug_info());
#endif
            for(int i=0; i<pi_uv->_size; i++){
                
                // * we only test the top i cuz the rest haven't been explored
                if(pi_uv->_data[i] == va){
                    is_va_in_topi = true;
                    break;
                }
                
                is_intersec_null = !std::binary_search(pi_uava->_data, pi_uava->_data+pi_uava->_size, pi_uv->_data[i]);

                if(is_intersec_null == false) break;
            }
#ifdef LOG_OUTPUT
                spdlog::trace("is_va_in_topi {}, is_intersec_null {}", is_va_in_topi, is_intersec_null);
#endif    
            if(is_va_in_topi == false && is_intersec_null == false){
                delta_m[anc_dep].UnionWith(*pi_uv, _buffer);
#ifdef LOG_OUTPUT
                spdlog::trace("at u {}, updated delta_m[{}]: {}", order[cur_depth], order[anc_dep], delta_m[anc_dep].debug_info());
#endif            
            }
        }


        if(cur_depth < qvcnt-1){
            memset(Tm[cur_depth+1], 0, max_candidates * sizeof(uint32_t));
            memset(pi_m_index[cur_depth+1], INVALID, max_candidates * sizeof(uint32_t));
            pi_m_cnt[cur_depth + 1] = pi_m_cnt[cur_depth];
            Tm_tot[cur_depth+1]=0;
            // matched_vidx[cur_depth+1]=INVALID;
        }

#ifdef LOG_OUTPUT
            spdlog::trace("Extend ({},{}) pi_m[{}][{}]: {}, pi_cnt: {}, delta_m[{}]: {}", cur_u, cur_v, cur_u, cur_v, pi_m[pi_m_index[cur_depth][cur_v_idx]].debug_info(), pi_m_cnt[cur_depth], cur_u, delta_m[cur_depth].debug_info());
#endif
        
    }

    void inline ReduceIndex(uint32_t depth, VertexID u, VertexID v_idx, VertexID **candidates, uint32_t *candidates_cnt){

        auto uv_idx = pi_m_index[depth][v_idx];
        if(Tm[depth][v_idx]){
            pi_m[uv_idx].SubtractWith(delta_m[depth], _buffer);
            matched_vidx[uv_idx] = v_idx;
        }

#ifdef LOG_OUTPUT
        spdlog::trace("Reduced to ({},{}) delta_m[{}]: {}", u, candidates[u][v_idx], u, delta_m[depth].debug_info());
        spdlog::trace("pi_m[{}][{}]: {}, pi_m_index[{}][{}]: {} ", u, candidates[u][v_idx], pi_m[uv_idx].debug_info(), u, v_idx, uv_idx);
#endif

        for(int i=1; i<pi_m[uv_idx]._size; i++){
            auto it = std::lower_bound(candidates[u], candidates[u]+candidates_cnt[u], pi_m[uv_idx]._data[i]);
            if(*it == pi_m[uv_idx]._data[i]){
                uint32_t eqv_idx = (it - candidates[u]);
                pi_m_index[depth][eqv_idx] = uv_idx;
#ifdef LOG_OUTPUT
                spdlog::trace("update pi_m_index[{}][{}] to {}", u, eqv_idx, uv_idx);
#endif
            }
        }
    }

    bool inline EquivalencePruneCheck(uint32_t cur_dep, VertexID u, VertexID v, uint32_t v_idx, uint32_t **candidates, size_t &ans){
        auto uv_idx = pi_m_index[cur_dep][v_idx];
        if(uv_idx != INVALID){
            // VertexID eqv = pi_m[uv_idx]._data[0];
            // uint32_t eqv_idx = matched_vidx[cur_dep];
            uint32_t eqv_idx = matched_vidx[uv_idx];
            // VertexID eqv = candidates[u][eqv_idx];
            // auto it = std::lower_bound(candidates[u], candidates[u] + candidates_cnt[u], eqv);
            // uint32_t eqv_idx = it - candidates[u];
            ans += Tm[cur_dep][eqv_idx];
            Tm_tot[cur_dep] += Tm[cur_dep][eqv_idx];
            if constexpr (Profiler::useProfiling)
                Profiler::getInst().veq_ap.cnt++;
// #ifdef LOG_OUTPUT
//             spdlog::trace("[{}][{}] Prune By eqv: {}, Tm[{}][{}]: {}, ", u, v, eqv, u, eqv, Tm[cur_dep][eqv_idx]);
//             spdlog::trace("pi_m[{}][{}]: {}, pi_m_index[{}][{}]: {}", u, v, pi_m[uv_idx].debug_info(), u, v_idx, uv_idx);
//             spdlog::trace("ans: {}", ans);
// #endif
            return true;
        }
        return false;
    }

    void inline SuccessMatch(uint32_t depth, uint32_t u, uint32_t v_idx){
        Tm[depth][v_idx] = 1;
        Tm_tot[depth] += 1;
    }

    void inline Backtrack(uint32_t depth, uint32_t u, uint32_t v_idx){
        Tm[depth][v_idx] = Tm_tot[depth+1];
        Tm_tot[depth] += Tm[depth][v_idx];
    }

    void inline InjectiveConflict(uint32_t depth, uint32_t v_idx, uint32_t conflict_u_dep, uint32_t conflict_v_idx){
        Tm[depth][v_idx] = 0;
        auto uv_idx = pi_index[depth][v_idx];
        auto con_uv_idx = pi_m_index[conflict_u_dep][conflict_v_idx];
        pi_m[con_uv_idx].IntersectWith(pi[uv_idx], _buffer);
        matched_vidx[con_uv_idx] = conflict_v_idx;
    }
}; 

template<
    bool Enable_DAFFailingPrune,
    bool Enable_GuPFailingPrune,
    bool Enable_GUPConflictDetection,
    bool Enable_BICEConflictDetection,
    bool Enable_VEQAutomorphismPrune
>
class PruningOpt{
public:

    std::conditional_t<Enable_DAFFailingPrune, DAFFailingPrune, std::monostate> daf_fp;
    std::conditional_t<Enable_GuPFailingPrune, GUPFailingPrune, std::monostate> gup_fp;
    std::conditional_t<Enable_GUPConflictDetection, GUPConflictDetection, std::monostate> gup_cd;
    std::conditional_t<Enable_BICEConflictDetection, BICEConflictDetection, std::monostate> bice_cd;
    std::conditional_t<Enable_VEQAutomorphismPrune, VEQAutomorphismPrune, std::monostate> veq_ap;

    void Init(const Graph& data, const Query& query, const Order &order, const Config &cfg){

        if constexpr (Enable_GUPConflictDetection){
            gup_cd.Init(data, query, order, cfg);
        }
        
        if constexpr (Enable_GuPFailingPrune){
            gup_fp.Init(query, order, cfg);
        }

        if constexpr (Enable_VEQAutomorphismPrune){
            veq_ap.Init(data, query, cfg, order);
        }

        if constexpr (Enable_BICEConflictDetection){
            bice_cd.Init(data, query, order, cfg);
        }

        if constexpr (Enable_DAFFailingPrune){
            daf_fp.Init(query, order);
        }

    }

    void inline PruneStatusBackUp(uint32_t depth, VertexID u){

        if constexpr (Enable_GuPFailingPrune){
            gup_fp.BoundingsBackUp(depth, u);
        }

        if constexpr (Enable_BICEConflictDetection){
            bice_cd.Q_PairBakeup(depth);
        }

    }

    void inline ExtendIndex(uint32_t depth, VertexID u, uint32_t v_idx, VertexID v, VertexID* embedding, uint32_t* idx_embedding, const Order& order){

        if constexpr (Enable_GuPFailingPrune){
            gup_fp.ExtendIndex(depth, u, v);
        }

        if constexpr (Enable_VEQAutomorphismPrune){
            veq_ap.ExtendIndex(depth, u, v, v_idx, embedding, idx_embedding, order);
        }

    }

    void inline ReduceIndex(uint32_t depth, VertexID u, uint32_t v_idx, VertexID v, VertexID **candidates, uint32_t *candidates_cnt){

        if constexpr (Enable_GuPFailingPrune){
            gup_fp.ReduceIndex(depth, v);
        }
        if constexpr (Enable_VEQAutomorphismPrune){
            veq_ap.ReduceIndex(depth, u, v_idx, candidates, candidates_cnt);
        }

    }

    void inline BacktrackCleanUp(uint32_t depth, VertexID u, uint32_t v_idx, VertexID v){

        if constexpr (Enable_GuPFailingPrune){
            gup_fp.BoundingsRecover(depth, u);
        }
        if constexpr (Enable_VEQAutomorphismPrune){
            veq_ap.Backtrack(depth, u, v_idx);
        }

    }

    bool inline BacktrackingPrune(uint32_t depth, VertexID u, uint32_t v_idx, VertexID v, uint32_t kcoreValue){
        bool gup_fp_flag = false;
        bool daf_fp_flag = false;

        if constexpr (Enable_GuPFailingPrune){
            gup_fp_flag = gup_fp.Nogood_V_Update(depth, v_idx, v, kcoreValue);
        }
        
        if constexpr (Enable_BICEConflictDetection){
            bice_cd.PruneUpdate(u, depth);
        }

        if constexpr (Enable_DAFFailingPrune){
            daf_fp_flag = daf_fp.PruneCheck(depth);
        }

        return gup_fp_flag | daf_fp_flag;
    }

    void inline SuccessMatch(uint32_t depth, VertexID u, uint32_t v_idx, VertexID v){

        if constexpr (Enable_GuPFailingPrune){
            gup_fp.SuccessMatch(depth, u, v);
        }
        if constexpr (Enable_VEQAutomorphismPrune){
            veq_ap.SuccessMatch(depth, u, v_idx);
        }
        if constexpr (Enable_DAFFailingPrune){
            daf_fp.SuccessMatch(depth, v);
        }

    }

    void inline FinalReturn(uint32_t depth){
        if constexpr (Enable_GuPFailingPrune){
            gup_fp.FinalReturn(depth);
        }
    }

    bool inline NoCandidatesPruneCheck(uint32_t cnt, uint32_t depth, VertexID u){
        bool gup_fp_flag = false;
        bool daf_fp_flag = false;

        if constexpr (Enable_GuPFailingPrune){
            gup_fp_flag = gup_fp.NoCandidatesConflictCheck(cnt, depth, u);
        }

        if constexpr (Enable_DAFFailingPrune){
            daf_fp_flag = daf_fp.NoCandidatesConflictCheck(cnt, depth);
        }

        return gup_fp_flag | daf_fp_flag;
    }

    void inline InjectiveConflict(uint32_t depth, VertexID u, uint32_t conf_depth, VertexID conf_u, uint32_t v_idx_at_dep, uint32_t v_idx_at_conf_dep){

        if constexpr (Enable_GuPFailingPrune){
            gup_fp.InjectiveConflict(depth, u, conf_depth, v_idx_at_dep);
        }

        if constexpr (Enable_VEQAutomorphismPrune){
            veq_ap.InjectiveConflict(depth, v_idx_at_dep, conf_depth, v_idx_at_conf_dep);
        }

        if constexpr (Enable_DAFFailingPrune){
            daf_fp.InjectiveConflict(depth, conf_depth);
        }

        // * To compatible with the whole framework
        // * We need add stack_size, since when BacktrackPrune, the stack_size will -1;
        if constexpr (Enable_BICEConflictDetection){
            for(auto uf: bice_cd.fnebs[u])
                bice_cd.stack_size[uf]++;
        }

    }

    bool inline ForwardPrune(uint32_t depth, VertexID u, uint32_t v_idx, VertexID v, bool *vis, uint32_t *reverse_embeddings_deps, VertexID **candidates, Edges *** edge_matrix, size_t &ans){
        bool gup_fp_flag = false;
        bool gup_cd_flag = false;
        bool veq_ap_flag = false;
        bool bice_cd_flag = false;
        
        if constexpr (Enable_VEQAutomorphismPrune){
            if(depth != 0)
                veq_ap_flag = veq_ap.EquivalencePruneCheck(depth, u, v, v_idx, candidates, ans);
        }

        if constexpr (Enable_GuPFailingPrune){
            gup_fp_flag = gup_fp.Nogood_V_Check(depth, v_idx, v);
        }

        if constexpr (Enable_GUPConflictDetection){
            gup_cd_flag = gup_cd.matchCheck(depth, v_idx, vis, reverse_embeddings_deps);
        }

        if constexpr (Enable_BICEConflictDetection){
            bice_cd_flag = bice_cd.IsConflict(u, v, v_idx, edge_matrix);
        }
        

        // reset the intermediate result for GuP Failing
        if constexpr (Enable_GuPFailingPrune){
            if constexpr (Enable_GUPConflictDetection){
                if(gup_cd_flag){
                    gup_fp._inter_result._mask = gup_cd.getCheckResult();
                }
                gup_fp._inter_result._useMask = gup_cd_flag;
            }

            if(veq_ap_flag)
                gup_fp._inter_result.setNoUse();

            if(bice_cd_flag)
                gup_fp._inter_result.setNoUse();
        }

        if constexpr (Enable_DAFFailingPrune){
            if(gup_fp_flag | gup_cd_flag | veq_ap_flag | bice_cd_flag){
                // daf_fp._fail_mask[depth - 1].set();
                daf_fp._fail_mask[depth].set();
            }
        }

        return gup_fp_flag | gup_cd_flag | veq_ap_flag | bice_cd_flag;
    }


};