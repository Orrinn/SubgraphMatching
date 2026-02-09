#pragma once
#include "graph.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include <assert.h>
#include "common_ops.h"
#include "common_type.h"
#include "api.h"
#include "prune.h"
#include "time.h"
#include <atomic>


template<typename Derived>
class ExploreBuffer{
public:
    uint32_t *pos{nullptr};
    uint32_t *tot{nullptr};
    uint32_t *embedding{nullptr};
    uint32_t *idx_embedding{nullptr};
    uint32_t *temp_buffer{nullptr};
    uint32_t **valid_candidate_idx{nullptr};
    bool *vis{nullptr};
    uint32_t *qv_depth{nullptr};
    uint32_t qvcnt{0};
    const LabelID* q_labels{nullptr}; // query labels // ! Note: q_lables is just a copy, do not use New or Delete on it.
    PreviousNeb* prevNeb{nullptr}; // ! NOTE:  prevNeb use VERTEX as idx, NOT the depth

    // ! Don't use unordered_map since it will introduce considerable overhead.
    uint32_t *reverse_embeddings_deps{nullptr};

    void inline getNextDataVertex(uint32_t cur_depth, VertexID u, const Config& cfg, VertexID& v, uint32_t& v_idx){
        static_cast<Derived*>(this)->getNextDataVertexImpl(cur_depth, u, cfg, v, v_idx);  
    }

    void inline allocate(const Graph &data, const Query &query, const Order& order, const Config& cfg){
        static_cast<Derived*>(this)->allocate(data, query, order, cfg);
    }

    template<typename FailingPrune>
    bool inline getNextCandidates(uint32_t depth, const Graph& data, const Order& order, const Config& cfg, FailingPrune *fp) {
        if constexpr (std::is_same<FailingPrune, GUPFailingPrune>::value){
            return static_cast<Derived*>(this)->getNextCandidatesImpl(depth, data, order, cfg, fp);
        } 
        else{
            return static_cast<Derived*>(this)->getNextCandidatesImpl(depth, data, order, cfg);
        }
        
    }

    bool inline getNextCandidates(uint32_t depth, const Graph& data, const Order& order, const Config& cfg) {
        return static_cast<Derived*>(this)->getNextCandidatesImpl(depth, data, order, cfg);
    }

    bool inline getNextCandidates(uint32_t depth, const Graph& data, const Order& order, const Config& cfg, uint32_t &conf_dep) {
        return static_cast<Derived*>(this)->getNextCandidatesImpl(depth, data, order, cfg, conf_dep);
    }

    void inline release(){
#ifdef LOG_OUTPUT
        spdlog::trace("buffer released");
#endif
        q_labels = nullptr;
        delete[] pos;
        delete[] tot;
        delete[] embedding;
        delete[] idx_embedding;
        delete[] vis;
        delete[] prevNeb;
        delete[] temp_buffer;
        delete[] qv_depth;
        delete[] reverse_embeddings_deps;
        for (uint32_t i = 0; i < qvcnt; ++i) {
            delete[] valid_candidate_idx[i];
        }

        delete[] valid_candidate_idx;
        static_cast<Derived*>(this)->releaseImpl();
    }

    void inline common_allocate(const Query &query, const Order& order, uint32_t qvcnt, uint32_t dvcnt, uint32_t max_can_cnt, uint32_t d_max_dgree){
        
        qv_depth = new uint32_t[qvcnt];
        for(int i=0; i<qvcnt; i++)
            qv_depth[order[i]] = i;

        q_labels = query.getVertexLabels();

        prevNeb = new PreviousNeb[qvcnt];
        GetPreviousNeb(query, order, prevNeb);

        reverse_embeddings_deps = new uint32_t[dvcnt];
        memset(reverse_embeddings_deps, -1, sizeof(uint32_t) * dvcnt);

        pos = new uint32_t[qvcnt];
        tot = new uint32_t[qvcnt];
        embedding = new uint32_t[qvcnt];
        idx_embedding = new uint32_t[qvcnt];
        vis = new bool[dvcnt];
        temp_buffer = new uint32_t[d_max_dgree];
        valid_candidate_idx = new uint32_t *[qvcnt];
        for (uint32_t i = 0; i < qvcnt; ++i) {
            valid_candidate_idx[i] = new uint32_t[max_can_cnt];
        }

        std::fill(vis, vis + dvcnt, false);
    }

};

class CacheBuffer: public ExploreBuffer<CacheBuffer>{
public:
    // for reuse
    VertexSet* vertexSets{nullptr};
    VertexSet* vertexNebSets{nullptr};

    void inline getNextDataVertexImpl(uint32_t cur_depth, VertexID u, const Config& cfg, VertexID& v, uint32_t& v_idx) {
        if(cur_depth == 0) [[unlikely]] {
            v_idx = this->valid_candidate_idx[cur_depth][this->pos[cur_depth]];
            v = v_idx;
        }
        else{
            uint32_t vsetId = cfg.reuse->plan->getIterVSetAt(cur_depth);
            v_idx = this->vertexSets[vsetId].getData(this->pos[cur_depth]);
            v = v_idx;
        }
    }

    void allocate(const Graph &data, const Query &query, const Order& order, const Config& cfg) {
        ReuseParam* reuseParam = cfg.reuse;
        qvcnt = query.getVertexCnt();
        uint32_t data_vertices_num = data.getVertexCnt();
        uint32_t data_max_dgree = data.getMaxDgree();
        uint32_t max_candidates_num = data_vertices_num;
        vertexSets = new VertexSet[reuseParam->plan->getTotOp()];
        for(int i=0; i<reuseParam->plan->getTotOp(); i++)
            vertexSets[i].alloc(data_max_dgree);

        vertexNebSets = new VertexSet[qvcnt];
        for(int i=0; i<qvcnt; i++)
            vertexNebSets[i].alloc(data_max_dgree);
        ExploreBuffer::common_allocate(query, order, qvcnt, data_vertices_num, max_candidates_num, data_max_dgree);
    }


    bool getNextCandidatesImpl(uint32_t depth, const Graph& data, const Order& order, const Config& cfg) {
        ReuseParam* reuseParam = cfg.reuse;
        // for loop reuse, we need to generate the intermediate result in the last loop
        uint32_t cur_dep = depth - 1; 
        const auto &opsByDepth = reuseParam->plan->getSetOps();
        const auto &opsArr = reuseParam->plan->getTotSetOpsArr();

        bool is_candidates_valid = true;

        for(auto vsetId: opsByDepth[cur_dep]){

            VertexID u = order[cur_dep];
            VertexID v = embedding[u];
            uint32_t vnebCnt = 0;

            const VertexSetIR &vsetIR = opsArr[vsetId];
            uint32_t pvsetId = vsetIR.getParentVSetID();
            LabelID required_label = vsetIR.getLabel();

            // const VertexID* vneb = data.getNeb(v, vnebCnt);
            const VertexID* vneb = data.getNebByLabel(v, vnebCnt, required_label);

            if(pvsetId == INVALID){
                // * This data (The neb of vertex v) wouldn't be modified, so we can safely remove the const
                vertexSets[vsetId].Init(const_cast<VertexID*>(vneb), vnebCnt);
            }
            else{
                // * This data (The neb of vertex v) wouldn't be modified, so we can safely remove the const
                vertexNebSets[u].Init(const_cast<VertexID*>(vneb), vnebCnt);
                if constexpr (Profiler::useProfiling){
                    Profiler::getInst().total_intersection++;
                }
                vertexSets[vsetId].IntersecOf(vertexSets[pvsetId], vertexNebSets[u]);
            }

            if(vertexSets[vsetId].getSize() == 0){
                is_candidates_valid = false;
                break;
            }
        }

        // set the data structure correctly
        uint32_t nextVSetId = reuseParam->plan->getIterVSetAt(depth);
        tot[depth] = vertexSets[nextVSetId].getSize();

        return is_candidates_valid;
    }

    // ! not implement the labeled version
    bool getNextCandidatesImpl(uint32_t depth, const Graph& data, const Order& order, const Config& cfg, uint32_t &conf_dep) {
        ReuseParam* reuseParam = cfg.reuse;
        // for loop reuse, we need to generate the intermediate result in the last loop
        uint32_t cur_dep = depth - 1; 
        const auto &opsByDepth = reuseParam->plan->getSetOps();
        const auto &opsArr = reuseParam->plan->getTotSetOpsArr();

        bool is_candidates_valid = true;

        for(auto vsetId: opsByDepth[cur_dep]){

            VertexID u = order[cur_dep];
            VertexID v = embedding[u];
            uint32_t vnebCnt = 0;
            const VertexID* vneb = data.getNeb(v, vnebCnt);

            const VertexSetIR &vsetIR = opsArr[vsetId];
            uint32_t pvsetId = vsetIR.getParentVSetID();
            if(pvsetId == INVALID){
                // * This data (The neb of vertex v) wouldn't be modified, so we can safely remove the const
                vertexSets[vsetId].Init(const_cast<VertexID*>(vneb), vnebCnt);
            }
            else{
                // * This data (The neb of vertex v) wouldn't be modified, so we can safely remove the const
                vertexNebSets[u].Init(const_cast<VertexID*>(vneb), vnebCnt);
                if constexpr (Profiler::useProfiling){
                    Profiler::getInst().total_intersection++;
                }
                vertexSets[vsetId].IntersecOf(vertexSets[pvsetId], vertexNebSets[u]);
            }

            if(vertexSets[vsetId].getSize() == 0){
                is_candidates_valid = false;
                conf_dep = reuseParam->plan->getTotSetOpsArr()[vsetId].getVDepth();
                break;
            }
        }

        // set the data structure correctly
        uint32_t nextVSetId = reuseParam->plan->getIterVSetAt(depth);
        tot[depth] = vertexSets[nextVSetId].getSize();

        return is_candidates_valid;
    }

    void releaseImpl(){
        if(vertexSets)
            delete[] vertexSets;
        if(vertexNebSets)
            delete[] vertexNebSets;
    }


};

class CandidatesBuffer: public ExploreBuffer<CandidatesBuffer>{
public:

    void inline getNextDataVertexImpl(uint32_t cur_depth, VertexID u, const Config& cfg, VertexID& v, uint32_t& v_idx) {
        v_idx = this->valid_candidate_idx[cur_depth][this->pos[cur_depth]];
        v = cfg.can->candidates[u][v_idx];
    }

    void allocate(const Graph &data, const Query &query, const Order& order, const Config& cfg) {
        CandidateParam* canParam = cfg.can;
        qvcnt = query.getVertexCnt();
        uint32_t data_vertices_num = data.getVertexCnt();
        uint32_t data_max_dgree = data.getMaxDgree();
        uint32_t max_candidates_num = data_vertices_num;
        max_candidates_num = canParam->candidates_count[0];

        for (uint32_t i = 1; i < qvcnt; ++i) {
            VertexID cur_vertex = i;
            uint32_t cur_candidate_num = canParam->candidates_count[cur_vertex];

            if (cur_candidate_num > max_candidates_num) {
                max_candidates_num = cur_candidate_num;
            }
        }

        ExploreBuffer::common_allocate(query, order, qvcnt, data_vertices_num, max_candidates_num, data_max_dgree);
    }

    template<typename FailingPrune = void>
    bool getNextCandidatesImpl(uint32_t depth, const Graph& data, const Order& order, const Config& cfg, FailingPrune *fp = nullptr) {
        CandidateParam* canParam = cfg.can;
        Edges ***edge_matrix = canParam->edge_matrix;
        uint32_t *&temp_buffer = this->temp_buffer;

        // static int total = 0;

        VertexID u = order[depth];
        VertexID prev_neb = prevNeb[u][0];
        uint32_t prev_embed_idx = idx_embedding[prev_neb];
        uint32_t valid_can_cnt = 0;

        // uint32_t c_i = 0;

        Edges &prev_edge = *edge_matrix[prev_neb][u];
        const uint32_t* prev_can = prev_edge.getNeb(prev_embed_idx, valid_can_cnt);
        // c_i += valid_can_cnt;
        // printf("intersection :\n");
        // for(int i=0; i<valid_can_cnt; i++)
        //     printf("%d ", prev_can[i]);
        // printf("\n");

        memcpy(valid_candidate_idx[depth], prev_can, sizeof(uint32_t)*valid_can_cnt);

        if constexpr (std::is_same<FailingPrune, GUPFailingPrune>::value){
            fp->UpdateBounding(depth, qv_depth[prev_neb]);
        }
        
        uint32_t temp_valid_cnt;

        for(int i=1; i<prevNeb[u].size(); i++){
            VertexID _prev_neb = prevNeb[u][i];
            VertexID prev_embed_idx = idx_embedding[_prev_neb];
            Edges &edge = *edge_matrix[_prev_neb][u];
            uint32_t cur_prev_can_cnt;
            const VertexID* cur_prev_can = edge.getNeb(prev_embed_idx, cur_prev_can_cnt);

            // c_i += cur_prev_can_cnt;

            if constexpr (Profiler::useProfiling){
                Profiler::getInst().total_intersection++;
            }
                
            Intersection(cur_prev_can, cur_prev_can_cnt, valid_candidate_idx[depth], valid_can_cnt, temp_buffer, temp_valid_cnt);

            if constexpr (std::is_same<FailingPrune, GUPFailingPrune>::value){
                if(valid_can_cnt != temp_valid_cnt)
                    fp->UpdateBounding(depth, qv_depth[_prev_neb]);
            }

            // for(int i=0; i<cur_prev_can_cnt; i++)
            //     printf("%d ", cur_prev_can[i]);
            // printf("\n");

            std::swap(temp_buffer, valid_candidate_idx[depth]);
            valid_can_cnt = temp_valid_cnt;

        }

        tot[depth] = valid_can_cnt;
        // printf("final intersection :\n");
        // for(int i=0; i<valid_can_cnt; i++)
        //     printf("%d ", valid_candidate_idx[depth][i]);
        // printf("\n");

        // printf("depth %d, c_i %d, l_i %d\n", depth, c_i, valid_can_cnt);
        // total += c_i;
        // total += valid_can_cnt;
        // printf("total %d\n", total);
        return valid_can_cnt!=0;
    }

    void constexpr releaseImpl() noexcept {}

};

class NaiveBuffer: public ExploreBuffer<NaiveBuffer>{
public:

    void inline getNextDataVertexImpl(uint32_t cur_depth, VertexID u, const Config& cfg, VertexID& v, uint32_t& v_idx) {
        v_idx = this->valid_candidate_idx[cur_depth][this->pos[cur_depth]];
        v = v_idx;
    }

    void allocate(const Graph &data, const Query &query, const Order& order, const Config& cfg) {
        qvcnt = query.getVertexCnt();
        uint32_t data_vertices_num = data.getVertexCnt();
        uint32_t data_max_dgree = data.getMaxDgree();

        ExploreBuffer::common_allocate(query, order, qvcnt, data_vertices_num, data_vertices_num, data_max_dgree);
    }


    bool getNextCandidatesImpl(uint32_t depth, const Graph& data, const Order& order, const Config& cfg) {

        uint32_t *&temp_buffer = this->temp_buffer;

        VertexID u = order[depth];
        LabelID u_label = q_labels[u];
        VertexID prev_neb = prevNeb[u][0];
        VertexID prev_embed = embedding[prev_neb];
        uint32_t prev_neb_cnt = 0;

        // const uint32_t* v_neb = data.getNeb(prev_embed, prev_neb_cnt);
        const uint32_t* v_neb = data.getNebByLabel(prev_embed, prev_neb_cnt, u_label);

        memcpy(valid_candidate_idx[depth], v_neb, sizeof(uint32_t)*prev_neb_cnt);
        
        uint32_t temp_valid_cnt;

        for(int i=1; i<prevNeb[u].size(); i++){
            uint32_t _prev_neb_cnt = 0;
            VertexID _prev_neb = prevNeb[u][i];
            VertexID _prev_embed = embedding[_prev_neb];
            // const VertexID* _v_neb = data.getNeb(_prev_embed, _prev_neb_cnt);
            const VertexID* _v_neb = data.getNebByLabel(_prev_embed, _prev_neb_cnt, u_label);

            if constexpr (Profiler::useProfiling){
                Profiler::getInst().total_intersection++;
            }

            Intersection(_v_neb, _prev_neb_cnt, valid_candidate_idx[depth], prev_neb_cnt, temp_buffer, temp_valid_cnt);

            std::swap(temp_buffer, valid_candidate_idx[depth]);
            prev_neb_cnt = temp_valid_cnt;

        }

        tot[depth] = prev_neb_cnt;
        return prev_neb_cnt!=0;
    }

    void constexpr releaseImpl() noexcept {}
};


// TODO: we should assemble a FAKE canParam for intermediate result reuse, for compatibility
template<typename BufferType>
size_t explore_Intersection(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, bool isProfile = false, uint64_t output_limit = -1){

    cfg.Check();

    BufferType buffer;
    
    buffer.allocate(data, query, order, cfg);

    size_t embed_cnt = 0;
    int cur_depth = 0;
    int qv_cnt = query.getVertexCnt();
    VertexID u_start = order[0];

    CandidateParam* canParam = nullptr;
    // ReuseParam* reuseParam = nullptr;

    if(cfg.useFilter) canParam = cfg.can;
    // else if(cfg.useCache) reuseParam = cfg.reuse;

    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = canParam->candidates_count[u_start];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both

    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
        
    auto start = std::chrono::high_resolution_clock::now();
    while (1)
    {
        while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){

            VertexID v=0;
            uint32_t v_idx = 0;
            VertexID u = order[cur_depth];

            // * We need CRPT to eliminate the if-else branch.
            // * According to our experimental results, using if-else branches can slow down the program by 2-3 times.
            buffer.getNextDataVertex(cur_depth, u, cfg, v, v_idx);

            if(buffer.vis[v]){
                buffer.pos[cur_depth] += 1;
                continue;
            }

            buffer.embedding[u] = v;
            buffer.idx_embedding[u] = v_idx;
            buffer.vis[v] = true;
            buffer.pos[cur_depth] += 1;
            buffer.reverse_embeddings_deps[v] = cur_depth;

            if(cur_depth == qv_cnt - 1) [[unlikely]] {
                embed_cnt += 1;
                buffer.vis[v] = false;
                // * We may don't have to maintain the reverse when backtracing
                // buffer.reverse_embeddings_deps[v] = INVALID;
                // for(int i=0; i<qv_cnt; i++)
                //     printf("%d ", buffer.embedding[i]);
                // printf("\n");
                if(output_limit != -1 && embed_cnt >= output_limit){
                    goto EXIT;
                }
            }
            else{
                cur_depth += 1;
                buffer.pos[cur_depth] = 0;

                buffer.getNextCandidates(cur_depth, data, order, cfg);
            }
        }

        cur_depth -= 1;
        if(cur_depth < 0) [[unlikely]] break;
        else{
            VertexID u = order[cur_depth];
            buffer.vis[buffer.embedding[u]] = false;
            // buffer.reverse_embeddings_deps[buffer.embedding[u]] = INVALID;
        }
    }
    
EXIT:

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time = duration.count();
    
    buffer.release();

    return embed_cnt;
}


template<
    typename BufferType
>
void _explore_recur(const Graph& data, const Query& query, const Order& order, const Config &cfg, BufferType* buffer, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    // std::cout << "Exploring at depth " << cur_dep << std::endl;

    VertexID u = order[cur_dep];
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(is_candidates_valid == false){
#ifdef LOG_OUTPUT
        spdlog::trace("No candidates at depth {}, skip", cur_dep);
#endif   
        return;
    }
    // if(buffer->tot[cur_dep] == 0) return;

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
    // while(buffer->pos[cur_dep] < buffer->tot[cur_dep]){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);
        // if(data.getVertexLabel(v) != query.getVertexLabel(u))
        //     continue;

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} (label {}) v {} (label {}) at depth {}, Embeddings {}", u, query.getVertexLabel(u), v, data.getVertexLabel(v),cur_dep, [&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif   

        if(buffer->vis[v]){
#ifdef LOG_OUTPUT
        spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif   
            continue;
        }

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().total_iter_count[cur_dep]++;
        }


        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            // std::cout << "Found Embeddings: [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", ");
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
#ifdef ENABLE_SAMPLE
            if constexpr (Profiler::useProfiling){
                if((ans & (Profiler::getInst().sample_inter - 1)) == 0){
                    // printf("aaa\n");
                    Profiler::getInst().timestamps.push_back(std::chrono::high_resolution_clock::now());
                }
            }
#endif
#ifdef ENABLE_OUTPUT_CMP
            std::cout << fmt::format("{},{}\n", ans, [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            if(ans == output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            buffer->embedding[u] = v;
            buffer->idx_embedding[u] = v_idx;
            buffer->vis[v] = true;
            buffer->reverse_embeddings_deps[v]=cur_dep;
            // failingPrune.ExtendIndex(cur_depth, u, v);

            _explore_recur<BufferType>(data, query, order, cfg, buffer, cur_dep + 1, ans, stop_flag, output_limit);
            if(stop_flag.load()) [[unlikely]]{
                return;
            }
            
            buffer->vis[v] = false;
            // buffer->reverse_embeddings_deps[v]=INVALID;
        }
    }
}

template<
    typename BufferType
>
void explore_Intersection_Recursive(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    BufferType buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    if constexpr (Profiler::useProfiling){
        Profiler::getInst().start_time = std::chrono::high_resolution_clock::now();
    }

    // std::cout << "Start exploring with depth " << cur_depth << ", tot: " << buffer.tot[cur_depth] << std::endl;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);
        if(data.getVertexLabel(v0) != query.getVertexLabel(u0))
            continue;

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().total_iter_count[cur_depth]++;
        }


        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur<BufferType>(data, query, order, cfg, &buffer, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}



void _explore_recur_cache_daf(const Graph& data, const Query& query, const Order& order, const Config &cfg, CacheBuffer* buffer, DAFFailingPrune* fp, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    uint32_t conf_dep = 0;
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg, conf_dep);
    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(is_candidates_valid == false){
#ifdef LOG_OUTPUT
        spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
#endif  
        fp->NoCandidatesConflictCheckCache(0, cur_dep, conf_dep);
        return;
    }
    else{
        fp->NoCandidatesConflictCheckCache(1, cur_dep, conf_dep);
    }
//     if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep)){
// #ifdef LOG_OUTPUT
//         spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
// #endif  
//         return;
//     }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep,[&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        if(buffer->vis[v]){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif
            fp->InjectiveConflict(cur_dep, buffer->reverse_embeddings_deps[v]);
            goto _PruneUpdate;
        }


        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            fp->SuccessMatch(cur_dep, v);
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // buffer->embedding[u] = v;
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            _explore_recur_cache_daf(data, query, order, cfg, buffer, fp, cur_dep + 1, ans, stop_flag,output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                break;
            }
        }

        buffer->vis[v] = false;
_PruneUpdate:
        if(fp->PruneCheck(cur_dep))
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
    }
}

void explore_Intersection_Cache_DAF(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CacheBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    DAFFailingPrune fp;
    fp.Init(query, order);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur_cache_daf(data, query, order, cfg, &buffer, &fp, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
        if(fp.PruneCheck(cur_depth)){
            buffer.pos[cur_depth] = buffer.tot[cur_depth];
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}


void _explore_recur_cache_daf_profile(const Graph& data, const Query& query, const Order& order, const Config &cfg, CacheBuffer* buffer, DAFFailingPrune* fp, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    uint32_t conf_dep = 0;
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg, conf_dep);
    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(is_candidates_valid == false){
#ifdef LOG_OUTPUT
        spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
#endif  
        fp->NoCandidatesConflictCheckCache(0, cur_dep, conf_dep);
        return;
    }
    else{
        fp->NoCandidatesConflictCheckCache(1, cur_dep, conf_dep);
    }
//     if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep)){
// #ifdef LOG_OUTPUT
//         spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
// #endif  
//         return;
//     }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        bool conflict = false;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep,[&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        if(buffer->vis[v]){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif
            fp->InjectiveConflict(cur_dep, buffer->reverse_embeddings_deps[v]);
            conflict = true;
            goto _PruneUpdate;
        }

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().total_iter_count[cur_dep]++;
        }


        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            fp->SuccessMatch(cur_dep, v);
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // buffer->embedding[u] = v;
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            _explore_recur_cache_daf_profile(data, query, order, cfg, buffer, fp, cur_dep + 1, ans, stop_flag,output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                break;
            }
        }

        buffer->vis[v] = false;
_PruneUpdate:
        if(fp->PruneCheck(cur_dep)){
            if constexpr (Profiler::useProfiling){
                if(buffer->tot[cur_dep] != (buffer->pos[cur_dep] + 1)){
                    Profiler::getInst().pruned_count[cur_dep]++;
                    for(int profile_index=buffer->pos[cur_dep]+1; profile_index<buffer->tot[cur_dep]; profile_index++){
                        uint32_t temp_id = cfg.reuse->plan->getIterVSetAt(cur_dep);
                        VertexID v_tmp = buffer->vertexSets[temp_id].getData(profile_index);
                        if(buffer->vis[v_tmp] == false) Profiler::getInst().pruned_iteration[cur_dep]++;
                    }
                }
            }
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
        }
    }
}

void explore_Intersection_Cache_DAF_profile(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CacheBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    DAFFailingPrune fp;
    fp.Init(query, order);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().total_iter_count[cur_depth]++;
        }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur_cache_daf_profile(data, query, order, cfg, &buffer, &fp, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
        if(fp.PruneCheck(cur_depth)){
            buffer.pos[cur_depth] = buffer.tot[cur_depth];
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}

void _explore_recur_cache_daf_filter(const Graph& data, const Query& query, const Order& order, const Config &cfg, CacheBuffer* buffer, DAFFailingPrune* fp, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    uint32_t conf_dep = 0;
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg, conf_dep);
    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);
    uint32_t total_skip=0;

    if(is_candidates_valid == false){
#ifdef LOG_OUTPUT
        spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
#endif  
        fp->NoCandidatesConflictCheckCache(0, cur_dep, conf_dep);
        return;
    }
    else{
        fp->NoCandidatesConflictCheckCache(1, cur_dep, conf_dep);
    }
//     if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep)){
// #ifdef LOG_OUTPUT
//         spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
// #endif  
//         return;
//     }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if(cfg.can->candidates_exsist[u][v] == false){
#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, v not in filter, continue", u, v, cur_dep);
#endif
            total_skip++;
            continue;
        }

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep,[&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        if(buffer->vis[v]){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif
            fp->InjectiveConflict(cur_dep, buffer->reverse_embeddings_deps[v]);
            goto _PruneUpdate;
        }


        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            fp->SuccessMatch(cur_dep, v);
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // buffer->embedding[u] = v;
#ifdef ENABLE_SAMPLE
            if constexpr (Profiler::useProfiling){
                if((ans & (Profiler::getInst().sample_inter - 1)) == 0){
                    // printf("aaa\n");
                    Profiler::getInst().timestamps.push_back(std::chrono::high_resolution_clock::now());
                }
            }
#endif
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            _explore_recur_cache_daf_filter(data, query, order, cfg, buffer, fp, cur_dep + 1, ans, stop_flag,output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                break;
            }
        }

        buffer->vis[v] = false;
_PruneUpdate:
        if(fp->PruneCheck(cur_dep))
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
    }

    if(total_skip == buffer->tot[cur_dep]) [[unlikely]] {
        fp->NoCandidatesConflictCheck(0, cur_dep);
    }
}

void explore_Intersection_Cache_DAF_Filter(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CacheBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    DAFFailingPrune fp;
    fp.Init(query, order);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    
    if constexpr (Profiler::useProfiling){
        Profiler::getInst().start_time = std::chrono::high_resolution_clock::now();
    }

    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);
        v0 = cfg.can->candidates[u0][v0_id];

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur_cache_daf_filter(data, query, order, cfg, &buffer, &fp, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
        if(fp.PruneCheck(cur_depth)){
            buffer.pos[cur_depth] = buffer.tot[cur_depth];
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}

void _explore_recur_cache_filter_profile(const Graph& data, const Query& query, const Order& order, const Config &cfg, CacheBuffer* buffer, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    uint32_t conf_dep = 0;
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg, conf_dep);
    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);


    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().total_iter_count[cur_dep]++;
        }

        if(cfg.can->candidates_exsist[u][v] == false){
#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, v not in filter, continue", u, v, cur_dep);
#endif
            continue;
        }

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().filter_hit_count[cur_dep]++;
        }

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep,[&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        if(buffer->vis[v]){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().conflict_count[cur_dep]++;
            }
            // goto _PruneUpdate;
            continue;
        }


        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // buffer->embedding[u] = v;
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            _explore_recur_cache_filter_profile(data, query, order, cfg, buffer, cur_dep + 1, ans, stop_flag,output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                break;
            }
        }

        buffer->vis[v] = false;
// _PruneUpdate:
    }

    // if(total_skip == buffer->tot[cur_dep]) [[unlikely]] {
    //     fp->NoCandidatesConflictCheck(0, cur_dep);
    // }
}

void explore_Intersection_Cache_Filter_profile(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CacheBuffer buffer;
    buffer.allocate(data, query, order, cfg);


    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
    
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);
        v0 = cfg.can->candidates[u0][v0_id];

        if(Profiler::useProfiling){
// since we iterate the candidates in asceding order, 
// therefore, the v0 in first depth is the total iteration that need be performed without filter
            Profiler::getInst().total_iter_count[cur_depth] = v0; 
            Profiler::getInst().filter_hit_count[cur_depth]++;
        }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur_cache_filter_profile(data, query, order, cfg, &buffer, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}

void _explore_recur_cache_filter(const Graph& data, const Query& query, const Order& order, const Config &cfg, CacheBuffer* buffer, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    uint32_t conf_dep = 0;
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg, conf_dep);
    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);


    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);
        
        if(cfg.can->candidates_exsist[u][v] == false){
#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, v not in filter, continue", u, v, cur_dep);
#endif
            continue;
        }

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep,[&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        if(buffer->vis[v]){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif
            // goto _PruneUpdate;
            continue;
        }


        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // buffer->embedding[u] = v;
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            _explore_recur_cache_filter(data, query, order, cfg, buffer, cur_dep + 1, ans, stop_flag,output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                break;
            }
        }

        buffer->vis[v] = false;
// _PruneUpdate:
    }

    // if(total_skip == buffer->tot[cur_dep]) [[unlikely]] {
    //     fp->NoCandidatesConflictCheck(0, cur_dep);
    // }
}

void explore_Intersection_Cache_Filter(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CacheBuffer buffer;
    buffer.allocate(data, query, order, cfg);


    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
    
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);
        v0 = cfg.can->candidates[u0][v0_id];

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur_cache_filter(data, query, order, cfg, &buffer, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}


void _explore_recur_baseline_daf(const Graph& data, const Query& query, const Order& order, const Config &cfg, NaiveBuffer* buffer, DAFFailingPrune* fp, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    uint32_t conf_dep = 0;
    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg, conf_dep);
    bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);

//     if(is_candidates_valid == false){
// #ifdef LOG_OUTPUT
//         spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
// #endif  
//         fp->NoCandidatesConflictCheckCache(0, cur_dep, conf_dep);
//         return;
//     }
//     else{
//         fp->NoCandidatesConflictCheckCache(1, cur_dep, conf_dep);
//     }
    if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep)){
#ifdef LOG_OUTPUT
        spdlog::trace("No candidates at depth {}, conf_dep {}, skip", cur_dep, conf_dep);
#endif  
        return;
    }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep,[&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        if(buffer->vis[v]){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif
            fp->InjectiveConflict(cur_dep, buffer->reverse_embeddings_deps[v]);
            goto _PruneUpdate;
        }


        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            fp->SuccessMatch(cur_dep, v);
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // buffer->embedding[u] = v;
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
        }
        else{
            _explore_recur_baseline_daf(data, query, order, cfg, buffer, fp, cur_dep + 1, ans, stop_flag,output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                break;
            }
        }

        buffer->vis[v] = false;
_PruneUpdate:
        if(fp->PruneCheck(cur_dep))
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
    }
}

void explore_Intersection_Baseline_DAF(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    NaiveBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    DAFFailingPrune fp;
    fp.Init(query, order);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    // for(int buffer.pos[cur_depth]=0; i<buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        _explore_recur_baseline_daf(data, query, order, cfg, &buffer, &fp, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
        if(fp.PruneCheck(cur_depth)){
            buffer.pos[cur_depth] = buffer.tot[cur_depth];
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}

void _explore_recur_GUP(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, GUPFailingPrune *fp, int cur_dep, size_t &ans, uint64_t output_limit){

    VertexID u = order[cur_dep];
    buffer->getNextCandidates(cur_dep, data, order, cfg, fp);

    fp->BoundingsBackUp(cur_dep, u);
    if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep, u)) {
        return;
    }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

#ifdef LOG_OUTPUT
        spdlog::trace("Extend: depth {}, u {}, v {}, partial embed [{}], boundings {}", 
            cur_dep,u,v,
            [buffer, &order, cur_dep](){
                std::vector<VertexID> v;
                for(int i=0; i<=cur_dep; i++) v.push_back(buffer->embedding[order[i]]);
                return fmt::format("{}", fmt::join(v, ","));
            }(),
            [fp, cur_dep](){
                std::vector<int> v;
                for(int i=0; i<fp->qvcnt; i++) v.push_back(fp->_bounding[cur_dep][i]);
                return fmt::format("{}", fmt::join(v, ""));
            }()
        );
#endif
        if(buffer->vis[v]){
            fp->InjectiveConflict(cur_dep, u, buffer->reverse_embeddings_deps[v], v_idx);
            goto _pruningUpdate;
        }
        
        if(fp->Nogood_V_Check(cur_dep, v_idx, v)){
            goto _pruningUpdate;
        }

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        fp->ExtendIndex(cur_dep, u, v);
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            fp->SuccessMatch(cur_dep, u, v);
            // buffer->embedding[u] = v;
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;32mFound Embedding\x1b[0m : [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
#endif
        }
        else{
            _explore_recur_GUP(data, query, order, cfg, buffer, fp, cur_dep + 1, ans, output_limit); 
            
            fp->BoundingsRecover(cur_dep, u);
        }

        buffer->vis[v] = false;
        fp->ReduceIndex(cur_dep, v);
_pruningUpdate:
        
#ifdef LOG_OUTPUT
        spdlog::trace("Reduced: depth {}, u {}, v {}, partial embed [{}], inter_result: {{{}}}", 
            cur_dep,u,v,
            [buffer, &order, cur_dep](){
                std::vector<VertexID> v;
                for(int i=0; i<=cur_dep; i++) v.push_back(buffer->embedding[order[i]]);
                return fmt::format("{}", fmt::join(v, ","));
            }(),
            fp->_inter_result.debug_info(query.getVertexCnt())
        );
#endif
        if(fp->Nogood_V_Update(cur_dep, v_idx, v, query.getKCoreValue(u)))
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
    }

    fp->FinalReturn(cur_dep);
}


void explore_Intersection_Recursive_GUP(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    GUPFailingPrune fp;
    fp.Init(query, order, cfg);
    fp.BoundingsBackUp(cur_depth, u0);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        fp.ExtendIndex(cur_depth, u0, v0);
#ifdef LOG_OUTPUT
        spdlog::trace("Extend: depth {}, u {}, v {}, partial embed [{}]", 
            cur_depth, u0, v0,
            [&buffer, &order, cur_depth](){
                std::vector<VertexID> vec;
                for(int i=0; i<=cur_depth; i++) vec.push_back(buffer.embedding[order[i]]);
                return fmt::format("{}", fmt::join(vec, ","));
            }()
        );
#endif
        _explore_recur_GUP(data, query, order, cfg, &buffer, &fp, cur_depth + 1, ans, output_limit);
        buffer.vis[v0] = false;
        fp.BoundingsRecover(cur_depth, u0);
        fp.ReduceIndex(cur_depth, v0);
        if(fp.Nogood_V_Update(0, v0_id, v0, query.getKCoreValue(u0)))
            buffer.pos[0] = buffer.tot[0];
        
        // buffer.pos[cur_depth]++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time = duration.count();

    buffer.release();
}

// Min priority queue.
static const auto extendable_vertex_compare = [](std::pair<std::pair<VertexID, uint32_t>, uint32_t> l, std::pair<std::pair<VertexID, uint32_t>, uint32_t> r) {
    if (l.first.second == 1 && r.first.second != 1) {
        return true;
    }
    else if (l.first.second != 1 && r.first.second == 1) {
        return false;
    }
    else
    {
        return l.second > r.second;
    }
};

typedef std::priority_queue<std::pair<std::pair<VertexID, uint32_t>, uint32_t>, std::vector<std::pair<std::pair<VertexID, uint32_t>, uint32_t>>,
        decltype(extendable_vertex_compare)> dpiso_min_pq;

void getCandidatesFor(VertexID u, CandidatesBuffer* buffer, const CandidateParam& canParam) {
    // VertexID previous_bn = bn[0];
    VertexID upre = buffer->prevNeb[u][0];
    Edges &e_pre = *canParam.edge_matrix[upre][u];
    uint32_t v_idx_pre = buffer->idx_embedding[upre];

    uint32_t pre_neb_cnt = 0;
    const uint32_t *pre_candidates = e_pre.getNeb(v_idx_pre, pre_neb_cnt);

    uint32_t valid_candidates_count = 0;
    for (uint32_t i = 0; i < pre_neb_cnt; ++i) {
        buffer->valid_candidate_idx[u][valid_candidates_count++] = pre_candidates[i];
    }

    uint32_t temp_count;
    for(int i=1; i<buffer->prevNeb[u].size() ;i++){
        upre = buffer->prevNeb[u][i];
        Edges &current_edge = *canParam.edge_matrix[upre][u];
        uint32_t vpre_id = buffer->idx_embedding[upre];

        uint32_t current_candidates_count = 0;
        const uint32_t *current_candidates = current_edge.getNeb(vpre_id, current_candidates_count);

        if constexpr (Profiler::useProfiling){
            Profiler::getInst().total_intersection++;
        }
        Intersection(current_candidates, current_candidates_count, buffer->valid_candidate_idx[u], valid_candidates_count, buffer->temp_buffer, temp_count);        
        // ComputeSetIntersection::ComputeCandidates(current_candidates, current_candidates_count, valid_candidate_index,
                                                //   valid_candidates_count,
                                                //   temp_buffer, temp_count);

        std::swap(buffer->temp_buffer, buffer->valid_candidate_idx[u]);
        valid_candidates_count = temp_count;
    }

    // idx_count[u] = valid_candidates_count;
    buffer->tot[u] = valid_candidates_count;
}

void updataExtendableV(const Query& query, CandidatesBuffer* buffer, const CandidateParam &can, VertexID cur_u, const FollowingNeb* followNeb, std::vector<dpiso_min_pq> &vec_rank_queue, uint32_t *extendable, uint32_t **weight_array){
    // std::vector<VertexID> vec;
    // std::vector<uint32_t> ext_vec;
    // for(int i=0; i<query.getVertexCnt(); i++) ext_vec.push_back(extendable[i]);
    // for(VertexID u: followNeb[cur_u]){
    //     if(extendable[u] - 1 == 0){
    //         vec.push_back(u);
    //     }
    // }
    // std::cout << fmt::format("u {}, extandle -1: {}, extanble v: {}, array: {}\n", cur_u, fmt::join(followNeb[cur_u], " "), fmt::join(vec, " "), fmt::join(ext_vec, " "));

    for(VertexID u: followNeb[cur_u]){
        extendable[u] -= 1;
        if (extendable[u] == 0) {
            getCandidatesFor(u, buffer, can);

            uint32_t weight = 0;
            for (uint32_t j = 0; j < buffer->tot[u]; ++j) {
                uint32_t idx = buffer->valid_candidate_idx[u][j];
                weight += weight_array[u][idx];
            }
            vec_rank_queue.back().emplace(std::make_pair(std::make_pair(u, query.getVertexDegree(u)), weight));
        }
    }
}

void updataExtendableV(const Query& query, CandidatesBuffer* buffer, const CandidateParam &can, VertexID cur_u, const FollowingNeb* followNeb, std::vector<dpiso_min_pq> &vec_rank_queue, uint32_t *extendable){
    // std::vector<VertexID> vec;
    // std::vector<uint32_t> ext_vec;
    // for(int i=0; i<query.getVertexCnt(); i++) ext_vec.push_back(extendable[i]);
    // for(VertexID u: followNeb[cur_u]){
    //     if(extendable[u] - 1 == 0){
    //         vec.push_back(u);
    //     }
    // }
    // std::cout << fmt::format("u {}, extandle -1: {}, extanble v: {}, array: {}\n", cur_u, fmt::join(followNeb[cur_u], " "), fmt::join(vec, " "), fmt::join(ext_vec, " "));

    for(VertexID u: followNeb[cur_u]){
        extendable[u] -= 1;
        if (extendable[u] == 0) {
            getCandidatesFor(u, buffer, can);
            vec_rank_queue.back().emplace(std::make_pair(std::make_pair(u, query.getVertexDegree(u)), buffer->tot[u]));
        }
    }
}

void restoreExtendableV(const FollowingNeb* followNeb, VertexID u, uint32_t *extendable) {
    // std::vector<uint32_t> ext_vec(extendable[0], extendable[query.getVertexCnt()]);
    // std::cout << fmt::format("u {}, extandle +1: {} \n", u, fmt::join(followNeb[u], " "));
    for (VertexID uf: followNeb[u]) {
        extendable[uf] += 1;
    }
}


void _explore_recur_DynamicOrder_DAF(const Graph& data, const Query& query, const Order& order, Order &match_order, const Config &cfg, CandidatesBuffer* buffer, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, std::vector<dpiso_min_pq>& vec_rank_queue, FollowingNeb *followNeb, uint32_t *extandble, uint32_t **weight_array, uint64_t output_limit){

    // VertexID u = order[cur_dep];
    VertexID u = match_order[cur_dep];

    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);

    // if(is_candidates_valid == false) return;
    if(buffer->tot[u] == 0) return;

    for(buffer->pos[u] = 0; buffer->pos[u]  < buffer->tot[u]; buffer->pos[u]++){
    // while(buffer->pos[cur_dep] < buffer->tot[cur_dep]){
        VertexID v;
        uint32_t v_idx;
        // buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);
        v_idx = buffer->valid_candidate_idx[u][buffer->pos[u]];
        v = cfg.can->candidates[u][v_idx];

        if(buffer->vis[v]){

            continue;
        }
        // if(cd->matchCheck(cur_dep, v_idx, buffer->vis, buffer->reverse_embeddings_deps)){
        //     continue;
        // }

        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            if(ans == output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
// #ifdef LOG_OUTPUT
//             spdlog::trace("Found Embeddings: [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
// #endif
        }
        else{
            buffer->embedding[u] = v;
            buffer->idx_embedding[u] = v_idx;
            buffer->vis[v] = true;
            buffer->reverse_embeddings_deps[v]=cur_dep;
            // failingPrune.ExtendIndex(cur_depth, u, v);

            vec_rank_queue.emplace_back(vec_rank_queue.back());
            updataExtendableV(query, buffer, *cfg.can, u, followNeb, vec_rank_queue, extandble, weight_array);

            VertexID next_u = vec_rank_queue.back().top().first.first;
            vec_rank_queue.back().pop();
            match_order[cur_dep + 1] = next_u;

            _explore_recur_DynamicOrder_DAF(data, query, order, match_order, cfg, buffer, cur_dep + 1, ans, stop_flag, vec_rank_queue, followNeb, extandble, weight_array, output_limit);
            if(stop_flag.load()) [[unlikely]]{
                return;
            }
            
            buffer->vis[v] = false;
            // buffer->reverse_embeddings_deps[v]=INVALID;
            vec_rank_queue.pop_back();
            restoreExtendableV(followNeb, u, extandble);

        }
    }
}

void explore_Intersection_DynamicOrder_DAF(const Graph& data, const Query& query, const Order& bfs_order, Order &match_order, const Config &cfg, uint32_t **weight_array, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, bfs_order, cfg);

    // ConflictDetection cd;
    // cd.Init(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = bfs_order[cur_depth];
    buffer.pos[u0] = 0;
    if(cfg.useFilter) buffer.tot[u0] = cfg.can->candidates_count[u0];
    else buffer.tot[u0] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[u0]; i++)
        buffer.valid_candidate_idx[u0][i] = i;

    uint32_t *extendable = new uint32_t[query.getVertexCnt()];
    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        extendable[i] = buffer.prevNeb[i].size();
    }

    FollowingNeb followNeb[query.getVertexCnt()];
    GetFollowingNeb(query, bfs_order, followNeb);

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    std::vector<dpiso_min_pq> vec_rank_queue;
    uint32_t cur_dep = 0;
    match_order.clear();
    match_order.reserve(query.getVertexCnt());
    match_order[cur_dep] = bfs_order[0];

    for(buffer.pos[u0] = 0; buffer.pos[u0]  < buffer.tot[u0]; buffer.pos[u0]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        // buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);
        v0_id = buffer.valid_candidate_idx[u0][buffer.pos[u0]];
        v0 = cfg.can->candidates[u0][v0_id];

        // if(cd.matchCheck(0, v0_id, buffer.vis, buffer.reverse_embeddings_deps)){
        //     continue;
        // }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        vec_rank_queue.emplace_back(dpiso_min_pq(extendable_vertex_compare));
        updataExtendableV(query, &buffer, *cfg.can, u0, followNeb, vec_rank_queue, extendable, weight_array);
        // updateExtendableVertex(idx_embedding, idx_count, valid_candidate_idx, edge_matrix, temp_buffer, weight_array,
        //                        tree, start_vertex, extendable,
        //                        vec_rank_queue, query_graph);

        VertexID next_u = vec_rank_queue.back().top().first.first;
        vec_rank_queue.back().pop();
        match_order[cur_dep + 1] = next_u;

        _explore_recur_DynamicOrder_DAF(data, query, bfs_order, match_order, cfg, &buffer, cur_depth + 1, ans, stop_flag, vec_rank_queue, followNeb, extendable, weight_array, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
        vec_rank_queue.pop_back();
        restoreExtendableV(followNeb, u0, extendable);
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}

void _explore_recur_DynamicOrder_VEQ(const Graph& data, const Query& query, const Order& order, Order &match_order, const Config &cfg, CandidatesBuffer* buffer, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, std::vector<dpiso_min_pq>& vec_rank_queue, FollowingNeb *followNeb, uint32_t *extandble, uint64_t output_limit){

    // VertexID u = order[cur_dep];
    VertexID u = match_order[cur_dep];

    // bool is_candidates_valid = buffer->getNextCandidates(cur_dep, data, order, cfg);

    // if(is_candidates_valid == false) return;
    if(buffer->tot[u] == 0) return;

    for(buffer->pos[u] = 0; buffer->pos[u]  < buffer->tot[u]; buffer->pos[u]++){
    // while(buffer->pos[cur_dep] < buffer->tot[cur_dep]){
        VertexID v;
        uint32_t v_idx;
        // buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);
        v_idx = buffer->valid_candidate_idx[u][buffer->pos[u]];
        v = cfg.can->candidates[u][v_idx];

        if(buffer->vis[v]){

            continue;
        }
        // if(cd->matchCheck(cur_dep, v_idx, buffer->vis, buffer->reverse_embeddings_deps)){
        //     continue;
        // }

        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            if(ans == output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
// #ifdef LOG_OUTPUT
//             spdlog::trace("Found Embeddings: [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
// #endif
        }
        else{
            buffer->embedding[u] = v;
            buffer->idx_embedding[u] = v_idx;
            buffer->vis[v] = true;
            buffer->reverse_embeddings_deps[v]=cur_dep;
            // failingPrune.ExtendIndex(cur_depth, u, v);

            vec_rank_queue.emplace_back(vec_rank_queue.back());
            updataExtendableV(query, buffer, *cfg.can, u, followNeb, vec_rank_queue, extandble);

            VertexID next_u = vec_rank_queue.back().top().first.first;
            vec_rank_queue.back().pop();
            match_order[cur_dep + 1] = next_u;

            _explore_recur_DynamicOrder_VEQ(data, query, order, match_order, cfg, buffer, cur_dep + 1, ans, stop_flag, vec_rank_queue, followNeb, extandble, output_limit);
            if(stop_flag.load()) [[unlikely]]{
                return;
            }
            
            buffer->vis[v] = false;
            // buffer->reverse_embeddings_deps[v]=INVALID;
            vec_rank_queue.pop_back();
            restoreExtendableV(followNeb, u, extandble);

        }
    }
}

void explore_Intersection_DynamicOrder_VEQ(const Graph& data, const Query& query, const Order& bfs_order, Order &match_order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, bfs_order, cfg);

    // ConflictDetection cd;
    // cd.Init(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = bfs_order[cur_depth];
    buffer.pos[u0] = 0;
    if(cfg.useFilter) buffer.tot[u0] = cfg.can->candidates_count[u0];
    else buffer.tot[u0] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[u0]; i++)
        buffer.valid_candidate_idx[u0][i] = i;

    uint32_t *extendable = new uint32_t[query.getVertexCnt()];
    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        extendable[i] = buffer.prevNeb[i].size();
    }

    FollowingNeb followNeb[query.getVertexCnt()];
    GetFollowingNeb(query, bfs_order, followNeb);

    struct timespec start, end;
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    std::vector<dpiso_min_pq> vec_rank_queue;
    uint32_t cur_dep = 0;
    match_order.clear();
    match_order.reserve(query.getVertexCnt());
    match_order[cur_dep] = bfs_order[0];

    for(buffer.pos[u0] = 0; buffer.pos[u0]  < buffer.tot[u0]; buffer.pos[u0]++){
    // while(buffer.pos[cur_depth] < buffer.tot[cur_depth]){
        VertexID v0;
        uint32_t v0_id;
        // buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);
        v0_id = buffer.valid_candidate_idx[u0][buffer.pos[u0]];
        v0 = cfg.can->candidates[u0][v0_id];

        // if(cd.matchCheck(0, v0_id, buffer.vis, buffer.reverse_embeddings_deps)){
        //     continue;
        // }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // failingPrune.ExtendIndex(cur_depth, u0, v0);

        vec_rank_queue.emplace_back(dpiso_min_pq(extendable_vertex_compare));
        updataExtendableV(query, &buffer, *cfg.can, u0, followNeb, vec_rank_queue, extendable);
        // updateExtendableVertex(idx_embedding, idx_count, valid_candidate_idx, edge_matrix, temp_buffer, weight_array,
        //                        tree, start_vertex, extendable,
        //                        vec_rank_queue, query_graph);

        VertexID next_u = vec_rank_queue.back().top().first.first;
        vec_rank_queue.back().pop();
        match_order[cur_dep + 1] = next_u;

        _explore_recur_DynamicOrder_VEQ(data, query, bfs_order, match_order, cfg, &buffer, cur_depth + 1, ans, stop_flag, vec_rank_queue, followNeb, extendable, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // buffer.pos[cur_depth]++;
        vec_rank_queue.pop_back();
        restoreExtendableV(followNeb, u0, extendable);
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    buffer.release();
}

size_t exploreDPisoStyle(const Graph &data, const Query &query, const Order &bfs_order, const Config &cfg,
                                        uint32_t **weight_array, uint32_t *order, size_t output_limit_num) {
    int max_depth = query.getVertexCnt();

    CandidatesBuffer buffer;
    buffer.allocate(data, query, bfs_order, cfg);

    uint32_t *extendable = new uint32_t[max_depth];
    for (uint32_t i = 0; i < max_depth; ++i) {
        extendable[i] = buffer.prevNeb[i].size();
    }

    FollowingNeb followNeb[max_depth];
    GetFollowingNeb(query, bfs_order, followNeb);

   // Evaluate the query.
    size_t embedding_cnt = 0;
    int cur_depth = 0;

    VertexID start_vertex = bfs_order[0];
    std::vector<dpiso_min_pq> vec_rank_queue;
    order[0] = start_vertex;

    for (uint32_t i = 0; i < cfg.can->candidates_count[start_vertex]; ++i) {
        VertexID v = cfg.can->candidates[start_vertex][i];
        buffer.embedding[start_vertex] = v;
        buffer.idx_embedding[start_vertex] = i;
        buffer.vis[v] = true;

        vec_rank_queue.emplace_back(dpiso_min_pq(extendable_vertex_compare));
        updataExtendableV(query, &buffer, *cfg.can, start_vertex, followNeb, vec_rank_queue, extendable, weight_array);

        VertexID u = vec_rank_queue.back().top().first.first;
        vec_rank_queue.back().pop();
        
        cur_depth += 1;
        order[cur_depth] = u;
        buffer.pos[u] = 0;
        while (cur_depth > 0) {
            while (buffer.pos[u] < buffer.tot[u]) {
                uint32_t valid_idx = buffer.valid_candidate_idx[u][buffer.pos[u]];
                v = cfg.can->candidates[u][valid_idx];

                if (buffer.vis[v]) {
                    buffer.pos[u] += 1;
                    continue;
                }
                buffer.embedding[u] = v;
                buffer.idx_embedding[u] = valid_idx;
                buffer.vis[v] = true;
                buffer.pos[u] += 1;

                if (cur_depth == max_depth - 1) {
                    embedding_cnt += 1;
                    buffer.vis[v] = false;

                    if (embedding_cnt >= output_limit_num) {
                        goto EXIT;
                    }
                } else {
                    cur_depth += 1;
                    vec_rank_queue.emplace_back(vec_rank_queue.back());
                    updataExtendableV(query, &buffer, *cfg.can, u, followNeb, vec_rank_queue, extendable, weight_array);

                    u = vec_rank_queue.back().top().first.first;
                    vec_rank_queue.back().pop();
                    buffer.pos[u] = 0;
                    order[cur_depth] = u;

                }
            }

            cur_depth -= 1;
            vec_rank_queue.pop_back();
            u = order[cur_depth];
            buffer.vis[buffer.embedding[u]] = false;
            restoreExtendableV(followNeb, u, extendable);

        }
    }

    // Release the buffer.
    EXIT:
    // releaseBuffer(max_depth, idx, idx_count, embedding, idx_embedding, temp_buffer, valid_candidate_idx,
    //               visited_vertices,
    //               bn, bn_count);
    buffer.release();

    return embedding_cnt;
}

void _explore_recur_VEQ(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, VEQAutomorphismPrune *ap, int cur_dep, size_t &ans, uint64_t output_limit){

    VertexID u = order[cur_dep];
    buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(buffer->tot[cur_dep] == 0){
        return;
    }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if(buffer->vis[v]){
            // continue;
            VertexID conflict_u = order[buffer->reverse_embeddings_deps[v]];
            ap->InjectiveConflict(cur_dep, v_idx, buffer->reverse_embeddings_deps[v], buffer->idx_embedding[conflict_u]);
            // goto _pruningUpdate;
            continue;
        }

        if(ap->EquivalencePruneCheck(cur_dep, u, v, v_idx, cfg.can->candidates, ans)){
            // std::cout << "pruning by equal \n";
            continue;
        }

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        ap->ExtendIndex(cur_dep, u, v, v_idx, buffer->embedding, buffer->idx_embedding, order);
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            ap->SuccessMatch(cur_dep, u, v_idx);
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;32mFound Embedding\x1b[0m : [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
#endif
            // buffer->embedding[u] = v;

        }
        else{
            _explore_recur_VEQ(data, query, order, cfg, buffer, ap, cur_dep + 1, ans, output_limit); 
            
            ap->Backtrack(cur_dep, u, v_idx);
        }

        buffer->vis[v] = false;
        ap->ReduceIndex(cur_dep, u, v_idx, cfg.can->candidates, cfg.can->candidates_count);
// _pruningUpdate:
        
    }

}


void explore_Intersection_Recursive_VEQ(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    VEQAutomorphismPrune ap;
    ap.Init(data, query, cfg, order);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        ap.ExtendIndex(cur_depth, u0, v0, v0_id, buffer.embedding, buffer.idx_embedding, order);
        _explore_recur_VEQ(data, query, order, cfg, &buffer, &ap, cur_depth + 1, ans, output_limit);
        buffer.vis[v0] = false;
        ap.Backtrack(cur_depth, u0, v0_id);
        ap.ReduceIndex(cur_depth, u0, v0_id, cfg.can->candidates, cfg.can->candidates_count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time = duration.count();

    buffer.release();
}


void _explore_recur_BICE_CD(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, BICEConflictDetection *cd, int cur_dep, size_t &ans, uint64_t output_limit){

    VertexID u = order[cur_dep];
    buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(buffer->tot[cur_dep] == 0){
        return;
    }

    cd->Q_PairBakeup(cur_dep);

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if(buffer->vis[v]){
            continue;
        }

        if(cd->IsConflict(u, v, v_idx, cfg.can->edge_matrix)){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("prune by conflict at u {}", u);
#endif  
            goto _pruningUpdate;
        }
#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {}, stack_size: {}", u, v, [&cd, &query](){
            std::vector<uint32_t> vec;
            for(int i=0; i<query.getVertexCnt(); i++) vec.push_back(cd->stack_size[i]);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;32mFound Embedding\x1b[0m : [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
#endif
            // buffer->embedding[u] = v;

        }
        else{
            _explore_recur_BICE_CD(data, query, order, cfg, buffer, cd, cur_dep + 1, ans, output_limit); 
        }

        buffer->vis[v] = false;
_pruningUpdate:
        cd->PruneUpdate(u, cur_dep);
#ifdef LOG_OUTPUT
        spdlog::trace("Reduced to u {} v {}, stack_size: {}", u, v, [&cd, &query](){
            std::vector<uint32_t> vec;
            for(int i=0; i<query.getVertexCnt(); i++) vec.push_back(cd->stack_size[i]);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif
    }

}

void explore_Intersection_Recursive_BICE_CD(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    BICEConflictDetection cd;
    cd.Init(data, query, order, cfg);
    cd.Q_PairBakeup(cur_depth);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        if(cd.IsConflict(u0, v0, v0_id, cfg.can->edge_matrix)){
            goto pruneUpdate;
        }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        _explore_recur_BICE_CD(data, query, order, cfg, &buffer, &cd, cur_depth + 1, ans, output_limit);
        buffer.vis[v0] = false;
pruneUpdate:
        cd.PruneUpdate(u0, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time = duration.count();

    buffer.release();
}

void _explore_recur_DAF_FP(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, DAFFailingPrune *fp, int cur_dep, size_t &ans, uint64_t output_limit){

    VertexID u = order[cur_dep];
    buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep)){
        return;
    }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if(buffer->vis[v]){
            // continue;
            fp->InjectiveConflict(cur_dep, buffer->reverse_embeddings_deps[v]);
            goto _PruneUpdate;
        }

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {}", u, v);
#endif

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
            fp->SuccessMatch(cur_dep, v);
#ifdef LOG_OUTPUT
            spdlog::trace("\x1b[1;32mFound Embedding\x1b[0m : [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
#endif
            // buffer->embedding[u] = v;

        }
        else{
            _explore_recur_DAF_FP(data, query, order, cfg, buffer, fp, cur_dep + 1, ans, output_limit); 
        }

        buffer->vis[v] = false;
_PruneUpdate:
        if(fp->PruneCheck(cur_dep))
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
    }

}

void explore_Intersection_Recursive_DAF_FP(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    DAFFailingPrune fp;
    fp.Init(query,order);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        _explore_recur_DAF_FP(data, query, order, cfg, &buffer, &fp, cur_depth + 1, ans, output_limit);
        buffer.vis[v0] = false;
// PruneUpdate:
        if(fp.PruneCheck(cur_depth))
            buffer.pos[cur_depth] = buffer.tot[cur_depth];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time = duration.count();

    buffer.release();
}

void _explore_recur_BICE_AP(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, BICEAutomorphismPrune *ap, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(buffer->tot[cur_dep] == 0){
        return;
    }

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if(buffer->vis[v]) goto _PruneUpdate_AP;

        #ifdef LOG_OUTPUT
            spdlog::trace("extend u {} v {}, {}", u, v, [&](){
                uint32_t cidx = ap->cell_index[cur_dep][v_idx];
                return fmt::format("vis[cell[{}]]:{}", cidx, ap->vis[cidx]);
            }());
        #endif

        if(ap->PruneCheck(cur_dep, v_idx)){
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().bice_ap.cnt++;
            }
            #ifdef LOG_OUTPUT
                spdlog::trace("extend u {} v {}, prune by BICE_AP", u, v);
            #endif
            goto _PruneUpdate_AP;
        }

        ap->ExtendIndex(cur_dep, v_idx, v);

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            // ans += 1;
            size_t res=0;
            ap->SuccessMatch(res);
            ans += res;
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
#ifdef LOG_OUTPUT
            if(res)
                spdlog::trace("\x1b[1;32mFound Embedding\x1b[0m : [{}], ans: {}", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "), ans);
#endif
            // buffer->embedding[u] = v;

        }
        else{
            _explore_recur_BICE_AP(data, query, order, cfg, buffer, ap, cur_dep + 1, ans, stop_flag, output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                return;
            }
        }

        buffer->vis[v] = false;
        ap->ReducedIndex(cur_dep, v_idx, v);
_PruneUpdate_AP:
        ap->PruneUpdate(cur_dep, v_idx, v);
        
    }

}

void explore_Intersection_Recursive_BICE_AP(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    BICEAutomorphismPrune ap;
    struct timespec pre_start, pre_end;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pre_start);
    ap.Init(data, query, cfg, order);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pre_end);
    if constexpr (Profiler::useProfiling){
        Profiler::getInst().prune_init_time = (pre_end.tv_sec - pre_start.tv_sec) * 1000.0 + (pre_end.tv_nsec - pre_start.tv_nsec) / 1e6;
    }
    
    struct timespec start, end;
    
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        if(buffer.vis[v0]) continue;

        if(ap.PruneCheck(cur_depth, v0_id)){
            if constexpr (Profiler::useProfiling){
                Profiler::getInst().bice_ap.cnt++;
            }
            goto PruneUpdate_AP;
        }

        ap.ExtendIndex(cur_depth, v0_id, v0);

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        
        _explore_recur_BICE_AP(data, query, order, cfg, &buffer, &ap, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        ap.ReducedIndex(cur_depth, v0_id, v0);
PruneUpdate_AP:
        ap.PruneUpdate(cur_depth, v0_id, v0);

        
    }
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;

    buffer.release();
}

template<
    bool Enable_DAFFailingPrune,
    bool Enable_GuPFailingPrune,
    bool Enable_GUPConflictDetection,
    bool Enable_BICEConflictDetection,
    bool Enable_VEQAutomorphismPrune
>
void _explore_recur_All(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, PruningOpt<Enable_DAFFailingPrune, Enable_GuPFailingPrune, Enable_GUPConflictDetection, Enable_BICEConflictDetection, Enable_VEQAutomorphismPrune>* prune, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    if constexpr (Enable_GuPFailingPrune){
        buffer->getNextCandidates(cur_dep, data, order, cfg, &prune->gup_fp);
    }
    else {
        buffer->getNextCandidates(cur_dep, data, order, cfg);
    }

    // fp->BoundingsBackUp(cur_dep, u);
    // if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep, u)) {
    //     return;
    // }
    if(prune->NoCandidatesPruneCheck(buffer->tot[cur_dep], cur_dep, u)){
#ifdef LOG_OUTPUT
        spdlog::trace("No candidates at depth {}, skip", cur_dep);
#endif   
        return;
    }

    prune->PruneStatusBackUp(cur_dep, u);

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} at depth {}, Embeddings {}", u, v, cur_dep, [&](){
            std::vector<VertexID> vec;
            for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
            vec.push_back(v);
            return fmt::format("{}", fmt::join(vec, ","));
        }());
#endif   
        if(buffer->vis[v]){
            // fp->InjectiveConflict(cur_dep, u, buffer->reverse_embeddings_deps[v], v_idx);
#ifdef LOG_OUTPUT
        spdlog::trace("access same vertex {} at depth {}, skip", v, cur_dep);
#endif   
            uint32_t conf_udep = buffer->reverse_embeddings_deps[v];
            uint32_t conf_u = order[conf_udep];
            uint32_t conf_vidx_udep = buffer->idx_embedding[conf_u];
            prune->InjectiveConflict(cur_dep, u, conf_udep, conf_u, v_idx, conf_vidx_udep);
            goto _pruningUpdate;
        }
        
        // if(fp->Nogood_V_Check(cur_dep, v_idx, v)){
        if(prune->ForwardPrune(cur_dep, u, v_idx, v, buffer->vis, buffer->reverse_embeddings_deps, cfg.can->candidates, cfg.can->edge_matrix, ans)){
// #ifdef ENABLE_SAMPLE
//             if constexpr (Profiler::useProfiling){
//                 if((ans & (Profiler::getInst().sample_inter - 1)) == 0){
//                     // printf("aaa\n");
//                     Profiler::getInst().timestamps.push_back(std::chrono::high_resolution_clock::now());
//                 }
//             }
// #endif
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
            goto _pruningUpdate;
        }

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        // fp->ExtendIndex(cur_dep, u, v);
        prune->ExtendIndex(cur_dep, u, v_idx, v, buffer->embedding, buffer->idx_embedding, order);
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            ans += 1;
#ifdef LOG_OUTPUT
            spdlog::trace("Found Embeddings: [{}]\n", [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, ","));
            }());
#endif
            // fp->SuccessMatch(cur_dep, u, v);
            prune->SuccessMatch(cur_dep, u, v_idx, v);
#ifdef ENABLE_SAMPLE
            if constexpr (Profiler::useProfiling){
                if((ans & (Profiler::getInst().sample_inter - 1)) == 0){
                    // printf("aaa\n");
                    Profiler::getInst().timestamps.push_back(std::chrono::high_resolution_clock::now());
                }
            }
#endif
#ifdef ENABLE_OUTPUT_CMP
            std::cout << fmt::format("{},{}\n", ans, [&](){
                std::vector<VertexID> vec;
                for(int i=0; i<cur_dep; i++) vec.push_back(buffer->embedding[order[i]]);
                vec.push_back(v);
                return fmt::format("{}", fmt::join(vec, " "));
            }());
#endif
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
            // buffer->embedding[u] = v;
        }
        else{
            _explore_recur_All(data, query, order, cfg, buffer, prune, cur_dep + 1, ans, stop_flag, output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                return;
            }
            // fp->BoundingsRecover(cur_dep, u);
            prune->BacktrackCleanUp(cur_dep, u, v_idx, v);
        }

        buffer->vis[v] = false;
        // fp->ReduceIndex(cur_dep, v);
        prune->ReduceIndex(cur_dep, u, v_idx, v, cfg.can->candidates, cfg.can->candidates_count);
_pruningUpdate:
        
// #ifdef LOG_OUTPUT
//         spdlog::trace("Reduced: depth {}, u {}, v {}, partial embed [{}]", 
//             cur_dep,u,v,
//             [buffer, &order, cur_dep](){
//                 std::vector<VertexID> v;
//                 for(int i=0; i<=cur_dep; i++) v.push_back(buffer->embedding[order[i]]);
//                 return fmt::format("{}", fmt::join(v, ","));
//             }()
//         );
// #endif
        // if(fp->Nogood_V_Update(cur_dep, v_idx, v, query.getKCoreValue(u)))
        if(prune->BacktrackingPrune(cur_dep, u, v_idx, v, query.getKCoreValue(u)))
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
    }

    // fp->FinalReturn(cur_dep);
    prune->FinalReturn(cur_dep);
}


template<
    bool Enable_DAFFailingPrune,
    bool Enable_GuPFailingPrune,
    bool Enable_GUPConflictDetection,
    bool Enable_BICEConflictDetection,
    bool Enable_VEQAutomorphismPrune
>
void explore_Intersection_All(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    PruningOpt<Enable_DAFFailingPrune, Enable_GuPFailingPrune, Enable_GUPConflictDetection, Enable_BICEConflictDetection, Enable_VEQAutomorphismPrune> prune;
    struct timespec pre_start, pre_end;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pre_start);
    prune.Init(data, query, order, cfg);
    // fp.BoundingsBackUp(cur_depth, u0);
    prune.PruneStatusBackUp(cur_depth, u0);

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pre_end);
    if constexpr (Profiler::useProfiling){
        Profiler::getInst().prune_init_time = (pre_end.tv_sec - pre_start.tv_sec) * 1000.0 + (pre_end.tv_nsec - pre_start.tv_nsec) / 1e6;
        Profiler::getInst().start_time = std::chrono::high_resolution_clock::now();
    }

    struct timespec start, end;
    
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        if(prune.ForwardPrune(cur_depth, u0, v0_id, v0, buffer.vis, buffer.reverse_embeddings_deps, cfg.can->candidates, cfg.can->edge_matrix ,ans)){
            if(ans >= output_limit) [[unlikely]] {
// #ifdef ENABLE_SAMPLE
//             if constexpr (Profiler::useProfiling){
//                 if((ans & (Profiler::getInst().sample_inter - 1)) == 0){
//                     // printf("aaa\n");
//                     Profiler::getInst().timestamps.push_back(std::chrono::high_resolution_clock::now());
//                 }
//             }
// #endif
                break;
            }
            goto pruneUpdate;
        }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        // fp.ExtendIndex(cur_depth, u0, v0);
        prune.ExtendIndex(cur_depth, u0, v0_id, v0, buffer.embedding, buffer.idx_embedding, order);
// #ifdef LOG_OUTPUT
//         spdlog::trace("Extend: depth {}, u {}, v {}, partial embed [{}]", 
//             cur_depth, u0, v0,
//             [&buffer, &order, cur_depth](){
//                 std::vector<VertexID> vec;
//                 for(int i=0; i<=cur_depth; i++) vec.push_back(buffer.embedding[order[i]]);
//                 return fmt::format("{}", fmt::join(vec, ","));
//             }()
//         );
// #endif
        _explore_recur_All(data, query, order, cfg, &buffer, &prune, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        // fp.BoundingsRecover(cur_depth, u0);
        prune.BacktrackCleanUp(cur_depth, u0, v0_id, v0);
        // fp.ReduceIndex(cur_depth, v0);
        prune.ReduceIndex(cur_depth, u0, v0_id, v0, cfg.can->candidates, cfg.can->candidates_count);
pruneUpdate:
        // if(fp.Nogood_V_Update(0, v0_id, v0, query.getKCoreValue(u0)))
        if(prune.BacktrackingPrune(cur_depth, u0, v0_id, v0, query.getKCoreValue(u0)))
            buffer.pos[0] = buffer.tot[0];
        
        // buffer.pos[cur_depth]++;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    // std::chrono::duration<double, std::milli> duration = end - start;
    // time = duration.count();
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
    
    buffer.release();
}


void _explore_recur_BICE(const Graph& data, const Query& query, const Order& order, const Config &cfg, CandidatesBuffer* buffer, BICEAutomorphismPrune *ap, DAFFailingPrune *fp, BICEConflictDetection *cd, int cur_dep, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit){

    VertexID u = order[cur_dep];
    buffer->getNextCandidates(cur_dep, data, order, cfg);

    if(buffer->tot[cur_dep] == 0){
        return;
    }
    // if(fp->NoCandidatesConflictCheck(buffer->tot[cur_dep], cur_dep)){
    //     return;
    // }
    cd->Q_PairBakeup(cur_dep);

    for(buffer->pos[cur_dep] = 0; buffer->pos[cur_dep]  < buffer->tot[cur_dep]; buffer->pos[cur_dep]++){
        VertexID v;
        uint32_t v_idx;
        buffer->getNextDataVertex(cur_dep, u, cfg, v, v_idx);

        if(buffer->vis[v]){
            // continue;
            goto _PruneUpdate_AP;
        }

        if(ap->PruneCheck(cur_dep, v_idx)){
            // continue;
#ifdef LOG_OUTPUT
            spdlog::trace("extend u {} v {} is skipped by cell {}", u, v, ap->cell_index[cur_dep][v_idx]);
#endif
            // fp->_fail_mask[cur_dep].set();
            // if (!fp->_fail_mask[cur_dep].test(cur_dep))
            //     fp->_fail_mask[cur_dep - 1] = fp->_fail_mask[cur_dep];
            // else
                fp->_fail_mask[cur_dep].set();
                fp->_fail_mask[cur_dep - 1] |= fp->_fail_mask[cur_dep];

            goto _PruneUpdate_AP;
        }

        ap->ExtendIndex(cur_dep, v_idx, v);

        if(cd->IsConflict(u, v, v_idx, cfg.can->edge_matrix, ap)){
            VertexID u_conf = 0;
            for(int i=0; i<query.getVertexCnt()-1; i++){
                if(cd->bp->q_pair[i+1] == cd->bp->NIL){
                    
                    u_conf = i;
                    break;
                }
            }
            std::vector<VertexID> conn_q;
            cd->bp->get_noninjective_connect(u_conf, cd->edges_stack, cd->edges_size_stack, cd->stack_size, cd->qv_depth, cd->qv_depth[u], ap, conn_q);
            for(auto uconn: conn_q){
                fp->_fail_mask[cur_dep] |= fp->_anscestor[cd->qv_depth[uconn]];
#ifdef LOG_OUTPUT
                spdlog::trace("accumulate u {} anc {} to failing mask, _fail_mask[{}]: {}", 
                    uconn, 
                    [&](){
                        std::vector<VertexID> vec;
                        for(int id=0; id<fp->_fail_mask.size(); id++) vec.push_back(fp->_anscestor[cd->qv_depth[uconn]][id]);
                        return fmt::format("{}", fmt::join(vec, ","));
                    }(),
                    cur_dep,
                    [&](){
                        std::vector<VertexID> vec;
                        for(int id=0; id<fp->_fail_mask.size(); id++) vec.push_back(fp->_fail_mask[cur_dep][id]);
                        return fmt::format("{}", fmt::join(vec, ","));
                    }()
                );
#endif
            }
            
            goto _PruneUpdate_CD;
        }
#ifdef LOG_OUTPUT
        spdlog::trace("extend u {} v {} cell {}", u, v, ap->cell_index[cur_dep][v_idx]);
#endif

        buffer->embedding[u] = v;
        buffer->idx_embedding[u] = v_idx;
        buffer->reverse_embeddings_deps[v] = cur_dep;
        buffer->vis[v] = true;
        
        
        if(cur_dep == query.getVertexCnt()-1) [[unlikely]] {
            // ans += 1;
            size_t res=0;
            ap->SuccessMatch(res);
            fp->SuccessMatch(cur_dep, v);
            ans += res;
#ifdef ENABLE_SAMPLE
            if constexpr (Profiler::useProfiling){
                if((ans & (Profiler::getInst().sample_inter - 1)) == 0){
                    // printf("aaa\n");
                    Profiler::getInst().timestamps.push_back(std::chrono::high_resolution_clock::now());
                }
            }
#endif
            if(ans >= output_limit) [[unlikely]] {
                stop_flag.store(true);
                return;
            }
#ifdef LOG_OUTPUT
            if(res)
                spdlog::trace("\x1b[1;32mFound Embedding\x1b[0m : [{}]", fmt::join(std::vector<VertexID>(buffer->embedding, buffer->embedding + query.getVertexCnt()), ", "));
#endif
            // buffer->embedding[u] = v;

        }
        else{
            _explore_recur_BICE(data, query, order, cfg, buffer, ap, fp, cd, cur_dep + 1, ans, stop_flag, output_limit); 
            if(stop_flag.load()) [[unlikely]]{
                return;
            }
        }

        buffer->vis[v] = false;
        ap->ReducedIndex(cur_dep, v_idx, v);
        if(fp->PruneCheck(cur_dep)){
            buffer->pos[cur_dep] = buffer->tot[cur_dep];
            // std::cout<<fmt::format("index: {}\n", fmt::join(ap->cell_index_by_depth[cur_dep], " "));
            for(auto index: ap->cell_index_by_depth[cur_dep]){
                ap->vis[index] = false;
            }
        }
_PruneUpdate_CD:
        cd->PruneUpdate(u, cur_dep);
_PruneUpdate_AP:
        ap->PruneUpdate(cur_dep, v_idx, v);
        
    }

}

void explore_Intersection_Recursive_BICE(const Graph& data, const Query& query, const Order& order, const Config &cfg, double &time, size_t &ans, std::atomic<bool>& stop_flag, uint64_t output_limit = -1){

    CandidatesBuffer buffer;
    buffer.allocate(data, query, order, cfg);

    int cur_depth = 0;

    VertexID u0 = order[cur_depth];
    buffer.pos[cur_depth] = 0;
    if(cfg.useFilter) buffer.tot[cur_depth] = cfg.can->candidates_count[u0];
    else buffer.tot[cur_depth] = data.getVertexCnt(); // useCache or don't use both
    
    for(int i=0; i<buffer.tot[cur_depth]; i++)
        buffer.valid_candidate_idx[cur_depth][i] = i;
    
    BICEAutomorphismPrune ap;
    DAFFailingPrune fp;
    BICEConflictDetection cd;
    struct timespec pre_start, pre_end;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pre_start);
    ap.Init(data, query, cfg, order);
    fp.Init(query, order);
    cd.Init(data, query, order, cfg);
    cd.Q_PairBakeup(cur_depth);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pre_end);
    
    struct timespec start, end;

    if constexpr (Profiler::useProfiling){
        Profiler::getInst().prune_init_time = (pre_end.tv_sec - pre_start.tv_sec) * 1000.0 + (pre_end.tv_nsec - pre_start.tv_nsec) / 1e6;
        Profiler::getInst().start_time = std::chrono::high_resolution_clock::now();
    }
    
    // auto start = std::chrono::high_resolution_clock::now();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
    for(buffer.pos[cur_depth] = 0; buffer.pos[cur_depth]  < buffer.tot[cur_depth]; buffer.pos[cur_depth]++){
        VertexID v0;
        uint32_t v0_id;
        buffer.getNextDataVertex(cur_depth, u0, cfg, v0, v0_id);

        if(buffer.vis[v0]) goto PruneUpdate_AP;

        if(ap.PruneCheck(cur_depth, v0_id)){
            // continue;
            // fp._fail_mask[cur_depth].set();
            goto PruneUpdate_AP;
        }

        ap.ExtendIndex(cur_depth, v0_id, v0);

        if(cd.IsConflict(u0, v0, v0_id, cfg.can->edge_matrix, &ap)){
            goto PruneUpdate_CD;
        }

        buffer.embedding[u0] = v0;
        buffer.idx_embedding[u0] = v0_id;
        buffer.vis[v0] = true;
        buffer.reverse_embeddings_deps[v0] = 0;
        
        _explore_recur_BICE(data, query, order, cfg, &buffer, &ap, &fp, &cd, cur_depth + 1, ans, stop_flag, output_limit);
        if(stop_flag.load()) [[unlikely]]{
            break;
        }
        buffer.vis[v0] = false;
        ap.ReducedIndex(cur_depth, v0_id, v0);
        // if(fp.PruneCheck(cur_depth)){
        //     buffer.pos[cur_depth] = buffer.tot[cur_depth];
        //     for(auto index: ap.cell_index_by_u[u0]){
        //         ap.vis[index] = false;
        //     }
        // }
PruneUpdate_CD:
        cd.PruneUpdate(u0, 0);
PruneUpdate_AP:
        ap.PruneUpdate(cur_depth, v0_id, v0);

        
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;
    // time = duration.count();
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;

    buffer.release();
}