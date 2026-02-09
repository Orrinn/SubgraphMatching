#pragma once
#include "common_type.h"
#include "graph.h"
#include "code_gen.h"
#include "vertexset.h"
#include <bitset>
#include <unordered_map>

class CandidateParam{
public:
    VertexID** candidates = nullptr;
    uint32_t* candidates_count = nullptr;
    Edges*** edge_matrix = nullptr;

    uint32_t _qv_cnt = 0;

    bool **candidates_exsist = nullptr;

    //only used for cell verification
    // std::vector<std::vector<uint32_t>> cell_index;
    // std::vector<std::vector<VertexID>> cell_set;

    // CandidateParam(VertexID **candidates, uint32_t *candidates_count, Edges ***edge_matrix):candidates(candidates), candidates_count(candidates_count), edge_matrix(edge_matrix){};
    CandidateParam(const Graph &data, const Query &query){

        _qv_cnt = query.getVertexCnt();
        uint32_t candidates_cnt = data.getMaxLabelFreq();

        candidates_count = new uint32_t[_qv_cnt];
        memset(candidates_count, 0, sizeof(uint32_t) * _qv_cnt);

        candidates = new uint32_t*[_qv_cnt];
        candidates_exsist = new bool*[_qv_cnt];

        for (int i = 0; i < _qv_cnt; ++i) {
            candidates[i] = new uint32_t[candidates_cnt];
            candidates_exsist[i] = new bool[data.getVertexCnt()];
            memset(candidates_exsist[i], 0, sizeof(bool) * data.getVertexCnt());
        }

        edge_matrix = new Edges **[_qv_cnt];
        for (int i = 0; i < _qv_cnt; ++i) {
            edge_matrix[i] = new Edges *[_qv_cnt];
            std::fill(edge_matrix[i], edge_matrix[i]+_qv_cnt, nullptr);
        }
    };

    // CandidateParam(const Graph &data, const Query &query, const Config& cfg, const Order &order){

    //     uint32_t _qv_cnt = query.getVertexCnt();
    //     uint32_t candidates_cnt = data.getMaxLabelFreq();

    //     candidates_count = new uint32_t[_qv_cnt];
    //     memset(candidates_count, 0, sizeof(uint32_t) * _qv_cnt);

    //     candidates = new uint32_t*[_qv_cnt];

    //     for (int i = 0; i < _qv_cnt; ++i) {
    //         candidates[i] = new uint32_t[candidates_cnt];
    //     }

    //     edge_matrix = new Edges **[_qv_cnt];
    //     for (int i = 0; i < _qv_cnt; ++i) {
    //         edge_matrix[i] = new Edges *[_qv_cnt];
    //     }

    //     ComputeStaticEqulCell(query, cfg, order, cell_index, cell_set);
    //     for (int u=0; u<_qv_cnt; u++){
    //         for(int vid=0; vid<cfg.can->candidates_count[u]; vid++){
    //             uint32_t cidx = cell_index[u][vid];
    //             candidates[u][candidates_count[u]++] = cidx;
    //         }
    //     }

        

        

    // };

    ~CandidateParam(){

#ifdef LOG_OUTPUT
        spdlog::trace("candidates deconstructed");
#endif
        if(candidates_exsist){
            for (int i = 0; i < _qv_cnt; ++i)
                delete[] candidates_exsist[i];
            delete[] candidates_exsist;
        }

        if(candidates){
            for (int i = 0; i < _qv_cnt; ++i)
                delete[] candidates[i];
            delete[] candidates;
        }

        if(edge_matrix){   
            for (int i = 0; i < _qv_cnt; ++i){
                for(int j=0; j<_qv_cnt;j++){        
                    if(edge_matrix[i][j]){
                        delete edge_matrix[i][j];
                        edge_matrix[i][j] = nullptr;
                    }
                }
                delete[] edge_matrix[i];
            }
            delete[] edge_matrix;
        }

        if(candidates_count) delete[] candidates_count;
    }

    // void ComputeStaticEqulCell(const Query &query, const Config& cfg, const Order &order, std::vector<std::vector<uint32_t>> &vec_index, std::vector<std::vector<VertexID>> &vec_set){
    //     std::vector<VertexID> tmp_vec;
    //     uint32_t vec_cnt = 0;
    //     uint32_t qvcnt = query.getVertexCnt();
    //     vec_index.resize(qvcnt);

    //     for(uint32_t dep=0; dep < qvcnt; dep++){
    //         VertexID u = order[dep];
    //         tmp_vec.resize(cfg.can->candidates_count[u]);
    //         vec_index[u].resize(cfg.can->candidates_count[u]);
    //         std::fill(vec_index[u].begin(), vec_index[u].end(), INVALID);
    //         uint32_t neb_cnt;
    //         const VertexID* nbrs = query.getNeb(u, neb_cnt);
    //         for(int v1_idx=0; v1_idx<cfg.can->candidates_count[u]; v1_idx++){
    //             if(vec_index[u][v1_idx] != INVALID) continue;
    //             tmp_vec.clear();
    //             tmp_vec.push_back(cfg.can->candidates[u][v1_idx]);
    //             vec_index[u][v1_idx] = vec_cnt++;
    //             for(int v2_idx=v1_idx + 1; v2_idx<cfg.can->candidates_count[u]; v2_idx++){
    //                 if(vec_index[u][v2_idx] != INVALID) continue;
    //                 bool isEqual = true;
    //                 for(int neb_idx=0; neb_idx < neb_cnt; neb_idx++){
    //                     VertexID u_neb = nbrs[neb_idx];
    //                     Edges* e = cfg.can->edge_matrix[u][u_neb];
    //                     uint32_t v1_neb_cnt, v2_neb_cnt;
    //                     const VertexID* v1_neb = e->getNeb(v1_idx, v1_neb_cnt);
    //                     const VertexID* v2_neb = e->getNeb(v2_idx, v2_neb_cnt);
    //                     if(v1_neb_cnt != v2_neb_cnt){
    //                         isEqual = false;
    //                         break;
    //                     }
    //                     for(int k=0; k<v1_neb_cnt; k++){
    //                         if(v1_neb[k] != v2_neb[k]){
    //                             isEqual = false;
    //                             break;
    //                         }
    //                     }
    //                     if(isEqual == false)
    //                         break;
    //                 }
    //                 if(isEqual == true){
    //                     tmp_vec.push_back(cfg.can->candidates[u][v2_idx]);
    //                     vec_index[u][v2_idx] = vec_index[u][v1_idx];
    //                 }
    //             }
    //             std::sort(tmp_vec.begin(), tmp_vec.end());
    //             vec_set.push_back(tmp_vec);
    //         }
    //     }
    // }
};


class ReuseParam{
public:
    Restriction *rests{nullptr};
    PlanIR *plan{nullptr};

    ReuseParam(Restriction *rests, PlanIR *plan):rests(rests), plan(plan){};
    ReuseParam()=default;
    ReuseParam(const Query& query, const Order &order){
        plan = new PlanIR(order, query);
    }

    ~ReuseParam(){
        if(plan) delete plan;
        if(rests) delete rests;
    }
};

class Config{
public:
    bool useFilter{false};
    std::string filterType{"gql"};
    std::string orderType{"gql"};
    
    bool useCache{false};

    bool usePrune{false};

    CandidateParam* can{nullptr};
    ReuseParam* reuse{nullptr};


    bool useDAFFailingPrune{false};
    bool useGuPFailingPrune{false};
    bool useGUPConflictDetection{false};
    bool useBICEConflictDetection{false};
    bool useVEQAutomorphismPrune{false};

    int gupConflictDetectionSize{3};

    // ~Config(){
    //     if(can) delete can;
    //     if(reuse) delete reuse;
    // }

    void Check() const{
        if(useFilter){
            if(can == nullptr){
                std::cout << "config error, use filter but canParam is nullprt\n";
                exit(-1);
            }
        }

        if(useCache){
            if(reuse == nullptr){
                std::cout << "config error, use cache but reuseParam is nullprt\n";
                exit(-1);
            }
        }
    }

};
