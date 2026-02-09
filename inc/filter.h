#pragma once

#include "graph.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <queue>
#include "table.h"
#include "common_ops.h"
#include "api.h"
#include "spectral.h"

using namespace std;

void sortCandidates(uint32_t **candidates, uint32_t *candidates_count, uint32_t num) {
    for (int i = 0; i < num; ++i) {
        std::sort(candidates[i], candidates[i] + candidates_count[i]);
    }
}

void NLFByVertex(const Graph& data, const Query& query, VertexID query_id, uint32_t *buffer, uint32_t& cnt){
    LabelID qLabel = query.getVertexLabel(query_id);
    int qDegree = query.getVertexDegree(query_id);
    
    const unordered_map<LabelID, uint32_t>* q_nlf = query.getVertexNLF(query_id);

    uint32_t tot_num;
    const VertexID* data_v_array = data.getVertexByLabel(qLabel, tot_num);

    cnt=0;
    for(int i=0; i<tot_num; i++){
        VertexID data_v = data_v_array[i];
        

        if(data.getVertexDegree(data_v) >= qDegree){
            const unordered_map<LabelID, uint32_t>* d_nlf = data.getVertexNLF(data_v);
            int is_valid = true;

            for(auto ele: *q_nlf){
                auto iter = d_nlf->find(ele.first);
                if (iter == d_nlf->end() || iter->second < ele.second) {
                    is_valid = false;
                    break;
                }
            }

            if(is_valid){
                if(buffer)
                    buffer[cnt++] = data_v;
                else
                    cnt++;
            }
        }
    }
}

void FilterParamConstruct(const Graph& data, const Query& query, CandidateParam &canParam){

    if(canParam.candidates == nullptr || canParam.candidates_count == nullptr){
        printf("error: filter candidates first, then evoke this function to construct candidates param and index;\n");
        exit(-1);
    }
    
    sortCandidates(canParam.candidates, canParam.candidates_count, query.getVertexCnt());

    buildTables(&data, &query, canParam.candidates, canParam.candidates_count, canParam.edge_matrix, nullptr);

    for(int i=0; i<query.getVertexCnt(); i++){
        memset(canParam.candidates_exsist[i], 0, sizeof(bool) * data.getVertexCnt());
        for(int j=0; j<canParam.candidates_count[i]; j++){
            VertexID v = canParam.candidates[i][j];
            canParam.candidates_exsist[i][v] = true;
        }
    }
}

bool _NLF_Filter(const Graph& data, const Query& query, uint32_t **&candidates, uint32_t *&candidates_cnt){
    // allocateBuffer(data, query, candidates, candidates_cnt);

    uint32_t qv_cnt = query.getVertexCnt();
    for(VertexID i=0; i<qv_cnt; i++){
        // uint32_t cnt;
        NLFByVertex(data, query, i, candidates[i], candidates_cnt[i]);
        if(candidates_cnt[i] == 0){
            printf("error on NLF filter, can't find candidates of %d\n", i);
            return false;
        }
    }
    return true;
}

void old_cheap(int* col_ptrs, int* col_ids, int* match, int* row_match, int n, int m) {
    int ptr;
    int i = 0;
    for(; i < n; i++) {
        int s_ptr = col_ptrs[i];
        int e_ptr = col_ptrs[i + 1];
        for(ptr = s_ptr; ptr < e_ptr; ptr++) {
            int r_id = col_ids[ptr];
            if(row_match[r_id] == -1) {
                match[i] = r_id;
                row_match[r_id] = i;
                break;
            }
        }
    }
}

void match_bfs(int* col_ptrs, int* col_ids, int* match, int* row_match, int* visited,
                        int* queue, int* previous, int n, int m) {
    int queue_ptr, queue_col, ptr, next_augment_no, i, j, queue_size,
            row, col, temp, eptr;

    old_cheap(col_ptrs, col_ids, match, row_match, n, m);

    memset(visited, 0, sizeof(int) * m);

    next_augment_no = 1;
    for(i = 0; i < n; i++) {
        if(match[i] == -1 && col_ptrs[i] != col_ptrs[i+1]) {
            queue[0] = i; queue_ptr = 0; queue_size = 1;

            while(queue_size > queue_ptr) {
                queue_col = queue[queue_ptr++];
                eptr = col_ptrs[queue_col + 1];
                for(ptr = col_ptrs[queue_col]; ptr < eptr; ptr++) {
                    row = col_ids[ptr];
                    temp = visited[row];

                    if(temp != next_augment_no && temp != -1) {
                        previous[row] = queue_col;
                        visited[row] = next_augment_no;

                        col = row_match[row];

                        if(col == -1) {
                            // Find an augmenting path. Then, trace back and modify the augmenting path.
                            while(row != -1) {
                                col = previous[row];
                                temp = match[col];
                                match[col] = row;
                                row_match[row] = col;
                                row = temp;
                            }
                            next_augment_no++;
                            queue_size = 0;
                            break;
                        } else {
                            // Continue to construct the match.
                            queue[queue_size++] = col;
                        }
                    }
                }
            }

            if(match[i] == -1) {
                for(j = 1; j < queue_size; j++) {
                    visited[match[queue[j]]] = -1;
                }
            }
        }
    }
}

bool verifyExactTwigIso(const Graph &data_graph, const Graph &query_graph, uint32_t data_vertex, uint32_t query_vertex,
                                   bool **valid_candidates, int *left_to_right_offset, int *left_to_right_edges,
                                   int *left_to_right_match, int *right_to_left_match, int* match_visited,
                                   int* match_queue, int* match_previous) {
    // Construct the bipartite graph between N(query_vertex) and N(data_vertex)
    uint32_t left_partition_size;
    uint32_t right_partition_size;
    const VertexID* query_vertex_neighbors = query_graph.getNeb(query_vertex, left_partition_size);
    const VertexID* data_vertex_neighbors = data_graph.getNeb(data_vertex, right_partition_size);

    uint32_t edge_count = 0;
    for (int i = 0; i < left_partition_size; ++i) {
        VertexID query_vertex_neighbor = query_vertex_neighbors[i];
        left_to_right_offset[i] = edge_count;

        for (int j = 0; j < right_partition_size; ++j) {
            VertexID data_vertex_neighbor = data_vertex_neighbors[j];

            if (valid_candidates[query_vertex_neighbor][data_vertex_neighbor]) {
                left_to_right_edges[edge_count++] = j;
            }
        }
    }
    left_to_right_offset[left_partition_size] = edge_count;

    memset(left_to_right_match, -1, left_partition_size * sizeof(int));
    memset(right_to_left_match, -1, right_partition_size * sizeof(int));

    match_bfs(left_to_right_offset, left_to_right_edges, left_to_right_match, right_to_left_match,
                               match_visited, match_queue, match_previous, left_partition_size, right_partition_size);
    for (int i = 0; i < left_partition_size; ++i) {
        if (left_to_right_match[i] == -1)
            return false;
    }

    return true;
}

void compactCandidates(uint32_t **&candidates, uint32_t *&candidates_count, uint32_t query_vertex_num) {
    for (uint32_t i = 0; i < query_vertex_num; ++i) {
        VertexID query_vertex = i;
        uint32_t next_position = 0;
        for (uint32_t j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];

            if (data_vertex != INVALID) {
                candidates[query_vertex][next_position++] = data_vertex;
            }
        }

        candidates_count[query_vertex] = next_position;
    }
}

bool isCandidateSetValid(uint32_t **&candidates, uint32_t *&candidates_count, uint32_t query_vertex_num) {
    for (uint32_t i = 0; i < query_vertex_num; ++i) {
        if (candidates_count[i] == 0){
            printf("candidates filter error, there is no candidates for %d, exit\n", i);
            return false;
        }
    }
    return true;
}


bool _GQL_Filter(const Graph& data, const Query &query, CandidateParam& canParam) {

    uint32_t **&candidates = canParam.candidates;
    uint32_t *&candidates_count = canParam.candidates_count;

    // Local refinement.
    if(_NLF_Filter(data, query, candidates, candidates_count) == false)
        return false;

    // Allocate buffer.
    uint32_t qvcnt = query.getVertexCnt();
    uint32_t dvcnt = data.getVertexCnt();

    bool** valid_candidates = new bool*[qvcnt];
    for (uint32_t i = 0; i < qvcnt; ++i) {
        valid_candidates[i] = new bool[dvcnt];
        memset(valid_candidates[i], 0, sizeof(bool) * dvcnt);
    }

    uint32_t query_graph_max_degree = query.getMaxDgree();
    uint32_t data_max_degree = data.getMaxDgree();

    int* left_to_right_offset = new int[query_graph_max_degree + 1];
    int* left_to_right_edges = new int[query_graph_max_degree * data_max_degree];
    int* left_to_right_match = new int[query_graph_max_degree];
    int* right_to_left_match = new int[data_max_degree];
    int* match_visited = new int[data_max_degree + 1];
    int* match_queue = new int[qvcnt];
    int* match_previous = new int[data_max_degree + 1];

    // Record valid candidate vertices for each query vertex.
    for (uint32_t i = 0; i < qvcnt; ++i) {
        VertexID query_vertex = i;
        for (uint32_t j = 0; j < candidates_count[query_vertex]; ++j) {
            VertexID data_vertex = candidates[query_vertex][j];
            valid_candidates[query_vertex][data_vertex] = true;
        }
    }

    // Global refinement.
    for (uint32_t l = 0; l < 2; ++l) {
        for (uint32_t i = 0; i < qvcnt; ++i) {
            VertexID query_vertex = i;
            for (uint32_t j = 0; j < candidates_count[query_vertex]; ++j) {
                VertexID data_vertex = candidates[query_vertex][j];

                if (data_vertex == INVALID)
                    continue;

                if (!verifyExactTwigIso(data, query, data_vertex, query_vertex, valid_candidates,
                                        left_to_right_offset, left_to_right_edges, left_to_right_match,
                                        right_to_left_match, match_visited, match_queue, match_previous)) {
                    candidates[query_vertex][j] = INVALID;
                    valid_candidates[query_vertex][data_vertex] = false;
                }
            }
        }
    }

    // Compact candidates.
    compactCandidates(candidates, candidates_count, qvcnt);

    // Release memory.
    for (uint32_t i = 0; i < qvcnt; ++i) {
        delete[] valid_candidates[i];
    }
    delete[] valid_candidates;
    delete[] left_to_right_offset;
    delete[] left_to_right_edges;
    delete[] left_to_right_match;
    delete[] right_to_left_match;
    delete[] match_visited;
    delete[] match_queue;
    delete[] match_previous;


    if(!isCandidateSetValid(candidates, candidates_count, qvcnt)){
        return false;
    }
    return true;
}



void generateCandidates(const Graph &data, const Graph &query, VertexID query_vertex,
                                       const std::vector<VertexID> &pivot_vertices, VertexID **candidates,
                                       uint32_t *candidates_count, uint32_t *flag, uint32_t *updated_flag) {
    LabelID query_vertex_label = query.getVertexLabel(query_vertex);
    uint32_t query_vertex_degree = query.getVertexDegree(query_vertex);

    const std::unordered_map<LabelID , uint32_t>* query_vertex_nlf = query.getVertexNLF(query_vertex);

    uint32_t count = 0;
    uint32_t updated_flag_count = 0;
    for (auto pivot_vertex: pivot_vertices) {

        for (uint32_t j = 0; j < candidates_count[pivot_vertex]; ++j) {
            VertexID v = candidates[pivot_vertex][j];

            if (v == INVALID)
                continue;
            uint32_t v_nbrs_count;
            const VertexID* v_nbrs = data.getNeb(v, v_nbrs_count);

            for (uint32_t k = 0; k < v_nbrs_count; ++k) {
                VertexID v_nbr = v_nbrs[k];
                LabelID v_nbr_label = data.getVertexLabel(v_nbr);
                uint32_t v_nbr_degree = data.getVertexDegree(v_nbr);

                if (flag[v_nbr] == count && v_nbr_label == query_vertex_label && v_nbr_degree >= query_vertex_degree) {
                    flag[v_nbr] += 1;

                    if (count == 0) {
                        updated_flag[updated_flag_count++] = v_nbr;
                    }
                }
            }
        }

        count += 1;
    }

    for (uint32_t i = 0; i < updated_flag_count; ++i) {
        VertexID v = updated_flag[i];
        if (flag[v] == count) {
            // NLF filter.
            const std::unordered_map<LabelID, uint32_t>* data_vertex_nlf = data.getVertexNLF(v);

            if (data_vertex_nlf->size() >= query_vertex_nlf->size()) {
                bool is_valid = true;

                for (auto element : *query_vertex_nlf) {
                    auto iter = data_vertex_nlf->find(element.first);
                    if (iter == data_vertex_nlf->end() || iter->second < element.second) {
                        is_valid = false;
                        break;
                    }
                }

                if (is_valid) {
                    candidates[query_vertex][candidates_count[query_vertex]++] = v;
                }
            }

        }
    }

    for (uint32_t i = 0; i < updated_flag_count; ++i) {
        uint32_t v = updated_flag[i];
        flag[v] = 0;
    }
}

void pruneCandidates(const Graph &data, const Query &query, VertexID query_vertex,
                                    const std::vector<VertexID> &pivot_vertices,  VertexID **candidates,
                                    uint32_t *candidates_count, uint32_t *flag, uint32_t *updated_flag) {
    LabelID query_vertex_label = query.getVertexLabel(query_vertex);
    uint32_t query_vertex_degree = query.getVertexDegree(query_vertex);

    uint32_t count = 0;
    uint32_t updated_flag_count = 0;
    for (auto pivot_vertex: pivot_vertices) {

        for (uint32_t j = 0; j < candidates_count[pivot_vertex]; ++j) {
            VertexID v = candidates[pivot_vertex][j];

            if (v == INVALID)
                continue;
            uint32_t v_nbrs_count;
            const VertexID* v_nbrs = data.getNeb(v, v_nbrs_count);

            for (uint32_t k = 0; k < v_nbrs_count; ++k) {
                VertexID v_nbr = v_nbrs[k];
                LabelID v_nbr_label = data.getVertexLabel(v_nbr);
                uint32_t v_nbr_degree = data.getVertexDegree(v_nbr);

                if (flag[v_nbr] == count && v_nbr_label == query_vertex_label && v_nbr_degree >= query_vertex_degree) {
                    flag[v_nbr] += 1;

                    if (count == 0) {
                        updated_flag[updated_flag_count++] = v_nbr;
                    }
                }
            }
        }

        count += 1;
    }

    for (uint32_t i = 0; i < candidates_count[query_vertex]; ++i) {
        uint32_t v = candidates[query_vertex][i];
        if (v == INVALID)
            continue;

        if (flag[v] != count) {
            candidates[query_vertex][i] = INVALID;
        }
    }

    for (uint32_t i = 0; i < updated_flag_count; ++i) {
        uint32_t v = updated_flag[i];
        flag[v] = 0;
    }
}


VertexID CFLStartVertex(const Graph& data, const Query& query) {
    auto rank_compare = [](std::pair<VertexID, double> l, std::pair<VertexID, double> r) {
        return l.second < r.second;
    };

    std::priority_queue<std::pair<VertexID, double>, std::vector<std::pair<VertexID, double>>, decltype(rank_compare)> rank_queue(rank_compare);

    // Compute the ranking.
    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        VertexID query_vertex = i;

        if (query.getKCoreLength() == 0 || query.getKCoreValue(query_vertex) > 1) {
            LabelID label = query.getVertexLabel(query_vertex);
            uint32_t degree = query.getVertexDegree(query_vertex);
            uint32_t frequency = data.getLabelFreq(label);
            double rank = frequency / (double) degree;
            rank_queue.push(std::make_pair(query_vertex, rank));
        }
    }

    // Keep the top-3.
    while (rank_queue.size() > 3) {
        rank_queue.pop();
    }
    
    VertexID start_vertex = 0;
    double min_score = data.getMaxLabelFreq() + 1;

    while (!rank_queue.empty()) {
        VertexID query_vertex = rank_queue.top().first;
        uint32_t count;
        NLFByVertex(data, query, query_vertex, nullptr, count);
        double cur_score = count / (double) query.getVertexDegree(query_vertex);

        // std::cout << "qv "<< query_vertex <<", cnt "<< count <<", cur_score " << cur_score << "\n";

        if (cur_score < min_score) {
            start_vertex = query_vertex;
            min_score = cur_score;
        }
        rank_queue.pop();
    }

    return start_vertex;
}


bool _CFL_Filter(const Graph &data, const Query &query, CandidateParam& canParam){
    
    TreeNode *tree = nullptr;
    VertexID **&candidates = canParam.candidates;
    uint32_t *&candidates_count = canParam.candidates_count;
    
    // allocateBuffer(data, query, candidates, candidates_count);

    VertexID startVertex = CFLStartVertex(data, query);

    Order construction_order;

    BFS(query, startVertex, tree, construction_order);

    // refinement vertices generation
    PreviousNeb *prevNeb = new PreviousNeb[query.getVertexCnt()];
    FollowingNeb *followNeb = new FollowingNeb[query.getVertexCnt()];
    
    GetPreviousNeb(query, construction_order, prevNeb);
    GetFollowingNeb(query, construction_order, followNeb);

    std::vector<VertexID> *fnebSameLevel, *fnebDiffLevel;
    fnebSameLevel = new std::vector<VertexID>[query.getVertexCnt()];
    fnebDiffLevel = new std::vector<VertexID>[query.getVertexCnt()];

    int _maxLevel = 0;
    for(int v=0; v<query.getVertexCnt(); v++){
        _maxLevel = (tree[v]._level > _maxLevel ? tree[v]._level : _maxLevel);

        fnebSameLevel[v].clear();
        fnebDiffLevel[v].clear();

        for(auto fn: followNeb[v]){
            if(tree[fn]._level == tree[v]._level)
                fnebSameLevel[v].push_back(fn);
            else if(tree[fn]._level > tree[v]._level)
                fnebDiffLevel[v].push_back(fn);
        }
    }

    int maxLevel = _maxLevel + 1;

    std::vector<std::vector<VertexID>> v_at_level(maxLevel);
    for(int l=0; l<maxLevel; l++)
        v_at_level[l].clear();

    for(auto v: construction_order)
        v_at_level[tree[v]._level].push_back(v);

    VertexID start_vertex = construction_order[0];
    NLFByVertex(data, query, start_vertex, candidates[start_vertex], candidates_count[start_vertex]);

    uint32_t* updated_flag = new uint32_t[data.getVertexCnt()];
    uint32_t* flag = new uint32_t[data.getVertexCnt()];
    std::fill(flag, flag + data.getVertexCnt(), 0);

    // Top-down generation.
    for (int i = 1; i < maxLevel; ++i) {
        // Forward generation.
        for (auto query_vertex: v_at_level[i]) {
            generateCandidates(data, query, query_vertex, prevNeb[query_vertex], candidates, candidates_count, flag, updated_flag);
        }

        // Backward prune.
        for (auto rit = v_at_level[i].rbegin(); rit != v_at_level[i].rend(); rit++) {
            VertexID query_vertex = *rit;
            // TreeNode& node = tree[query_vertex];

            if (fnebSameLevel[query_vertex].size() > 0) {
                pruneCandidates(data, query, query_vertex, fnebSameLevel[query_vertex], candidates, candidates_count, flag, updated_flag);
            }
        }
    }

    // Bottom-up refinement.
    for (int i = maxLevel - 2; i >= 0; --i) {
        for (auto query_vertex: v_at_level[i]) {
            if (fnebDiffLevel[query_vertex].size() > 0) {
                pruneCandidates(data, query, query_vertex, fnebDiffLevel[query_vertex], candidates, candidates_count, flag, updated_flag);
            }
        }
    }


    compactCandidates(candidates, candidates_count, query.getVertexCnt());

    delete[] updated_flag;
    delete[] flag;
    delete[] prevNeb;
    delete[] followNeb;
    delete[] tree;
    delete[] fnebDiffLevel;
    delete[] fnebSameLevel;
    if(!isCandidateSetValid(candidates, candidates_count, query.getVertexCnt())){
        printf("CFL filter invalid, exit\n");
        return false;
    }
    return true;
}

VertexID CECIStartVertex(const Graph &data, const Query &query) {
    double min_score = data.getVertexCnt();
    VertexID start_vertex = 0;

    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        uint32_t degree = query.getVertexDegree(i);
        uint32_t count = 0;
        NLFByVertex(data, query, i, nullptr, count);
        double cur_score = count / (double)degree;
        if (cur_score < min_score) {
            min_score = cur_score;
            start_vertex = i;
        }
    }

    return start_vertex;
}

bool _CECI_Filter(const Graph &data, const Query &query, CandidateParam &canParam) {

    uint32_t **&candidates = canParam.candidates;
    uint32_t *&candidates_count = canParam.candidates_count; 
    Order construction_order;
    TreeNode *tree = nullptr;  
    
    std::vector<std::unordered_map<VertexID, std::vector<VertexID >>> TE_Candidates;
    std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> NTE_Candidates;

    VertexID startVertex = CECIStartVertex(data, query);

    BFS(query, startVertex, tree, construction_order);

    PreviousNeb *prevNeb = new PreviousNeb[query.getVertexCnt()];
    FollowingNeb *followNeb = new FollowingNeb[query.getVertexCnt()];

    GetPreviousNeb(query, construction_order, prevNeb);
    GetFollowingNeb(query, construction_order, followNeb);

    // allocateBuffer(data, query, candidates, candidates_count);

    uint32_t query_vertices_count = query.getVertexCnt();
    uint32_t data_vertices_count = data.getVertexCnt();
    // Find the pivots.
    VertexID root = construction_order[0];
    NLFByVertex(data, query, root, candidates[root], candidates_count[root]);

    if (candidates_count[root] == 0){
        std::cout << "CECI finds candidates = 0, exit;\n";
        return false;
    }

    // TE_Candidates construction and filtering.
    std::vector<uint32_t> updated_flag(data_vertices_count);
    std::vector<uint32_t> flag(data_vertices_count);
    std::fill(flag.begin(), flag.end(), 0);
    std::vector<bool> visited_query_vertex(query_vertices_count);
    std::fill(visited_query_vertex.begin(), visited_query_vertex.end(), false);

    visited_query_vertex[root] = true;

    TE_Candidates.resize(query_vertices_count);

    for (uint32_t i = 1; i < query_vertices_count; ++i) {
        VertexID u = construction_order[i];
        // TreeNode& u_node = tree[u];
        VertexID u_p = tree[u]._parent;

        uint32_t u_label = query.getVertexLabel(u);
        uint32_t u_degree = query.getVertexDegree(u);
        const std::unordered_map<LabelID, uint32_t>* u_nlf = query.getVertexNLF(u);
        candidates_count[u] = 0;

        visited_query_vertex[u] = true;
        VertexID* frontiers = candidates[u_p];
        uint32_t frontiers_count = candidates_count[u_p];

        for (uint32_t j = 0; j < frontiers_count; ++j) {
            VertexID v_f = frontiers[j];

            if (v_f == INVALID)
                continue;

            uint32_t nbrs_cnt;
            const VertexID* nbrs = data.getNeb(v_f, nbrs_cnt);

            auto iter_pair = TE_Candidates[u].emplace(v_f, std::vector<VertexID>());
            for (uint32_t k = 0; k < nbrs_cnt; ++k) {
                VertexID v = nbrs[k];

                if (data.getVertexLabel(v) == u_label && data.getVertexDegree(v) >= u_degree) {

                    // NLF check
                    const std::unordered_map<LabelID, uint32_t>* v_nlf = data.getVertexNLF(v);

                    if (v_nlf->size() >= u_nlf->size()) {
                        bool is_valid = true;

                        for (auto element : *u_nlf) {
                            auto iter = v_nlf->find(element.first);
                            if (iter == v_nlf->end() || iter->second < element.second) {
                                is_valid = false;
                                break;
                            }
                        }

                        if (is_valid) {
                            iter_pair.first->second.push_back(v);
                            if (flag[v] == 0) {
                                flag[v] = 1;
                                candidates[u][candidates_count[u]++] = v;
                            }
                        }
                    }
                }
            }

            if (iter_pair.first->second.empty()) {
                frontiers[j] = INVALID;
                for (VertexID u_c : tree[u_p]._children) {
                    if (visited_query_vertex[u_c]) {
                        TE_Candidates[u_c].erase(v_f);
                    }
                }
            }
        }

        if (candidates_count[u] == 0){
            std::cout << "CECI finds candidates = 0, exit;\n";
            exit(-1);
            return false;
        }
            

        for (uint32_t j = 0; j < candidates_count[u]; ++j) {
            VertexID v = candidates[u][j];
            flag[v] = 0;
        }
    }

    // NTE_Candidates construction and filtering.
    NTE_Candidates.resize(query_vertices_count);
    for (auto& value : NTE_Candidates) {
        value.resize(query_vertices_count);
    }

    for (uint32_t i = 1; i < query_vertices_count; ++i) {
        VertexID u = construction_order[i];

        uint32_t u_label = query.getVertexLabel(u);
        uint32_t u_degree = query.getVertexDegree(u);

        const std::unordered_map<LabelID, uint32_t> *u_nlf = query.getVertexNLF(u);

        for (VertexID u_p: prevNeb[u]) {
            
            VertexID *frontiers = candidates[u_p];
            uint32_t frontiers_count = candidates_count[u_p];

            for (uint32_t j = 0; j < frontiers_count; ++j) {
                VertexID v_f = frontiers[j];

                if (v_f == INVALID)
                    continue;

                uint32_t nbrs_cnt;
                const VertexID *nbrs = data.getNeb(v_f, nbrs_cnt);

                auto iter_pair = NTE_Candidates[u][u_p].emplace(v_f, std::vector<VertexID>());
                for (uint32_t k = 0; k < nbrs_cnt; ++k) {
                    VertexID v = nbrs[k];

                    if (data.getVertexLabel(v) == u_label && data.getVertexDegree(v) >= u_degree) {

                        // NLF check
                        const std::unordered_map<LabelID, uint32_t> *v_nlf = data.getVertexNLF(v);

                        if (v_nlf->size() >= u_nlf->size()) {
                            bool is_valid = true;

                            for (auto element : *u_nlf) {
                                auto iter = v_nlf->find(element.first);
                                if (iter == v_nlf->end() || iter->second < element.second) {
                                    is_valid = false;
                                    break;
                                }
                            }

                            if (is_valid) {
                                iter_pair.first->second.push_back(v);
                            }
                        }
                    }
                }

                if (iter_pair.first->second.empty()) {
                    frontiers[j] = INVALID;
                    for (VertexID u_c : tree[u_p]._children) {
                        TE_Candidates[u_c].erase(v_f);
                    }
                }
            }
        }
    }

    // Reverse BFS refine.
    std::vector<std::vector<uint32_t>> cardinality(query_vertices_count);
    for (uint32_t i = 0; i < query_vertices_count; ++i) {
        cardinality[i].resize(candidates_count[i], 1);
    }

    std::vector<uint32_t> local_cardinality(data_vertices_count);
    std::fill(local_cardinality.begin(), local_cardinality.end(), 0);

    for (int i = query_vertices_count - 1; i >= 0; --i) {
        VertexID u = construction_order[i];
        TreeNode& u_node = tree[u];

        uint32_t flag_num = 0;
        uint32_t updated_flag_count = 0;

        // Compute the intersection of TE_Candidates and NTE_Candidates.
        for (uint32_t j = 0; j < candidates_count[u]; ++j) {
            VertexID v = candidates[u][j];

            if (v == INVALID)
                continue;

            if (flag[v] == flag_num) {
                flag[v] += 1;
                updated_flag[updated_flag_count++] = v;
            }
        }

        for (VertexID u_bn: prevNeb[u]) {
            flag_num += 1;
            for (auto iter = NTE_Candidates[u][u_bn].begin(); iter != NTE_Candidates[u][u_bn].end(); ++iter) {
                for (auto v : iter->second) {
                    if (flag[v] == flag_num) {
                        flag[v] += 1;
                    }
                }
            }
        }

        flag_num += 1;

        // Get the cardinality of the candidates of u.
        for (uint32_t j = 0; j < candidates_count[u]; ++j) {
            VertexID v = candidates[u][j];
            if (v != INVALID && flag[v] == flag_num) {
                local_cardinality[v] = cardinality[u][j];
            }
            else {
                cardinality[u][j] = 0;
            }
        }

        VertexID u_p = u_node._parent;
        VertexID* frontiers = candidates[u_p];
        uint32_t frontiers_count = candidates_count[u_p];

        // Loop over TE_Candidates.
        for (uint32_t j = 0; j < frontiers_count; ++j) {
            VertexID v_f = frontiers[j];

            if (v_f == INVALID) {
                cardinality[u_p][j] = 0;
                continue;
            }

            uint32_t temp_score = 0;
            for (auto iter = TE_Candidates[u][v_f].begin(); iter != TE_Candidates[u][v_f].end();) {
                VertexID v = *iter;
                temp_score += local_cardinality[v];
                if (local_cardinality[v] == 0) {
                    iter = TE_Candidates[u][v_f].erase(iter);
                    for (VertexID u_c : u_node._children) {
                        TE_Candidates[u_c].erase(v);
                    }

                    for (VertexID u_c : followNeb[u]) {
                        NTE_Candidates[u_c][u].erase(v);
                    }
                }
                else {
                    ++iter;
                }
            }

            cardinality[u_p][j] *= temp_score;
        }

        // Clear updated flag.
        for (uint32_t j = 0; j < updated_flag_count; ++j) {
            flag[updated_flag[j]] = 0;
            local_cardinality[updated_flag[j]] = 0;
        }
    }

    compactCandidates(candidates, candidates_count, query_vertices_count);
    sortCandidates(candidates, candidates_count, query_vertices_count);


    for (uint32_t i = 0; i < query_vertices_count; ++i) {
        if (candidates_count[i] == 0) {
            std::cout << "CECI finds candidates = 0, exit;\n";
            return false;
        }
    }

    for (uint32_t i = 1; i < query_vertices_count; ++i) {
        VertexID u = construction_order[i];
        TreeNode& u_node = tree[u];

        // Clear TE_Candidates.
        {
            VertexID u_p = u_node._parent;
            auto iter = TE_Candidates[u].begin();
            while (iter != TE_Candidates[u].end()) {
                VertexID v_f = iter->first;
                if (!std::binary_search(candidates[u_p], candidates[u_p] + candidates_count[u_p], v_f)) {
                    iter = TE_Candidates[u].erase(iter);
                }
                else {
                    std::sort(iter->second.begin(), iter->second.end());
                    iter++;
                }
            }
        }

        // Clear NTE_Candidates.
        {
            for (VertexID u_p : prevNeb[u]) {
                auto iter = NTE_Candidates[u][u_p].end();
                while (iter != NTE_Candidates[u][u_p].end()) {
                    VertexID v_f = iter->first;
                    if (!std::binary_search(candidates[u_p], candidates[u_p] + candidates_count[u_p], v_f)) {
                        iter = NTE_Candidates[u][u_p].erase(iter);
                    }
                    else {
                        std::sort(iter->second.begin(), iter->second.end());
                        iter++;
                    }
                }
            }
        }
    }

    delete[] prevNeb;
    delete[] followNeb;
    delete[] tree;
    return true;
}

/*
    @ DAF/VEQ start
*/
VertexID DAFStartVertex(const Graph &data, const Query &query) {
    double min_score = data.getVertexCnt();
    VertexID start_vertex = 0;

    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        uint32_t degree = query.getVertexDegree(i);
        if (degree <= 1)
            continue;

        uint32_t count = 0;
        NLFByVertex(data, query, i, nullptr, count);
        double cur_score = count / (double)degree;
        if (cur_score < min_score) {
            min_score = cur_score;
            start_vertex = i;
        }
    }

    return start_vertex;
}

bool _DAF_Filter(const Graph &data, const Query &query, CandidateParam &candParam) {

    uint32_t **&candidates = candParam.candidates;
    uint32_t *&candidates_count = candParam.candidates_count;

    Order construction_order;
    
    TreeNode *tree = nullptr;

    if(_NLF_Filter(data, query, candidates, candidates_count) == false)
        return false;

    VertexID startVertex = DAFStartVertex(data, query);

    BFS(query, startVertex, tree, construction_order);

    PreviousNeb *prevNeb = new PreviousNeb[query.getVertexCnt()];
    FollowingNeb *followNeb = new FollowingNeb[query.getVertexCnt()];

    GetPreviousNeb(query, construction_order, prevNeb);
    GetFollowingNeb(query, construction_order, followNeb);

    uint32_t query_vertices_num = query.getVertexCnt();
    uint32_t* updated_flag = new uint32_t[data.getVertexCnt()];
    uint32_t* flag = new uint32_t[data.getVertexCnt()];
    std::fill(flag, flag + data.getVertexCnt(), 0);

    // The number of refinement is k. According to the original paper, we set k as 3.
    for (uint32_t k = 0; k < 3; ++k) {
        if (k % 2 == 0) {
            for (int i = 1; i < query_vertices_num; ++i) {
                VertexID query_vertex = construction_order[i];
                pruneCandidates(data, query, query_vertex, prevNeb[query_vertex], candidates, candidates_count, flag, updated_flag);
            }
        }
        else {
            for (int i = query_vertices_num - 2; i >= 0; --i) {
                VertexID query_vertex = construction_order[i];
                pruneCandidates(data, query, query_vertex, followNeb[query_vertex], candidates, candidates_count, flag, updated_flag);
            }
        }
    }

    compactCandidates(candidates, candidates_count, query.getVertexCnt());

    delete[] updated_flag;
    delete[] flag;
    delete[] prevNeb;
    delete[] followNeb;
    delete[] tree;
    if(!isCandidateSetValid(candidates, candidates_count, query.getVertexCnt())){
        std::cout << "DAF Filter finds candidates = 0, exit\n";
        return false;
    }

    return true;
}

// bool _Pilos_Filter(const Graph& data, const Query& query, int twohop, uint32_t **&candidates, uint32_t *&candidates_count, float **&EWeight, float **&eigenVD1, int alpha, int beta)
// {
//     auto start = std::chrono::high_resolution_clock::now();
//     uint32_t **candidates1 = NULL;
//     uint32_t *candidates_count1 = NULL;
//     int sizq = query.getVertexCnt();
//     uint32_t Eprun = sizq - 3;
//     Eprun = 30;
//     MatrixXd eigenVq1(sizq, Eprun);
//     int oMax = sizq * 3;
//     oMax = 10000;
//     MTcalc12(query_graph, query_graph->getGraphMaxDegree(), eigenVq1, true, Eprun, oMax);
//     float **eigenQ = NULL;
//     eigenQ = new float *[sizq];
//     for (uint32_t i = 0; i < sizq; ++i)
//     {
//         eigenQ[i] = new float[Eprun];
//         for (uint32_t j = 0; j < Eprun; j++)
//         {
//             eigenQ[i][j] = eigenVq1(i, j);
//         }
//     }
//     return PILOS(data, query, eigenQ, twohop, candidates, candidates_count, EWeight, eigenVD1, alpha, beta);
// }



inline bool Filter_NLF(const Graph& data, const Query& query, CandidateParam &canParam){
    if(_NLF_Filter(data, query, canParam.candidates, canParam.candidates_count) == false)
        return false;

    FilterParamConstruct(data, query, canParam);
    return true;
}

inline bool Filter_GQL(const Graph& data, const Query &query, CandidateParam& canParam){
    if(_GQL_Filter(data, query, canParam) == false)
        return false;
    
    FilterParamConstruct(data, query, canParam);
    return true;
}

inline bool Filter_CFL(const Graph& data, const Query &query, CandidateParam& canParam){
    if(_CFL_Filter(data, query, canParam) == false)
        return false;
    
    FilterParamConstruct(data, query, canParam);
    return true;
}

inline bool Filter_CECI(const Graph& data, const Query &query, CandidateParam& canParam){
    if(_CECI_Filter(data, query, canParam) == false)
        return false;
    
    FilterParamConstruct(data, query, canParam);
    return true;
}

inline bool Filter_DAF(const Graph& data, const Query& query, CandidateParam& canParam){
    if(_DAF_Filter(data, query, canParam) == false)
        return false;

    FilterParamConstruct(data, query, canParam);
    return true;
}

inline bool Filter_Pilos(const Graph& data, const Query& query, CandidateParam& canParam) {
    int alpha = 125;
    int beta = 0;
    float **eigenVD1 = data.getEigenValue();
    if(_Pilos_Filter(data, query, 2, canParam, eigenVD1, alpha, beta) == false)
        return false;

    FilterParamConstruct(data, query, canParam);
    return true;
}