#pragma once
#include "graph.h"
#include <queue>
#include <vector>
#include "assert.h"
#include <iostream>
#include <algorithm>
#include "config.h"

#if INTERSECTION_AVX2 
#include <immintrin.h>
#include <x86intrin.h>
#endif


// class Graph;
#if INTERSECTION_AVX2
void IntersectionAVX2(const VertexID* larray, const uint32_t l_count, const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    uint32_t lc = l_count;
    uint32_t rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        uint32_t tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    uint32_t li = 0;
    uint32_t ri = 0;

    __m256i per_u_order = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    __m256i per_v_order = _mm256_set_epi32(3, 2, 1, 0, 3, 2, 1, 0);
    VertexID* cur_back_ptr = cn;

    auto size_ratio = (rc) / (lc);
    if (size_ratio > 2) {
        if (li < lc && ri + 7 < rc) {
            __m256i u_elements = _mm256_set1_epi32(larray[li]);
            __m256i v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));

            while (true) {
                __m256i mask = _mm256_cmpeq_epi32(u_elements, v_elements);
                auto real_mask = _mm256_movemask_epi8(mask);
                if (real_mask != 0) {
                    // at most 1 element
                    *cur_back_ptr = larray[li];
                    cur_back_ptr += 1;
                }
                if (larray[li] > rarray[ri + 7]) {
                    ri += 8;
                    if (ri + 7 >= rc) {
                        break;
                    }
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                } else {
                    li++;
                    if (li >= lc) {
                        break;
                    }
                    u_elements = _mm256_set1_epi32(larray[li]);
                }
            }
        }
    } else {
        if (li + 1 < lc && ri + 3 < rc) {
            __m256i u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
            __m256i u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
            __m256i v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
            __m256i v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);

            while (true) {
                __m256i mask = _mm256_cmpeq_epi32(u_elements_per, v_elements_per);
                auto real_mask = _mm256_movemask_epi8(mask);
                if (real_mask << 16 != 0) {
                    *cur_back_ptr = larray[li];
                    cur_back_ptr += 1;
                }
                if (real_mask >> 16 != 0) {
                    *cur_back_ptr = larray[li + 1];
                    cur_back_ptr += 1;
                }


                if (larray[li + 1] == rarray[ri + 3]) {
                    li += 2;
                    ri += 4;
                    if (li + 1 >= lc || ri + 3 >= rc) {
                        break;
                    }
                    u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
                    u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                    v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);
                } else if (larray[li + 1] > rarray[ri + 3]) {
                    ri += 4;
                    if (ri + 3 >= rc) {
                        break;
                    }
                    v_elements = _mm256_loadu_si256((__m256i *) (rarray + ri));
                    v_elements_per = _mm256_permutevar8x32_epi32(v_elements, per_v_order);
                } else {
                    li += 2;
                    if (li + 1 >= lc) {
                        break;
                    }
                    u_elements = _mm256_loadu_si256((__m256i *) (larray + li));
                    u_elements_per = _mm256_permutevar8x32_epi32(u_elements, per_u_order);
                }
            }
        }
    }

    cn_count = (uint32_t)(cur_back_ptr - cn);
    if (li < lc && ri < rc) {
        while (true) {
            while (larray[li] < rarray[ri]) {
                ++li;
                if (li >= lc) {
                    return;
                }
            }
            while (larray[li] > rarray[ri]) {
                ++ri;
                if (ri >= rc) {
                    return;
                }
            }
            if (larray[li] == rarray[ri]) {
                // write back
                cn[cn_count++] = larray[li];

                ++li;
                ++ri;
                if (li >= lc || ri >= rc) {
                    return;
                }
            }
        }
    }
    return;
}

#endif

void IntersectionMerge(const VertexID* larray, const uint32_t l_count,const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count) {
    cn_count = 0;

    if (l_count == 0 || r_count == 0)
        return;

    uint32_t lc = l_count;
    uint32_t rc = r_count;

    if (lc > rc) {
        auto tmp = larray;
        larray = rarray;
        rarray = tmp;

        uint32_t tmp_count = lc;
        lc = rc;
        rc = tmp_count;
    }

    uint32_t li = 0;
    uint32_t ri = 0;

    while (true) {
        if (larray[li] < rarray[ri]) {
            li += 1;
            if (li >= lc) {
                return;
            }
        }
        else if (larray[li] > rarray[ri]) {
            ri += 1;
            if (ri >= rc) {
                return;
            }
        }
        else {
            cn[cn_count++] = larray[li];

            li += 1;
            ri += 1;
            if (li >= lc || ri >= rc) {
                return;
            }
        }
    }
}

void UnionMerge(const VertexID* larray, const uint32_t l_count, const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count){
    uint32_t li = 0, ri = 0;
    cn_count = 0;
    
    while (li < l_count && ri < r_count) {
        if (larray[li] < rarray[ri]) {
            cn[cn_count++] = larray[li++];
        } else if (larray[li] > rarray[ri]) {
            cn[cn_count++] = rarray[ri++];
        } else {
            cn[cn_count++] = larray[li];
            li++;
            ri++;
        }
    }
    
    while (li < l_count) cn[cn_count++] = larray[li++];
    while (ri < r_count) cn[cn_count++] = rarray[ri++];
}

void SubtractionMerge(const VertexID* larray, const uint32_t l_count, const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count){
    uint32_t li = 0, ri = 0;
    cn_count = 0;
    
    while (li < l_count && ri < r_count) {
        if (larray[li] < rarray[ri]) {
            cn[cn_count++] = larray[li++];
        } else if (larray[li] > rarray[ri]) {
            ri++;
        } else {
            li++;
            ri++;
        }
    }
    
    while (li < l_count) cn[cn_count++] = larray[li++];
}

inline void Union(const VertexID* larray, const uint32_t l_count, const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count){
    UnionMerge(larray, l_count, rarray, r_count, cn, cn_count);
}

inline void Subtraction(const VertexID* larray, const uint32_t l_count, const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count){
    SubtractionMerge(larray, l_count, rarray, r_count, cn, cn_count);
}


inline void Intersection(const VertexID* larray, const uint32_t l_count, const VertexID* rarray, const uint32_t r_count, VertexID* cn, uint32_t &cn_count){
#if INTERSECTION_AVX2
    IntersectionAVX2(larray, l_count, rarray, r_count, cn, cn_count);
#else
    IntersectionMerge(larray, l_count, rarray, r_count, cn, cn_count);
#endif
}

/*
 @ only should be called for query graph.
*/
void _GetNeb(const Query &query, const Order& order, std::vector<VertexID> *pnebs, bool isFollowing){

    if(query.getVertexCnt() != order.size()){
        std::cout << "query size not equal to order size\n";
        assert(query.getVertexCnt() == order.size());
    }

    uint32_t *vertex_pos = new uint32_t[query.getVertexCnt()];
    // pnebs = new std::vector<VertexID>[query.getVertexCnt()];

    for(int i=0; i<query.getVertexCnt(); i++){
        vertex_pos[order[i]] = i;
        pnebs[i].clear();
    }

    for(auto v: order){
        uint32_t v_pos = vertex_pos[v];
        uint32_t nebs_cnt;
        const VertexID* nebs = query.getNeb(v, nebs_cnt);
        for(int i=0; i<nebs_cnt; i++){
            VertexID neb = nebs[i];
            if(isFollowing){
                if(vertex_pos[neb] > v_pos){
                    pnebs[v].push_back(neb);
                }
            }
            else{
                if(vertex_pos[neb] < v_pos){
                    pnebs[v].push_back(neb);
                }
            }
        }
        std::sort(pnebs[v].begin(), pnebs[v].end(), 
          [&vertex_pos](const VertexID& a, const VertexID& b) {
              return vertex_pos[a] < vertex_pos[b];
          });
    }

    delete[] vertex_pos;

}

/*
 @ only should be called for query graph.
*/
inline void GetPreviousNeb(const Query &query, const Order &order, PreviousNeb *prevNeb){
    _GetNeb(query, order, prevNeb, false);
}

/*
 @ only should be called for query graph. 
*/
inline void GetFollowingNeb(const Query &query, const Order &order, FollowingNeb *followNeb){
    _GetNeb(query, order, followNeb, true);
}

/*
 @ only should be called for query graph.
*/
void BFS(const Graph& graph, VertexID start_vertex, TreeNode *&tree, Order &bfs_order) {

    uint32_t vertex_num = graph.getVertexCnt();

    std::queue<VertexID> bfs_queue;
    std::vector<bool> visited(vertex_num, false);

    tree = new TreeNode[vertex_num];

    bfs_order.clear();

    // uint32_t visited_vertex_count = 0;
    bfs_queue.push(start_vertex);
    visited[start_vertex] = true;
    tree[start_vertex]._level = 0;
    tree[start_vertex]._id = start_vertex;

    while(!bfs_queue.empty()) {
        const VertexID u = bfs_queue.front();
        bfs_queue.pop();
        bfs_order.push_back(u);

        uint32_t u_nbrs_count;
        const VertexID* u_nbrs = graph.getNeb(u, u_nbrs_count);
        for (uint32_t i = 0; i < u_nbrs_count; ++i) {
            VertexID u_nbr = u_nbrs[i];

            if (!visited[u_nbr]) {
                bfs_queue.push(u_nbr);
                visited[u_nbr] = true;
                tree[u_nbr]._id = u_nbr;
                tree[u_nbr]._parent = u;
                tree[u_nbr]._level = tree[u]._level + 1;
                tree[u]._children.push_back(u_nbr);
            }
        }
    }
}

/*
@ only should be called for data graph, calculate the k-hop pairs for plocal
*/
void GetPairsCnt(const Graph& graph, VertexID start_vertex, int alpha, size_t& res) {

    uint32_t vertex_num = graph.getVertexCnt();

    std::queue<std::pair<VertexID, int>> bfs_queue;
    std::vector<bool> visited(vertex_num, false);

    int _res = 0;


    // uint32_t visited_vertex_count = 0;
    bfs_queue.push(std::make_pair(start_vertex, 0));
    visited[start_vertex] = true;

    while(!bfs_queue.empty()) {
        const auto p = bfs_queue.front();
        bfs_queue.pop();
        VertexID u = p.first;
        int level = p.second;

        if(level == alpha)
            break;
        
        _res ++;

        uint32_t u_nbrs_count;
        const VertexID* u_nbrs = graph.getNeb(u, u_nbrs_count);
        for (uint32_t i = 0; i < u_nbrs_count; ++i) {
            VertexID u_nbr = u_nbrs[i];

            if (!visited[u_nbr]) {
                bfs_queue.push(std::make_pair(u_nbr, level+1));
                visited[u_nbr] = true;
            }
        }
    }

    res = _res - 1;
}

void GetAllPairDistance(const Query &query, uint32_t **khop_map){

    uint32_t qv_cnt = query.getVertexCnt();

    for(int u=0; u<qv_cnt; u++){
        uint32_t u_nbrs_cnt = 0;
        const VertexID *u_nbrs = query.getNeb(u, u_nbrs_cnt);
        for(int idx = 0; idx < u_nbrs_cnt; idx++){
            VertexID u_nbr = u_nbrs[idx];
            khop_map[u][u_nbr] = 1;
        }
        khop_map[u][u] = 0;
    }

    for(int k=0; k<qv_cnt; k++){
        for(int i=0; i<qv_cnt; i++){
            for(int j=0; j<qv_cnt; j++){
                if(j==i) continue;
                khop_map[i][j] = std::min(khop_map[i][j], khop_map[i][k] + khop_map[k][j]);
            }
        }
    }
}