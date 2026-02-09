#pragma once
#include "graph.h"
#include <queue>
#include <vector>
#include <algorithm>
#include "filter.h"
#include <random>
#include "vertexset.h"
#include "api.h"

VertexID GQLStartVertex(const Query& query, const CandidateParam& canParam) {
    /**
     * Select the vertex with the minimum number of candidates as the start vertex.
     * Tie Handling:
     *  1. degree
     *  2. label id
     */

    uint32_t start_vertex = 0;


    for (uint32_t i = 1; i <query.getVertexCnt(); ++i) {
        VertexID cur_vertex = i;

        if (canParam.candidates_count[cur_vertex] < canParam.candidates_count[start_vertex]) {
            start_vertex = cur_vertex;
        }
        else if (canParam.candidates_count[cur_vertex] == canParam.candidates_count[start_vertex]
                && query.getVertexDegree(cur_vertex) > query.getVertexDegree(start_vertex)) {
            start_vertex = cur_vertex;
        }
    }

    return start_vertex;
}

// IVE's MDE ordering
void getMaxIsolated(const Query &query, Order &order, uint32_t &isolate_v_idx){
    uint32_t qv_cnt = query.getVertexCnt();
    int *degree_copy = new int[qv_cnt];
    bool *vis = new bool[qv_cnt];

    order.clear();

    std::fill(vis, vis+qv_cnt, false);
    
    VertexID u_start = 0;

    for(int i=0; i<qv_cnt; i++){
        degree_copy[i] = query.getVertexDegree(i);
        if(degree_copy[i] == query.getMaxDgree())
            u_start = i;
    }

    auto cmp = [&degree_copy](VertexID u1, VertexID u2){
        return degree_copy[u1]<degree_copy[u2];
    };

    std::priority_queue<VertexID, std::vector<VertexID>, decltype(cmp)> pq(cmp);
    pq.push(u_start);
    vis[u_start] = true;

    uint32_t order_cnt = 0;
    isolate_v_idx = 0;

    while (!pq.empty())
    {
        VertexID u = pq.top(); pq.pop();
        uint32_t neb_cnt;

        if(degree_copy[u] == 0 && isolate_v_idx == 0)
            isolate_v_idx = order_cnt;

        const VertexID* nebs = query.getNeb(u, neb_cnt);
        for(int i=0; i<neb_cnt; i++)
            degree_copy[nebs[i]]--;
        
        // need resconstruct since priority changed
        std::vector<VertexID> elements;
        while (!pq.empty())
        {
            elements.push_back(pq.top());
            pq.pop();
        }
        for(auto e: elements)
            pq.push(e);
        
        for(int i=0; i<neb_cnt; i++){
            if(vis[nebs[i]] == false){
                pq.push(nebs[i]);
                vis[nebs[i]] = true;
            }
        }

        order_cnt ++;
        order.push_back(u);
    }


    delete[] degree_copy;
    delete[] vis;
}

bool isValid(const Order &order, const Query &query){
    uint32_t size = order.size();
    if(size != query.getVertexCnt()) return false;
    
    for(int i=1; i<size; i++){
        bool valid = false;
        VertexID vi = order[i];
        uint32_t vi_neb_cnt = 0;
        const VertexID* vi_neb = query.getNeb(vi, vi_neb_cnt);
        for(int j=0; j<i; j++){
            VertexID vj = order[j];
            if(std::binary_search(vi_neb, vi_neb+vi_neb_cnt, vj)){
                valid = true;
                break;
            }
        }
        if(valid == false) return false;
    }

    return true;
}

std::vector<Order> orderGenerate(const Query &query){
    std::vector<Order> permutations;
    uint32_t size = query.getVertexCnt();
    Order base(size);
    for (int i = 0; i < size; ++i) {
        base[i] = i;
    }

    do{
        // for(auto v: base){
        //     std::cout <<v <<" ";
        // }
        // std::cout<<'\n';
        if(isValid(base, query))
            permutations.push_back(base);
    }while(std::next_permutation(base.begin(), base.end()));
    
    return permutations;
}

std::pair<VertexID, VertexID> QSIStartEdge(const Graph &query, const CandidateParam& canParam) {
    /**
     * Select the edge with the minimum number of candidate edges.
     * Tie Handling:
     *  1. the sum of the degree values of the end points of an edge (prioritize the edge with smaller degree).
     *  2. label id
     */
    uint32_t min_value = std::numeric_limits<uint32_t>::max();
    uint32_t min_degree_sum = std::numeric_limits<uint32_t>::max();

    std::pair<VertexID, VertexID> start_edge = std::make_pair(0, 1);
    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        VertexID begin_vertex = i;
        uint32_t nbrs_cnt;
        const uint32_t* nbrs = query.getNeb(begin_vertex, nbrs_cnt);

        for (uint32_t j = 0; j < nbrs_cnt; ++j) {
            VertexID end_vertex = nbrs[j];
            uint32_t cur_value = (*canParam.edge_matrix[begin_vertex][end_vertex])._e_cnt;
            uint32_t cur_degree_sum = query.getVertexDegree(begin_vertex) + query.getVertexDegree(end_vertex);

            if (cur_value < min_value || (cur_value == min_value
                && (cur_degree_sum < min_degree_sum))) {
                min_value = cur_value;
                min_degree_sum = cur_degree_sum;

                start_edge = query.getVertexDegree(begin_vertex) < query.getVertexDegree(end_vertex) ?
                        std::make_pair(end_vertex, begin_vertex) : std::make_pair(begin_vertex, end_vertex);
            }
        }
    }

    return start_edge;
}


void generateRootToLeafPaths(TreeNode *tree_node, VertexID cur_vertex, std::vector<uint32_t> &cur_path,
                                                std::vector<std::vector<uint32_t>> &paths) {
    TreeNode& cur_node = tree_node[cur_vertex];
    cur_path.push_back(cur_vertex);

    if (cur_node._children.size() == 0) {
        paths.emplace_back(cur_path);
    }
    else {
        for (VertexID next_vertex : cur_node._children) {
            generateRootToLeafPaths(tree_node, next_vertex, cur_path, paths);
        }
    }

    cur_path.pop_back();
}

void estimatePathEmbeddsingsNum(std::vector<uint32_t> &path, Edges ***edge_matrix, std::vector<size_t> &estimated_embeddings_num) {
    assert(path.size() > 1);
    std::vector<size_t> parent;
    std::vector<size_t> children;

    estimated_embeddings_num.resize(path.size() - 1);
    Edges& last_edge = *edge_matrix[path[path.size() - 2]][path[path.size() - 1]];
    children.resize(last_edge._v_cnt);

    size_t sum = 0;
    for (uint32_t i = 0; i < last_edge._v_cnt; ++i) {
        children[i] = last_edge._offset[i + 1] - last_edge._offset[i];
        sum += children[i];
    }

    estimated_embeddings_num[path.size() - 2] = sum;

    for (int i = path.size() - 2; i >= 1; --i) {
        uint32_t begin = path[i - 1];
        uint32_t end = path[i];

        Edges& edge = *edge_matrix[begin][end];
        parent.resize(edge._v_cnt);

        sum = 0;
        for (uint32_t j = 0; j < edge._v_cnt; ++j) {

            size_t local_sum = 0;
            for (uint32_t k = edge._offset[j]; k < edge._offset[j + 1]; ++k) {
                uint32_t nbr = edge._edge[k];
                local_sum += children[nbr];
            }

            parent[j] = local_sum;
            sum += local_sum;
        }

        estimated_embeddings_num[i - 1] = sum;
        parent.swap(children);
    }
}

uint32_t generateNoneTreeEdgesCount(const Query &query, TreeNode *tree_node, std::vector<uint32_t> &path) {
    uint32_t non_tree_edge_count = query.getVertexDegree(path[0]) - tree_node[path[0]]._children.size();

    for (uint32_t i = 1; i < path.size(); ++i) {
        VertexID vertex = path[i];
        non_tree_edge_count += query.getVertexDegree(vertex) - tree_node[vertex]._children.size() - 1;
    }

    return non_tree_edge_count;
}

void generateCorePaths(const Query &query, TreeNode *tree_node, VertexID cur_vertex,
                                          std::vector<uint32_t> &cur_core_path, std::vector<std::vector<uint32_t>> &core_paths) {
    TreeNode& node = tree_node[cur_vertex];
    cur_core_path.push_back(cur_vertex);

    bool is_core_leaf = true;
    for (VertexID child : node._children) {
        if (query.getKCoreValue(child) > 1) {
            generateCorePaths(query, tree_node, child, cur_core_path, core_paths);
            is_core_leaf = false;
        }
    }

    if (is_core_leaf) {
        core_paths.emplace_back(cur_core_path);
    }
    cur_core_path.pop_back();
}

void generateTreePaths(const Query &query, TreeNode *tree_node, VertexID cur_vertex, std::vector<uint32_t> &cur_tree_path, std::vector<std::vector<uint32_t>> &tree_paths) {
    TreeNode& node = tree_node[cur_vertex];
    cur_tree_path.push_back(cur_vertex);

    bool is_tree_leaf = true;
    for (VertexID child : node._children) {
        if (query.getVertexDegree(child) > 1) {
            generateTreePaths(query, tree_node, child, cur_tree_path, tree_paths);
            is_tree_leaf = false;
        }
    }

    if (is_tree_leaf && cur_tree_path.size() > 1) {
        tree_paths.emplace_back(cur_tree_path);
    }
    cur_tree_path.pop_back();
}

void generateLeaves(const Query &query, std::vector<uint32_t> &leaves) {
    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        VertexID cur_vertex = i;
        if (query.getVertexDegree(cur_vertex) == 1) {
            leaves.push_back(cur_vertex);
        }
    }
}


void updateValidVertices(const Query &query, VertexID query_vertex, std::vector<bool> &visited, std::vector<bool> &adjacent) {
    visited[query_vertex] = true;
    uint32_t nbr_cnt;
    const uint32_t* nbrs = query.getNeb(query_vertex, nbr_cnt);

    for (uint32_t i = 0; i < nbr_cnt; ++i) {
        uint32_t nbr = nbrs[i];
        adjacent[nbr] = true;
    }
}

void gupOrderCore(const Query& query, const CandidateParam &candParam, const Order& bfs_order, Order &order, std::vector<bool> &ordered){

    assert(order.size() == 1);
    assert(ordered[order[0]] == true);

    uint32_t qvcnt = query.getVertexCnt();
    // order.clear();

    std::vector<std::vector<double>> weights(qvcnt, std::vector<double>(qvcnt, 0.0));
    for(uint32_t v=0; v<qvcnt; v++){
        if(query.getKCoreValue(v) >= 2){
            uint32_t v_neb_cnt;
            const VertexID* vnebs = query.getNeb(v, v_neb_cnt);
            for(uint32_t vnebs_idx=0; vnebs_idx<v_neb_cnt; vnebs_idx++){
                VertexID v_neb = vnebs[vnebs_idx];
                if(query.getKCoreValue(v_neb) >= 2 && bfs_order[v] < bfs_order[v_neb]){

                    double weight = 0.0;
                    weight = (double)(*candParam.edge_matrix[v][v_neb])._e_cnt;
                    weight = weight / (double)(candParam.candidates_count[v]);
                    weight = weight / (double)(candParam.candidates_count[v_neb]);
                    weights[v][v_neb] = weight;
                    weights[v_neb][v] = weight;
                }
            }
        }
    }

    std::priority_queue<
        std::pair<double, int>,
        std::vector<std::pair<double, int>>,
        std::greater<std::pair<double, int>>
    > pq;
    std::vector<bool> queued(qvcnt, false);
    std::vector<double> selectivity(qvcnt, 1.0);

    auto cost = [&](VertexID u,
        const std::vector<bool> &ordered,
        const std::vector<bool> &queued,
        const std::vector<double> &selectivity) -> double
    {
        double c = selectivity[u] * static_cast<double>(candParam.candidates_count[u]);
        uint32_t u_neb_cnt;
        const VertexID* u_nebs = query.getNeb(u, u_neb_cnt);
        for (uint32_t i = 0; i < u_neb_cnt; ++i) {
            VertexID u_nbr = u_nebs[i];
            if (queued[u_nbr] && !ordered[u_nbr]) {
                c *= weights[u][u_nbr] * selectivity[u_nbr];
            }
        }
        return c;
    };

    VertexID v = order[0];
    while(1){
        uint32_t v_neb_cnt;
        const VertexID* v_nebs = query.getNeb(v, v_neb_cnt);
        for(uint32_t vnebs_idx=0; vnebs_idx<v_neb_cnt; vnebs_idx++){
            VertexID v_neb = v_nebs[vnebs_idx];
            if(query.getKCoreValue(v_neb) >= 2 && ordered[v_neb] == false){
                queued[v_neb] = true;
                selectivity[v_neb] *= weights[v][v_neb];
            }
        }

        for(uint32_t vnebs_idx=0; vnebs_idx<v_neb_cnt; vnebs_idx++){
            VertexID v_neb = v_nebs[vnebs_idx];
            if(query.getKCoreValue(v_neb) >= 2 && ordered[v_neb] == false){
                double c = cost(v_neb, ordered, queued, selectivity);
                pq.push(std::make_pair(c, v_neb));
            }
        }

        while(1){
            if(pq.empty()) return;
            auto top = pq.top();
            pq.pop();
            v = top.second;
            if(!ordered[v]) break;
        }

        order.push_back(v);
        ordered[v] = true;
    }
}

std::vector<size_t> count_embed(VertexID u, const Query& query, const CandidateParam& candParam, const TreeNode* bfstree, std::vector<bool> &ordered, std::vector<double> &nembeds){

    std::vector<size_t> npaths = std::vector<size_t>(candParam.candidates_count[u], 1);
    uint32_t u_neb_cnt;
    const VertexID* u_nbrs = query.getNeb(u, u_neb_cnt);
    for(uint32_t i=0; i<u_neb_cnt; i++){
        VertexID u_neb = u_nbrs[i];
        if(ordered[u_neb] || bfstree[u_neb]._parent != u) continue;

        auto _npaths = count_embed(u_neb, query, candParam, bfstree, ordered, nembeds);
        size_t npsum = 0;
        for (size_t iv = 0; iv < candParam.candidates_count[u]; iv++) {
            uint32_t v_neb_cnt = 0;
            const VertexID* v_nbrs = candParam.edge_matrix[u][u_neb]->getNeb(iv, v_neb_cnt);
            size_t np = 0;
            for (int ii =0; ii < v_neb_cnt; ii++) {
                uint32_t _iv_neb = v_nbrs[ii];
                np += _npaths[_iv_neb];
            }
            npaths[iv] *= np;
            npsum += np;
        }
        nembeds[u_neb] = static_cast<double>(npsum) / candParam.candidates_count[u];
    }
    return npaths;
}

void gupOrderTree(const Query& query, const CandidateParam &candParam, const TreeNode* bfstree, Order &order, std::vector<bool> &ordered){

    using Item = std::tuple<int, double, VertexID>;
    using MinHeap = std::priority_queue<Item, std::vector<Item>, std::greater<Item>>;
    MinHeap minHeap;
    std::vector<double> nembeds(query.getVertexCnt(), 0.0);

    for(int u=0; u<query.getVertexCnt(); u++){
        uint32_t u_neb_cnt;
        const VertexID* u_nbrs = query.getNeb(u, u_neb_cnt);
        bool has_unordered_neb = std::any_of(u_nbrs, u_nbrs + u_neb_cnt, [&ordered](VertexID v) { return !ordered[v]; });
        if(ordered[u] && has_unordered_neb) {
            count_embed(u, query, candParam, bfstree, ordered, nembeds);
            for(int ui=0; ui<u_neb_cnt; ui++){
                auto _u = u_nbrs[ui];
                if(!ordered[_u]){
                    minHeap.push(std::make_tuple(0, nembeds[_u], _u));
                }
            }
        }
    }

    while(!minHeap.empty()){
        auto [d, _, u] = minHeap.top();
        minHeap.pop();

        assert(ordered[u] == false);

        order.push_back(u);
        ordered[u] = true;
        uint32_t u_neb_cnt;
        const VertexID* u_nbrs = query.getNeb(u, u_neb_cnt);
        for(int ui=0; ui<u_neb_cnt; ui++){
            auto _u = u_nbrs[ui];
            if(!ordered[_u]){
                minHeap.push(std::make_tuple(d - 1, nembeds[_u], _u));
            }
        }
    }
}

VertexID _GuP_start_vertex(const Query &query, const CandidateParam &candParam) {

    bool is_tree_or_cycle = true;
    for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
        if (query.getVertexDegree(i) > 2) {
            is_tree_or_cycle = false;
            break;
        }
    }

    assert(is_tree_or_cycle == false);

    // select_root_core
    uint32_t max_kcore = 0;
    for(int i=0; i<query.getVertexCnt(); i++){
        max_kcore = std::max(max_kcore, query.getKCoreValue(i));
    }

    std::vector<VertexID> cand;
    cand.clear();
    for(int i=0; i<query.getVertexCnt(); i++){
        if(query.getKCoreValue(i) == max_kcore) cand.push_back(i);
    }

    for(int k= max_kcore; k>=1; k--){

        std::vector<std::pair<VertexID, double>> ps;
        ps.reserve(query.getVertexCnt());
        // for (VertexID u = 0; u < query.getVertexCnt(); ++u) {
            // if (query.getKCoreValue(u) != k) continue;
        for (VertexID u : cand) {
            int coredeg = 0;
            uint32_t u_neb_cnt;
            const VertexID* u_nebs = query.getNeb(u, u_neb_cnt);
            for (uint32_t i = 0; i < u_neb_cnt; ++i) {
                VertexID uneb = u_nebs[i];
                if (query.getKCoreValue(uneb) >= k) {
                    coredeg++;
                }
            }
            if (coredeg == 0) continue;
            double weight = candParam.candidates_count[u] / (double)coredeg;
            ps.emplace_back(u, weight);
        }

        if (ps.empty()) continue;

        auto it_min = std::min_element(
            ps.begin(), ps.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; });
        double minp = it_min->second;

        auto equal_eps = [minp](double x) {
            double ax = std::fabs(x);
            double am = std::fabs(minp);
            double scale = std::max(1.0, std::max(ax, am));
            return std::fabs(x - minp) <= (1e-12 * scale);
        };

        std::vector<VertexID> mins;
        mins.reserve(ps.size());
        for (auto& [u, pen] : ps) {
            if (equal_eps(pen)) mins.push_back(u);
        }

        if (mins.size() == 1 || k == 1) {
            return mins.front();
        }
    }

    return cand.front(); // Fallback, should not happen
}

void Order_GuP(const Graph &data, const Query &query, const CandidateParam &candParam, Order &order){
    
    order.clear();
    std::vector<bool> ordered(query.getVertexCnt(), false);

    VertexID root;
    CandidateParam _nlf(data, query);
    Filter_NLF(data, query, _nlf);
    root = _GuP_start_vertex(query, _nlf);

    assert(query.getKCoreLength() == 0 || query.getKCoreValue(root) >= 2);

    Order bfs_order;
    TreeNode *tree = nullptr;
    BFS(query, root, tree, bfs_order);

    order.push_back(root);
    ordered[root] = true;
    gupOrderCore(query, candParam, bfs_order, order, ordered);
    gupOrderTree(query, candParam, tree, order, ordered);

    assert(order.size() == query.getVertexCnt());

    delete[] tree;
}

void Order_GQL(const Graph &data, const Query &query, const CandidateParam &candParam, Order &order){

    order.clear();

    std::vector<bool> visited_vertices(query.getVertexCnt(), false);
    std::vector<bool> adjacent_vertices(query.getVertexCnt(), false);
    order.resize(query.getVertexCnt(), 0);

    VertexID start_vertex = GQLStartVertex(query, candParam);
    order[0] = start_vertex;
    
    updateValidVertices(query, start_vertex, visited_vertices, adjacent_vertices);

    for (uint32_t i = 1; i < query.getVertexCnt(); ++i) {
        VertexID next_vertex = 0;
        uint32_t min_value = data.getVertexCnt() + 1;
        for (uint32_t j = 0; j < query.getVertexCnt(); ++j) {
            VertexID cur_vertex = j;

            if (!visited_vertices[cur_vertex] && adjacent_vertices[cur_vertex]) {
                if (candParam.candidates_count[cur_vertex] < min_value) {
                        min_value = candParam.candidates_count[cur_vertex];
                        next_vertex = cur_vertex;
                }
                else if (candParam.candidates_count[cur_vertex] == min_value && query.getVertexDegree(cur_vertex) > query.getVertexDegree(next_vertex)) {
                        next_vertex = cur_vertex;
                }
            }
        }
        updateValidVertices(query, next_vertex, visited_vertices, adjacent_vertices);
        order[i] = next_vertex;
    }


}


void Order_CECI(const Graph &data, const Query &query, Order& order){
    VertexID startVertex = CECIStartVertex(data, query);

    TreeNode *tree = nullptr;

    BFS(query, startVertex, tree, order);

    delete[] tree;
}

void Order_QSI(const Graph& data, const Query& query, const CandidateParam& canParam, Order& order) {

    std::vector<bool> visited_vertices(query.getVertexCnt(), false);
    std::vector<bool> adjacent_vertices(query.getVertexCnt(), false);

    order.clear();
    order.resize(query.getVertexCnt());

    std::pair<VertexID, VertexID> start_edge =  QSIStartEdge(query, canParam);
    order[0] = start_edge.first;
    order[1] = start_edge.second;

    updateValidVertices(query, start_edge.first, visited_vertices, adjacent_vertices);
    updateValidVertices(query, start_edge.second, visited_vertices, adjacent_vertices);

    for (uint32_t l = 2; l < query.getVertexCnt(); ++l) {
        uint32_t min_value = std::numeric_limits<uint32_t>::max();
        uint32_t max_degree = 0;
        uint32_t max_adjacent_selected_vertices = 0;
        std::pair<VertexID, VertexID> selected_edge;
        for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
            VertexID begin_vertex = i;
            if (visited_vertices[begin_vertex]) {
                uint32_t nbrs_cnt;
                const VertexID *nbrs = query.getNeb(begin_vertex, nbrs_cnt);
                for (uint32_t j = 0; j < nbrs_cnt; ++j) {
                        VertexID end_vertex = nbrs[j];

                        if (!visited_vertices[end_vertex]) {
                            uint32_t cur_value = (*canParam.edge_matrix[begin_vertex][end_vertex])._e_cnt;
                            uint32_t cur_degree = query.getVertexDegree(end_vertex);
                            uint32_t adjacent_selected_vertices = 0;
                            uint32_t end_vertex_nbrs_count;
                            const VertexID *end_vertex_nbrs = query.getNeb(end_vertex, end_vertex_nbrs_count);

                            for (uint32_t k = 0; k < end_vertex_nbrs_count; ++k) {
                                VertexID end_vertex_nbr = end_vertex_nbrs[k];

                                if (visited_vertices[end_vertex_nbr]) {
                                    adjacent_selected_vertices += 1;
                                }
                            }

                            if (cur_value < min_value || (cur_value == min_value && adjacent_selected_vertices < max_adjacent_selected_vertices)
                                    || (cur_value == min_value && adjacent_selected_vertices == max_adjacent_selected_vertices && cur_degree > max_degree)) {
                                selected_edge = std::make_pair(begin_vertex, end_vertex);
                                min_value = cur_value;
                                max_degree = cur_degree;
                                max_adjacent_selected_vertices = adjacent_selected_vertices;
                            }
                        }
                }
            }
        }

        order[l] = selected_edge.second;
        updateValidVertices(query, selected_edge.second, visited_vertices, adjacent_vertices);
    }
}

void Order_VF2PP(const Graph &data, const Query &query, Order &order) {
    uint32_t query_vertices_num = query.getVertexCnt();

    uint32_t property_count = 0;
    std::vector<std::vector<uint32_t>> properties(query_vertices_num);
    std::vector<bool> order_type(query_vertices_num, true);     // True: Ascending, False: Descending.
    std::vector<uint32_t> vertices;
    std::vector<bool> in_matching_order(query_vertices_num, false);

    for (uint32_t i = 0; i < query_vertices_num; ++i) {
        properties[i].resize(3);
    }

    // Select the root vertex with the rarest node labels and the largest degree.
    property_count = 2;
    order_type[0] = true;
    order_type[1] = false;

    for (uint32_t u = 0; u < query_vertices_num; ++u) {
        vertices.push_back(u);
        properties[u][0] = data.getLabelFreq(query.getVertexLabel(u));
        properties[u][1] = query.getVertexDegree(u);
    }

    auto order_lambda = [&properties, &order_type, property_count](uint32_t l, uint32_t r) -> bool {
        for (uint32_t x = 0; x < property_count; ++x) {
            if (properties[l][x] == properties[r][x])
                continue;

            if (order_type[0]) {
                return properties[l][x] < properties[r][x];
            }
            else {
                return properties[l][x] > properties[r][x];
            }
        }

        return l < r;
    };

    order.clear();
    order.resize(query_vertices_num, 0);
    std::stable_sort(vertices.begin(), vertices.end(), order_lambda);
    VertexID startVertex = vertices[0];
    in_matching_order[startVertex] = true;
    order[0] = startVertex;

    vertices.clear();
    TreeNode* tree = nullptr;
    
    Order bfs_order;
    bfs_order.clear();

    BFS(query, startVertex, tree, bfs_order);

    property_count = 3;
    order_type[0] = false;
    order_type[1] = false;
    order_type[2] = true;

    uint32_t level = 1;
    uint32_t count = 1;
    uint32_t num_vertices_in_matching_order = 1;
    while (num_vertices_in_matching_order < query_vertices_num) {
        // Get the vertices in current level.
        while (count < query_vertices_num) {
            uint32_t u = bfs_order[count];
            if (tree[u]._level == level) {
                vertices.push_back(u);
                count += 1;
            }
            else {
                level += 1;
                break;
            }
        }

        // Process a level in the BFS tree.
        while(!vertices.empty()) {
            // Set property.
            for (auto u : vertices) {
                uint32_t un_count;
                const uint32_t* un = query.getNeb(u, un_count);

                uint32_t bn_count = 0;
                for (uint32_t i = 0; i < un_count; ++i) {
                    uint32_t uu = un[i];
                    if (in_matching_order[uu]) {
                        bn_count += 1;
                    }
                }

                properties[u][0] = bn_count;
                properties[u][1] = query.getVertexDegree(u);
                properties[u][2] = data.getLabelFreq(query.getVertexLabel(u));
            }


            std::sort(vertices.begin(), vertices.end(), order_lambda);
            order[num_vertices_in_matching_order++] = vertices[0];

            in_matching_order[vertices[0]] = true;

            vertices.erase(vertices.begin());
        }
    }


    delete[] tree;
}

void Order_CFL(const Graph &data, const Query &query, const CandidateParam &canParam, Order &order) {

    Edges ***edge_matrix = canParam.edge_matrix;
    uint32_t *candidates_count = canParam.candidates_count;

    TreeNode *tree = nullptr;
    Order bfs_order;

    VertexID startVertex = CFLStartVertex(data, query);

    BFS(query, startVertex, tree, bfs_order);

    uint32_t query_vertices_num = query.getVertexCnt();
    VertexID root_vertex = bfs_order[0];
    order.clear();
    order.resize(query.getVertexCnt(), 0);
    std::vector<bool> visited_vertices(query_vertices_num, false);

    std::vector<std::vector<uint32_t>> core_paths;
    std::vector<std::vector<std::vector<uint32_t>>> forests;
    std::vector<uint32_t> leaves;

    generateLeaves(query, leaves);
    if (query.getKCoreValue(root_vertex) > 1) {
        std::vector<uint32_t> temp_core_path;
        generateCorePaths(query, tree, root_vertex, temp_core_path, core_paths);
        for (uint32_t i = 0; i < query_vertices_num; ++i) {
            VertexID cur_vertex = i;
            if (query.getKCoreValue(cur_vertex) > 1) {
                std::vector<std::vector<uint32_t>> temp_tree_paths;
                std::vector<uint32_t> temp_tree_path;
                generateTreePaths(query, tree, cur_vertex, temp_tree_path, temp_tree_paths);
                if (!temp_tree_paths.empty()) {
                    forests.emplace_back(temp_tree_paths);
                }
            }
        }
    }
    else {
        std::vector<std::vector<uint32_t>> temp_tree_paths;
        std::vector<uint32_t> temp_tree_path;
        generateTreePaths(query, tree, root_vertex, temp_tree_path, temp_tree_paths);
        if (!temp_tree_paths.empty()) {
            forests.emplace_back(temp_tree_paths);
        }
    }

    // Order core paths.
    uint32_t selected_vertices_count = 0;
    order[selected_vertices_count++] = root_vertex;
    visited_vertices[root_vertex] = true;

    if (!core_paths.empty()) {
        std::vector<std::vector<size_t>> paths_embededdings_num;
        std::vector<uint32_t> paths_non_tree_edge_num;
        for (auto& path : core_paths) {
            uint32_t non_tree_edge_num = generateNoneTreeEdgesCount(query, tree, path);
            paths_non_tree_edge_num.push_back(non_tree_edge_num + 1);

            std::vector<size_t> path_embeddings_num;
            estimatePathEmbeddsingsNum(path, edge_matrix, path_embeddings_num);
            paths_embededdings_num.emplace_back(path_embeddings_num);
        }

        // Select the start path.
        double min_value = std::numeric_limits<double>::max();
        uint32_t selected_path_index = 0;

        for (uint32_t i = 0; i < core_paths.size(); ++i) {
            double cur_value = paths_embededdings_num[i][0] / (double) paths_non_tree_edge_num[i];

            if (cur_value < min_value) {
                min_value = cur_value;
                selected_path_index = i;
            }
        }


        for (uint32_t i = 1; i < core_paths[selected_path_index].size(); ++i) {
            order[selected_vertices_count] = core_paths[selected_path_index][i];
            selected_vertices_count += 1;
            visited_vertices[core_paths[selected_path_index][i]] = true;
        }

        core_paths.erase(core_paths.begin() + selected_path_index);
        paths_embededdings_num.erase(paths_embededdings_num.begin() + selected_path_index);
        paths_non_tree_edge_num.erase(paths_non_tree_edge_num.begin() + selected_path_index);

        while (!core_paths.empty()) {
            min_value = std::numeric_limits<double>::max();
            selected_path_index = 0;

            for (uint32_t i = 0; i < core_paths.size(); ++i) {
                uint32_t path_root_vertex_idx = 0;
                for (uint32_t j = 0; j < core_paths[i].size(); ++j) {
                    VertexID cur_vertex = core_paths[i][j];

                    if (visited_vertices[cur_vertex])
                        continue;

                    path_root_vertex_idx = j - 1;
                    break;
                }

                double cur_value = paths_embededdings_num[i][path_root_vertex_idx] / (double)candidates_count[core_paths[i][path_root_vertex_idx]];
                if (cur_value < min_value) {
                    min_value = cur_value;
                    selected_path_index = i;
                }
            }

            for (uint32_t i = 1; i < core_paths[selected_path_index].size(); ++i) {
                if (visited_vertices[core_paths[selected_path_index][i]])
                    continue;

                order[selected_vertices_count] = core_paths[selected_path_index][i];
                selected_vertices_count += 1;
                visited_vertices[core_paths[selected_path_index][i]] = true;
            }

            core_paths.erase(core_paths.begin() + selected_path_index);
            paths_embededdings_num.erase(paths_embededdings_num.begin() + selected_path_index);
        }
    }

    // Order tree paths.
    for (auto& tree_paths : forests) {
        std::vector<std::vector<size_t>> paths_embededdings_num;
        for (auto& path : tree_paths) {
            std::vector<size_t> path_embeddings_num;
            estimatePathEmbeddsingsNum(path, edge_matrix, path_embeddings_num);
            paths_embededdings_num.emplace_back(path_embeddings_num);
        }

        while (!tree_paths.empty()) {
            double min_value = std::numeric_limits<double>::max();
            uint32_t selected_path_index = 0;

            for (uint32_t i = 0; i < tree_paths.size(); ++i) {
                uint32_t path_root_vertex_idx = 0;
                for (uint32_t j = 0; j < tree_paths[i].size(); ++j) {
                    VertexID cur_vertex = tree_paths[i][j];

                    if (visited_vertices[cur_vertex])
                        continue;

                    path_root_vertex_idx = j == 0 ? j : j - 1;
                    break;
                }

                double cur_value = paths_embededdings_num[i][path_root_vertex_idx] / (double)candidates_count[tree_paths[i][path_root_vertex_idx]];
                if (cur_value < min_value) {
                    min_value = cur_value;
                    selected_path_index = i;
                }
            }

            for (uint32_t i = 0; i < tree_paths[selected_path_index].size(); ++i) {
                if (visited_vertices[tree_paths[selected_path_index][i]])
                    continue;

                order[selected_vertices_count] = tree_paths[selected_path_index][i];
                selected_vertices_count += 1;
                visited_vertices[tree_paths[selected_path_index][i]] = true;
            }

            tree_paths.erase(tree_paths.begin() + selected_path_index);
            paths_embededdings_num.erase(paths_embededdings_num.begin() + selected_path_index);
        }
    }

    // Order the leaves.
    while (!leaves.empty()) {
        double min_value = std::numeric_limits<double>::max();
        uint32_t selected_leaf_index = 0;

        for (uint32_t i = 0; i < leaves.size(); ++i) {
            VertexID vertex = leaves[i];
            double cur_value = candidates_count[vertex];

            if (cur_value < min_value) {
                min_value = cur_value;
                selected_leaf_index = i;
            }
        }

        if (!visited_vertices[leaves[selected_leaf_index]]) {
            order[selected_vertices_count] = leaves[selected_leaf_index];
            selected_vertices_count += 1;
            visited_vertices[leaves[selected_leaf_index]] = true;
        }
        leaves.erase(leaves.begin() + selected_leaf_index);
    }

    delete[] tree;
}

// For DAF order. 
void computeWeightArray(const Graph& data, const Query& query, const CandidateParam& canParam, Order& bfs_order, uint32_t **&weight_array) {
    uint32_t qvcnt = query.getVertexCnt();
    
    bfs_order.clear();

    VertexID start_vertex = DAFStartVertex(data, query);

    TreeNode *tree = nullptr;
    BFS(query, start_vertex, tree, bfs_order);

    // // Compute weight array.
    // weight_array = new uint32_t*[qvcnt];
    // for (uint32_t i = 0; i < qvcnt; ++i) {
    //     weight_array[i] = new uint32_t[canParam.candidates_count[i]];
    //     std::fill(weight_array[i], weight_array[i] + candidates_count[i], std::numeric_limits<uint32_t>::max());
    // }

    PreviousNeb prevNeb[qvcnt];
    GetPreviousNeb(query, bfs_order, prevNeb);

    for (int i = qvcnt - 1; i >= 0; --i) {
        VertexID vertex = bfs_order[i];
        TreeNode& node = tree[vertex];
        bool set_to_one = true;

        for(auto child: node._children){
            // TreeNode& child_node = tree[child];
            // if (child_node.bn_count_ == 1) {
            if (prevNeb[child].size() == 1){
                set_to_one = false;
                Edges& cur_edge = *canParam.edge_matrix[vertex][child];
                for (uint32_t k = 0; k < canParam.candidates_count[vertex]; ++k) {
                    // uint32_t cur_candidates_count = cur_edge.offset_[k + 1] - cur_edge.offset_[k];
                    uint32_t cur_candidates_count = 0;
                    const uint32_t* cur_candidates = cur_edge.getNeb(k, cur_candidates_count);

                    uint32_t weight = 0;

                    for (uint32_t l = 0; l < cur_candidates_count; ++l) {
                        uint32_t candidates = cur_candidates[l];
                        weight += weight_array[child][candidates];
                    }

                    if (weight < weight_array[vertex][k])
                        weight_array[vertex][k] = weight;
                }
            }
        }

        if (set_to_one) {
            std::fill(weight_array[vertex], weight_array[vertex] + canParam.candidates_count[vertex], 1);
        }
    }

    delete[] tree;
}

void computeProbability(const Graph& data, const Query& query, double *prob){
    uint32_t data_vertices_num = data.getVertexCnt();
    uint32_t query_vertices_num = query.getVertexCnt();
    uint32_t max_degree = data.getMaxDgree();
    uint32_t* degrees = new uint32_t[max_degree + 1];
    memset(degrees, 0, (max_degree + 1)*sizeof(uint32_t));

    
    for (uint32_t i = 0; i < data_vertices_num; i++){
        degrees[data.getVertexDegree(i)]++;
    }
    for (uint32_t i = 1; i <= max_degree; i++){
        degrees[i] += degrees[i-1];
    }
    if (degrees[max_degree]!=data_vertices_num){
        std::cout << " degrees[max_degree]!=data_vertices_num " << std::endl;
    }

    for (uint32_t i=0;i<query_vertices_num;i++) {
        uint32_t degree = query.getVertexDegree(i);
        uint32_t label_num = data.getLabelFreq(query.getVertexLabel(i));
        uint32_t degree_num = degree > max_degree ? 
                        0 
                        : data_vertices_num - (degree != 0 ? 
                                               degrees[degree - 1] 
                                               : 0);
        prob[i] = ((double)label_num / (double)data_vertices_num)
                    * ((double)degree_num / (double)data_vertices_num);
    }

    delete[] degrees;
}

void Order_VF3(const Graph& data, const Query& query, Order &order) {
    uint32_t query_vertices_num = query.getVertexCnt();

    order.clear();
    order.resize(query_vertices_num, 0);
    
    double* prob = new double[query_vertices_num];
    computeProbability(data, query, prob);
    
    std::vector<std::vector<double>> properties(query_vertices_num);
    std::vector<bool> order_type(query_vertices_num, true);     
    std::vector<uint32_t> vertices(query_vertices_num);

    for (uint32_t i = 0; i < query_vertices_num; ++i) {
        vertices[i] = i;              
        properties[i].resize(3);      
        properties[i][0] = 0;         
        properties[i][1] = prob[i];   
        properties[i][2] = data.getVertexDegree(query.getVertexLabel(i)); 
    }
    
    order_type[0] = false;
    order_type[1] = true;
    order_type[2] = false;
    auto order_lambda = [&properties, &order_type](uint32_t l, uint32_t r) -> bool {
        for (uint32_t x = 0; x < 3; ++x) {
            if (properties[l][x] == properties[r][x])
                continue;
            if (order_type[x]) {
                return properties[l][x] < properties[r][x];
            }
            else {
                return properties[l][x] > properties[r][x];
            }
        }
        return l < r;
    };

    
    std::sort(vertices.begin(),vertices.end(),order_lambda);
    order[0] = vertices[0];
    vertices.erase(vertices.begin());


    
    for (uint32_t i = 1; i < query_vertices_num; i++) {
        
        uint32_t un_count;
        const uint32_t *un = query.getNeb(order[i - 1], un_count);
        for (uint32_t j = 0; j < un_count; j++) {
            properties[un[j]][0] += 1; 
        }

        
        std::sort(vertices.begin(), vertices.end(), order_lambda);
        order[i] = vertices[0];
        vertices.erase(vertices.begin());

        
        for (uint32_t j = 0; j < i; ++j) {
            if (query.hasEdge(order[i], order[j])) {
                break;
            }
        }
    }

    delete[] prob;
}

void Order_RI(const Graph &query, Order& order) {
    uint32_t query_vertices_num = query.getVertexCnt();
    // order = new uint32_t[query_vertices_num];
    order.clear();
    order.resize(query_vertices_num, 0);

    std::vector<bool> visited(query_vertices_num, false);
    
    order[0] = 0;
    for (uint32_t i = 1; i < query_vertices_num; ++i) {
        if (query.getVertexDegree(i) > query.getVertexDegree(order[0])) {
            order[0] = i;
        }
    }
    visited[order[0]] = true;
    
    std::vector<uint32_t> tie_vertices;
    std::vector<uint32_t> temp;

    for (uint32_t i = 1; i < query_vertices_num; ++i) {
        
        uint32_t max_bn = 0;
        for (uint32_t u = 0; u < query_vertices_num; ++u) {
            if (!visited[u]) {
                
                uint32_t cur_bn = 0;
                for (uint32_t j = 0; j < i; ++j) {
                    uint32_t uu = order[j];
                    if (query.hasEdge(u, uu)) {
                        cur_bn += 1;
                    }
                }

                
                if (cur_bn > max_bn) {
                    max_bn = cur_bn;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                } else if (cur_bn == max_bn) {
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            uint32_t count = 0;
            std::vector<uint32_t> u_fn;
            for (auto u : temp) {
                uint32_t un_count;
                const uint32_t* un = query.getNeb(u, un_count);
                for (uint32_t j = 0; j < un_count; ++j) {
                    if (!visited[un[j]]) {
                        u_fn.push_back(un[j]);
                    }
                }
                
                uint32_t cur_count = 0;
                for (uint32_t j = 0; j < i; ++j) {
                    uint32_t uu = order[j];
                    uint32_t uun_count;
                    const uint32_t* uun = query.getNeb(uu, uun_count);
                    uint32_t common_neighbor_count = 0;
                    uint32_t tmp_arr[uun_count];
                    Intersection(uun, uun_count, u_fn.data(), (uint32_t)u_fn.size(), tmp_arr, common_neighbor_count);
                    if (common_neighbor_count > 0) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();
                
                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count){
                    tie_vertices.push_back(u);
                }
            }
        }

        if (tie_vertices.size() != 1) {
            temp.swap(tie_vertices);
            tie_vertices.clear();

            uint32_t count = 0;
            std::vector<uint32_t> u_fn;
            for (auto u : temp) {
                
                uint32_t un_count;
                const uint32_t* un = query.getNeb(u, un_count);
                for (uint32_t j = 0; j < un_count; ++j) {
                    if (!visited[un[j]]) {
                        u_fn.push_back(un[j]);
                    }
                }

                uint32_t cur_count = 0;
                for (auto uu : u_fn) {
                    bool valid = true;

                    for (uint32_t j = 0; j < i; ++j) {
                        if (query.hasEdge(uu, order[j])) {
                            valid = false;
                            break;
                        }
                    }

                    if (valid) {
                        cur_count += 1;
                    }
                }

                u_fn.clear();

                
                if (cur_count > count) {
                    count = cur_count;
                    tie_vertices.clear();
                    tie_vertices.push_back(u);
                }
                else if (cur_count == count){
                    tie_vertices.push_back(u);
                }
            }
        }

        order[i] = tie_vertices[0];

        visited[order[i]] = true;
        for (uint32_t j = 0; j < i; ++j) {
            if (query.hasEdge(order[i], order[j])) {
                break;
            }
        }

        tie_vertices.clear();
        temp.clear();
    }
}

/*
    GPM's cost model based order generation
*/

size_t GetTriCnt(const Graph& data){

    size_t tri_cnt = 0;

    VertexSet *vsets = new VertexSet[3];
    for (int i = 0; i < 3; i++)
        vsets[i].alloc(data.getMaxDgree());

    for(VertexID v0=0; v0<data.getVertexCnt(); v0++){
        uint32_t v0_neb_cnt;
        const VertexID* v0_neb = data.getNeb(v0, v0_neb_cnt);
        vsets[0].Init(const_cast<VertexID*>(v0_neb), v0_neb_cnt);
        for(uint32_t v1_id = 0; v1_id < v0_neb_cnt; v1_id++){
            uint32_t v1_neb_cnt;
            VertexID v1 = vsets[0].getData(v1_id);
            const VertexID* v1_neb = data.getNeb(v1, v1_neb_cnt);
            vsets[1].Init(const_cast<VertexID*>(v1_neb), v1_neb_cnt);
            if(v1 <= v0) continue;
            vsets[2].IntersecOf(vsets[0], vsets[1]);
            for(uint32_t v2_id = 0; v2_id < vsets[2].getSize(); v2_id++){
                VertexID v2 = vsets[2].getData(v2_id);
                if(v2 <= v1 || v2 == v0) continue;
                tri_cnt++;
            }
        }
    }

    delete[] vsets;

    return tri_cnt;
}

double GetPlocal(const Graph& data, int sample_cnt, int alpha){
    std::random_device rd;
    std::default_random_engine rand(rd());
    std::uniform_int_distribution<int> dist(0, data.getVertexCnt()-1);

    double plocal = 0.;
    for(int i=0; i < sample_cnt; i++){
        VertexID startVertex = dist(rand);
        size_t res=0;
        GetPairsCnt(data, startVertex, alpha, res);
        double _plocal = res * 1.0 / data.getVertexCnt();  // _plocal = d(startVertex,v)<8 / all (startVertex,v)
        plocal = ((i * plocal) + _plocal) / (i+1);
    }

    return plocal;
}

// Graphpi's 2-phase order elimination
std::vector<Order> orderReduce(const Query &query, const std::vector<Order> &orders, int iso_v_idx){
    
    int size = orders[0].size();
    std::vector<Order> ret;

    for(auto o: orders){
        int isValid_phase1 = 1;
        int isValid_phase2 = 1;
        // Phase 1: connectivity verify
        for(int i=1; i<size; i++){
            VertexID u = o[i];
            int isConnect = 0;
            for(int j=0; j<i; j++){
                VertexID upre = o[j];
                if(query.hasEdge(upre, u))
                    isConnect = 1;
                if(isConnect) break;
            }
            if(isConnect == 0){
                isValid_phase1 = 0;
                break;
            }
        }

        // Phase 2: Verify if innermost k-vertices are independent
        if(isValid_phase1){

            for(int i=iso_v_idx; i<size; i++){
                VertexID u = o[i];
                int isConnect=0;
                for(int j=i+1; j<size; j++){
                    VertexID v = o[j];
                    if(query.hasEdge(u,v))
                        isConnect = 1;
                    if(isConnect){
                        isValid_phase2 = 0;
                        break;
                    }
                }
                if(isValid_phase2 == 0)
                    break;
            } 
        }

        if(isValid_phase1 && isValid_phase2)
            ret.push_back(o);
    }

    return ret;

}

// min_val is for early exit
// Automine/Graphzero model
double AZ_Model(const Graph &data, const Query &query, const Order &order, double min_val){

    double ret=1.0;

    double p = data.getEdgeCnt() * 1.0 / data.getVertexCnt() * data.getVertexCnt();
    double np_pow[query.getMaxDgree()];
    double _p_pow[query.getMaxDgree()];
    np_pow[0] = data.getVertexCnt();
    _p_pow[0] = 1.0;
    for(int i=1; i<query.getMaxDgree(); i++){
        np_pow[i] = np_pow[i-1] * p;
        _p_pow[i] = _p_pow[i-1] * (1-p);
    }

    PreviousNeb *prevNeb = new PreviousNeb[query.getVertexCnt()];

    GetPreviousNeb(query, order, prevNeb);

    for(int i=0; i<order.size(); i++){
        VertexID u = order[i];
        // ret *= (np_pow[prevNeb[u].size()] * _p_pow[i - prevNeb[u].size()]);
        ret *= np_pow[prevNeb[u].size()];
        if(ret > min_val){
            delete[] prevNeb;
            return std::numeric_limits<double>::max();
        }
    }

    delete[] prevNeb;

    return ret;
}

// Decomine naive model
double DM_Model(const Graph &data, const Query &query, const Order &order, double plocal, int alpha, double min_val){

    int qv_cnt = query.getVertexCnt();
    uint32_t **khop_map = new uint32_t* [qv_cnt];
    for(int i=0; i<qv_cnt; i++){
        khop_map[i] = new uint32_t[qv_cnt];
        memset(khop_map[i], -1, sizeof(int) * qv_cnt);
    }

    GetAllPairDistance(query, khop_map);

    int n = data.getVertexCnt();

    double p = data.getEdgeCnt() * 1.0 / data.getVertexCnt() * data.getVertexCnt();
    double p_pow[query.getMaxDgree()];
    double plocal_pow[query.getMaxDgree()];
    p_pow[0] = 1.0;
    plocal_pow[0] = 1.0;
    for(int i=1; i<query.getMaxDgree(); i++){
        p_pow[i] = p_pow[i-1] * p;
        plocal_pow[i] = plocal_pow[i-1] * plocal;
    }
    

    PreviousNeb *prevNeb = new PreviousNeb[qv_cnt];

    GetPreviousNeb(query, order, prevNeb);

    double ret=1.0;

    for(VertexID u: order){

        if(prevNeb[u].size() >= 2){
            int plocal_cnt = 0;
            int p_cnt = 0;
            VertexID s = prevNeb[u][0];
            for(VertexID t: prevNeb[u]){
                if(s==t) continue;
                if(khop_map[s][t] + 1 < alpha) plocal_cnt++;
                else p_cnt++;
            }
            
            ret *= (n * p_pow[1] * plocal_pow[plocal_cnt] * p_pow[p_cnt]);
        }
        else
            ret *= (n*p_pow[prevNeb[u].size()]);

        
        if(ret > min_val){
            delete[] prevNeb;
            return std::numeric_limits<double>::max();
        }
    }


    for(int i=0; i<qv_cnt; i++)
        delete[] khop_map[i];
    delete[] khop_map;

    delete[] prevNeb;

    return ret;
}

double GP_Model_Plain(const Graph &data, const Query &query, const Order &order, const Restriction& rest, size_t tri_cnt, double min_val) {
    int max_degree = query.getMaxDgree();

    double p_size[max_degree];
    double pp_size[max_degree];

    uint32_t e_cnt = 2 * data.getEdgeCnt();
    uint32_t v_cnt = data.getVertexCnt();

    uint32_t size = query.getVertexCnt();

    double p0 = e_cnt * 1.0 / v_cnt / v_cnt;
    double p1 = tri_cnt * 1.0 * v_cnt / e_cnt / e_cnt; 

    // printf("vcnt: %d, ecnt: %d, p0: %.6f, p1: %.6f\n", v_cnt, e_cnt, p0, p1);

    std::vector<uint32_t> vidx_in_order(order.size());
    for(int i=0; i<order.size(); i++)
        vidx_in_order[order[i]] = i;
    
    p_size[0] = v_cnt;
    for(int i = 1;i < max_degree; ++i) {
        p_size[i] = p_size[i-1] * p0;
    }
    pp_size[0] = 1;
    for(int i = 1; i < max_degree; ++i) {
        pp_size[i] = pp_size[i-1] * p1;
    }
    
    std::vector<int> invariant_size[size];
    for(int i = 0; i < size; ++i) invariant_size[i].clear();
    
    // prinf order

    // printf("order: ");
    // for(int i = 0; i < size; ++i) {
    //     printf("%d ", order[i]);
    // }
    // printf("\n");

    PreviousNeb prevNeb[size];
    GetPreviousNeb(query, order, prevNeb);


    double val = 1;
    for(int i = size - 1; i >= 0; --i) {
        int cnt_forward = prevNeb[order[i]].size();

        if(i < size-1){
            val += p_size[1] * prevNeb[order[i+1]].size();
        }
        
        // val += 1;

        if( i ) {
            val *= p_size[1] * pp_size[ cnt_forward - 1 ];
        }
        else {
            val *= p_size[0];
        }

        // printf("depth: %d, cost: %.6f, cnt_forward: %d", i, val, cnt_forward);
        // printf("\n");
    }

    return val;
}

double GP_Model_Promot(const Graph &data, const Query &query, const Order &order, const Restriction& rest, size_t tri_cnt, double min_val) {
    int max_degree = query.getMaxDgree();

    double p_size[max_degree];
    double pp_size[max_degree];

    uint32_t e_cnt = 2 * data.getEdgeCnt();
    uint32_t v_cnt = data.getVertexCnt();

    uint32_t size = query.getVertexCnt();

    double p0 = e_cnt * 1.0 / v_cnt / v_cnt;
    double p1 = tri_cnt * 1.0 * v_cnt / e_cnt / e_cnt; 

    // printf("vcnt: %d, ecnt: %d, p0: %.6f, p1: %.6f\n", v_cnt, e_cnt, p0, p1);

    std::vector<uint32_t> vidx_in_order(order.size());
    for(int i=0; i<order.size(); i++)
        vidx_in_order[order[i]] = i;
    
    p_size[0] = v_cnt;
    for(int i = 1;i < max_degree; ++i) {
        p_size[i] = p_size[i-1] * p0;
    }
    pp_size[0] = 1;
    for(int i = 1; i < max_degree; ++i) {
        pp_size[i] = pp_size[i-1] * p1;
    }
    
    std::vector<int> invariant_size[size];
    for(int i = 0; i < size; ++i) invariant_size[i].clear();
    
    // prinf order

    // printf("order: ");
    // for(int i = 0; i < size; ++i) {
    //     printf("%d ", order[i]);
    // }
    // printf("\n");

    PlanIR plan(order, query);
    const auto &opsByDeps = plan.getSetOps();
    const auto &opsArr = plan.getTotSetOpsArr();

    PreviousNeb prevNeb[size];
    GetPreviousNeb(query, order, prevNeb);


    double val = 1;
    for(int i = size - 1; i >= 0; --i) {


        for(auto vsetId: opsByDeps[i]){
            int sub_exp_cnt = opsArr[vsetId].getEdgeIdx().size();
            if(sub_exp_cnt > 1){
                val += p_size[1] * pp_size[sub_exp_cnt - 2] + p_size[1];
            }
        }
                

        // for(int j = 0; j < invariant_size[i].size(); ++j)
        //     if(invariant_size[i][j] > 1) 
        //         val += p_size[1] * pp_size[invariant_size[i][j] - 2] + p_size[1];
        val += 1;
        if( i ) {
            val *= p_size[1] * pp_size[ opsArr[plan.getIterVSetAt(i)].getEdgeIdx().size() - 1 ];
        }
        else {
            val *= p_size[0];
        }

        // printf("depth: %d, cost: %.6f, Invariant: ", i, val);
        // for(auto vsetId: opsByDeps[i]){
        //     int sub_exp_cnt = opsArr[vsetId].getEdgeIdx().size();
        //     printf("%d ", sub_exp_cnt);
        // }
        // printf("\n");
    }

    return val;
}

double GP_Model(const Graph &data, const Query &query, const Order &order, const Restriction& rest, size_t tri_cnt, double min_val, bool useCodePromotion = true){
    if (useCodePromotion)
        return GP_Model_Promot(data, query, order, rest, tri_cnt, min_val);
    else
        return GP_Model_Plain(data, query, order, rest, tri_cnt, min_val);

}

// double GP_Model(const Graph &data, const Query &query, const Order &order, const Restriction& rest, size_t tri_cnt, double min_val, bool useCodePromotion = true){

    
//     double p1 = data.getEdgeCnt() * 2.0 / (data.getVertexCnt() * data.getVertexCnt());
//     double p2 = (tri_cnt * 1.0 * data.getVertexCnt()) / (4.0 * data.getEdgeCnt() * data.getEdgeCnt());
//     // double p1 = data.getEdgeCnt() * 1.0 / (1.0 * data.getVertexCnt() ) /(1.0 * data.getVertexCnt());
//     // double p2 = (tri_cnt * 1.0 * data.getVertexCnt()) / (1.0 * data.getEdgeCnt() )/ (1.0 * data.getEdgeCnt());
//     // cout << fmt::format("tri_cnt: {}, p1: {}, p2: {}", tri_cnt, p1, p2)<<"\n";

//     double np1 = data.getVertexCnt() * p1;

//     double p2_pow[query.getMaxDgree()];
//     p2_pow[0] = 1;
//     for(int i=1; i<query.getMaxDgree();i++)
//         p2_pow[i] = p2_pow[i-1] * p2;

//     uint32_t size = query.getVertexCnt();

//     // print order
//     std::cout<<"order: ";
//     for(int i=0; i<size; i++)
//         std::cout<<order[i]<<" ";
//     std::cout<<'\n';

//     PlanIR plan(order, query);
//     const auto &opsByDeps = plan.getSetOps();
//     const auto &opsArr = plan.getTotSetOpsArr();
//     // std::cout<<plan.debugOutput()<<'\n';

//     // Todo: restriction filter
//     // f_i here is the (1-f_i) term in the paper.

//     // size_t rest_filter_cnt[rest.size()];
//     // for(int i=0; i<rest.size(); i++) rest_filter_cnt[i] = 0;

//     // size_t tot = 1;
//     // for(int i=0; i<size; i++){
//     //     tot *= (i+1);
        
//     // }
//     double sum[rest.size()];
//     for(int i=0; i<rest.size(); i++) sum[i]=1.0;

//     vector<double> f_i(size, 1.0);
//     // for(int i = size - 1; i >= 0; --i){
//     //     for(int j = 0; j < rest.size(); ++j)
//     //         if(rest[j].second == order[i])
//     //             f_i[order[i]] *= sum[j];
//     // }

//     double cost_i[size];
//     for(int i=0; i<size; i++) cost_i[i] = 1.0;
    
//     PreviousNeb *preNeb = new PreviousNeb[size];

//     GetPreviousNeb(query, order, preNeb);

//     std::vector<uint32_t> vidx_in_order(order.size());
//     for(int i=0; i<order.size(); i++)
//         vidx_in_order[order[i]] = i;

//     // when use code promotion and common subexpression elimination.
//     // std::vector<int> intermediate_res[size];

//     std::vector<std::vector<int>> inter_res;
//     inter_res.resize(size);
//     for(int i=0; i<size; i++) inter_res[i].clear();

//     for(int i=size-1; i>=0; i--){

//         // for(auto it = preNeb[order[i]].rbegin(); it != preNeb[order[i]].rend(); it++){
//         //     VertexID upre = *it;
//         //     inter_res[vidx_in_order[upre]].push_back(c--);
//         // }
//         // for(int j=i-1; j>=0; j--)
//         //     if(query.hasEdge(j,i))
//         //         inter_res[j].push_back(c--);

//         if(i == size-1){
//             auto preCnt = opsArr[plan.getIterVSetAt(i)].getEdgeIdx().size();
//             double l_i = np1 * p2_pow[ preCnt - 1 ];
//             // cost_i[i] = f_i[i] * l_i;
//             cost_i[i] = f_i[i] * l_i * 2; 
//         }
//         else{
//             auto preCnt = 0;
//             if(i) preCnt = opsArr[plan.getIterVSetAt(i)].getEdgeIdx().size();
//             double l_i = 0.0, c_i = 0.0;
//             if(i) l_i = np1 * p2_pow[ preCnt - 1 ];
//             else l_i = data.getVertexCnt() * 1.0;
//             if(useCodePromotion){
//                 for(auto vsetId: opsByDeps[i]){
//                     // int subexpression_len = ;
//                     uint32_t pvsetId = opsArr[vsetId].getParentVSetID();
//                     if(pvsetId != INVALID){
//                         int subexpression_len = opsArr[pvsetId].getEdgeIdx().size();
//                         if(subexpression_len > 1) c_i += (np1*p2_pow[subexpression_len - 2] + np1); // |N(v1)| + |N(v2)| or |N(v1) \cup N(v2)| + |N(v3)|
//                     }
//                 }
//                 // for(int j=0; j<inter_res[i].size(); j++){
//                 //     int subexpression_len = inter_res[i][j];
//                 //     if(subexpression_len > 1) c_i += (np1*p2_pow[subexpression_len - 2] + np1); // |N(v1)| + |N(v2)| or |N(v1) \cup N(v2)| + |N(v3)|
//                 // }   
//             }
//             else{
//                 c_i = preCnt * np1; // |N(v1)| + |N(v2)| + ... + |N(v_k)|
//             }
//             // cost_i[i] = f_i[i] * l_i * (c_i + cost_i[i+1]);
//             cost_i[i] = f_i[i] * l_i * (c_i + cost_i[i+1] + 1);
//         }

//         //depth: 0, cost: 3047484442.409045, inter_res: 1 1 1 1 1 
//         std::cout<< fmt::format("depth: {}, cost: {}, inter_res: {}", i, cost_i[i], fmt::join(opsByDeps[i], " "))<<"\n";
//     }

//     // for(int i=size-1; i>=0; i--){
//     //     std::cout << "depth "<< i << ":" << cost_i[i]<<"\n";
//     // }

//     delete[] preNeb;

//     return cost_i[0];
// }

// // Graphpi/Graphset model
// double GP_Model(const Graph &data, const Query &query, const Order &order, const Restriction& rest, size_t tri_cnt, double min_val, bool useCodePromotion = true){

    
//     double p1 = data.getEdgeCnt() * 2.0 / (data.getVertexCnt() * data.getVertexCnt());
//     double p2 = (tri_cnt * 1.0 * data.getVertexCnt()) / (4.0 * data.getEdgeCnt() * data.getEdgeCnt());
//     // double p1 = data.getEdgeCnt() * 1.0 / (1.0 * data.getVertexCnt() ) /(1.0 * data.getVertexCnt());
//     // double p2 = (tri_cnt * 1.0 * data.getVertexCnt()) / (1.0 * data.getEdgeCnt() )/ (1.0 * data.getEdgeCnt());
//     // cout << fmt::format("tri_cnt: {}, p1: {}, p2: {}", tri_cnt, p1, p2)<<"\n";

//     double np1 = data.getVertexCnt() * p1;

//     double p2_pow[query.getMaxDgree()];
//     p2_pow[0] = 1;
//     for(int i=1; i<query.getMaxDgree();i++)
//         p2_pow[i] = p2_pow[i-1] * p2;

//     uint32_t size = query.getVertexCnt();

//     // print order
//     std::cout<<"order: ";
//     for(int i=0; i<size; i++)
//         std::cout<<order[i]<<" ";
//     std::cout<<'\n';

//     // Todo: restriction filter
//     // f_i here is the (1-f_i) term in the paper.

//     // size_t rest_filter_cnt[rest.size()];
//     // for(int i=0; i<rest.size(); i++) rest_filter_cnt[i] = 0;

//     // size_t tot = 1;
//     // for(int i=0; i<size; i++){
//     //     tot *= (i+1);
        
//     // }
//     double sum[rest.size()];
//     for(int i=0; i<rest.size(); i++) sum[i]=1.0;

//     vector<double> f_i(size, 1.0);
//     // for(int i = size - 1; i >= 0; --i){
//     //     for(int j = 0; j < rest.size(); ++j)
//     //         if(rest[j].second == order[i])
//     //             f_i[order[i]] *= sum[j];
//     // }

//     double cost_i[size];
//     for(int i=0; i<size; i++) cost_i[i] = 1.0;
    
//     PreviousNeb *preNeb = new PreviousNeb[size];

//     GetPreviousNeb(query, order, preNeb);

//     std::vector<uint32_t> vidx_in_order(order.size());
//     for(int i=0; i<order.size(); i++)
//         vidx_in_order[order[i]] = i;

//     // when use code promotion and common subexpression elimination.
//     // std::vector<int> intermediate_res[size];

//     std::vector<std::vector<int>> inter_res;
//     inter_res.resize(size);
//     for(int i=0; i<size; i++) inter_res[i].clear();

//     for(int i=size-1; i>=0; i--){

//         int preCnt = preNeb[order[i]].size();
//         int c = preCnt;

//         for(auto it = preNeb[order[i]].rbegin(); it != preNeb[order[i]].rend(); it++){
//             VertexID upre = *it;
//             inter_res[vidx_in_order[upre]].push_back(c--);
//         }
//         // for(int j=i-1; j>=0; j--)
//         //     if(query.hasEdge(j,i))
//         //         inter_res[j].push_back(c--);

//         if(i == size-1){
//             double l_i = np1 * p2_pow[ preCnt - 1 ];
//             // cost_i[i] = f_i[i] * l_i;
//             cost_i[i] = f_i[i] * l_i * 2; 
//         }
//         else{
//             double l_i = 0.0, c_i = 0.0;
//             if(i) l_i = np1 * p2_pow[ preCnt - 1 ];
//             else l_i = data.getVertexCnt() * 1.0;
//             if(useCodePromotion){
//                 for(int j=0; j<inter_res[i].size(); j++){
//                     int subexpression_len = inter_res[i][j];
//                     if(subexpression_len > 1) c_i += (np1*p2_pow[subexpression_len - 2] + np1); // |N(v1)| + |N(v2)| or |N(v1) \cup N(v2)| + |N(v3)|
//                 }   
//             }
//             else{
//                 c_i = preCnt * np1; // |N(v1)| + |N(v2)| + ... + |N(v_k)|
//             }
//             // cost_i[i] = f_i[i] * l_i * (c_i + cost_i[i+1]);
//             cost_i[i] = f_i[i] * l_i * (c_i + cost_i[i+1] + 1);
//         }

//         //depth: 0, cost: 3047484442.409045, inter_res: 1 1 1 1 1 
//         std::cout<< fmt::format("depth: {}, cost: {}, inter_res: {}", i, cost_i[i], fmt::join(inter_res[i], " "))<<"\n";
//     }

//     // for(int i=size-1; i>=0; i--){
//     //     std::cout << "depth "<< i << ":" << cost_i[i]<<"\n";
//     // }

//     delete[] preNeb;

//     return cost_i[0];
// }

Order Sample_Single_Order(const Query &query, const Order& _iso, uint32_t iso_v_idx){

    Order order;
    order.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, iso_v_idx - 1);

    uint32_t qvcnt = query.getVertexCnt();
    bool *vis = new bool[qvcnt];
    memset(vis, 0, sizeof(bool) * qvcnt);
    // uint32_t *deps_by_idx = new uint32_t[qvcnt];

    // for(int i=0; i<qvcnt; i++)
    //     deps_by_idx[_iso[i]] = i;
    
    for(int i=iso_v_idx; i<qvcnt; i++)
        vis[_iso[i]] = true;

    VertexID startVertex = _iso[dist(gen)];
    order.push_back(startVertex);
    vis[startVertex] = true;

    while (order.size() < iso_v_idx)
    {
        std::uniform_int_distribution<> _dist(0, order.size() - 1);

        VertexID v = order[_dist(gen)];
        uint32_t nebs_cnt;
        const VertexID* nebs = query.getNeb(v, nebs_cnt);
        if(nebs_cnt == 0) continue;
        
        std::vector<VertexID> unvisited_neb;
        unvisited_neb.clear();
        for(int i=0; i<nebs_cnt; i++){
            VertexID vneb = nebs[i];
            if(vis[vneb]==false) unvisited_neb.push_back(vneb);
        }

        if(unvisited_neb.size()==0) continue;

        std::uniform_int_distribution<> _neb_dist(0, unvisited_neb.size() - 1);
        VertexID v_next = unvisited_neb[_neb_dist(gen)];
        order.push_back(v_next);
        vis[v_next] = true;
    }

    for(int i=iso_v_idx; i<qvcnt; i++)
        order.push_back(_iso[i]);
    

    delete[] vis;
    // delete[] deps_by_idx;

    return order;
}

std::vector<Order> Sample_Orders(const Graph &data, const Query &query, const Order& _iso, uint32_t iso_v_idx, uint32_t sample_cnt = 2e4, uint32_t max_fail = 1e3){

    std::vector<Order> orders;
    orders.clear();
    uint32_t fail_cnt = 0;
    while (orders.size() < sample_cnt)
    {
        auto o = Sample_Single_Order(query, _iso, iso_v_idx);
        // auto o = Sample_Single_Order(query, _iso, query.getVertexCnt());
        if(std::find(orders.begin(), orders.end(), o) == orders.end())
            orders.push_back(o);
        else{
            fail_cnt ++;
            if(fail_cnt >= max_fail)
                break;
        }
    }

    return orders;
}

void Order_GraphZero(const Graph &data, const Query &query, Order &order){

    vector<Order> orders;
    Order _iso;
    uint32_t k;
    getMaxIsolated(query, _iso, k);

    if(query.getVertexCnt() < 13){
        vector<Order> _orders = orderGenerate(query);
        orders = orderReduce(query, _orders, k);
    }
    else{
        orders = Sample_Orders(data, query, _iso, k, 2e4, 1e3);
    }

    Order tmp;
    Order_VF2PP(data, query, tmp);
    orders.push_back(tmp);
    Order_VF3(data, query, tmp);
    orders.push_back(tmp);
    Order_RI(query, tmp);
    orders.push_back(tmp);
     
    double minCost = std::numeric_limits<double>::max();
    int min_index = 0;
    for(int i=0; i<orders.size(); i++){
        Order _order = orders[i];
        double cost = AZ_Model(data, query, _order, minCost);
        if(cost < minCost){
            min_index = i;
            minCost = cost;
        }
    }

    order = orders[min_index];
}

void Order_DecoMine(const Graph &data, const Query &query, Order &order){

    vector<Order> orders;
    Order _iso;
    uint32_t k;
    getMaxIsolated(query, _iso, k);

    if(query.getVertexCnt() < 13){
        vector<Order> _orders = orderGenerate(query);
        orders = orderReduce(query, _orders, k);
    }
    else{
        printf("for query with vertices > 13, the permutation methods is too slow, advanced methods not implemented yet\n");
        exit(-1);
    }

    int sample_cnt = 0.1 * data.getVertexCnt();
    int alpha = 8;
    double plocal = GetPlocal(data, sample_cnt, alpha);

    double minCost = std::numeric_limits<double>::max();
    int min_index = 0;
    for(int i=0; i<orders.size(); i++){
        Order _order = orders[i];
        double cost = DM_Model(data, query, _order, plocal, alpha, minCost);
        if(cost < minCost){
            min_index = i;
            minCost = cost;
        }
    }

    order = orders[min_index];
}

void Order_GraphPi(const Graph &data, const Query &query, Order &order, bool usePromotion = true){
    vector<Order> orders;
    Order _iso;
    uint32_t k;
    getMaxIsolated(query, _iso, k);

    if(query.getVertexCnt() < 13){
        vector<Order> _orders = orderGenerate(query);
        orders = orderReduce(query, _orders, k);
    }
    else{
        orders = Sample_Orders(data, query, _iso, k, 5e4, 1e3);
    }

    Order tmp;
    Order_VF2PP(data, query, tmp);
    orders.push_back(tmp);
    Order_VF3(data, query, tmp);
    orders.push_back(tmp);
    Order_RI(query, tmp);
    orders.push_back(tmp);

    Restriction rest;
    rest.clear();

    // this is accepteble since the preprocessing time in the largest dataset is no longer than 10s
    size_t tri_cnt = GetTriCnt(data);

    double minCost = std::numeric_limits<double>::max();
    int min_index = 0;
    for(int i=0; i<orders.size(); i++){
        // std::cout<< "check : " << i << "\n";
        Order _order = orders[i];
        double cost = GP_Model(data, query, _order, rest, tri_cnt, minCost, usePromotion);
        if(cost < minCost){
            min_index = i;
            minCost = cost;
        }
    }

    order = orders[min_index];
}