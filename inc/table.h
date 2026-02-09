#pragma once
#include <vector>
#include <algorithm>
#include "graph.h"
#include "api.h"

void buildTables(const Graph *data_graph, const Graph *query_graph, uint32_t **candidates, uint32_t *candidates_count, Edges ***edge_matrix, uint32_t* order) {
    uint32_t query_vertices_num = query_graph->getVertexCnt();
    uint32_t* flag = new uint32_t[data_graph->getVertexCnt()];
    uint32_t* updated_flag = new uint32_t[data_graph->getVertexCnt()];
    std::fill(flag, flag + data_graph->getVertexCnt(), 0);

    for (uint32_t i = 0; i < query_vertices_num; ++i)
        for (uint32_t j = 0; j < query_vertices_num; ++j) 
            edge_matrix[i][j] = NULL;
    

    std::vector<VertexID> build_table_order(query_vertices_num);
    for (uint32_t i = 0; i < query_vertices_num; ++i) 
        build_table_order[i] = i;
    

    std::sort(build_table_order.begin(), build_table_order.end(), [query_graph](VertexID l, VertexID r) {
        if (query_graph->getVertexDegree(l) == query_graph->getVertexDegree(r)) {
            return l < r;
        }
        return query_graph->getVertexDegree(l) > query_graph->getVertexDegree(r);
    });

    std::vector<uint32_t> temp_edges(data_graph->getEdgeCnt() * 2);
    std::vector<uint32_t> temp_edges_v(data_graph->getEdgeCnt() * 2);

    for (auto u : build_table_order) {
        uint32_t u_nbrs_count;
        const VertexID* u_nbrs = query_graph->getNeb(u, u_nbrs_count);

        uint32_t updated_flag_count = 0;

        for (uint32_t i = 0; i < u_nbrs_count; ++i) {
            VertexID u_nbr = u_nbrs[i];

            if (edge_matrix[u][u_nbr] != NULL)
                continue;

            if (updated_flag_count == 0) {
                for (uint32_t j = 0; j < candidates_count[u]; ++j) {
                    VertexID v = candidates[u][j];
                    flag[v] = j + 1;
                    updated_flag[updated_flag_count++] = v;
                }
            }

            edge_matrix[u_nbr][u] = new Edges;
            edge_matrix[u_nbr][u]->_v_cnt = candidates_count[u_nbr];
            edge_matrix[u_nbr][u]->_offset = new uint32_t[candidates_count[u_nbr] + 1];

            edge_matrix[u][u_nbr] = new Edges;
            edge_matrix[u][u_nbr]->_v_cnt = candidates_count[u];
            edge_matrix[u][u_nbr]->_offset = new uint32_t[candidates_count[u] + 1];
            std::fill(edge_matrix[u][u_nbr]->_offset, edge_matrix[u][u_nbr]->_offset + candidates_count[u] + 1, 0);

            uint32_t local_edge_count = 0;
            uint32_t local_max_degree = 0;

            for (uint32_t j = 0; j < candidates_count[u_nbr]; ++j) {
                VertexID v = candidates[u_nbr][j];
                edge_matrix[u_nbr][u]->_offset[j] = local_edge_count;

                uint32_t v_nbrs_count;
                const VertexID* v_nbrs = data_graph->getNeb(v, v_nbrs_count);

                uint32_t local_degree = 0;

                for (uint32_t k = 0; k < v_nbrs_count; ++k) {
                    VertexID v_nbr = v_nbrs[k];

                    if (flag[v_nbr] != 0) {
                        uint32_t position = flag[v_nbr] - 1;
                        temp_edges[local_edge_count] = position;
                        temp_edges_v[local_edge_count] = candidates[u][position];
                        local_edge_count++;
                        edge_matrix[u][u_nbr]->_offset[position + 1] += 1;
                        local_degree += 1;
                    }
                }

                if (local_degree > local_max_degree) {
                    local_max_degree = local_degree;
                }
            }

            edge_matrix[u_nbr][u]->_offset[candidates_count[u_nbr]] = local_edge_count;
            edge_matrix[u_nbr][u]->_max_degree = local_max_degree;
            edge_matrix[u_nbr][u]->_e_cnt = local_edge_count;
            edge_matrix[u_nbr][u]->_edge = new uint32_t[local_edge_count];
            edge_matrix[u_nbr][u]->_edge_v = new uint32_t[local_edge_count];
            std::copy(temp_edges.begin(), temp_edges.begin() + local_edge_count, edge_matrix[u_nbr][u]->_edge);
            std::copy(temp_edges_v.begin(), temp_edges_v.begin() + local_edge_count, edge_matrix[u_nbr][u]->_edge_v);

            edge_matrix[u][u_nbr]->_e_cnt = local_edge_count;
            edge_matrix[u][u_nbr]->_edge = new uint32_t[local_edge_count];
            edge_matrix[u][u_nbr]->_edge_v = new uint32_t[local_edge_count];

            local_max_degree = 0;
            for (uint32_t j = 1; j <= candidates_count[u]; ++j) {
                if (edge_matrix[u][u_nbr]->_offset[j] > local_max_degree) {
                    local_max_degree = edge_matrix[u][u_nbr]->_offset[j];
                }
                edge_matrix[u][u_nbr]->_offset[j] += edge_matrix[u][u_nbr]->_offset[j - 1];
            }

            edge_matrix[u][u_nbr]->_max_degree = local_max_degree;

            for (uint32_t j = 0; j < candidates_count[u_nbr]; ++j) {
                uint32_t begin = j;
                for (uint32_t k = edge_matrix[u_nbr][u]->_offset[begin]; k < edge_matrix[u_nbr][u]->_offset[begin + 1]; ++k) {
                    uint32_t end = edge_matrix[u_nbr][u]->_edge[k];

                    edge_matrix[u][u_nbr]->_edge[edge_matrix[u][u_nbr]->_offset[end]] = begin;
                    edge_matrix[u][u_nbr]->_edge_v[edge_matrix[u][u_nbr]->_offset[end]] = candidates[u_nbr][begin];
                    edge_matrix[u][u_nbr]->_offset[end]++;
                }
            }

            for (uint32_t j = candidates_count[u]; j >= 1; --j) {
                edge_matrix[u][u_nbr]->_offset[j] = edge_matrix[u][u_nbr]->_offset[j - 1];
            }
            edge_matrix[u][u_nbr]->_offset[0] = 0;
        }

        for (uint32_t i = 0; i < updated_flag_count; ++i) {
            VertexID v = updated_flag[i];
            flag[v] = 0;
        }
    }

    delete[] flag;
    delete[] updated_flag;
}