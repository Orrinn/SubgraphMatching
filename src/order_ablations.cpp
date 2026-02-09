#include <vector>
#include "graph.h"
#include <iostream>
#include "filter.h"
#include "table.h"
#include "explore.h"
#include "order.h"
#include "config.h"

#include <random>
#include "code_gen.h"
#include "api.h"
#include <future>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>

using namespace std;

vector<string> order_types = {
    "QSI",
    "RI",
    "GQL",
    "CFL",
    "CECI",
    "VF2PP",
    "VF3",
    "DAF",
    "VEQ",
    "GraphPi",
    "GraphZero"
};

vector<string> explore_types = {
    "filter",
    "cache"
};

vector<string> order_compare_filter(const Graph* data_g, const Query* query_g, string order_type, atomic<bool>& stop_flag, size_t output_limit = 1e5){

    const Graph& data = *data_g;
    const Query& query = *query_g;
    Order order;
    Config cfg;
    double time=0;
    size_t ans=0;
    double order_overhead = 0.0;
    vector<string> vec;
    vec.clear();
    Profiler::getInst().reset();
    
    CandidateParam can(data, query);
    if(Filter_GQL(data, query, can)==false)
        goto Ret;

    order.clear();
    

    if(order_type != "DAF" && order_type != "VEQ"){
        if(order_type == "GQL"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_GQL(data, query, can, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "CECI"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_CECI(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
            
        else if(order_type == "CFL"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_CFL(data, query, can, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "GraphPi"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_GraphPi(data, query, order, false);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "GraphZero"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_GraphZero(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "QSI"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_QSI(data, query, can, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "RI"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_RI(query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "VF2PP"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_VF2PP(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "VF3"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_VF3(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();     
        }
        
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else{
        Order match_order;
        if(order_type == "DAF"){
            Order bfs_order;
            uint32_t **weight_array = new uint32_t*[query.getVertexCnt()];
            for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
                weight_array[i] = new uint32_t[can.candidates_count[i]];
                std::fill(weight_array[i], weight_array[i] + can.candidates_count[i], std::numeric_limits<uint32_t>::max());
            }
            auto start = std::chrono::high_resolution_clock::now();
            computeWeightArray(data, query, can, bfs_order, weight_array);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();     

            cfg.useFilter = true;
            cfg.can = &can;

            explore_Intersection_DynamicOrder_DAF(data, query, bfs_order, match_order, cfg, weight_array, time, ans, stop_flag, output_limit);

            for (uint32_t i = 0; i < query.getVertexCnt(); ++i) {
                delete[] weight_array[i];
            }
            delete[] weight_array;
        }
        else if(order_type == "VEQ"){
            Order bfs_order;

            auto start = std::chrono::high_resolution_clock::now();
            VertexID start_vertex = DAFStartVertex(data, query);
            TreeNode *tree = nullptr;
            BFS(query, start_vertex, tree, bfs_order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();     

            cfg.useFilter = true;
            cfg.can = &can;

            explore_Intersection_DynamicOrder_VEQ(data, query, bfs_order, match_order, cfg, time, ans, stop_flag, output_limit);
            
            delete[] tree;
        }
    }

Ret:
    vec.push_back("filter");
    vec.push_back(order_type);
    vec.push_back(fmt::format("{:.2f}", order_overhead));
    vec.push_back(fmt::format("{:.2f}", time));
    vec.push_back(fmt::format("{}", ans));
    vec.push_back(fmt::format("{}", output_limit));
    vec.push_back(fmt::format("{}", Profiler::getInst().total_intersection));
    return vec; 
}

vector<string> order_compare_cache(const Graph* data_g, const Query* query_g, string order_type, atomic<bool>& stop_flag, size_t output_limit = 1e5){

    const Graph& data = *data_g;
    const Query& query = *query_g;
    Order order;
    Config cfg;
    double time=0;
    size_t ans=0;
    double order_overhead = 0.0;
    vector<string> vec;
    vec.clear();
    Profiler::getInst().reset();
    
    CandidateParam can(data, query);
    if(Filter_GQL(data, query, can)==false)
        goto Ret;

    order.clear();
    

    if(order_type != "DAF" && order_type != "VEQ"){
        if(order_type == "GQL"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_GQL(data, query, can, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "CECI"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_CECI(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
            
        else if(order_type == "CFL"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_CFL(data, query, can, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "GraphPi"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_GraphPi(data, query, order, true);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "GraphZero"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_GraphZero(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "QSI"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_QSI(data, query, can, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "RI"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_RI(query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "VF2PP"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_VF2PP(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();
        }
        else if(order_type == "VF3"){
            auto start = std::chrono::high_resolution_clock::now();
            Order_VF3(data, query, order);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            order_overhead = duration.count();     
        }


        ReuseParam reuse(query, order);
        cfg.useCache = true;
        cfg.reuse = &reuse;
        explore_Intersection_Recursive<CacheBuffer>(data, query, order, cfg, time, ans, stop_flag, output_limit);
        
    }

Ret:
    vec.push_back("cache");
    vec.push_back(order_type);
    vec.push_back(fmt::format("{:.2f}", order_overhead));
    vec.push_back(fmt::format("{:.2f}", time));
    vec.push_back(fmt::format("{}", ans));
    vec.push_back(fmt::format("{}", output_limit));
    vec.push_back(fmt::format("{}", Profiler::getInst().total_intersection));
    return vec; 
}

string run_order_with_timeout(const Graph* data_g, const Query* query_g, string order_type, string explore_type, size_t output_limit = 1e5, uint32_t timeout = 3600){

    std::atomic<bool> stop_flag(false);

    if(explore_type == "filter"){
        auto future = std::async(std::launch::async, [&]() {
            return order_compare_filter(data_g, query_g, order_type, stop_flag, output_limit);
        });

        if (future.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {
            // future.get();
            stop_flag.store(true);
            vector<string> vec;
            vec = future.get();
            
            return fmt::format("{}", fmt::join(vec, ",")); 
        }

        return fmt::format("{}", fmt::join(future.get(), ","));
    }
    else{
        auto future = std::async(std::launch::async, [&]() {
            return order_compare_cache(data_g, query_g, order_type, stop_flag, output_limit);
        });

        if (future.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {
            // future.get();
            stop_flag.store(true);
            vector<string> vec;
            vec = future.get();
            
            return fmt::format("{}", fmt::join(vec, ",")); 
        }

        return fmt::format("{}", fmt::join(future.get(), ","));
    }
}


void process_query(const Graph* data_g, string data_file, string query_name, ofstream *file, size_t output_limit = 1e5, uint32_t timeout = 3600){
    vector<string> vec;
    vec.clear();
    vec.push_back(data_file);
    vec.push_back(fmt::format("{}", query_name));
    string query_info = fmt::format("{}", fmt::join(vec, ","));

    std::cout << fmt::format("Running Query: [{}] on Data: [{}], timeout: [{}], output_limit: [{}]\n", query_name, data_file, timeout, output_limit);

    string query_path = fmt::format("{}/data/query/{}.txt", PROJECT_SOURCE_DIR, query_name);
    // std::shared_ptr<Query> query_g = make_shared<Query>();
    Query *query_g = new Query();
    query_g->Load(query_path);

    for(string explore_type: explore_types){
    // string explore_type = "cache";
        for(auto order_type: order_types){

            if((order_type == "VEQ" || order_type == "DAF") && explore_type == "cache") continue;

            cout << fmt::format("Running Order Type: [{}], Explore Type: [{}]\n", order_type, explore_type);

            string res = run_order_with_timeout(data_g, query_g, order_type, explore_type, output_limit, timeout);
            string local_record = fmt::format("{},{}\n", query_info, res);
            // printf("count: %d\n", query_g.use_count());
            *file << local_record << std::flush;
        }
    }
    
    delete query_g;
}

void output_directory_check(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        std::cout << "Output Path does not exist, creating: " << path << std::endl;
        std::filesystem::create_directories(path);
    }
}


int main(int argc, char **argv){

#ifdef LOG_OUTPUT
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
#endif


    vector<string> header({"data", "query_name", "explore", "order", "order_overhead", "time_ms", "ans", "output_limit", "intersection_cnt"});

    string heads = fmt::format("{}", fmt::join(header, ","));

    // string data_file = "citeseer";
    // int query_size = 8;
    string data_file = string(argv[1]);
    string query_name = string(argv[2]);
    // string query_file = string(argv[2]);

    string default_output_path = fmt::format("{}/res", PROJECT_SOURCE_DIR);
    string output_path = (argc > 3) ? string(argv[3]) : default_output_path;

    string file_name = extractFileName(__FILE__);
    string output_dir = fmt::format("{}/{}", output_path, file_name);
    string log_file = fmt::format("{}/{}/{}_{}.csv", output_path, file_name, data_file, query_name);
    output_directory_check(output_dir);
    ofstream file(log_file);
    if(file.is_open()){
        file << heads << "\n";
    }
    else{
        cout << "cannot create / open file at" << log_file << "\n";
        exit(-1);
    }
    string data_path = fmt::format("{}/data/data/{}.txt", PROJECT_SOURCE_DIR, data_file);
    Graph* data_g = new Graph();
    data_g->Load(data_path);

    process_query(data_g, data_file, query_name, &file, 1e5, 300);


    delete data_g;

    return 1;
}
