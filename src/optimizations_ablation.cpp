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
#include <map>
#include <future>
#include <filesystem>

using namespace std;


vector<string> filter_types = {
    "nlf",
    "gql",
    "cfl",
    "ceci",
    "daf",
    "pilos"
};

vector<string> explore_optim = {
    "naive",
    "cache",
    "filter"
};


vector<string> filter_compare(const Graph* data_g, const Query* query_g, const Order& order, string filter_type, string order_type, string explore_optim, atomic<bool>& stop_flag, std::promise<void> start_promise, size_t output_limit = 1e5){    

    std::cout << fmt::format("Using exploration [{}]\n", explore_optim);

    Config cfg;

    size_t ans=0;
    double time = 0.0;
    double prep_time_ms = 0.0;
    double avg_candidates = data_g->getVertexCnt();
    Profiler::getInst().reset();
    if(explore_optim == "cache"){
        ReuseParam reuse(*query_g, order);
        cfg.useCache = true;
        cfg.reuse = &reuse;
        start_promise.set_value();
        explore_Intersection_Recursive<CacheBuffer>(*data_g, *query_g, order, cfg, time, ans, stop_flag, output_limit);
    }
    else if(explore_optim == "naive"){
        start_promise.set_value();
        explore_Intersection_Recursive<NaiveBuffer>(*data_g, *query_g, order, cfg, time, ans, stop_flag, output_limit);
    }
    else{

        CandidateParam canParam(*data_g, *query_g);

        std::cout << fmt::format("Using filter [{}], start filtering...\n", filter_type);

        auto start = std::chrono::high_resolution_clock::now();
        if(filter_type == "nlf")
            Filter_NLF(*data_g, *query_g, canParam);
        else if(filter_type == "gql")
            Filter_GQL(*data_g, *query_g, canParam);
        else if(filter_type == "cfl")
            Filter_CFL(*data_g, *query_g, canParam);
        else if(filter_type == "ceci")
            Filter_CECI(*data_g, *query_g, canParam);
        else if(filter_type == "daf")
            Filter_DAF(*data_g, *query_g, canParam);
        else if(filter_type == "pilos")
            Filter_Pilos(*data_g, *query_g, canParam);
        else{
            filter_type = "null";
        }

        auto end = std::chrono::high_resolution_clock::now();
        prep_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        size_t _total=0;
        for(int i=0; i<query_g->getVertexCnt(); i++){
            _total += canParam.candidates_count[i];
        }
        avg_candidates = (double)_total / query_g->getVertexCnt();

        std::cout << fmt::format("Filtering takes [{:.2f}] ms, with [{:.2f}] avg candidates\n", prep_time_ms, avg_candidates);

        start_promise.set_value();

        cfg.useFilter = true;
        cfg.can = &canParam;
        std::cout << fmt::format("Start exploring with filter [{}], order [{}], output_limit [{}]\n", filter_type, order_type, output_limit);
        explore_Intersection_Recursive<CandidatesBuffer>(*data_g, *query_g, order, cfg, time, ans, stop_flag, output_limit);
    }

    vector<string> vec;
    vec.push_back(filter_type);
    vec.push_back(order_type);
    vec.push_back(explore_optim);
    vec.push_back(fmt::format("{:.2f}", time));
    vec.push_back(fmt::format("{}", ans));
    vec.push_back(fmt::format("{}", output_limit));
    vec.push_back(fmt::format("{}", Profiler::getInst().total_intersection));
    vec.push_back(fmt::format("{:.2f}", prep_time_ms));
    vec.push_back(fmt::format("{:.2f}", avg_candidates));

    return vec;
}


string inline filter_intersec_test_timeout(const Graph* data_g, const Query* query_g, const Order& order, string filter_type, string order_type, string explore_optim, size_t output_limit = 1e5, uint32_t timeout = 3600){

    atomic<bool> stop_flag(false);
    std::promise<void> start_promise;
    std::future<void> start_future = start_promise.get_future();

    auto future = std::async(std::launch::async, [&]() {
        return filter_compare(data_g, query_g, order, filter_type, order_type, explore_optim, stop_flag, std::move(start_promise), output_limit);
    });

    start_future.get();

    if (future.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {
        stop_flag.store(true);
        vector<string> vec;
        vec = future.get();
        return fmt::format("{}", fmt::join(vec, ",")); 
    }

    return fmt::format("{}", fmt::join(future.get(), ","));
}


void process_query(const Graph *data_g, string data_file, string query_name, ofstream *file, size_t output_limit = 1e5, uint32_t timeout = 3600){
    vector<string> vec;
    vec.clear();
    vec.push_back(data_file);
    vec.push_back(query_name);
    string query_info = fmt::format("{}", fmt::join(vec, ","));

    string query_path = fmt::format("{}/data/query/{}.txt", PROJECT_SOURCE_DIR, query_name);
    Query *query_g = new Query();
    query_g->Load(query_path);

    std::cout << fmt::format("Running Query: [{}] on Data: [{}], timeout: [{}], output_limit: [{}]\n", query_name, data_file, timeout, output_limit);

    std::cout << "generating order...\n";
    Order order;
    Order_RI(*query_g, order);
    std::cout << fmt::format("order: [{}]\n", fmt::join(order, ","));
    
    for(auto eo: explore_optim){
        string res;
        if (eo == "filter"){
            for(auto ft: filter_types){
                res = filter_intersec_test_timeout(data_g, query_g, order, ft, "RI", eo, output_limit, timeout);
                string local_record = fmt::format("{},{}\n", query_info, res);
                *file << local_record << std::flush;
            }
        }
        else{
            res = filter_intersec_test_timeout(data_g, query_g, order, "null", "RI", eo, output_limit, timeout);
            string local_record = fmt::format("{},{}\n", query_info, res);
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


    vector<string> header({"data", "query_name", "filter_type", "order_type", "explore_optimization", "time_ms", "ans", "output_limit", "intersection_cnt", "prep_tims_ms", "avg_candidates"});

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
        std::cout << "open log at : " << log_file << "\n"; 
    }
    else{
        std::cout << "cannot create / open file at" << log_file << "\n";
        exit(-1);
    }

    string data_path = fmt::format("{}/data/data/{}.txt", PROJECT_SOURCE_DIR, data_file);
    Graph *data_g = new Graph();
    data_g->Load(data_path);
    data_g->LoadEigenIndex(fmt::format("{}/data/egienIndex/{}12A_500.csv", PROJECT_SOURCE_DIR, data_file));

    // for(int query_idx = )
    
    process_query(data_g, data_file, query_name, &file, 1e5, 300);
    

    delete data_g;

    return 1;
}
