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

#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>

using namespace std;

vector<string> prune_types = {
    "FP:DAF",
    "FP:GuP",
    "CD:BICE",
    "AP:BICE",
    "AP:VEQ"
};

vector<string> prune_compare(const Graph* data_g, const Query* query_g, string prune_type, atomic<bool>& stop_flag, size_t output_limit = 1e5){

    const Graph& data = *data_g;
    const Query& query = *query_g;
    Order order;
    Config cfg;
    double time=0;
    size_t ans=0;
    vector<string> vec;
    vec.clear();
    Profiler::getInst().reset();
    double prune_mem_overhead = 0.0;
    size_t prune_counts = 0;
    
    CandidateParam can(data, query);
    if(Filter_GQL(data, query, can)==false)
        goto Ret;

    order.clear();

    Order_RI(query, order);
    
    
    cfg.useFilter = true;
    cfg.can = &can;
    if(prune_type == "FP:DAF"){
        explore_Intersection_All<true, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag, output_limit);
        prune_mem_overhead = Profiler::getInst().daf_fp.mem_overhead_KB;
        prune_counts = Profiler::getInst().daf_fp.cnt;
    }
    else if(prune_type == "FP:GuP"){
        explore_Intersection_All<false, true, false, false, false>(data, query, order, cfg, time, ans, stop_flag, output_limit);
        prune_mem_overhead = Profiler::getInst().gup_fp.mem_overhead_KB;
        prune_counts = Profiler::getInst().gup_fp.cnt;
    }
    else if(prune_type == "CD:BICE"){
        explore_Intersection_All<false, false, false, true, false>(data, query, order, cfg, time, ans, stop_flag, output_limit);
        prune_mem_overhead = Profiler::getInst().bice_cd.mem_overhead_KB;
        prune_counts = Profiler::getInst().bice_cd.cnt;
    }
    else if(prune_type == "AP:VEQ"){
        explore_Intersection_All<false, false, false, false, true>(data, query, order, cfg, time, ans, stop_flag, output_limit);
        prune_mem_overhead = Profiler::getInst().veq_ap.mem_overhead_KB;
        prune_counts = Profiler::getInst().veq_ap.cnt;
    }
    else if(prune_type == "AP:BICE"){
        explore_Intersection_Recursive_BICE_AP(data, query, order, cfg, time, ans, stop_flag, output_limit);
        prune_mem_overhead = Profiler::getInst().bice_ap.mem_overhead_KB;
        prune_counts = Profiler::getInst().bice_ap.cnt;
    }

    std::cout << fmt::format("Running {}, Init Time {:.2f} ms\n", prune_type, Profiler::getInst().prune_init_time);
        
Ret:
    vec.push_back(prune_type);
    vec.push_back(fmt::format("{:.2f}", time));
    vec.push_back(fmt::format("{}", ans));
    vec.push_back(fmt::format("{}", output_limit));
    vec.push_back(fmt::format("{:.2f}", prune_mem_overhead));
    vec.push_back(fmt::format("{:.2f}", Profiler::getInst().prune_init_time));
    vec.push_back(fmt::format("{}", prune_counts));
    vec.push_back(fmt::format("{}", Profiler::getInst().total_intersection));
    return vec; 
}

string run_prune_with_timeout(const Graph* data_g, const Query* query_g, string prune_type, size_t output_limit = 1e5, uint32_t timeout = 3600){

    std::atomic<bool> stop_flag(false);

    auto future = std::async(std::launch::async, [&]() {
        return prune_compare(data_g, query_g, prune_type, stop_flag, output_limit);
    });

    if (future.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {

        stop_flag.store(true);
        vector<string> vec;
        vec = future.get();
        
        return fmt::format("{}", fmt::join(vec, ",")); 
    }

    return fmt::format("{}", fmt::join(future.get(), ","));
}


void process_query(const Graph* data_g, string data_file, string query_name, ofstream *file, size_t output_limit = 1e5, uint32_t timeout = 3600){
    vector<string> vec;
    vec.clear();
    vec.push_back(data_file);
    vec.push_back(fmt::format("{}", query_name));
    string query_info = fmt::format("{}", fmt::join(vec, ","));

    std::cout << fmt::format("Running Query: [{}] on Data: [{}], timeout: [{}], output_limit: [{}]\n", query_name, data_file, timeout, output_limit);
    
    string query_path = fmt::format("{}/data/query/{}.txt", PROJECT_SOURCE_DIR, query_name);
    Query *query_g = new Query();
    query_g->Load(query_path);

    for(auto prune_type: prune_types){
        std::cout << fmt::format("Running Prune Type: [{}]\n", prune_type);
        string res = run_prune_with_timeout(data_g, query_g, prune_type, output_limit, timeout);
        string local_record = fmt::format("{},{}\n", query_info, res);
        *file << local_record << std::flush;
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


    vector<string> header({"data", "query_name", "prune_type", "time_ms", "ans", "output_limit", "memory_overhead", "prune_init_time","prune_cnt", "intersection_cnt"});

    string heads = fmt::format("{}", fmt::join(header, ","));

    string data_file = string(argv[1]);
    string query_name = string(argv[2]);

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
