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

vector<string> algs = {
    "RI",
    "RI+cache+filter+fs",
    "QSI",
    "GQL",
    "CFL",
    "CECI",
    "VF2PP",
    "VF3",
    "DAF",
    "GuP",
    "GraphPi",
    "GraphZero",
    "BICE",
    "VEQ",
    "Pilos"
};

vector<string> alg_compare(const Graph* data_g, const Query* query_g, string alg, atomic<bool>& stop_flag, std::promise<void> start_promise, size_t output_limit = 1e5){

    const Graph& data = *data_g;
    const Query& query = *query_g;
    double time=0;
    size_t ans=0;
    vector<string> vec;
    vec.clear();
    Profiler::getInst().reset();
    double prep_time_ms = 0.0;
    double avg_candidates = 0.0;
    if(alg == "RI"){
        CandidateParam can(data, query);
        if(Filter_NLF(data, query, can) == false){
            goto Ret;
        }

        Order order;
        Order_RI(query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "QSI"){
        CandidateParam can(data, query);
        if(Filter_NLF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_QSI(data, query, can, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "GQL"){
        CandidateParam can(data, query);
        if(Filter_GQL(data, query, can)==false)
            goto Ret;

        Order order;
        Order_GQL(data, query, can, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "CFL"){
        CandidateParam can(data, query);
        if(Filter_CFL(data, query, can) == false)
            goto Ret;

        Order order;
        Order_CFL(data, query, can, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "CECI"){
        CandidateParam can(data, query);
        if(Filter_CECI(data, query, can) == false)
            goto Ret;

        Order order;
        Order_CECI(data, query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "VF2PP"){
        CandidateParam can(data, query);
        if(Filter_NLF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_VF2PP(data, query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "VF3"){
        CandidateParam can(data, query);
        if(Filter_NLF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_VF3(data, query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "DAF"){
        CandidateParam can(data, query);
        if(Filter_DAF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_RI(query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<true, false, false, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "GuP"){
        CandidateParam can(data, query);
        if(Filter_DAF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_GuP(data, query, can, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, true, true, false, false>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "VEQ"){
        CandidateParam can(data, query);
        if(Filter_DAF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_RI(query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_All<false, false, false, false, true>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "BICE"){
        CandidateParam can(data, query);
        if(Filter_DAF(data, query, can) == false)
            goto Ret;

        Order order;
        Order_RI(query, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;
        explore_Intersection_Recursive_BICE(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "GraphPi"){
        Order order;
        Order_GraphPi(data, query, order, true);

        start_promise.set_value();

        ReuseParam reuse(*query_g, order);

        Config cfg;
        cfg.useCache = true;
        cfg.reuse = &reuse;
        explore_Intersection_Recursive<CacheBuffer>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "GraphZero"){
        Order order;
        Order_GraphZero(data, query, order);

        start_promise.set_value();

        ReuseParam reuse(*query_g, order);

        Config cfg;
        cfg.useCache = true;
        cfg.reuse = &reuse;
        explore_Intersection_Recursive<CacheBuffer>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "RI+cache+filter+fs"){
        Order order;
        Order_RI(query, order);

        CandidateParam can(data, query);
        if(Filter_GQL(data, query, can) == false)
            goto Ret;

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;

        ReuseParam reuse(*query_g, order);
        cfg.useCache = true;
        cfg.reuse = &reuse;
        explore_Intersection_Cache_DAF_Filter(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }
    else if(alg == "Pilos"){
        Order order;

        CandidateParam can(data, query);
        if(Filter_Pilos(data, query, can) == false)
            goto Ret;

        Order_GQL(data, query, can, order);

        start_promise.set_value();

        Config cfg;
        cfg.useFilter = true;
        cfg.can = &can;

        explore_Intersection_All<true, false, false, false, true>(data, query, order, cfg, time, ans, stop_flag,output_limit);
    }

Ret:
    vec.push_back(alg);
    vec.push_back(fmt::format("{:.2f}", time));
    vec.push_back(fmt::format("{}", ans));
    vec.push_back(fmt::format("{}", output_limit));
    vec.push_back(fmt::format("{}", Profiler::getInst().total_intersection));
    return vec; 
}

string run_alg_with_timeout(const Graph* data_g, const Query* query_g, string alg, size_t output_limit = 1e5, uint32_t timeout = 3600){

    std::atomic<bool> stop_flag(false);
    std::promise<void> start_promise;
    std::future<void> start_future = start_promise.get_future();

    auto future = std::async(std::launch::async, [&]() {
        return alg_compare(data_g, query_g, alg, stop_flag, std::move(start_promise) ,output_limit);
    });

    start_future.get();

    if (future.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {
        // future.get();
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
    // std::shared_ptr<Query> query_g = make_shared<Query>();
    Query *query_g = new Query();
    query_g->Load(query_path);

    for(auto alg: algs){
        cout << fmt::format("Running Algorithm: [{}]\n", alg);
        string res = run_alg_with_timeout(data_g, query_g, alg, output_limit, timeout);
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

// #ifdef LOG_OUTPUT
//     spdlog::set_level(spdlog::level::trace);
//     spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
// #endif


    vector<string> header({"data", "query_name", "algorithm", "time_ms", "ans", "output_limit", "intersection_cnt"});

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
    data_g->LoadEigenIndex(fmt::format("{}/data/egienIndex/{}12A_500.csv", PROJECT_SOURCE_DIR, data_file));

    process_query(data_g, data_file, query_name, &file, 1e5, 300);

    delete data_g;

    return 1;
}
