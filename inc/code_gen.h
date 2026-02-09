#pragma once
#include "graph.h"
#include <vector>
#include <algorithm>
#include <assert.h>
#include <sstream>
#include <fmt/format.h>
#include <optional>
#include <chrono>

typedef struct{
    int threadNum{1};
    std::vector<VertexID> order{};
    std::string query_g_path;
    std::string data_g_path;
    double time{0};
    bool isProfile{false};
}Context;

#ifdef _USE_DEPRECATED
class VertexSetIR{
private:
    uint32_t _vid{INVALID};
    uint32_t _parent_id{INVALID};
    uint32_t _depth{INVALID};
    std::vector<VertexID> _edges;
    std::vector<VertexID> _restriction; // _restriction[_] = 5 means f(v) < f(5), f is embedding mapping
    std::vector<VertexID> _follow_vertices;

public:
    VertexSetIR(uint32_t vid): _vid(vid){
        _edges.clear();
        _restriction.clear();
        _follow_vertices.clear();
    }

    VertexSetIR(){
        _edges.clear();
        _restriction.clear();
        _follow_vertices.clear();
    }

    // generate the VertexSetIR before the depth.
    VertexSetIR(const VertexSetIR &vir, const std::vector<VertexID> &depthsByVertex, uint32_t depth): _depth(depth){  
        const auto& edges = vir.getEdges();
        const auto& res = vir.getRestricitons();

        _edges.clear();
        _restriction.clear();
        _follow_vertices.clear();

        for(auto vid: edges){
            if(depthsByVertex[vid] <= depth) _edges.push_back(vid);
        }

        for(auto vid: res){
            if(depthsByVertex[vid] <= depth) _restriction.push_back(vid);
        }

        std::sort(_edges.begin(), _edges.end());
        std::sort(_restriction.begin(), _restriction.end());
    }

    void addEdge(VertexID u){
        _edges.push_back(u);
        std::sort(_edges.begin(), _edges.end());
    }

    void addRestriction(VertexID u){
        _restriction.push_back(u);
        std::sort(_restriction.begin(), _restriction.end());
    }

    void addFollowVetex(VertexID u){
        _follow_vertices.push_back(u);
        std::sort(_follow_vertices.begin(), _follow_vertices.end());
    }

    const std::vector<VertexID>& getEdges() const{return _edges;}
    const std::vector<VertexID>& getRestricitons() const{return _restriction;}
    const std::vector<VertexID>& getFollowVertices() const{return _follow_vertices;}
    const uint32_t getVSetID() const{return _vid;}
    void setVSetID(uint32_t id){_vid = id;}
    const VertexID getDepth() const{return _depth;}
    void setDepth(uint32_t dep){_depth = dep;}
    const uint32_t getParentVSetID() const{return _parent_id;}
    void setParentVSetID(uint32_t _pa_id) {_parent_id = _pa_id;}

    bool hasEdge(VertexID v) const{
        return (std::find(_edges.begin(), _edges.end(), v) != _edges.end());
    }
    bool hasRestriction(VertexID v) const{
        return (std::find(_restriction.begin(), _restriction.end(), v) != _restriction.end());
    }

    // used for std::sort();
    bool operator<(const VertexSetIR& rhs) const{
        const auto &rhsEdges = rhs.getEdges();
        const auto &rhsRestriction = rhs.getRestricitons();

        if(_depth < rhs.getDepth()) return true;
        else if(_edges.size() < rhsEdges.size()) return true;
        else if(_restriction.size() < rhsRestriction.size()) return true;
        
        for(int i=0; i<_edges.size(); i++){
            if(_edges[i]<rhsEdges[i]) return false;
            if(_edges[i]>rhsEdges[i]) return true;
        }

        for(int i=0; i<_restriction.size(); i++){
            if(_restriction[i]<rhsRestriction[i]) return false;
            if(_restriction[i]>rhsRestriction[i]) return true;
        }

        return false;
    }

    // used only in iter_set; for std::find function;
    bool operator==(const VertexSetIR &rhs) const{
        return _edges == rhs.getEdges() &&
               _restriction == rhs.getRestricitons() &&
               _depth == rhs.getDepth();
    }

    // edge-induced
    bool isSupersetOf(const VertexSetIR &rhs) const {
        if (_depth > rhs.getDepth()) return false;
        const auto &rhs_e = rhs.getEdges();
        const auto &rhs_r = rhs.getRestricitons();
        for(int i=0; i<_edges.size();i++)
            if(std::find(rhs_e.begin(), rhs_e.end(), _edges[i]) == rhs_e.end()) return false; // for vertex-induced, this should be excatly same for rhs_e and _edges before k-th depth;
        for(int i=0; i<_restriction.size();i++)
            if(std::find(rhs_r.begin(), rhs_r.end(), _restriction[i]) == rhs_r.end()) return false;
        return true;
    }

    friend std::ostream &operator<<(std::ostream &out, const VertexSetIR &vir){
        out << "/* VSet(" << vir.getVSetID() << ", " << vir.getDepth() << ") In-Edges: ";
        for (auto e: vir.getEdges()){
            out << e << " ";
        }
        out << "Restricts: ";
        for (auto r: vir.getRestricitons()){
            out << r << " ";
        }
        out << "*/\n";
        return out;
    }
};


class PlanIR{
private:
    std::vector<VertexID> _order;
    std::vector<std::vector<VertexSetIR>> _setOps; // orgnaized by depth
    std::vector<VertexSetIR> _iterSet;
    std::vector<uint32_t> _depthsByVertex; // _depthsByVertex[vid] = depth;

    std::vector<VertexSetIR> _tot_setOpsArr; // all in an array

    uint32_t _tot_op{0};


public:
    PlanIR() = default;
    PlanIR(const std::vector<VertexID> &order, const Query& query):_order(order){
        uint32_t qvcnt = query.getVertexCnt();
        int maxDepth = qvcnt - 1;

        _depthsByVertex.resize(order.size());
        for(int i=0; i<order.size(); i++)
            _depthsByVertex[order[i]] = i;

        // for(int i=0; i<order.size(); i++)
        //     std::cout<<_depthsByVertex[i]<<" ";
        // std::cout<<"\n";

        _setOps.resize(maxDepth);
        for(int i=0; i<maxDepth; i++)
            _setOps.at(i).clear();

        _iterSet.resize(maxDepth);    
        
        std::vector<VertexSetIR> vertexIRs(qvcnt);

        for(int vid=0; vid<qvcnt; vid++){
            VertexSetIR &vIR = vertexIRs.at(vid);
            vIR.setVSetID(vid);
            uint32_t nebCnt=0;
            const VertexID* neb = query.getNeb(vid, nebCnt);
            
            for(int idx=0; idx<nebCnt; idx++)
                vIR.addEdge(neb[idx]);
            // std::cout << vIR;
        }

        // std::cout<< "----------------" <<'\n';

        for(int dep=0; dep<maxDepth; dep++){
            for(int followDep = dep+1; followDep<qvcnt; followDep++){
                const VertexSetIR &vIR = vertexIRs[order[followDep]];
                VertexSetIR ops = VertexSetIR(vIR, _depthsByVertex, dep);
                // std::cout<<ops;
                if(ops.getEdges().size())
                    _setOps.at(dep).push_back(ops);
            }
            uint32_t iter_v = order[dep+1];
            const VertexSetIR &iter_vIR = vertexIRs[iter_v];
            VertexSetIR iter_vset = VertexSetIR(iter_vIR, _depthsByVertex, dep);
            if(iter_vset.getEdges().size() <= 0){
                printf("Invalid schedule: iter_vset at depth %d is empty\n", dep+1);
                exit(-1);
            }
            _iterSet.at(dep) = iter_vset;
        }

        _tot_setOpsArr.clear();
        uint32_t nextId = 0;
        for(auto &ops: _setOps){
            std::sort(ops.begin(), ops.end());
            ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
            for(auto &op: ops){
                op.setVSetID(nextId);
                _tot_setOpsArr.push_back(op);
                nextId++;
            }
        }
        _tot_op = nextId;

        for(int i=0; i<_tot_op; i++){
        if(i != _tot_setOpsArr[i].getVSetID()){
                printf("error, tot ops wrong, exit\n");
                assert(i == _tot_setOpsArr[i].getVSetID());
            }
        }

        for(int i=0; i<_tot_op; i++){
            auto &op = _tot_setOpsArr[i];
            auto pvset = getParentVSet(op);
            if(pvset.has_value()){
                op.setParentVSetID(pvset->getVSetID());
            }
        }

        // TODO add used_vertices maintain;
        for(int dep=0; dep<maxDepth; dep++){
            auto &ops = _setOps[dep];
            for(auto &op: ops){
                for(int followDep = dep+1; followDep<qvcnt; followDep++){
                    const VertexSetIR &vIR = vertexIRs[order[followDep]];
                    VertexSetIR _op = VertexSetIR(vIR, _depthsByVertex, dep);
                    if(op == _op){
                        op.addFollowVetex(order[followDep]);
                    }
                }
            }
        }

        for(int dep=0; dep<maxDepth; dep++){
            auto &ops = _setOps[dep];
            VertexSetIR vIR = *std::find(ops.begin(), ops.end(), _iterSet[dep]);
            if(vIR.getVSetID() == INVALID){
                printf("iter_vset at depth %d Invalid, id = -1\n", dep);
                exit(-1);
            }
            _iterSet[dep] = vIR;
        }

    }

    const std::vector<VertexID>& getOrder() const{
        return _order;
    }

    const std::vector<std::vector<VertexSetIR>>& getSetOps() const{
        return _setOps;
    }
    
    const std::vector<VertexSetIR>& getIterVSet() const{
        return _iterSet;
    }

    const uint32_t getTotOp() const {return _tot_op;}

    const std::vector<VertexSetIR> getSetOpsArray() const {return _tot_setOpsArr;} 

    bool isLastOp(const VertexSetIR &op) const{
        return op.getDepth() == _iterSet.size()-1;
    }

    std::optional<VertexSetIR> getParentVSet(VertexSetIR vset) const{
        std::optional<VertexSetIR> parentVSet;
        // VertexSetIR parentVSet;
        const auto &vset_r = vset.getRestricitons();
        VertexID cur_dep_v = _order[vset.getDepth()];
        // restriction bounded
        for(const auto& op: _setOps.at(vset.getDepth())){
            const auto &op_r = op.getRestricitons();
            if(op.isSupersetOf(vset) && 
               op.getVSetID() != vset.getVSetID() &&
               op.getRestricitons().size() + 1 == vset.getRestricitons().size() &&
               std::find(vset_r.begin(), vset_r.end(), cur_dep_v) != vset_r.end() &&
               std::find(op_r.begin(), op_r.end(), cur_dep_v) == op_r.end()
            ){
                parentVSet = op;
            }
            if(parentVSet.has_value()) return parentVSet;
        }

        // prefix reuse
        if(vset.getDepth() > 0){
            const auto &ops = _setOps.at(vset.getDepth()-1);
            for(const auto &op: ops){
                if(op.isSupersetOf(vset) &&
                   op.getVSetID() != vset.getVSetID() &&
                   (!parentVSet.has_value() || parentVSet->isSupersetOf(op))
                ){
                    parentVSet = op;
                }
            }
        }

        return parentVSet;
    }
};

std::string indentGen(int depth){
    return std::string(depth, '\t');
}

std::string includeGen(Context &ctx){
    std::ostringstream code;
    // code << "#include \"command.h\"\n";
    code << "#include <iostream>\n";
    code << "#include \"graph.h\"\n";
    code << "#include \"code_gen.h\"\n";
    code << "#include \"vertexset.h\"\n";
    if(ctx.isProfile) code << "#include \"common_type.h\"\n";

    return code.str();
}

std::string readNebGen(const PlanIR &plan, uint32_t dep){
    std::ostringstream code;
    VertexID cur_v = plan.getOrder()[dep];
    if(dep > 0){
        const auto &iter_sets = plan.getIterVSet();
        const auto &iter_op = iter_sets.at(dep-1);
        code << indentGen(dep+1) << fmt::format("VertexID v{id} = vsets[{iter_id}].getData(v{id}_id);\n",
                                                fmt::arg("id", cur_v),
                                                fmt::arg("iter_id", iter_op.getVSetID()));
    }
    code << indentGen(dep+1) << fmt::format("uint32_t v{id}_neb_cnt = 0;\n", fmt::arg("id", cur_v));
    code << indentGen(dep+1) << fmt::format("const VertexID* v{id}_neb = data.getNeb(v{id}, v{id}_neb_cnt);\n",
                                            fmt::arg("id", cur_v));
    code << indentGen(dep+1) << fmt::format("v_neb_set[{id}].Init(v{id}_neb, v{id}_neb_cnt);\n",
                                            fmt::arg("id", cur_v));
    code << indentGen(dep+1) << fmt::format("v_neb_set[{id}].setVid(v{id});\n", fmt::arg("id", cur_v));

    return code.str();
}

std::string opGen(const PlanIR &plan, const VertexSetIR &op, bool isProfile=false){
    std::string code;
    auto pvset = plan.getParentVSet(op);
    const auto &op_e = op.getEdges();
    const auto &op_r = op.getRestricitons();
    int op_dep = op.getDepth();
    VertexID op_dep_v = plan.getOrder()[op_dep];

    bool has_edge_to_v = op.hasEdge(op_dep_v);
    bool has_res_to_v = op.hasRestriction(op_dep_v);

    if(pvset.has_value()){
        if(pvset->getDepth() == op.getDepth()){ // reserved for restriction 
            
        }
        else if(pvset->getDepth() == op.getDepth() - 1){ // prefix reuse    
            if(has_edge_to_v){
                if(isProfile){
                    code += fmt::format("auto _s{depth} = std::chrono::high_resolution_clock::now();\n", fmt::arg("depth", op.getDepth()+1));
                }
                code += fmt::format("vsets[{op_id}].IntersecOf(vsets[{parent_id}], v_neb_set[{vid}]);\n",
                            fmt::arg("op_id", op.getVSetID()),
                            fmt::arg("parent_id", pvset->getVSetID()),
                            fmt::arg("vid", op_dep_v));
                if(isProfile){
                    code += fmt::format("auto _e{depth} = std::chrono::high_resolution_clock::now();\n", fmt::arg("depth", op.getDepth()+1));
                    code += fmt::format("std::chrono::duration<double> _d{depth} = _e{depth} - _s{depth};\n", fmt::arg("depth", op.getDepth()+1));
                    code += fmt::format("Profile::getInst().intersect_time[{depth}] += _d{depth}.count();\n", fmt::arg("depth", op.getDepth()+1));
                    code += fmt::format("Profile::getInst().intersect_tot += _d{depth}.count();\n", fmt::arg("depth", op.getDepth()+1));
                    code += fmt::format("Profile::getInst().intersect_cnt[{depth}] += 1;\n", fmt::arg("depth", op.getDepth()+1));
                }
            }
            else{
                if(has_res_to_v){ //reserved for restriction

                }
                else{
                    code += fmt::format("vsets[{op_id}].RemoveFrom(vsets[{parent_id}], v_neb_set[{vid}].getVid());\n",
                                        fmt::arg("op_id", op.getVSetID()),
                                        fmt::arg("parent_id", pvset->getVSetID()),
                                        fmt::arg("vid", op_dep_v));

                    // if(plan.isLastOp(op))
                    //     code += fmt::format("global_ans += vsets[{op_id}].getSize();\n",fmt::arg("op_id", op.getVSetID()));
                }
            }

            if(plan.isLastOp(op))
                code += fmt::format("global_ans += vsets[{op_id}].getSize();\n",fmt::arg("op_id", op.getVSetID()));
        }
    }
    else{  // no parent
        if(has_res_to_v);
        else{
            code += fmt::format("vsets[{op_id}].Init(v_neb_set[{vid}].getDataPtr(), v_neb_set[{vid}].getSize());\n", 
            fmt::arg("op_id", op.getVSetID()),
            fmt::arg("vid", op_dep_v));
        }

        const auto &order = plan.getOrder();
        for(int cur_dep=0; cur_dep<op_dep; cur_dep++){
            VertexID v_at_dep = order[cur_dep];
            if(op.hasRestriction(v_at_dep)){

            }
            else{
                code += fmt::format("vsets[{op_id}].RemoveFrom(vsets[{op_id}], v_neb_set[{vid}].getVid())\n;",
                                    fmt::arg("op_id", op.getVSetID()),
                                    fmt::arg("vid", v_at_dep));
            }
        }

        code += fmt::format("if (vsets[{op_id}].getSize() == 0) continue;\n", fmt::arg("op_id", op.getVSetID()));
        if(plan.isLastOp(op))
            code += fmt::format("global_ans += vsets[{op_id}].getSize();\n",fmt::arg("op_id", op.getVSetID()));
    }

    return code;
}

std::string iterGen(const PlanIR& plan, uint32_t dep){
    uint32_t max_dep = plan.getOrder().size() - 1;
    if (dep >= max_dep-1) {
        return "";
    } else {
        const auto &iter_op = plan.getIterVSet()[dep];
        VertexID cur_v = plan.getOrder()[dep+1];
        return fmt::format(
                "for (uint32_t v{id}_id = 0; v{id}_id < vsets[{iter_id}].getSize(); v{id}_id++) {left} // loop-{dep} begin\n",
                fmt::arg("left", "{"),
                fmt::arg("id", cur_v),
                fmt::arg("iter_id", iter_op.getVSetID()),
                fmt::arg("dep", dep + 1));
    }
}

std::string mainGen(const Context &ctx){
    std::ostringstream code;
    code << "int main(int argc, char **argv){ // main begin\n";
    // code << indentGen(1) << "MatchingCommand command(argc, argv);\n";
    code << indentGen(1) << fmt::format("std::string query_g_path = \"{query_g_path}\";\n", fmt::arg("query_g_path", ctx.query_g_path));
    code << indentGen(1) <<  fmt::format("std::string data_g_path = \"{data_g_path}\";\n", fmt::arg("data_g_path", ctx.data_g_path));
    code << indentGen(1) <<  "Graph* data_g = new Graph();\n";
    code << indentGen(1) <<  "data_g->Load(data_g_path);\n";
    code << indentGen(1) <<  "Query* query_g = new Query();\n";
    code << indentGen(1) <<  "query_g->Load(query_g_path);\n";
    code << indentGen(1) <<  "Context ctx;\n";

    code << indentGen(1) << fmt::format("ctx.threadNum = {threadNum};\n", fmt::arg("threadNum", ctx.threadNum));
    std::string orderInstance = fmt::format("{{{}}}", fmt::join(ctx.order, ", "));
    std::string orderPrintInst = fmt::format("{}", fmt::join(ctx.order, " "));
    // code << indentGen(1) << fmt::format("std::cout << \"threadNum = {threadNum}\\n\";\n", fmt::arg("threadNum", ctx.threadNum));
    code << indentGen(1) << fmt::format("std::vector<VertexID> order = {orderIns};\n", fmt::arg("orderIns", orderInstance));
    // code << indentGen(1) << fmt::format("std::cout << \"order: {orderIns}\\n\";\n", fmt::arg("orderIns", orderPrintInst));
    code << indentGen(1) << "PlanIR plan(order, *query_g);\n";
    code << indentGen(1) << "unsigned long long ans = query(plan, *data_g, ctx);\n";
    // code << indentGen(1) << "std::cout<< \"ans: \"<< ans << \"\\n\";\n";
    code << indentGen(1) << fmt::format("std::string orderInst = \"{orderInst}\";\n", fmt::arg("orderInst", orderPrintInst));
    if(ctx.isProfile){
        code << indentGen(1) << "std::vector<std::string> intersec_time_format(query_g->getVertexCnt());\n";
        code << indentGen(1) << "std::vector<double> inter_time(query_g->getVertexCnt());\n";
        code << indentGen(1) << "std::vector<uint32_t> inter_cnt(query_g->getVertexCnt());\n";
        code << indentGen(1) << "for(int i=0; i<query_g->getVertexCnt();i++){\n";
        code << indentGen(1) << "inter_time[order[i]] = Profile::getInst().intersect_time[i];\n";
        code << indentGen(1) << "inter_cnt[order[i]] = Profile::getInst().intersect_cnt[i];\n";
        code << indentGen(1) << "}\n";
        
        code << indentGen(1) << "for(int i=0; i<query_g->getVertexCnt(); i++){\n";
        code << indentGen(1) << "intersec_time_format[i] = fmt::format(\"{:.2f}\", inter_time[i]);\n";
        code << indentGen(1) << "}\n";
        
        code << indentGen(1) << "std::string intersecTimeInst = fmt::format(\"{}\", fmt::join(intersec_time_format, \",\"));\n";
        code << indentGen(1) << "std::string intersecCntInst = fmt::format(\"{}\", fmt::join(inter_cnt, \",\"));\n";
        
        code << indentGen(1) << "printf(\"%.2f,%llu,%s,%d,%.2f,%s,%s\", ctx.time, ans, orderInst.c_str(), plan.getTotOp(),Profile::getInst().intersect_tot, intersecTimeInst.c_str(), intersecCntInst.c_str());\n";
    }
    else{
        code << indentGen(1) << "printf(\"%.2f,%llu,%s,%d\",ctx.time,ans,orderInst.c_str(),plan.getTotOp());\n";
    }
    
    code <<"} // main end \n";
    return code.str();
}

std::string queryGen(const PlanIR& plan, const Context& ctx){
    std::ostringstream code;
    uint32_t max_dep = plan.getOrder().size()-1;
    const auto& setOps = plan.getSetOps();
    VertexID first_v = plan.getOrder()[0];

    code << "unsigned long long query(const PlanIR& plan, const Graph &data, Context& ctx){ // query begin\n";
    code << "unsigned long long global_ans = 0;\n";
    if(ctx.isProfile) code << "Profile::getInst().reset(plan.getOrder().size());\n";
    code << "auto start = std::chrono::high_resolution_clock::now();\n";
    code << "#pragma omp parallel num_threads(ctx.threadNum) reduction(+: global_ans)\n{\n";
    code << "uint32_t totOp = plan.getTotOp();\n";
    code << "VertexSet *vsets = new VertexSet[totOp];\n";
    code << "for(int i=0; i<totOp; i++) vsets[i].alloc(data.getMaxDgree());\n";
    code << fmt::format("VertexSet *v_neb_set = new VertexSet[{tot_iter}];\n", fmt::arg("tot_iter", plan.getOrder().size()));
    code << fmt::format("for(int i=0; i<{tot_iter}; i++) v_neb_set[i].alloc(data.getMaxDgree());\n", fmt::arg("tot_iter", plan.getOrder().size()));
    code << "#pragma omp for schedule(dynamic) nowait\n";
    code << fmt::format("for (uint32_t v{id} = 0; v{id} < data.getVertexCnt(); v{id}++) {{ // loop-0 begin\n",
                        fmt::arg("id", first_v));
    
    for (int dep = 0; dep < max_dep; dep++) {
        // code for reading adj from the graph
        code << readNebGen(plan, dep);
        // code for computation at this loop
        const auto &ops = setOps.at(dep);
        for (const auto &op: ops) {
            code << opGen(plan, op, ctx.isProfile) << op;
        }
        // code for iterating next loop
        if (dep == max_dep - 1) continue;
        code << iterGen(plan, dep);
    }

    for(int dep = max_dep-1; dep >=0; dep--){
        code << "} // loop-" << std::to_string(dep) << " end\n";
    }

    
    // code << "for(int i=0; i<totOp; i++) {delete *(vsets+i);}\n"; 
    code << "delete[] vsets;\n";
    code << "} // omp-end\n";
    code << "auto end = std::chrono::high_resolution_clock::now();\n";
    code << "std::chrono::duration<double> duration = end - start;\n";
    
    code << "ctx.time = duration.count();\n";
    // code << "printf(\"Execution time: %.2f s\\n\", duration.count());\n";
    code << "return global_ans;\n";
    code << "} // query-end\n" << "\n\n";

    return code.str();
}

std::string codeGen(const PlanIR& plan, Context &ctx){
    std::ostringstream code;

    code << includeGen(ctx);

    code << queryGen(plan, ctx);
    
    code << mainGen(ctx);

    return code.str();
}

#else

class VertexSetIR{
private:
    uint32_t _vid{INVALID};
    uint32_t _parent_id{INVALID};
    uint32_t _depth{INVALID}; // this is the depth where vertex set been calculated 
    std::vector<VertexID> _edge_idx;
    uint32_t _v_depth{INVALID}; // this is  the depth where vertex should orginally be calculated

    LabelID _v_label{INVALID}; // this is the label of vertex

public:
    VertexSetIR(uint32_t vid): _vid(vid){
        _edge_idx.clear();
    }

    VertexSetIR(){
        _edge_idx.clear();
    }

    // generate the VertexSetIR before the depth.
    VertexSetIR(const VertexSetIR &vir, uint32_t depth, uint32_t v_dep): _depth(depth), _v_depth(v_dep), _v_label(vir.getLabel()){  
        const auto& edgeIdx = vir.getEdgeIdx();

        _edge_idx.clear();

        for(auto vid: edgeIdx){
            if(vid <= depth) _edge_idx.push_back(vid);
        }

        std::sort(_edge_idx.begin(), _edge_idx.end());
    }

    void addEdgeIdx(uint32_t u_idx){
        _edge_idx.push_back(u_idx);
        std::sort(_edge_idx.begin(), _edge_idx.end());
    }

    const std::vector<VertexID> getEdge(const Order& order) const{
        std::vector<VertexID> _e(_edge_idx.size());
        for(int i=0; i<_edge_idx.size(); i++)
            _e[i] = order[_edge_idx[i]];
        return _e;
    }

    const std::vector<VertexID>& getEdgeIdx() const{return _edge_idx;}
    const uint32_t getVSetID() const{return _vid;}
    void setVSetID(uint32_t id){_vid = id;}
    const VertexID getDepth() const{return _depth;}
    void setDepth(uint32_t dep){_depth = dep;}
    const uint32_t getParentVSetID() const{return _parent_id;}
    void setParentVSetID(uint32_t _pa_id) {_parent_id = _pa_id;}
    const uint32_t getVDepth() const{return _v_depth;}
    const LabelID getLabel() const { return _v_label; }
    void setLabel(LabelID label) { _v_label = label; }

    bool hasEdge(VertexID v) const{
        return (std::find(_edge_idx.begin(), _edge_idx.end(), v) != _edge_idx.end());
    }

    // used for std::sort();
    bool operator<(const VertexSetIR& rhs) const{
        const auto &rhsEdgeIdx = rhs.getEdgeIdx();

        if(this == &rhs) return false;

        if(_depth != rhs.getDepth()) return _depth < rhs.getDepth();

        if (_v_label != rhs.getLabel()) return _v_label < rhs.getLabel();
        
        if(_edge_idx.size() != rhsEdgeIdx.size()) return _edge_idx.size() < rhsEdgeIdx.size();
        
        for(int i=0; i<_edge_idx.size(); i++){
            if(_edge_idx[i] != rhsEdgeIdx[i])
                return _edge_idx[i] < rhsEdgeIdx[i];
        }

        return false; // if all are same, return false;
    }

    bool operator==(const VertexSetIR &rhs) const{
        return _v_label == rhs.getLabel() && _edge_idx == rhs.getEdgeIdx();
    }

    // edge-induced
    bool isSupersetOf(const VertexSetIR &rhs) const {
        if (_depth > rhs.getDepth()) return false;
        const auto &rhs_e = rhs.getEdgeIdx();
        for(int i=0; i<_edge_idx.size();i++)
            if(std::binary_search(rhs_e.begin(), rhs_e.end(), _edge_idx[i]) == false) return false; // for vertex-induced, this should be excatly same for rhs_e and _edge_idx before k-th depth;
        return true;
    }

    std::string debugOutput(const Order& order) const{
        std::ostringstream out;
        const auto _e = getEdge(order);
        out << "/* VSet( id:" << getVSetID() << ", label:" << getLabel() << ", depth:" << getDepth() << ", parentId:"<< getParentVSetID() <<") In-Edges Index: ";
        for (auto e: getEdgeIdx()){
            out << e << " ";
        }
        out << "In-Edges : ";
        for (auto e: _e){
            out << e << " ";
        }
        out << "*/\n";
        return out.str();
    }
};


class PlanIR{
    typedef uint32_t Depth;
    typedef uint32_t VSetId;
private:
    
    std::vector<VertexID> _order;
    std::vector<std::vector<VSetId>> _setOps; // orgnaized by depth, _setOps[depth]
    std::vector<VSetId> _iterSet;             // vset_id = _iterSet[depth]
    std::vector<VertexID> _depthsByVertex; // _depthsByVertex[vid] = depth;

    std::vector<VertexSetIR> _tot_setOpsArr; // all in an array

    uint32_t _tot_op{0};


public:
    PlanIR() = default;
    PlanIR(const std::vector<VertexID> &order, const Query& query):_order(order){
        uint32_t qvcnt = query.getVertexCnt();
        // int maxDepth = qvcnt - 1;

        _depthsByVertex.resize(order.size());
        for(int i=0; i<order.size(); i++)
            _depthsByVertex[order[i]] = i;

        // for(int i=0; i<order.size(); i++)
        //     std::cout<<_depthsByVertex[i]<<" ";
        // std::cout<<"\n";
        
        std::vector<VertexSetIR> vertexIRs(qvcnt);

        for(int vid=0; vid<qvcnt; vid++){
            VertexSetIR &vIR = vertexIRs.at(vid);
            vIR.setVSetID(vid);
            vIR.setLabel(query.getVertexLabel(vid));
            uint32_t nebCnt=0;
            const VertexID* neb = query.getNeb(vid, nebCnt);
            
            for(int idx=0; idx<nebCnt; idx++)
                vIR.addEdgeIdx(_depthsByVertex[neb[idx]]);
            // std::cout << vIR;
        }

        // std::cout<< "----------------" <<'\n';

        std::vector<VertexSetIR> _all_ops;
        std::vector<VertexSetIR> _iter_tmp;
        _all_ops.clear();
        _iter_tmp.resize(qvcnt);

        for(int dep=0; dep<qvcnt-1; dep++){
            for(int followDep = dep+1; followDep<qvcnt; followDep++){
                const VertexSetIR &vIR = vertexIRs[order[followDep]];
                VertexSetIR ops = VertexSetIR(vIR, dep, followDep);
                // std::cout<<ops;
                if(ops.getEdgeIdx().size())
                    _all_ops.push_back(ops);
            }
            uint32_t iter_v = order[dep+1];
            const VertexSetIR &iter_vIR = vertexIRs[iter_v];
            VertexSetIR iter_vset = VertexSetIR(iter_vIR, dep, dep+1);
            if(iter_vset.getEdgeIdx().size() <= 0){
                printf("Invalid schedule: iter_vset at depth %d is empty\n", dep+1);
                exit(-1);
            }
            _iter_tmp.at(dep+1) = iter_vset;
        }

        std::sort(_all_ops.begin(), _all_ops.end());

        uint32_t op_id = 0;
        VertexSetIR &op0 = _all_ops[0];
        op0.setVSetID(op_id);
        _tot_setOpsArr.push_back(op0);
        op_id++;
        for(int i=1; i<_all_ops.size(); i++){
            int isFirst = 1;
            for(int j=0; j<i; j++){
                if(_all_ops[i] == _all_ops[j]){
                    isFirst = 0;
                    break;
                }
            }

            if(isFirst){
                VertexSetIR &op = _all_ops[i];
                op.setVSetID(op_id);
                _tot_setOpsArr.push_back(op);
                op_id++;
            }
        }

        _all_ops.clear();
        _tot_op = _tot_setOpsArr.size();

        for(int i=1; i<_tot_setOpsArr.size(); i++){
            auto &op = _tot_setOpsArr[i];
            const auto &edge = op.getEdgeIdx();
            std::vector<VertexID> prefix(edge.begin(), edge.begin() + edge.size() - 1);
            for(int j=0; j<i; j++){
                if(prefix == _tot_setOpsArr[j].getEdgeIdx() && op.getLabel() == _tot_setOpsArr[j].getLabel()){
                    op.setParentVSetID(_tot_setOpsArr[j].getVSetID());
                    break;
                }
            }
        }

        _setOps.resize(qvcnt);
        for(int i=0; i<qvcnt; i++)
            _setOps.at(i).clear();

        _iterSet.resize(qvcnt, INVALID);    

        for(int i=0; i<_tot_setOpsArr.size(); i++){
            const auto &op = _tot_setOpsArr[i];
            _setOps.at(op.getDepth()).push_back(op.getVSetID());
        }

        for(int i=0; i<_iter_tmp.size(); i++){
            const auto &op_tmp = _iter_tmp[i];
            for(const auto &op: _tot_setOpsArr){
                if(op_tmp == op){
                    _iterSet[i] = op.getVSetID();
                    break;
                }
            }
        }

        // assert(0);
    }

    const std::vector<VertexID>& getOrder() const{
        return _order;
    }

    const std::vector<std::vector<VSetId>>& getSetOps() const{
        return _setOps;
    }
    
    const std::vector<VSetId>& getIterVSet() const{
        return _iterSet;
    }

    const VSetId getIterVSetAt(Depth depth) const{
        return _iterSet[depth];
    }

    const std::vector<VertexSetIR>& getTotSetOpsArr() const{
        return _tot_setOpsArr;
    }

    const uint32_t getTotOp() const {return _tot_op;}

    const std::vector<VertexSetIR> getSetOpsArray() const {return _tot_setOpsArr;} 

    bool isLastOp(uint32_t cur_dep) const{
        return cur_dep == _iterSet.size()-2;
    }

    std::string debugOutput() const{
        std::ostringstream out;
        for(auto op: _tot_setOpsArr)
            out << op.debugOutput(_order);
        return out.str();
    }
};



std::string indentGen(int depth){
    return std::string(depth, '\t');
}

std::string includeGen(Context &ctx){
    std::ostringstream code;
    // code << "#include \"command.h\"\n";
    code << "#include <iostream>\n";
    code << "#include \"graph.h\"\n";
    code << "#include \"code_gen.h\"\n";
    code << "#include \"vertexset.h\"\n";
    code << "#include \"common_type.h\"\n";

    return code.str();
}

std::string readNebGen(const PlanIR &plan, uint32_t dep){
    std::ostringstream code;
    VertexID cur_v = plan.getOrder()[dep];
    if(dep > 0){
        const auto &iter_sets = plan.getIterVSet();
        const auto &iter_op = plan.getTotSetOpsArr()[iter_sets.at(dep)];
        code << indentGen(dep+1) << fmt::format("VertexID v{id} = vsets[{iter_id}].getData(v{id}_id);\n",
                                                fmt::arg("id", cur_v),
                                                fmt::arg("iter_id", iter_op.getVSetID()));
        code << indentGen(dep+1) << fmt::format("if(embeddings.hasVertex(v{})) continue;\n", cur_v);
        code << indentGen(dep+1) << fmt::format("embeddings.push(v{});\n", cur_v);
    }
    code << indentGen(dep+1) << fmt::format("uint32_t v{id}_neb_cnt = 0;\n", fmt::arg("id", cur_v));
    code << indentGen(dep+1) << fmt::format("const VertexID* v{id}_neb = data.getNeb(v{id}, v{id}_neb_cnt);\n",
                                            fmt::arg("id", cur_v));
    code << indentGen(dep+1) << fmt::format("v_neb_set[{id}].Init(v{id}_neb, v{id}_neb_cnt);\n",
                                            fmt::arg("id", cur_v));
    code << indentGen(dep+1) << fmt::format("v_neb_set[{id}].setVid(v{id});\n", fmt::arg("id", cur_v));

    return code.str();
}

std::string opGen(const PlanIR &plan, const VertexSetIR &op, int cur_dep, bool isProfile=false){
    std::string code;
    uint32_t pvsetId = op.getParentVSetID();
    
    // const auto &op_e = op.getEdgeIdx();
    int op_dep = op.getDepth();
    VertexID op_dep_v = plan.getOrder()[op_dep];

    if(pvsetId != INVALID){

       // prefix reuse    
        if(isProfile){
            code += fmt::format("auto _s{depth} = std::chrono::high_resolution_clock::now();\n", fmt::arg("depth", op.getDepth()+1));
        }
        code += fmt::format("vsets[{op_id}].IntersecOf(vsets[{parent_id}], v_neb_set[{vid}]);\n",
                    fmt::arg("op_id", op.getVSetID()),
                    fmt::arg("parent_id", pvsetId),
                    fmt::arg("vid", op_dep_v));
        if(isProfile){
            code += fmt::format("auto _e{depth} = std::chrono::high_resolution_clock::now();\n", fmt::arg("depth", op.getDepth()+1));
            code += fmt::format("std::chrono::duration<double> _d{depth} = _e{depth} - _s{depth};\n", fmt::arg("depth", op.getDepth()+1));
            code += fmt::format("Profile::getInst().intersect_time[{depth}] += _d{depth}.count();\n", fmt::arg("depth", op.getDepth()+1));
            code += fmt::format("Profile::getInst().intersect_tot += _d{depth}.count();\n", fmt::arg("depth", op.getDepth()+1));
            code += fmt::format("Profile::getInst().intersect_cnt[{depth}] += 1;\n", fmt::arg("depth", op.getDepth()+1));
        }

        if(plan.isLastOp(cur_dep))
            code += fmt::format("global_ans += subtract(embeddings, vsets[{op_id}]);\n",fmt::arg("op_id", op.getVSetID()));
        
    }
    else{  // no parent
        code += fmt::format("vsets[{op_id}].Init(v_neb_set[{vid}].getDataPtr(), v_neb_set[{vid}].getSize());\n", 
        fmt::arg("op_id", op.getVSetID()),
        fmt::arg("vid", op_dep_v));

        code += fmt::format("if (vsets[{op_id}].getSize() == 0) {{\nembeddings.pop();\ncontinue;\n}}\n", fmt::arg("op_id", op.getVSetID()));
        if(plan.isLastOp(cur_dep))
            code += fmt::format("global_ans += subtract(embeddings, vsets[{op_id}]);\n",fmt::arg("op_id", op.getVSetID()));
    }

    return code;
}

// std::string opGen(const PlanIR &plan, const VertexSetIR &op, bool isProfile=false){
//     std::string code;
//     uint32_t pvsetId = op.getParentVSetID();
    
//     const auto &op_e = op.getEdgeIdx();
//     int op_dep = op.getDepth();
//     VertexID op_dep_v = plan.getOrder()[op_dep];

//     bool has_edge_to_v = op.hasEdge(op_dep_v);

//     if(pvset.has_value()){
//         if(pvset->getDepth() == op.getDepth()){ // reserved for restriction 
            
//         }
//         else if(pvset->getDepth() == op.getDepth() - 1){ // prefix reuse    
//             if(has_edge_to_v){
//                 if(isProfile){
//                     code += fmt::format("auto _s{depth} = std::chrono::high_resolution_clock::now();\n", fmt::arg("depth", op.getDepth()+1));
//                 }
//                 code += fmt::format("vsets[{op_id}].IntersecOf(vsets[{parent_id}], v_neb_set[{vid}]);\n",
//                             fmt::arg("op_id", op.getVSetID()),
//                             fmt::arg("parent_id", pvset->getVSetID()),
//                             fmt::arg("vid", op_dep_v));
//                 if(isProfile){
//                     code += fmt::format("auto _e{depth} = std::chrono::high_resolution_clock::now();\n", fmt::arg("depth", op.getDepth()+1));
//                     code += fmt::format("std::chrono::duration<double> _d{depth} = _e{depth} - _s{depth};\n", fmt::arg("depth", op.getDepth()+1));
//                     code += fmt::format("Profile::getInst().intersect_time[{depth}] += _d{depth}.count();\n", fmt::arg("depth", op.getDepth()+1));
//                     code += fmt::format("Profile::getInst().intersect_tot += _d{depth}.count();\n", fmt::arg("depth", op.getDepth()+1));
//                     code += fmt::format("Profile::getInst().intersect_cnt[{depth}] += 1;\n", fmt::arg("depth", op.getDepth()+1));
//                 }
//             }
//             else{
//                 if(has_res_to_v){ //reserved for restriction

//                 }
//                 else{
//                     code += fmt::format("vsets[{op_id}].RemoveFrom(vsets[{parent_id}], v_neb_set[{vid}].getVid());\n",
//                                         fmt::arg("op_id", op.getVSetID()),
//                                         fmt::arg("parent_id", pvset->getVSetID()),
//                                         fmt::arg("vid", op_dep_v));

//                     // if(plan.isLastOp(op))
//                     //     code += fmt::format("global_ans += vsets[{op_id}].getSize();\n",fmt::arg("op_id", op.getVSetID()));
//                 }
//             }

//             if(plan.isLastOp(op))
//                 code += fmt::format("global_ans += vsets[{op_id}].getSize();\n",fmt::arg("op_id", op.getVSetID()));
//         }
//     }
//     else{  // no parent
//         if(has_res_to_v);
//         else{
//             code += fmt::format("vsets[{op_id}].Init(v_neb_set[{vid}].getDataPtr(), v_neb_set[{vid}].getSize());\n", 
//             fmt::arg("op_id", op.getVSetID()),
//             fmt::arg("vid", op_dep_v));
//         }

//         const auto &order = plan.getOrder();
//         for(int cur_dep=0; cur_dep<op_dep; cur_dep++){
//             VertexID v_at_dep = order[cur_dep];
//             if(op.hasRestriction(v_at_dep)){

//             }
//             else{
//                 code += fmt::format("vsets[{op_id}].RemoveFrom(vsets[{op_id}], v_neb_set[{vid}].getVid())\n;",
//                                     fmt::arg("op_id", op.getVSetID()),
//                                     fmt::arg("vid", v_at_dep));
//             }
//         }

//         code += fmt::format("if (vsets[{op_id}].getSize() == 0) continue;\n", fmt::arg("op_id", op.getVSetID()));
//         if(plan.isLastOp(op))
//             code += fmt::format("global_ans += vsets[{op_id}].getSize();\n",fmt::arg("op_id", op.getVSetID()));
//     }

//     return code;
// }

std::string iterGen(const PlanIR& plan, uint32_t dep){
    std::ostringstream code;
    uint32_t max_dep = plan.getOrder().size() - 1;
    if (dep >= max_dep-1) {
        code << "";
    } else {
        const auto iter_vset_id = plan.getIterVSet()[dep+1];
        VertexID cur_v = plan.getOrder()[dep+1];
        code << fmt::format(
                "for (uint32_t v{id}_id = 0; v{id}_id < vsets[{iter_id}].getSize(); v{id}_id++) {left} // loop-{dep} begin\n",
                fmt::arg("left", "{"),
                fmt::arg("id", cur_v),
                fmt::arg("iter_id", iter_vset_id),
                fmt::arg("dep", dep + 1));
    }
    return code.str();
}

std::string mainGen(const Context &ctx){
    std::ostringstream code;
    code << "int main(int argc, char **argv){ // main begin\n";
    // code << indentGen(1) << "MatchingCommand command(argc, argv);\n";
    code << indentGen(1) << fmt::format("std::string query_g_path = \"{query_g_path}\";\n", fmt::arg("query_g_path", ctx.query_g_path));
    code << indentGen(1) <<  fmt::format("std::string data_g_path = \"{data_g_path}\";\n", fmt::arg("data_g_path", ctx.data_g_path));
    code << indentGen(1) <<  "Graph* data_g = new Graph();\n";
    code << indentGen(1) <<  "data_g->Load(data_g_path);\n";
    code << indentGen(1) <<  "Query* query_g = new Query();\n";
    code << indentGen(1) <<  "query_g->Load(query_g_path);\n";
    code << indentGen(1) <<  "Context ctx;\n";

    code << indentGen(1) << fmt::format("ctx.threadNum = {threadNum};\n", fmt::arg("threadNum", ctx.threadNum));
    std::string orderInstance = fmt::format("{{{}}}", fmt::join(ctx.order, ", "));
    std::string orderPrintInst = fmt::format("{}", fmt::join(ctx.order, " "));
    // code << indentGen(1) << fmt::format("std::cout << \"threadNum = {threadNum}\\n\";\n", fmt::arg("threadNum", ctx.threadNum));
    code << indentGen(1) << fmt::format("std::vector<VertexID> order = {orderIns};\n", fmt::arg("orderIns", orderInstance));
    // code << indentGen(1) << fmt::format("std::cout << \"order: {orderIns}\\n\";\n", fmt::arg("orderIns", orderPrintInst));
    code << indentGen(1) << "PlanIR plan(order, *query_g);\n";
    code << indentGen(1) << "unsigned long long ans = query(plan, *data_g, ctx);\n";
    // code << indentGen(1) << "std::cout<< \"ans: \"<< ans << \"\\n\";\n";
    code << indentGen(1) << fmt::format("std::string orderInst = \"{orderInst}\";\n", fmt::arg("orderInst", orderPrintInst));
    if(ctx.isProfile){
        code << indentGen(1) << "std::vector<std::string> intersec_time_format(query_g->getVertexCnt());\n";
        code << indentGen(1) << "std::vector<double> inter_time(query_g->getVertexCnt());\n";
        code << indentGen(1) << "std::vector<uint32_t> inter_cnt(query_g->getVertexCnt());\n";
        code << indentGen(1) << "for(int i=0; i<query_g->getVertexCnt();i++){\n";
        code << indentGen(1) << "inter_time[order[i]] = Profile::getInst().intersect_time[i];\n";
        code << indentGen(1) << "inter_cnt[order[i]] = Profile::getInst().intersect_cnt[i];\n";
        code << indentGen(1) << "}\n";
        
        code << indentGen(1) << "for(int i=0; i<query_g->getVertexCnt(); i++){\n";
        code << indentGen(1) << "intersec_time_format[i] = fmt::format(\"{:.2f}\", inter_time[i]);\n";
        code << indentGen(1) << "}\n";
        
        code << indentGen(1) << "std::string intersecTimeInst = fmt::format(\"{}\", fmt::join(intersec_time_format, \",\"));\n";
        code << indentGen(1) << "std::string intersecCntInst = fmt::format(\"{}\", fmt::join(inter_cnt, \",\"));\n";
        
        code << indentGen(1) << "printf(\"%.2f,%llu,%s,%d,%.2f,%s,%s\", ctx.time, ans, orderInst.c_str(), plan.getTotOp(),Profile::getInst().intersect_tot, intersecTimeInst.c_str(), intersecCntInst.c_str());\n";
    }
    else{
        code << indentGen(1) << "printf(\"%.2f,%llu,%s,%d\",ctx.time,ans,orderInst.c_str(),plan.getTotOp());\n";
    }
    
    code <<"} // main end \n";
    return code.str();
}

// void depthGen(const PlanIR &plan, const Context &ctx, int cur_depth){

//     std::ostringstream code;

//     VertexID v_id = plan.getOrder()[cur_depth];

//     code << fmt::format("uint32_t v{}_neb_cnt;\n", v_id);
//     code << fmt::format("const v")
    
    

// }

std::string queryGen(const PlanIR& plan, const Context& ctx){
    std::ostringstream code;
    uint32_t max_dep = plan.getOrder().size()-1;
    // uint32_t max_dep = plan.getOrder().size();
    const auto& setOps = plan.getSetOps();
    VertexID first_v = plan.getOrder()[0];

    code << "unsigned long long query(const PlanIR& plan, const Graph &data, Context& ctx){ // query begin\n";
    code << "unsigned long long global_ans = 0;\n";
    if(ctx.isProfile) code << "Profile::getInst().reset(plan.getOrder().size());\n";
    code << "auto start = std::chrono::high_resolution_clock::now();\n";
    code << "#pragma omp parallel num_threads(ctx.threadNum) reduction(+: global_ans)\n{\n";
    code << "VertexSet embeddings;\n";
    code << "embeddings.alloc(plan.getOrder().size());\n";
    code << "uint32_t totOp = plan.getTotOp();\n";
    code << "VertexSet *vsets = new VertexSet[totOp];\n";
    code << "for(int i=0; i<totOp; i++) vsets[i].alloc(data.getMaxDgree());\n";
    code << fmt::format("VertexSet *v_neb_set = new VertexSet[{tot_iter}];\n", fmt::arg("tot_iter", plan.getOrder().size()));
    code << fmt::format("for(int i=0; i<{tot_iter}; i++) v_neb_set[i].alloc(data.getMaxDgree());\n", fmt::arg("tot_iter", plan.getOrder().size()));
    code << "#pragma omp for schedule(dynamic) nowait\n";
    code << fmt::format("for (uint32_t v{id} = 0; v{id} < data.getVertexCnt(); v{id}++) {{ // loop-0 begin\n",
                        fmt::arg("id", first_v));
    code << fmt::format("embeddings.push(v{});\n", first_v);
    
    for (int dep = 0; dep < max_dep; dep++) {
        // code for reading adj from the graph
        code << readNebGen(plan, dep);
        // code for computation at this loop
        const auto &ops_id = setOps.at(dep);
        for (const auto op_id: ops_id) {
            const auto& op = plan.getTotSetOpsArr()[op_id];
            code << opGen(plan, op, dep, ctx.isProfile) << op.debugOutput(ctx.order);
        }
        // code for iterating next loop
        if (dep == max_dep - 1) continue;
        code << iterGen(plan, dep);
    }

    for(int dep = max_dep-1; dep >=0; dep--){
        code << "embeddings.pop();\n";
        code << "} // loop-" << std::to_string(dep) << " end\n";
    }

    
    // code << "for(int i=0; i<totOp; i++) {delete *(vsets+i);}\n"; 
    code << "delete[] vsets;\n";
    code << "} // omp-end\n";
    code << "auto end = std::chrono::high_resolution_clock::now();\n";
    code << "std::chrono::duration<double> duration = end - start;\n";
    
    code << "ctx.time = duration.count();\n";
    // code << "printf(\"Execution time: %.2f s\\n\", duration.count());\n";
    code << "return global_ans;\n";
    code << "} // query-end\n" << "\n\n";

    return code.str();
}

std::string subtractGen(){
    std::ostringstream code;

    code << "uint64_t subtract(const VertexSet &embeddings, const VertexSet &lastVSet){\n";
    code << "uint64_t res = lastVSet.getSize();\n";
    code << "for(int i=0; i<embeddings.getSize(); i++){\n";
    code << "    const auto data_ptr = lastVSet.getDataPtr();\n";
    code << "    if(std::binary_search(data_ptr, data_ptr + lastVSet.getSize(), embeddings.getData(i)))\n";
    code << "    res--;\n";
    code << "}\n";
    code << "return res;\n";
    code << "}\n";

    return code.str();
}

std::string codeGen(const PlanIR& plan, Context &ctx){
    std::ostringstream code;

    code << includeGen(ctx);

    code << subtractGen();

    code << queryGen(plan, ctx);
    
    code << mainGen(ctx);

    return code.str();
}

#endif