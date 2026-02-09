#pragma once
#include <stdint.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cstdio>
#include <algorithm>
#include "common_type.h"


class Edges {
public:
    uint32_t* _offset;
    uint32_t* _edge;    // v = candidates[u_nbr][_edge[idx]]  // store the vertices index in candidates 
    uint32_t* _edge_v;  // v = _edge_v[idx] // store the vertices in candidates
    uint32_t _v_cnt;
    uint32_t _e_cnt;
    uint32_t _max_degree;
public:
    Edges() {
        _offset = NULL;
        _edge = NULL;
        _edge_v = NULL;
        _v_cnt = 0;
        _e_cnt = 0;
        _max_degree = 0;
    }

    ~Edges() {
#ifdef LOG_OUTPUT
        spdlog::trace("Edges release");
#endif
        delete[] _offset;
        delete[] _edge;
        delete[] _edge_v;
    }

    const inline uint32_t* getNeb(const uint32_t vertex_id, uint32_t &cnt){
        cnt = _offset[vertex_id+1] - _offset[vertex_id];
        return _edge + _offset[vertex_id];
    }

    const inline uint32_t* getNeb_V(const uint32_t vertex_id, uint32_t &cnt){
        cnt = _offset[vertex_id+1] - _offset[vertex_id];
        return _edge_v + _offset[vertex_id];        
    }
};

class Graph{

protected:
    std::string path;
    uint32_t _v_cnt{0};
    uint32_t _e_cnt{0};
    uint32_t _label_cnt{0};
    uint32_t _max_degree{0};
    uint32_t _max_label_freq{0};

    uint32_t* _offset{nullptr};
    VertexID* _neb{nullptr};
    LabelID* _labels{nullptr};

    uint32_t* _label_offset{nullptr};
    VertexID* _neb_label{nullptr};

    VertexID* _vertex_by_labels{nullptr};
    uint32_t* _vertex_by_labels_offset{nullptr};

    float** _eigenValue{nullptr};

    std::unordered_map<LabelID, uint32_t> _labels_freq;
    std::unordered_map<LabelID, uint32_t>* _nlf;

    void buildNLF(){
        _nlf = new std::unordered_map<LabelID, uint32_t>[_v_cnt];
        for (int i = 0; i < _v_cnt; ++i) {
            uint32_t count;
            const VertexID * neighbors = getNeb(i, count);

            for (int j = 0; j < count; ++j) {
                VertexID u = neighbors[j];
                LabelID label = getVertexLabel(u);
                if (_nlf[i].find(label) == _nlf[i].end()) {
                    _nlf[i][label] = 0;
                }

                _nlf[i][label] += 1;
            }
        }
    }

    void buildLabelIndex(){
        _vertex_by_labels = new VertexID[_v_cnt];
        _vertex_by_labels_offset = new uint32_t[_label_cnt + 1];
        _vertex_by_labels_offset[0] = 0;

        uint32_t total = 0;
        for (int i = 0; i < _label_cnt; ++i) {
            _vertex_by_labels_offset[i + 1] = total;
            total += _labels_freq[i];
        }

        for (int i = 0; i < _v_cnt; ++i) {
            LabelID label = _labels[i];
            _vertex_by_labels[_vertex_by_labels_offset[label + 1]++] = i;
        }
    }

    void BuildLabelOffset() {
        size_t labels_offset_size = (size_t)_v_cnt * _label_cnt + 1;
        _label_offset = new uint32_t[labels_offset_size];
        std::fill(_label_offset, _label_offset + labels_offset_size, 0);
    
        _neb_label = new VertexID[_e_cnt * 2];
        memcpy(_neb_label, _neb, sizeof(VertexID) * _e_cnt * 2);

        for (uint32_t i = 0; i < _v_cnt; ++i) {
            std::sort(_neb_label + _offset[i], _neb_label + _offset[i + 1],
                [this](const VertexID u, const VertexID v) -> bool {
                    return _labels[u] == _labels[v] ? u < v : _labels[u] < _labels[v];
                });
        }
    
        for (uint32_t i = 0; i < _v_cnt; ++i) {
            LabelID previous_label = 0;
            LabelID current_label = 0;
    
            labels_offset_size = i * _label_cnt;
            _label_offset[labels_offset_size] = _offset[i];
    
            for (uint32_t j = _offset[i]; j < _offset[i + 1]; ++j) {
                current_label = _labels[_neb_label[j]];
    
                if (current_label != previous_label) {
                    for (uint32_t k = previous_label + 1; k <= current_label; ++k) {
                        _label_offset[labels_offset_size + k] = j;
                    }
                    previous_label = current_label;
                }
            }
    
            for (uint32_t l = current_label + 1; l <= _label_cnt; ++l) {
                _label_offset[labels_offset_size + l] = _offset[i + 1];
            }
        }
    }


public:
    Graph(){
        _v_cnt = _e_cnt = _label_cnt = _max_degree = 0;
        _offset = nullptr;
        _neb = nullptr;
        _labels = nullptr;
        _nlf = nullptr;
        _vertex_by_labels = _vertex_by_labels_offset = nullptr;
        _label_offset = nullptr;
        _neb_label = nullptr;
        _eigenValue = nullptr;

        _labels_freq.clear();
    }

    virtual ~Graph(){
#ifdef LOG_OUTPUT
        if(typeid(*this) == typeid(Graph))
            spdlog::trace("data graph deconstructed");
        else
            spdlog::trace("query graph deconstructed");
#endif
        if(_offset){delete[] _offset; _offset=nullptr;}
        if(_neb){delete[] _neb; _neb=nullptr;}
        if(_labels){delete[] _labels; _labels=nullptr;}
        if(_nlf){delete[] _nlf; _nlf=nullptr;}
        if(_vertex_by_labels){delete[] _vertex_by_labels; _vertex_by_labels=nullptr;}
        if(_vertex_by_labels_offset){delete[] _vertex_by_labels_offset; _vertex_by_labels_offset=nullptr;}
        if(_label_offset){delete[] _label_offset; _label_offset=nullptr;}
        if(_neb_label){delete[] _neb_label; _neb_label=nullptr;}
        if(_eigenValue){
            for (int i = 0; i < _v_cnt; ++i) {
                delete[] _eigenValue[i];
            }
            delete[] _eigenValue;
            _eigenValue = nullptr;
        }
    }

    virtual void Load(const std::string& file_path){

        std::ifstream infile(file_path);

        if(!infile.is_open()){
            std::cout<<"can't open "<< file_path <<'\n';
            exit(-1);
        }

        path = file_path;

        char type;

        infile>>type>>_v_cnt>>_e_cnt;

        _offset = new uint32_t[_v_cnt+1];
        _neb = new VertexID[_e_cnt*2];
        _labels = new LabelID[_v_cnt];

        memset(_offset, 0, sizeof(uint32_t)*(_v_cnt+1));
        memset(_neb, 0, sizeof(VertexID)*(_e_cnt*2));
        memset(_labels, 0, sizeof(LabelID)*(_v_cnt));

        _offset[0] = 0;

        std::vector<uint32_t> neighbors_offset(_v_cnt, 0);
        LabelID max_label_id=0;

        while (infile >> type)
        {
            if(type == 'v'){
                VertexID id;
                LabelID label;
                uint32_t degree;
                infile>>id>>label>>degree;
                _labels[id] = label;
                _max_degree = degree > _max_degree? degree: _max_degree;
                _offset[id + 1] = _offset[id] + degree;

                if(_labels_freq.find(label) == _labels_freq.end()){
                    _labels_freq[label] = 0;
                    max_label_id = max_label_id>label?max_label_id:label;
                }

                _labels_freq[label] += 1;
                
            }
            else{
                VertexID u,v;
                infile >> u >> v;
                
                uint32_t offset = _offset[u] + neighbors_offset[u];
                _neb[offset] = v;

                offset = _offset[v] + neighbors_offset[v];
                _neb[offset] = u;

                neighbors_offset[u] += 1;
                neighbors_offset[v] += 1;
            }
        }
        
        infile.close();
        _label_cnt = _labels_freq.size() > (max_label_id + 1) ? _labels_freq.size() : max_label_id + 1;

        for (auto e : _labels_freq)
            _max_label_freq = _max_label_freq > e.second? _max_label_freq: e.second;
        

        for (int i = 0; i < _v_cnt; ++i) 
            std::sort(_neb + _offset[i], _neb + _offset[i+1]);

        buildLabelIndex();
        buildNLF();
        BuildLabelOffset();

    }

    void LoadEigenIndex(std::string fileToOpen){
        // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
        // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

        // the input is the file: "fileToOpen.csv":
        // a,b,c
        // d,e,f
        // This function converts input file data into the Eigen matrix format

        // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
        // M=[a b c
        //    d e f]
        // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
        // later on, this vector is mapped into the Eigen matrix format
        _eigenValue = new float*[_v_cnt];
        for (int i = 0; i < _v_cnt; ++i) {
            _eigenValue[i] = new float[35];
        }

        // in this object we store the data from the matrix
        std::ifstream matrixDataFile(fileToOpen);

        // this variable is used to store the row of the matrix that contains commas
        std::string matrixRowString;

        // this variable is used to store the matrix entry;
        std::string matrixEntry;

        // this variable is used to track the number of rows
        int matrixRowNumber = 0;
        int colNum = 0;

        while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
        {
            std::stringstream matrixRowStringStream(matrixRowString); // convert matrixRowString that is a string to a stream variable.
            colNum = 0;
            while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
            {
                this->_eigenValue[matrixRowNumber][colNum] = stof(matrixEntry);
                colNum++;
            }
            matrixRowNumber++; // update the column numbers
        }
    }

    float** getEigenValue() const {
        // assert(_eigenValue != nullptr);
        return _eigenValue;
    }


    const uint32_t getVertexCnt() const{
        return _v_cnt;
    };

    const uint32_t getVertexDegree(const VertexID id) const {
        return _offset[id + 1] - _offset[id];
    }

    const uint32_t getEdgeCnt() const{
        return _e_cnt;
    };

    const uint32_t getLabelCnt() const{
        return _label_cnt;
    };

    const uint32_t getMaxDgree() const{
        return _max_degree;
    };

    const uint32_t getMaxLabelFreq() const {
        return _max_label_freq;
    }

    const uint32_t getLabelFreq(LabelID label) const{
        return _labels_freq.find(label) == _labels_freq.end() ? 0 : _labels_freq.at(label);
    }

    const VertexID* getNeb(VertexID id, uint32_t &neb_count) const{
        neb_count = _offset[id+1] - _offset[id];
        return _neb + _offset[id];
    };

    const VertexID *getNebByLabel(const VertexID id, uint32_t &count, const LabelID label) const
    {
        uint32_t offset = id * _label_cnt + label;
        count = _label_offset[offset + 1] - _label_offset[offset];
        return _neb_label + _label_offset[offset];
    }

    const LabelID getVertexLabel(VertexID id) const{
        return _labels[id];
    }

    const LabelID* getVertexLabels() const{
        return _labels;
    }

    const std::unordered_map<LabelID, uint32_t>* getVertexNLF(VertexID id) const{
        return _nlf + id;
    }

    const VertexID* getVertexByLabel(const LabelID id, uint32_t& count) const {
        count = _vertex_by_labels_offset[id + 1] - _vertex_by_labels_offset[id];
        return _vertex_by_labels + _vertex_by_labels_offset[id];
    }

    const bool hasEdge(VertexID u, VertexID v) const{
        return std::binary_search(_neb + _offset[u], _neb + _offset[u+1], v);
    }

};

class Query: public Graph{

    uint32_t _kcore_length{0};
    int* _kcore_value{nullptr};

    void KCore(const Query &query, int *core_table){
        int vertices_count = query.getVertexCnt();
        int max_degree = query.getMaxDgree();

        int* vertices = new int[vertices_count];          // Vertices sorted by degree.
        int* position = new int[vertices_count];          // The position of vertices in vertices array.
        int* degree_bin = new int[max_degree + 1];      // Degree from 0 to max_degree.
        int* offset = new int[max_degree + 1];          // The offset in vertices array according to degree.

        std::fill(degree_bin, degree_bin + (max_degree + 1), 0);

        for (int i = 0; i < vertices_count; ++i) {
            int degree = query.getVertexDegree(i);
            core_table[i] = degree;
            degree_bin[degree] += 1;
        }

        int start = 0;
        for (int i = 0; i < max_degree + 1; ++i) {
            offset[i] = start;
            start += degree_bin[i];
        }

        for (int i = 0; i < vertices_count; ++i) {
            int degree = query.getVertexDegree(i);
            position[i] = offset[degree];
            vertices[position[i]] = i;
            offset[degree] += 1;
        }

        for (int i = max_degree; i > 0; --i) {
            offset[i] = offset[i - 1];
        }
        offset[0] = 0;

        for (int i = 0; i < vertices_count; ++i) {
            int v = vertices[i];

            uint32_t count;
            const VertexID * neighbors = query.getNeb(v, count);

            for(int j = 0; j < count; ++j) {
                int u = neighbors[j];

                if (core_table[u] > core_table[v]) {

                    // Get the position and vertex which is with the same degree
                    // and at the start position of vertices array.
                    int cur_degree_u = core_table[u];
                    int position_u = position[u];
                    int position_w = offset[cur_degree_u];
                    int w = vertices[position_w];

                    if (u != w) {
                        // Swap u and w.
                        position[u] = position_w;
                        position[w] = position_u;
                        vertices[position_u] = w;
                        vertices[position_w] = u;
                    }

                    offset[cur_degree_u] += 1;
                    core_table[u] -= 1;
                }
            }
        }

        delete[] vertices;
        delete[] position;
        delete[] degree_bin;
        delete[] offset;
    }

    void buildKCore(){
        _kcore_value = new int[_v_cnt];
        memset(_kcore_value, 0, sizeof(int) * _v_cnt);
        KCore(*this, _kcore_value);

        for (VertexID i = 0; i < _v_cnt; ++i) {
            if (_kcore_value[i] > 1) {
                _kcore_length += 1;
            }
        }
    }

public: 

    void Load(const std::string& file_path) override{
        Graph::Load(file_path);

        buildKCore();
    }

    ~Query() override{
        if(_kcore_value != nullptr){delete[] _kcore_value; _kcore_value = nullptr;}
#ifdef LOG_OUTPUT
            spdlog::trace("query graph deconstructed");
#endif
    }

    uint32_t getKCoreLength() const{
        return _kcore_length;
    }

    uint32_t getKCoreValue(VertexID v) const{
        return _kcore_value[v];
    }

};

class TreeNode {
public:
    VertexID _id=0;
    uint32_t _level=0;

    VertexID _parent;
    std::vector<VertexID> _children;
    size_t _estimated_embeddings_num;
public:
    TreeNode() {
        _id = 0;
        _parent = 0;
        _level = 0;
        _estimated_embeddings_num = 0;
        _children.clear();
    }

    ~TreeNode() {
        _children.clear();
    }
};
