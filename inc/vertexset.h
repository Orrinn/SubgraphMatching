#pragma once

#include "common_type.h"
#include "common_ops.h"

class VertexSet{
public:

    VertexSet() = default;

    void alloc(uint32_t capacity){
        if(_allocate && _data!=nullptr && _cap >= capacity)
            _size = 0;
        else{
            _size = 0;
            _allocate = true;
            _data = new VertexID[capacity];
            _cap = capacity;
        }
    }

    void Init(VertexID* data, uint32_t size){

        if(_allocate && _data != nullptr)
            delete[] _data;
        
        _allocate = false;
        _size = size;
        _data = data;

        // for(int i=0; i<size; i++) _data[i] = data[i];
        // _size = size;
    }

    ~VertexSet(){
        if(_allocate && _data!=nullptr)
            delete[] _data;
    }

    uint32_t IntersecOf(const VertexSet &set1, const VertexSet &set2){
        Intersection(set1.getDataPtr(), set1.getSize(), set2.getDataPtr(), set2.getSize(), _data, _size);
        return _size;
    }

    uint32_t RemoveFrom(const VertexSet &set1, uint32_t u){
        uint32_t l = 0;
        uint32_t size1 = set1.getSize();
        const VertexID* data1 = set1.getDataPtr();
        if (size1 > 64) {
            uint32_t count = size1;
            while (count > 0) {
                uint32_t it = l;
                uint32_t step = count / 2;
                it += step;
                if (data1[it] < u) {
                    l = ++it;
                    count -= step + 1;
                }
                else count = step;
            }
        }
        else {
            while (l < size1 && data1[l] < u) l++;
        }

        if(l<size1 && data1[l] == u){
            _size = size1-1;
            for(int i=0; i<l; i++) _data[i] = set1.getData(i);
            for(int i=l; i<_size; i++) _data[i] = set1.getData(i+1);
        }
        else{
            // _data = const_cast<VertexID*>(data1);
            for(int i=0; i<size1; i++)
                _data[i] = set1.getData(i);
            _size = size1;
        }

        _vid = INVALID;
        return _size;
    }

    bool hasVertex(VertexID u) const{
        for(int i=0; i<_size; i++){
            if(_data[i] == u)
                return true;
        }
        return false;
    }

    const VertexID* getDataPtr() const{return _data;}
    const uint32_t getSize() const{return _size;}
    const uint32_t getVid() const{return _vid;}
    void setVid(uint32_t v) {_vid = v;}
    const uint32_t getCapacity() const{return _cap;}
    const VertexID getData(uint32_t id) const{return _data[id];}
    void push(VertexID vid){
        assert(_size < _cap);
        _data[_size++] = vid;
    }
    VertexID pop(){
        assert(_size > 0);
        return _data[_size--];
    }


private:
    VertexID* _data{nullptr};
    uint32_t _size{0};
    uint32_t _cap{0};
    VertexID _vid{INVALID};
    bool _allocate{false};
};