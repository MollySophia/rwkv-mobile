#ifndef BACKEND_H
#define BACKEND_H

#include <string>
#include <vector>

namespace rwkvmobile {

class execution_provider {
public:
    virtual int init(void * extra) { }
    virtual int init(std::string model_path, void * extra) { }
    virtual int load_model(std::string model_path) = 0;
    virtual int eval(int id, std::vector<float> &logits) = 0;
    virtual int eval(std::vector<int> ids, std::vector<float> &logits) = 0;
    virtual int get_state(std::vector<float> &state) { }
    virtual int set_state(std::vector<float> state) { }
    virtual int release_model() { };
    virtual int release() { };
};

}

#endif