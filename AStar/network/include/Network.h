#ifndef NETWORK_H_
#define NETWORK_H_
#include "../../module/src/NPuzzle.hpp"
#include <memory>

class Network {
public:
    using Problem = NPuzzle<5>;
    using Board = Problem::State;
    using Value = Problem::Value;

    static std::shared_ptr<Network> Create(std::string type);

    virtual ~Network() {};
    
    virtual Value predict(const Board& board, const Board& goal) = 0;
};

#endif // !NETWORK_H_