#ifndef ASTAR_HPP_
#define ASTAR_HPP_
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "lib/fiboheap/fiboqueue.h"

namespace ch = std::chrono;

template <typename State, typename Value>
struct AStarNode {
    AStarNode* parent = nullptr;
    Value g = 0.0;
    Value h = 0.0;
    State state = State();
    Value f() { return g + h; }
};

namespace std {
template <typename State, typename Value>
struct less<AStarNode<State, Value>*> {
    using Node = AStarNode<State, Value>;
    bool operator()(Node* left, Node* right) const {
        return left->f() < right->f();
    }
};
}

template <class Duration>
struct DurationGuard {
    Duration& duration;
    ch::time_point<ch::system_clock> start;
    DurationGuard(Duration& d) : duration(d), start(ch::system_clock::now()) {}
    ~DurationGuard() { duration = ch::duration_cast<Duration>(ch::system_clock::now() - start); }
};

template <typename Problem>
class AStarSearch { 
public:
    using State = typename Problem::State;
    using Value = typename Problem::Value;
    using Node  = AStarNode<State, Value>;

public:
    AStarSearch(Problem problem) : m_problem(problem) {}

    std::vector<State> findPath(State init, State goal) {
        // Initializing
        DurationGuard durationGuard(m_searchTime);
        auto initNode = getNode(std::move(init));
        auto goalNode = getNode(std::move(goal));
        initNode->h = m_problem.h(initNode->state, goalNode->state);
        this->addOpen(initNode);

        // Running
        while (true) {
            // Pop node with best f value
            auto current = this->popOpen();
            this->addClose(current);

            // Prepare adjacent nodes
            for (auto& state : m_problem.getNeighbors(current->state)) {
                auto neighbor = getNode(std::move(state));
                auto g = current->g + m_problem.g(current->state, neighbor->state);
                if (inClose(neighbor)) {
                    continue;
                } else if (inOpen(neighbor)) {
                    if (neighbor->g > g) { // Found a better path
                        neighbor->g = g;
                        neighbor->parent = current;
                        updateOpen(neighbor);
                    } // h value has been calced
                } else {
                    neighbor->parent = current;
                    neighbor->g = g;
                    neighbor->h = m_problem.h(neighbor->state, goalNode->state);
                    this->addOpen(neighbor);
                }
            }

            // Stop check
            if (m_open.empty()) {
                return std::vector<State>();
            }
            if (inOpen(goalNode)) {
                std::vector<State> path;
                auto pathNode = goalNode;
                while (pathNode != nullptr) {
                    path.push_back(std::move(pathNode->state));
                    pathNode = pathNode->parent;
                }
                std::reverse(path.begin(), path.end());
                return path;
            }
        }
    }

    std::size_t statesExplorered() {
        return m_nodePool.size();
    }

    ch::milliseconds timeElapsed() {
        return m_searchTime;
    }

    void clear() {
        m_nodePool.clear();
        m_open.clear();
        m_close.clear();
    }

private:
    Node* getNode(State&& state) {
        auto hash = std::hash<State>()(state);
        auto iter = m_nodePool.find(hash);
        if (iter == m_nodePool.end()) {
            iter = m_nodePool.insert({ hash, std::make_unique<Node>() }).first;
            iter->second->state = std::move(state);
        }
        return iter->second.get(); 
    }

    bool inClose(Node* node) { 
        return m_close.count(node); 
    }

    void addClose(Node* node) {
        m_close.insert(node);
    }

    bool inOpen(Node* node) { 
        return m_open.count(node);
    }

    void addOpen(Node* node) {
        m_open.push(node);
    }

    Node* popOpen() {
        auto minF = m_open.top();
        m_open.pop();
        return minF;
    }

    void updateOpen(Node* node) {
        auto heapNode = m_open.findNode(node);
        m_open.decrease_key(heapNode, node);
    }

private:
    Problem m_problem;
    ch::milliseconds m_searchTime;
    std::unordered_map<std::size_t, std::unique_ptr<Node>> m_nodePool;
    std::unordered_set<Node*> m_close;
    FibQueue<Node*> m_open;
};

#endif // !ASTAR_HPP_
