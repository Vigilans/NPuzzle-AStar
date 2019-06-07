#include <iostream>
#include <filesystem>
#include <cassert>
#include "../module/src/AStar.hpp"
#include "../module/src/NPuzzle.hpp"
#include "../network/include/Network.h"

using namespace std;
using namespace std::filesystem;

constexpr size_t N = 5;
using Problem = NPuzzle<N>;
using State = Problem::State;

void ScrambleTest() {
    for (const auto& board : State::Scrambled(20, true)) {
        cout << board << endl;
    }
}

void SolutionTest() {
    auto problem = Problem{ &HammingDistance<N> };
    //auto problem = Problem{ &ManhattanDistance<N> };
    auto astar = AStarSearch{ problem };

    auto states = State::Scrambled(30, true);
    auto start = states.back();
    //auto start = State::Random();

    auto path = astar.findPath(start, State::Ordered());

    for (const auto& board : path) {
        cout << board << endl;
    }

    cout << "Path length: " << path.size() << endl;
    cout << "States explorered: " << astar.statesExplorered() << endl;
    cout << "Time Elapsed: " << astar.timeElapsed().count() << "ms" << endl;

    astar.clear();
}

void GenerateTest() {
    auto problem = Problem{ &ManhattanDistance<N> };
    auto astar = AStarSearch{ problem };
    auto dataset = std::vector<State>();
    while (dataset.size() < 200000) {
        auto boards = State::Scrambled(30, true);
        if (boards.size() != 31) exit(1);
        auto path = astar.findPath(boards.back(), State::Ordered());
        dataset.insert(dataset.end(), path.begin(), path.end());
        cout << "Path length " << path.size() << ", ";
        cout << "States " << astar.statesExplorered() << ", ";
        cout << "Time " << astar.timeElapsed().count() << "ms" << ", ";
        cout << "Current dataset size: " << dataset.size() << endl;
        astar.clear();
    }
    cout << "Dataset generated with size " << dataset.size() << endl;
}

void NetworkTest() {
    auto mlp = Network::Create("mlp");
    auto problem = Problem{ [mlp](auto& a, auto& b) { return mlp->predict(a, b); } };
    auto astar = AStarSearch{ std::move(problem) };
    for (int i = 0; i < 10; ++i) {
        auto boards = State::Scrambled(10, true);
        cout << "Predicted length: " << mlp->predict(boards.back(), State::Ordered()) << ", ";
        auto path = astar.findPath(boards.back(), State::Ordered());
        cout << "Path length " << path.size() - 1 << ", ";
        cout << "States " << astar.statesExplorered() << ", ";
        cout << "Time " << astar.timeElapsed().count() << "ms" << "\n";
        astar.clear();
        //cout << mlp->predict(State::Ordered(), State::Ordered()) << endl;
    }
}

int main() {
    //ScrambleTest();
    //SolutionTest();
    //GenerateTest();
    NetworkTest();
    return 0;
}