#include "AStar.hpp"
#include "NPuzzle.hpp"
#include <iostream>

using namespace std;

int main() {
    constexpr size_t N = 5;
    using Problem = NPuzzle<N>;
    using State = Problem::State;

    //for (const auto& board : State::Scrambled(20, true)) {
    //    cout << board << endl;
    //}

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

    return 0;
}