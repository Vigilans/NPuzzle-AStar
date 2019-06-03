#ifndef NPUZZLE_HPP_
#define NPUZZLE_HPP_
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <bitset>
#include <functional>
#include <iomanip>
#include <sstream>

constexpr int Factorial(int n) {
    if (n == 0) return 1;
    return n * Factorial(n - 1);
}

std::mt19937 Rnd = std::mt19937(time(nullptr));

template <std::size_t N>
struct Board {
    enum Direction {
        Left, Right, Up, Down
    };

    constexpr static std::size_t Size = N * N;

    std::vector<std::size_t> state;
    std::size_t blank;

    static Board Random();
    static Board Ordered();
    static std::vector<Board> Scrambled(int maxSteps, bool fixed = false);

    static std::tuple<int, int> GetOffset(Direction dir) {
        switch (dir) {
            case Left:  return { -1, 0 };
            case Right: return { 1, 0 };
            case Up:    return { 0, -1 };
            case Down:  return { 0, 1 };
        }
    }

    static std::tuple<int, int> GetPose(std::size_t index) {
        return { index % N, index / N };
    }

    static std::size_t GetIndex(int x, int y) {
        return y * N + x;
    }

    bool validMove(Direction dir) const {
        auto [x, y] = GetPose(blank);
        auto [dx, dy] = GetOffset(dir);
        return (x + dx >= 0) && (x + dx < N) && (y + dy >= 0) && (y + dy < N);
    }

    Board move(Direction dir) const {
        auto [x, y] = GetPose(blank);
        auto [dx, dy] = GetOffset(dir);
        auto tile = GetIndex(x + dx, y + dy);
        Board next = *this;
        next.blank = tile;
        std::swap(next.state[blank], next.state[tile]);
        return next;
    }

    std::size_t permutationRank() const { // Myrvold & Ruskey O(n) Pefect hash for board
        std::array<std::size_t, Size> tmp, inv, stack;

        for (int i = 0; i < Size; ++i) {
            tmp[i] = state[i];
            inv[state[i]] = i;
        }

        for (int n = Size; n > 1; --n) {
            int s = tmp[n - 1];
            std::swap(tmp[n - 1], tmp[inv[n - 1]]);
            std::swap(inv[s], inv[n - 1]);
            stack[n - 1] = s;
        }

        for (int i = 2; i < Size; ++i) {
            stack[i] += (i + 1) * stack[i - 1];
        }

        return stack[Size - 1];
    }
};

template <std::size_t N>
constexpr int ManhattanDistance(const Board<N>& a, const Board<N>& b) {
    std::size_t distance = 0;
    // We assume that b is always ordered to speed up calculating
    for (int i = 0; i < Board<N>::Size; ++i) {
        auto [x1, y1] = Board<N>::GetPose(a.state[i]);
        auto [x2, y2] = Board<N>::GetPose(i);
        distance += abs(x2 - x1) + abs(y2 - y1);
    }
    return distance;
}

template <std::size_t N>
constexpr int HammingDistance(const Board<N>& a, const Board<N>& b) {
    std::size_t distance = 0;
    for (int i = 0; i < Board<N>::Size; ++i) {
        distance += (a.state[i] != b.state[i]);
    }
    return distance;
}

template <std::size_t N>
struct NPuzzle {
    using State = Board<N>;
    using Value = int;

    std::function<Value(const State& a, const State& b)> h;

    Value g(const State& a, const State& b) { return 1; }

    std::vector<State>& getNeighbors(const State& s) {
        static std::vector<State> neighbors(4);
        neighbors.clear();
        for (int i = 0; i < 4; ++i) {
            auto dir = State::Direction(i);
            if (s.validMove(dir)) {
                neighbors.push_back(s.move(dir));
            }
        }
        return neighbors;
    };
};

template <std::size_t N>
bool operator==(const Board<N>& lhs, const Board<N>& rhs) {
    return lhs.state == rhs.state;
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& os, const Board<N>& b) {
    const std::size_t width = 1 + (int)std::log10(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            os << std::setw(width) << b.state[b.GetIndex(j, i)] << " ";
        }
        os << "\n";
    }
    return os;
}

namespace std {

template <std::size_t N>
struct hash<Board<N>> {
    size_t operator()(const Board<N>& board) const {
        return board.permutationRank();
    }
};

template <std::size_t N>
constexpr std::string to_string(const Board<N>& board) {
    static stringstream ss; 
    ss.str(""), ss << board; 
    auto str = ss.str();
    str.pop_back(); // Remove "\n"
    str.pop_back(); // Remove " "
    return str;
}

}

template <std::size_t N> Board<N> Board<N>::Random() {
    auto board = Board::Ordered();
    auto& state = board.state;
    std::shuffle(state.begin(), state.end(), Rnd);
    board.blank = std::find(state.begin(), state.end(), 0) - state.begin();
    return board;
}

template <std::size_t N> Board<N> Board<N>::Ordered() {
    Board board;
    board.state.resize(Board::Size);
    board.blank = 0;
    std::iota(board.state.begin(), board.state.end(), 0);
    return board;
}

template <std::size_t N> std::vector<Board<N>> Board<N>::Scrambled(int maxSteps, bool fixed) {
    auto steps = fixed ? maxSteps : (maxSteps + Rnd() % maxSteps) / 2;
    std::vector<Board> trace{ Board::Ordered() };
    std::vector<std::bitset<4>> dirTested{ 0b0000 };
    std::unordered_set<Board> visited{ trace.back() };
    for (int i = 0; i < steps; ++i) {
        while (true) {
            const auto& board = trace.back();
            auto& dirTest = dirTested.back();
            auto dir = Direction(Rnd() % 4);
            while (dirTest.to_ullong() & (1ull << dir)) { // Ensure it is a new direction
                dir = Direction((dir + 1) % 4);
            }
            dirTest |= (1ull << dir);
            if (board.validMove(dir)) {
                auto next = board.move(dir);
                if (!visited.count(next)) { // Prevent misprediction of solution length
                    visited.insert(next);
                    trace.push_back(next);
                    dirTested.push_back(0b0000);
                    break;
                }
            }
            if (dirTest == 0b1111) { // No possible next step could be found
                if (!fixed) { // If not fixed, we just return the current trace
                    return trace;
                } else { // Else, we must ensure that trace reaches maxSteps
                    do { // Using backtrack to find another way
                        trace.pop_back();
                        dirTested.pop_back();
                    } while (dirTested.back().count() < 4); // must contains an available dir
                }
            }
        }
    }
    return trace;
}

#endif // !NPUZZLE_HPP_
