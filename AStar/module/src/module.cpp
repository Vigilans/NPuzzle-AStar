#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <sstream>
#include <variant>
#include "AStar.hpp"
#include "NPuzzle.hpp"

namespace py = pybind11;
using namespace std;
using namespace py::literals; // Import the `_a` literal

constexpr size_t N = 5; // 24-Digits
using AStar = AStarSearch<NPuzzle<N>>;
using HeuristicFunc = std::function<AStar::Value(const AStar::State&, const AStar::State&)>;

PYBIND11_MODULE(Module, mod) {
    mod.doc() = "NPuzzle.rl Core module";

    // Definition of Player enum class
    py::enum_<Board<N>::Direction>(mod, "Direction", "Blank tile's moving direction")
        .value("left",  Board<N>::Direction::Left)
        .value("right", Board<N>::Direction::Right)
        .value("up",    Board<N>::Direction::Up)
        .value("down",  Board<N>::Direction::Down);

    py::class_<Board<N>>(mod, "Board", "24-Digits Board")
        .def(py::init<>())
        .def_readonly("blank_tile", &Board<N>::blank)
        .def_property_readonly("board", [](const Board<N>& b) {
            return py::array_t({ N, N }, b.state.data());
        })
        .def_property_readonly("state", [](const Board<N>& b) { 
            py::array_t<float> state({ N * N });
            std::copy(b.state.begin(), b.state.end(), state.mutable_data());
            return state;
        })
        .def("move", [](const Board<N>& b, Board<N>::Direction dir) { return b.validMove(dir) ? b.move(dir) : b; })
        .def("__eq__",   [](const Board<N>& a, const Board<N>& b) { return a == b; })
        .def("__len__",  [](const Board<N>& b) { return N * N; })
        .def("__hash__", [](const Board<N>& b) { return b.permutationRank(); })
        .def("__str__",  [](const Board<N>& b) { return to_string(b); })
        .def("__repr__", [](const Board<N>& b) { auto [x, y] = Board<N>::GetPose(b.blank); return py::str("Board(blank: [{},{}], board:\n{}\n)").format(x, y, to_string(b)); })
        .def_static("ordered",   Board<N>::Ordered)
        .def_static("random",    Board<N>::Random)
        .def_static("scrambled", Board<N>::Scrambled)
        .def_property_readonly_static("N", [](py::object) { return N; })
        .def_property_readonly_static("size", [](py::object) { return N * N; })
        .def_static("get_pose",  Board<N>::GetPose)
        .def_static("get_index", Board<N>::GetIndex);

    py::class_<AStar>(mod, "AStar", "A* Search Tree on 24-Digits problem")
        .def(py::init([](variant<string, HeuristicFunc> arg) {
            if (arg.index() == 0) {
                auto choice = get<string>(arg);
                if (choice == "manhattan") {
                    return AStar{ NPuzzle<N>{ &ManhattanDistance<N> } };
                } else if (choice == "hamming") {
                    return AStar{ NPuzzle<N>{ &HammingDistance<N> } };
                } else {
                    throw invalid_argument("bad choice");
                }
            } else {
                auto func = get<HeuristicFunc>(arg);
                return AStar{ NPuzzle<N>{ func } };
            }
        }))
        .def("find_path", &AStar::findPath)
        .def("reset", &AStar::clear)
        .def("run", [](AStar& astar, Board<N> start) {
            auto path = astar.findPath(std::move(start), Board<N>::Ordered());
            auto result = py::make_tuple(path, path.size(), astar.statesExplorered(), astar.timeElapsed());
            return astar.clear(), result;
        }, "Result: <path, path_length, states_explorered, time_elapsed>")
        .def_property_readonly("states_explorered", &AStar::statesExplorered)
        .def_property_readonly("time_elapsed", &AStar::timeElapsed)
        .def("__repr__", [](const AStar& b) { return py::str("A* Search Tree on 24-Digits problem"); });
}