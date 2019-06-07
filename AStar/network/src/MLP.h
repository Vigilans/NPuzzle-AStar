#ifndef MLP_H_
#define MLP_H_
#include "MXNet.h"

class MLP : public MXNet {
public:
    MLP() : MXNet(
        "model/mlp-symbol.json", 
        "model/mlp-symbol.params",
        { 1, 625 }
    ), m_onehot({}) { }

    virtual mx::NDArray transform(const Board& board, const Board&) override {
        for (int i = 0; i < Board::Size; ++i) {
            m_onehot[i][board.state[i]] = 1.0;
        }
        auto state = mx::NDArray(m_inputShape, GlobalContext, false);
        state.SyncCopyFromCPU(&m_onehot[0][0], m_inputShape.Size());
        for (int i = 0; i < Board::Size; ++i) {
            m_onehot[i][board.state[i]] = 0.0;
        }
        return state;
    }

    std::array<std::array<float, Board::Size>, Board::Size> m_onehot;
};

#endif // !MLP_H_

