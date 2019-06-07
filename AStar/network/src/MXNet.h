#ifndef ASTAR_MXNET_H_
#define ASTAR_MXNET_H_
#include <Network.h>
#include <mxnet-cpp/MxNetCpp.h>

namespace mx = mxnet::cpp;

class MXNet : public Network {
public:
    using ArrayMap = std::map<std::string, mx::NDArray>;

    static mx::Context GlobalContext;

    MXNet(std::string modelFile, std::string paramsFile, mx::Shape inputShape) {
        m_inputShape = std::move(inputShape);
        this->loadModel(std::move(modelFile), std::move(paramsFile));
    }

    virtual ~MXNet() { MXNotifyShutdown(); }

    virtual mx::NDArray transform(const Board& a, const Board& b) = 0;

    virtual Value predict(const Board& board, const Board& goal) override {
        auto input = transform(board, goal);

        input.CopyTo(&m_executor->arg_arrays[0]);

        // Run the neural network
        m_executor->Forward(false);

        auto output = m_executor->outputs[0].Copy(mx::Context::cpu());
        output.WaitToRead();
        
        return std::round(*output.GetData());
    }

private:
    void loadModel(std::string modelFile, std::string paramsFile) {
        ArrayMap parameters;
        m_net = mx::Symbol::Load(modelFile);
        mx::NDArray::Load(paramsFile, nullptr, &parameters);
        m_argMap["data"] = mx::NDArray(m_inputShape, GlobalContext, false);
        for (const auto& [key, arr] : parameters) {
            //std::cout << key << std::endl;
            //std::cout << arr << std::endl;
            if (key.substr(0, 4) == "arg:") {
                auto name = key.substr(4, key.size() - 4);
                m_argMap[name] = arr.Copy(GlobalContext);
            }
            if (key.substr(0, 4) == "aux:") {
                auto name = key.substr(4, key.size() - 4);
                m_auxMap[name] = arr.Copy(GlobalContext);
            }
        }
        mx::NDArray::WaitAll();
        m_executor = std::unique_ptr<mx::Executor>(m_net.SimpleBind(
            GlobalContext, m_argMap, ArrayMap(),
            std::map<std::string, mx::OpReqType>(), m_auxMap
        ));
        mx::NDArray::WaitAll();
    }

protected:
    mx::Symbol m_net;
    std::unique_ptr<mx::Executor> m_executor;
    mx::Shape m_inputShape;
    ArrayMap m_argMap;
    ArrayMap m_auxMap;
};

#endif // !ASTAR_MXNET_H_
