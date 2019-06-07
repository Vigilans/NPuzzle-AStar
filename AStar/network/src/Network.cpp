#include <Network.h>
#include "MXNet.h"
#include "MLP.h"

using namespace std;
using namespace mx;

Context MXNet::GlobalContext = Context::cpu();

shared_ptr<Network> Network::Create(string type) {
    if (type == "mlp") {
        return make_shared<MLP>();
    }
}