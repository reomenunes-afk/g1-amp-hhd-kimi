#include <mujoco/mujoco.h>
#include <iostream>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: urdf2mjcf <input.urdf> <output.xml>" << std::endl;
        return 1;
    }
    const char* urdf_path = argv[1];
    const char* out_path = argv[2];

    char load_err[1000] = {0};
    mjModel* m = mj_loadXML(urdf_path, nullptr, load_err, sizeof(load_err));
    if (!m) {
        std::cerr << "Failed to load URDF: " << load_err << std::endl;
        return 1;
    }

    char save_err[1000] = {0};
    if (mj_saveLastXML(out_path, m, save_err, sizeof(save_err)) == 0) {
        std::cerr << "Failed to save MJCF: " << save_err << std::endl;
        mj_deleteModel(m);
        return 1;
    }

    std::cout << "Saved MJCF to " << out_path << std::endl;
    mj_deleteModel(m);
    return 0;
}
