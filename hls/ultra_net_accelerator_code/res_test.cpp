#include <iostream>
#include <fstream>

using namespace std;

void load_data(const char *path, char *ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}


int main(int argc, char const *argv[])
{
    int res[10][20][36];
    load_data("data/boat6_0_res.bin", (char *)res, sizeof(res));

    for (int i=0; i < 36; i ++) {
        cout << res[0][0][i] << " " ;
    }
    int res_cof[10][20] = {0};

    int i_max = 0;
    int j_max = 0;
    int max = -999999;
    for (int i=0; i < 10; i ++) {
        for (int j=0; j < 20; j ++) {
            int temp = res[i][j][4];
            if (temp > max) {
                max = temp;
                
                cout << "i " << i << " j " << j << endl;
            }
            
        }
        cout << endl;
    }
    return 0;
}
