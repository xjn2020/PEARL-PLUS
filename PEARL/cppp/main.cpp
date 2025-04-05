#include <cstdlib>
#include <string.h>
#include <string>
#include <cstdlib>
#include <string.h>
#include <string>
#include <iostream>
#include <ctime>

#include "model.h"
#include "infer.h"

using namespace std;

void usage() 
{
  cout << "Training Usage:" << endl
       << "btm est <K> <W> <alpha> <beta> <n_iter> <save_step> <docs_pt> <model_dir>\n"
       << "\tK  int, number of topics, like 20" << endl
       << "\tW  int, size of vocabulary" << endl
       << "\talpha   double, Pymmetric Dirichlet prior of P(z), like 1.0" << endl
       << "\tbeta    double, Pymmetric Dirichlet prior of P(w|z), like 0.01" << endl
       << "\tn_iter  int, number of iterations of Gibbs sampling" << endl
       << "\tsave_step   int, steps to save the results" << endl
       << "\tdocs_pt     string, path of training docs" << endl
       << "\tmodel_dir   string, output directory" << endl
       << "Inference Usage:" << endl
       << "btm inf <K> <docs_pt> <model_dir>" << endl
       << "\tK  int, number of topics, like 20" << endl
       << "\tdocs_pt     string, path of training docs" << endl
       << "\tmodel_dir  string, output directory" << endl;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        usage();
        return 1;
    }

    //// load parameters from std input
    // sum_b $K $W $alpha $beta $E $niter $dwid_pt $model_dir $cos_sim_pt $biterm_num
    string type(argv[1]);
    int K = atoi(argv[2]) + 1;                  // topic num
	int W = atoi(argv[3]);            // vocabulary size
    double alpha = atof(argv[4]);    // hyperparameters of p(z)
    double beta = atof(argv[5]);     // hyperparameters of p(w|z)
    int E = atoi(argv[6]);
    int n_iter = atoi(argv[7]);
    string docs_pt(argv[8]);
    string dir(argv[9]);
    string cos_sim_pt(argv[10]);
    int biterm_num = atoi(argv[11]); 

    clock_t start = clock();    // time start
    Model model(K, W, alpha, beta, n_iter, cos_sim_pt, biterm_num);  // cos_sim_dir, biterm_num

    cout << "#####################Begin iteration#####################" << endl;
    for (int i = 1; i < E + 1; i ++ )
    {
        cout << "#######################" << i << '/' << E << "#######################" << endl;

        // estimation
	    model.run(docs_pt, dir, i);   // i

	    // inference
	    Infer inf(type, K);
        inf.run(docs_pt, dir, i);
    }

    clock_t end = clock();  // time end
	printf("cost %fs\n", double(end - start)/CLOCKS_PER_SEC);
}
