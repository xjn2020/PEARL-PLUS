#ifndef _MODEL_H
#define _MODEL_H

#include <vector>
#include <fstream>
#include <string>
#include "biterm.h"
#include "doc.h"
#include "pvec.h"
#include "pmat.h"

using namespace std;

class Model
{
  public:
    vector<Biterm> bs;

  protected:
    int W;				// vocabulary size
    int K;				// number of topics
    int n_iter;			// maximum number of iteration of Gibbs Sampling
    double alpha;			// hyperparameters of p(z)
    double beta;			// hyperparameters of p(w|z)
    string cos_sim_pt;
    
    // sample recorders (counters)
    Pvec<int> nb_z;	// n(b|z), size K*1  denotes how many biterms are assigned to the topic z
    Pmat<int> nwz;	  // n(w,z), size K*W  times of the word w assigned to the topic z

    Pvec<double> pw_b;   // the background word distribution
    Pmat<double> cos_sim;
    Pvec<double> pZ;
    Pmat<double> pw_Z; 

    // If true, the topic 0 is set to a background topic that 
    // equals to the emperiacal word dsitribution. It can filter 
    // out common words
    bool has_background;      // AAAI23 biterm set generation

  public:
    // construct function
    Model(int K, int W, double a, double b, int n_iter, string cos_sim_pt, int biterm_num, bool has_b = false):
    K(K), W(W), alpha(a), beta(b), 
    n_iter(n_iter), cos_sim_pt(cos_sim_pt), has_background(has_b)
    {
      pw_b.resize(W);     // background word distribution
      nwz.resize(K, W);   // the number of times of the word w assigned to the topic z
      nb_z.resize(K);   // denotes how many biterms are assigned to the topic z
      cos_sim.resize(biterm_num, K);
      pZ.resize(K);
      pw_Z.resize(K, W);
    }
    
    // run estimate procedures
    void run(string docs_pt, string res_dir, int idx);
    
  private:
    // intialize memeber varibles and biterms
    void model_init();
    
    // load from docs
    void load_docs(string docs_pt); 

    void load_cos_sim_mat();
    
    // update estimate of a biterm
    void update_biterm(Biterm& bi, int idx, int b);
    
    // reset topic proportions for biterm b
    void reset_biterm_topic(Biterm& bi);
    
    // assign topic proportions for biterm b
    void assign_biterm_topic(Biterm& bi, int k);
    
    // compute condition distribution p(z|b)
    void compute_pz_b(Biterm& bi, Pvec<double>& p, int idx, int b);

    void save_res(string res_dir);
    void save_pz(string pt);
    void save_pw_z(string pt);
};

#endif
