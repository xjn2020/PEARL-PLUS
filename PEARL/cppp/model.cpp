#include <cassert>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <ctime>
#include <algorithm>
#include <vector>

#include "sampler.h"
#include "str_util.h"
#include "model.h"

void Model::run(string doc_pt, string res_dir, int idx)
{

  if (idx == 1)
  {
    load_docs(doc_pt);
        model_init();
    load_cos_sim_mat();
  }

  // Gibbs sampling
  for (int it = 1; it < n_iter + 1; ++it)
  {
	  cout << "\riter " << it << '/' << n_iter;
	  fflush(stdout);   // clear
	  for (int b = 0; b < bs.size(); ++b)
      {
	    update_biterm(bs[b], idx, b);
	  }
  }

//  if (idx == 1) {cos_sim.free_memory();}

  // save theta phi
  string out_dir = res_dir+"k" + str_util::itos(K) + "." + str_util::itos(idx) + "."; //kK.idx.
  save_res(out_dir);
}

void Model::model_init() {
    // 先将nbz和nwz归零#########################################
  srand(time(NULL));  // use current time as random seed
  // random initialize
  for (vector<Biterm>::iterator b = bs.begin(); b != bs.end(); ++b) {
	int k = Sampler::uni_sample(K); // 0...K-1
	assign_biterm_topic(*b, k);
  }
}

// input, each line is a doc
// format: wid  wid  wid ...
void Model::load_docs(string dfile) {
  cout << "load docs: " << dfile << endl;
  ifstream rf( dfile.c_str() );
  if (!rf) {
	cout << "file not find:" << dfile << endl;
	exit(-1);
  }

  string line;
  while(getline(rf, line)) {
	Doc doc(line);
	doc.gen_biterms(bs);

	// statistic the exmperial word distribution
	for (int i = 0; i < doc.size(); ++i) {
	  int w = doc.get_w(i);
	  pw_b[w] += 1;     // background word distribution
	}
  }
  
  pw_b.normalize();
}

void Model::load_cos_sim_mat()
{
  cout << "load cos_sim_mat:" << cos_sim_pt << endl;
  ifstream rf( cos_sim_pt.c_str() );
  if (!rf)
  {
    cout << "file not find:" << cos_sim_pt << endl;
	  exit(-1);
  }

  string line;
  int i = 0;
  while(getline(rf, line))
  {
    Doc doc(line, 0);
    for (int j = 0; j < doc.len(); ++j)
    {
      double w = doc.get_s(j);
      cos_sim[i][j] = w;
//      cout <<cos_sim[i][j]<<endl;
    }
    i ++ ;
  }
}

// sample procedure for ith biterm 
void Model::update_biterm(Biterm& bi, int idx, int b) {
  reset_biterm_topic(bi);
  
  // compute p(z|b)
  Pvec<double> pz;
  compute_pz_b(bi, pz, idx, b);

  // sample topic for biterm b
  int k = Sampler::mult_sample(pz.to_vector());
  assign_biterm_topic(bi, k);
}

// reset topic assignment of biterm i
void Model::reset_biterm_topic(Biterm& bi) {
  int k = bi.get_z();
  // not is the background topic
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  
  nb_z[k] -= 1;	// update number of biterms in topic K
  nwz[k][w1] -= 1;	// update w1's occurrence times in topic K
  nwz[k][w2] -= 1;
  assert(nb_z[k] > -10e-7 && nwz[k][w1] > -10e-7 && nwz[k][w2] > -10e-7);
  bi.reset_z();
}

// compute p(z|w_i, w_j)
void Model::compute_pz_b(Biterm& bi, Pvec<double>& pz, int idx, int b) 
{
  pz.resize(K);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  
  double pw1k, pw2k, pk;
  for (int k = 0; k < K; ++k) 
  {
    // avoid numerical problem by mutipling W
    if (has_background && k == 0) 
    {
      pw1k = pw_b[w1];
      pw2k = pw_b[w2];
    }
    else 
    {
      pw1k = (nwz[k][w1] + beta) / (2 * nb_z[k] + W * beta);
      pw2k = (nwz[k][w2] + beta) / (2 * nb_z[k] + 1 + W * beta);
    }
    pk = (nb_z[k] + alpha) / (bs.size() + K * alpha);

    if (idx == 1)
    {
      pz[k] = cos_sim[b][k] * pk * pw1k * pw2k;
    }
    else
    {
      pz[k] = pw_Z[k][w1] * pw_Z[k][w2] * pk * pw1k * pw2k;
    }
  }

  //pz.normalize();
}

// assign topic k to biterm i
void Model::assign_biterm_topic(Biterm& bi, int k) {
  bi.set_z(k);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  nb_z[k] += 1;   // how many biterms are assigned to the topic z
  nwz[k][w1] += 1;  // the number of times of the word w assigned to the topic z
  nwz[k][w2] += 1;
}

void Model::save_res(string dir) {
  // theta
  string pt = dir + "pz";
  cout << "\nwrite p(z): " << pt << endl;
  save_pz(pt);
  
  // phi
  string pt2 = dir + "pw_z";
  cout << "write p(w|z): " << pt2 << endl;
  save_pw_z(pt2);
}

// p(z) is determinated by the overall proportions
// of biterms in it
void Model::save_pz(string pt) 
{
  for (int i = 0; i < K; i ++)
  {
    pZ[i] = double(nb_z[i]);
  }
  pZ.normalize(alpha);
  pZ.write(pt);
}

void Model::save_pw_z(string pt) 
{
  ofstream wf(pt.c_str());
  for (int k = 0; k < K; k++) 
  {
	  for (int w = 0; w < W; w++) 
	    pw_Z[k][w] = (nwz[k][w] + beta) / (nb_z[k] * 2 + W * beta);

	  wf << pw_Z[k].str() << endl;
  }
}
