#ifndef _DOC_H
#define _DOC_H

#include <string>
#include <vector>
#include <list>
#include <cassert>
#include <sstream>

#include "biterm.h"

using namespace std;

class Doc {
private:
  vector<int> ws;	// word sequence
  vector<double> cos_sim_line;
  
public: 
  Doc(const string& s) {read_doc(s);}
  Doc(const string& s, int flag) {read_cos_sim(s);}

  int size() const {return ws.size();}
  int len() const {return cos_sim_line.size();}
  
  const vector<int>& get_ws() const {return ws;}
  
  const int get_w(int i) const {
	assert(i < ws.size());
	return ws[i];
  }

  const double get_s(int i) const
  {
    assert(i < cos_sim_line.size());
    return cos_sim_line[i];
  }

  /**
  * Extract biterms from a document
  *   `win`:  window size for biterm extraction
  *   `bs`: the output biterms
  */
  void gen_biterms(vector<Biterm>& bs, int win = 15) const 
  {
	  if (ws.size() < 2) return;
	
	  for (int i = 0; i < ws.size()-1; ++i) 
	    for (int j = i+1; j < int(ws.size()); ++j) 
		    bs.push_back( Biterm(ws[i], ws[j]) );
  }

private:
  void read_doc(const string& s) {
    istringstream iss(s);
	  int w;
    while (iss >> w)  ws.push_back(w);
  }
  void read_cos_sim(const string& s)
  {
    istringstream iss(s);
	double w;
    while (iss >> w)  cos_sim_line.push_back(w);
  }
};
  
#endif

