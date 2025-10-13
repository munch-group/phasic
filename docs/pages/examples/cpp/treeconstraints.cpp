#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include <list>
#include <cstdlib>
#include <limits>
#include <exception>

#include "treeconstraints.h"

/** Stuff for working with sets and pairs: *********************************************************/

std::set<int> set_intersection(std::set<int> s1, std::set<int> s2) {
  // Calculates intersection of two sets of ints.
  std::set<int> s_inter;
  set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), inserter(s_inter, s_inter.begin()));
  return s_inter;

}

std::set<int> set_union(std::set<int> s1, std::set<int> s2) {
  // Calculate union of two sets of ints.
  std::set<int> s_union;
  set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), inserter(s_union, s_union.begin()));
  return s_union;

}

bool set_includes(std::set<int> s1, std::set<int> s2) {
  // Test if a set of ints is included in an other set of ints.
  bool answer;
  answer = includes(s1.begin(), s1.end(), s2.begin(), s2.end());
  return answer;
}

struct pair_compare {
  bool operator() (const int *first, const int *second) const
  { 
    if (first[0] < second[0]) {
      return true;
    } else if (first[0] == second[0] && first[1] < second[1]) {
      return true;
    } else {
      return false;
    }
  }
};

bool pair_identity (int *first, int *second) {
  // testing identiry of two pairs for use in list unique.
  if (first[0] == second[0] && first[1] == second[1]) {
    delete [] second;
    return true;
  } else {
    return false;
  }
}

bool pair_sort (int *first, int *second) {
  // Sorting of pairs.
  if (first[0] < second[0]) {
    return true;
  } else if (first[0] == second[0] && first[1] < second[1]) {
    return true;
  } else {
    return false;
  }
}

void printPairsSet(std::set<int *> s) {
  // Prints a set of pair tuples to the stdout: 
  std::set<int *>::iterator it;
  std::cout << '(';
  for (it=s.begin() ; it != s.end(); it++ ) {
    if (it != s.begin())
      std::cout << ", ";
    std::cout << '[' << **it << ',' << *(*it+1) << ']';
  }
  std::cout << ')' << std::endl;
}

/** Utility functions: *****************************************************************************/

int intMax(int a, int b) {

  return a > b ? a : b;
}

int intMin(int a, int b) {

  return a < b ? a : b;
}

bool findInIntVector(int n, std::vector<int> v) {

  int vSize = v.size();
  for (int i=0; i<vSize; i++) {
    if (n == v[i]) {
      return true;
    }
  }
  return false;
}

void swapInts(int *p1, int *p2) {
  int tmp;
  tmp = *p1;
  *p1 = *p2;
  *p2 = tmp;
}

std::vector<int*> get_pairwise_combinations(std::vector<int> elements) {
  std::vector<int*> pairList;
  for (unsigned int i=0; i<elements.size(); i++) {
    for (unsigned int j=i+1; j<elements.size(); j++) {
      int* pair = new int[2];
      pair[0] = intMin(elements[i], elements[j]);
      pair[1] = intMax(elements[i], elements[j]);
      pairList.push_back(pair);
    }
  }
  return pairList;
}

std::vector<std::string> stringSplit(std::string str, std::string delim) {
  int cutAt;
  std::vector<std::string> results;
  while( (cutAt = str.find_first_of(delim)) != str.npos ) {
    if(cutAt > 0) {
      results.push_back(str.substr(0,cutAt));
    }
    str = str.substr(cutAt+1);
  }
  if(str.length() > 0) {
    results.push_back(str);
  }
  return results;
}

/** Global variables: ******************************************************************************/

// Maximal number of allowed sequences:
#define MAX_OTUS 1000

// Maximal allowed sequence length:
#define MAX_ALIGN_LENGTH 2000

// // Array of array alignment[<sequence_nr>][base_nr]
// char **alignment;

// Array of arrays each specifying a constraint.  (name integers)
std::vector<std::set<int> > backboneSetsList;

// Array of tuples specifying allowed pairs.     (index integers)
std::list<int *> allowedPairsList;

// Array of otus not part of back bone constraint.
std::vector<int> unconstrainedOTUs;

int N, K, L;

// Number of sequences (we need this for the other dimension of the alignmnet)
int nrOTUs;


// char *Node::getSubTreeString(char *buffer) {

//   if (left != NULL && right != NULL) {
//     //char leftBuffer[1000000];
//     char *leftBuffer = new char[10000];
//     //char rightBuffer[1000000];
//     char *rightBuffer = new char[10000];
//     if ((*left).leafSet.size() > (*right).leafSet.size()) {
//       if (branchLengths == 1) {
// 	sprintf(buffer, "(%s:%.5f,%s:%.5f)", (*left).getSubTreeString(leftBuffer), distLeft, (*right).getSubTreeString(rightBuffer), distRight);
//       } else {
// 	sprintf(buffer, "(%s,%s)", (*left).getSubTreeString(rightBuffer), (*right).getSubTreeString(leftBuffer));
//       }
//     } else {
//       if (branchLengths == 1) {
// 	sprintf(buffer, "(%s:%.5f,%s:%.5f)", (*right).getSubTreeString(rightBuffer), distRight, (*left).getSubTreeString(leftBuffer), distLeft);
//       } else {
// 	sprintf(buffer, "(%s,%s)", (*right).getSubTreeString(rightBuffer), (*left).getSubTreeString(leftBuffer));
//       }
//     }
//     delete [] leftBuffer;
//     delete [] rightBuffer;
//   } else {
//     sprintf(buffer,"%d",name);
//   }
//   return buffer;
// }

void freeExternMemmory(void) {
  
  extern int N;
  extern std::list<int *> allowedPairsList;
    
  std::list<int *>::iterator it;
  for ( it = allowedPairsList.begin() ; it != allowedPairsList.end() ; ++it )
    delete [] *it;
  allowedPairsList.clear();
}

/** Calculating and maintaining the list of pairs allowed to join: *********************************/

bool updateConstraints(int i, int j) {

  extern std::vector<std::set<int> > backboneSetsList;

  bool removedConstraint = false;

  int nrBackboneSets = backboneSetsList.size();
  for (int k=nrBackboneSets-1; k>=0; k--) {
    if (set_member(j+1, backboneSetsList[k])) {
      backboneSetsList[k].erase(j+1);
      if (backboneSetsList[k].size() == 1) {
	removedConstraint = true;
	backboneSetsList.erase(backboneSetsList.begin() + k);
      }
    }
  }

  return removedConstraint;
} 

void computeAllowedPairs(void) {

  extern int N;
  extern std::vector<std::set<int> > backboneSetsList;
  extern std::list<int *> allowedPairsList;

  std::list<int *>::iterator it;
  for ( it = allowedPairsList.begin() ; it != allowedPairsList.end() ; ++it )
    delete [] *it;
  allowedPairsList.clear();

  for (int i=0; i<N; i++) {
    int inConstraint = 0;
    std::set<int> constrainedSet;
    
    int nrBackboneSets = backboneSetsList.size();
    for (int j=0; j<nrBackboneSets; j++ ) {
      std::set<int> s = backboneSetsList[j];
      if (set_member(i+1, s)) {
	inConstraint = 1;
	std::vector<int> nodeIndeces;
	std::set<int>::iterator sit;
	for (sit=s.begin() ; sit != s.end(); sit++ ) {
	  if (!set_member(*sit, constrainedSet))
	    nodeIndeces.push_back(*sit - 1);
	}
	std::vector<int *> pairs = get_pairwise_combinations(nodeIndeces);
	int nrPairs = pairs.size();
	for (int k=0; k<nrPairs; k++) {
	  allowedPairsList.push_back(pairs[k]);
	}
	break;
      } else {
	constrainedSet = set_union(constrainedSet, s);	
      }
    }
    if (!inConstraint) {
      for (int j=0; j<N; j++) {
	if (i != j) {
	  //int *pair = (int *) calloc(2, sizeof(int));
	  int *pair = new int[2];
	  pair[0] = intMin(i, j);
	  pair[1] = intMax(i, j);
	  allowedPairsList.push_back(pair);
	}
      }      
    }
    allowedPairsList.sort(pair_sort);
    allowedPairsList.unique(pair_identity);
  }
}

void updateAllowedPairs(int i, int j) {

  extern std::list<int *> allowedPairsList;
  extern std::vector<std::set<int> > backboneSetsList;
  extern std::vector<int> unconstrainedOTUs;

  std::list<int *> newList;

  bool removedConstraint = updateConstraints(i, j);

  std::list<int *>::iterator it;
  for ( it=allowedPairsList.begin() ; it != allowedPairsList.end(); it++ ) {
        
    if ((*it)[0] == j || (*it)[1] == j) {

      if (findInIntVector(j, unconstrainedOTUs))
	continue;
      
      if ((*it)[0] == i && (*it)[1] == j || 
	  (*it)[0] == j && (*it)[1] == i) { // we may have swopped them.
	continue;
      } else if ((*it)[0] == j) {
	//int *p = (int *) calloc(2, sizeof(int));
	int *p = new int[2];
	p[0] = intMin(i, (*it)[1]);
	p[1] = intMax(i, (*it)[1]);
	newList.push_back(p);
      } else if ((*it)[1] == j) {
	//int *p = (int *) calloc(2, sizeof(int));
	int *p = new int[2];
	p[0] = intMin(i, (*it)[0]);
	p[1] = intMax(i, (*it)[0]);
	newList.push_back(p);
      } else {
	std::cout << "WARNING: Not supposed to happen" << std::endl;
	exit(1);
      }
    } else {
      int *p = new int[2];
      p[0] = (*it)[0];
      p[1] = (*it)[1];
      newList.push_back(p);
      //newList.push_back((*it));
    }
  }

  if (removedConstraint) {
    // if we removed a set it means that i now represents all otus
    // from that set. So we find the first set in the updated
    // constraints list that includes i and then add all
    // combinations of i to items in this set:
    std::set<int> constrainedSet;

    int nrBackboneSets = backboneSetsList.size();
    for (int k=0; k<nrBackboneSets; k++ ) {
	std::set<int> s = backboneSetsList[k];
	if (set_member(i+1, s)) {

	  std::vector<int> nodeIndeces;
	  std::set<int>::iterator sit;
	  for (sit=s.begin() ; sit != s.end(); sit++ ) {
	    if (!set_member(*sit, constrainedSet))
	      nodeIndeces.push_back(*sit - 1);
	  }

	  std::vector<int *> pairs = get_pairwise_combinations(nodeIndeces);
	  std::vector<int *>::iterator lit;
	  for ( lit=pairs.begin() ; lit != pairs.end(); lit++ ) {
	    newList.push_back(*lit);
	  }
	  break;
	} else {
	  constrainedSet = set_union(constrainedSet, s);
	}
    }  
  }
  newList.sort(pair_sort);

  newList.unique(pair_identity);  
  for ( std::list<int *>::iterator it=allowedPairsList.begin() ; it != allowedPairsList.end(); it++ )
    delete *it;
  allowedPairsList.clear();
  allowedPairsList = newList;
}

/** Finding a pair to join and create a new node: **************************************************/

int *findPair(void) {

  extern std::list<int *> allowedPairsList;

  std::vector<int *> pairList;

  std::list<int *>::iterator it;
  for ( it=allowedPairsList.begin() ; it != allowedPairsList.end(); it++ ) {
    int i = **it;
    int j = *(*it+1);
      int *p = new int[2];
      p[0] = intMin(i, j); 
      p[1] = intMax(i, j); 
      pairList.push_back(p);
  }

  // Free the memory for the pairs we don't return:
  int pidx = rand() % pairList.size();
  int pairListSize = pairList.size();
  for ( int i=0; i<pairListSize; i++ ) {
    if (i != pidx)
      delete [] pairList[i];
  }
  return pairList[pidx];
}

/***************************************************************************************************/


void pick_pairs(int a_nrOTUs, int a_nrBackboneSets, char **a_backboneSetsList)
{
  	std::cout << "hello world" << std::endl;


  extern int nrOTUs, N, L, K; //, alignmentLength;
  extern std::vector<std::set<int> > backboneSetsList;
  extern std::list<int *> allowedPairsList;
  extern std::vector<int> unconstrainedOTUs;

  int i, j;

  // Populate teh global variables:
  nrOTUs = a_nrOTUs;

  // The initial number of leaves
  N = nrOTUs;
  // The number of clusters created so far (including the first N "leaf culsters" we start with)
  K = N;
  // The number of nodes left to be joined:
  L = N;

  // Turn lists specifying unconstrained sets in to set objects:
  std::string s;
  std::string delim = " ";
  std::vector<std::string> splitList;
  int otu;
  
  backboneSetsList.clear();

  for (i=0; i<a_nrBackboneSets; i++) {
    s.assign(a_backboneSetsList[i]);
    splitList = stringSplit(s, delim);
    std::set<int> constSet;
    for (unsigned int j=0; j<splitList.size(); j++) {
      otu = atoi(splitList[j].c_str());
      constSet.insert(otu);
    }
    backboneSetsList.push_back(constSet);
  }

  // Find the otus that are not part of the backbone constraint:
  std::set<int> unionSet;
  for (unsigned int i=0; i<backboneSetsList.size(); i++) {
    unionSet = set_union(unionSet, backboneSetsList[i]);
  }

  unconstrainedOTUs.clear();
  for (i=0; i<N; i++) {
    if (!set_member(i+1, unionSet))
      unconstrainedOTUs.push_back(i);
  }

  // Compute the pairs that are allowed to join at first:
  computeAllowedPairs();

  // Main loop:
  while (K < 2 * N - 2) {
    int *p = findPair();

    if (findInIntVector(p[0], unconstrainedOTUs)) {
      // If i is not in the backbone sets we don't want i to represent the child
      // nodes. If they are both unconstrained swopping is not needed but does not matter.      
      swapInts(&p[0], &p[1]);
    }

    for (auto i: backboneSetsList)
      std::cout << i << std::endl;

  // std::list<int *>::iterator it;
  // for ( it=allowedPairsList.begin() ; it != allowedPairsList.end(); it++ ) {
  //   int i = **it;
  //   int j = *(*it+1);
  //   std::cout << i+1 << ' ' << j+1 << std::endl;
  // }

    std::cout << "picked " << p[0]+1 << " " << p[1]+1 << std::endl;

    updateAllowedPairs(p[0], p[1]);

    K += 1;
  }


  // only one join remains and this should be reflected in allowedPairsList..
  if (allowedPairsList.size() != 1) {
    std::cout << "WARNING: allowedPairsList is not one - it is " << allowedPairsList.size() << std::endl;
    exit(1);
  }
  int *p = allowedPairsList.front();

  // Free up memory:
  freeExternMemmory();

}

std::list<int *> init_constraints(int a_nrOTUs, int a_nrBackboneSets, char **a_backboneSetsList)
{
  extern int nrOTUs, N, L, K; //, alignmentLength;
  extern std::vector<std::set<int> > backboneSetsList;
  extern std::list<int *> allowedPairsList;
  extern std::vector<int> unconstrainedOTUs;

  int i, j;

  // Populate teh global variables:
  nrOTUs = a_nrOTUs;

  // The initial number of leaves
  N = nrOTUs;
  // The number of clusters created so far (including the first N "leaf culsters" we start with)
  K = N;
  // The number of nodes left to be joined:
  L = N;

  // Turn lists specifying unconstrained sets in to set objects:
  std::string s;
  std::string delim = " ";
  std::vector<std::string> splitList;
  int otu;
  
  backboneSetsList.clear();

  for (i=0; i<a_nrBackboneSets; i++) {
    s.assign(a_backboneSetsList[i]);
    splitList = stringSplit(s, delim);
    std::set<int> constSet;
    for (unsigned int j=0; j<splitList.size(); j++) {
      otu = atoi(splitList[j].c_str());
      constSet.insert(otu);
    }
    backboneSetsList.push_back(constSet);
  }

  // Find the otus that are not part of the backbone constraint:
  std::set<int> unionSet;
  for (unsigned int i=0; i<backboneSetsList.size(); i++) {
    unionSet = set_union(unionSet, backboneSetsList[i]);
  }

  unconstrainedOTUs.clear();
  for (i=0; i<N; i++) {
    if (!set_member(i+1, unionSet))
      unconstrainedOTUs.push_back(i);
  }

  // Compute the pairs that are allowed to join at first:
  computeAllowedPairs();

  return allowedPairsList;
}

std::list<int *> update_constraints(int *p)
{

  // give as arg a list of all otus that have coalesced
  // we can assume all coelescences so far have obeyed the constraints
  // we should look in each set 


    if (findInIntVector(p[0], unconstrainedOTUs)) {
      // If i is not in the backbone sets we don't want i to represent the child
      // nodes. If they are both unconstrained swopping is not needed but does not matter.      
      swapInts(&p[0], &p[1]);
    } 

    updateAllowedPairs(p[0], p[1]);

    // if (allowedPairsList.size() == 1) {
    // // Free up memory:
    // freeExternMemmory();
    // }

  return allowedPairsList;
}


int main(void) {

// char *a_alignment[11] = {
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGAGCCGGCATAATTGGTACCGCCTTAAGCCTCCTTATCCGAGCAGAACTAGGTCAACCAGGAACCCTCCTAGGAGACGACCAAATCTATAATGTTATTGTTACTGCCCATGCTTTCGTAATGATCTTCTTTATAGTAATACCCATTATAATTGGCGGATTTGGTAACTGATTAGTTCCCCTAATAATTGGTGCCCCTGACATAGCATTCCCACGTATAAACAATATAAGCTTCTGATTACTCCCTCCATCATTCCTCCTACTCCTAGCCTCATCTACAATTGAAGCCGGAGTGGGCACTGGATGAACTGTCTATCCTCCACTAGCCGGTAACCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTTCATCTTGCAGGTATTTCTTCAATCCTAGGTGCTATTAACTTCATCACTACTGCAATTAATATAAAACCACCAACCCTATCACAATATCAAACTCCCCTATTTGTATGATCCGTCCTAATTACCGCAGTTCTTCTTCTCCTCTCCCTTCCAGTCCTTGCTGCTGGCATCACCATGCTATTAACAGACCGCAATCTTAACACTACGTTCTTCGACCCAGCAGGAGGTGGAGATCCAGTTTTATACCAACATCTCTTCTGATTCTTTGGCCA------------",
//   (char*)"--ACTATATTTAATTTTTGGCGCCTGAGCCGGTATGATTGGTACAGCCCTAAGCCTCCTTATCCGAGCAGAACTAGGACAACCAGGGACTCTCCTAGGAGATGACCAAATCTATAATGTAATTGTCACTGCCCATGCCTTCGTAATAATCTTCTTTATAGTAATGCCCATTATAATTGGTGGATTTGGCAACTGATTGGTCCCCCTAATAATTGGTGCTCCTGACATAGCATTCCCACGTATAAATAACATAAGCTTCTGACTACTCCCCCCATCATTCCTCCTGCTCCTAGCCTCATCTACAATTGAAGCTGGAGTAGGCACCGGATGAACTGTTTACCCACCATTAGCCGGTAATCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTCCACCTTGCAGGTGTTTCTTCAATCCTAGGTGCTATTAACTTCATCACCACTGCAATTAATATAAAACCACCAGCCCTATCACAATACCAAACCCCCCTATTTGTATGATCCGTCTTAATTACCGCAGTTCTCCTCCTCCTCTCTCTCCCAGTTCTTGCTGCTGGCATCACCATACTATTAACAGACCGCAATCTTAACACTACGNTCTTCGATCCAGCAGGAGGCGGAGATCCAGTCTTATATCAGCATCTCTNCTGATTCTTTGGGCACCCAGAAGNCTA",
//   (char*)"--ACTATATNTAATTTTTGGCGCCTGAGCCGGCATAATTGGTACCGCCCTAAGCCTCNTTATCCGAGCAGAACTAGGACAACCAGGAACCCTCCTAGGAGACGACCAAATCTATAATGTAATTGTCACTGCCCATGCTTTCGTAATGATCTTCTTTATAGTAATACCCATTATAATTGGTGGATTTGGTAACTGATTAGTTCCCCTAATAATTGGTGCCCCCGACATAGCATTCCCACGTATAAACAATATAAGCTTCTGACTACTCCCCCCATCGTTCCTCCTACTCCTAGCCTCATCTACAATTGAAGCCGGAGTGGGCACTGGATGAACTGTCTACCCTCCACTAGCCGGTAACCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTTCACCTTGCAGGTATTTCTTCAATCCTAGGTGCTATTAACTTCATCACTACTGCAATTAATATAAAACCACCAACCCTATCACAATACCAAACCCCCCTATTTGTATGATCTGTCCTAATTACCGCAGTTCTTCTTCTCCTTTCCCTTCCAGTCCTTGCTGCTGGCATCACCATGCTATTAACTGACCGCAATCTCAACACTACATTCTTCGACCCAGCAGGAGGCGGAGATCCAGTTTTATACCAACATCTCTTCTGATTTTTTGGCCACCCAGAAGTC--",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGAGCCGGCATAATCGGTACAGCCCTAAGCCTCCTCATCCGAGCAGAACTAGGACAGCCAGGAACCCTCCTAGGAGATGACCAGATCTACAATGTAATCGTCACTGCTCATGCCTTCGTAATAATCTTCTTTATAGTAATACCTATTATAATTGGTGGATTTGGCAACTGACTGGTCCCCCTAATAATTGGCGCCCCCGACATAGCATTCCCACGTATAAACAATATAAGCTTCTGACTACTCCCCCCATCATTCCTCCTGCTCCTAGCTTCGTCTACAATTGAAGCTGGAGTAGGCACCGGATGAACTGTTTACCCACCATTAGCCGGTAACCTAGCCCATGCCGGAGCTTCAGTAGACCTGGCCATCTTCTCCCTTCACCTTGCAGGTGTCTCCTCAATCCTAGGTGCTATTAACTTTATCACTACCGCAATCAATATAAAACCACCAGCCCTATCACAATATCAAACACCTCTATTCGTATGATCCGTCTTAATCACCGCAGTTCTTCTCCTCCTCTCCCTCCCAGTTCTTGCTGCTGGCATCACCATACTATTAACAGACCGCAATCTTAACACTACATTCTTTGACCCAGCAGGAGGCGGAGATCCAGTTTTATACCAACACCTTTTCTGATTTTTCGGTCACCCAG-------",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGGGCCGGCATAATCGGTACAGCCCTAAGCCTTCTCATCCGAGCAGAACTAGGACAACCAGGAACCCTCCTAGGAGATGACCAGATCTACAATGTAATCGTCACTGCTCATGCCTTCGTGATAATCTTCTTTATAGTAATACCCATTATAATTGGTGGATTTGGCAACTGACTAGTCCCCCTAATAATTGGCGCCCCCGACATAGCATTCCCACGTATAAACAATATAAGCTTCTGACTACTCCCCCCATCATTCCTCCTGCTCCTAGCCTCGTCTACAATTGAAGCTGGAGTAGGCACCGGATGAACTGTTTACCCGCCATTAGCCGGTAACCTAGCCCATGCCGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTCCACCTTGCAGGTGTCTCCTCAATCCTAGGTGCTATTAACTTTATCACTACCGCAATCAATATAAAACCACCAGCCCTATCACAATATCAAACACCTCTATTCGTATGATCCGTCTTAATCACCGCAGTTCTTCTCCTCCTCTCCCTCCCAGTTCTTGCTGCTGGCATCACCATACTATTAACAGACCGCAATCTTAACACTACATTCTTTGACCCAGCAGGAGGCGGAGATCCAGTTTTATACCAACACCTCTTCTGATTTTTCGGTCACCCAGAAGTC--",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGAGCCGGCATGATTGGCACAGCCCTAAGCCTCCTTATCCGAGCAGAACTAGGACAACCAGGGACTCTCCTAGGAGACGACCAAATCTATAATGTAATCGTTACTGCCCATGCTTTTGTAATGATCTTTTTTATAGTAATGCCCATTATAATTGGTGGATTTGGCAACTGATTGGTCCCCCTAATAATCGGTGCCCCTGACATAGCATTCCCACGTATAAACAATATAAGCTTCTGACTACTCCCCCCATCATTCCTCCTGCTCCTAGCCTCATCTACAATTGAAGCTGGAGTAGGCACCGGATGAACTGTCTACCCACCATTAGCCGGTAATCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCTATCTTCTCCCTCCACCTTGCAGGTATCTCTTCAATCCTAGGTGCTATTAACTTCATCACTACTGCAATTAACATAAAACCACCAGCCCTATCACAATACCAAACCCCTCTATTTGTATGATCTGTCTTAATTACCGCAGTTCTCCTCCTCCTCTCCCTTCCAGTTCTTGCTGCTGGCATCACCATACTGTTAACAGACCGTAATCTTAACACTACGTTCTTCGACCCAGCAGGAGGCGGAGATCCAATTTTATATCAACATCTATTCTGATTTTTTGGTCACCCAGAAGTC--",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGAGCCGGCATGATTGGTACAGCCCTAAGCCTCCTTATCCGAGCAGAACTAGGGCAACCAGGGACTCTCCTAGGAGACGACCAAATCTATAATGTAATTGTCACTGCCCATGCTTTCGTAATAATCTTCTTTATAGTAATGCCCATTATAATTGGTGGATTTGGCAACTGACTGGTCCCCCTAATAATTGGTGCCCCTGACATAGCATTTCCACGTATAAATAATATAAGCTTCTGACTACTCCCCCCATCATTCCTCCTGCTCCTAGCCTCATCTACAATTGAAGCCGGGGTAGGTACCGGATGAACTGTTTACCCACCATTAGCTGGTAATCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTCCACCTTGCAGGTGTTTCTTCAATCCTAGGTGCTATTAACTTCATCACCACTGCAATTAACATAAAACCACCAGCCCTATCACAATACCAAACCCCCTTATTTGTATGATCCGTCTTAATTACCGCAGTTCTACTTCTCCTCTCCCTTCCAGTTCTTGCTGCTGGCATCACCATACTATTAACAGACCGCAATCTTAACACTACGTTCTTCGACCCAGCAGGAGGTGGAGATCCAGTCTTATATCAACATCTCTTCTGATTTTTTGGTCACCCAGAAGT---",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGAGCCGGCATAATTGGTACCGCCCTAAGCCTCCTTATCCGAGCAGAACTAGGACAACCAGGAACCCTCCTAGGAGACGACCAAATCTATAATGTAATTGTCACTGCCCATGCTTTCGTAATGATCTTCTTTATAGTAATACCCATTATAATTGGTGGATTTGGTAACTGATTAGTTCCCCTAATAATTGGTGCCCCCGACATAGCATTCCCACGTATAAACAATATAAGCTTCTGACTACTCCCCCCATCGTTCCTCCTACTCCTAGCCTCATCTACAATTGAAGCCGGAGTGGGCACTGGATGAACTGTCTACCCTCCACTAGCCGGTAACCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTTCACCTTGCAGGTATTTCTTCAATCCTAGGTGCTATTAACTTCATCACTACTGCAATTAATATAAAACCACCAACCCTATCACAATACCAAACCCCCCTATTTGTATGATCTGTCCTAATTACCGCAGTTCTTCTTCTCCTTTCCCTTCCAGTCCTTGCTGCTGGCATCACCATGCTATTAACTGACCGCAATCTCAACACTACATTCTTCGACCCAGCAGGAGGCGGAGATCCAGTTTTATACCAACATCTCTTCTGATTTTTTGGCCACCCAGAAGTC--",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGGGCCGGCATAATCGGTACAGCCCTAAGCCTCCTCATCCGAGCAGAACTAGGACAACCAGGAACCCTCCTAGGAGATGACCAGATCTACAATGTAATCGTCACTGCTCATGCCTTCGTAATAATCTTCTTCATAGTAATGCCCATTATAATTGGTGGATTTGGCAACTGACTAGTCCCTCTAATAATTGGTGCCCCCGACATAGCATTTCCACGTATAAACAATATAAGCTTTTGACTACTCCCCCCATCATTCCTCCTACTCCTAGCCTCATCTACAGTTGAAGCTGGAGTGGGTACCGGATGGACTGTTTACCCACCATTAGCCGGTAATCTAGCCCATGCCGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTTCACCTTGCAGGTGTCTCCTCTATTCTGGGTGCCATTAACTTCATCACTACCGCAATCAATATAAAACCACCAGCCCTATCACAATATCAAACCCCTCTATTCGTATGATCCGTCCTAATCACCGCAGTTCTTCTTCTCCTCTCCCTCCCAGTTCTTGCTGCTGGCATCACTATACTATTAACAGACCGCAATCTTAACACCACATTCTTTGACCCAGCAGGAGGTGGAGATCCAGTTTTATATCAACACCTCTTCTGATTTTTTGGTCACCCAGAAGTC--",
//   (char*)"--ACTATATTTAATTTTTGGCGCCTGAGCCGGCATGATTGGTACAGCCCTAAGCCTCCTTATCCGAGCAGAACTAGGGCAACCAGGGACTCTCCTAGGAGACGACCAAATCTATAATGTTATTGTCACCGCCCATGCCTTCGTAATAATCTTCTTCATAGTAATGCCTATTATAATTGGTGGATTTGGCAACTGATTGGTCCCCCTAATAATTGGTGCCCCTGACATAGCATTTCCACGTATAAATAATATGAGCTTCTGACTACTCCCCCCATCATTCCTCCTGCTCCTAGCCTCATCTACAATTGAAGCTGGAGTAGGCACCGGATGGACTGTTTACCCACCATTAGCCGGTAATCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTCCACCTTGCAGGTGTTTCTTCAATCCTAGGTGCTATTAACTTCATCACCACTGCAATTAATATAAAACCACCAGCCCTATCACAATACCAAACCCCCCTATTTGTATGATCCGTCTTAATTACCGCAGTTCTCCTTCTCCTCTCCCTCCCAGTTCTTGCTGCTGGCATCACCATACTATTAACAGACCGCAATCTTAACACTACGTTCTTCGACCCAGCAGGAGGCGGAGACCCAGTCTTATATCAACATCTC-----------------------------",
//   (char*)"ACACTATATTTAATTTTTGGCGCCTGAGCCGGCATGATTGGTACAGCCCTAAGCCTCCTTATCCGAGCAGAACTAGGGCAACCAGGGACCCTCCTAGGAGACGACCAAATCTATAATGTAATTGTCACTGCCCATGCCTTCGTAATAATCTTCTTTATAGTAATACCCATTATAATTGGTGGATTTGGCAACTGATTGGTCCCCCTAATAATTGGTGCCCCTGACATAGCATTTCCACGTATAAATAATATAAGCTTCTGACTACTCCCTCCATCATTCCTCCTGCTCCTAGCCTCATCTACAATTGAAGCTGGAGTAGGCACCGGATGAACTGTTTACCCACCATTAGCCGGTAATCTAGCCCATGCTGGAGCTTCAGTAGACCTAGCCATCTTCTCCCTCCACCTTGCAGGTGTTTCTTCAATCCTAGGTGCTATTAACTTCATCACCACTGCAATTAATATAAAACCACCAGCCCTATCACAATACCAAACCCCCCTATTTGTATGATCCGTCTTAATTACCGCAGTTCTCCTTCTCCTCTCCCTCCCAGTTCTTGCTGCTGGCATCACCATACTATTAACAGACCGCAATCTTAACACTACATTCTTCGACCCAGCAGGAGGCGGAGATCCAGTCTTATATCAACATCTCTTCTGATTCTTTGGCCACACAGAAGTC--"
// };
// // int a_nrBackboneSets = 1;
// // char *a_backboneSetsList[1] = {
// //   (char*)"3 8 1 7 2 6 10 4 5 9"
// // };
// int a_nrBackboneSets = 6;
// char *a_backboneSetsList[6] = {
//   (char*)"4 5",
//   (char*)"3 8",
//   (char*)"7 2",
//   (char*)"6 10",
//   (char*)"3 8 1 7 2 6 10",
//   (char*)"3 8 1 7 2 6 10 4 5 9"
// };
// int a_nrOTUs = 11;

  // char *a_alignment[5] = {
  //   (char*)"TGAAAAAAAAAAAAAAAAAAA", 
  //   (char*)"CGGAAAAAAAAAAAAAAAAAA", 
  //   (char*)"CCCAAAAAAAAAAAAAAAAAA", 
  //   (char*)"CCCAAAAAAAAAAAAAAAAAA", 
  //   (char*)"CCTAAAAAAAAAAAAAAAAAA"
  // };
  int a_nrBackboneSets = 3;
  char *a_backboneSetsList[3] = {
    (char*)"1 4", 
    (char*)"2 3",
    (char*)"1 2 3 4"
  };
  int a_nrOTUs = 7;

 pick_pairs(a_nrOTUs, a_nrBackboneSets, a_backboneSetsList);

  // std::list<int *> pairs = init_constraints(a_nrOTUs, a_nrBackboneSets, a_backboneSetsList);

  // std::list<int *>::iterator it;
  // for ( it=pairs.begin() ; it != pairs.end(); it++ ) {
  //   int i = **it;
  //   int j = *(*it+1);
  //   std::cout << i << ' ' << j << std::endl;
  // }


  // p = pick_pair()

  std::cout << "NOTE THAT SET NUMBERS ARE 1 BASED AND PAIR NUBERS ARE 0 BASED. AND EACH COALESCENT REPLACES TEH PAIR WITH THE NODE WITH THE LOWEST NUMBER " << std::endl;



  return 1;
}



// use structs instead of integer labels

struct lineage_group
{
  int id;
  int count;
};

inline bool operator<(const lineage_group& lhs, const lineage_group& rhs)
{
  return lhs.id < rhs.id;
}

// don't remove a constraint set when one element remains, do it when the last element has lost its last member