#include <Rcpp.h>
#include "stdint.h"
#include "stdlib.h"

using namespace Rcpp;


/**
 * Struct representing lineage properties corresponding to a state vector index:
 * descendants: number of decendendants for lineages represented by the state vector index.
 * population: the current population of lineages represented by the state vector index.
 */struct properties_single_locus {
    int descendants;
    int population;
};

/**
 * Struct representing lineage properties corresponding to a state vector index:
 * descendants_l1: number of decendendants at locus 1 for lineages represented by the state vector index.
 * descendants_l2: number of decendendants at locus 2 for lineages represented by the state vector index.
 * population: the current population of lineages represented by the state vector index.
 */
struct properties_two_locus {
    int descendants_l1;
    int descendants_l2;
    int population;
};

/**
 * Struct representing lineage properties corresponding to a state vector index:
 * descendants: number of decendendants for lineages represented by the state vector index.
 * is_derived: whether the lineage carries the derived mutation.
 * population: the current population of lineages represented by the state vector index.
 */
struct properties_single_locus_derived {
    int descendants;
    int is_derived;
    int population;
};

/**
 * Struct representing lineage properties corresponding to a state vector index:
 * descendants_l1: number of decendendants at locus 1 for lineages represented by the state vector index.
 * descendants_l2: number of decendendants at locus 2 for lineages represented by the state vector index.
 * is_derived: whether the lineage carries the derived mutation.
 * population: the current population of lineages represented by the state vector index.
 */
struct properties_two_locus_derived {
    int descendants_l1;
    int descendants_l2;
    int is_derived;
    int population;
};

/**
 * Converts a zero-based state vector index to a props
 *
 * @param s Sample size.
 * @param i Index to be converted.
 * @return Container struct for `index`.
 */
static inline properties_single_locus _index_to_props_single_locus(int s, int i) {
    properties_single_locus props;
    props.population = i / pow((s+1),1) + 1;
    i = fmod(i, pow((s+1),1));
    props.descendants = i;
    return props;
}

// [[Rcpp::export]]
List index_to_props_single_locus(int s, int i) {
    i  -= 1;
    properties_single_locus props = _index_to_props_single_locus(s, i);
    return List::create(
        Named("descendants") = props.descendants,
        _["population"] = props.population
        );
}

/**
 * Converts a zero-based state vector index to a props
 *
 * @param s Sample size.
 * @param i Index to be converted.
 * @return Container struct for `index`.
 */static inline properties_two_locus _index_to_props_two_locus(int s, int i) {
    properties_two_locus props;
    props.population = i / pow((s+1),2) + 1;
    i = fmod(i, pow((s+1),2));
    props.descendants_l2 = i / pow((s+1),1);
    i = fmod(i, pow((s+1),1));
    props.descendants_l1 = i;
    return props;
}

// [[Rcpp::export]]
List index_to_props_two_locus(int s, int i) {
    i  -= 1;
    properties_two_locus props = _index_to_props_two_locus(s, i);
    return List::create(
        Named("descendants_l1") = props.descendants_l1,
        _["descendants_l2"] = props.descendants_l2, 
        _["population"] = props.population
        );
}

/**
 * Converts a zero-based state vector index to a props
 *
 * @param s Sample size.
 * @param i Index to be converted.
 * @return Container struct for `index`.
 */
static inline properties_single_locus_derived _index_to_props_single_locus_derived(int s, int i) {
    properties_single_locus_derived props;
    props.population = i / pow((s+1),2) + 1;
    i = fmod(i, pow((s+1),2));
    props.is_derived = i / pow((s+1),1);
    i = fmod(i, pow((s+1),1));
    props.descendants = i;
    return props;
}

// [[Rcpp::export]]
List index_to_props_single_locus_derived(int s, int i) {
    i  -= 1;
    properties_single_locus_derived props = _index_to_props_single_locus_derived(s, i);
    return List::create(
        Named("descendants") = props.descendants,
        _["is_derived"] = props.is_derived,
        _["population"] = props.population
        );
}
/**
 * Converts a zero-based state vector index to a props
 *
 * @param s Sample size.
 * @param i Index to be converted.
 * @return Container struct for `index`.
 */
static inline properties_two_locus_derived _index_to_props_two_locus_derived(int s, int i) {
    properties_two_locus_derived props;
    props.population = i / pow((s+1),3) + 1;
    i = fmod(i, pow((s+1),3));
    props.is_derived = i / pow((s+1),2);
    i = fmod(i, pow((s+1),2));
    props.descendants_l2 = i / pow((s+1),1);
    i = fmod(i, pow((s+1),1));
    props.descendants_l1 = i;
    return props;
}

// [[Rcpp::export]]
List index_to_props_two_locus_derived(int s, int i) {
    i  -= 1;
    properties_two_locus_derived props = _index_to_props_two_locus_derived(s, i);
    return List::create(
        Named("descendants_l1") = props.descendants_l1,
        _["descendants_l2"] = props.descendants_l2, 
        _["is_derived"] = props.is_derived, 
        _["population"] = props.population
        );
}

/**
 * Converts a lineage properties to a state vector index.
 *
 * @param s Sample size.
 * @param a Number of decendants of lineage at locus one.
 * @param p Integer label of the population where the lineage is located (defaults to 1)
 * @return State vector index.
 */
static inline int _props_to_index_single_locus(int s, int a, int p=1) {
        return 
            a * pow((s+1),0) +
            (p-1) * pow((s+1),1);
}

// [[Rcpp::export]]
int props_to_index_single_locus(int s, int a, int p=1)  {
    return _props_to_index_single_locus(s, a, p=p) + 1;
}

/**
 * Converts a lineage properties to a state vector index.
 *
 * @param s Sample size.
 * @param a Number of decendants of lineage at locus one.
 * @param b Number of decendants of lineage at locus two.
 * @param p Integer label of the population where the lineage is located (defaults to 1)
 * @return State vector index.
 */
static inline int _props_to_index_two_locus(int s, int a, int b, int p=1) {
        return 
            a * pow((s+1),0) + 
            b * pow((s+1),1) +
            (p-1) * pow((s+1),2);
}

// [[Rcpp::export]]
int props_to_index_two_locus(int s, int a, int b, int p=1)  {
    return _props_to_index_two_locus(s, a, b, p=p) + 1;
}

/**
 * Converts a lineage properties to a state vector index.
 *
 * @param s Sample size.
 * @param a Number of decendants of lineage at locus one.
 * @param d Whether the lineage carries the derived varian (0 or 1).
 * @param p Integer label of the population where the lineage is located (defaults to 1)
 * @return State vector index.
 */
static inline int _props_to_index_single_locus_derived(int s, int a, int d, int p=1) {
        return 
            a * pow((s+1),0) + 
            d * pow((s+1),1) +
            (p-1) * pow((s+1),2);
}

// [[Rcpp::export]]
int props_to_index_single_locus_derived(int s, int a, int d, int p=1)  {
    return _props_to_index_single_locus_derived(s, a, d, p=p) + 1;
}

/**
 * Converts a lineage properties to a state vector index.
 *
 * @param s Sample size.
 * @param a Number of decendants of lineage at locus one.
 * @param b Number of decendants of lineage at locus two.
 * @param d Whether the lineage carries the derived varian (0 or 1).
 * @param p Integer label of the population where the lineage is located (defaults to 1)
 * @return State vector index.
 */
static inline int _props_to_index_two_locus_derived(int s, int a, int b, int d, int p=1) {
        return 
            a * pow((s+1),0) + 
            b * pow((s+1),1) + 
            d * pow((s+1),2) +
            (p-1) * pow((s+1),3);
}

// [[Rcpp::export]]
int props_to_index_two_locus_derived(int s, int a, int b, int d, int p=1)  {
    return _props_to_index_two_locus_derived(s, a, b, d, p=p) + 1;
}

