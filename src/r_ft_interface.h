
#ifndef R_FASTTEXT_FASTTEXT_H
#define R_FASTTEXT_FASTTEXT_H

#include <thread>

#include <Rcpp.h>
#include "fasttext.h"

using namespace Rcpp;
using namespace fasttext;

double cosine_similarity(std::vector<double> a, std::vector<double> b);

#endif
