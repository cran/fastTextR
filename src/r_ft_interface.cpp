
// [[Rcpp::plugins("cpp11")]]

#include "r_ft_interface.h"

double cosine_similarity(std::vector<double> a, std::vector<double> b) {
    if ( a.size() != b.size() )
        return( -1.0 );
    double ab, a2, b2;
    ab = 0.0; a2 = 0.0; b2 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        ab += (a[i] * b[i]);
        a2 += pow(a[i], 2);
        b2 += pow(b[i], 2);
    }
    return ( ab / (sqrt(a2) * sqrt(b2)) );
}

namespace fasttext {

// ###########################
// Args
// ###########################
void Args::set_input_file(std::string input_file) {
    input = input_file;
}

void Args::set_output_file(std::string output_file) {
    output = output_file;
}

void Args::set_test_file(std::string test_file) {
    test = test_file;
}

void Args::set_pretrained_vec(std::string pretrained_file) {
    pretrainedVectors = pretrained_file;
}

void Args::init_from_list(Rcpp::List control) {
    std::string method = control["method"];
    if ( method == "supervised" ) {
        model = model_name::sup;
        loss = loss_name::softmax;
        minCount = 1;
        minn = 0;
        maxn = 0;
        lr = 0.1;
    } else if ( method == "cbow" ) {
        model = model_name::cbow;
    } else if ( method == "skipgram" ) {
        model = model_name::sg;
    } else {
        Rcpp::stop("unkown method!");
    }
    
    std::string xloss = control["loss"];
    if ( xloss == "softmax" ) {
            loss = loss_name::softmax;
    } else if ( xloss == "hs" ) {
        loss = loss_name::hs;
    } else if ( xloss == "ns" ) {
        loss = loss_name::ns;
    } else {
        Rcpp::stop("unkown loss!");
    }

    input             = Rcpp::as<std::string>( control["input"] );
    test              = Rcpp::as<std::string>( control["test"] );
    output            = Rcpp::as<std::string>( control["output"] );
    lr                = control["learning_rate"];    
    lrUpdateRate      = control["learn_update"];
    dim               = control["word_vec_size"];
    ws                = control["window_size"];
    epoch             = control["epoch"];
    minCount          = control["min_count"];
    minCountLabel     = control["min_count_label"];
    neg               = control["neg"];
    wordNgrams        = control["max_len_ngram"];        
    bucket            = control["nbuckets"];
    minn              = control["min_ngram"];
    maxn              = control["max_ngram"];
    thread            = control["nthreads"];
    t                 = control["threshold"];
    label             = Rcpp::as<std::string>( control["label"] );
    verbose           = control["verbose"];
    pretrainedVectors = Rcpp::as<std::string>( control["pretrained_vectors"] );
}

Rcpp::List Args::args_to_list() {
    Rcpp::List x;
    x["input"] = input;
    x["test"] = test;
    x["output"] = output;
    x["learning_rate"] = lr;
    x["learn_update"] = lrUpdateRate;
    x["word_vec_size"] = dim;
    x["window_size"] = ws;
    x["epoch"] = epoch;
    x["min_count"] = minCount;
    x["min_count_label"] = minCountLabel;
    x["neg"] = neg;
    x["max_len_ngram"] = wordNgrams;  
    if ( model == model_name::cbow ) {
        x["method"] = "cbow";
    } else if ( model == model_name::sg ) {
        x["method"] = "skipgram";
    } if ( model == model_name::sup ) {
        x["method"] = "supervised";
    }
    
    if ( loss == loss_name::hs ) {
        x["loss"] = "hs";
    } else if ( loss == loss_name::ns ) {
        x["loss"] = "ns";
    } else if ( loss == loss_name::softmax ) {
        x["loss"] = "softmax";
    }

    x["nbuckets"] = bucket;
    x["min_ngram"] = minn;
    x["max_ngram"] = maxn;
    x["nthreads"] = thread;
    x["threshold"] = t;
    x["label"] = label;
    x["verbose"] = verbose;
    x["pretrained_vectors"] = pretrainedVectors;
    return x;   
}

// ###########################
// FastText
// ###########################
// ###########################
//  R - Modifications
// ###########################
std::shared_ptr<Args>& FastText::get_args() {
    return args_;
}

std::shared_ptr<Dictionary>& FastText::get_dict() {
    return dict_;
}

std::shared_ptr<Matrix>& FastText::get_input() {
    return input_;
}

std::shared_ptr<Matrix>& FastText::get_output() {
    return output_;
}

std::shared_ptr<Model>& FastText::get_model() {
    return model_;
}

int32_t FastText::get_dim() {
    return args_->dim;
}

int64_t FastText::get_token_count() {
    int64_t token_count = tokenCount;
    return token_count;
}

std::string FastText::get_model_type() {
    std::string model_type = "";
    if ( args_->model == model_name::cbow ) {
        model_type = "cbow";
    } else if ( args_->model == model_name::sg ) {
        model_type = "skipgram";
    } if ( args_->model == model_name::sup ) {
        model_type = "supervised";
    }
    return model_type;
}

std::vector<double>  FastText::r_test(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::vector<double> res;
  res.push_back( precision / (k * nexamples) );    // P@k Precision
  res.push_back( precision / nlabels );            // R@k Recall
  res.push_back( static_cast<double>(nexamples) );
  res.push_back( precision ); // number of correctly predicted items
  return res;
}

void FastText::r_predict(std::istream& in, std::ofstream& ofs, int32_t k, bool print_prob) {
    std::vector<std::pair<real,std::string>> predictions;
    while (in.peek() != EOF) {
        predict(in, k, predictions);
        if (predictions.empty()) {
            ofs << "n/a" << std::endl;
            continue;
        }

        for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            if (it != predictions.cbegin()) {
                ofs << ' ';
            }
            ofs << it->second;
            if (print_prob) {
                ofs << ' ' << exp(it->first);
            }
        }
        ofs << std::endl;
    }
}

void FastText::r_predict(std::istream& in, int32_t k, bool print_prob, PairVector& r_pred) const {
    std::vector<std::pair<real,std::string>> predictions;
  
    while (in.peek() != EOF) {
        predict(in, k, predictions);
        if ( predictions.empty() ) {
            // Rcpp::Rcout << "n/a" << std::endl;
            r_pred.labels.push_back( "NA" );
            r_pred.prob.push_back( -1.0 );
            continue;
        }
        for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            if (it != predictions.cbegin()) {
                // Rcpp::Rcout << ' ';
            }
            // Rcpp::Rcout << it->second;
            r_pred.labels.push_back( it->second );
            if (print_prob) {
                // Rcpp::Rcout << ' ' << exp(it->first);
                r_pred.prob.push_back( exp(it->first) );
            }
        }
        // Rcpp::Rcout << std::endl;
    }
}

void FastText::r_predict(std::vector<std::string>& newdata, int32_t k, 
                         bool print_prob, PairVector& r_pred) const {
    std::vector<std::pair<real,std::string>> predictions;
    
    for (size_t i=0; i < newdata.size(); i++) {
        std::istringstream iss (newdata[i]);
        predict(iss, k, predictions);
        if ( predictions.empty() ) {
            // Rcpp::Rcout << "n/a" << std::endl;
            r_pred.labels.push_back( "NA" );
            r_pred.prob.push_back( -1.0 );
            continue;
        }
        for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            if (it != predictions.cbegin()) {
                // Rcpp::Rcout << ' ';
            }
            // Rcpp::Rcout << it->second;
            r_pred.labels.push_back( it->second );
            if (print_prob) {
                // Rcpp::Rcout << ' ' << exp(it->first);
                r_pred.prob.push_back( exp(it->first) );
            }
        }
    }
}

void FastText::r_word_vector(std::vector<std::string>& words, 
                             std::vector<double>& word_vec) {
    std::string word;
    int32_t dim = args_->dim;
    std::vector<double> vec(dim, 0);
    int32_t words_len = words.size();
    int64_t input_ncol = input_->n_;
    int64_t i;
    for (int32_t m = 0; m < words_len; m++) {
        word = words[m];
        
        // getVector(vec, word);
        const std::vector<int32_t>& ngrams = dict_->getNgrams(word);
        double ngram_len = ngrams.size();

        for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
            if (it != ngrams.end()) {
                i = (int64_t)*it;
            }
            // vec.addRow(*input_, *it);
            for (int64_t j = 0; j < input_ncol; j++) {
                vec[j] += input_->data_[i * input_ncol + j];
            }
        }
        for (int32_t n = 0; n < dim; n++) {
            if ( ngram_len > 0.5 ) {
                vec[n] /= ngram_len;
            }
            word_vec.push_back( vec[n] ); // the fasttext type real is the same as float
            vec[n] = 0.0;
        }
    }
}

// NOTE: word_vec has to have alredy the correct length
void FastText::r_get_word_vector(std::string& word, std::vector<double>& word_vec) {
    
    int32_t dim = args_->dim;
    int64_t input_ncol = input_->n_;
    std::fill(word_vec.begin(), word_vec.end(), 0);
    int64_t i;

    if ( input_ncol != dim )
        Rcpp::stop("dimension missmatch! This should never happen!");
        
    // getVector(vec, word);
    const std::vector<int32_t>& ngrams = dict_->getNgrams(word);
    double ngram_len = ngrams.size();

    for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
        // if (it != ngrams.end()) {
            i = (int64_t)*it;
        // }
        // vec.addRow(*input_, *it);
        for (int64_t j = 0; j < input_ncol; j++) {
            word_vec[j] += input_->data_[i * input_ncol + j];
        }
    }
    for (int32_t n = 0; n < dim; n++) {
        if ( ngram_len > 0.5 ) {
            word_vec[n] /= ngram_len;
        }
    }
}

void FastText::k_most_silmilar(std::string word, int k,
                     std::vector<double>& similarity,
                     std::vector<std::string>& sim_words) {
    double min_sim = 0;
    int32_t nrow = dict_->nwords();
    int32_t dim = args_->dim;
    std::string word_m;
    std::vector<double> word_vec_word(dim);
    std::vector<double> word_vec(dim);
    double sim;

    r_get_word_vector(word, word_vec_word);

    for (int32_t m = 0; m < nrow; m++) {
        word_m = dict_->getWord(m);
        if ( word == word_m )
            continue;
        // Would be better if I don't use push_back in 'r_get_word_vector'
        // since the size never changes.
        r_get_word_vector(word_m, word_vec);
        sim = cosine_similarity(word_vec_word, word_vec);
        // I don't handle ties!
        if ( sim > min_sim ) {
            for (int32_t i = 0; i < k; i++) {
                if ( similarity[i] < sim ) {
                    similarity.insert(similarity.begin() + i, sim);
                    sim_words.insert(sim_words.begin() + i, word_m);
                    similarity.pop_back();
                    sim_words.pop_back();
                    break;
                }
            }
            min_sim = similarity.back();
        }
    }
}

} // end namespace

// ###########################
// R - Interface
// ###########################

// [[Rcpp::export]]
int Rft_dict_get_nwords(SEXP r_dict) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    int32_t n = dict->nwords();
    return n;
}

// [[Rcpp::export]]
int Rft_dict_get_nlabels(SEXP r_dict) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    int32_t n = dict->nlabels();
    return n;
}

// [[Rcpp::export]]
int Rft_dict_get_ntokens(SEXP r_dict) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    int32_t n = dict->ntokens();
    return n;
}

// [[Rcpp::export]]
std::string Rft_dict_get_word(SEXP r_dict, int i) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::string word = dict->getWord(i - 1);
    return word;
}

// [[Rcpp::export]]
std::vector<std::string> Rft_dict_get_words(SEXP r_dict, std::vector<int> ind) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::vector<std::string> words;
    for ( size_t i=0; i < ind.size(); i++ ) {
        words.push_back( dict->getWord(ind[i] - 1) );
    }
    return words;
}

// [[Rcpp::export]]
std::vector<std::string> Rft_dict_get_all_words(SEXP r_dict) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::vector<std::string> words;
    int32_t nwords = dict->nwords();
    for ( int32_t i=0; i < nwords; i++ ) {
        words.push_back( dict->getWord( i ) );
    }
    return words;
}

// [[Rcpp::export]]
std::string Rft_dict_get_label(SEXP r_dict, int i) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::string label = dict->getLabel(i - 1);
    return label;
}

// [[Rcpp::export]]
std::vector<std::string> Rft_dict_get_labels(SEXP r_dict, std::vector<int> ind) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::vector<std::string> labels;
    for ( size_t i=0; i < ind.size(); i++ ) {
        labels.push_back( dict->getLabel(ind[i] - 1) );
    }
    return labels;
}

// [[Rcpp::export]]
std::vector<std::string> Rft_dict_get_all_labels(SEXP r_dict) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::vector<std::string> labels;
    int32_t nlabels = dict->nlabels(); // FIXME: here shouldn't be nlabels but most likely
    // size_ but than I don't really need this function.
    for ( int32_t i=0; i < nlabels; i++ ) {
        labels.push_back( dict->getLabel( i ) );
    }
    return labels;
}

// [[Rcpp::export]]
SEXP Rft_dict_read_from_file(SEXP r_dict, std::string file_name) {
    Rcpp::XPtr<Dictionary>dict(r_dict);
    std::ifstream ifs(file_name);
    dict->readFromFile(ifs);
    ifs.close();
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP Rft_ft_dim_input_matrix(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    NumericVector dim(2);
    dim[0] = (double)ft->get_input()->m_;
    dim[1] = (double)ft->get_input()->n_;
    return dim;
}

// [[Rcpp::export]]
SEXP Rft_ft_dim_output_matrix(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    NumericVector dim(2);
    dim[0] = (double)ft->get_output()->m_;
    dim[1] = (double)ft->get_output()->n_;
    return dim;
}

// [[Rcpp::export]]
SEXP Rft_ft_get_args(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    Rcpp::XPtr<Args>ptr(ft->get_args().get(), false);
    return ptr;
}

// [[Rcpp::export]]
SEXP Rft_ft_get_dict(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    Rcpp::XPtr<Dictionary>ptr(ft->get_dict().get(), false);
    return ptr;
}

// [[Rcpp::export]]
SEXP Rft_ft_get_input_matrix(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    int64_t nrow = ft->get_input()->m_;
    int64_t ncol = ft->get_input()->n_;
    if ( (nrow > 2147483647) | (ncol > 2147483647) ) {
        Rcpp::stop("matrix is to big for R!");
    }
    NumericMatrix mat(nrow, ncol);
    for (int64_t m=0; m < nrow; m++) {
        for (int64_t n=0; n < ncol; n++) {
            mat(m, n) = ft->get_input()->data_[m * ncol + n];
        }
    }
    return mat;
}

// [[Rcpp::export]]
NumericMatrix Rft_ft_get_output_matrix(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    int64_t nrow = ft->get_output()->m_;
    int64_t ncol = ft->get_output()->n_;
    if ( (nrow > 2147483647) | (ncol > 2147483647) ) {
        Rcpp::stop("matrix is to big for R!");
    }
    NumericMatrix mat(nrow, ncol);
    for (int64_t m=0; m < nrow; m++) {
        for (int64_t n=0; n < ncol; n++) {
            mat(m, n) = ft->get_output()->data_[m * ncol + n];
        }
    }
    return mat;
}

// [[Rcpp::export]]
SEXP Rft_ft_get_model(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    Rcpp::XPtr<Model>ptr(ft->get_model().get(), false);
    return ptr;
}

// [[Rcpp::export]]
double Rft_ft_get_token_count(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    double token_count = ft->get_token_count();
    return token_count;
}

// [[Rcpp::export]]
std::string Rft_ft_get_model_type(SEXP r_ft) {
    Rcpp::XPtr<FastText>ft(r_ft);
    std::string model_type = ft->get_model_type();
    return model_type;
}

// Constructors
// ============
// [[Rcpp::export]]
SEXP Rft_new_model() {
    Rcpp::XPtr<FastText> p(new FastText(), true);
    return p;   
}

// [[Rcpp::export]]
SEXP Rft_new_args() {
    Rcpp::XPtr<Args> p(new Args(), true);
    return p;   
}

// [[Rcpp::export]]
SEXP Rft_new_dict(SEXP control) {
    std::shared_ptr<Args> a = std::make_shared<Args>();
    a->init_from_list(control);
    Rcpp::XPtr<Dictionary> p(new Dictionary(a), true);
    return p;   
}

// Load / Save Model
// =================
// [[Rcpp::export]]
SEXP Rft_load_model(std::string file_name) {
    Rcpp::XPtr<FastText> fast_text(new FastText(), true);
    fast_text->loadModel( file_name );
    return fast_text;
}

// [[Rcpp::export]]
SEXP Rft_save_model(SEXP ft, std::string file_name) {
    Rcpp::XPtr<FastText>fast_text(ft);
    fast_text->get_args()->set_output_file(file_name);
    fast_text->saveModel();
    if ( fast_text->get_model_type() != "supervised" ) {
        fast_text->saveVectors();
    }
    return R_NilValue;
}

// Typecast
// ========
// [[Rcpp::export]]
List Rft_args_to_list(SEXP args) {
    Rcpp::XPtr<Args>ptr(args);
    return ptr->args_to_list();
}

// Train
// =====
// [[Rcpp::export]]
SEXP Rft_train(SEXP control, int save_to_file) {
    std::shared_ptr<Args> a = std::make_shared<Args>();
    a->init_from_list(control);

    // In this case we just init the args and run the model as normal
    if ( save_to_file > 0 ) {
        FastText fast_text;
        fast_text.train(a, 0);
        return R_NilValue;
    }
    Rcpp::XPtr<FastText> fast_text(new FastText(), true);
    fast_text->train(a, 1);
    return fast_text;
}

// Test
// ====
// [[Rcpp::export]]
std::vector<double> Rft_test(SEXP ft, std::string test_file, int k_most_likely) {
    Rcpp::XPtr<FastText>fast_text(ft);

    std::ifstream ifs(test_file);
    if (!ifs.is_open()) 
        Rcpp::stop("Test file cannot be opened!");

    std::vector<double> res = fast_text->r_test(ifs, k_most_likely);
    ifs.close();

    return res;
}

// Predict
// =======
// [[Rcpp::export]]
SEXP Rft_predict_to_file(SEXP ft, std::string newdata_file, std::string result_file, 
                         int k_most_likely, bool print_prob) {
    Rcpp::XPtr<FastText>fast_text(ft);

    std::ifstream ifs(newdata_file);
    if (!ifs.is_open()) {
        Rcpp::stop("Prediction file cannot be opened!");
    }

    std::ofstream ofs(result_file);
    if ( !ofs.is_open() ) {
        Rcpp::stop("Result file cannot be opened for saving!");
    }

    fast_text->r_predict(ifs, ofs, k_most_likely, print_prob);

    ifs.close();
    ofs.close();

    return R_NilValue;
}

// NOTE: The model is already loaded at this time.
// [[Rcpp::export]]
SEXP Rft_predict(SEXP ft, std::string newdata_file, int k_most_likely, bool print_prob) {
    Rcpp::XPtr<FastText>fast_text(ft);

    std::ifstream ifs(newdata_file);
    if (!ifs.is_open()) {
        Rcpp::stop("Prediction file cannot be opened!");
    }

    // FastText::r_predict(std::istream& in, int32_t k, PairVector& r_pred)
    PairVector r_pred;
    fast_text->r_predict(ifs, k_most_likely, print_prob, r_pred);
    
    return List::create(_["label"] = r_pred.labels, _["prob"] = r_pred.prob);
}

// [[Rcpp::export]]
SEXP Rft_vec_predict(SEXP ft, std::vector<std::string> newdata,
                     int k_most_likely, bool print_prob) {
    Rcpp::XPtr<FastText>fast_text(ft);
    
    PairVector r_pred;
    fast_text->r_predict(newdata, k_most_likely, print_prob, r_pred);

    return List::create(_["label"] = r_pred.labels, _["prob"] = r_pred.prob);
}

// [[Rcpp::export]]
std::vector<double> Rft_get_word_vectors(SEXP ft, std::vector<std::string> words) {
    Rcpp::XPtr<FastText>fast_text(ft);

    std::vector<double> word_vec;
    fast_text->r_word_vector(words, word_vec);

    return word_vec;
}

// [[Rcpp::export]]
SEXP Rft_get_all_word_vectors(SEXP ft){
    Rcpp::XPtr<FastText>fast_text(ft);

    std::shared_ptr<fasttext::Dictionary>& dict = fast_text->get_dict();

    int32_t ncol = fast_text->get_dim();
    int32_t nrow = dict->nwords();
    
    Rcpp::NumericMatrix mat(nrow, ncol);
    std::string word;
    std::vector<std::string> rownames;
    std::vector<double> word_vec(ncol);

    for (int32_t m = 0; m < nrow; m++) {
        word = dict->getWord(m);
        rownames.push_back( word );
        fast_text->r_get_word_vector(word, word_vec);
        for (int32_t n = 0; n < ncol; n++) {
            mat(m, n) = word_vec[n];
        }
        word_vec.clear();
    }
    Rcpp::List dimnames = Rcpp::List::create(rownames, R_NilValue);
    mat.attr("dimnames") = dimnames;
    return mat;
}

// [[Rcpp::export]]
double Rft_similarity(SEXP ft, std::string word_a, std::string word_b){
    Rcpp::XPtr<FastText>fast_text(ft);

    int32_t dim = fast_text->get_dim();
    std::vector<double> word_vec_a(dim);
    std::vector<double> word_vec_b(dim);

    fast_text->r_get_word_vector(word_a, word_vec_a);
    fast_text->r_get_word_vector(word_b, word_vec_b);

    return cosine_similarity(word_vec_a, word_vec_b);
}

// [[Rcpp::export]]
SEXP Rft_k_most_silmilar(SEXP ft, std::vector<std::string> words, int k) {
    Rcpp::XPtr<FastText>fast_text(ft);

    std::string word;
    std::vector<double> similarities(k);
    std::vector<std::string> sim_words(k);
    List ret(words.size());
    NumericVector rvec(k);
    for (size_t i = 0; i < words.size(); i++) {
        std::fill(similarities.begin(), similarities.begin()+k, 0);
        std::fill(sim_words.begin(), sim_words.begin()+k, "");
        word = words[i];
        fast_text->k_most_silmilar(word, k, similarities, sim_words);
        rvec = Rcpp::wrap(similarities);
        rvec.attr("names") = sim_words;
        ret[i] = rvec;
    }
    return ret;
}

int is_break(std::string s) {
    if ( s.length() < 4 ) return 0;
    if ( s.compare(0, 4, "<br>") == 0 ) return 3;
    if ( s.compare(0, 5, "<br >") == 0 ) return 4;
    if ( s.compare(0, 6, "<br />") == 0 ) return 5;
    return 0;
}

// s1 == " " | s1 == "\a" | s1 == "\b" | s1 == "\f" 
// | s1 == "\n" | s1 == "\r" | s1 == "\t" | s1 == "\v"
int is_control_char(std::string s1) {
    if ( (s1 == " ") | (s1 == "\a") | (s1 == "\b") | (s1 == "\f") 
       | (s1 == "\n") | (s1 == "\r") | (s1 == "\t") | (s1 == "\v") ) {
        return 1;
    }
    return 0;
}

// s1 == "\'" | s1 == "\"" | s1 == "." | s1 == "," | s1 == "("
//   | s1 == ")" | s1 == "!" | s1 == "\?"
int is_punctation(std::string s1) {
    if ( (s1 == "\'") | (s1 == "\"") | (s1 == ".") | (s1 == ",") | (s1 == "(")
       | (s1 == ")") | (s1 == "!") | (s1 == "\?") ) {
        return 1;
    }
    return 0;
}

// ## add space
// ## "'", '"', ".", ",", "(", ")", "!", "?", ""
// ## replace with " "
// ## "<br />", ";", ":", 
// ## "<br />"
// [[Rcpp::export]]
std::vector<std::string> clean_text(std::vector<std::string> x) {
    std::vector<std::string> y(x.size());
    std::string s0, s1, s2;
    for (size_t i = 0; i < x.size(); i++) {
        s1 = ""; s2 = ""; y[i] = "";
        size_t n = x[i].length();
        for (size_t j = 0; j < n; j++) {
            s1 = x[i].at(j);
            if ( j + 1 <  n ) {
                s2 = x[i].at(j + 1);
            }

            if ( is_punctation(s1) ) {
                if ( s0 != " " ) {
                    y[i] += " ";
                }
                y[i] += s1;
                if ( j + 1 <  n ) {
                    if ( s2 != " ") {
                        y[i] += " ";       
                    }
                }
            } else if ( is_control_char(s1) ) {
                s1 = " ";
                if ( s0 != " " ) {
                    y[i] += " ";
                }
            } else if ( (s1 == ";") | (s1 == ":") ) {
                if ( s0 != " " ) {
                    y[i] += " ";
                }
            } else if ( (s1 == "<") & (s2 == "b") & (n - j > 4) ) {
                size_t len = ( 6 < (n - j) ) ? 6 : (n-j);
                j += is_break(x[i].substr(j, len));
                if ( j > 0 ) {
                    if ( s0 != " " ) {
                        y[i] += " ";       
                    }
                } else {
                    y[i] += s1;    
                }
            } else {
                y[i] += s1;
            }
            s0 = y[i].back();
        }
    }
    return y;
}

