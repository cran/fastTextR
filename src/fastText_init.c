#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP fastTextR_clean_text(SEXP);
extern SEXP fastTextR_Rft_args_to_list(SEXP);
extern SEXP fastTextR_Rft_dict_get_all_labels(SEXP);
extern SEXP fastTextR_Rft_dict_get_all_words(SEXP);
extern SEXP fastTextR_Rft_dict_get_label(SEXP, SEXP);
extern SEXP fastTextR_Rft_dict_get_labels(SEXP, SEXP);
extern SEXP fastTextR_Rft_dict_get_nlabels(SEXP);
extern SEXP fastTextR_Rft_dict_get_ntokens(SEXP);
extern SEXP fastTextR_Rft_dict_get_nwords(SEXP);
extern SEXP fastTextR_Rft_dict_get_word(SEXP, SEXP);
extern SEXP fastTextR_Rft_dict_get_words(SEXP, SEXP);
extern SEXP fastTextR_Rft_dict_read_from_file(SEXP, SEXP);
extern SEXP fastTextR_Rft_ft_dim_input_matrix(SEXP);
extern SEXP fastTextR_Rft_ft_dim_output_matrix(SEXP);
extern SEXP fastTextR_Rft_ft_get_args(SEXP);
extern SEXP fastTextR_Rft_ft_get_dict(SEXP);
extern SEXP fastTextR_Rft_ft_get_input_matrix(SEXP);
extern SEXP fastTextR_Rft_ft_get_model(SEXP);
extern SEXP fastTextR_Rft_ft_get_model_type(SEXP);
extern SEXP fastTextR_Rft_ft_get_output_matrix(SEXP);
extern SEXP fastTextR_Rft_ft_get_token_count(SEXP);
extern SEXP fastTextR_Rft_get_all_word_vectors(SEXP);
extern SEXP fastTextR_Rft_get_word_vectors(SEXP, SEXP);
extern SEXP fastTextR_Rft_k_most_silmilar(SEXP, SEXP, SEXP);
extern SEXP fastTextR_Rft_load_model(SEXP);
extern SEXP fastTextR_Rft_new_args();
extern SEXP fastTextR_Rft_new_dict(SEXP);
extern SEXP fastTextR_Rft_new_model();
extern SEXP fastTextR_Rft_predict(SEXP, SEXP, SEXP, SEXP);
extern SEXP fastTextR_Rft_predict_to_file(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP fastTextR_Rft_save_model(SEXP, SEXP);
extern SEXP fastTextR_Rft_similarity(SEXP, SEXP, SEXP);
extern SEXP fastTextR_Rft_test(SEXP, SEXP, SEXP);
extern SEXP fastTextR_Rft_train(SEXP, SEXP);
extern SEXP fastTextR_Rft_vec_predict(SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"fastTextR_clean_text",               (DL_FUNC) &fastTextR_clean_text,               1},
    {"fastTextR_Rft_args_to_list",         (DL_FUNC) &fastTextR_Rft_args_to_list,         1},
    {"fastTextR_Rft_dict_get_all_labels",  (DL_FUNC) &fastTextR_Rft_dict_get_all_labels,  1},
    {"fastTextR_Rft_dict_get_all_words",   (DL_FUNC) &fastTextR_Rft_dict_get_all_words,   1},
    {"fastTextR_Rft_dict_get_label",       (DL_FUNC) &fastTextR_Rft_dict_get_label,       2},
    {"fastTextR_Rft_dict_get_labels",      (DL_FUNC) &fastTextR_Rft_dict_get_labels,      2},
    {"fastTextR_Rft_dict_get_nlabels",     (DL_FUNC) &fastTextR_Rft_dict_get_nlabels,     1},
    {"fastTextR_Rft_dict_get_ntokens",     (DL_FUNC) &fastTextR_Rft_dict_get_ntokens,     1},
    {"fastTextR_Rft_dict_get_nwords",      (DL_FUNC) &fastTextR_Rft_dict_get_nwords,      1},
    {"fastTextR_Rft_dict_get_word",        (DL_FUNC) &fastTextR_Rft_dict_get_word,        2},
    {"fastTextR_Rft_dict_get_words",       (DL_FUNC) &fastTextR_Rft_dict_get_words,       2},
    {"fastTextR_Rft_dict_read_from_file",  (DL_FUNC) &fastTextR_Rft_dict_read_from_file,  2},
    {"fastTextR_Rft_ft_dim_input_matrix",  (DL_FUNC) &fastTextR_Rft_ft_dim_input_matrix,  1},
    {"fastTextR_Rft_ft_dim_output_matrix", (DL_FUNC) &fastTextR_Rft_ft_dim_output_matrix, 1},
    {"fastTextR_Rft_ft_get_args",          (DL_FUNC) &fastTextR_Rft_ft_get_args,          1},
    {"fastTextR_Rft_ft_get_dict",          (DL_FUNC) &fastTextR_Rft_ft_get_dict,          1},
    {"fastTextR_Rft_ft_get_input_matrix",  (DL_FUNC) &fastTextR_Rft_ft_get_input_matrix,  1},
    {"fastTextR_Rft_ft_get_model",         (DL_FUNC) &fastTextR_Rft_ft_get_model,         1},
    {"fastTextR_Rft_ft_get_model_type",    (DL_FUNC) &fastTextR_Rft_ft_get_model_type,    1},
    {"fastTextR_Rft_ft_get_output_matrix", (DL_FUNC) &fastTextR_Rft_ft_get_output_matrix, 1},
    {"fastTextR_Rft_ft_get_token_count",   (DL_FUNC) &fastTextR_Rft_ft_get_token_count,   1},
    {"fastTextR_Rft_get_all_word_vectors", (DL_FUNC) &fastTextR_Rft_get_all_word_vectors, 1},
    {"fastTextR_Rft_get_word_vectors",     (DL_FUNC) &fastTextR_Rft_get_word_vectors,     2},
    {"fastTextR_Rft_k_most_silmilar",      (DL_FUNC) &fastTextR_Rft_k_most_silmilar,      3},
    {"fastTextR_Rft_load_model",           (DL_FUNC) &fastTextR_Rft_load_model,           1},
    {"fastTextR_Rft_new_args",             (DL_FUNC) &fastTextR_Rft_new_args,             0},
    {"fastTextR_Rft_new_dict",             (DL_FUNC) &fastTextR_Rft_new_dict,             1},
    {"fastTextR_Rft_new_model",            (DL_FUNC) &fastTextR_Rft_new_model,            0},
    {"fastTextR_Rft_predict",              (DL_FUNC) &fastTextR_Rft_predict,              4},
    {"fastTextR_Rft_predict_to_file",      (DL_FUNC) &fastTextR_Rft_predict_to_file,      5},
    {"fastTextR_Rft_save_model",           (DL_FUNC) &fastTextR_Rft_save_model,           2},
    {"fastTextR_Rft_similarity",           (DL_FUNC) &fastTextR_Rft_similarity,           3},
    {"fastTextR_Rft_test",                 (DL_FUNC) &fastTextR_Rft_test,                 3},
    {"fastTextR_Rft_train",                (DL_FUNC) &fastTextR_Rft_train,                2},
    {"fastTextR_Rft_vec_predict",          (DL_FUNC) &fastTextR_Rft_vec_predict,          4},
    {NULL, NULL, 0}
};

void R_init_fastTextR(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}


