# -----------------------------------------------------------
#  ft.control
#  ==========
#' @title Default Control Settings
#' @description A auxiliary function for defining the control variables.
#' @param loss a character string giving the name of the loss function 
#'             allowed values are \code{'softmax'}, \code{'hs'} and \code{'ns'}.
#' @param learning_rate a numeric giving the learning rate, the default value is \code{0.05}.
#' @param learn_update an integer giving after how many tokens the learning rate
#'                     should be updated. The default value is \code{100L}, which
#'                     means the learning rate is updated every 100 tokens. 
#' @param word_vec_size an integer giving the length (size) of the word vectors.
#' @param window_size an integer giving the size of the context window.
#' @param epoch an integer giving the number of epochs.
#' @param min_count an integer giving the minimal number of word occurences.
#' @param min_count_label and integer giving the minimal number of label occurences.
#' @param neg an integer giving how many negatives are sampled (only used if loss is \code{"ns"}).
#' @param max_len_ngram an integer giving the maximum length of ngrams used.
#' @param nbuckets an integer giving the number of buckets.
#' @param min_ngram an integer giving the minimal ngram length.
#' @param max_ngram an integer giving the maximal ngram length.
#' @param nthreads an integer giving the number of threads.
#' @param threshold a numeric giving the sampling threshold.
#' @param label a character string specifying the label prefix (default is \code{'__label__'}).
#' @param verbose an integer giving the verbosity level, the default value
#'                is \code{0L} and shouldn't be changed since Rcpp::Rcout 
#'                cann't handle the traffic.
#' @param pretrained_vectors a character string giving the file path
#'                           to the pretrained word vectors which are used 
#'                           for the supervised learning.
#' @return a list with the control variables.
#' @examples
#' ft.control(learning_rate=0.1)
ft.control <- function(loss = c("softmax", "hs", "ns"), 
                       learning_rate=0.05, learn_update=100L, word_vec_size=5L, 
                       window_size=5L, epoch=5L, min_count=5L, min_count_label=0L,
                       neg=5L, max_len_ngram=1L, nbuckets=2000000L, min_ngram=3L,
                       max_ngram=6L, nthreads=1L, threshold=1e-4, label="__label__", 
                       verbose=0, pretrained_vectors="") {
    loss <- match.arg(loss)
    as.list(environment())
}

.control_arguments <- c("input", "test", "output", "learning_rate", "learn_update", 
                        "word_vec_size", "window_size", "epoch", "min_count", 
                        "min_count_label", "neg", "max_len_ngram", "nbuckets", 
                        "min_ngram", "max_ngram", "nthreads", "threshold", "label", 
                        "verbose", "pretrained_vectors")

# -----------------------------------------------------------
#  fasttext
#  ========
#' @title Train a Model
#' @description Train a new word representation model or supervised 
#'              classification model.
#' @param input a character string giving the location of the input file.
## @param output a character string giving the location of the output file.
#' @param method a character string giving the method, possible values are 
#'               \code{'supervised'}, \code{'cbow'} and \code{'skipgram'}.
#' @param control a list giving the control variables, for more information
#'                see \code{\link{ft.control}}.
#' @examples
#' \dontrun{
#' model <- fasttext("my_data.txt", method="supervised", 
#'                   control = ft.control(nthreads = 1L))
#' }
## fasttext <- function(input, output = "", 
##                      method = c("supervised", "cbow", "skipgram"), 
##                      control = ft.control()) {
fasttext <- function(input, method = c("supervised", "cbow", "skipgram"), 
                     control = ft.control()) {
    output <- ""
    method <- match.arg(method)
    stopifnot( is.character(input), is.character(output) )
    stopifnot( file.exists(input) )

    save_to_file <- if ( !nchar(output) ) 0L else 1L

    control$input <- input
    control$method <- method
    control$output <- output
    control$test <- ""
    if ( !all(.control_arguments %in% names(control)) ) {
        i <- which(!.control_arguments %in% names(control))
        if ( length(i) == 1) {
            stop("control argument '", .control_arguments[i], "' is missing")
        } else {
            stop("control arguments ", 
                 paste(shQuote(.control_arguments[i]), collapse=", "), 
                 " are missing")
        }
    }
    env <- new.env(parent<-emptyenv())
    env$pointer <- Rft_train(control, save_to_file)
    .wrap_model( env )
}

evaluate <- function(model, known_labels, newdata, k) UseMethod("evaluate")

# -----------------------------------------------------------
#  evaluate
#  ========
## @title Evaluate the Model
## @description Evaluate the quality of the predictions.
##              For the model evaluation precision and recall are used.
## @param model an object inheriting from \code{'fasttext'}.
## @param known_labels a character vector giving the known labels.
## @param newdata a character vector giving new data for the evaluation.
## @param k an integer giving the number of labels to be returned.
## @examples
## \dontrun{
## ft.test(model, test_file)
## }
evaluate.supervised_model <- function(model, known_labels, newdata, k=1L) {
    stopifnot( inherits(model, "fasttext") )
    if ( k != 1L ) 
        stop("k != 1 is not implemented yet")
    pred <- predict(model, known_labels, newdata, k = k)
    confusion_matrix <- table(known_labels = known_labels, predicted = pred[,1])
    list(confusion_matrix = confusion_matrix, 
         precision = precision(confusion_matrix),
         recall = recall(confusion_matrix),
         accuracy_score = accuracy_score(confusion_matrix))
}

.evaluate_supervised_model <- function(model, known_labels, newdata_file, k=1L) {
    stopifnot(is.character(newdata_file), file.exists(newdata_file))
    stopifnot( inherits(model, "fasttext") )
    pred <- predict(model, newdata_file, k = k)
    confusion_matrix <- table(known_labels = known_labels, predicted = pred[,1])
    list(confusion_matrix = confusion_matrix, 
         precision = precision(confusion_matrix),
         recall = recall(confusion_matrix),
         accuracy_score = accuracy_score(confusion_matrix))
}

precision <- function(confusion_matrix) {
    diag(confusion_matrix) / colSums(confusion_matrix)
}

recall <- function(confusion_matrix) {
    diag(confusion_matrix) / rowSums(confusion_matrix)
}

accuracy_score <- function(confusion_matrix) {
    sum(diag(confusion_matrix)) / sum(confusion_matrix)
}

ft.test <- function(model, test_file, k=1L) {
    stopifnot(is.character(test_file), file.exists(test_file))
    stopifnot( inherits(model, "fasttext") )
    x <- Rft_test(model$pointer, test_file, as.integer(k))
    names(x) <- c("precision", "recall", "nexamples", "number_of_correctly_predicted")
    return( x[1:3] )
}

# -----------------------------------------------------------
#  predict
#  =======
#' @title Predict using a Previously Trained Model
#' @description Predict values based on a previously trained model.
#' @param object an object inheriting from \code{'fasttext'}.
#' @param newdata a character vector giving the new data.
#' @param newdata_file a character string giving the location of to the new data.
#' @param result_file a character string naming a file.
#' @param k an integer giving the number of labels to be returned.
#' @param prob a logical if true the probabilities are also returned.
#' @param ... currently not used. 
#' @return \code{NULL} if a \code{'result_file'} is given otherwise 
#'         if \code{'prob'} is true a \code{data.frame} with the predicted labels 
#'         and the corresponding probabilities, if \code{'prob'} is false a 
#'         character vector with the predicted labels.
#' @examples
#' \dontrun{
#' predict(object, newdata)
#' }
#' @name predict.supervised_model
#' @rdname predict.supervised_model
predict.supervised_model <- function(object, newdata = character(), newdata_file = "", 
                                     result_file = "", k = 1L, prob = FALSE, ...) {
    stopifnot( inherits(object, "fasttext") )
    if ( missing(newdata) ) {
        stopifnot(is.character(newdata_file), is.character(result_file), 
                  file.exists(newdata))
        if ( missing(result_file) ) {
            pred <- Rft_predict(object$pointer, newdata_file, as.integer(k), 
                                as.logical(prob))
        } else {
            pred <- Rft_predict_to_file(object$pointer, newdata_file, result_file, 
                                        as.integer(k), as.logical(prob))
            return( NULL )
        }
    } else {
        pred <- Rft_vec_predict(object$pointer, newdata, as.integer(k), as.logical(prob))
    }
    if ( !prob ) {
        if ( k > 1 ) {
            pred <- matrix(pred[[1]], ncol=k, byrow=TRUE)
            colnames(pred) <- sprintf("best_%s", seq_len(k))
            return( pred )
        }
        return( pred[[1]] )
    }
    if ( k > 1 ) {
        pred[[1]] <- matrix(pred[[1]], ncol=k, byrow=TRUE)
        pred[[2]] <- matrix(pred[[2]], ncol=k, byrow=TRUE)
        cn <- sprintf("best_%s", seq_len(k))
        colnames(pred[[1]]) <- cn
        colnames(pred[[2]]) <- cn
    }
    return( pred )
}

# -----------------------------------------------------------
#  save.fasttext
#  =============
#' @title Save Model
#' @description Save the model to a file.
#' @param model an object inheriting from \code{"fasttext"}.
#' @param file a character string giving the name of the file.
#' @examples
#' \dontrun{
#' save.fasttext(model = m, file = "data.model")
#' }
save.fasttext <- function(model, file) {
    stopifnot( is.character(file), inherits(model, "fasttext") )
    Rft_save_model(model$pointer, file)
    return(invisible(NULL))
}

# -----------------------------------------------------------
#  read.fasttext
#  =============
#' @title Read Model
#' @description Read a previously saved model from file.
#' @param file a character string giving the name of the file
#'             to be read in.
#' @return an object inheriting from \code{"fasttext"}.
#' @examples
#' \dontrun{
#' model <- read.fasttext( "dbpedia.bin" )
#' }
read.fasttext <- function(file) {
    if ( !file.exists(file) ) {
        stop("cannot open file '", file, "': No such file or directory")
    }
    env <- new.env(parent = emptyenv())
    env$pointer <- try(Rft_load_model(file), silent = TRUE)
    if ( inherits(env$pointer, "try-error") ) {
        stop("version doesn't fit. The model was created by a different version than 'fastTextR' uses.")
    }
    .wrap_model( env )
}

# -----------------------------------------------------------
#  get_words
#  =========
#' @title Get Words
#' @description Obtain all the words from a previously trained model.
#' @param model an object inheriting from \code{"fasttext"}.
#' @return a character vector.
#' @examples
#' \dontrun{
#' get_words(model)
#' }
get_words <- function(model) {
    stopifnot( inherits(model, "fasttext") )
    Rft_dict_get_all_words(Rft_ft_get_dict(model$pointer))
}

# -----------------------------------------------------------
#  get_word_vectors
#  ================
#' @title Get Word Vectors
#' @description Obtain word vectors from a previously trained model.
#' @param model an object inheriting from \code{"fasttext"}.
#' @param words a character vector giving the words.
#' @return a matrix containing the word vectors.
#' @examples
#' \dontrun{
#' get_word_vectors(model, c("word", "vector"))
#' }
get_word_vectors <- function(model, words) {
    stopifnot(is.character(words), all(nchar(words) > 0))
    stopifnot( inherits(model, "fasttext") )
    word_vec <- Rft_get_word_vectors(model$pointer, words)
    word_vec <- matrix(word_vec, nrow=length(words), byrow=TRUE)
    rownames(word_vec) <- words
    word_vec
}

# -----------------------------------------------------------
#  ft.word_vector
#  ==============
## get_word_vectors <- function(model, words=NULL) {
##     Rft_get_all_word_vectors(model$pointer)
## }

.wrap_model <- function(env) {
    env$model_type <- Rft_ft_get_model_type( env$pointer )
    env$dict <- Rft_ft_get_dict( env$pointer )
    env$nwords <- Rft_dict_get_nwords( env$dict )
    env$ntoken <- Rft_dict_get_ntokens( env$dict )
    env$nlabels <- Rft_dict_get_nlabels( env$dict )
    class(env) <- c(sprintf("%s_model", env$model_type), "fasttext")
    env
}

print.cbow_model <- function(x, ...) {
    cat("fastText", shQuote(x$model_type), "model:", "\n")
    cat(sprintf("    %s tokens, %s words", 
        format(x$ntoken, big.mark=","),
        format(x$nwords, big.mark=",")), "\n")
}

print.skipgram_model <- function(x, ...) {
    cat("fastText", shQuote(x$model_type), "model:", "\n")
    cat(sprintf("    %s tokens, %s words", 
        format(x$ntoken, big.mark=","),
        format(x$nwords, big.mark=",")), "\n")
}

print.supervised_model <- function(x, ...) {
    cat("fastText", shQuote(x$model_type), "model:", "\n")
    cat(sprintf("    %s tokens, %s words, %s labels", 
        format(x$ntoken, big.mark=","),
        format(x$nwords, big.mark=","),
        format(x$nlabels, big.mark=",")), "\n")
}

# -----------------------------------------------------------
#  normalize
#  =========
#' @title Normalize
#' @description Applies normalization to a given text.
#' @param txt a character vector to be normalized.
#' @return a character vector.
#' @examples
#' \dontrun{
#' normalize(some_text)
#' }
normalize <- function(txt) {
    clean_text(txt)
}
