# fastTextR

**fastTextR** is an **R** interface to the [fastText](https://github.com/facebookresearch/fastText)
library. It can be used to **word representation learning** *(Bojanowski et al., 2016)* and 
**supervised text classification** *(Joulin et al., 2016)*.
Particularly the advantage of **fastText** to other software is that, 
it was designed for biggish data.

The following examples show how to use **fastTextR** and are based on the
examples provided in the **fastText** library.

## Text Classification
### Download Data    
```{r}
fn <- "dbpedia_csv.tar.gz"

if ( !file.exists(fn) ) {
    download.file("https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz",
                  fn)
    untar(fn)
}
```

### Normalize Data
In **fastText** labels are typically marked with `__label__1` to `__label__k`.
Since **fastText** relies at the order of the trainings data it is important
to ensure the order of the trainings data follows no particular pattern
(which is done here with `sample`). The function `normalize` mimics
the data preparation steps of the bash function `normalize_text`
as shown in 
[classification-example.sh](https://github.com/facebookresearch/fastText/blob/master/classification-example.sh).

```{r}
library("fastText")

train <- sample(sprintf("__label__%s", readLines("dbpedia_csv/train.csv")))
head(train)

train <- normalize(train)
writeLines(train, con = "dbpedia.train")

test <- readLines("dbpedia_csv/test.csv")
test <- normalize(test)
labels <- gsub("\\D", "", substr(test, 1, 4))
test <- substr(test, 5, max(nchar(test)))
head(test)
head(labels)
```

### Train Model
After the data preparation the model can be trained and is saved to 
the file `"dbpedia.bin"`.
```{r}
cntrl <- ft.control(word_vec_size = 10L, learning_rate = 0.1, max_len_ngram = 2L, 
                    min_count = 1L, nbuckets = 10000000L, epoch = 5L, nthreads = 20L)

model <- fasttext(input = "dbpedia.train", method = "supervised", control = cntrl)
save.fasttext(model, "dbpedia")
```

### Read Model
A previously trained model can be loaded via the function `read.fasttext`.
```{r}
model <- read.fasttext( "dbpedia.bin" )
```

### Predict / Test Model
To perform prediction the function `predict` can be used.
```{r}
test.pred <- predict(model, test, k = 1L, prob = TRUE)
str(test.pred)
test.pred <- predict(model, test, k = 1L, prob = FALSE)
str(test.pred)

confusion_matrix <- table(labels, gsub("\\D", "", test.pred$label))
confusion_matrix

sum(diag(confusion_matrix)) / sum(confusion_matrix)
```

## Word Representation Learning
### Download Data
```{r}
fn <- "enwik9.zip"
if ( !file.exists(fn) ) {
    url <- "http://mattmahoney.net/dc/enwik9.zip"
    download.file(url, fn)
    unzip(fn)
}

fn <- "rw.zip"
if ( !file.exists(fn) ) {
    url <- "http://stanford.edu/~lmthang/morphoNLM/rw.zip"
    download.file(url, fn)
    unzip(fn)
}
```

### Prepare Data
```{r}
The function `clean_wiki` mimics the data preparation steps of the perl 
script `wikifil.pl` 
(`https://github.com/facebookresearch/fastText/blob/master/wikifil.pl`).

clean_wiki <- function(x) {
    stopifnot(is.character(x))
    x <- gsub("[[:cntrl:]]", " ", x)
    x <- gsub("<.*>", "", x, perl = TRUE)  ## remove xml tags
    x <- gsub("&amp", "&", x, perl = TRUE) ## decode URL encoded chars
    x <- gsub("&lt", "<", x, perl = TRUE)
    x <- gsub("&gt", ">", x, perl = TRUE)
    x <- gsub("<ref[^<]*<\\/ref>", "", x, perl = TRUE) ## remove references <ref...> ... </ref>
    x <- gsub("<[^>]*>", "", x, perl = TRUE)           ## remove xhtml tags
    x <- gsub("\\[http:[^] ]*", "[", x, perl = TRUE)   ## remove normal url, preserve visible text
    x <- gsub("\\|thumb", "", x, perl = TRUE) ## remove images links, preserve caption
    x <- gsub("\\|left", "", x, perl = TRUE)
    x <- gsub("\\|right", "", x, perl = TRUE)
    x <- gsub("\\|\\d+px", "", x, perl = TRUE)
    x <- gsub("\\[\\[image:[^\\[\\]]*\\|", "", x, perl = TRUE)
    x <- gsub("\\[\\[category:([^|\\]]*)[^]]*\\]\\]", "[[\\1]]", x, perl = TRUE) ## show categories without markup
    x <- gsub("\\[\\[[a-z\\-]*:[^\\]]*\\]\\]", "", x, perl = TRUE) ## remove links to other languages
    x <- gsub("\\[\\[[^\\|\\]]*\\|", "[[", x, perl = TRUE) ## remove wiki url, preserve visible text
    x <- gsub("\\{\\{[^\\}]*\\}\\}", "", x, perl = TRUE) ## remove {{icons}} and {tables}
    x <- gsub("\\{[^\\}]*\\}", "", x, perl = TRUE)
    x <- gsub("\\[", "", x, perl = TRUE) ## remove [ and ]
    x <- gsub("\\]", "", x, perl = TRUE)
    x <- gsub("&[^;]*;", " ", x, perl = TRUE) ## remove URL encoded chars

    # convert to lowercase letters and spaces, spell digits
    x <- tolower(x)
    x <- gsub("0", " zero ", x, perl = TRUE)
    x <- gsub("1", " one ", x, perl = TRUE)
    x <- gsub("2", " two ", x, perl = TRUE)
    x <- gsub("3", " three ", x, perl = TRUE)
    x <- gsub("4", " four ", x, perl = TRUE)
    x <- gsub("5", " five ", x, perl = TRUE)
    x <- gsub("6", " six ", x, perl = TRUE)
    x <- gsub("7", " seven ", x, perl = TRUE)
    x <- gsub("8", " eight ", x, perl = TRUE)
    x <- gsub("9", " nine ", x, perl = TRUE)

    x <- gsub("[[:punct:]]", " ", x)
    x
}

library(XML)
html <- htmlParse("enwik9", encoding = "UTF-8")
txt <- xpathSApply(html, "//text", xmlValue) 
txt <- grep("#redirect", txt, value = TRUE, ignore.case = TRUE, invert = TRUE)
txt <- clean_wiki_pearl(txt)
txt <- paste(txt, collapse = " ")
txt <- gsub("\\s+", " ", txt)
writeLines(txt, con = "fil9")
```

### Train Model
```{r}
cntrl <- ft.control(learning_rate = 0.025, word_vec_size = 5, epoch = 1, 
                    nthreads = 10L)

model <- fasttext("fil9", "skipgram", cntrl)
model

save.fasttext(model, "fil9_skipgram_model")
```

### Load Model
```{r}
model <- read.fasttext("fil9_skipgram_model.bin")
```

### Obtain Word Vectors
```{r}
queries <- readLines("rw/rw.txt")
queries <- unlist(lapply(strsplit(queries, "\\t"), head, 2))
queries <- tolower(queries)

word_vectors <- get_words(model, queries)
```

## References

[1] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

