## ----eval=FALSE---------------------------------------------------------------
#  library("fastTextR")

## ----eval=FALSE---------------------------------------------------------------
#  model <- ft_load("cc.en.300.bin")

## ----eval=FALSE---------------------------------------------------------------
#  ft_word_vectors(model, c("asparagus", "pidgey", "yellow"))[,1:5]

## ----eval=FALSE---------------------------------------------------------------
#  ft_sentence_vectors(model, c("Poets have been mysteriously silent on the subject of cheese", "Who did not let the gorilla into the ballet"))[,1:5]

## ----eval=FALSE---------------------------------------------------------------
#  ft_nearest_neighbors(model, 'asparagus', k = 5L)

## ----eval=FALSE---------------------------------------------------------------
#  ft_analogies(model, c("berlin", "germany", "france"))

