library(mlbench)
library(gridExtra)
library(ggplot2)
library(randomForest)
library(randomForestSRC)
library(mlr)

newdata = as.data.frame(mlbench.spirals(n=200))
blobs = ggplot(data=newdata, aes(x=x.1, y=x.2, color=classes)) + geom_point() + coord_fixed(ratio=1)
grid.arrange(blobs, ncol=1, nrow =1)

dev.new()
classif_forest = makeLearner("classif.randomForest", ntree=128)
task = makeClassifTask(id="dataML",data=newdata,target="classes")
plotLearnerPrediction(classif_forest, task, features = c("x.1", "x.2"))
#