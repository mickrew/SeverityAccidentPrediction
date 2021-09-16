import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

public class Classifier {
    private Instances train;
    private Instances test;
    private String attrSel; // Attribute Selection applied on Training & Test Sets
    private String startDate;
    private String endDate;
    private ArrayList<Result> results;

    public Classifier(){
        results = new ArrayList<>();
    }
    public Classifier(Instances trainingSet, Instances testSet, String attrSelName, String startDate, String endDate){
        results = new ArrayList<>();
        updateClassifier(trainingSet, testSet, attrSelName, startDate, endDate);
    }

    public void updateClassifier(Instances trainingSet, Instances testSet, String attrSelName, String startDate, String endDate){
        train = trainingSet;
        test = testSet;
        attrSel = attrSelName;
        this.startDate = startDate;
        this.endDate = endDate;
    }

    public void setClass(){
        train.setClassIndex(0);
        test.setClassIndex(0);
    }

    public ArrayList<Result> getResults(){ return results;}
    
    public void j48(String options) throws Exception{
        //setClass();
        //building
        J48 tree = new J48();
        if(options != null)
            tree.setOptions(weka.core.Utils.splitOptions(options));
        tree.buildClassifier(train);
        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(tree,test);
        /****/
        System.out.println(evalTs.toSummaryString("Results Test:\n", false));
        System.out.println(evalTs.toMatrixString());
        System.out.println(evalTs.pctCorrect());
        /****/
        addEvalResults(evalTs, "J48");
    }
    
    public void randomForest(String options) throws Exception{
        //setClass();
        //building
        RandomForest rForest = new RandomForest();
        if(options != null)
            rForest.setOptions(weka.core.Utils.splitOptions(options));
        rForest.buildClassifier(train);
        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(rForest,test);
        /****/
        System.out.println(evalTs.toSummaryString("Results Test:\n", false));
        System.out.println(evalTs.toMatrixString());
        System.out.println(evalTs.pctCorrect());
        /****/
        addEvalResults(evalTs, "RandomForest");
    }

    public void naiveBayes(String options) throws Exception{
        //setClass();
        //building
        NaiveBayes nBayes = new NaiveBayes();
        if(options != null)
            nBayes.setOptions(weka.core.Utils.splitOptions(options));
        nBayes.buildClassifier(train);
        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(nBayes,test);
        /****/
        System.out.println(evalTs.toSummaryString("Results Test:\n", false));
        System.out.println(evalTs.toMatrixString());
        System.out.println(evalTs.pctCorrect());
        /****/
        addEvalResults(evalTs, "NaiveBayes");
    }

    private void addEvalResults(Evaluation eval, String classifier) throws Exception{
        Result r = new Result(classifier);
        r.attrSel = attrSel;
        r.startDate = startDate;
        r.endDate = endDate;
        r.classSamples = eval.getClassPriors();
        for(int i=0; i<4; i++) {
            r.classTPR[i] = eval.truePositiveRate(i );
            r.classTPR[i] = eval.falsePositiveRate(i);
            r.precision[i] = eval.precision(i);
            r.recall[i] = eval.recall(i);
            r.fMeasure[i] = eval.fMeasure(i);
        }
        r.weightedTPR = eval.weightedTruePositiveRate();
        r.weightedFPR = eval.weightedFalsePositiveRate();
        r.weightedPrecision = eval.weightedPrecision();
        r.weightedRecall = eval.weightedRecall();
        r.weightedFMeasure = eval.weightedFMeasure();
        r.summaryEval = eval.toSummaryString();
        r.confusionMatrix = eval.toMatrixString();
        results.add(r);
    }
}

/*
 ----------------- J48 Options
        -U          Use unpruned tree.
        -O          Do not collapse tree.
        -C <pruning confidence>     Set confidence threshold for pruning. (default 0.25)
        -M <minimum number of instances>     Set minimum number of instances per leaf. (default 2)
        -R          Use reduced error pruning.
        -N <number of folds>     Set number of folds for reduced error pruning. One fold is used as pruning set. (default 3)
        -B          Use binary splits only.
        -S          Don't perform subtree raising.
        -L          Do not clean up after the tree has been built.
        -A          Laplace smoothing for predicted probabilities.
        -J          Do not use MDL correction for info gain on numeric attributes.
        -Q <seed>   Seed for random data shuffling (default 1).
        -doNotMakeSplitPointActualValue    Do not make split point actual value.

 ----------------- RandomForest Options
        -P          Size of each bag, as a percentage of the training set size. (default 100)
        -O          Calculate the out of bag error.
        -store-out-of-bag-predictions                Whether to store out of bag predictions in internal evaluation object.
        -output-out-of-bag-complexity-statistics     Whether to output complexity-based statistics when out-of-bag evaluation is performed.
        -print      Print the individual classifiers in the output
        -attribute-importance       Compute and output attribute importance (mean impurity decrease method)
        -I <num>    Number of iterations (i.e., the number of trees in the random forest).(current value 100)
        -num-slots <num>            Number of execution slots.(default 1 - i.e. no parallelism)(use 0 to auto-detect number of cores)
        -K <number of attributes>   Number of attributes to randomly investigate. (default 0)(<1 = int(log_2(#predictors)+1)).
        -M <minimum number of instances>    Set minimum number of instances per leaf.(default 1)
        -V <minimum variance for split>     Set minimum numeric class variance proportion of train variance for split (default 1e-3).
        -S <num>        Seed for random number generator.(default 1)
        -depth <num>    The maximum depth of the tree, 0 for unlimited.(default 0)
        -N <num>        Number of folds for backfitting (default 0, no backfitting).
        -U              Allow unclassified instances.
        -B              Break ties randomly when several attributes look equally good.
        -output-debug-info          If set, classifier is run in debug mode and may output additional info to the console
        -do-not-check-capabilities  If set, classifier capabilities are not checked before classifier is built (use with caution).
        -num-decimal-places    The number of decimal places for the output of numbers in the model (default 2).
        -batch-size    The desired batch size for batch prediction  (default 100).

 ----------------- NaiveBayes Options
        -K     Use kernel density estimator rather than normal distribution for numeric attributes
        -D     Use supervised discretization to process numeric attributes
        -O     Display model in old format (good when there are many classes)
*/
