import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class AttrSelectedClassifier {
    private List<Instances> datasets = new ArrayList<>();
    private boolean crossValid = false;
    private String startDate;
    private String endDate;

    private AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();

    private J48 j48;
    private RandomForest randomForest;
    private NaiveBayes naiveBayes;
    private CfsSubsetEval cfs_Eval;
    private InfoGainAttributeEval InfoGain_Eval;
    private PrincipalComponents pca_Eval;
    private BestFirst BestF_Search;
    private GreedyStepwise greedyStepWise_Search;
    private Ranker ranker_Search;

    private Evaluation evaluation;

    Timer timer = new Timer();


    public AttrSelectedClassifier(List<Instances> datasets, boolean crossValid, String startDate, String endDate) {
        this.crossValid = crossValid;
        this.datasets.addAll(datasets);
        if (crossValid == false && (datasets.size() < 2)) {
            System.err.println("Both Training and Test sets must be given, but only one was");
            System.exit(1);
        }
        this.startDate = startDate;
        this.endDate = endDate;
    }

    public Result start(String attrEvalName, String AttrEvalOptions, String AttrSearchOptions, String classifierName, String classifierOptions) throws Exception{
        for(Instances dataset : datasets)
            dataset.setClassIndex(0);

        classifier = new AttributeSelectedClassifier();

        switch (classifierName) {
            case "J48":
                j48(classifierOptions);
                break;
            case "RANDOM_FOREST":
                randomForest(classifierOptions);
                break;
            case "NAIVE_BAYES":
                naiveBayes(classifierOptions);
                break;
            default:
                System.err.println("Error: Utility.Classifier String Parameter is not correctly defined");
                System.exit(1);
        }

        switch (attrEvalName) {
            case "CFS_BESTFIRST":
                cfs(AttrEvalOptions);
                bestFirst(AttrSearchOptions);
                break;
            case "CFS_GREEDYSTEPWISE":
                cfs(AttrEvalOptions);
                greedyStepWise(AttrSearchOptions);
                break;
            case "INFOGAIN_RANKER":
                infoGain(AttrEvalOptions);
                ranker(AttrSearchOptions);
                break;
            default:
                System.err.println("Error: Attribute Selection Evaluation String Parameter is not correctly defined");
                System.exit(1);
        }
        evaluation = new Evaluation(datasets.get(0));

        timer.startTimer();
        /** ======= Performing Cross Validation =======+*/
        if(crossValid == true){
            evaluation.crossValidateModel(classifier, datasets.get(0), 10, new Random(System.currentTimeMillis()));
        }else{
            /**======= Performing Training/Test Validation =======+*/

            classifier.buildClassifier(datasets.get(0));
            evaluation.evaluateModel(classifier,datasets.get(1));
        }
        /*
        classifier.setClassifier(j48);
        classifier.setEvaluator(cfs_Eval);
        classifier.setSearch(BestF_Search);
        evaluation.evaluateModel(classifier,datasets.get(1));
        */
        timer.stopTimer();


        ///****/
        //System.out.println(evaluation.toSummaryString("Results:\n", false));
        //System.out.println(evaluation.toClassDetailsString());
        //System.out.println(evaluation.toMatrixString());
        //System.out.println(evaluation.pctCorrect());
        ///****/

        return evalResult(evaluation, classifierName, attrEvalName, timer.getTime());
    }



    /***************************** Attribute Evaluation Methods *****************************/


    private void cfs(String optionsEval) throws Exception {
        cfs_Eval = new CfsSubsetEval();
        if (optionsEval != null)
            cfs_Eval.setOptions(weka.core.Utils.splitOptions(optionsEval));
        classifier.setEvaluator(cfs_Eval);
    }

    private void bestFirst(String optionsSearch) throws Exception {
        BestF_Search = new BestFirst();
        if (optionsSearch != null)
            BestF_Search.setOptions(weka.core.Utils.splitOptions(optionsSearch));
        classifier.setSearch(BestF_Search);
    }

    public void greedyStepWise(String optionsSearch) throws Exception {
        greedyStepWise_Search = new GreedyStepwise();
        if (optionsSearch != null)
            greedyStepWise_Search.setOptions(weka.core.Utils.splitOptions(optionsSearch));
        classifier.setSearch(greedyStepWise_Search);
    }

    private void infoGain(String optionsEval) throws Exception {
        InfoGain_Eval = new InfoGainAttributeEval();
        if (optionsEval != null)
            InfoGain_Eval.setOptions(weka.core.Utils.splitOptions(optionsEval));
        classifier.setEvaluator(InfoGain_Eval);
    }

    public void ranker(String optionsSearch) throws Exception {
        ranker_Search = new Ranker();
        if (optionsSearch != null)
            ranker_Search.setOptions(weka.core.Utils.splitOptions(optionsSearch));
        classifier.setSearch(ranker_Search);
    }

    private void pca(String optionsEval) throws Exception {
        pca_Eval = new PrincipalComponents();
        if (optionsEval != null) {
            String[] optEvalArray = weka.core.Utils.splitOptions(optionsEval);
            pca_Eval.setOptions(optEvalArray);
        }
        classifier.setEvaluator(pca_Eval);
    }
    /****************************************************************************************/

    /****************************** Classifiers Methods *************************************/

    private void j48(String classifierOptions) throws Exception {
        j48 = new J48();
        if(classifierOptions != null)
            j48.setOptions(weka.core.Utils.splitOptions(classifierOptions));
        classifier.setClassifier(j48);
    }

    private void randomForest(String classifierOptions) throws Exception {
        randomForest = new RandomForest();
        if(classifierOptions != null)
            randomForest.setOptions(weka.core.Utils.splitOptions(classifierOptions));
        classifier.setClassifier(randomForest);
    }

    private void naiveBayes(String classifierOptions) throws Exception {
        naiveBayes = new NaiveBayes();
        if(classifierOptions != null)
            naiveBayes.setOptions(weka.core.Utils.splitOptions(classifierOptions));
        classifier.setClassifier(naiveBayes);
    }

    /****************************************************************************************/

    /******************************* Result Evaluation **************************************/

    private Result evalResult(Evaluation eval, String classifierName, String attrSelName, String time) throws Exception{
        Result r = new Result();
        r.classifier = classifierName;
        r.attrSel = attrSelName;
        r.startDate = startDate;
        r.endDate = endDate;
        r.timeRequired = time;
        r.accuracy = eval.pctCorrect();
        r.totSamples = eval.numInstances();
        r.classSamples = eval.getClassPriors();
        for(int i=0; i<4; i++) {
            r.classTPR[i] = eval.truePositiveRate(i);
            r.classFPR[i] = eval.falsePositiveRate(i);
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
        return r;
    }
    /****************************************************************************************/
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
