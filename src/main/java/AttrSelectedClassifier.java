import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
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
                System.err.println("Error: Classifier String Parameter is not correctly defined");
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

            // =====>Model Complexity extraction and printing to Console
            //       Problem: RandomForest and CrossValidation clean all results and model at the end of evaluation.
            /*
            if(classifier.getClassifier() instanceof J48) {
                System.out.println(j48.measureNumLeaves());
            }
            */
        }
        // ====> automatic computed time extraction:
        //       Problem: RandomForest and CrossValidation clean all results and model at the end of evaluation.
        //double time = classifier.measureTime();
        timer.stopTimer();


        ///****/
        //System.out.println(evaluation.toSummaryString("Results:\n", false));
        //System.out.println(evaluation.toClassDetailsString());
        //System.out.println(evaluation.toMatrixString());
        //System.out.println(evaluation.pctCorrect());
        ///****/

        return Visualizer.evalResult(evaluation, classifierName, attrEvalName,timer.getTime(), startDate, endDate);
    }



    /***************************** Attribute Evaluation Methods *****************************/

    /*
    ------------- CfsSubsetEval options
          -M        Treat missing values as a separate value.
          -L        Don't include locally predictive attributes.
          -Z        Precompute the full correlation matrix at the outset, rather than compute correlations lazily (as needed) during the search. Use this in conjuction with parallel processing in order to speed up a backward search.
          -P <int>  The size of the thread pool, for example, the number of cores in the CPU. (default 1)
          -E <int>  The number of threads to use, which should be >= size of thread pool. (default 1)
          -D        Output debugging info.
     */
    private void cfs(String optionsEval) throws Exception {
        cfs_Eval = new CfsSubsetEval();
        if (optionsEval != null)
            cfs_Eval.setOptions(weka.core.Utils.splitOptions(optionsEval));
        classifier.setEvaluator(cfs_Eval);
    }

    /*
    ------------- BestFirst
        -P <start set>     Specify a starting set of attributes. Eg. 1,3,5-7.
        -D <0 = backward | 1 = forward | 2 = bi-directional>       Direction of search. (default = 1).
        -N <num>    Number of non-improving nodes to consider before terminating search.
        -S <num>    Size of lookup cache for evaluated subsets. Expressed as a multiple of the number of attributes in the data set. (default = 1)

    */
    private void bestFirst(String optionsSearch) throws Exception {
        BestF_Search = new BestFirst();
        if (optionsSearch != null)
            BestF_Search.setOptions(weka.core.Utils.splitOptions(optionsSearch));
        classifier.setSearch(BestF_Search);
    }

    /*
    ------------- GreedyStepwise options
         -C                 Use conservative forward search
         -B                 Use a backward search instead of a forward one.
         -P <start set>     Specify a starting set of attributes. Eg. 1,3,5-7.
         -R                 Produce a ranked list of attributes.
         -T <threshold>     Specify a theshold by which attributes may be discarded from the ranking. Use in conjuction with -R
         -N <num to select> Specify number of attributes to select
         -num-slots <int>   The number of execution slots, for example, the number of cores in the CPU. (default 1)
         -D                 Print debugging output
    */
    public void greedyStepWise(String optionsSearch) throws Exception {
        greedyStepWise_Search = new GreedyStepwise();
        if (optionsSearch != null)
            greedyStepWise_Search.setOptions(weka.core.Utils.splitOptions(optionsSearch));
        classifier.setSearch(greedyStepWise_Search);
    }

    /*
    ------------- InfoGain options
        -M      treat missing values as a separate value.
        -B      just binarize numeric attributes instead of properly discretizing them.
    */
    private void infoGain(String optionsEval) throws Exception {
        InfoGain_Eval = new InfoGainAttributeEval();
        if (optionsEval != null)
            InfoGain_Eval.setOptions(weka.core.Utils.splitOptions(optionsEval));
        classifier.setEvaluator(InfoGain_Eval);
    }

    /*
    ------------- Ranker options
        -P <start set>  Specify a starting set of attributes. Eg. 1,3,5-7. Any starting attributes specified are ignored during the ranking.
        -T <threshold>  Specify a theshold by which attributes may be discarded from the ranking.
        -N <num to select>  Specify number of attributes to select

     */
    public void ranker(String optionsSearch) throws Exception {
        ranker_Search = new Ranker();
        if (optionsSearch != null)
            ranker_Search.setOptions(weka.core.Utils.splitOptions(optionsSearch));
        classifier.setSearch(ranker_Search);
    }

    /*
    ------------- Principal Components Analysis Options
        -C      Center (rather than standardize) the data and compute PCA using the covariance (rather than the correlation) matrix.
        -R      Retain enough PC attributes to account for this proportion of variance in the original data. (default = 0.95)
        -O      Transform through the PC space and back to the original space.
        -A      Maximum number of attributes to include in transformed attribute names. (-1 = include all)
     */
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
    */
    private void j48(String classifierOptions) throws Exception {
        j48 = new J48();
        if(classifierOptions != null)
            j48.setOptions(weka.core.Utils.splitOptions(classifierOptions));
        classifier.setClassifier(j48);
    }

    /*
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
     */
    private void randomForest(String classifierOptions) throws Exception {
        randomForest = new RandomForest();
        if(classifierOptions != null)
            randomForest.setOptions(weka.core.Utils.splitOptions(classifierOptions));
        classifier.setClassifier(randomForest);
    }

    /*
     ----------------- NaiveBayes Options
        -K     Use kernel density estimator rather than normal distribution for numeric attributes
        -D     Use supervised discretization to process numeric attributes
        -O     Display model in old format (good when there are many classes)
     */
    private void naiveBayes(String classifierOptions) throws Exception {
        naiveBayes = new NaiveBayes();
        if(classifierOptions != null)
            naiveBayes.setOptions(weka.core.Utils.splitOptions(classifierOptions));
        classifier.setClassifier(naiveBayes);
    }

    /****************************************************************************************/

}
