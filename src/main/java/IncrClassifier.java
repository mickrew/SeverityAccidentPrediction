
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;


public class IncrClassifier {
    private Instances newDataset;
    private String startDate;
    private String endDate;

    private UpdateableClassifier updateableClassifier;
    private weka.classifiers.Classifier classifierInterface;
    private Evaluation evaluation;

    private NaiveBayesUpdateable nBayesUpdatable;
    //private IBk ibk;
    private HoeffdingTree hoeffdingTree;


    public IncrClassifier(Instances datasetStructure, String classifierName, String classifierOptions) throws Exception {
        /**============== Instantiation of chosen Incremental Classifier ==============**/
        switch (classifierName) {
            case "NAIVE_BAYES_UPDATABLE":
                nBayesUpdatable = new NaiveBayesUpdateable();
                if (classifierOptions != null)
                    nBayesUpdatable.setOptions(weka.core.Utils.splitOptions(classifierOptions));
                updateableClassifier = nBayesUpdatable;
                classifierInterface = nBayesUpdatable;
                break;

            case "HOEFFDING_TREE":
                hoeffdingTree = new HoeffdingTree();
                if (classifierOptions != null)
                    hoeffdingTree.setOptions(weka.core.Utils.splitOptions(classifierOptions));
                updateableClassifier = hoeffdingTree;
                classifierInterface = hoeffdingTree;
                break;
            default:
                System.err.println("Error: Classifier String Parameter is not correctly defined");
                System.exit(1);
        }
        /**============== Input header definition ==============**/
        if (datasetStructure.size() > 1) {
            System.err.println("Error: datasetStructure should be an empty set");
            System.exit(1);
            //datasetStructure.
        }

    }

    public void update(Instances dataset, String startDate, String endDate) throws Exception {
        newDataset = dataset;
        this.startDate = startDate;
        this.endDate = endDate;
        newDataset.setClassIndex(0);

        //timer.startTimer();
        /**======= Performing Incremental Update of Model =======**/
        for (Instance sample : dataset)
            updateableClassifier.updateClassifier(sample);
/*
            classifier.buildClassifier(datasets.get(0));
            evaluation.evaluateModel(classifier,datasets.get(1));


        double time = classifier.measureTime();
        //timer.stopTimer();

*/
        ///****/
        //System.out.println(evaluation.toSummaryString("Results:\n", false));
        //System.out.println(evaluation.toClassDetailsString());
        //System.out.println(evaluation.toMatrixString());
        //System.out.println(evaluation.pctCorrect());
        ///****/

/*
        // load data
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(args[0]));
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes() - 1);

        // train NaiveBayes
        NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
        nb.buildClassifier(structure);
        Instance current;
        while ((current = loader.getNextInstance(structure)) != null)
            nb.updateClassifier(current);

        // output generated model
        System.out.println(nb);
*/
    }
}
