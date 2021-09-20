
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class IncrClassifier {

    private String startDate;
    private String endDate;
    private String classifierName;
    private boolean isFirstTime = true;
    private Timer timer = new Timer();
    private Instances structure;
    private NaiveBayesUpdateable nb;
    private HoeffdingTree ht;
    private Evaluation evaluation;
    private UpdateableClassifier updateableClassifier;

    public IncrClassifier(){}

    public void buildIncrClassifier(String classifierName, String classifierOptions) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("TrainSetFiltered.arff"));
        structure = loader.getStructure();
        structure.setClassIndex(0);


        /**============== Instantiation of chosen Incremental Classifier ==============**/
        this.classifierName = classifierName;
        switch (classifierName) {
            case "NAIVE_BAYES_UPDATABLE":
                nb = new NaiveBayesUpdateable();
                updateableClassifier = nb;
                nb.buildClassifier(structure);
                break;
            case "HOEFFDING_TREE":
                ht = new HoeffdingTree();
                ht.buildClassifier(structure);
                updateableClassifier = ht;
                break;
            default:
                System.err.println("Error: Classifier String Parameter is not correctly defined");
                System.exit(1);
        }
    }

    public List<Result> update(String startDate, String endDate) throws Exception {
        this.startDate = startDate;
        this.endDate = endDate;

        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("TrainSetFiltered.arff"));
        structure = loader.getStructure();
        structure.setClassIndex(0);

        Instance current;
        int count = 0;
        while ((current = loader.getNextInstance(structure)) != null) {
            try {
                nb.updateClassifier(current);
                ht.updateClassifier(current);
            }catch(IndexOutOfBoundsException iobex){

            }
        }
        loader = new ArffLoader();
        loader.setFile(new File("TestSetFiltered.arff"));
        Instances instancesTrain = loader.getDataSet();
        instancesTrain.setClassIndex(0);

        List<Result> resultList = new ArrayList<>();

        evaluation = new Evaluation(instancesTrain);
        evaluation.crossValidateModel(nb, instancesTrain, 10, new Debug.Random(1));
        resultList.add(Visualizer.evalResult(evaluation, "NAIVE_BAYES_UPDATABLE", null, timer.getTime(), startDate, endDate));

        evaluation.crossValidateModel(ht, instancesTrain, 10, new Debug.Random(1));
        resultList.add(Visualizer.evalResult(evaluation, "HOEFFDING_TREE", null, timer.getTime(), startDate, endDate));

        timer.stopTimer();

        return resultList;
    }
}