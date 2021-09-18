
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;


public class IncrClassifier {
    //private static List<IncrClassifier> incrClassifier = new ArrayList<>();
    private String startDate;
    private String endDate;
    private String classifierName;
    private boolean isFirstTime = true;
    private Timer timer = new Timer();

    private UpdateableClassifier updateableClassifier;
    private weka.classifiers.Classifier classifierInterface;
    private Evaluation evaluation;

    private NaiveBayesUpdateable nBayesUpdatable;
    private HoeffdingTree hoeffdingTree;


    public IncrClassifier(String classifierName, String classifierOptions){
        try{
        /**============== Instantiation of chosen Incremental Classifier ==============**/
        this.classifierName = classifierName;
        switch (classifierName) {
            case "NAIVE_BAYES_UPDATABLE":
                nBayesUpdatable = new NaiveBayesUpdateable();
                if (classifierOptions != null)
                    nBayesUpdatable.setOptions(weka.core.Utils.splitOptions(classifierOptions));
                updateableClassifier = nBayesUpdatable;
                classifierInterface = nBayesUpdatable;
                break;
/*
            case "HOEFFDING_TREE":
                hoeffdingTree = new HoeffdingTree();
                if (classifierOptions != null)
                    hoeffdingTree.setOptions(weka.core.Utils.splitOptions(classifierOptions));
                updateableClassifier = hoeffdingTree;
                classifierInterface = hoeffdingTree;
                break;
*/
            default:
                System.err.println("Error: Classifier String Parameter is not correctly defined");
                System.exit(1);
        }
        }catch(Exception e){
            System.err.println("Error: Exception thrown at IncrClassifier constructor");
            System.exit(1);
        }
    }
/*
    public static IncrClassifier getInstance(String classifierName, String classifierOptions) throws Exception{
        /**============== already built classifier: to be updated ==============**/
/*        for(IncrClassifier currentClassifier: incrClassifier){
            if(currentClassifier.classifierName == classifierName)
                return currentClassifier;
        }
        /**============== First definition ==============**/
/*        IncrClassifier newClassifier = new IncrClassifier(classifierName, classifierOptions);
        incrClassifier.add(newClassifier);
        return newClassifier;
    }*/

    public Result update(List<Instances> datasets, String startDate, String endDate) throws Exception {
        this.startDate = startDate;
        this.endDate = endDate;

        timer.startTimer();
        /**============== First Model Build ==============**/
        if (isFirstTime) {
            classifierInterface.buildClassifier(datasets.get(0));
            isFirstTime = false;
        } else {
            /**======= Performing Incremental Update of Model =======**/
            for (Instance sample : datasets.get(0))
                updateableClassifier.updateClassifier(sample);
        }
        evaluation = new Evaluation(datasets.get(0));
        evaluation.evaluateModel(classifierInterface, datasets.get(1));
        timer.stopTimer();

        return Visualizer.evalResult(evaluation, classifierName, null, timer.getTime(), startDate, endDate);
    }
}