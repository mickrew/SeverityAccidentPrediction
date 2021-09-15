import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.ArrayList;
import java.util.List;


class Result{
    String classifier;
    String attrSel;
    Double accuracy;
    Double[] classTPR;
    Double[] classFPR;
    Double[] precision;
    Double[] recall;
    Double[] fMeasure;
    Double weightedTPR;
    Double weightedFPR;
    Double weightedPrecision;
    Double weightedRecall;
    Double weightedFMeasure;

    public Result(String classifierName, String attrSelName){
        classifier = classifierName;
        attrSel = attrSelName;
        classTPR = new Double[4];
        classFPR = new Double[4];
        precision = new Double[4];
        recall = new Double[4];
        fMeasure = new Double[4];
    }
}

public class Classifier {
    private Instances train;
    private Instances test;
    private ArrayList<Result> results;
    private String attrSel;

    public Classifier(Instances trainingSet, Instances testSet, String attrSelName){
        train = trainingSet;
        test = testSet;
        results = new ArrayList<>();
        attrSel = attrSelName;
    }



    public void J48(String[] options) throws Exception{
        //building
        if(options == null) {
            options = new String[1];
            options[0] = "-U";
        }
        J48 tree = new J48();
        tree.setOptions(options);
        tree.buildClassifier(train);

        // Evaluation: training Set
        //Evaluation eval = new Evaluation(train);
        //eval.evaluateModel(tree,train);
        //String resultEvalTrain = eval.toSummaryString("Results Training:\n", false);

        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(tree,test);
        printEvalResults(evalTs);

        String resultEvaltest = evalTs.toSummaryString("Results Test:\n", false);
        System.out.println(evalTs.toMatrixString());
        System.out.println(evalTs.pctCorrect());
    }

    private void printEvalResults(Evaluation eval, String Classifier){
        Result r = new Result();
        for(int i=0; i<4; i++) {
            classTPR[i] = eval.truePositiveRate(i + 1);
            classTPR[i] = eval.falsePositiveRate(i + 1);
            precision[i] = eval.precision(i+1);
            recall[i] = eval.recall(i+1);
            fMeasure[i] = eval.fMeasure(i+1);
        }


    }

    private void comparingResults(){

    }
}
