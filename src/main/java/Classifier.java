import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.ArrayList;
import java.util.List;


public class Classifier {
    private Instances train;
    private Instances test;
    private List<String> results;
    private String resultsFile;

    public Classifier(Instances trainingSet, Instances testSet, String file){
        train = trainingSet;
        test = testSet;
        results = new ArrayList<>();
        resultsFile = file;
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
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(tree,train);
        System.out.println(eval.toSummaryString("Results Training:\n", false));

        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(tree,test);
        System.out.println(evalTs.toSummaryString("Results Test:\n", false));

        System.out.println(evalTs.toMatrixString());
        System.out.println(evalTs.pctCorrect());
    }

    private void comparingResults(){

    }
}
