import weka.attributeSelection.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
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

public class myClassifier {
    private Instances train;
    private Instances test;
    private String startDate;
    private String endDate;
    Timer timer = new Timer();

    public myClassifier() {}

    public myClassifier(Instances trainingSet, Instances testSet, String startDate, String endDate){
        train = trainingSet;
        test = testSet;
        this.startDate = startDate;
        this.endDate = endDate;
        setClass();
    }

    public void setClass(){
        train.setClassIndex(0);
        test.setClassIndex(0);
    }

    public Result j48(String options) throws Exception{
        setClass();
        J48 tree = new J48();
        if(options != null)
            tree.setOptions(weka.core.Utils.splitOptions(options));
        timer.startTimer();
        tree.buildClassifier(train);
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(tree,train);
        timer.stopTimer();
        ///****/
        //System.out.println(evaluation.toSummaryString("Results:\n", false));
        //System.out.println(evaluation.toClassDetailsString());
        //System.out.println(evaluation.toMatrixString());
        //System.out.println(evaluation.pctCorrect());
        ///****/
        return Visualizer.evalResult(evaluation,"J48","noAttrSel", timer.getTime(),startDate,endDate);
    }

    public Result randomForest(String options) throws Exception{
        RandomForest rForest = new RandomForest();
        if(options != null)
            rForest.setOptions(weka.core.Utils.splitOptions(options));
        timer.startTimer();
        rForest.buildClassifier(train);
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(rForest,test);
        timer.stopTimer();
        return Visualizer.evalResult(evaluation,"RANDOM_FOREST","noAttrSel", timer.getTime(),startDate,endDate);
    }

    public Result naiveBayes(String options) throws Exception{
        NaiveBayes nBayes = new NaiveBayes();
        if(options != null)
            nBayes.setOptions(weka.core.Utils.splitOptions(options));
        timer.startTimer();
        nBayes.buildClassifier(train);
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(nBayes,test);
        timer.stopTimer();
        return Visualizer.evalResult(evaluation,"NAIVE_BASYES","noAttrSel", timer.getTime(),startDate,endDate);
    }
}

