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
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

public class Classifier {
    private Instances train;
    private Instances test;
    private String attrSel; // Attribute Selection applied on Training & Test Sets
    private String startDate;
    private String endDate;
    private ArrayList<Result> results = new ArrayList<>();;
    Timer timer = new Timer();
    private FileWriter fileWriter;
    private FileWriter fileWriterIncr;
    private PrintWriter printWriter;
    private PrintWriter printWriterIncr;
    private String outputFile;
    private String incrOutputFile;


    public Classifier(String outputFile) throws IOException{
        this.outputFile = outputFile;
        incrOutputFile = "Incremental"+outputFile;
        File f = new File(outputFile);
        if(f.exists()) {
            f.delete();
        }
        f = new File(incrOutputFile);
        if(f.exists()) {
            f.delete();
        }
    }

    /*
    public Classifier(Instances trainingSet, Instances testSet, String attrSelName, String startDate, String endDate, String outputFile){
        this(outputFile);
        updateClassifier(trainingSet, testSet, attrSelName, startDate, endDate);
    }
    */

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
        timer.startTimer();
        tree.buildClassifier(train);
        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(tree,test);
        timer.stopTimer();
        ///****/
        //System.out.println(evalTs.toSummaryString("Results Test:\n", false));
        //System.out.println(evalTs.toMatrixString());
        //System.out.println(evalTs.pctCorrect());
        ///****/
        addEvalResults(evalTs, "J48", timer.getTime());
    }

    public void randomForest(String options) throws Exception{
        //setClass();
        //building
        RandomForest rForest = new RandomForest();
        if(options != null)
            rForest.setOptions(weka.core.Utils.splitOptions(options));
        timer.startTimer();
        rForest.buildClassifier(train);
        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(rForest,test);
        timer.stopTimer();
        ///****/
        //System.out.println(evalTs.toSummaryString("Results Test:\n", false));
        //System.out.println(evalTs.toMatrixString());
        //System.out.println(evalTs.pctCorrect());
        ///****/
        addEvalResults(evalTs, "RandomForest", timer.getTime());
    }

    public void naiveBayes(String options) throws Exception{
        //setClass();
        //building
        NaiveBayes nBayes = new NaiveBayes();
        if(options != null)
            nBayes.setOptions(weka.core.Utils.splitOptions(options));
        timer.startTimer();
        nBayes.buildClassifier(train);
        // Evaluation: test set
        Evaluation evalTs = new Evaluation(train);
        evalTs.evaluateModel(nBayes,test);
        timer.stopTimer();
        ///****/
        //System.out.println(evalTs.toSummaryString("Results Test:\n", false));
        //System.out.println(evalTs.toMatrixString());
        //System.out.println(evalTs.pctCorrect());
        ///****/
        addEvalResults(evalTs, "NaiveBayes", timer.getTime());
    }

    private void addEvalResults(Evaluation eval, String classifier, String time) throws Exception{
        Result r = new Result(classifier);
        r.attrSel = attrSel;
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
        results.add(r);
        printResult(r);
    }

    public void printResult(Result r) throws IOException {
        fileWriter = new FileWriter(outputFile,true);
        printWriter = new PrintWriter(fileWriter);
        fileWriterIncr = new FileWriter(incrOutputFile,true);
        printWriterIncr = new PrintWriter(fileWriterIncr);

        printSingleResult(printWriter, r);
        printIncrementalResult(printWriterIncr,r);

        fileWriter.close();
        printWriter.close();
        fileWriterIncr.close();
        printWriterIncr.close();
    }

    private void printSingleResult(PrintWriter printWriter1, Result r){
        NumberFormat formatter = new DecimalFormat("#.###");

        printWriter1.printf("-----------------------------------------------------------------------------------------\n");
        printWriter1.printf("%-12s%-12s%-16s%-20s%-20s%-20s\n", r.startDate,r.endDate,r.classifier, r.attrSel,"Accuracy: "+formatter.format(r.accuracy), "ClassifierTime:"+r.timeRequired);
        printWriter1.printf("%-60s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n","", "Per Class:","#Samples", "TPR", "FPR", "Precision", "Recall","F-measure");
        for(int i=0; i<4; i++)
            printWriter1.printf("%-60s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Sev" + (i + 1) + ":", r.classSamples[i], formatter.format(r.classTPR[i]), formatter.format(r.classFPR[i]), formatter.format(r.precision[i]), formatter.format(r.recall[i]), formatter.format(r.fMeasure[i]));

        printWriter1.printf("%-60s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Weighted:",r.totSamples,formatter.format(r.weightedTPR), formatter.format(r.weightedFPR), formatter.format(r.weightedPrecision), formatter.format(r.weightedRecall), formatter.format(r.weightedFMeasure));
    }

    private void printIncrementalResult(PrintWriter printWriter2,Result newR){
        int index = results.size();
        Result oldR = new Result();
        boolean found = false;  // found result of previous time windows computed by same classifier
        if(index >= 2) {
            index -= 2; // index =index -1-1;
                        // since indexing counting starts from 1 but .get() starts counting from 0
                        // and last index corresponds to current results to be compared with previous ones
            while (index >= 0) {
                oldR = results.get(index);
                if (oldR.classifier == newR.classifier && oldR.attrSel == newR.attrSel) {
                    found = true;
                    break;
                }
                index--;
            }
        }
        if(found){
            NumberFormat formatter = new DecimalFormat("#.###");
            printWriter2.printf("-----------------------------------------------------------------------------------------\n");
            printWriter2.printf("%-12s%-12s%-16s%-20s%-10s%-+10.3f%-20s\n", newR.startDate,newR.endDate,newR.classifier, newR.attrSel,"Accuracy: ",newR.accuracy-oldR.accuracy, "ClassifierTime:"+newR.timeRequired);
            printWriter2.printf("%-60s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n","", "Per Class:","#Samples", "TPR", "FPR", "Precision", "Recall","F-measure");
            double newSum=0, oldSum=0;
            for(int i=0; i<4; i++) {
                printWriter2.printf("%-60s%-12s%-+10.0f%-+10.3f%-+10.3f%-+10.3f%-+10.3f%-+10.3f\n", "", "Sev" + (i + 1) + ":", newR.classSamples[i]-oldR.classSamples[i], newR.classTPR[i]-oldR.classTPR[i], newR.classFPR[i]-oldR.classFPR[i], newR.precision[i]-oldR.precision[i], newR.recall[i]-oldR.recall[i], newR.fMeasure[i]-oldR.fMeasure[i]);
                newSum += newR.classSamples[i];
                oldSum += oldR.classSamples[i];
            }
            printWriter2.printf("%-60s%-12s%-+10.0f%-+10.3f%-+10.3f%-+10.3f%-+10.3f%-+10.3f\n", "", "Weighted:", newSum-oldSum,newR.weightedTPR-oldR.weightedTPR, newR.weightedFPR-oldR.weightedFPR, newR.weightedPrecision-oldR.weightedPrecision, newR.weightedRecall-oldR.weightedRecall, newR.weightedFMeasure-oldR.weightedFMeasure);
        }
        else{
            printSingleResult(printWriterIncr, newR);
        }
    }

    private String addPlus(double value){
        if(value >= 0)
            return "+"+value;
        return ""+value;
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
