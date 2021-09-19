import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import weka.classifiers.Evaluation;

public class Visualizer{

    private ArrayList<Result> results = new ArrayList<>();
    private FileWriter fileWriter;
    private FileWriter fileWriterIncr;
    private PrintWriter printWriter;
    private PrintWriter printWriterIncr;
    private String outputFile;
    private String incrOutputFile;


    public Visualizer(String outputFile) throws IOException {
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

    public void addResult(Result r) throws IOException{
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
        printWriter1.printf("%-12s%-12s%-16s%-20s%-20s%-20s\n", r.startDate,r.endDate,r.classifier, r.attrSel,"Accuracy: "+formatter.format(r.accuracy)+"%", "ClassifierTime: "+r.timeRequired + "s");
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
            printWriter2.printf("%-12s%-12s%-16s%-20s%-10s%-+7.3f%s  %-20s\n", newR.startDate,newR.endDate,newR.classifier, newR.attrSel,"Accuracy: ",(newR.accuracy-oldR.accuracy),"%", "ClassifierTime: "+newR.timeRequired +"s");
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
    /******************************* Evaluation Result Extraction **************************************/
    public static Result evalResult(Evaluation eval, String classifierName, String attrSelName, String time, String startDate, String endDate) throws Exception{
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
        /****************************************************************************************/
    }
}