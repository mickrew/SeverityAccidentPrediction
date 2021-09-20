import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import weka.classifiers.Evaluation;

public class Visualizer{

    private ArrayList<Result> results = new ArrayList<>();
    private FileWriter fileWriter;
    private FileWriter fileWriterIncr;
    private PrintWriter printWriter;
    private PrintWriter printWriterIncr;
    private String incrOutputFile;
    private boolean isFirstTimeAcc=true;
    private boolean isFirstTimeResults=true;

    public Visualizer() {}

    public void addResult(String outputFile, Result r) throws IOException{
        results.add(r);
        printResult(outputFile, r);
    }

    public void printResult(String outputFile, Result r) throws IOException {
        if(isFirstTimeResults) {
            incrOutputFile = "Incremental" + outputFile;
            File f = new File(outputFile);
            if (f.exists()) {
                f.delete();
            }
            f = new File(incrOutputFile);
            if (f.exists()) {
                f.delete();
            }
            isFirstTimeResults = false;
        }

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
        r.attrSel = (attrSelName==null)?" No Attr Sel":attrSelName;
        r.startDate = startDate;
        r.endDate = endDate;
        r.timeRequired = time;
        r.accuracy = eval.pctCorrect();
        r.totSamples = eval.numInstances();
        //casi in cui count della severity1 è 0
        if (eval.getClassPriors().length < 4){
            r.classSamples = new double[4];
            r.classSamples[0]=0.0;
            r.classSamples[1]=eval.getClassPriors()[0];
            r.classSamples[2]=eval.getClassPriors()[1];
            r.classSamples[3]=eval.getClassPriors()[2];
        } else
            r.classSamples = eval.getClassPriors();

        for(int i=0; i<4; i++) {
            try {
                r.classTPR[i] = eval.truePositiveRate(i);

                r.classFPR[i] = eval.falsePositiveRate(i);
                r.precision[i] = eval.precision(i);
                r.recall[i] = eval.recall(i);
                r.fMeasure[i] = eval.fMeasure(i);
            } catch (ArrayIndexOutOfBoundsException e){
                r.classTPR[i] = 0.0;
                r.classFPR[i] = 0.0;
                r.precision[i] = 0.0;
                r.recall[i] = 0.0;
                r.fMeasure[i] = 0.0;
            }

        }
        r.weightedTPR = eval.weightedTruePositiveRate();
        r.weightedFPR = eval.weightedFalsePositiveRate();
        r.weightedPrecision = eval.weightedPrecision();
        r.weightedRecall = eval.weightedRecall();
        r.weightedFMeasure = eval.weightedFMeasure();
        r.summaryEval = eval.toString();
        r.confusionMatrix = eval.toMatrixString();
        return r;
        /****************************************************************************************/
    }

    public void printResultAcc(Result r) throws Exception{
        if(isFirstTimeAcc){
            File dir = new File("statistics");
            for(File file: dir.listFiles()) {
                if (!file.isDirectory())
                    file.delete();
            }
            isFirstTimeAcc=false;
        }
        //Date date = new Date();
        //SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");

        //String filenameAcc = "statistics\\accuracy_"+r.classifier+"_"+r.attrSel+".csv";
        String filenameFmeasure = "statistics\\fmeasure_"+r.classifier+"_"+r.attrSel+".csv";

        FileWriter fileWriterAcc = new FileWriter(filenameFmeasure,true);
        PrintWriter printWriterAcc = new PrintWriter(fileWriterAcc);

        printAccuracy(printWriterAcc, r);

        fileWriterAcc.close();
        printWriterAcc.close();
    }

    public void printAccuracy(PrintWriter printWriter3, Result r){
        NumberFormat formatter = new DecimalFormat("#.###");
        //printWriter3.printf("%s,%s,%s\n", r.startDate,r.endDate,formatter.format(r.accuracy).replace(",","."));
        printWriter3.printf("%s,%s,", r.startDate,r.endDate);
        for(int i=0; i<r.fMeasure.length; i++)
            printWriter3.printf("%s,",formatter.format(r.fMeasure[i]).replace(",","."));
        printWriter3.printf("%s\n",formatter.format(r.weightedFMeasure).replace(",","."));
    }

}