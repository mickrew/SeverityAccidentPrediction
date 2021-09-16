import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

public class Visualizer{

    public static void printResults(List<Result> results, String fileName) throws IOException {
        String[] fileList = {fileName, "incremental"+fileName};
        for(int k=0; k<2;k++){
            File file = new File(fileList[k]);
            PrintWriter printWriter = new PrintWriter(file);
            printWriter.printf("\tSliding Window Incremental Analysis: Severity Accident Prediction\n\n");
            printWriter.printf("%-10s%-10s\n", "Start","End");
            Result r_new;
            Result r = new Result();
            for(int j=0; j<results.size(); j++){
                if(k==0 || j==0){
                    r = results.get(j);
                    printSingleResult(printWriter, r);
                } else {
                    r_new = results.get(j);
                    printIncrementalResult(printWriter, r, r_new);
                    r = r_new;
                }
            }
            printWriter.close();
        }

    }

    private static void printSingleResult(PrintWriter printWriter, Result r){
        NumberFormat formatter = new DecimalFormat("#.##");

        printWriter.printf("-----------------------------------------------------------------------------------------\n");
        printWriter.printf("%-12s%-12s%-6s%-20s%-15s%-20s\n", r.startDate,r.endDate,r.classifier, r.attrSel,"Accuracy: "+formatter.format(r.accuracy), "ClassifierTime:"+r.timeRequired);
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n","", "Per Class:","#Samples", "TPR", "FPR", "Precision", "Recall","F-measure");
        double sum=0;
        for(int i=0; i<4; i++) {
            printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Sev" + (i + 1) + ":", r.classSamples[i], formatter.format(r.classTPR[i]), formatter.format(r.classFPR[i]), formatter.format(r.precision[i]), formatter.format(r.recall[i]), formatter.format(r.fMeasure[i]));
            sum += r.classSamples[i];
        }
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Weighted:", sum,formatter.format(r.weightedTPR), formatter.format(r.weightedFPR), formatter.format(r.weightedPrecision), formatter.format(r.weightedRecall), formatter.format(r.weightedFMeasure));
    }

    private static void printIncrementalResult(PrintWriter printWriter, Result newR, Result oldR){
        NumberFormat formatter = new DecimalFormat("#.##");
        printWriter.printf("-----------------------------------------------------------------------------------------\n");
        printWriter.printf("%-12s%-12s%-6s%-20s%-15s%-20s\n", newR.startDate,newR.endDate,newR.classifier, newR.attrSel,"Accuracy: "+formatter.format((newR.accuracy-oldR.accuracy)), "ClassifierTime:"+newR.timeRequired);
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n","", "Per Class:","#Samples", "TPR", "FPR", "Precision", "Recall","F-measure");
        double newSum=0, oldSum=0;
        for(int i=0; i<4; i++) {
            printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Sev" + (i + 1) + ":", newR.classSamples[i]-oldR.classSamples[i], formatter.format(newR.classTPR[i]-oldR.classTPR[i]), formatter.format(newR.classFPR[i]-oldR.classFPR[i]), formatter.format(newR.precision[i]-oldR.precision[i]), formatter.format(newR.recall[i]-oldR.recall[i]), formatter.format(newR.fMeasure[i]-oldR.fMeasure[i]));
            newSum += newR.classSamples[i];
            oldSum += oldR.classSamples[i];
        }
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Weighted:", newSum-oldSum,formatter.format(newR.weightedTPR-oldR.weightedTPR), formatter.format(newR.weightedFPR-oldR.weightedFPR), formatter.format(newR.weightedPrecision-oldR.weightedPrecision), formatter.format(newR.weightedRecall-oldR.weightedRecall), formatter.format(newR.weightedFMeasure-oldR.weightedFMeasure));
    }
}