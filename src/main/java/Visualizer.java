import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
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
        printWriter.printf("-----------------------------------------------------------------------------------------\n");
        printWriter.printf("%-12s%-12s%-6s%-20s%-15s%-20s\n", r.startDate,r.endDate,r.classifier, r.attrSel,"Accuracy: "+r.accuracy, "ClassifierTime:"+r.timeRequired);
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n","", "Per Class:","#Samples", "TPR", "FPR", "Precision", "Recall","F-measure");
        double sum=0;
        for(int i=0; i<4; i++) {
            printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Sev" + (i + 1) + ":", r.classSamples[i], r.classTPR[i], r.classFPR[i], r.precision[i], r.recall[i], r.fMeasure[i]);
            sum += r.classSamples[i];
        }
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Weighted:", sum,r.weightedTPR, r.weightedFPR, r.weightedPrecision, r.weightedRecall, r.weightedFMeasure);
    }

    private static void printIncrementalResult(PrintWriter printWriter, Result newR, Result oldR){
        printWriter.printf("-----------------------------------------------------------------------------------------\n");
        printWriter.printf("%-12s%-12s%-6s%-20s%-15s%-20s\n", newR.startDate,newR.endDate,newR.classifier, newR.attrSel,"Accuracy: "+(newR.accuracy-oldR.accuracy), "ClassifierTime:"+newR.timeRequired);
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n","", "Per Class:","#Samples", "TPR", "FPR", "Precision", "Recall","F-measure");
        double newSum=0, oldSum=0;
        for(int i=0; i<4; i++) {
            printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Sev" + (i + 1) + ":", newR.classSamples[i]-oldR.classSamples[i], newR.classTPR[i]-oldR.classTPR[i], newR.classFPR[i]-oldR.classFPR[i], newR.precision[i]-oldR.precision[i], newR.recall[i]-oldR.recall[i], newR.fMeasure[i]-oldR.fMeasure[i]);
            newSum += newR.classSamples[i];
            oldSum += oldR.classSamples[i];
        }
        printWriter.printf("%-50s%-12s%-10s%-10s%-10s%-10s%-10s%-10s\n", "", "Weighted:", newSum-oldSum,newR.weightedTPR-oldR.weightedTPR, newR.weightedFPR-oldR.weightedFPR, newR.weightedPrecision-oldR.weightedPrecision, newR.weightedRecall-oldR.weightedRecall, newR.weightedFMeasure-oldR.weightedFMeasure);
    }
}