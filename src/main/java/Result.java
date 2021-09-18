import java.io.*;

public class Result implements Serializable{
    public String classifier;
    public String attrSel;
    public String startDate;
    public String endDate;
    public String timeRequired;
    public double[] classSamples;
    public double totSamples;
    public double accuracy;
    public double[] classTPR;
    public double[] classFPR;
    public double[] precision;
    public double[] recall;
    public double[] fMeasure;
    public double weightedTPR;
    public double weightedFPR;
    public double weightedPrecision;
    public double weightedRecall;
    public double weightedFMeasure;
    public String summaryEval;
    public String confusionMatrix;


    public Result(){
        classSamples = new double[4];
        classTPR = new double[4];
        classFPR = new double[4];
        precision = new double[4];
        recall = new double[4];
        fMeasure = new double[4];
    }

    public Result(String classifierName, String attrSelName, String startDate, String endDate, String classifierTime,
                  double[] classSamples, double totSamples, double accuracy, double[] classTPR, double[] classFPR,
                  double[] precision, double[] recall, double[] fMeasure, double weightedTPR, double weightedFPR,
                  double weightedPrecision, double weightedRecall, double weightedFMeasure,
                  String summaryEval, String confusionMatrix){
        this();
        this.classifier = classifierName;
        this.attrSel = attrSelName;
        this.startDate = startDate;
        this.endDate = endDate;
        this.timeRequired = classifierTime;
        this.classSamples = classSamples;
        this.totSamples = totSamples;
        this.accuracy = accuracy;
        this.classTPR = classTPR;
        this.classFPR = classFPR;
        this.precision = precision;
        this.recall = recall;
        this.fMeasure = fMeasure;
        this.weightedTPR = weightedTPR;
        this.weightedFPR = weightedFPR;
        this.weightedPrecision = weightedPrecision;
        this.weightedRecall = weightedRecall;
        this.weightedFMeasure = weightedFMeasure;
        this.summaryEval = summaryEval;
        this.confusionMatrix = confusionMatrix;
    }



    public String getResult(){
        return summaryEval;
    }
}
