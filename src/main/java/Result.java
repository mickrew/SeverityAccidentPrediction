import java.io.*;

public class Result implements Serializable{
    public String classifier;
    public String attrSel;
    public double[] classSamples;
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

    public Result(String classifierName, String attrSelName){
        classifier = classifierName;
        attrSel = attrSelName;
        classSamples = new double[4];
        classTPR = new double[4];
        classFPR = new double[4];
        precision = new double[4];
        recall = new double[4];
        fMeasure = new double[4];

    }

    public String getResult(){
        return summaryEval;
    }
}
