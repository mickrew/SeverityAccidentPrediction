import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Preprocessor {
    private static SpreadSubsample spreadFilter;
    private static NumericCleaner cleanerFilterTemperature;
    private static NumericCleaner cleanerFilterPressure;
    private static NumericCleaner cleanerFilterWindspeed;
    private static NumericCleaner cleanerFilterVisibility;
    private static ReplaceMissingValues replaceFilter;
    private static ClassAssigner classAssigner;
    private static Remove removeFilter;
    private static NumericToNominal numericToNominal;
    private static SortLabels sortLabelsFilter;


    public static List<Instances> filter (Instances train, Instances test, int maxSpread) throws Exception {
        System.out.println("------------------------------------");
        System.out.println("===> Start PreProcessing");

        MultiFilter mf = new MultiFilter();
        mf.setFilters(new Filter[] {cleanerFilterVisibility, cleanerFilterWindspeed, cleanerFilterPressure, cleanerFilterTemperature, replaceFilter});

        //System.out.println("Applying filter: ClassAssigner");
        classAssigner = new ClassAssigner();
        classAssigner.setClassIndex("2");

        //System.out.println("Applying filter: SpreadSubsample");
        spreadFilter = new SpreadSubsample();
        spreadFilter.setMaxCount(maxSpread);
        //spreadFilter.setInputFormat(train);

        // S=Random Seed; M=Max Class Distr. Spread; W=mantain weight; X=max # samples
        // String opt = "-S 1 -M 10.0 -W no -X 30000";
        // System.out.println("\tfilter options:");
        // System.out.println("\tSeed: 1\tMax Class Spread: 10.0\tweight: no\tMax #samples: 30000");
        //String[] optArray = weka.core.Utils.splitOptions(opt);
        //spreadFilter.setOptions(optArray);
        //spreadFilter.setInputFormat(train);


        String[] op = new String[]{"-R","1,3,4,7,8,10,11,18,21,23,38,43,44,46"};
        Remove rmv = new Remove();
        rmv.setOptions(op);
        rmv.setInputFormat(train);

        /*
        removeFilter.setAttributeIndices("1,3,4,7,8,10-12,17-21,23,38,43,45-47");
        removeFilter.setInputFormat(train);
        */

        sortLabelsFilter = new SortLabels();
        sortLabelsFilter.setAttributeIndices("first-last");

        numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices("1,36");

        cleanerFilterTemperature = new NumericCleaner();
        cleanerFilterTemperature.setAttributeIndices("13");
        cleanerFilterTemperature.setMaxDefault(Double.NaN);
        cleanerFilterTemperature.setMaxThreshold(130.0);
        cleanerFilterTemperature.setMinDefault(Double.NaN);
        cleanerFilterTemperature.setMinThreshold(-130.0);

        cleanerFilterPressure = new NumericCleaner();
        cleanerFilterPressure.setAttributeIndices("15");
        cleanerFilterPressure.setMaxDefault(Double.NaN);
        cleanerFilterPressure.setMaxThreshold(32.06);
        cleanerFilterPressure.setMinDefault(Double.NaN);
        cleanerFilterPressure.setMinThreshold(25.0);

        cleanerFilterVisibility = new NumericCleaner();
        cleanerFilterVisibility.setAttributeIndices("16");
        cleanerFilterVisibility.setMaxDefault(Double.NaN);
        cleanerFilterVisibility.setMaxThreshold(10.1);
        cleanerFilterVisibility.setMinDefault(Double.NaN);
        cleanerFilterVisibility.setMinThreshold(0.0);

        cleanerFilterWindspeed = new NumericCleaner();
        cleanerFilterWindspeed.setAttributeIndices("18");
        cleanerFilterWindspeed.setMaxDefault(Double.NaN);
        cleanerFilterWindspeed.setMaxThreshold(254.1);
        cleanerFilterWindspeed.setMinDefault(Double.NaN);
        cleanerFilterWindspeed.setMinThreshold(0.0);

        replaceFilter = new ReplaceMissingValues();


        // configures the Filter based on train instances and returns filtered instances: both training and test set

        Instances newTrain = Filter.useFilter(train, rmv);

        sortLabelsFilter.setInputFormat(newTrain);
        numericToNominal.setInputFormat(newTrain);
        cleanerFilterTemperature.setInputFormat(newTrain);
        cleanerFilterPressure.setInputFormat(newTrain);
        cleanerFilterVisibility.setInputFormat(newTrain);
        cleanerFilterWindspeed.setInputFormat(newTrain);
        replaceFilter.setInputFormat(newTrain);


        newTrain = Filter.useFilter(newTrain, numericToNominal);
        newTrain = Filter.useFilter(newTrain, cleanerFilterVisibility);
        newTrain = Filter.useFilter(newTrain, cleanerFilterWindspeed);
        newTrain = Filter.useFilter(newTrain, cleanerFilterPressure);
        newTrain = Filter.useFilter(newTrain, cleanerFilterTemperature);
        newTrain = Filter.useFilter(newTrain, replaceFilter);
        newTrain = Filter.useFilter(newTrain, sortLabelsFilter);

        Instances newTest = Filter.useFilter(test, rmv);

        /*
        numericToNominal.setInputFormat(newTest);
        cleanerFilterTemperature.setInputFormat(newTest);
        cleanerFilterPressure.setInputFormat(newTest);
        cleanerFilterVisibility.setInputFormat(newTest);
        cleanerFilterWindspeed.setInputFormat(newTest);
        replaceFilter.setInputFormat(newTest);
         */

        newTest = Filter.useFilter(newTest, numericToNominal);
        newTest = Filter.useFilter(newTest, cleanerFilterVisibility);
        newTest = Filter.useFilter(newTest, cleanerFilterWindspeed);
        newTest = Filter.useFilter(newTest, cleanerFilterPressure);
        newTest = Filter.useFilter(newTest, cleanerFilterTemperature);
        newTest = Filter.useFilter(newTest, replaceFilter);
        newTest = Filter.useFilter(newTest, sortLabelsFilter);


        ArffSaver saver = new ArffSaver();
        saver.setInstances(newTrain);
        saver.setFile(new File("TrainSetFiltered.arff"));
        saver.writeBatch();

        saver = new ArffSaver();
        saver.setInstances(newTrain);
        saver.setFile(new File("TestSetFiltered.arff"));
        saver.writeBatch();

        //Instances newTrain = Filter.useFilter(train, mf);
        System.out.println("Training Set filtered");
        //Instances newTest = Filter.useFilter(test, mf);
        System.out.println("Test Set filtered");

        List<Instances> list = new ArrayList<>();
        newTrain.setClassIndex(0);
        newTest.setClassIndex(0);
        list.add(newTrain);
        list.add(newTest);


        return list;
    }

    public static void setRemoveFilter(Remove removeFilter) {
        Preprocessor.removeFilter = removeFilter;
    }
}
