import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.ClassAssigner;
import weka.filters.unsupervised.attribute.NumericCleaner;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

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

    public static List<Instances> filter (Instances train, Instances test, int maxSpread) throws Exception {
        System.out.println("------------------------------------");
        System.out.println("===> Start PreProcessing");

        MultiFilter mf = new MultiFilter();
        mf.setFilters(new Filter[] {cleanerFilterVisibility, cleanerFilterWindspeed, cleanerFilterPressure, cleanerFilterTemperature, replaceFilter});

        System.out.println("Applying filter: ClassAssigner");
        classAssigner = new ClassAssigner();
        classAssigner.setClassIndex("2");

        System.out.println("Applying filter: SpreadSubsample");
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

        System.out.println("Applying filter: NumericCleaner ");
        cleanerFilterTemperature = new NumericCleaner();
        cleanerFilterTemperature.setAttributeIndices("9");
        cleanerFilterTemperature.setMaxDefault(Double.NaN);
        cleanerFilterTemperature.setMaxThreshold(130.0);
        cleanerFilterTemperature.setMinDefault(Double.NaN);
        cleanerFilterTemperature.setMinThreshold(-130.0);
        cleanerFilterTemperature.setInputFormat(train);

        cleanerFilterPressure = new NumericCleaner();
        cleanerFilterPressure.setAttributeIndices("11");
        cleanerFilterPressure.setMaxDefault(Double.NaN);
        cleanerFilterPressure.setMaxThreshold(32.06);
        cleanerFilterPressure.setMinDefault(Double.NaN);
        cleanerFilterPressure.setMinThreshold(25.0);
        cleanerFilterPressure.setInputFormat(train);

        cleanerFilterVisibility = new NumericCleaner();
        cleanerFilterVisibility.setAttributeIndices("12");
        cleanerFilterVisibility.setMaxDefault(Double.NaN);
        cleanerFilterVisibility.setMaxThreshold(10.1);
        cleanerFilterVisibility.setMinDefault(Double.NaN);
        cleanerFilterVisibility.setMinThreshold(0.0);
        cleanerFilterVisibility.setInputFormat(train);

        cleanerFilterWindspeed = new NumericCleaner();
        cleanerFilterWindspeed.setAttributeIndices("14");
        cleanerFilterWindspeed.setMaxDefault(Double.NaN);
        cleanerFilterWindspeed.setMaxThreshold(254.1);
        cleanerFilterWindspeed.setMinDefault(Double.NaN);
        cleanerFilterWindspeed.setMinThreshold(0.0);
        cleanerFilterWindspeed.setInputFormat(train);

        replaceFilter = new ReplaceMissingValues();
        replaceFilter.setInputFormat(train);


        // configures the Filter based on train instances and returns filtered instances: both training and test set

        Filter.useFilter(train, classAssigner);
        Filter.useFilter(train, spreadFilter);
        Filter.useFilter(train, cleanerFilterVisibility);
        Filter.useFilter(train, cleanerFilterWindspeed);
        Filter.useFilter(train, cleanerFilterPressure);
        Filter.useFilter(train, cleanerFilterTemperature);
        Filter.useFilter(train, replaceFilter);
        Instances newTrain = new Instances(train);

        Instances newTest = Filter.useFilter(train, classAssigner);
        newTrain = Filter.useFilter(train, cleanerFilterVisibility);
        newTrain = Filter.useFilter(train, cleanerFilterWindspeed);
        newTrain = Filter.useFilter(train, cleanerFilterPressure);
        newTrain = Filter.useFilter(train, cleanerFilterTemperature);
        newTrain = Filter.useFilter(train, replaceFilter);



        //Instances newTrain = Filter.useFilter(train, mf);
        System.out.println("\t> Training Set filtered");
        //Instances newTest = Filter.useFilter(test, mf);
        System.out.println("\t> Test Set filtered");

        List<Instances> list = new ArrayList<>();
        list.add(newTrain);
        list.add(newTest);
        return list;
    }
}
