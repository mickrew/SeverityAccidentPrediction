import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.filters.supervised.instance.SpreadSubsample;

import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;

public class Preprocessor {

    public static void filter (String fileTrain, String fileTest) throws Exception {
        System.out.println("------------------------------------");
        System.out.println("===> Start PreProcessing");

        // DataSource source = new DataSource(fileTrain);
        // Instances train = source.getDataSet();
        Instances train = ConverterUtils.DataSource.read(fileTrain);
        train.setClassIndex(1);
        //ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(fileTest);
        //Instances test = source2.getDataSet();
        Instances test = ConverterUtils.DataSource.read(fileTest);
        test.setClassIndex(1);

        System.out.println("Applying filter: SpreadSubsample");
        SpreadSubsample filter = new SpreadSubsample();
        filter.setDistributionSpread(0);
        filter.setInputFormat(train);

        String opt = "-S 1 -M 10.0 -W no -X 30000"; // S=Random Seed; M=Max Class Distr. Spread; W=mantain weight; X=max # samples
        System.out.println("\tfilter options:");
        System.out.println("\tSeed: 1\tMax Class Spread: 10.0\tweight: no\tMax #samples: 30000");
        String[] optArray = weka.core.Utils.splitOptions(opt);
        filter.setOptions(optArray);
        filter.setInputFormat(train);

        // configures the Filter based on train instances and returns filtered instances: both training and test set
        Instances newTrain = Filter.useFilter(train, filter);
        System.out.println("\t> Training Set filtered");
        Instances newTest = Filter.useFilter(test, filter);
        System.out.println("\t> Test Set filtered");

        ArffSaver saver1 = new ArffSaver();
        saver1.setInstances(newTrain);
        ArffSaver saver2 = new ArffSaver();
        saver2.setInstances(newTrain);

        saver1.setFile(new File(fileTrain));
        saver2.setFile(new File(fileTest));

        saver1.writeBatch();
        saver2.writeBatch();
    }
}
