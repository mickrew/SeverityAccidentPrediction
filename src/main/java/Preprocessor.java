import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SpreadSubsample;
import java.util.ArrayList;
import java.util.List;

public class Preprocessor {

    public static List<Instances> filter (Instances train, Instances test) throws Exception {
        System.out.println("------------------------------------");
        System.out.println("===> Start PreProcessing");

        System.out.println("Applying filter: SpreadSubsample");
        SpreadSubsample filter = new SpreadSubsample();
        filter.setDistributionSpread(0);
        filter.setInputFormat(train);

        // S=Random Seed; M=Max Class Distr. Spread; W=mantain weight; X=max # samples
        String opt = "-S 1 -M 10.0 -W no -X 30000";
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

        List<Instances> list = new ArrayList<>();
        list.add(newTrain);
        list.add(newTest);
        return list;
    }
}
