import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class Driver {
    public static void main(String[] args) throws Exception {
        ManageCSV manager = new ManageCSV();

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        String dateString = "2016-06-30 12:00:00";
        Date date = sdf.parse(dateString);

        //manager.getTuples(date, 3);
        //manager.reduceList();
        //manager.writeCSV("templeReduced.csv");


        double trainPercentage = 66.0;
        int randomSeed = 1;

        //CSVLoader source = new CSVLoader();
        DataSource source = new DataSource("temple.arff");
        //source.setSource(new File("templeReduced.csv"));

        final Instances dataSet = source.getDataSet();
        dataSet.randomize(new Random(randomSeed));

        int trainSize = (int)Math.round(dataSet.numInstances() * trainPercentage / 100);
        int testSize = dataSet.numInstances() - trainSize;

        Instances train = new Instances(dataSet, 0, trainSize);
        //train.setClassIndex(2);

        Instances test = new Instances(dataSet, trainSize, testSize);

        int[] numInstancesSeverity = manager.getCountSeverity();
        List<Instances> dataFiltered = Preprocessor.filter(train, test, numInstancesSeverity[3]);

        AttributeSelection attSel = new AttributeSelection(dataFiltered.get(0), dataFiltered.get(1));
        List<List<Instances>> listAttrSel = new ArrayList<>();

        List<Instances> list1 = attSel.cfs_BestFirst(null,null);
        List<Instances> list2 = attSel.cfs_GreedyStepWise(null,null);
        List<Instances> list3 = attSel.InfoGain_Ranker(null,null);
        //List<Instances> list4 = attSel.PCA_Ranker(null,null);
        listAttrSel.add(list1);
        listAttrSel.add(list2);
        listAttrSel.add(list3);
        //listAttrSel.add(list4);

        Date dateTimeStart = new Date();
        Date dateTimeEnd = new Date();
        long start = dateTimeStart.getTime();
        long end = 0L;
        for(List<Instances> datasets : listAttrSel) {
            Classifier classifier1 = new Classifier(datasets.get(0), datasets.get(1), "");
            classifier1.j48(null);
            classifier1.randomForest(null);
            classifier1.naiveBayes(null);
            dateTimeEnd = new Date();
            end = dateTimeEnd.getTime();
            System.out.println("Time: " + Long.toString((end-start)/1000));
            System.out.println("____________________________________________________________________________________________________________");
        }

        System.out.println("fine");

    }
}
