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

        manager.getTuplesFast(date, 3);
        manager.reduceList();
        manager.writeCSV("templeReduced.csv");
        //manager.saveARFF(new File("templeReduced.csv"));


        double trainPercentage = 66.0;
        int randomSeed = 1;

        CSVLoader source = new CSVLoader();
        //DataSource source = new DataSource("templeLoad.arff");
        source.setMissingValue("nan");
        source.setNominalAttributes("1-4,12-20,21,27,30-47,49,50");
        source.setNumericAttributes("5-9,11,22-26,28,29,48");
        source.setStringAttributes("10");
        source.setSource(new File("templeReduced.csv"));

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

        Classifier classifier = new Classifier();
        for(List<Instances> datasets : listAttrSel) {
            classifier.updateClassifier(datasets.get(0),datasets.get(1), "cfs_BestFirst", "2000-01-01","2000-03-01");

            classifier.j48(null);
            classifier.randomForest(null);
            classifier.naiveBayes(null);
        }

        List<String> attrNames = new ArrayList<>();
        attrNames.add("cfs_BestFirst");
        attrNames.add("cfs_GreedyStepWise");
        attrNames.add("InfoGain_Ranker");

        for(int i=0; i<listAttrSel.size(); i++){
            List<Instances> datasets = listAttrSel.get(i);
            classifier.updateClassifier(datasets.get(0),datasets.get(1), attrNames.get(i), "2000-01-01","2000-03-01");
            classifier.j48(null);
            classifier.randomForest(null);
            classifier.naiveBayes(null);
        }

        Visualizer.printResults(classifier.getResults(),"results.txt");
        System.out.println("fine");

    }
}
