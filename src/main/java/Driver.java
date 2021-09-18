import org.apache.commons.lang3.time.DateUtils;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;


public class Driver {
    public static List<Instances> loadDataSplitTrainTest() throws Exception {

        ManageCSV manager = new ManageCSV();
        CSVLoader source = new CSVLoader();
        //DataSource source = new DataSource("templeLoad.arff");
        source.setMissingValue("nan");
        source.setNominalAttributes("1-4,12-20,21,27,30-47,49,50");
        source.setNumericAttributes("5-9,11,22-26,28,29,48");
        source.setStringAttributes("10");
        source.setSource(new File("templeReduced.csv"));

        double trainPercentage = 66.0;
        int randomSeed = 1;
        final Instances dataSet = source.getDataSet();
        dataSet.randomize(new Random(randomSeed));

        int trainSize = (int)Math.round(dataSet.numInstances() * trainPercentage / 100);
        int testSize = dataSet.numInstances() - trainSize;

        Instances train = new Instances(dataSet, 0, trainSize);
        Instances test = new Instances(dataSet, trainSize, testSize);

        List<Instances> dataNotFiltered = new ArrayList<>();
        dataNotFiltered.add(train);
        dataNotFiltered.add(test);

        return dataNotFiltered;
    }

    public static void main(String[] args) throws Exception {
        Timer timer = new Timer();
        timer.startTimer();
        ManageCSV manager = new ManageCSV();
        Classifier classifier = new Classifier("results.txt");


        List<String> attrNames = new ArrayList<>();
        attrNames.add("cfs_BestFirst");
        attrNames.add("cfs_GreedyStepWise");
        attrNames.add("InfoGain_Ranker");

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        SimpleDateFormat sdf1 = new SimpleDateFormat("yyyy-MM-dd");
        String dateString = "2020-01-01 00:00:00";
        Date dateStart = sdf.parse(dateString);
        Date dateEnd;
        /*************/
        final int DRIFT =1;
        final int GRANULARITY = 4;
        final int NUM_ITERATION = 2;
        /*************/
        manager.setGranularity(GRANULARITY);
        int lastGranularity= manager.getGranularity();

        for (int j= 0; j<NUM_ITERATION; j++) {
            lastGranularity= manager.getGranularity();

            dateStart = DateUtils.addWeeks(dateStart, DRIFT*j);

            System.out.println("==========================================");
            System.out.println("===> Start Reading");
            dateEnd = manager.getTuplesFromDB(dateStart);
            manager.writeCSV("temple.csv");

            manager.reduceList();

            manager.writeCSV("templeReduced.csv");
            //manager.saveARFF(new File("templeReduced.csv"));

            List<Instances> dataNotFiltered = loadDataSplitTrainTest();

            int[] numInstancesSeverity = manager.getCountSeverity();
            List<Instances> dataFiltered = Preprocessor.filter(dataNotFiltered.get(0), dataNotFiltered.get(1), numInstancesSeverity[3]);

            AttributeSelection attSel = new AttributeSelection(dataFiltered.get(0), dataFiltered.get(1));
            List<List<Instances>> listAttrSel = new ArrayList<>();

            List<Instances> list1 = attSel.cfs_BestFirst(null, null);
            //List<Instances> list2 = attSel.cfs_GreedyStepWise(null, null);
            //List<Instances> list3 = attSel.InfoGain_Ranker(null, null);
            //List<Instances> list4 = attSel.PCA_Ranker(null,null);
            listAttrSel.add(list1);
            //listAttrSel.add(list2);
            //listAttrSel.add(list3);
            //listAttrSel.add(list4);

            /*
            for(List<Instances> datasets : listAttrSel) {
                classifier.updateClassifier(datasets.get(0),datasets.get(1), "cfs_BestFirst", "2000-01-01","2000-03-01");

                classifier.j48(null);
                classifier.randomForest(null);
                classifier.naiveBayes(null);
            }
            */

            System.out.println("------------------------------------");
            System.out.println("===> Start Classifing");
            for (int i = 0; i < listAttrSel.size(); i++) {

                List<Instances> datasets = listAttrSel.get(i);
                classifier.updateClassifier(datasets.get(0), datasets.get(1), attrNames.get(i), sdf1.format(dateStart), sdf1.format(dateEnd));
                System.out.println("J48 is running");
                classifier.j48(null);
                System.out.println("RandomForest is running");
                classifier.randomForest(null);
                //classifier.naiveBayes(null);
            }
        }
        //Visualizer.printResults(classifier.getResults(),"results.txt");
        timer.stopTimer();
        System.out.println("fine " + timer.getTime());

    }
}
