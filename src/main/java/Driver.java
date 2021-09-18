import org.apache.commons.lang3.time.DateUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class Driver {
    /***********************************************************************************************************/
    /**************************** DRIVER FOR INCREMENTAL CLASSIFIER EXECUTION **********************************/
    /***********************************************************************************************************/

    private static SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private static SimpleDateFormat sdf1 = new SimpleDateFormat("yyyy-MM-dd");
    private static ArrayList<Integer> numTuples = new ArrayList<>();
    /*************/
    private static double PERCENTAGESPLIT = 66.0;
    private final static int randomSeed = (int)System.currentTimeMillis();
    private final static int DRIFT =1;
    private final static int NUM_ITERATION = 2;
    private final static String dateString = "2016-02-01 00:00:00";

    private static boolean CROSS_VALIDATION = true;
    private static int GRANULARITY = 4;
    /*************/

    private static boolean prova = true;
    private static NaiveBayesUpdateable nBayesUpdatable = new NaiveBayesUpdateable();
    private static UpdateableClassifier updateableClassifier = nBayesUpdatable;
    private static weka.classifiers.Classifier classifierInterface = nBayesUpdatable;
    private static IncrClassifier incrNaiveBayes = new IncrClassifier("NAIVE_BAYES_UPDATABLE",null);

    public static List<Instances> loadDataSplitTrainTest(double trainPercentage) throws Exception {
        ManageCSV manager = new ManageCSV();
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

        String nameFile = dateString.split(" ")[0] + "_" + String.valueOf(NUM_ITERATION)+ "_DR" + String.valueOf(DRIFT) + "_GR" + String.valueOf(GRANULARITY) + "_" + ".txt";
        Visualizer incrVisualizer = new Visualizer("results\\" +"updateable"+nameFile);
        /*List<String> attrNames = new ArrayList<>();
        attrNames.add("cfs_BestFirst");
        attrNames.add("cfs_GreedyStepWise");
        attrNames.add("InfoGain_Ranker");
        */
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        SimpleDateFormat sdf1 = new SimpleDateFormat("yyyy-MM-dd");
        Date dateStart = sdf.parse(dateString);
        Date nextDateStart = dateStart;
        Date dateEnd;

        manager.setGranularity(GRANULARITY);
        //int lastGranularity= manager.getGranularity();

        for (int j= 0; j<NUM_ITERATION; j++) {
            System.out.println("==========================================");
            System.out.println("Num Iteration: " + Integer.valueOf(j+1) + "/" + Integer.valueOf(NUM_ITERATION));
            //lastGranularity= manager.getGranularity();

            nextDateStart = DateUtils.addWeeks(nextDateStart, DRIFT*j);

            System.out.println("==========================================");
            System.out.println("===> Start Reading");
            dateEnd = manager.getTuplesFromDB(nextDateStart);
            manager.writeCSV("temple.csv");

            manager.reduceList();
            manager.writeCSV("templeReduced.csv");
            //manager.saveARFF(new File("templeReduced.csv"));

            numTuples.add(manager.getCountTuples());

            List<Instances> dataNotFiltered = loadDataSplitTrainTest(PERCENTAGESPLIT);

            int[] numInstancesSeverity = manager.getCountSeverity();
            List<Instances> dataFiltered = Preprocessor.filter(dataNotFiltered.get(0), dataNotFiltered.get(1), numInstancesSeverity[3]);


            /**************************** CLASSIFICATION PHASE **********************************/
            System.out.println("------------------------------------");
            System.out.println("===> Start Classifying");

            for(Instances dataset: dataFiltered)
                dataset.setClassIndex(0);

            //IncrClassifier incrNaiveBayes = IncrClassifier.getInstance("NAIVE_BAYES_UPDATABLE",null);
            //IncrClassifier incrHoeffdingTree = IncrClassifier.getInstance("HOEFFDING_TREE",null);

            System.out.println("NaiveBayesUpdatable is running (updating its model)");
            Result naiveBayesResult = incrNaiveBayes.update(dataFiltered,sdf1.format(dateStart),sdf1.format(dateEnd));
            System.out.println("HoeffdingTree is running (updating its model)");
            //Result hoeffdingTreeResult = incrHoeffdingTree.update(dataFiltered,sdf1.format(dateStart),sdf1.format(dateEnd));
            incrVisualizer.addResult(naiveBayesResult);
            //incrVisualizer.addResult(hoeffdingTreeResult);





            /*
            if(prova==true){
                classifierInterface.buildClassifier(dataFiltered.get(0));
            }else
                for (Instance sample : dataFiltered.get(0))
                    updateableClassifier.updateClassifier(sample);
                    //nBayesUpdatable.updateClassifier(sample);
            Evaluation evaluation = new Evaluation(dataFiltered.get(0));
            evaluation.evaluateModel(classifierInterface, dataFiltered.get(1));
            Result r = Visualizer.evalResult(evaluation, "NAIVE_BAYES_UPDATABLE", null, "", "startDate", "endDate");
            incrVisualizer.addResult(r);
            */
        }
        Integer sum = 0;
        for(Integer i : numTuples)
            sum += i;

        System.out.println("\nMean of tuples taken: " + sum);

        timer.stopTimer();
        NumberFormat formatter = new DecimalFormat("##");

        System.out.println("\n Application time: " + timer.getTime()+"s");
    }
}
