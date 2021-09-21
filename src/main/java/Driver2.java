import org.apache.commons.lang3.time.DateUtils;
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

public class Driver2 {
    /***********************************************************************************************************/
    /************************ DRIVER FOR ATTRIBUTE SELECTION CLASSIFIER EXECUTION ******************************/
    /***********************************************************************************************************/

    private static SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private static SimpleDateFormat sdf1 = new SimpleDateFormat("yyyy-MM-dd");
    private static ArrayList<Integer> numTuples = new ArrayList<>();
    private static IncrClassifier incrClassifier = new IncrClassifier();
    private static Visualizer visualizer = new Visualizer();

    /*************/
    private static double PERCENTAGESPLIT = 66.0;
    private final static int randomSeed = (int)System.currentTimeMillis();

    private final static String J48 = "J48";
    private final static String RANDOM_FOREST = "RANDOM_FOREST";
    private final static String NAIVE_BAYES = "NAIVE_BAYES";
    private final static String CFS_BESTFIRST = "CFS_BESTFIRST";
    private final static String CFS_GREEDYSTEPWISE = "CFS_GREEDYSTEPWISE";
    private final static String INFO_GAIN = "INFOGAIN_RANKER";
    private final static String NAIVE_BAYES_UPDATABLE = "NAIVE_BAYES_UPDATABLE";
    private final static String HOEFFDING_TREE = "HOEFFDING_TREE";

    private final static String DATE_STRING_START_DATASET = "2017-02-01 00:00:00";
    private final static String DATELIMIT = "2020-12-31 23:59:59";

    private final static boolean FIXED_GRANULARITY = true;
    private final static int THRESHOLD_DIMENSION_TRAININGSET = 10000;
    private final static boolean RUN_UPDATABLE_CLASSIFIER = true;
    private final static boolean LOAD_ALL_TRAINING_AVAILABLE = true;

    private final static int DRIFT =4;
    private final static int GRANULARITY = 4;
    private final static int NUM_ITERATION = 256;
    private final static boolean CROSS_VALIDATION = false;

    private final static String TRAININGSET = "TrainingSet.csv";
    private final static String TESTSET ="TestSet.csv";


    /*************/

    private static String suffixNameFile = DATE_STRING_START_DATASET.split(" ")[0] + "_" + String.valueOf(NUM_ITERATION)+
                                    "_DR" + String.valueOf(DRIFT) + "_GR" + String.valueOf(GRANULARITY)+
                                    (CROSS_VALIDATION?"_CR":"" ) + ".txt";

    public static Instances loadData(String file) throws Exception {
        CSVLoader source = new CSVLoader();
        //DataSource source = new DataSource("templeLoad.arff");
        source.setMissingValue("nan");
        source.setNominalAttributes("1-4,12-20,21,27,30-47,49,50");
        source.setNumericAttributes("5-9,11,22-26,28,29,48");
        source.setStringAttributes("10");
        source.setSource(new File(file));

        Instances dataSet = source.getDataSet();

        return dataSet;
    }

    public static List<Instances> loadDataSplitTrainTest(double trainPercentage) throws Exception {

        CSVLoader source = new CSVLoader();

        source.setMissingValue("nan");
        source.setNominalAttributes("1-4,12-20,21,27,30-47,49,50");
        source.setNumericAttributes("5-9,11,22-26,28,29,48");
        source.setStringAttributes("10");
        source.setSource(new File(TRAININGSET));

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
        if(CROSS_VALIDATION)
            PERCENTAGESPLIT = 100.0;
        Timer timer = new Timer();
        timer.startTimer();
        ManageCSV manager = new ManageCSV(sdf.parse(DATE_STRING_START_DATASET));

        /**************** with Attribute Selection *******************/

        List<String> classifiersNames = new ArrayList<>();
        List<String> attrSelectionNames = new ArrayList<>();

        // 1 Classifier
        classifiersNames.add(J48);
        attrSelectionNames.add(CFS_BESTFIRST);
        // 2 Classifier
        classifiersNames.add(RANDOM_FOREST);
        attrSelectionNames.add(CFS_GREEDYSTEPWISE);
        /// 3 Classifier
        classifiersNames.add(NAIVE_BAYES);
        attrSelectionNames.add(INFO_GAIN);

        Date dateStartDataset = sdf.parse(DATE_STRING_START_DATASET);
        Date dateStartTraining;
        Date dateEndTraining;
        Date prevDateEndTraining;
        Date dateLimit = sdf.parse(DATELIMIT);

        int countTrainingSet = 0;
        manager.setGranularity(GRANULARITY);

        boolean lastIteration=false;

        for (int j= 0; j<NUM_ITERATION && !lastIteration; j++) {
            System.out.println("==========================================");
            System.out.println("Num Iteration: " + Integer.valueOf(j+1) + "/" + Integer.valueOf(NUM_ITERATION));

            dateStartTraining = DateUtils.addWeeks(dateStartDataset, DRIFT*j);

            prevDateEndTraining = DateUtils.addWeeks(dateStartTraining, manager.getGranularity());

            if (prevDateEndTraining.getTime() > dateLimit.getTime()) {
                lastIteration = true;
                dateEndTraining=dateLimit;
            } else
                dateEndTraining=prevDateEndTraining;


            System.out.println("==========================================");
            System.out.println("===> Start Reading");
            System.out.println("-----------------");
            System.out.println("Read Training Set");
            System.out.println("-----------------");

            if (countTrainingSet > THRESHOLD_DIMENSION_TRAININGSET)
                dateStartDataset = DateUtils.addWeeks(dateStartDataset, manager.getGranularity());
            if (LOAD_ALL_TRAINING_AVAILABLE)
                dateStartTraining=dateStartDataset;
            dateEndTraining = manager.getTuplesFromDB(dateStartTraining, FIXED_GRANULARITY, dateEndTraining, dateLimit);

            //manager.writeCSV("temple.csv");
            manager.printCoutnSeverity();
            manager.reduceList();
            countTrainingSet = manager.getCountTuples();

            manager.printCoutnSeverity();
            manager.writeCSV(TRAININGSET);
            System.out.println("-------------");
            System.out.println("Read Test Set");
            System.out.println("-------------");

            Date dateStartTest = dateEndTraining;

            Date dateEndTestSet = DateUtils.addWeeks(dateStartTest, (int) manager.getGranularity()/2);
            if (dateEndTestSet.getTime() > dateLimit.getTime()) {
                dateEndTestSet=dateLimit;
            }
            dateEndTestSet = manager.getTuplesFromDB(dateStartTest, FIXED_GRANULARITY, dateEndTestSet, dateLimit);

            manager.writeCSV(TESTSET);
            manager.printCoutnSeverity();


            numTuples.add(manager.getCountTuples());

            Instances trainingSet = loadData(TRAININGSET);
            Instances testSet = loadData(TESTSET);

            int[] numInstancesSeverity = manager.getCountSeverity();

            //dataFiltered
            //0 - TrainingSet
            //1 - TestSet
            List<Instances> dataFiltered = Preprocessor.filter(trainingSet, testSet, numInstancesSeverity[3]);

            /*
            List<Instances> dataNotFilteredProva = new ArrayList<>();
            dataNotFilteredProva = loadDataSplitTrainTest(PERCENTAGESPLIT);
            List<Instances> dataFilteredProva = Preprocessor.filter(dataNotFilteredProva.get(0), dataNotFilteredProva.get(1), numInstancesSeverity[3]);
            */

            if(j==0 && RUN_UPDATABLE_CLASSIFIER) {
                incrClassifier.buildIncrClassifier(NAIVE_BAYES_UPDATABLE, null);
                incrClassifier.buildIncrClassifier(HOEFFDING_TREE, null);
            }

            System.out.println("------------------------------------");
            System.out.println("===> Start Classifying");


            if(classifiersNames.size() != attrSelectionNames.size()){
                System.err.println("Error: classifier definition is wrong!");
                System.exit(1);
            }

            for(int i=0; i < classifiersNames.size(); i++){
                AttrSelectedClassifier classifier = new AttrSelectedClassifier(dataFiltered,CROSS_VALIDATION,
                                                    sdf1.format(dateStartTraining), sdf1.format(dateEndTraining));

                System.out.println(classifiersNames.get(i)+" is running");
                Result r = classifier.start(attrSelectionNames.get(i),null, null,
                                            classifiersNames.get(i),null);

                visualizer.addResult("results\\" + r.classifier + "_"+ r.attrSel + "_"+ suffixNameFile, r);
                visualizer.printResultAcc(r);
            }

            System.out.println("NAIVE_BAYES_UPDATABLE and HOEFFDING_TREE are running concurrently");

            if(j!=0 && RUN_UPDATABLE_CLASSIFIER) {
                for(Result ur: incrClassifier.update(sdf1.format(dateStartTraining), sdf1.format(dateEndTraining))) {
                    visualizer.addResult("results\\" +ur.classifier +"_"+ ur.attrSel + "_"+ suffixNameFile, ur);
                    visualizer.printResultAcc(ur);
                }
            }

        }

        Integer sum = 0;
        for(Integer i : numTuples)
            sum += i;
        if(RUN_UPDATABLE_CLASSIFIER)
            System.out.println("\nMean of tuples taken: " + sum/numTuples.size());

        timer.stopTimer();

        System.out.println("\nApplication time: " + timer.getTime()+"s");

    }
}
