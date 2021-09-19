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
    /*************/
    private static double PERCENTAGESPLIT = 66.0;
    private final static int randomSeed = (int)System.currentTimeMillis();
    private final static int DRIFT =1;
    private final static int NUM_ITERATION = 5;
    private final static String dateString = "2018-02-01 00:00:00";

    private static boolean CROSS_VALIDATION = false;
    private static int GRANULARITY = 10;
    /*************/

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
        if(CROSS_VALIDATION)
            PERCENTAGESPLIT = 100.0;
        Timer timer = new Timer();
        timer.startTimer();
        ManageCSV manager = new ManageCSV();

        String nameFile = dateString.split(" ")[0] + "_" + String.valueOf(NUM_ITERATION)+ "_DR" + String.valueOf(DRIFT) + "_GR" + String.valueOf(GRANULARITY) + (CROSS_VALIDATION?"_CR":"" ) + ".txt";
        Visualizer visualizer = new Visualizer("results\\" +nameFile);
        List<String> attrNames = new ArrayList<>();
        attrNames.add("cfs_BestFirst");
        attrNames.add("cfs_GreedyStepWise");
        attrNames.add("InfoGain_Ranker");


        Date dateStart = sdf.parse(dateString);
        Date dateEnd, dateBegin = sdf.parse(dateString);

        manager.setGranularity(GRANULARITY);
        //int lastGranularity= manager.getGranularity();

        for (int j= 0; j<NUM_ITERATION; j++) {
            System.out.println("==========================================");
            System.out.println("Num Iteration: " + Integer.valueOf(j+1) + "/" + Integer.valueOf(NUM_ITERATION));
            //lastGranularity= manager.getGranularity();

            dateStart = DateUtils.addWeeks(sdf.parse(dateString), DRIFT*j);

            System.out.println("==========================================");
            System.out.println("===> Start Reading");
            dateEnd = manager.getTuplesFromDB(dateStart);
            manager.writeCSV("temple.csv");
            manager.printCoutnSeverity();
            manager.reduceList();
            manager.printCoutnSeverity();
            manager.writeCSV("templeReduced.csv");
            //manager.saveARFF(new File("templeReduced.csv"));

            numTuples.add(manager.getCountTuples());

            List<Instances> dataNotFiltered = loadDataSplitTrainTest(PERCENTAGESPLIT);

            int[] numInstancesSeverity = manager.getCountSeverity();
            List<Instances> dataFiltered = Preprocessor.filter(dataNotFiltered.get(0), dataNotFiltered.get(1), numInstancesSeverity[3]);



            System.out.println("------------------------------------");
            System.out.println("===> Start Classifying");

            List<String> classifiersNames = new ArrayList<>();
            List<String> attrSelectionNames = new ArrayList<>();
            /** 1 Classifier **/
            classifiersNames.add("J48");
            attrSelectionNames.add("CFS_BESTFIRST");
            /** 2 Classifier **/
            classifiersNames.add("RANDOM_FOREST");
            attrSelectionNames.add("CFS_GREEDYSTEPWISE");

            if(classifiersNames.size() != attrSelectionNames.size()){
                System.err.println("Error: classifier definition is wrong!");
                System.exit(1);
            }

            for(int i=0; i < classifiersNames.size(); i++){
                AttrSelectedClassifier classifier = new AttrSelectedClassifier(dataFiltered,CROSS_VALIDATION,
                                                    sdf1.format(dateStart), sdf1.format(dateEnd));
                System.out.println(classifiersNames.get(i)+" is running");
                Result r = classifier.start(attrSelectionNames.get(i),null, null,
                                            classifiersNames.get(i),null);

                //visualizer.addResult(r);
                visualizer.printResultAcc(r);
            }
        }

        Integer sum = 0;
        for(Integer i : numTuples)
            sum += i;

        System.out.println("\nMean of tuples taken: " + sum/numTuples.size());

        timer.stopTimer();
        NumberFormat formatter = new DecimalFormat("##");
        /*
        Double totalSecs =  Double.parseDouble(timer.getTime());
        Double minutes = (totalSecs % 3600) / 60;
        Double seconds = totalSecs % 60;


        String timeString = String.format("%02d:%02d", minutes, seconds);
        System.out.println("\nApplication time: " + timeString);
        */

        System.out.println("\nApplication time: " + timer.getTime()+"s");

    }
}
