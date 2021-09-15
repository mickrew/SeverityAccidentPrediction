import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

public class Driver {
    public static void main(String[] args) throws Exception {
        ManageCSV manager = new ManageCSV();

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        String dateString = "2016-06-30 12:00:00";
        Date date = sdf.parse(dateString);

        manager.getTuples(date, 3);
        manager.reduceList();
        manager.writeCSV("templeReduced.csv");


        double trainPercentage = 66.0;
        int randomSeed = 1;

        DataSource source = new DataSource("templeReduced.csv");

        final Instances dataSet = source.getDataSet();
        dataSet.randomize(new Random(randomSeed));

        int trainSize = (int)Math.round(dataSet.numInstances() * trainPercentage / 100);
        int testSize = dataSet.numInstances() - trainSize;

        Instances train = new Instances(dataSet, 0, trainSize);
        Instances test = new Instances(dataSet, trainSize, testSize);
        int[] numInstancesSeverity4 = manager.getCountSeverity();

        List<Instances> data = Preprocessor.filter(train, test, numInstancesSeverity4[3]);



        System.out.println("fine");

    }
}
