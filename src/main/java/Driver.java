import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class Driver {
    public static void main(String[] args) throws IOException, ParseException {
        ManageCSV manager = new ManageCSV();

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        String dateString = "2016-06-30 12:00:00";
        Date date = sdf.parse(dateString);

        manager.getTuples(date, 3);
        manager.reduceList();
        manager.writeCSV("templeReduced.csv");
        System.out.println("fine");

    }
}
