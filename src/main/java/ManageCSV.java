import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang3.time.DateUtils;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

public class ManageCSV {

    String header = "ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight, Duration, Weekday, Hour\n";
    CSVReader reader = new CSVReader(new FileReader("AccidentListTot.csv"));
    int[] countSeverity = new int[4];
    int countTuples =0;

    ArrayList<String[]> list;

    public ManageCSV() throws FileNotFoundException {
    }

    private void initializeCountSeverity(){
        countSeverity[0]=0;
        countSeverity[1]=0;
        countSeverity[2]=0;
        countSeverity[3]=0;
    }

    public ArrayList<String[]> getTuples(Date dateStart, int granularity) throws IOException {
        String header = "ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight, Duration, Weekday, Hour\n";

        FileWriter write = new FileWriter("temple.csv");
        CSVReader reader = new CSVReader(new FileReader("AccidentListTot.csv"));
        CSVWriter csvWriter = new CSVWriter(write);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        Date date = new Date();
        Date dateEnd = DateUtils.addMonths(dateStart, granularity);
        String[] nextLine;

        write.write(header);
        int count=0;
        nextLine = reader.readNext();
        initializeCountSeverity();
        while ((nextLine = reader.readNext()) != null) {

            if (count % 100000 == 0)
                System.out.println(count);

            count++;

            try {
                date = sdf.parse(nextLine[2]);
            } catch (ParseException e) {
                e.printStackTrace();
            }
            if (date.getTime() >= dateStart.getTime() && date.getTime() <= dateEnd.getTime()){

                int severity = Integer.valueOf(nextLine[1]);
                if (severity <=4 && severity>=1)
                    countSeverity[severity-1]++;

                nextLine = checkValues(nextLine);

                csvWriter.writeNext(nextLine);
            }
        }
        countTuples = count-1;
        return list;
    }

    private String[] checkValues(String[] nextLine) {
        Double temperature;
        Double pressure;
        Double visibility;
        Double windSpeed;


        try {
            temperature = Double.valueOf(nextLine[21]);
            if(temperature<-130 && temperature>130)
                nextLine[21] = "nan";

        } catch (Exception e) {
            nextLine[21] = "nan";
        }

        try {
            pressure = Double.valueOf(nextLine[24]);
            if(pressure<25 && pressure>32.06)
                nextLine[21] = "nan";
        } catch (Exception e) {
            nextLine[24] = "nan";
        }

        try {
            visibility = Double.valueOf(nextLine[25]);
            if(visibility<0 && visibility>10.01)
                nextLine[21] = "nan";

        } catch (Exception e) {
            nextLine[25] = "nan";
        }

        try {
            windSpeed = Double.valueOf(nextLine[27]);
            if(windSpeed<0 && windSpeed>254)
                nextLine[21] = "nan";
        } catch (Exception e) {
            nextLine[27] = "nan";
        }

        return nextLine;
    }


}
