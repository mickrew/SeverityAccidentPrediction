import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;

public class SplitMonth {

    public static int yearIndex(String year){
        if (year.contains("2016")){
            return 0;
        }
        if (year.contains("2017")){
            return 1;
        }
        if (year.contains("2018")){
            return 2;
        }
        if (year.contains("2019")){
            return 3;
        }
        if (year.contains("2020")){
            return 4;
        }
        return 0;
    }



    public static void prova(String[] args) throws IOException {

        String header = "ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight, Duration, Weekday, Hour";
        CSVReader reader = new CSVReader(new FileReader("AccidentListTot.csv"));
        FileWriter write = new FileWriter("temple.csv");
        CSVWriter csvWriter = new CSVWriter(write);

        String date = "";
        String year;
        String month;

        int count=0;
        String[] nextLine = reader.readNext();
        ArrayList<String[]> list = new ArrayList<>();

        for(int i=0; i<12;i++){
            list.add(new String[]{""});
        }



        while ((nextLine = reader.readNext()) != null) {

            if (count % 100000 == 0)
                System.out.println(count);

            count++;

            date = nextLine[2];

            //year = date.split("-")[0];
            month = date.split("-")[1];
            //Integer yearInt = yearIndex(year);
            Integer monthInt = Integer.valueOf(month);



        }

    }
}
