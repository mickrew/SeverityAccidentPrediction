import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class SplitMonth1 {
    public static void prova(String[] args) throws IOException {

        ArrayList<FileWriter> listWriter = new ArrayList<>();
        ArrayList<CSVWriter> listCSVWriter = new ArrayList<>();
        String header = "ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight, Duration, Weekday, Hour\n";
        CSVReader reader = new CSVReader(new FileReader("AccidentList2020.csv"));

        for(int i =1; i<=12 ; i++){
            FileWriter write = new FileWriter("2020-"+String.valueOf(i)+".csv");
            write.write(header);
            listWriter.add(write);

            CSVWriter csvWriter = new CSVWriter(write);
            listCSVWriter.add(csvWriter);
        }

        String[] nextLine = reader.readNext();
        int count = 0;
        String date = "";
        String month;

        while ((nextLine = reader.readNext()) != null) {

            if (count%10000==0)
                System.out.println(count);

            count++;

            date = nextLine[2];

            month = date.split("-")[1];
            Integer monthInt = Integer.valueOf(month);
            listCSVWriter.get(monthInt-1).writeNext(nextLine);
        }

        for(int i=0; i<12 ; i++){
            listWriter.get(i).flush();
            listWriter.get(i).close();

            //listCSVWriter.get(i).flush();
            //listCSVWriter.get(i).close();
        }
        for(int i=1; i<=12 ; i++){
            CSVLoader source = new CSVLoader();
            source.setSource(new File("2020-"+String.valueOf(i)+".csv"));
            ArffSaver saver = new ArffSaver();

            Instances dataSet = source.getDataSet();

            saver.setInstances(dataSet);
            saver.setFile(new File("2020-"+String.valueOf(i)+".arff"));
            saver.writeBatch();
        }

    }
}
