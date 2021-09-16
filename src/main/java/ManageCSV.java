import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang3.time.DateUtils;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class ManageCSV {

    private String header = "ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight, Duration, Weekday, Hour\n";

    private int[] countSeverity = new int[4];
    private int countTuples =0;
    private Double percentageSeverity4 = 1.0;
    private Double percentageSeverity2 = 0.7;
    private Double percentageSeverity1 = 0.7;
    private Double percentageSeverity3 = 0.8;

    ArrayList<String[]> list = new ArrayList<>();

    public ManageCSV() throws FileNotFoundException {
    }

    private void initializeCountSeverity(){
        countSeverity[0]=0; //Severity 1
        countSeverity[1]=0; //Severity 2
        countSeverity[2]=0; //Severity 3
        countSeverity[3]=0; //Severity 4
    }


    public void reduceList(){

        //int[] tmpCountSeverity = countSeverity;
        Iterator itr = list.iterator();
        String[] tmp;
        while(itr.hasNext()){
           tmp = (String[]) itr.next();
           if (tmp[1].contains("2")){
               if (countSeverity[1]>=percentageSeverity2*countSeverity[3]){
                   itr.remove();
                   countSeverity[1]--;
                   continue;
               }
           }
            if (tmp[1].contains("3")){
                if (countSeverity[2]>=percentageSeverity3*countSeverity[3]){
                    itr.remove();
                    countSeverity[2]--;
                    continue;
                }
            }
            if (tmp[1].contains("1")){
                if (countSeverity[0]>=percentageSeverity1*countSeverity[3]){
                    itr.remove();
                    countSeverity[0]--;
                    continue;
                }
            }
        }

    }

    public void getTuplesFromDB(Date dateStart, int granularity) throws IOException {
        FileWriter write = new FileWriter("temple.csv");
        CSVWriter csvWriter = new CSVWriter(write);
        write.write(header);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        Date dateEnd = DateUtils.addMonths(dateStart, granularity);

        ArrayList<String> nameFiles = new ArrayList<>();
        for(int i =0; i<= granularity; i++){
            Date tmp = DateUtils.addMonths(dateStart, i);
            String nameFile = String.valueOf("data\\" + (tmp.getYear()+1900) + "-" + String.valueOf(tmp.getMonth()) + ".csv");
            nameFiles.add(nameFile);
        }

        CSVReader reader;
        Date date = new Date();

        String[] nextLine;
        initializeCountSeverity();

        int count=0;

        for(int i = 0; i<= granularity; i++){
            reader = new CSVReader(new FileReader(nameFiles.get(i)));
            nextLine = reader.readNext(); //read the header from file
            while ((nextLine = reader.readNext()) != null) {

                if (count % 10000 == 0)
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
                    this.list.add(nextLine);
                    csvWriter.writeNext(nextLine);
                }
            }
            reader.close();

        }
        System.out.println(count);
        System.out.println("Saving files ... ... ... ");
        write.flush();
        write.close();
        //csvWriter.close();
        countTuples = count-1;

        Timer t = new Timer();
        t.startTimer();

        CSVLoader source = new CSVLoader();
        source.setSource(new File("temple.csv"));
        System.out.println("File .csv saved ");

        t.stopTimer();
        t.printTimer();

        /*
        t.startTimer();

        ArffSaver saver = new ArffSaver();

        Instances dataSet = source.getDataSet();

        saver.setInstances(dataSet);
        saver.setFile(new File("temple.arff"));
        System.out.println("File .arff saved ");
        saver.writeBatch();
        t.stopTimer();
        t.printTimer();
        */

    }

    public void getTuples(Date dateStart, int granularity) throws IOException {
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
                this.list.add(nextLine);
                csvWriter.writeNext(nextLine);
            }
        }
        write.flush();
        write.close();
        this.list=list;
        countTuples = count-1;
        //return list;
    }

    private String[] checkValues(String[] nextLine) {
        Double temperature;
        Double pressure;
        Double visibility;
        Double windSpeed;


        try {
            temperature = Double.valueOf(nextLine[21]);
            if(temperature<-130 && temperature>130)
                nextLine[21] = String.valueOf(Double.NaN);

        } catch (Exception e) {
            nextLine[21] = String.valueOf(Double.NaN);
        }

        try {
            pressure = Double.valueOf(nextLine[24]);
            if(pressure<25 && pressure>32.06)
                nextLine[21] = String.valueOf(Double.NaN);
        } catch (Exception e) {
            nextLine[24] = String.valueOf(Double.NaN);
        }

        try {
            visibility = Double.valueOf(nextLine[25]);
            if(visibility<0 && visibility>10.01)
                nextLine[21] = String.valueOf(Double.NaN);

        } catch (Exception e) {
            nextLine[25] = String.valueOf(Double.NaN);
        }

        try {
            windSpeed = Double.valueOf(nextLine[27]);
            if(windSpeed<0 && windSpeed>254)
                nextLine[21] = String.valueOf(Double.NaN);
        } catch (Exception e) {
            nextLine[27] = String.valueOf(Double.NaN);
        }

        return nextLine;
    }


    //save in csv the current list
    public void writeCSV(String name) throws IOException {
        FileWriter write = new FileWriter(name);
        write.write(header);
        CSVWriter csvWriter = new CSVWriter(write);
        Iterator itr = list.iterator();
        String[] tmp;
        while(itr.hasNext()){
            csvWriter.writeNext((String[]) itr.next());
        }
        csvWriter.close();
        write.close();

    }

    public void saveARFF(File file) throws IOException {

        CSVLoader source = new CSVLoader();
        source.setSource(file);
        ArffSaver saver = new ArffSaver();

        Instances dataSet = source.getDataSet();

        saver.setInstances(dataSet);
        saver.setFile(new File("temple.arff"));
        saver.writeBatch();
    }





    public int getCountTuples() {
        return countTuples;
    }

    public String getHeader() {
        return header;
    }

    public ArrayList<String[]> getList() {
        return list;
    }

    public int[] getCountSeverity() {
        return countSeverity;
    }
}
