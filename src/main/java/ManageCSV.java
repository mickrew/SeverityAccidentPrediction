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
    private final static Double percentageSeverity4 = 1.0;
    private final static Double percentageSeverity2 = 0.8;
    private final static Double percentageSeverity1 = 0.7;
    private final static Double percentageSeverity3 = 0.8;
    private final static int MAX_SEVERITY4 = 8000;
    private int THRESHOLD = 75000;
    private int granularity = 4;
    private Date startDataset;


    private ArrayList<String[]> list = new ArrayList<>();

    public ManageCSV(Date dastestart) throws FileNotFoundException {
        startDataset=dastestart;
    }

    private void initializeCountSeverity(){
        countSeverity[0]=0; //Severity 1
        countSeverity[1]=0; //Severity 2
        countSeverity[2]=0; //Severity 3
        countSeverity[3]=0; //Severity 4
    }


    public void  reduceList(){

        Collections.shuffle(list);

        Iterator itr = list.iterator();
        String[] tmp;

        if (countSeverity[3]>MAX_SEVERITY4){
            while(itr.hasNext()){
                tmp = (String[]) itr.next();
                if (tmp[1].contains("4")){
                    if (countSeverity[3]>MAX_SEVERITY4){
                        itr.remove();
                        countSeverity[3]--;
                    } else {
                        break;
                    }
                }
            }
        }

        itr = list.iterator();

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
        System.out.println("Reduced to\t" + list.size() + " tuples");
        countTuples = list.size();
    }

    public Date getTuplesFromDB(Date dateStartTraining, boolean fixedGranularity, Date dateEndTraining, Date dateLimit) throws IOException, ParseException {

        list.clear();

        FileWriter write = new FileWriter("temple.csv");
        CSVWriter csvWriter = new CSVWriter(write);
        write.write(header);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        SimpleDateFormat sdf1 = new SimpleDateFormat("yyyy-MM-dd");

        Date tmp = dateStartTraining;

        ArrayList<String> nameFiles = new ArrayList<>();

        for(int i =0; tmp.getTime()<=dateEndTraining.getTime(); i++){
            tmp = DateUtils.addMonths(dateStartTraining, i);
            if (tmp.getTime()>dateLimit.getTime())
                break;
            String month = String.valueOf(tmp.getMonth()+1);
            String nameFile = String.valueOf("data\\" + (tmp.getYear()+1900) + "-" + month + ".csv");
            nameFiles.add(nameFile);
        }

        CSVReader reader = null;
        Date date = new Date();

        String[] nextLine;
        initializeCountSeverity();

        int count=0;
        boolean check=false;

        System.out.println("Granularity: " + granularity);


        for(int i = 0; i< nameFiles.size(); i++) {
            reader = new CSVReader(new FileReader(nameFiles.get(i)));
            nextLine = reader.readNext(); //read the header from file
            while ((nextLine = reader.readNext()) != null) {

                //if (count % 10000 == 0) System.out.println(count);

                count++;

                try {
                    date = sdf.parse(nextLine[2]);
                } catch (ParseException e) {
                    e.printStackTrace();
                }
                if (date.getTime() >= dateStartTraining.getTime() && date.getTime() <= dateEndTraining.getTime()) {

                    int severity = Integer.valueOf(nextLine[1]);
                    if (severity <= 4 && severity >= 1)
                        countSeverity[severity - 1]++;
                    nextLine = checkValues(nextLine);
                    this.list.add(nextLine);
                    csvWriter.writeNext(nextLine);
                }
            }

        }
        if(!fixedGranularity) {
            if (list.size() > 1.5 * THRESHOLD) {

                granularity -= 1;// (int) (0.5 * granularity);
                Date dateLastEnd = dateEndTraining;
                dateEndTraining = DateUtils.addWeeks(dateStartTraining, granularity);

                System.out.println("reduce granularity: " + list.size() + " is over  " + 1.5 * THRESHOLD + "\tnew value = " + granularity);
                //System.out.println("Change dateEnd from "+sdf.format(dateLastEnd)+" to " + sdf.format(dateEnd));
                //this.list.clear();
                //count =0;
                //the granularity cannot go down 1

                if (granularity < 1)
                    granularity = 1;
                check = true;
            } else if (list.size() < 0.5 * THRESHOLD) {
                granularity += 1;//(int) (1.5 * granularity);
                System.out.println("New granularity: " + granularity);
                check = true;
            }
        }


        if (reader!=null)
            reader.close();



        System.out.println("Range dates from " + sdf1.format(dateStartTraining) + " to " + sdf1.format(dateEndTraining));
        //System.out.println("Read\t" + count  + " tuples");
        System.out.println("Extracted\t" + list.size() + " tuples");

        //System.out.println("Saving files ... ... ... ");
        write.flush();
        write.close();
        //csvWriter.close();
        countTuples = list.size();

        Timer t = new Timer();
        t.startTimer();



        t.stopTimer();
        //t.printTimer();

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

        return dateEndTraining;
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
        System.out.println("File" + name +" saved ");
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

    public void printCoutnSeverity(){
        for (int i=0; i< countSeverity.length; i++){
            System.out.println("Severity" + Integer.valueOf(i+1) +": " + countSeverity[i]);
        }
    }

    public void deleteFile(String nameFile){
        File f = new File(nameFile);
        if (f.exists()) {
            f.delete();
        }
    }

    public int getGranularity(){
        return granularity;
    }

    public void setGranularity(int granularity) {
        this.granularity = granularity;
    }
}
