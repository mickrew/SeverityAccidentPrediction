import java.util.Date;

public class Timer {
    private Date dateTimeStart;
    long start= 0L;
    private Date dateTimeEnd;
    long end= 0L;
    private Date datePauseTimer;
    long pause= 0L;
    private Date dateResumeTimer;
    long resume= 0L;
    private long timePaused = 0L;


    public Timer(){

    }

    public void startTimer (){
        dateTimeStart = new Date();
        start = dateTimeStart.getTime();
    }

    public void stopTimer(){
        dateTimeEnd = new Date();
        end = dateTimeEnd.getTime();
    }

    public void printTimer(){
        double timePassed = (double) (end-start-timePaused)/1.0;
        System.out.println("Timer: "+  String.valueOf((timePassed)/1000));
    }

    public void pauseTimer(){
        datePauseTimer = new Date();
        pause = datePauseTimer.getTime();
    }

    public void resumeTimer(){
        dateResumeTimer = new Date();
        resume = dateResumeTimer.getTime();
        timePaused += resume-pause;
    }

    public String getTime() {
        double timePassed = (double) (end-start-timePaused)/1.0;
        return String.valueOf((timePassed)/1000);
    }
}
