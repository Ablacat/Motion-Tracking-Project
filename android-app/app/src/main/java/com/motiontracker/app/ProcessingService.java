package com.motiontracker.app;

import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.*;
import android.util.Log;
import java.util.concurrent.*;

/** Foreground job: survives activity rotation/backgrounding and offers cancellation. */
public final class ProcessingService extends Service {
    public static final String PROCESS="com.motiontracker.PROCESS",CANCEL="com.motiontracker.CANCEL";
    public static final String UPDATE="com.motiontracker.UPDATE",PREFS="processing";
    private static final String CHANNEL="video-analysis";
    private static final int NOTIFICATION=100;
    public static volatile boolean running=false;
    private volatile boolean cancelled=false;
    private final ExecutorService executor=Executors.newSingleThreadExecutor();
    private long lastProgress;

    @Override public void onCreate() {
        super.onCreate();
        getSystemService(NotificationManager.class).createNotificationChannel(
            new NotificationChannel(CHANNEL,"Video analysis",NotificationManager.IMPORTANCE_LOW));
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId) {
        if(intent==null) {stopSelf();return START_NOT_STICKY;}
        if(CANCEL.equals(intent.getAction())) {
            cancelled=true;
            if(!running) stopSelf();
            return START_NOT_STICKY;
        }
        if(!PROCESS.equals(intent.getAction()) || running) return START_NOT_STICKY;
        Uri source=intent.getData();
        if(source==null) {stopSelf();return START_NOT_STICKY;}
        running=true;cancelled=false;
        startForeground(NOTIFICATION,notification(0,"Starting analysis…",true));
        update(0,"Starting analysis…",true);
        executor.execute(()-> {
            PowerManager.WakeLock wakeLock=getSystemService(PowerManager.class).newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,"MotionTracker:analysis");
            wakeLock.acquire(4*60*60*1000L);
            String message;boolean success=false;
            try {
                VideoProcessor.Output output=VideoProcessor.process(this,source,()->cancelled,(percent,text)-> {
                    long now=SystemClock.elapsedRealtime();
                    if(now-lastProgress>=250 || percent>=96) {lastProgress=now;update(percent,text,true);}
                });
                getSharedPreferences(PREFS,MODE_PRIVATE).edit().putString("video",output.video.toString())
                    .putString("csv",output.csv.toString()).apply();
                message=output.summary;success=true;
            } catch(CancellationException e) {message="Analysis cancelled. Your original video is unchanged.";}
            catch(Exception e) {
                Log.e("MotionTracker","Processing failed",e);
                message="Analysis failed: "+(e.getMessage()==null?e.getClass().getSimpleName():e.getMessage());
            } catch(OutOfMemoryError e) {message="Not enough memory to analyse this video. Select a shorter clip.";}
            finally {if(wakeLock.isHeld()) wakeLock.release();}
            running=false;update(success?100:0,message,false);
            stopForeground(STOP_FOREGROUND_REMOVE);
            postNotification(notification(success?100:0,
                success?"Analysis complete. Tap to view results.":message,false));
            stopSelf();
        });
        return START_NOT_STICKY;
    }
    private void update(int percent,String message,boolean active) {
        getSharedPreferences(PREFS,MODE_PRIVATE).edit().putInt("percent",percent)
            .putString("message",message).putBoolean("running",active).apply();
        if(active) postNotification(notification(percent,message,true));
        sendBroadcast(new Intent(UPDATE).setPackage(getPackageName()));
    }
    private void postNotification(Notification notification) {
        if(Build.VERSION.SDK_INT<33 || checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)==PackageManager.PERMISSION_GRANTED)
            getSystemService(NotificationManager.class).notify(NOTIFICATION,notification);
    }
    private Notification notification(int percent,String text,boolean active) {
        PendingIntent open=PendingIntent.getActivity(this,0,new Intent(this,MainActivity.class),
            PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder=new Notification.Builder(this,CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_upload).setContentTitle("Motion Tracker")
            .setContentText(text).setContentIntent(open).setOngoing(active).setOnlyAlertOnce(true)
            .setStyle(new Notification.BigTextStyle().bigText(text));
        if(active) {
            PendingIntent cancel=PendingIntent.getService(this,1,new Intent(this,ProcessingService.class).setAction(CANCEL),
                PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE);
            builder.setProgress(100,percent,false).addAction(new Notification.Action.Builder(
                android.R.drawable.ic_menu_close_clear_cancel,"Cancel",cancel).build());
        } else builder.setAutoCancel(true);
        return builder.build();
    }
    @Override public void onDestroy() {cancelled=true;executor.shutdownNow();super.onDestroy();}
    @Override public IBinder onBind(Intent intent) {return null;}
}
