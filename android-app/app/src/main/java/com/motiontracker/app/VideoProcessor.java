package com.motiontracker.app;

import android.content.*;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.OpenCVLoader;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.*;

/** Two-pass offline processing, matching the desktop metric/display rules. */
public final class VideoProcessor {
    public interface Progress {void update(int percent,String message);}
    public static final class Output {
        public final Uri video,csv;public final String summary;
        Output(Uri video,Uri csv,String summary) {this.video=video;this.csv=csv;this.summary=summary;}
    }
    private static final int[][] LIMBS={{5,7},{7,9},{6,8},{8,10},{11,13},{13,15},{12,14},{14,16},
        {15,24},{24,20},{20,15},{16,25},{25,21},{21,16}};
    private static final int[][] TORSO={{17,18},{18,19},{18,5},{18,6},{19,11},{19,12}};

    public static Output process(Context context,Uri source,VideoIO.Cancellation cancellation,Progress progress) throws Exception {
        long started=System.nanoTime();
        File job=new File(context.getCacheDir(),"analysis-"+UUID.randomUUID());
        if(!job.mkdirs()) throw new IOException("Cannot create processing directory");
        File input=new File(job,"input"),video=new File(job,"annotated.mp4"),csv=new File(job,"measurements.csv");
        Uri savedVideo=null,savedCsv=null;
        try {
            progress.update(0,"Copying selected video…");
            try(InputStream in=context.getContentResolver().openInputStream(source);OutputStream out=new FileOutputStream(input)) {
                if(in==null) throw new IOException("Cannot read the selected video");
                byte[] buffer=new byte[65536];int count;
                while((count=in.read(buffer))!=-1) {VideoIO.check(cancellation);out.write(buffer,0,count);}
            }
            if(!OpenCVLoader.initLocal()) throw new IOException("OpenCV could not start on this device");
            VideoIO.Info info=VideoIO.inspect(input);
            List<MotionAnalysis.Frame> frames=new ArrayList<>();
            final float[][][] previous={null};
            progress.update(3,"Loading pose models…");
            try(RtmPoseEngine engine=new RtmPoseEngine(context)) {
                VideoIO.decode(input,info,cancellation,(image,time,index)-> {
                    float[][] raw=engine.estimate(image);
                    MotionAnalysis.Frame frame=MotionAnalysis.smooth(time,raw,previous[0]);
                    frames.add(frame);
                    if(frame.complete) previous[0]=frame.points;
                    if(index%5==0) progress.update(5+(int)(60*fraction(time,info.durationUs)),
                        "Analysing frame "+(index+1)+"…");
                });
            }
            double reference=MotionAnalysis.reference(frames);
            if(!Double.isFinite(reference)) throw new IOException("No complete pose was detected. Use a clear full-body view of one lifter.");
            List<Integer> starts=MotionAnalysis.liftStarts(frames,reference,info.fps);
            List<MotionAnalysis.Result> results=MotionAnalysis.analyse(frames,starts,reference);
            progress.update(66,"Rendering annotated video…");
            final MotionAnalysis.Result[] display={null};
            try(VideoIO.Encoder encoder=new VideoIO.Encoder(video,info)) {
                int rendered=VideoIO.decode(input,info,cancellation,(image,time,index)-> {
                    if(index>=frames.size() || time!=frames.get(index).timeUs)
                        throw new IOException("Video timestamps changed between analysis and rendering");
                    MotionAnalysis.Frame frame=frames.get(index);
                    MotionAnalysis.Result result=results.get(index);
                    if(result.complete) display[0]=result;
                    draw(image,frame.points,display[0]);encoder.write(image,time,cancellation);
                    if(index%5==0) progress.update(66+(int)(29.*(index+1)/frames.size()),
                        "Rendering frame "+(index+1)+" of "+frames.size()+"…");
                });
                if(rendered!=frames.size()) throw new IOException("Video frame count changed during rendering");
                encoder.finish(cancellation);
                if(!encoder.isFinished()) throw new IOException("Video encoder did not finish");
            }
            writeCsv(csv,frames,results);
            progress.update(96,"Saving video and measurements…");
            String stamp=new SimpleDateFormat("yyyyMMdd_HHmmss_SSS",Locale.US).format(new Date());
            savedVideo=publish(context,video,"deadlift_"+stamp+".mp4","video/mp4",true,cancellation);
            savedCsv=publish(context,csv,"deadlift_"+stamp+".csv","text/csv",false,cancellation);
            VideoIO.check(cancellation);
            int complete=0;for(MotionAnalysis.Frame f:frames) if(f.complete) complete++;
            double elapsed=(System.nanoTime()-started)/1e9;
            String summary=String.format(Locale.US,"%d frames · %d lifts · %d complete frames · %.0f seconds\nSilent MP4 saved to Movies/MotionTracker; CSV to Download/MotionTracker.",
                frames.size(),starts.size(),complete,elapsed);
            if(starts.isEmpty()) summary+="\nNo lift onset detected; relative spine/knee values are unavailable.";
            return new Output(savedVideo,savedCsv,summary);
        } catch(Exception e) {
            if(savedVideo!=null) context.getContentResolver().delete(savedVideo,null,null);
            if(savedCsv!=null) context.getContentResolver().delete(savedCsv,null,null);
            throw e;
        } finally {
            // Only this job's known temporary files are removed. User videos are untouched.
            input.delete();video.delete();csv.delete();job.delete();
        }
    }
    private static double fraction(long value,long total) {
        return total>0?Math.max(0,Math.min(1,(double)value/total)):0;
    }

    private static Uri publish(Context context,File source,String name,String mime,boolean video,
                               VideoIO.Cancellation cancellation) throws IOException {
        ContentValues values=new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME,name);values.put(MediaStore.MediaColumns.MIME_TYPE,mime);
        values.put(MediaStore.MediaColumns.RELATIVE_PATH,(video?Environment.DIRECTORY_MOVIES:Environment.DIRECTORY_DOWNLOADS)+"/MotionTracker");
        values.put(MediaStore.MediaColumns.IS_PENDING,1);
        Uri uri=context.getContentResolver().insert(video?MediaStore.Video.Media.EXTERNAL_CONTENT_URI:MediaStore.Downloads.EXTERNAL_CONTENT_URI,values);
        if(uri==null) throw new IOException("Cannot create gallery output");
        boolean done=false;
        try {
            try(InputStream in=new FileInputStream(source);OutputStream out=context.getContentResolver().openOutputStream(uri)) {
                if(out==null) throw new IOException("Cannot write gallery output");
                byte[] buffer=new byte[65536];int count;
                while((count=in.read(buffer))!=-1) {VideoIO.check(cancellation);out.write(buffer,0,count);}
            }
            values.clear();values.put(MediaStore.MediaColumns.IS_PENDING,0);
            context.getContentResolver().update(uri,values,null,null);done=true;return uri;
        } finally {if(!done) context.getContentResolver().delete(uri,null,null);}
    }

    private static void draw(Mat image,float[][] points,MotionAnalysis.Result result) {
        double scale=Math.max(.5,image.cols()/1080.);
        int thickness=Math.max(1,(int)Math.round(3*scale)),radius=Math.max(2,(int)Math.round(4*scale));
        for(int[] edge:LIMBS) line(image,points,edge,new Scalar(0,255,0),thickness);
        for(int[] edge:TORSO) line(image,points,edge,new Scalar(0,255,255),thickness);
        for(int i:MotionAnalysis.REQUIRED) if(MotionAnalysis.finite(points[i]))
            Imgproc.circle(image,new Point(points[i][0],points[i][1]),radius,new Scalar(0,0,255),-1,Imgproc.LINE_AA,0);
        ResultOverlay.draw(image,result);
    }
    private static void line(Mat image,float[][] points,int[] edge,Scalar colour,int thickness) {
        if(MotionAnalysis.finite(points[edge[0]]) && MotionAnalysis.finite(points[edge[1]]))
            Imgproc.line(image,new Point(points[edge[0]][0],points[edge[0]][1]),new Point(points[edge[1]][0],points[edge[1]][1]),colour,thickness,Imgproc.LINE_AA,0);
    }
    private static String cell(double value) {return Double.isFinite(value)?Double.toString(value):"";}
    private static void writeCsv(File file,List<MotionAnalysis.Frame> frames,List<MotionAnalysis.Result> results) throws IOException {
        try(Writer out=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file),StandardCharsets.UTF_8))) {
            out.write("frame,time_s,side,complete,lift_number,lift_start_time_s,hip_angle_deg,min_hip_angle_deg,knee_angle_deg,ankle_angle_deg,knee_change_deg,ankle_change_deg,timing_baseline_time_s,knee_onset_time_s,ankle_onset_time_s,ankle_minus_knee_onset_s,knee_change_before_ankle_deg,spine_length_px,spine_reference_length_px,spine_curvature_estimate_deg,spine_lift_baseline_deg,spine_curvature_change_deg\n");
            for(int i=0;i<frames.size();i++) {
                MotionAnalysis.Result result=results.get(i);
                for(int side=0;side<2;side++) {
                    MotionAnalysis.Side r=result.sides[side];
                    out.write(i+","+cell(result.timeUs/1e6)+","+(side==0?"left":"right")+","+(result.complete?1:0)+","+(result.liftNumber>0?result.liftNumber:"")+",");
                    double[] values={result.liftStart,r.hip,r.minHip,r.knee,r.ankle,r.kneeChange,r.ankleChange,r.timingBaseline,r.kneeOnset,r.ankleOnset,r.ankleOnset-r.kneeOnset,r.kneeBeforeAnkle,
                        result.spineLength,result.reference,result.absoluteCurvature,result.spineBaseline,result.spineChange};
                    for(int j=0;j<values.length;j++) {if(j>0) out.write(',');out.write(cell(values[j]));}
                    out.write('\n');
                }
            }
        }
    }
    private VideoProcessor() {}
}
