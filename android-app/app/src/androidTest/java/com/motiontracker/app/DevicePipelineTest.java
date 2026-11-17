package com.motiontracker.app;

import android.content.Context;
import android.content.ContentValues;
import android.content.Intent;
import android.media.MediaFormat;
import android.net.Uri;
import android.provider.MediaStore;
import android.widget.VideoView;
import androidx.lifecycle.Lifecycle;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.json.*;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.*;
import java.util.*;
import static org.junit.Assert.*;

/** Run on the S23 (or an ARM64 emulator): validates native inference and codecs. */
@RunWith(AndroidJUnit4.class)
public class DevicePipelineTest {
    private static byte[] read(InputStream in) throws IOException {
        ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] b=new byte[65536];int n;
        while((n=in.read(b))!=-1) out.write(b,0,n);return out.toByteArray();
    }
    @Test public void nativeModelsMatchPythonKeypoints() throws Exception {
        assertTrue(OpenCVLoader.initLocal());
        Context target=InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context tests=InstrumentationRegistry.getInstrumentation().getContext();
        File image=new File(target.getCacheDir(),"pose-fixture.png");JSONObject expected;
        try(InputStream in=tests.getAssets().open("pose-fixture.png");OutputStream out=new FileOutputStream(image)) {out.write(read(in));}
        try(InputStream in=tests.getAssets().open("pose-fixture.json")) {expected=new JSONObject(new String(read(in),java.nio.charset.StandardCharsets.UTF_8));}
        Mat bgr=Imgcodecs.imread(image.getPath());assertFalse(bgr.empty());
        try(RtmPoseEngine engine=new RtmPoseEngine(target)) {
            float[] box=new float[4];for(int i=0;i<4;i++) box[i]=(float)expected.getJSONArray("bbox").getDouble(i);
            RtmPoseEngine.Pose pose=engine.inferPose(bgr,box);
            JSONArray points=expected.getJSONArray("points"),scores=expected.getJSONArray("scores");
            for(int joint:MotionAnalysis.REQUIRED) {
                assertEquals(scores.getDouble(joint),pose.scores[joint],.03);
                assertEquals(points.getJSONArray(joint).getDouble(0),pose.xy[joint][0],3);
                assertEquals(points.getJSONArray(joint).getDouble(1),pose.xy[joint][1],3);
            }
            float[][] detected=engine.estimate(bgr);
            assertTrue("Detector/pose pipeline must find every required point in the fixture",MotionAnalysis.ready(detected));
        } finally {bgr.release();image.delete();}
    }
    @Test public void codecRoundTripKeepsFramesAndPresentationTimes() throws Exception {
        assertTrue(OpenCVLoader.initLocal());
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        File file=new File(context.getCacheDir(),"codec-test.mp4");
        MediaFormat format=MediaFormat.createVideoFormat("video/avc",320,240);
        format.setInteger(MediaFormat.KEY_FRAME_RATE,30);format.setLong(MediaFormat.KEY_DURATION,200_000);
        VideoIO.Info info=new VideoIO.Info(format);
        long[] timestamps={0,33_333,70_000,110_000,150_000,190_000};
        Mat green=new Mat(240,320,CvType.CV_8UC3,new Scalar(10,200,20));
        try {
            try(VideoIO.Encoder encoder=new VideoIO.Encoder(file,info)) {
                for(long timestamp:timestamps) encoder.write(green,timestamp,()->false);
                encoder.finish(()->false);assertTrue(encoder.isFinished());
            }
            List<Long> actual=new ArrayList<>();
            int count=VideoIO.decode(file,VideoIO.inspect(file),()->false,(frame,time,index)-> {
                actual.add(time);assertEquals(320,frame.cols());assertEquals(240,frame.rows());
                double[] pixel=frame.get(120,160);assertTrue(pixel[1]>pixel[0]+100 && pixel[1]>pixel[2]+100);
            });
            assertEquals(timestamps.length,count);
            for(int i=0;i<count;i++) assertEquals(timestamps[i],actual.get(i).longValue());
            format.setInteger(MediaFormat.KEY_ROTATION,90);
            VideoIO.decode(file,new VideoIO.Info(format),()->false,(frame,time,index)-> {
                assertEquals(240,frame.cols());assertEquals(320,frame.rows());
            });
        } finally {green.release();file.delete();}
    }

    @Test public void largeOverlayUsesGreenForNegativeSpineAndRedForLargeResults() throws Exception {
        assertTrue(OpenCVLoader.initLocal());
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context tests=InstrumentationRegistry.getInstrumentation().getContext();
        File fixture=new File(context.getCacheDir(),"overlay-fixture.png");
        try(InputStream in=tests.getAssets().open("pose-fixture.png");OutputStream out=new FileOutputStream(fixture)) {out.write(read(in));}
        Mat portrait=Imgcodecs.imread(fixture.getPath()),landscape=new Mat();
        MotionAnalysis.Result result=new MotionAnalysis.Result(0,true);
        result.spineChange=-12.3;result.sides[0].minHip=95;result.sides[1].minHip=120;
        result.sides[0].kneeBeforeAnkle=20;result.sides[1].kneeBeforeAnkle=31;
        try {
            org.opencv.imgproc.Imgproc.resize(portrait,landscape,new Size(1280,720));
            ResultOverlay.draw(portrait,result);ResultOverlay.draw(landscape,result);
            assertTrue(Imgcodecs.imwrite(new File(context.getExternalFilesDir(null),"overlay-preview.png").getPath(),portrait));
            assertTrue(Imgcodecs.imwrite(new File(context.getExternalFilesDir(null),"overlay-landscape.png").getPath(),landscape));
            // The full negative-spine range must use the exact green accent in native BGR.
            double scale=Math.min(portrait.cols()/720.,portrait.rows()*.30/256.);
            int pad=Math.max(4,(int)Math.round(12*scale)),gap=Math.max(2,(int)Math.round(8*scale));
            int row=Math.max(24,(int)Math.round(72*scale));
            int top=portrait.rows()-(3*row+2*gap+2*pad);
            double[] colour=portrait.get(top+pad+row/2,pad+1);
            assertArrayEquals(new double[]{153,211,52},colour,0);
        } finally {portrait.release();landscape.release();fixture.delete();}
    }

    @Test public void previewLoopsAcrossEndsUntilStopIsPressed() throws Exception {
        assertTrue(OpenCVLoader.initLocal());
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        File file=new File(context.getCacheDir(),"loop-preview-test.mp4");
        MediaFormat format=MediaFormat.createVideoFormat("video/avc",320,240);
        format.setInteger(MediaFormat.KEY_FRAME_RATE,30);format.setLong(MediaFormat.KEY_DURATION,400_000);
        Mat frame=new Mat(240,320,CvType.CV_8UC3,new Scalar(20,100,180));
        Uri uri=null;
        try {
            try(VideoIO.Encoder encoder=new VideoIO.Encoder(file,new VideoIO.Info(format))) {
                for(int i=0;i<12;i++) encoder.write(frame,i*33_333L,()->false);
                encoder.finish(()->false);
            }
            ContentValues values=new ContentValues();
            values.put(MediaStore.Video.Media.DISPLAY_NAME,"motion-tracker-loop-test.mp4");
            values.put(MediaStore.Video.Media.MIME_TYPE,"video/mp4");
            uri=context.getContentResolver().insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI,values);
            assertNotNull(uri);
            try(InputStream in=new FileInputStream(file);OutputStream out=context.getContentResolver().openOutputStream(uri)) {
                assertNotNull(out);out.write(read(in));
            }
            Intent intent=new Intent(context,PreviewActivity.class).setData(uri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            try(ActivityScenario<PreviewActivity> scenario=ActivityScenario.launch(intent)) {
                int wraps=0,previous=-1;long deadline=System.currentTimeMillis()+15_000;
                boolean[] looping={false};int[] position={-1};
                while(System.currentTimeMillis()<deadline && wraps<2) {
                    scenario.onActivity(activity->{
                        VideoView video=activity.findViewById(R.id.preview_video);
                        looping[0]=activity.isLooping() && video.isPlaying();
                        position[0]=looping[0]?video.getCurrentPosition():-1;
                    });
                    if(looping[0] && previous>=0 && position[0]+30<previous) wraps++;
                    if(position[0]>=0) previous=position[0];
                    Thread.sleep(50);
                }
                assertTrue("Preview should be actively looping",looping[0]);
                assertTrue("Preview should cross the end and restart at least twice",wraps>=2);
                scenario.onActivity(activity->{
                    activity.findViewById(R.id.preview_stop).performClick();
                    assertFalse(((VideoView)activity.findViewById(R.id.preview_video)).isPlaying());
                });
                // finish() stops playback immediately; window/Activity destruction is
                // asynchronous and may wait for the emulator's closing animation.
                long closedDeadline=System.currentTimeMillis()+5_000;
                while(scenario.getState()!=Lifecycle.State.DESTROYED && System.currentTimeMillis()<closedDeadline)
                    Thread.sleep(50);
                assertEquals(Lifecycle.State.DESTROYED,scenario.getState());
            }
        } finally {
            frame.release();file.delete();
            if(uri!=null) context.getContentResolver().delete(uri,null,null);
        }
    }
}
