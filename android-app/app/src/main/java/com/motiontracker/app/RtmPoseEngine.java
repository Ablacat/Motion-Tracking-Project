package com.motiontracker.app;

import android.content.Context;
import org.json.JSONObject;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import ai.onnxruntime.*;
import java.io.*;
import java.nio.FloatBuffer;
import java.security.MessageDigest;
import java.util.*;

/** Native-Android port of RTMLib's YOLOX + balanced Halpe26 preprocessing/decoding. */
public final class RtmPoseEngine implements AutoCloseable {
    private final OrtEnvironment environment=OrtEnvironment.getEnvironment();
    private final OrtSession detector,pose;
    private final float[] detectorInput=new float[3*640*640],poseInput=new float[3*192*256];
    private final byte[] detectorPixels=new byte[3*640*640],posePixels=new byte[3*192*256];
    private float[] trackedBox;
    private int frameNumber;
    public int detectorRuns;

    public RtmPoseEngine(Context context) throws Exception {
        JSONObject manifest;
        try(InputStream in=context.getAssets().open("models/models.json")) {
            manifest=new JSONObject(new String(readAll(in),java.nio.charset.StandardCharsets.UTF_8));
        }
        try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(2);
            options.setInterOpNumThreads(1);
            options.addConfigEntry("session.intra_op.allow_spinning","0");
            options.addConfigEntry("session.inter_op.allow_spinning","0");
            detector=environment.createSession(modelFile(context,"detector.onnx",manifest).getPath(),options);
            OrtSession loadedPose;
            try { loadedPose=environment.createSession(modelFile(context,"pose.onnx",manifest).getPath(),options); }
            catch(Exception e) {detector.close();throw e;}
            pose=loadedPose;
        }
    }

    private static File modelFile(Context context,String name,JSONObject manifest) throws Exception {
        JSONObject entry=manifest.getJSONObject(name);
        String hash=entry.getString("sha256");long size=entry.getLong("bytes");
        File dir=new File(context.getFilesDir(),"models");
        if(!dir.isDirectory() && !dir.mkdirs()) throw new IOException("Cannot create model directory");
        File target=new File(dir,name),stamp=new File(dir,name+".sha256");
        if(target.length()==size && stamp.isFile()) {
            try(InputStream in=new FileInputStream(stamp)) {
                if(new String(readAll(in),java.nio.charset.StandardCharsets.UTF_8).equals(hash)) return target;
            }
        }
        File temporary=new File(dir,name+".part");
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream in=context.getAssets().open("models/"+name);OutputStream out=new FileOutputStream(temporary)) {
            byte[] buffer=new byte[65536];int count;
            while((count=in.read(buffer))!=-1) {out.write(buffer,0,count);digest.update(buffer,0,count);}
        }
        StringBuilder actual=new StringBuilder();
        for(byte b:digest.digest()) actual.append(String.format(Locale.US,"%02x",b&255));
        if(temporary.length()!=size || !actual.toString().equals(hash)) {
            temporary.delete();throw new IOException("Bundled model failed verification: "+name);
        }
        java.nio.file.Files.move(temporary.toPath(),target.toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        try(Writer writer=new FileWriter(stamp)) {writer.write(hash);}
        return target;
    }

    public static final class Pose {
        public final float[][] xy;
        public final float[] scores;
        Pose(float[][] xy,float[] scores) {this.xy=xy;this.scores=scores;}
        float[][] accepted(int width,int height) {
            float[][] accepted=MotionAnalysis.emptyPoints();
            for(int i=0;i<26;i++) if(scores[i]>=MotionAnalysis.MIN_SCORE && MotionAnalysis.finite(xy[i])
                    && xy[i][0]>=0 && xy[i][0]<width && xy[i][1]>=0 && xy[i][1]<height)
                accepted[i]=xy[i].clone();
            return accepted;
        }
    }

    private boolean requiredPresent(Pose value,int width,int height) {
        if(value==null) return false;
        float[][] accepted=value.accepted(width,height);
        for(int joint:MotionAnalysis.REQUIRED) if(accepted[joint]==null) return false;
        return true;
    }

    /** Detector crops are reused; all pose landmarks are inferred on every frame. */
    public float[][] estimate(Mat image) throws Exception {
        boolean detect=trackedBox==null || frameNumber%10==0;
        Pose selected=detect?detectAndPose(image):inferPose(image,trackedBox);
        if(!detect && !requiredPresent(selected,image.cols(),image.rows())) selected=detectAndPose(image);
        trackedBox=null;
        if(requiredPresent(selected,image.cols(),image.rows())) {
            float x1=Float.POSITIVE_INFINITY,y1=x1,x2=Float.NEGATIVE_INFINITY,y2=x2;
            for(int j=0;j<26;j++) if(selected.scores[j]>=MotionAnalysis.MIN_SCORE && MotionAnalysis.finite(selected.xy[j])) {
                x1=Math.min(x1,selected.xy[j][0]);y1=Math.min(y1,selected.xy[j][1]);
                x2=Math.max(x2,selected.xy[j][0]);y2=Math.max(y2,selected.xy[j][1]);
            }
            float mx=Math.max((x2-x1)*.2f,20),my=Math.max((y2-y1)*.2f,20);
            trackedBox=new float[]{Math.max(0,x1-mx),Math.max(0,y1-my),
                Math.min(image.cols(),x2+mx),Math.min(image.rows(),y2+my)};
        }
        frameNumber++;
        return selected==null?MotionAnalysis.emptyPoints():selected.accepted(image.cols(),image.rows());
    }

    private Pose detectAndPose(Mat frame) throws Exception {
        detectorRuns++;
        Pose best=null;double bestArea=-1;
        for(float[] box:detect(frame)) {
            Pose candidate=inferPose(frame,box);
            double x1=Double.POSITIVE_INFINITY,y1=x1,x2=Double.NEGATIVE_INFINITY,y2=x2;
            int accepted=0;
            for(int j=0;j<17;j++) if(candidate.scores[j]>=MotionAnalysis.MIN_SCORE && MotionAnalysis.finite(candidate.xy[j])) {
                accepted++;x1=Math.min(x1,candidate.xy[j][0]);y1=Math.min(y1,candidate.xy[j][1]);
                x2=Math.max(x2,candidate.xy[j][0]);y2=Math.max(y2,candidate.xy[j][1]);
            }
            double area=(x2-x1)*(y2-y1);
            if(accepted>=4 && area>bestArea) {bestArea=area;best=candidate;}
        }
        return best;
    }

    private List<float[]> detect(Mat image) throws Exception {
        // RTMLib pads on bottom/right using BGR=114; YOLOX input is NOT divided by 255.
        double ratio=Math.min(640./image.rows(),640./image.cols());
        int width=(int)(image.cols()*ratio),height=(int)(image.rows()*ratio);
        Mat padded=new Mat(640,640,CvType.CV_8UC3,new Scalar(114,114,114));
        Mat resized=new Mat(),roi=padded.submat(0,height,0,width);
        try {
            Imgproc.resize(image,resized,new Size(width,height));resized.copyTo(roi);
            padded.get(0,0,detectorPixels);
        } finally {roi.release();resized.release();padded.release();}
        chw(detectorPixels,detectorInput,640*640,false);
        List<float[]> boxes=new ArrayList<>();
        try(OnnxTensor tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(detectorInput),new long[]{1,3,640,640});
            OrtSession.Result output=detector.run(Collections.singletonMap(detector.getInputNames().iterator().next(),tensor))) {
            // Bundled end-to-end YOLOX graph already performs NMS: [1,N,5] = xyxy,score.
            float[][][] values=(float[][][])output.get(0).getValue();
            if(values.length!=1) throw new IOException("Unexpected detector output batch");
            for(float[] row:values[0]) {
                if(row.length!=5) throw new IOException("Expected end-to-end YOLOX [1,N,5]");
                if(row[4]<=.3f) continue;
                float[] box={(float)(row[0]/ratio),(float)(row[1]/ratio),
                    (float)(row[2]/ratio),(float)(row[3]/ratio)};
                if(box[2]>box[0] && box[3]>box[1]) boxes.add(box);
            }
        }
        return boxes;
    }

    /** Public for the on-device parity test against Python RTMLib. */
    public Pose inferPose(Mat image,float[] box) throws Exception {
        double cx=(box[0]+box[2])*.5,cy=(box[1]+box[3])*.5;
        double sw=(box[2]-box[0])*1.25,sh=(box[3]-box[1])*1.25;
        if(sw>sh*192./256) sh=sw*256./192;else sw=sh*192./256;
        Mat transform=new Mat(2,3,CvType.CV_64F),crop=new Mat();
        transform.put(0,0,192/sw,0,96-cx*192/sw,0,256/sh,128-cy*256/sh);
        try {
            Imgproc.warpAffine(image,crop,transform,new Size(192,256),Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT,new Scalar(0,0,0));
            crop.get(0,0,posePixels);
        } finally {transform.release();crop.release();}
        chw(posePixels,poseInput,192*256,true);
        float[][] xy=new float[26][2];float[] scores=new float[26];
        try(OnnxTensor tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(poseInput),new long[]{1,3,256,192});
            OrtSession.Result output=pose.run(Collections.singletonMap(pose.getInputNames().iterator().next(),tensor))) {
            float[][] sx=((float[][][])output.get(0).getValue())[0];
            float[][] sy=((float[][][])output.get(1).getValue())[0];
            if(sx.length!=26 || sy.length!=26) throw new IOException("Expected 26 Halpe26 keypoints");
            for(int joint=0;joint<26;joint++) {
                int ix=argmax(sx[joint]),iy=argmax(sy[joint]);
                scores[joint]=(sx[joint][ix]+sy[joint][iy])*.5f;
                double px=scores[joint]>0?ix/2.:-.5,py=scores[joint]>0?iy/2.:-.5;
                xy[joint][0]=(float)(px/192*sw+cx-sw*.5);
                xy[joint][1]=(float)(py/256*sh+cy-sh*.5);
            }
        }
        return new Pose(xy,scores);
    }
    private static int argmax(float[] values) {
        int best=0;for(int i=1;i<values.length;i++) if(values[i]>values[best]) best=i;return best;
    }
    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] buffer=new byte[4096];int count;
        while((count=in.read(buffer))!=-1) out.write(buffer,0,count);
        return out.toByteArray();
    }
    private static void chw(byte[] bgr,float[] output,int pixels,boolean normalise) {
        double[] mean={123.675,116.28,103.53},std={58.395,57.12,57.375};
        // RTMLib normalises BGR directly with these values: do not swap red and blue.
        for(int channel=0;channel<3;channel++) for(int i=0;i<pixels;i++) {
            int value=bgr[3*i+channel]&255;
            output[channel*pixels+i]=(float)(normalise?(value-mean[channel])/std[channel]:value);
        }
    }
    @Override public void close() throws OrtException {try {pose.close();} finally {detector.close();}}
}
