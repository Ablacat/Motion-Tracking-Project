package com.motiontracker.app;

import org.junit.Test;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import static org.junit.Assert.*;

public class MotionAnalysisTest {
    @Test public void requestedLengthRule() {
        assertEquals(0,MotionAnalysis.curvature(100,100),1e-9);
        assertEquals(45,MotionAnalysis.curvature(100*Math.sqrt(2)/2,100),1e-9);
        assertEquals(60,MotionAnalysis.curvature(50,100),1e-9);
        assertEquals(0,MotionAnalysis.curvature(101,100),1e-9);
        assertTrue(Double.isNaN(MotionAnalysis.curvature(0,100)));
        assertTrue(Double.isNaN(MotionAnalysis.curvature(100,Double.NaN)));
    }
    private static float[] rotate(float[] vector,double degrees) {
        double r=Math.toRadians(degrees),c=Math.cos(r),s=Math.sin(r);
        return new float[]{(float)(c*vector[0]-s*vector[1]),(float)(s*vector[0]+c*vector[1])};
    }
    private static float[] minus(float[] a,float[] b) {return new float[]{a[0]-b[0],a[1]-b[1]};}
    private static float[] plus(float[] a,float[] b) {return new float[]{a[0]+b[0],a[1]+b[1]};}
    private static MotionAnalysis.Frame frame(int number,double kneeAngle,double ankleAngle,double curve) {
        float[][] points=MotionAnalysis.emptyPoints();
        for(int j:MotionAnalysis.REQUIRED) points[j]=new float[]{3,3};
        for(int side=0;side<2;side++) {
            int[] joints=MotionAnalysis.SIDES[side];
            float[] hip={2*side,0},knee={2*side,1};
            float[] shoulder=plus(hip,rotate(minus(knee,hip),170));
            float[] ankle=plus(knee,rotate(minus(hip,knee),kneeAngle));
            float[] toe=plus(ankle,rotate(minus(knee,ankle),ankleAngle));
            float[][] all={shoulder,hip,knee,ankle,toe};
            for(int i=0;i<5;i++) points[joints[i]]=all[i];
        }
        points[17]=new float[]{1,-1.8f};points[18]=new float[]{1,(float)(-1.2*Math.cos(Math.toRadians(curve)))};
        points[19]=new float[]{1,0};
        return new MotionAnalysis.Frame(number*100_000L,points);
    }
    @Test public void setupIsIgnoredAndEachLiftResetsBothBaselines() {
        List<MotionAnalysis.Frame> frames=Arrays.asList(frame(0,150,100,20),frame(1,175,125,45),
            frame(2,140,95,60),frame(3,150,95,45),frame(4,155,105,50),
            frame(5,150,100,30),frame(6,160,100,60),frame(7,170,110,65));
        List<MotionAnalysis.Result> results=MotionAnalysis.analyse(frames,Arrays.asList(2,5),1.2);
        assertTrue(Double.isNaN(results.get(1).spineChange));
        assertTrue(Double.isNaN(results.get(1).maxKneeBeforeAnkle()));
        assertEquals(0,results.get(2).spineChange,1e-6);
        assertEquals(-15,results.get(3).spineChange,1e-4);
        assertEquals(10,results.get(4).maxKneeBeforeAnkle(),1e-4);
        assertEquals(0,results.get(5).spineChange,1e-6);
        assertEquals(30,results.get(6).spineChange,1e-4);
        assertEquals(10,results.get(7).maxKneeBeforeAnkle(),1e-4);
        assertEquals(2,results.get(7).liftNumber);
    }
    @Test public void gapDoesNotRebaseSpineOrTimingMidLift() {
        List<MotionAnalysis.Frame> frames=Arrays.asList(frame(0,150,100,60),frame(1,160,100,50),
            new MotionAnalysis.Frame(200_000,MotionAnalysis.emptyPoints()),frame(3,170,110,45));
        List<MotionAnalysis.Result> results=MotionAnalysis.analyse(frames,Collections.singletonList(0),1.2);
        assertFalse(results.get(2).complete);
        assertEquals(-15,results.get(3).spineChange,1e-4);
        assertTrue(Double.isNaN(results.get(3).maxKneeBeforeAnkle()));
    }
    @Test public void filteringDoesNotInventMissingPoints() {
        float[][] full=frame(0,150,100,30).points;
        assertTrue(MotionAnalysis.ready(full));
        for(int joint:MotionAnalysis.REQUIRED) {
            float[][] partial=full.clone();partial[joint]=null;
            MotionAnalysis.Frame smoothed=MotionAnalysis.smooth(0,partial,full);
            assertNull(smoothed.points[joint]);assertFalse(smoothed.complete);
        }
    }
    @Test public void overlayShowsOnlyRequestedMetricsAndHigherSignedKneeValue() {
        MotionAnalysis.Result result=new MotionAnalysis.Result(0,true);
        result.sides[0].minHip=80;result.sides[1].minHip=120;
        result.sides[0].kneeBeforeAnkle=-20;result.sides[1].kneeBeforeAnkle=-5;
        assertEquals(80,result.minHip(),0);assertEquals(-5,result.maxKneeBeforeAnkle(),0);
        assertEquals(3,result.overlay().length);
        for(String line:result.overlay()) assertFalse(line.contains("length"));
    }
    @Test public void matchesDesktopOnAll234SavedFrames() throws Exception {
        InputStream stream=getClass().getResourceAsStream("/desktop_reference.csv");assertNotNull(stream);
        List<MotionAnalysis.Frame> frames=new ArrayList<>();List<String[]> expected=new ArrayList<>();
        try(BufferedReader reader=new BufferedReader(new InputStreamReader(stream,StandardCharsets.UTF_8))) {
            reader.readLine();String line;
            while((line=reader.readLine())!=null) {
                String[] cells=line.split(",",-1);float[][] points=MotionAnalysis.emptyPoints();
                for(int j=0;j<26;j++) {
                    float x=Float.parseFloat(cells[1+j*2]),y=Float.parseFloat(cells[2+j*2]);
                    if(Float.isFinite(x)&&Float.isFinite(y)) points[j]=new float[]{x,y};
                }
                frames.add(new MotionAnalysis.Frame(Long.parseLong(cells[0]),points));expected.add(cells);
            }
        }
        assertEquals(234,frames.size());
        double reference=MotionAnalysis.reference(frames);
        List<Integer> starts=MotionAnalysis.liftStarts(frames,reference,30);
        assertEquals(Arrays.asList(107,213),starts);
        List<MotionAnalysis.Result> actual=MotionAnalysis.analyse(frames,starts,reference);
        for(int i=0;i<frames.size();i++) {
            String[] e=expected.get(i);MotionAnalysis.Result r=actual.get(i);
            assertValue(e[53],r.absoluteCurvature);assertValue(e[54],r.spineBaseline);assertValue(e[55],r.spineChange);
            assertValue(e[56],r.sides[0].minHip);assertValue(e[57],r.sides[1].minHip);
            assertValue(e[58],r.sides[0].kneeBeforeAnkle);assertValue(e[59],r.sides[1].kneeBeforeAnkle);
        }
    }
    private static void assertValue(String expected,double actual) {
        if(expected.isEmpty()) assertTrue("Expected missing value",Double.isNaN(actual));
        else assertEquals(Double.parseDouble(expected),actual,.02);
    }
}
