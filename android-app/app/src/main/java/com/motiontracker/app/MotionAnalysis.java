package com.motiontracker.app;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Geometry and per-repetition analysis. No Android dependency: tested on the JVM. */
public final class MotionAnalysis {
    public static final int HEAD = 17, NECK = 18, PELVIS = 19;
    public static final int[] REQUIRED = {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25};
    public static final int[][] SIDES = {{5,11,13,15,20}, {6,12,14,16,21}};
    public static final double MIN_SCORE = .5, ALPHA = .35, CHANGE_THRESHOLD = 5;

    /** Coordinates use native Halpe26 indices, in upright/output image pixels. */
    public static final class Frame {
        public final long timeUs;
        public final float[][] points;
        public final boolean complete;
        public Frame(long timeUs, float[][] points) {
            this.timeUs = timeUs;
            this.points = points;
            this.complete = ready(points);
        }
    }

    public static double angle(float[] a, float[] vertex, float[] b) {
        if (!finite(a) || !finite(vertex) || !finite(b)) return Double.NaN;
        double ux = a[0]-vertex[0], uy = a[1]-vertex[1];
        double vx = b[0]-vertex[0], vy = b[1]-vertex[1];
        double denominator = Math.hypot(ux,uy)*Math.hypot(vx,vy);
        if (denominator <= 1e-9) return Double.NaN;
        return Math.toDegrees(Math.acos(clamp((ux*vx+uy*vy)/denominator,-1,1)));
    }
    static double clamp(double x, double low, double high) { return Math.max(low,Math.min(high,x)); }
    static boolean finite(float[] xy) {
        return xy != null && xy.length >= 2 && Float.isFinite(xy[0]) && Float.isFinite(xy[1]);
    }
    public static double length(float[][] p) {
        if (!finite(p[NECK]) || !finite(p[PELVIS])) return Double.NaN;
        double length = Math.hypot(p[NECK][0]-p[PELVIS][0],p[NECK][1]-p[PELVIS][1]);
        return length > 1e-9 ? length : Double.NaN;
    }
    public static double curvature(double length, double reference) {
        if (!Double.isFinite(length) || !Double.isFinite(reference) || length <= 0 || reference <= 0)
            return Double.NaN;
        return Math.toDegrees(Math.acos(clamp(length/reference,0,1)));
    }
    public static boolean ready(float[][] p) {
        if (p == null || p.length != 26) return false;
        for (int joint : REQUIRED) if (!finite(p[joint])) return false;
        for (int[] side : SIDES)
            for (int i=0;i<3;i++)
                if (!Double.isFinite(angle(p[side[i]],p[side[i+1]],p[side[i+2]]))) return false;
        return Double.isFinite(length(p)) && Double.isFinite(angle(p[HEAD],p[NECK],p[PELVIS]));
    }
    public static float[][] emptyPoints() { return new float[26][]; }

    /** Missing detections stay missing. Only complete frames become the next history. */
    public static Frame smooth(long ptsUs, float[][] raw, float[][] previous) {
        float[][] points = emptyPoints();
        for (int i=0;i<26;i++) {
            if (!finite(raw[i])) continue;
            float[] old = previous != null && finite(previous[i]) ? previous[i] : raw[i];
            points[i] = new float[]{(float)(ALPHA*raw[i][0]+(1-ALPHA)*old[0]),
                (float)(ALPHA*raw[i][1]+(1-ALPHA)*old[1])};
        }
        return new Frame(ptsUs,points);
    }

    public static double reference(List<Frame> frames) {
        double longest = Double.NaN;
        for (Frame f : frames) if (f.complete) {
            double value = length(f.points);
            if (!Double.isFinite(longest) || value > longest) longest = value;
        }
        return longest;
    }

    private static double meanY(float[][] p,int a,int b) { return (p[a][1]+p[b][1])/2.; }
    private static double kneeMean(float[][] p) {
        return (angle(p[11],p[13],p[15])+angle(p[12],p[14],p[16]))/2.;
    }

    /** Wrists stand in for bar movement. Backdate confirmed ascent to the setup bottom. */
    public static List<Integer> liftStarts(List<Frame> frames,double reference,double nominalFps) {
        List<Integer> starts = new ArrayList<>();
        if (!Double.isFinite(reference)) return starts;
        double rise = reference*.02;
        int confirm = Math.max(2,(int)Math.ceil(nominalFps*.10));
        int candidate = -1, ascending = 0;
        double candidateHand=0,candidateHip=0,candidateKnee=0;
        boolean waitingForSetup = true;
        for (int i=0;i<frames.size();i++) {
            Frame f=frames.get(i);
            if (!f.complete) {candidate=-1;ascending=0;continue;}
            float[][] p=f.points;
            double handY=meanY(p,9,10),ankleY=meanY(p,15,16);
            double hipY=meanY(p,11,12),kneeY=meanY(p,13,14),knee=kneeMean(p);
            double hand=ankleY-handY,hip=ankleY-hipY;
            if (!waitingForSetup) {
                if (knee>=165 && handY<kneeY) waitingForSetup=true;
                continue;
            }
            if (candidate<0) {
                if (handY>=kneeY && knee<165) {
                    candidate=i;candidateHand=hand;candidateHip=hip;candidateKnee=knee;
                }
                continue;
            }
            if (hip<=candidateHip) {
                candidate=i;candidateHand=hand;candidateHip=hip;candidateKnee=knee;ascending=0;
                continue;
            }
            boolean rising=hand-candidateHand>=rise && hip-candidateHip>=rise && knee-candidateKnee>=2;
            ascending=rising?ascending+1:0;
            if (ascending>=confirm) {
                starts.add(candidate);candidate=-1;ascending=0;waitingForSetup=false;
            }
        }
        return starts;
    }

    public static final class Side {
        public double hip=Double.NaN,minHip=Double.NaN,knee=Double.NaN,ankle=Double.NaN;
        public double kneeChange=Double.NaN,ankleChange=Double.NaN;
        public double kneeBeforeAnkle=Double.NaN,kneeOnset=Double.NaN,ankleOnset=Double.NaN;
        public double timingBaseline=Double.NaN;
    }
    public static final class Result {
        public final long timeUs;
        public final boolean complete;
        public final Side[] sides={new Side(),new Side()};
        public int liftNumber;
        public double liftStart=Double.NaN,spineLength=Double.NaN,reference=Double.NaN;
        public double absoluteCurvature=Double.NaN,spineBaseline=Double.NaN,spineChange=Double.NaN;
        Result(long pts,boolean complete) {this.timeUs=pts;this.complete=complete;}
        public double minHip() {
            return extremum(sides[0].minHip,sides[1].minHip,false);
        }
        public double maxKneeBeforeAnkle() {
            return extremum(sides[0].kneeBeforeAnkle,sides[1].kneeBeforeAnkle,true);
        }
        private static double extremum(double a,double b,boolean higher) {
            if (!Double.isFinite(a)) return b;
            if (!Double.isFinite(b)) return a;
            return higher?Math.max(a,b):Math.min(a,b);
        }
        public String[] overlay() {
            return new String[]{"Spine angle change from lift start: "+display(spineChange),
                "Lowest hip angle: "+display(minHip()),
                "Knee change before ankle: "+display(maxKneeBeforeAnkle())};
        }
    }
    private static final class Timing {
        double knee,ankle,time,kneeOnset=Double.NaN,ankleOnset=Double.NaN;
        double previousKnee=0,kneeBeforeAnkle=Double.NaN;
        Timing(double knee,double ankle,double time) {this.knee=knee;this.ankle=ankle;this.time=time;}
    }
    private static String display(double angle) {
        return Double.isFinite(angle)?String.format(Locale.US,"%.1f deg",angle):"--";
    }

    /** Angle deltas use the detected start of each lift, never the start of the video. */
    public static List<Result> analyse(List<Frame> frames,List<Integer> starts,double reference) {
        List<Result> results=new ArrayList<>(frames.size());
        double[] minima={Double.NaN,Double.NaN};
        Timing[] timing=new Timing[2];
        int next=0,lift=0,start=-1,spineBaselineStart=-1;
        double spineBaseline=Double.NaN;
        for (int i=0;i<frames.size();i++) {
            Frame frame=frames.get(i);
            if (next<starts.size() && starts.get(next)==i) {start=i;lift=++next;}
            Result result=new Result(frame.timeUs,frame.complete);
            results.add(result);
            if (!frame.complete) {Arrays.fill(timing,null);continue;}
            result.liftNumber=lift;
            result.liftStart=start<0?Double.NaN:frames.get(start).timeUs/1e6;
            result.spineLength=length(frame.points);result.reference=reference;
            result.absoluteCurvature=curvature(result.spineLength,reference);
            if (i==start) {spineBaseline=result.absoluteCurvature;spineBaselineStart=start;}
            if (start>=0 && spineBaselineStart==start) {
                result.spineBaseline=spineBaseline;
                result.spineChange=result.absoluteCurvature-spineBaseline;
            }
            double seconds=frame.timeUs/1e6;
            for (int side=0;side<2;side++) {
                int[] j=SIDES[side];float[][] p=frame.points;Side r=result.sides[side];
                r.hip=angle(p[j[0]],p[j[1]],p[j[2]]);
                r.knee=angle(p[j[1]],p[j[2]],p[j[3]]);
                r.ankle=angle(p[j[2]],p[j[3]],p[j[4]]);
                if (!Double.isFinite(minima[side]) || r.hip<minima[side]) minima[side]=r.hip;
                r.minHip=minima[side];
                if (i==start) timing[side]=new Timing(r.knee,r.ankle,seconds);
                Timing t=timing[side];
                if (t==null) continue;
                r.kneeChange=r.knee-t.knee;r.ankleChange=r.ankle-t.ankle;
                if (!Double.isFinite(t.kneeOnset) && Math.abs(r.kneeChange)>=CHANGE_THRESHOLD)
                    t.kneeOnset=seconds;
                if (!Double.isFinite(t.ankleOnset) && Math.abs(r.ankleChange)>=CHANGE_THRESHOLD) {
                    t.ankleOnset=seconds;t.kneeBeforeAnkle=t.previousKnee;
                }
                r.kneeOnset=t.kneeOnset;r.ankleOnset=t.ankleOnset;
                r.kneeBeforeAnkle=t.kneeBeforeAnkle;r.timingBaseline=t.time;
                t.previousKnee=r.kneeChange;
            }
        }
        return results;
    }
    private MotionAnalysis() {}
}
