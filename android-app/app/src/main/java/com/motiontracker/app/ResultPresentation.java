package com.motiontracker.app;

import java.util.Locale;

/** Continuous display colour scale. These are magnitude ranges, not clinical limits. */
public final class ResultPresentation {
    public static final int GREEN=0x34D399,YELLOW=0xFBBF24,RED=0xF85050,NEUTRAL=0xBDC9D5;
    // A full green -> amber -> red sweep occurs over these angle ranges.
    public static final double SPINE_RED_DEG=45,HIP_RED_DEG=180,KNEE_RED_DEG=45;
    public static final String[] LABELS={"Spine change","Lowest hip angle","Knee before ankle"};

    public static double value(MotionAnalysis.Result result,int metric) {
        if(result==null) return Double.NaN;
        switch(metric) {
            case 0:return result.spineChange;
            case 1:return result.minHip();
            case 2:return result.maxKneeBeforeAnkle();
            default:throw new IllegalArgumentException("Unknown metric: "+metric);
        }
    }
    public static int colour(int metric,double value) {
        if(!Double.isFinite(value)) return NEUTRAL;
        double magnitude,limit;
        switch(metric) {
            // Decreasing curvature must stay green regardless of its absolute magnitude.
            case 0:magnitude=Math.max(0,value);limit=SPINE_RED_DEG;break;
            case 1:magnitude=Math.abs(value);limit=HIP_RED_DEG;break;
            case 2:magnitude=Math.abs(value);limit=KNEE_RED_DEG;break;
            default:throw new IllegalArgumentException("Unknown metric: "+metric);
        }
        double fraction=MotionAnalysis.clamp(magnitude/limit,0,1);
        return fraction<=.5?mix(GREEN,YELLOW,fraction*2):mix(YELLOW,RED,(fraction-.5)*2);
    }
    private static int mix(int a,int b,double fraction) {
        int red=(int)Math.round(((a>>16)&255)*(1-fraction)+((b>>16)&255)*fraction);
        int green=(int)Math.round(((a>>8)&255)*(1-fraction)+((b>>8)&255)*fraction);
        int blue=(int)Math.round((a&255)*(1-fraction)+(b&255)*fraction);
        return red<<16|green<<8|blue;
    }
    public static String number(double value) {
        return Double.isFinite(value)?String.format(Locale.US,"%.1f deg",value):"--";
    }
    private ResultPresentation() {}
}
