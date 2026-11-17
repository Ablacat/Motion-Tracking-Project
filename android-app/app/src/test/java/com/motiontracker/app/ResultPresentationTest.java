package com.motiontracker.app;

import org.junit.Test;
import static org.junit.Assert.*;

public class ResultPresentationTest {
    @Test public void everyNegativeSpineValueStaysGreen() {
        for(double angle:new double[]{-0.01,-5,-45,-90,-180,-10_000,0})
            assertEquals(ResultPresentation.GREEN,ResultPresentation.colour(0,angle));
    }
    @Test public void scaleRunsContinuouslyThroughAmberAndSaturatesAtRed() {
        assertEquals(ResultPresentation.GREEN,ResultPresentation.colour(0,0));
        assertEquals(ResultPresentation.YELLOW,ResultPresentation.colour(0,22.5));
        assertEquals(ResultPresentation.RED,ResultPresentation.colour(0,45));
        assertEquals(ResultPresentation.RED,ResultPresentation.colour(0,180));
        assertNotEquals(ResultPresentation.GREEN,ResultPresentation.colour(0,10));
        assertNotEquals(ResultPresentation.RED,ResultPresentation.colour(0,30));
        int before=ResultPresentation.colour(0,22.499),after=ResultPresentation.colour(0,22.501);
        for(int shift:new int[]{0,8,16}) assertTrue(Math.abs(((before>>shift)&255)-((after>>shift)&255))<=1);
    }
    @Test public void otherMetricsUseTheirMagnitudeAndMissingValuesAreNeutral() {
        assertEquals(ResultPresentation.YELLOW,ResultPresentation.colour(1,90));
        assertEquals(ResultPresentation.RED,ResultPresentation.colour(1,180));
        assertEquals(ResultPresentation.colour(2,30),ResultPresentation.colour(2,-30));
        assertEquals(ResultPresentation.RED,ResultPresentation.colour(2,-45));
        assertEquals(ResultPresentation.NEUTRAL,ResultPresentation.colour(0,Double.NaN));
        assertEquals(ResultPresentation.NEUTRAL,ResultPresentation.colour(2,Double.POSITIVE_INFINITY));
    }
    @Test public void numericSignsAndMeasurementsRemainUnchanged() {
        assertEquals("-12.3 deg",ResultPresentation.number(-12.3));
        assertEquals("--",ResultPresentation.number(Double.NaN));
        MotionAnalysis.Result result=new MotionAnalysis.Result(0,true);
        result.spineChange=-20;result.sides[0].minHip=80;result.sides[1].minHip=100;
        result.sides[0].kneeBeforeAnkle=15;result.sides[1].kneeBeforeAnkle=25;
        assertEquals(-20,ResultPresentation.value(result,0),0);
        assertEquals(80,ResultPresentation.value(result,1),0);
        assertEquals(25,ResultPresentation.value(result,2),0);
    }
}
