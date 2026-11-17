package com.motiontracker.app;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/** Large, high-contrast result cards baked into the exported video. */
public final class ResultOverlay {
    public static void draw(Mat image,MotionAnalysis.Result result) {
        // Portrait values are about 4.5x the previous text size. Limit panel height on
        // landscape clips so the cards do not occupy most of the action area.
        double scale=Math.min(image.cols()/720.,image.rows()*.30/256.);
        int pad=Math.max(4,(int)Math.round(12*scale));
        int gap=Math.max(2,(int)Math.round(8*scale));
        int rowHeight=Math.max(24,(int)Math.round(72*scale));
        int panelHeight=3*rowHeight+2*gap+2*pad;
        int top=Math.max(0,image.rows()-panelHeight);
        Imgproc.rectangle(image,new Point(0,top),new Point(image.cols(),image.rows()),new Scalar(10,16,23),-1);
        for(int metric=0;metric<3;metric++) {
            double value=ResultPresentation.value(result,metric);
            int rgb=ResultPresentation.colour(metric,value);
            Scalar accent=new Scalar(rgb&255,(rgb>>8)&255,(rgb>>16)&255);
            int y=top+pad+metric*(rowHeight+gap);
            Imgproc.rectangle(image,new Point(pad,y),new Point(image.cols()-pad,y+rowHeight),new Scalar(28,36,44),-1);
            Imgproc.rectangle(image,new Point(pad,y),new Point(pad+Math.max(3,5*scale),y+rowHeight),accent,-1);
            String label=ResultPresentation.LABELS[metric],number=ResultPresentation.number(value);
            int[] numberBase=new int[1],labelBase=new int[1];
            double numberFont=1.8*scale,labelFont=.70*scale;
            int numberWeight=Math.max(2,(int)Math.round(3*scale));
            Size numberSize=Imgproc.getTextSize(number,Imgproc.FONT_HERSHEY_SIMPLEX,numberFont,numberWeight,numberBase);
            Size labelSize=Imgproc.getTextSize(label,Imgproc.FONT_HERSHEY_SIMPLEX,labelFont,1,labelBase);
            double right=image.cols()-2*pad;
            double labelX=pad+Math.max(8,20*scale);
            // Keep full numeric values legible if an unusually long value is exported.
            double numberRoom=right-(labelX+labelSize.width+16*scale);
            if(numberSize.width>numberRoom) {
                numberFont*=Math.max(.1,numberRoom/numberSize.width);
                numberSize=Imgproc.getTextSize(number,Imgproc.FONT_HERSHEY_SIMPLEX,numberFont,numberWeight,numberBase);
            }
            double numberY=y+(rowHeight+numberSize.height-numberBase[0])/2.;
            double labelY=y+(rowHeight+labelSize.height-labelBase[0])/2.;
            Imgproc.putText(image,label,new Point(labelX,labelY),Imgproc.FONT_HERSHEY_SIMPLEX,labelFont,
                new Scalar(230,235,240),1,Imgproc.LINE_AA,false);
            Imgproc.putText(image,number,new Point(right-numberSize.width,numberY),Imgproc.FONT_HERSHEY_SIMPLEX,
                numberFont,accent,numberWeight,Imgproc.LINE_AA,false);
        }
    }
    private ResultOverlay() {}
}
