package com.motiontracker.app;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.media.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.concurrent.CancellationException;

/** Streaming Android hardware decode/encode. No video bitmaps are retained in memory. */
public final class VideoIO {
    public interface Cancellation {boolean cancelled();}
    public interface FrameConsumer {void accept(Mat uprightBgr,long timeUs,int index) throws Exception;}
    public static void check(Cancellation cancellation) {
        if(cancellation.cancelled() || Thread.currentThread().isInterrupted()) throw new CancellationException();
    }
    public static final class Info {
        public final int width,height,rotation;
        public final long durationUs;
        public final double fps;
        Info(MediaFormat format) throws IOException {
            int rawWidth=format.getInteger(MediaFormat.KEY_WIDTH),rawHeight=format.getInteger(MediaFormat.KEY_HEIGHT);
            rotation=format.containsKey(MediaFormat.KEY_ROTATION)?format.getInteger(MediaFormat.KEY_ROTATION):0;
            if(rotation!=0 && rotation!=90 && rotation!=180 && rotation!=270)
                throw new IOException("Unsupported video rotation: "+rotation);
            int w=rotation%180==0?rawWidth:rawHeight,h=rotation%180==0?rawHeight:rawWidth;
            double scale=Math.min(1.,1280./Math.max(w,h));
            width=Math.max(2,((int)(w*scale)/2)*2);height=Math.max(2,((int)(h*scale)/2)*2);
            durationUs=format.containsKey(MediaFormat.KEY_DURATION)?format.getLong(MediaFormat.KEY_DURATION):0;
            double rate=format.containsKey(MediaFormat.KEY_FRAME_RATE)?format.getNumber(MediaFormat.KEY_FRAME_RATE).doubleValue():30.;
            fps=Double.isFinite(rate) && rate>0?rate:30.;
            if(format.containsKey(MediaFormat.KEY_COLOR_TRANSFER)) {
                int transfer=format.getInteger(MediaFormat.KEY_COLOR_TRANSFER);
                if(transfer==MediaFormat.COLOR_TRANSFER_ST2084 || transfer==MediaFormat.COLOR_TRANSFER_HLG)
                    throw new IOException("This version supports SDR video. Export this HDR clip as SDR and select it again.");
            }
        }
    }
    private static int videoTrack(MediaExtractor extractor) throws IOException {
        for(int i=0;i<extractor.getTrackCount();i++) {
            String mime=extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if(mime!=null && mime.startsWith("video/")) return i;
        }
        throw new IOException("The selected file contains no video track");
    }
    public static Info inspect(File file) throws IOException {
        MediaExtractor extractor=new MediaExtractor();
        try {extractor.setDataSource(file.getPath());return new Info(extractor.getTrackFormat(videoTrack(extractor)));}
        finally {extractor.release();}
    }

    public static int decode(File file,Info info,Cancellation cancellation,FrameConsumer consumer) throws Exception {
        MediaExtractor extractor=new MediaExtractor();MediaCodec decoder=null;
        Mat bgr=new Mat(),upright=new Mat(),scaled=new Mat();
        boolean started=false;
        try {
            extractor.setDataSource(file.getPath());int track=videoTrack(extractor);
            extractor.selectTrack(track);MediaFormat format=extractor.getTrackFormat(track);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            // Buffer output is rotated explicitly below, once, before pose estimation.
            format.setInteger(MediaFormat.KEY_ROTATION,0);
            decoder=MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME));
            decoder.configure(format,null,null,0);decoder.start();started=true;
            boolean inputEnd=false,outputEnd=false;int count=0;long firstPts=-1,lastPts=-1;
            long lastOutput=System.nanoTime();MediaCodec.BufferInfo bufferInfo=new MediaCodec.BufferInfo();
            while(!outputEnd) {
                check(cancellation);
                if(!inputEnd) {
                    int input=decoder.dequeueInputBuffer(10_000);
                    if(input>=0) {
                        ByteBuffer buffer=decoder.getInputBuffer(input);
                        if(buffer==null) throw new IOException("Decoder input buffer unavailable");
                        buffer.clear();int size=extractor.readSampleData(buffer,0);
                        if(size<0) {
                            decoder.queueInputBuffer(input,0,0,0,MediaCodec.BUFFER_FLAG_END_OF_STREAM);inputEnd=true;
                        } else {
                            decoder.queueInputBuffer(input,0,size,extractor.getSampleTime(),0);extractor.advance();
                        }
                    }
                }
                int output=decoder.dequeueOutputBuffer(bufferInfo,10_000);
                if(output>=0) {
                    try {
                        if(bufferInfo.size>0) {
                            Image image=decoder.getOutputImage(output);
                            if(image==null) throw new IOException("This decoder does not expose YUV images. Try an H.264 SDR MP4.");
                            try {imageToBgr(image,bgr);} finally {image.close();}
                            if(info.rotation==90) Core.rotate(bgr,upright,Core.ROTATE_90_CLOCKWISE);
                            else if(info.rotation==180) Core.rotate(bgr,upright,Core.ROTATE_180);
                            else if(info.rotation==270) Core.rotate(bgr,upright,Core.ROTATE_90_COUNTERCLOCKWISE);
                            else bgr.copyTo(upright);
                            Imgproc.resize(upright,scaled,new Size(info.width,info.height));
                            if(firstPts<0) firstPts=bufferInfo.presentationTimeUs;
                            long pts=bufferInfo.presentationTimeUs-firstPts;
                            if(pts<=lastPts) throw new IOException("Video frame timestamps are not strictly increasing");
                            lastPts=pts;
                            consumer.accept(scaled,pts,count++);
                        }
                        outputEnd=(bufferInfo.flags&MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0;
                    } finally {decoder.releaseOutputBuffer(output,false);}
                    lastOutput=System.nanoTime();
                } else if(System.nanoTime()-lastOutput>30_000_000_000L) {
                    throw new IOException("Video decoder timed out");
                }
            }
            if(count==0) throw new IOException("Video contains no decoded frames");
            return count;
        } finally {
            bgr.release();upright.release();scaled.release();extractor.release();
            if(decoder!=null) {try {if(started) decoder.stop();} finally {decoder.release();}}
        }
    }

    private static void imageToBgr(Image image,Mat output) throws IOException {
        if(image.getFormat()!=ImageFormat.YUV_420_888) throw new IOException("Unsupported decoder image format; use 8-bit SDR video");
        Rect crop=image.getCropRect();int width=crop.width(),height=crop.height();
        if((width&1)!=0 || (height&1)!=0) throw new IOException("Video has odd YUV dimensions; export with even dimensions");
        byte[] nv21=new byte[width*height*3/2];Image.Plane[] planes=image.getPlanes();
        readPlane(planes[0],crop.left,crop.top,width,height,nv21,0,1);
        readPlane(planes[2],crop.left/2,crop.top/2,width/2,height/2,nv21,width*height,2);
        readPlane(planes[1],crop.left/2,crop.top/2,width/2,height/2,nv21,width*height+1,2);
        Mat yuv=new Mat(height+height/2,width,CvType.CV_8UC1);
        try {yuv.put(0,0,nv21);Imgproc.cvtColor(yuv,output,Imgproc.COLOR_YUV2BGR_NV21);}
        finally {yuv.release();}
    }
    private static void readPlane(Image.Plane plane,int x,int y,int width,int height,byte[] output,int offset,int step) {
        ByteBuffer buffer=plane.getBuffer().duplicate();int base=buffer.position();
        int rowStride=plane.getRowStride(),pixelStride=plane.getPixelStride();
        for(int row=0;row<height;row++) {
            int source=base+(y+row)*rowStride+x*pixelStride;
            if(step==1 && pixelStride==1) {
                buffer.position(source);buffer.get(output,offset+row*width,width);
            } else for(int col=0;col<width;col++) output[offset+(row*width+col)*step]=buffer.get(source+col*pixelStride);
        }
    }

    public static final class Encoder implements AutoCloseable {
        private final MediaCodec codec;
        private final MediaMuxer muxer;
        private final int width,height;
        private final Mat yuv=new Mat();
        private final byte[] planar;
        private boolean muxerStarted=false,finished=false,codecStarted=false;
        private int track=-1;private long lastPts=-1;
        private final MediaCodec.BufferInfo info=new MediaCodec.BufferInfo();
        public Encoder(File file,Info video) throws IOException {
            width=video.width;height=video.height;planar=new byte[width*height*3/2];
            MediaFormat format=MediaFormat.createVideoFormat("video/avc",width,height);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            format.setInteger(MediaFormat.KEY_BIT_RATE,Math.max(2_000_000,(int)(width*height*video.fps*.15)));
            format.setInteger(MediaFormat.KEY_FRAME_RATE,(int)Math.round(Math.min(120,video.fps)));
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL,1);
            format.setInteger(MediaFormat.KEY_COLOR_STANDARD,MediaFormat.COLOR_STANDARD_BT601_NTSC);
            format.setInteger(MediaFormat.KEY_COLOR_TRANSFER,MediaFormat.COLOR_TRANSFER_SDR_VIDEO);
            format.setInteger(MediaFormat.KEY_COLOR_RANGE,MediaFormat.COLOR_RANGE_LIMITED);
            codec=MediaCodec.createEncoderByType("video/avc");
            MediaMuxer created=null;
            try {
                codec.configure(format,null,null,MediaCodec.CONFIGURE_FLAG_ENCODE);codec.start();codecStarted=true;
                created=new MediaMuxer(file.getPath(),MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            } catch(IOException|RuntimeException e) {
                if(created!=null) created.release();
                try {if(codecStarted) codec.stop();} finally {codec.release();}
                throw e;
            }
            muxer=created;
        }
        public void write(Mat bgr,long pts, Cancellation cancellation) throws IOException {
            check(cancellation);Imgproc.cvtColor(bgr,yuv,Imgproc.COLOR_BGR2YUV_I420);yuv.get(0,0,planar);
            int input;long deadline=System.nanoTime()+30_000_000_000L;
            while((input=codec.dequeueInputBuffer(10_000))<0) {
                check(cancellation);drain(false,cancellation);
                if(System.nanoTime()>deadline) throw new IOException("Video encoder input timed out");
            }
            Image image=codec.getInputImage(input);
            if(image==null) throw new IOException("This encoder does not support flexible YUV input");
            // getInputImage belongs to the codec; queueInputBuffer invalidates it.
            Image.Plane[] planes=image.getPlanes();
            writePlane(planes[0],planar,0,width,height);
            writePlane(planes[1],planar,width*height,width/2,height/2);
            writePlane(planes[2],planar,width*height+width*height/4,width/2,height/2);
            codec.queueInputBuffer(input,0,planar.length,pts,0);lastPts=pts;drain(false,cancellation);
        }
        private static void writePlane(Image.Plane plane,byte[] source,int offset,int width,int height) {
            ByteBuffer buffer=plane.getBuffer().duplicate();int base=buffer.position();
            int rowStride=plane.getRowStride(),pixelStride=plane.getPixelStride();
            for(int row=0;row<height;row++) {
                if(pixelStride==1) {buffer.position(base+row*rowStride);buffer.put(source,offset+row*width,width);}
                else for(int col=0;col<width;col++) buffer.put(base+row*rowStride+col*pixelStride,source[offset+row*width+col]);
            }
        }
        private boolean drain(boolean end, Cancellation cancellation) throws IOException {
            long deadline=System.nanoTime()+30_000_000_000L;
            while(true) {
                check(cancellation);
                int index=codec.dequeueOutputBuffer(info,end?10_000:0);
                if(index==MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if(muxerStarted) throw new IOException("Encoder format changed twice");
                    track=muxer.addTrack(codec.getOutputFormat());muxer.start();muxerStarted=true;
                } else if(index>=0) {
                    try {
                        ByteBuffer buffer=codec.getOutputBuffer(index);
                        if((info.flags&MediaCodec.BUFFER_FLAG_CODEC_CONFIG)!=0) info.size=0;
                        if(info.size>0) {
                            if(!muxerStarted || buffer==null) throw new IOException("Encoder output arrived without a valid format");
                            buffer.position(info.offset);buffer.limit(info.offset+info.size);muxer.writeSampleData(track,buffer,info);
                        }
                        if((info.flags&MediaCodec.BUFFER_FLAG_END_OF_STREAM)!=0) return true;
                    } finally {codec.releaseOutputBuffer(index,false);}
                } else if(index==MediaCodec.INFO_TRY_AGAIN_LATER && !end) return false;
                if(System.nanoTime()>deadline) throw new IOException("Video encoder output timed out");
            }
        }
        public void finish(Cancellation cancellation) throws IOException {
            int input;long deadline=System.nanoTime()+30_000_000_000L;
            while((input=codec.dequeueInputBuffer(10_000))<0) {
                check(cancellation);drain(false,cancellation);
                if(System.nanoTime()>deadline) throw new IOException("Encoder could not finish");
            }
            codec.queueInputBuffer(input,0,0,lastPts+1,MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            finished=drain(true,cancellation);
        }
        @Override public void close() {
            yuv.release();
            try {if(codecStarted) codec.stop();} finally {
                codec.release();
                try {if(muxerStarted) muxer.stop();} finally {muxer.release();}
            }
        }
        public boolean isFinished() {return finished;}
    }
    private VideoIO() {}
}
