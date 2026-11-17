package com.motiontracker.app;

import android.app.Activity;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.view.*;
import android.widget.*;

/** App-owned preview: MediaPlayer loops continuously until paused, stopped or closed. */
public final class PreviewActivity extends Activity {
    private VideoView video;
    private MediaPlayer player;
    private Uri uri;
    private int positionMs;
    private boolean playWhenReady=true,visible=false,prepared=false;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        uri=getIntent().getData();
        if(uri==null) {finish();return;}
        if(state!=null) {positionMs=state.getInt("position",0);playWhenReady=state.getBoolean("playing",true);}
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundColor(Color.BLACK);
        LinearLayout toolbar=new LinearLayout(this);toolbar.setGravity(Gravity.CENTER_VERTICAL);
        int padding=dp(12);toolbar.setPadding(padding,padding,padding,padding);
        TextView title=new TextView(this);title.setText(R.string.preview_title);title.setTextSize(22);title.setTextColor(Color.WHITE);
        toolbar.addView(title,new LinearLayout.LayoutParams(0,-2,1));
        Button stop=new Button(this);stop.setId(R.id.preview_stop);stop.setText(R.string.stop_preview);stop.setAllCaps(false);
        stop.setOnClickListener(v->{playWhenReady=false;video.stopPlayback();finish();});
        toolbar.addView(stop,new LinearLayout.LayoutParams(-2,-2));root.addView(toolbar);
        FrameLayout holder=new FrameLayout(this);
        video=new VideoView(this);video.setId(R.id.preview_video);
        holder.addView(video,new FrameLayout.LayoutParams(-1,-1,Gravity.CENTER));
        root.addView(holder,new LinearLayout.LayoutParams(-1,0,1));
        TextView error=new TextView(this);error.setTextColor(Color.WHITE);error.setPadding(padding,padding,padding,padding);
        error.setVisibility(View.GONE);root.addView(error);
        setContentView(root);
        MediaController controls=new MediaController(this);controls.setAnchorView(video);video.setMediaController(controls);
        video.setOnPreparedListener(mediaPlayer->{
            player=mediaPlayer;prepared=true;
            error.setVisibility(View.GONE);
            mediaPlayer.setLooping(true);
            mediaPlayer.setScreenOnWhilePlaying(true);
            if(positionMs>0) video.seekTo(positionMs);
            if(visible && playWhenReady) video.start();
        });
        video.setOnErrorListener((mediaPlayer,what,extra)->{
            playWhenReady=false;error.setText(R.string.preview_error);error.setVisibility(View.VISIBLE);return true;
        });
    }
    private int dp(int value) {return Math.round(value*getResources().getDisplayMetrics().density);}
    @Override protected void onStart() {
        super.onStart();
        if(video!=null) {prepared=false;video.setVideoURI(uri);}
    }
    @Override protected void onResume() {
        super.onResume();visible=true;
        if(prepared && playWhenReady) video.start();
    }
    @Override protected void onPause() {
        visible=false;
        if(prepared) {positionMs=video.getCurrentPosition();playWhenReady=video.isPlaying();video.pause();}
        super.onPause();
    }
    @Override protected void onStop() {
        // Release the codec outside the preview. onStart prepares it again on return.
        if(video!=null) video.stopPlayback();prepared=false;player=null;
        super.onStop();
    }
    @Override protected void onSaveInstanceState(Bundle state) {
        state.putInt("position",prepared?video.getCurrentPosition():positionMs);
        state.putBoolean("playing",visible && prepared?video.isPlaying():playWhenReady);
        super.onSaveInstanceState(state);
    }
    boolean isLooping() {return prepared && player!=null && player.isLooping();}
}
