package com.motiontracker.app;

import android.Manifest;
import android.app.Activity;
import android.content.*;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;
import android.view.*;
import android.widget.*;
import androidx.core.content.ContextCompat;

/** Small gallery-first UI. Processing stays in a foreground service, never the UI thread. */
public final class MainActivity extends Activity {
    private static final int PICK_VIDEO=10,NOTIFICATION_PERMISSION=11;
    private Button select,cancel,preview,shareVideo,shareCsv;
    private TextView status;private ProgressBar progress;
    private final BroadcastReceiver updates=new BroadcastReceiver() {
        @Override public void onReceive(Context context,Intent intent) {refresh();}
    };

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        LinearLayout layout=new LinearLayout(this);layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(dp(24),dp(32),dp(24),dp(24));layout.setBackgroundColor(Color.rgb(17,28,37));
        TextView title=text("Motion Tracker",30);title.setTypeface(null,android.graphics.Typeface.BOLD);layout.addView(title);
        TextView intro=text("Choose a deadlift video from your gallery. Analysis runs entirely on your phone.",17);
        intro.setPadding(0,dp(16),0,dp(20));layout.addView(intro);
        select=button("Choose video",layout);select.setOnClickListener(v->pickVideo());
        progress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);progress.setMax(100);
        layout.addView(progress,new LinearLayout.LayoutParams(-1,dp(12)));
        status=text("Select a clear full-body video of one lifter.",15);status.setPadding(0,dp(16),0,dp(16));layout.addView(status);
        cancel=button("Cancel analysis",layout);cancel.setOnClickListener(v->startService(
            new Intent(this,ProcessingService.class).setAction(ProcessingService.CANCEL)));
        preview=button("Play result",layout);preview.setOnClickListener(v->openResult());
        shareVideo=button("Share video",layout);shareVideo.setOnClickListener(v->share("video","video/mp4"));
        shareCsv=button("Share measurements CSV",layout);shareCsv.setOnClickListener(v->share("csv","text/csv"));
        ScrollView scroll=new ScrollView(this);scroll.setFillViewport(true);scroll.addView(layout);setContentView(scroll);
        // A killed process cannot resume a partially encoded video; the original is safe.
        SharedPreferences prefs=getSharedPreferences(ProcessingService.PREFS,MODE_PRIVATE);
        if(prefs.getBoolean("running",false) && !ProcessingService.running)
            prefs.edit().putBoolean("running",false).putString("message","Previous analysis was interrupted. Select the video to retry.").apply();
    }
    @Override protected void onStart() {
        super.onStart();
        IntentFilter filter=new IntentFilter(ProcessingService.UPDATE);
        ContextCompat.registerReceiver(this,updates,filter,ContextCompat.RECEIVER_NOT_EXPORTED);
        refresh();
    }
    @Override protected void onStop() {unregisterReceiver(updates);super.onStop();}
    private int dp(int value) {return Math.round(value*getResources().getDisplayMetrics().density);}
    private TextView text(String value,int sp) {
        TextView text=new TextView(this);text.setText(value);text.setTextSize(sp);text.setTextColor(Color.WHITE);return text;
    }
    private Button button(String text,LinearLayout parent) {
        Button button=new Button(this);button.setText(text);button.setAllCaps(false);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,dp(54));params.topMargin=dp(6);
        parent.addView(button,params);return button;
    }
    private void pickVideo() {
        SharedPreferences prefs=getSharedPreferences(ProcessingService.PREFS,MODE_PRIVATE);
        if(Build.VERSION.SDK_INT>=33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED
                && !prefs.getBoolean("notificationAsked",false)) {
            prefs.edit().putBoolean("notificationAsked",true).apply();
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},NOTIFICATION_PERMISSION);
            return;
        }
        launchPicker();
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] results) {
        super.onRequestPermissionsResult(request,permissions,results);
        // Notification permission is optional; gallery processing also works if declined.
        if(request==NOTIFICATION_PERMISSION) launchPicker();
    }
    private void launchPicker() {
        Intent intent=Build.VERSION.SDK_INT>=33?new Intent(MediaStore.ACTION_PICK_IMAGES):
            new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
        try {startActivityForResult(intent,PICK_VIDEO);} catch(ActivityNotFoundException e) {
            startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("video/*")
                .addCategory(Intent.CATEGORY_OPENABLE),PICK_VIDEO);
        }
    }
    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
        if(requestCode==PICK_VIDEO && resultCode==RESULT_OK && data!=null && data.getData()!=null) {
            Intent job=new Intent(this,ProcessingService.class).setAction(ProcessingService.PROCESS)
                .setData(data.getData()).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            try {startForegroundService(job);select.setEnabled(false);status.setText(R.string.starting_analysis);}
            catch(RuntimeException e) {status.setText(getString(R.string.cannot_start_analysis,e.getMessage()));}
        }
    }
    private void refresh() {
        SharedPreferences prefs=getSharedPreferences(ProcessingService.PREFS,MODE_PRIVATE);
        boolean active=prefs.getBoolean("running",false);
        select.setEnabled(!active);cancel.setVisibility(active?View.VISIBLE:View.GONE);
        progress.setVisibility(active?View.VISIBLE:View.GONE);
        progress.setProgress(prefs.getInt("percent",0));
        status.setText(prefs.getString("message","Select a clear full-body video of one lifter."));
        boolean video=prefs.contains("video")&&!active,csv=prefs.contains("csv")&&!active;
        preview.setVisibility(video?View.VISIBLE:View.GONE);shareVideo.setVisibility(video?View.VISIBLE:View.GONE);
        shareCsv.setVisibility(csv?View.VISIBLE:View.GONE);
    }
    private Uri result(String key) {
        String value=getSharedPreferences(ProcessingService.PREFS,MODE_PRIVATE).getString(key,null);
        return value==null?null:Uri.parse(value);
    }
    private void openResult() {
        Uri uri=result("video");if(uri==null) return;
        startActivity(new Intent(this,PreviewActivity.class).setData(uri));
    }
    private void share(String key,String mime) {
        Uri uri=result(key);if(uri==null) return;
        Intent intent=new Intent(Intent.ACTION_SEND).setType(mime).putExtra(Intent.EXTRA_STREAM,uri)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.setClipData(ClipData.newRawUri("Motion Tracker result",uri));
        startActivity(Intent.createChooser(intent,"Share result"));
    }
}
