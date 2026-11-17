# Motion Tracker for Android

An offline Android app for the Galaxy S23 / Android 13. Select an existing gallery video, analyse it on the phone, then play or share the annotated MP4 and measurements CSV. No server, account, Python installation on the phone, internet permission, or broad gallery access is required.

## Install the prepared APK

Use `dist/MotionTracker-S23-debug.apk`. It is a signed **debug build for ARM64 phones**, suitable for sideloading and testing.

1. Transfer the APK to the S23 by USB or Quick Share.
2. Open it in **My Files**. If prompted, allow that app to install unknown apps, then install **Motion Tracker**.
3. Open Motion Tracker and tap **Choose video**. Notifications are optional but allow progress/cancellation from the notification panel.
4. Select a clear, full-body, single-lifter SDR video from the gallery. Keep the app running; a foreground service handles processing if the activity is backgrounded.
5. Tap **Play result**, **Share video**, or **Share measurements CSV** after completion. The in-app preview loops automatically. Tap the video for pause/seek controls, or press **Stop** or Back to close it.

Results are saved to **Movies/MotionTracker** and **Download/MotionTracker**. Video export is silent, matching the desktop tracker. The original video is never modified. The app copies the selected source into its private cache during processing and removes its temporary job files afterward. Leave enough free space for the source copy, annotated export, and the models.

With Android platform-tools installed and USB debugging authorised:

```powershell
adb install -r dist/MotionTracker-S23-debug.apk
```

The native inference and codec pipeline still needs testing on your actual S23. Build/test results are recorded in `VALIDATION.md`; a successful build is not a measured phone benchmark.

## Build from source

Requirements: **JDK 17**, Android Studio/Android SDK **platform 34**, SDK build tools, and **Python 3.11+** only for model preparation. The wrapper uses **Gradle 8.9** with Android Gradle Plugin **8.7.0**. Minimum Android version is **10 / API 29**; the supplied APK targets API 34 and ARM64.

Open this directory in Android Studio, select JDK 17, and let it create `local.properties` with your SDK location. Alternatively set `ANDROID_HOME` to your installed SDK location. Do not copy another machine's `local.properties`.

From this directory:

```powershell
python tools/prepare_models.py
.\gradlew.bat -Pabis=arm64-v8a testDebugUnitTest assembleDebug lintDebug
```

macOS/Linux:

```bash
python3 tools/prepare_models.py
sh gradlew -Pabis=arm64-v8a testDebugUnitTest assembleDebug lintDebug
```

The APK is `app/build/outputs/apk/debug/app-debug.apk`. Model preparation first tries `~/.cache/rtmlib/hub/checkpoints`; otherwise it downloads the pinned OpenMMLab checkpoints, verifies their SHA-256 hashes, and stages them in `app/src/main/assets/models`. Models are excluded from source control/source ZIPs but included in the prepared APK. `checkModels` stops a build if weights are missing. Network access is needed for the first dependency/model downloads, **not** when running the APK.

## What is tracked

- **Spine angle change from lift start:** the absolute shortening estimate is `degrees(acos(current neck–central-hip distance / longest complete-frame distance))`. Subtract the angle at each detected lift start. It resets to 0° per lift, is blank before the first lift, and is signed afterward.
- **Lowest hip angle:** the lowest shoulder–hip–knee angle over both sides and all complete frames of the clip.
- **Knee change before ankle:** per-side knee-angle change from lift start, frozen on the frame before ankle change first reaches 5°. The display uses the higher signed value of the two sides. A 5° threshold also defines knee motion onset for CSV timing.

The spine **length is hidden in the overlay** but retained in the CSV. The absolute length-based spine angle is a heuristic, not anatomical spinal curvature: camera distance, perspective, rotation, clothing, and detection errors can affect it. It cannot distinguish flexion from extension.

The launch page shows the video controls and processing status, without a metric list or explanatory footer. Newly analysed videos display large numeric results on dark cards. Result colours follow a continuous green → amber → red magnitude scale: positive spine changes and absolute knee changes reach red at 45°, and hip angles at 180°. All negative spine changes stay green; missing values are grey. These are adjustable display ranges in `ResultPresentation.java`, not clinical or safety thresholds. Existing exported videos need to be re-analysed to get the larger coloured overlay.

Lift starts are estimated from the lowest setup hip position followed by sustained hand/hip ascent and knee extension. Wrists act as a bar-motion proxy; the bar itself is not detected. Standing completion arms the next repetition. Detection is designed for conventional full-range deadlifts; unusual/partial repetitions may require a different onset detector. When a lift cannot be detected, its relative metrics remain unavailable.

Every decoded frame gets pose inference. Only frames with all 19 required mapped landmarks and valid angles update the display. Missing frames are blank in the CSV, while the video holds the last complete display. A gap invalidates knee/ankle timing until the next lift; it never silently creates a new mid-lift baseline. Spine change keeps the original per-lift baseline across gaps.

## Processing and performance

1. Decode the gallery clip into upright BGR frames. Preserve actual presentation timestamps and handle 0°/90°/180°/270° rotation. Resize to an even output size with a maximum dimension of 1280 pixels.
2. Run the same balanced **YOLOX-M + RTMPose-M Halpe26** ONNX models as the desktop code. Reuse a padded person crop, re-detect every 10 frames, and retry fresh detection on the same frame if tracked landmarks are lost. Use two CPU inference threads and disable idle thread spinning.
3. Retain only coordinates/timestamps in memory. Find the longest valid spine length and each repetition's start, then calculate both-side measurements.
4. Decode again and render the three metrics and skeleton. Encode H.264 MP4 through Android MediaCodec and publish the outputs using MediaStore.

Supported input is an unencrypted **8-bit SDR video** supported by the phone's decoder, typically H.264 or HEVC. HDR/10-bit clips are explicitly rejected rather than silently producing incorrect colours. Export them as SDR before selecting them. This version does not preserve audio. Processing time and thermal behaviour must be measured on the S23; desktop speedups do not predict phone performance.

## Tests

```powershell
.\gradlew.bat -Pabis=arm64-v8a testDebugUnitTest lintDebug
.\gradlew.bat -Pabis=arm64-v8a assembleDebugAndroidTest
# With your test phone connected and authorised (this installs/runs test APKs):
.\gradlew.bat -Pabis=arm64-v8a connectedDebugAndroidTest
```

- JVM tests check the requested cosine rule, signed per-lift spine changes, per-lift knee timing, gap handling, required-point gating, three-line overlay, continuous colours and negative-spine handling, and parity against all 234 saved desktop frames. The fixture is checked in; Python is not required to run these tests.
- Device tests check native pose coordinates/scores against a Python-generated reference, successful person detection, H.264 encode/decode, variable frame timestamps, frame count, colours, 90° rotation, portrait/landscape overlays, and repeated preview looping until Stop. Their code compiles into a separate test APK; compiling it does not run it. The four tests have passed on the Android 13 Galaxy_S23 emulator; see `VALIDATION.md`.
- Developer-only `tools/generate_test_fixtures.py` regenerates fixtures from the original desktop videos/CSVs and requires the desktop Python environment (`rtmlib`, `opencv-python`, `numpy`, `onnxruntime`).

For an x86-64 emulator build use `-Pabis=x86_64`; `dist/MotionTracker-emulator-debug.apk` is prepared for that architecture. The distributed S23 APK only includes `arm64-v8a`. A local `abis=x86_64` setting in `gradle.properties` can keep Android Studio's Run button configured for an emulator; the explicit command-line flags above override it for phone builds.

## Release signing

The supplied debug APK is for sideloading. For a distributable release, use **Android Studio → Build → Generate Signed App Bundle / APK → APK**, create/choose a private keystore, and select the release variant. Keep that keystore private and backed up: future updates need the same signing key. `assembleRelease` produces an unsigned APK until release signing is configured. Review current target-SDK/store requirements before store submission; this project is prepared for direct deployment to the S23.

## Source layout

| File | Responsibility |
|---|---|
| `MainActivity.java` | Gallery selection, progress, playback/share actions |
| `PreviewActivity.java` | Looping in-app playback, pause/seek controls, Stop and lifecycle cleanup |
| `ProcessingService.java` | Foreground job, notifications, cancellation, wake lock |
| `RtmPoseEngine.java` | Model verification/loading, YOLOX preprocessing, crop reuse, affine pose preprocessing, SimCC decoding |
| `VideoIO.java` | Timestamp-aware streaming decode, YUV conversion, rotation, H.264 encode/mux |
| `MotionAnalysis.java` | Pure Java geometry, lift detection, baselines and metric state |
| `VideoProcessor.java` | Two-pass pipeline, skeleton/metric overlay, CSV, gallery export |
| `ResultOverlay.java`, `ResultPresentation.java` | Large result cards and continuous magnitude colours |

ONNX Runtime and OpenCV versions are pinned in `app/build.gradle`. Upstream references and notices are in `THIRD_PARTY_NOTICES.md`.
