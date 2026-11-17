# Third-party components

This project ports RTMLib preprocessing and postprocessing for the existing models, and uses the following upstream components. Preserve applicable upstream notices when redistributing.

| Component | Version / provenance | Upstream licence |
|---|---|---|
| RTMLib (YOLOX/RTMPose preprocessing and decoding reference) | https://github.com/Tau-J/rtmlib | Apache-2.0: https://github.com/Tau-J/rtmlib/blob/main/LICENSE |
| MMPose / RTMPose checkpoint source | OpenMMLab `rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605` | https://github.com/open-mmlab/mmpose/blob/main/LICENSE |
| YOLOX / detector checkpoint source | OpenMMLab `yolox_m_8xb8-300e_humanart-c2c7a14a` | https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE |
| ONNX Runtime Android | 1.23.2 | MIT: https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| OpenCV Android | 4.12.0 | Apache-2.0: https://github.com/opencv/opencv/blob/4.12.0/LICENSE |
| AndroidX Core | 1.13.1 | Apache-2.0: https://android.googlesource.com/platform/frameworks/support/ |
| AndroidX test libraries | Test only | Apache-2.0: https://github.com/android/android-test |
| JUnit | 4.13.2, test only | EPL-1.0: https://github.com/junit-team/junit4/blob/r4.13.2/LICENSE-junit.txt |
| Gradle wrapper | 8.9 | Apache-2.0: https://github.com/gradle/gradle/blob/v8.9.0/LICENSE |

Exact model URLs, sizes and SHA-256 values are recorded in `app/src/main/assets/models/models.json`; `tools/prepare_models.py` pins those values. Model training-data terms and any conditions on the checkpoints should also be reviewed before public/commercial redistribution. A framework's licence alone is not a guarantee about all training data.
