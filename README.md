#vision-based-spatiotemporal-control
This project implements a real-time vision system that analyzes video streams to detect risky behaviors and automatically adjusts flow control. I built it to explore how deep learning can work with traditional control theory for safety applications.
The system uses MobileNetV2 for visual feature extraction and a bidirectional LSTM to understand temporal patterns across 30-frame windows. The risk predictions feed into a PID controller that manages valve flow with safety interlocks at critical thresholds.
Testing was done on three datasets including pool surveillance footage and industrial valve sessions. The entire pipeline runs at around 25-30 FPS on Google Colab's GPU. Main dependencies are TensorFlow, OpenCV, and NumPy. 
