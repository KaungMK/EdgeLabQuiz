# Raspberry Pi Lab Notes for Quiz Preparation

## Table of Contents
1. [Sound Analytics with Raspberry Pi](#sound-analytics-with-raspberry-pi)
2. [Image Analytics with Raspberry Pi](#image-analytics-with-raspberry-pi)
3. [Video Analytics with Raspberry Pi](#video-analytics-with-raspberry-pi)
4. [Real-time Inference of Deep Learning Models on Edge Device](#real-time-inference-of-deep-learning-models-on-edge-device)
5. [IoT Communications: MQTT](#iot-communications-mqtt)
6. [AWS IoT: Real-Time Device Data to AWS IoT Core](#aws-iot-real-time-device-data-to-aws-iot-core)
7. [Raspberry Pi 400 Setup with Webcam](#raspberry-pi-400-setup-with-webcam)

---

## Sound Analytics with Raspberry Pi

### Key Concepts
- Sound analytics involves capturing, analyzing, and visualizing audio data
- Applications include voice-enabled services, healthcare applications, and audio chatbots
- Python libraries for sound processing include PyAudio, SoundDevice, Librosa

### Setup Process
1. **Hardware prerequisites**:
   - Raspberry Pi with Raspbian OS
   - USB microphone
   - Optional speaker/USB headset for playback

2. **Software preparation**:
   ```bash
   # System updates
   sudo apt update
   sudo apt upgrade
   
   # Create virtual environment to avoid library conflicts
   sudo apt install python3-venv
   python3 -m venv audio
   source audio/bin/activate
   
   # Install necessary libraries
   sudo apt install portaudio19-dev
   pip3 install pyaudio
   pip3 install sounddevice
   pip3 install scipy matplotlib
   pip install librosa
   ```

3. **Testing microphone**:
   ```bash
   arecord --duration=10 test.wav
   aplay test.wav
   ```

### Sound Processing Techniques

#### Audio Visualization
- Time series representation of captured audio
- Frequency spectrum analysis using Fourier Transform
- Sample code available for both PyAudio and SoundDevice approaches

#### Audio Filtering
- Bandpass filtering - keeping only specific frequency ranges
- Can be implemented using sample code for either PyAudio or SoundDevice

#### Audio Feature Extraction
- **Spectrogram**: Visual representation of frequencies over time
- **Chromogram**: Representation of pitch classes (relevant for music analysis)
- **Mel-Spectrogram**: Uses the Mel Scale (perceptual scale of pitches) instead of frequency
- **MFCC (Mel Frequency Cepstral Coefficients)**: Representation of short-term power spectrum

#### Speech Recognition
- Requires additional libraries:
  ```bash
  sudo apt-get install flac
  pip install pocketsphinx
  pip install SpeechRecognition
  ```
- CMUSphinx (offline, edge-based inference) vs. Google Speech Recognition API (cloud-based)
- Offline models are typically less effective but don't require internet connectivity

### Key Takeaways
- Sound analytics involves transforming time-domain to frequency-domain for better interpretation
- Different features (spectrogram, MFCC) highlight different aspects of audio
- Tradeoff between offline (edge) and online (cloud) processing for speech recognition
- Virtual environments help avoid library conflicts

---

## Image Analytics with Raspberry Pi

### Key Concepts
- Edge Computer Vision (ECV) performs image processing on resource-constrained devices
- Benefits include real-time processing, enhanced privacy, reduced network dependency
- ECV recognized by Gartner as a top emerging technology of 2023

### Setup Process
1. **Environment preparation**:
   ```bash
   # Create and activate virtual environment
   sudo apt install python3-venv
   python3 -m venv image
   source image/bin/activate
   
   # Install OpenCV
   pip install opencv-python
   
   # Install scikit-image
   pip install scikit-image
   ```

### Image Processing Techniques

#### Color Segmentation
- Separating image into different colors (RGB) based on intensity ranges
- OpenCV functions to read frames from webcam and process by color channels

#### Feature Extraction
- Histogram of Gradients (HoG): Extracts gradient patterns from image patches
- Important considerations:
  - Converting RGB to grayscale before HoG feature extraction
  - Resizing impacts frame rate (downsampling speeds up computation)
  - Patch size affects feature granularity and processing speed

#### Face Detection and Landmark Extraction
- Using MediaPipe for lightweight, on-device ML processing:
  ```bash
  pip install mediapipe
  ```
- Benefits over traditional OpenCV methods:
  - Faster processing
  - Lightweight models using TensorFlow Lite
  - More accurate with fewer resources

- Alternative OpenCV approach using Haar Cascades:
  - Requires downloading model file (`haarcascade_frontalface_alt2.xml`)
  - Generally slower but doesn't require additional libraries

### Key Takeaways
- Resource constraints are critical in edge computing - image size affects performance
- Feature extraction (HoG) is fundamental to many computer vision tasks
- MediaPipe provides optimized models for edge devices
- Tradeoffs between accuracy, speed, and resource utilization

---

## Video Analytics with Raspberry Pi

### Key Concepts
- Video analytics extends image processing to temporal dimension
- Edge video analytics challenges: real-time processing with limited resources
- Applications include surveillance, motion detection, gesture recognition

### Setup Process
```bash
# Create and activate virtual environment (or reuse "image" environment)
sudo apt install python3-venv
python3 -m venv video
source video/bin/activate

# Install OpenCV
pip install opencv-python

# Install MediaPipe
pip install mediapipe
```

### Video Processing Techniques

#### Optical Flow
- Technique to track moving objects between frames
- Two approaches:
  1. Lucas-Kanade: Sparse optical flow (tracks specific points)
  2. Farneback: Dense optical flow (tracks all pixels)
- Visualized as streamlines or directional arrows
- Parameter tuning affects accuracy and performance

#### MediaPipe Applications

1. **Hand Landmark Detection**:
   ```bash
   # Download model
   wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```
   - Identifies 21 landmarks on each hand
   - Can be used to detect specific gestures or count fingers

2. **Hand Gesture Recognition**:
   ```bash
   # Download model
   wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
   ```
   - Pre-trained model recognizes common gestures (e.g., victory sign)

3. **Object Detection**:
   ```bash
   # Download EfficientDet model
   wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
   ```
   - Can detect common objects in video frames
   - Useful for video summarization or selective recording

### Key Takeaways
- Video analytics adds temporal dimension to image processing
- Optical flow is fundamental for motion tracking
- MediaPipe provides optimized, pre-trained models for various video tasks
- Applications range from gesture control to surveillance

---

## Real-time Inference of Deep Learning Models on Edge Device

### Key Concepts
- Edge devices have limited resources but privacy/security benefits
- Optimization techniques enable running deep learning models on constrained hardware
- Quantization reduces model size and increases inference speed

### Setup Process
```bash
# Create and activate virtual environment
sudo apt install python3-venv
python3 -m venv dlonedge
source dlonedge/bin/activate

# Install PyTorch and OpenCV
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy --upgrade
```

### Deep Learning Optimization Techniques

#### Model Performance Without Optimization
- MobileNetV2 without optimization achieves only 5-6 FPS on Raspberry Pi 4B
- Target performance: 30 FPS for real-time applications

#### Quantization
- Converting 32-bit floating-point weights/activations to 8-bit integers
- Reduces model size and computation requirements
- With quantization, MobileNetV2 approaches 30 FPS

#### Quantization Methods
1. **Post-Training Quantization**:
   - Applied after model training
   - Simpler to implement but may cause accuracy loss
   - Converts parameters from 32-bit float to 8-bit integer

2. **Quantization-Aware Training**:
   - Inserts fake quantization operators during training
   - Model learns to be robust to quantization
   - Generally better accuracy but requires retraining

### Key Takeaways
- Quantization is crucial for running deep learning models on edge devices
- Tradeoff between model accuracy and inference speed
- Post-training quantization is easier but quantization-aware training gives better results
- Target performance metrics (e.g., FPS) should guide optimization choices

---

## IoT Communications: MQTT

### Key Concepts
- MQTT (Message Queue Telemetry Transport): lightweight publish/subscribe messaging protocol
- Ideal for IoT due to small code footprint and minimal bandwidth requirements
- Runs over TCP/IP or similar network protocols

### MQTT Components
1. **MQTT Broker**: Intermediary that receives messages from publishers and delivers to subscribers
2. **Topic**: Namespace for messages (e.g., "test/topic")
3. **MQTT Client**: Device that publishes messages or subscribes to topics

### Setup Process

#### Installing and Configuring MQTT Broker
```bash
# Update package list
sudo apt update

# Install Mosquitto broker
sudo apt install mosquitto

# Edit configuration
sudo nano /etc/mosquitto/mosquitto.conf

# Add these lines to config file:
listener 1883
allow_anonymous true

# Start Mosquitto manually
sudo mosquitto -c /etc/mosquitto/mosquitto.conf
```

#### Enable Mosquitto to Run on Boot (Optional)
```bash
# Start and enable service
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# Restart to apply configuration
sudo systemctl restart mosquitto

# Check status
systemctl status mosquitto

# Disable and stop (if needed)
sudo systemctl disable mosquitto
sudo systemctl stop mosquitto
```

#### Setting Up MQTT Clients

1. **Install Python MQTT Library**:
   ```bash
   # Activate virtual environment
   source myenv/bin/activate
   
   # Install Paho MQTT
   pip install paho-mqtt
   ```

2. **Create Publisher Client**:
   ```python
   import paho.mqtt.client as mqtt
   import time

   client = mqtt.Client("Publisher")
   client.connect("localhost", 1883)  # Change "localhost" to broker IP

   while True:
       client.publish("test/topic", "Hello, MQTT!")
       time.sleep(5)
   ```

3. **Create Subscriber Client**:
   ```python
   import paho.mqtt.client as mqtt

   def on_message(client, userdata, message):
       print(f"Received message '{message.payload.decode()}' on topic '{message.topic}'")

   client = mqtt.Client("Subscriber")
   client.on_message = on_message
   client.connect("localhost", 1883)  # Change "localhost" to broker IP
   client.subscribe("test/topic")
   client.loop_forever()
   ```

### Testing MQTT Communication
1. Run subscriber in one terminal
2. Run publisher in another terminal
3. Observe messages being received by subscriber

### Key Takeaways
- MQTT uses a publish/subscribe model instead of request/response
- MQTT broker manages message routing between clients
- Topics organize messages and allow filtering by subscribers
- Connection configuration requires broker address (IP) and port (typically 1883)

---

## AWS IoT: Real-Time Device Data to AWS IoT Core

### Key Concepts
- AWS IoT Core: Cloud service for connecting edge devices
- Supports large volume of messages with routing to AWS services
- Communication protocols: MQTT, HTTP, WebSocket

### Setup Process

#### Creating IoT Thing in AWS IoT Core
1. Go to AWS IoT Core console (region: ap-southeast-1)
2. Navigate to Manage > All Devices > Things
3. Create single thing with a unique name
4. Auto-generate certificate
5. Create IoT policy with appropriate permissions
6. Download security files:
   - Device Certificate (aws-certificate.pem.crt)
   - Public Key File (aws-public.pem.key)
   - Private Key File (aws-private.pem.key)
7. Attach certificate to IoT Thing and IoT policy

#### Setting Up Raspberry Pi for AWS IoT Core
```bash
# Create and activate virtual environment
sudo apt install python3-venv
python3 -m venv awsiotcore
source awsiotcore/bin/activate

# Install required libraries
pip install paho-mqtt
pip install --upgrade psutil
```

#### Connecting to AWS IoT Core via MQTT
- Download required files:
  - Python script (pipython.py)
  - Root CA certificate
  - Three security files from AWS
- Update endpoint in script:
  ```python
  client.connect("xxxxxxxx-ats.iot.ap-southeast-1.amazonaws.com", 8883, 60)
  ```
- Test connection by subscribing to topic "device/data" in AWS IoT MQTT Test Client
- Run Python script and verify data appears in MQTT console

#### Ingesting Data into DynamoDB via IoT Rule
1. Create a rule in AWS IoT Core
2. Set up SQL statement to select data from MQTT topic
3. Create DynamoDB table with appropriate primary key (hostname) and sort key (timestamp)
4. Configure rule action to insert data into DynamoDB
5. Create or select IAM role with necessary permissions
6. Run Python script to send data
7. Verify data appears in DynamoDB table

### Key Takeaways
- AWS IoT Core provides secure connectivity for edge devices
- Certificate-based authentication ensures secure communication
- IoT rules enable automated processing of device data
- DynamoDB provides scalable storage for time-series device data

---

## Raspberry Pi 400 Setup with Webcam

### Key Concepts
- Raspberry Pi 400 is an all-in-one keyboard computer
- Can be configured for headless operation via SSH and VNC
- Supports webcams for image and video capture

### Setup Process

#### Installing Raspberry Pi OS
1. Download and install Raspberry Pi Imager
2. Configure OS settings (hostname, username, password, WiFi)
3. Enable SSH for headless access
4. Write OS to microSD card

#### Initial Configuration
```bash
# Update system
sudo apt update
sudo apt upgrade

# Enable VNC
sudo apt install haveged  # Optional: for headless setup
sudo raspi-config  # Navigate to Interfacing Options > VNC > Yes
```

#### Webcam Setup and Testing

1. **Connect Webcam** and verify detection:
   ```bash
   lsusb
   ```

2. **Image Capture**:
   ```bash
   # Install fswebcam
   sudo apt install fswebcam
   
   # Capture image
   fswebcam -r 1280x720 --no-banner image.jpg
   ```

3. **Video Recording**:
   ```bash
   # Install ffmpeg
   sudo apt install ffmpeg
   
   # Record video
   ffmpeg -f v4l2 -framerate 25 -video_size 640x480 -i /dev/video0 output.mp4
   
   # Play video
   sudo apt install vlc
   vlc output.mp4
   ```

4. **Audio Recording** (if webcam has microphone):
   ```bash
   # List audio devices
   arecord -l
   
   # Record audio (replace card/device numbers)
   arecord -D plughw:2,0 -d 10 test.wav
   
   # Play audio
   aplay test.wav
   ```

#### Python Environment Setup
```bash
# Install virtual environment
sudo apt install python3-venv

# Create virtual environment
python3 -m venv myenv

# Activate virtual environment
source myenv/bin/activate

# Install OpenCV in virtual environment
pip install opencv-python

# Upgrade pip
pip install --upgrade pip
```

### Python Scripting for Webcam Applications

#### Basic Image Capture
- Python with OpenCV can access webcam feeds and process frames
- Example for capturing a single frame:
  ```python
  import cv2
  
  # Initialize webcam
  cap = cv2.VideoCapture(0)
  
  # Check if webcam opened successfully
  if not cap.isOpened():
      print("Error: Could not open webcam")
      exit()
  
  # Capture a single frame
  ret, frame = cap.read()
  
  # Save the frame
  if ret:
      cv2.imwrite('captured_image.jpg', frame)
      print("Image captured successfully")
  else:
      print("Failed to capture image")
  
  # Release the webcam
  cap.release()
  ```

#### Motion Detection
- More advanced application using frame differencing
- Core steps:
  1. Capture reference frame
  2. Capture current frame
  3. Calculate absolute difference between frames
  4. Apply thresholding and dilation to enhance motion areas
  5. Find contours in the processed difference
  6. Draw bounding boxes around detected motion

- Key OpenCV functions:
  - `cv2.absdiff()`: Calculate absolute difference between frames
  - `cv2.cvtColor()`: Convert between color spaces (RGB to grayscale)
  - `cv2.GaussianBlur()`: Reduce noise before processing
  - `cv2.threshold()`: Binarize image based on pixel intensity
  - `cv2.dilate()`: Enhance and connect regions of interest
  - `cv2.findContours()`: Identify boundaries of detected motion

### Key Takeaways
- Raspberry Pi can be set up headless using SSH and VNC
- Virtual environments help manage Python package dependencies
- Webcams can be accessed through command-line tools or Python libraries
- OpenCV provides powerful functions for real-time image/video processing
- Motion detection requires comparing sequential frames and identifying differences
- Processing parameters (blur, threshold, contour area) affect detection sensitivity

---

## Study Checklist for Quiz Preparation

### Sound Analytics
- [ ] Explain the purpose and applications of sound analytics on edge devices
- [ ] List the key Python libraries for audio processing (PyAudio/SoundDevice, Librosa)
- [ ] Describe the difference between time-domain and frequency-domain representations
- [ ] Explain the purpose of audio features: spectrogram, chromogram, Mel-spectrogram, MFCC
- [ ] Compare offline (edge) vs. online (cloud) speech recognition

### Image Analytics
- [ ] Define Edge Computer Vision (ECV) and its benefits
- [ ] Explain the process of color segmentation using OpenCV
- [ ] Describe the Histogram of Gradients (HoG) feature and its applications
- [ ] Compare MediaPipe and traditional OpenCV methods for face detection
- [ ] Explain how image resizing impacts processing performance

### Video Analytics
- [ ] Describe the challenges of video analytics on edge devices
- [ ] Explain optical flow and its two main approaches (Lucas-Kanade and Farneback)
- [ ] List MediaPipe models used for hand tracking, gesture recognition, and object detection
- [ ] Explain how to extract landmarks from detected hands
- [ ] Describe a practical application of video analytics (e.g., gesture control)

### Deep Learning on Edge
- [ ] Define model quantization and explain its importance for edge devices
- [ ] Compare post-training quantization vs. quantization-aware training
- [ ] Explain how quantization affects model size and inference speed
- [ ] Describe the performance metrics for real-time inference (FPS)

### MQTT
- [ ] Define MQTT and explain why it's suitable for IoT applications
- [ ] List and describe the three main components of MQTT
- [ ] Explain the publish/subscribe model vs. request/response
- [ ] Outline the steps to set up an MQTT broker and clients
- [ ] Describe the structure of MQTT topics and their purpose

### AWS IoT Core
- [ ] Explain the purpose of AWS IoT Core and its integration with edge devices
- [ ] List the steps to create an IoT Thing in AWS IoT Core
- [ ] Describe the security mechanisms for AWS IoT (certificates)
- [ ] Explain how IoT Rules route data to AWS services
- [ ] Describe the process of ingesting IoT data into DynamoDB

### Raspberry Pi Setup
- [ ] List the steps to set up a Raspberry Pi headless using SSH and VNC
- [ ] Explain how to capture images and video using command-line tools
- [ ] Describe the process of creating and using Python virtual environments
- [ ] Explain the core components of motion detection using OpenCV

---

## Sample Quiz Questions

1. **Q**: What is the primary advantage of using a virtual environment for Python projects on Raspberry Pi?
   **A**: It isolates project dependencies, preventing conflicts between packages installed via `apt` and `pip`, and allows different projects to use different versions of the same package.

2. **Q**: Explain the difference between Lucas-Kanade and Farneback optical flow methods.
   **A**: Lucas-Kanade is a sparse optical flow method that tracks specific points between frames, while Farneback is a dense optical flow method that computes motion for all pixels in the frame.

3. **Q**: Why is model quantization important for deep learning on edge devices?
   **A**: Quantization reduces model size and computation requirements by converting 32-bit floating-point weights and activations to 8-bit integers, enabling faster inference with less memory usage.

4. **Q**: In MQTT communication, what is the role of the broker?
   **A**: The broker receives messages from publishers, manages topic subscriptions, and delivers messages to subscribers who have subscribed to the relevant topics.

5. **Q**: How does the Mel-spectrogram differ from a regular spectrogram?
   **A**: A Mel-spectrogram uses the Mel scale (perceptual scale of pitches) instead of frequency on the y-axis and uses the decibel scale for color intensity, making it better aligned with human hearing perception.

6. **Q**: What are the three security files required to connect a device to AWS IoT Core?
   **A**: Device Certificate (aws-certificate.pem.crt), Public Key File (aws-public.pem.key), and Private Key File (aws-private.pem.key).

7. **Q**: What OpenCV functions are typically used in a basic motion detection pipeline?
   **A**: `cv2.absdiff()` for frame comparison, `cv2.cvtColor()` for grayscale conversion, `cv2.GaussianBlur()` for noise reduction, `cv2.threshold()` for binarization, `cv2.dilate()` for enhancement, and `cv2.findContours()` for motion area detection.

8. **Q**: Compare the strengths and weaknesses of edge-based vs. cloud-based speech recognition.
   **A**: Edge-based (like CMUSphinx) works offline but is less accurate; cloud-based (like Google Speech API) is more accurate but requires internet connectivity and may have privacy implications.

9. **Q**: What is the purpose of IoT Rules in AWS IoT Core?
   **A**: IoT Rules allow automatic processing of messages from devices, enabling routing of data to AWS services like DynamoDB, Lambda, or S3 based on message content or topic.

10. **Q**: Explain why MediaPipe is particularly suitable for edge devices like Raspberry Pi.
    **A**: MediaPipe provides lightweight, optimized machine learning models using TensorFlow Lite, designed specifically for on-device inference with minimal resource requirements while maintaining good accuracy.
