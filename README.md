# Ad-Matay

An app that allows skipping commercial break on recorded TV shows by using computer vision elements.

I first thought of it about 2 years ago, but I ditched it too fast to be significant. Back then I made a little presentation of my blueprints in order to consult with friends about it, [it is still available here](https://youtu.be/E45f3Y3l9NQ?si=hlVrtUqbBtFeZCYh)

The app relies on the prior knowledge we have on the TV channels, which changes the icons and/or presenting timers on commercial breaks.
The app allows capturing the content of the TV in several ways (TBD below),
then uses YOLO and CV methods to isolate the TV and the ROIs.
The ROIs for our case are the top-left, top-right corners, which are then compared to the "ad icons" of the channels.
There is an optional functionality for OCR to get the exact skipping required, but it is currently too slow and therefore commented-out.


![](demo_images/Ad-Matay%20simple%20demo%20image.png)

It includes a testing GUI to examine the process that is also presenting the stages:

![](https://github.com/omer-re/Ad-matay/blob/892f3eb6742470a26b130ccf675cad70049b63a9/demo_images/app_demo_gui_mute.gif)

![LINK TO VIDEO OF THAT GIF](https://github.com/omer-re/Ad-matay/blob/892f3eb6742470a26b130ccf675cad70049b63a9/demo_images/app_demo_gui_mute.mp4)

## Table of Contents

1. [Key concepts](#key-concepts)
2. [Design](#design)
3. [High-level block diagram](#high-level-block-diagram)
   - [Why using a single worker per class?](#why-using-a-single-worker-per-class)
   - [Why Threads over Processes?](#why-threads-over-processes)
4. [Setup](#setup)
   - [Setting up with Linux packages](#setting-up-with-linux-packages)
   - [Setting up with conda](#setting-up-with-conda)
5. [Modules](#modules)
   - [Frame fetcher](#frame-fetcher)
     - [Why not using the ADB snapshot as default?](#why-not-using-the-adb-snapshot-as-default)
   - [TV detector](#tv-detector)
     - [Multiframe approach](#multiframe-approach)
   - [LPR](#lpr)
   - [TV Controller](#tv-controller)
6. [Future Features](#future-features)
   

# Key concepts

I am a true believer in the ability to innovate using existing building blocks.
Modern developer's ability to use existing pieces of code and methods, connecting them together in a never-seen-before way to create value.
I find it fascinating when I run into such plug&play, well built, well explained repo, and therefore I tried to be one.

**This can be generalized to a template for "vision triggered apps":**
1. Frames are captured from various sources.
2. Enhancing, normalizing frames using CV techniques or algebraic transformations.
3. Feeding frames into first stage which uses fast methods (like YOLO, CV techniques) for initial detection and reducing frames to ROI. 
This stage requires contextual planing and trade-offs, with some prior knowledge (color spaces for example).
4. Applying advance, usually slower, methods on the minimized ROIs (like OCR). Minimizing ROIs support both noise-reduction (ignoring false detection) and performances (less area to scan) purposes.
5. According to results, apply relatively simple logic to trigger actions.

This time it is skipping ads, but it could be easily converted for detecting bus lines numbers, Wolt bikers hiding their IDs or opening a paid parking gate for members' cars only.


## Design

The app is designed to be highly modular, and each class was built to allow easy integration for components.
I have used *consumer-producer* design pattern, stages have easy communication between them using input and output cyclic queues.

That architecture offers several benefits:

 - Allows users to easily integrate my classes in other apps due to simple i/o.
 - Adding other modules to use the input in parallel (the queues are readable, input source will not be held preemptively).
For example, another app that preform parental control can use the input simultaneously.
 - Cyclic queues' size can be adjusted, they limit memory resources consumptions even if there are pace differences between stages.
That is obviously a context related tradeoff I made, while other needs might require 0 frame drops which derives different solutions.
 - It allows using parallelism relatively easy as there's no need for many shared resources.
 - Using the queues has the by-product of "ZOH" (zero order hold), meaning that we can still hold the last valid input until new updates arrives.
 - **REAL OOP**: each class has its own `main()` which allows running and testing it separately and independently.


## High-level block diagram 

Attached is an explained scheme:
[Link to PDF for better view](https://drive.google.com/file/d/14PyFBqQjhArn8FWC8BHDP_0aOVB6A2bj/view?usp=sharing)
<img src="https://i.imgur.com/DPVruKH.png" width="800"/>
<img src="demo_images/Ad-Matay%20simple%20demo%20image.png" width="800"/>

#### Why using a single worker per class? 
From the nature of the challenge I am trying to solve, there's no much value for a true real-time process.
Sampling in a frequency of once a second (or even a once a few seconds) is sufficient for those needs.
I am using a very small queue (N=1,2) as I couldn't justify piling up frames if I can't process them on the next module in a timely manner.
The RPI5 has its limitations, and the tv detection plus lpr are taking about 2-3 seconds at the moment.

#### Why Threads over Processes?
1. My app's modules were built to be a part of a larger app, and therefore hogging resources won't be cool at all. Keeping it as a single process will allow lighter HW to use it and also other developers to use multiprocessing in their larger app.
2. Further down the roadmap I will convert it to use the Hailo kit especially for improving performances and timing, on their documentation they refer to multithreading rather than multiprocessing, so I expect the upgrade and transition to be better this way.
Anyhow, future improvements will most likely derive rethinking the workers' distribution.


# Setup
Please refer to `set_up` dir and create your virtual env. I'd rather use conda, but a venv should do as well.
I have added the packages (`apt instal...`) list of my device as well, as I ran into some errors while creating it.
If your env isn't working, consider installing my packages as well.

#### Setting up with Linux packages

1. Go to: `cd <repo root>/set_up/install_packages_ <with or without> _versions.sh`
2. Set permissions: `chmod +x install_packages_ <with or without> _versions.sh`
3. `./install_packages_ <with or without> _versions.sh`

#### Setting up with conda

1. Install miniforge or some conda suits your devices.
2. Create virtual env by `conda env create -f set_up/environment.yml`.


## Modules
Please note that each module contains `main()` function to allow testing its basic functionalities independently,
meaning that you can run `(<conda _env>) python tv_detector.py` for example.

### Frame fetcher
This module's responsibility is taking the input source and push frames of it into the frame_queue.
Input sources can vary: USB camera, IP Camera, Pre-recorded video file, HDMI stream, ADB snapshots.
The purpose is to allow the users to use whatever is easier for them to do.

USB camera, IP Camera, Pre - recorded video file - are treated as "raw", and will have to be processed by the tv detector to get them "normalized" into perfect tv segmented rectangle.
HDMI stream, ADB snapshots - are the "normalized" input, and they are (obviously) offering much easier detection due to their superior quality.
In case of HDMI stream, ADB snapshots most of the tv_detector process will be skipped.

#### Why not using the ADB snapshot as default?
Sending snapshot commands via adb gives astonishing frames for inputs, with crisp images and even high enough frame rate.
However, many of the content apps for android TV uses SAFE_FLAG to prevent users from taking screenshots of the content.
The results are usually black screen with only system icons on it (volume indicator for example).
That block makes the ADB method to be unreliable for most crowd, but a GREAT option for some (in case the app developers you're using have missed it, or if you are using a rooted android).

![](https://github.com/omer-re/Ad-matay/blob/4b9757d5fa1b327cdb24febb1d20a9f4552e2b57/demo_images/adb_locked.png)


### TV detector
Given a frame from a camera which contains the TV, we'd like to detect the corners of the screen in order to normalize and transform.
That's not that simple, as even YOLO8 segmentation models tend to miss some of the TV, even in a relatively good conditions.
Unfortunately, the misses are usually on the top-bottom edges, where most of our data is:
Using bigger YoloV8 models had the same problem, therefore I returned using nano.

![](https://github.com/omer-re/Ad-matay/blob/4b9757d5fa1b327cdb24febb1d20a9f4552e2b57/demo_images/mask_tv_challenge4.png)

The solution I have used is a multi-stage approach, where I use the Yolo segment but also refining it with basic CV methods of corner detection.
Once corners are detected, we can ues perspective transform to make the side-view a straight rectangle,
cropping the TV, passing it as an ROI, ignoring the rest.

- Note: in order to prevent false detection of objects on frame (including TVs on what's shown on TV) I am picking the largest TV from all TV detections (by area).

#### Multiframe approach
Assuming enough frames, we can segment the TV by averaging frames and looking for diffs.
Problem is that on many cases, like news for example, much of the frame is static and therefore diffs aren't shown.
It can be relevant if we'd like to trade time for compute resources, as it can be applied on a much lighter HW, not using YOLO or such ML/DL methods.
Diffing>Averaging>Masking it or watershedding it.

![](https://github.com/omer-re/Ad-matay/blob/637030a5850c258a6f2eb2b0d4387b3e9cbdd8ea/demo_images/50_frames_buffer.png)


### LPR
This module is built to indicate whether we are currently watching ads or not.
It is relying on the different icons on the top corners and the timer that is usually presented, counting down back to the show.

Initially I have tried to implement that model using License Plates Recognition, therefore the name.
LPR has much in common with the need to "understand" what's the timer on the top-left corner is showing.
I have struggled getting it to preform well and therefore it is currently commented out, meaning that the indication is just binary, whether we're on ads or not.
(For example, Mnist, EasyOCR, Tesseract are too slow).

There are more "old school" techniques I consider that might resolve it, for example:
    - KNN for numbers, which are usually standard font.
    - OCR+refinement for time properties like MM=[0,59]
    - OCR with "clock obeys time rules" meaning that if it is currently :15,  next sample has X seconds gap,
    we should expect something in the areas of :15-X seconds.
    - Old school: Detecting ":" or sampling frequency to detect areas that changes **EXCATLY** every 1 second,
    considering it as the font color. applying TH/colorspace mask for this color, then we expect text to be contrasted and clearer.


The detection is done for each corner independently, 
each frame's corners are compared to the collection of references I collected in advance.
The comparison is done by feature extraction with Meta's Dino, then comparing the cosine similarity of their vectors.

For some noise reduction, I designed it to toggle between Ad - Non-ad only after some N consecutive frames of the same state.



### TV Controller
Once we detected a commercial break, we'd like to skip it or notifying for it (playing sound alerting that commercial break has passed)
We can transmit it to the TV using several ways, each simulated the remote control action in a different way:
- **IR (infra-red) -** good old remote control method. Recording the TV's signal once then reusing it.
  - Pros: 99% of the TVs has IR receiver.
  - Cons: Requires another HW (even if cheap). Requires setting it up for each specific TV model.
- **ADB command -** simulating press is a basic adb command that can be used.
  - Pros: such commands are easy to use and are usually standard.
  - Cons: Requires being on the same local network as the TV.
- **Bluetooth -** Given that many remote controllers are a BT device, we can mimic their actions.
  - Pros: Very standard protocol allows easy setup.
  - Cons: Requires pairing it like a new remote.
- **Virtual keyboard -** This way is simulating having a keyboard connected to the TV's USB, then relying on the standard protocols to pass "fast forward" button.
  - Pros: Robust way to communicate, should be a plug&play thing.
  - Cons: Requires direct USB connection from the RPI to the TV, which might limit us, especially if using a usb camera which also requires considering were to place the camera and RPI.


## Future Features

 - Improving timings using faster CV elements and simplified OCR (techniques like KNN might offer interesting approach).
 - Enabling OCR of the top-left corner in relevant timing (<2 secs/frame).
 - Finding a way to bypass adb SAFE_FLAG to be able to get input using adb. Accessibility features of android or some privileged android app might resolve it.
 - Refactoring stages to use Hailo's LPR demo, harnessing the RPI's AI Hat's abilities. 