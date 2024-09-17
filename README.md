# Ad-Matay

An app that allows skipping commercial break on recorded TV shows by using computer vision elements.

The app relies on the prior knowledge we have on the TV channels, which changes the icons and/or presenting timers on commercial breaks.
The app allows capturing the content of the TV in several ways (TBD below),
then uses YOLO and CV methods to isolate the TV and the ROIs.
The ROIs for our case are the top-left, top-right corners, which are then compared to the "ad icons" of the channels.
There is an optional functionality for OCR to get the exact skipping required, but it is currently too slow and therefore commented-out.


![](https://github.com/omer-re/Ad-matay/blob/637030a5850c258a6f2eb2b0d4387b3e9cbdd8ea/demo_images/Ad-Matay%20simple%20demo%20image.png)

It includes a testing GUI to examine the process that is also presenting the stages:
![](https://github.com/omer-re/Ad-matay/blob/637030a5850c258a6f2eb2b0d4387b3e9cbdd8ea/demo_images/demo_works_7sec_delay.png)

## TLDR & Block diagram

The app is designed to be highly modular, and each class was built to allow easy integration for components.
I have used *consumer-producer* design pattern, and every stage (class) in the process is reading from a cyclic queue, and outputs to another cyclic queue.
That architecture offers several benefits:
- allows users to both use my classes in other apps with easy integration.
- also, adding other modules to use the input in parallel (the queues are readable, input source will not be held preemptively).
For example, another app that preform parental control can use the input simultaneously.

- It allows using parallelism relatively easy as there's no need for many shared resources.
- Using the queues has the by-product of "ZOH" (zero order hold), meaning that we can still hold the last valid input until new updates arrives.
- I chose using multithreading over multiprocessing as some context is shared, but converting it to multiprocessing can be relatively simple (mainly adapting the queues mechanism).

<img src="https://github.com/omer-re/Ad-matay/blob/637030a5850c258a6f2eb2b0d4387b3e9cbdd8ea/demo_images/ad_matay_scheme.png"/>

#### Why using a single worker per class?
From the nature of the challenge I am trying to solve, there's no much value for a true real-time process.
Sampling in a frequency of once a second (or even a once a few seconds) is sufficient for those needs.
I am using a very small queue (N=1,2) as I couldn't justify piling up frames if I can't process them on the next module in a timely manner.
The RPI5 has its limitations, and the tv detection plus lpr are taking about 2-3 seconds at the moment.

Further down the roadmap I will convert it to use the Hailo kit and then I expect it to work faster, which will then derive rethinking the workers distribution.


## Frame fetcher
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


## TV detector
Given a frame from a camera which contains the TV, we'd like to detect the corners of the screen in order to normalize and transform.
That's not that simple, as even YOLO8 segmentation models tend to miss some of the TV, even in a relatively good conditions.
Unfortunately, the misses are usually on the top-bottom edges, where most of our data is:


