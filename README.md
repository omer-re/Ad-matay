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

![](https://github.com/omer-re/Ad-matay/blob/637030a5850c258a6f2eb2b0d4387b3e9cbdd8ea/demo_images/ad_matay_scheme.png)

### Why using a single worker per class?
From the nature of the challenge I am trying to solve, there's no much value for a true real-time process.
Sampling in a frequency of once a second (or even a once a few seconds) is sufficient for those needs.
I am using a very small queue (N=1,2) as I couldn't justify piling up frames if I can't process them on the next module in a timely manner.
The RPI5 has its limitations, and the tv detection plus lpr are taking about 2-3 seconds at the moment.

Further down the roadmap I will convert it to use the Hailo kit and then I expect it to work faster, which will then derive rethinking the workers distribution.



