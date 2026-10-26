# Implementation Plan: AI-Powered Foreign Object Debris (FOD) Detection System

## 1. Introduction
A strict six-day execution timeline is established by this document to ensure all technical requirements for the OpenCV AI Competition 2026 are met prior to the October 26 deadline. The implementation is divided into daily milestones, with strict definitions of done to prevent scope creep. Critical technical directives are provided to mitigate known risks during the build phase.

## 2. Daily Execution Schedule

### Day 1: Physical Mock-up and OpenCV Core Mathematics
*   **Core Engineering Tasks:**
    1. A mock engine bay must be constructed using a cardboard box, tangled wires, and assorted hardware to simulate a dense mechanical environment.
    2. A local Python script must be written to capture frames from a local webcam or smartphone.
    3. The captured RGB frames must be converted to edge-space using OpenCV edge detection algorithms, such as `cv2.Canny()` or `cv2.Sobel()`.
    4. Absolute difference calculations (`cv2.absdiff()`) must be implemented to compare two static edge-space images.
*   **Definition of Done:** 
    The background engine structure is successfully subtracted, and a dropped tool is mathematically isolated by the local Python script using edge outlines. Color is entirely ignored by the algorithm.

### Day 2: Pose Matching and Vision-Language Model (VLM) Testing
*   **Core Engineering Tasks:**
    1. Feature extraction and matching algorithms must be implemented using `cv2.ORB_create()` and `cv2.BFMatcher()`.
    2. Logic must be programmed to search a continuous "After" video feed for a specific frame that geometrically matches a saved "Before" frame.
    3. Homography matrices (`cv2.findHomography()`) must be calculated to perfectly align the matched image pair, correcting for minor perspective shifts.
    4. A minimal Python script utilizing the AWS `boto3` library must be written to transmit the cropped anomaly to Amazon Bedrock (Claude 3.5 Sonnet).
*   **Definition of Done:** 
    A slightly misaligned photograph is successfully aligned, the background is subtracted, the anomaly is cropped, and a "FOD" or "PART" classification is successfully returned and printed in the terminal by the AWS Bedrock API.

### Day 3: Amazon Web Services (AWS) Backend Deployment
*   **Core Engineering Tasks:**
    1. An AWS EC2 Graviton (ARM64) instance (such as `c7g.medium` or `t4g.small`) must be provisioned.
    2. The Cloud-Optimized OpenCV Library (COOL) must be deployed via the AWS Marketplace, or OpenCV must be compiled from source with ARM optimizations (`-D ENABLE_NEON=ON`) enabled.
    3. The Python scripts developed during Day 1 and Day 2 must be wrapped into a FastAPI backend service.
    4. Application Programming Interfaces (APIs) for `/upload_baseline`, `/upload_after`, and `/evaluate_fod` must be exposed to the internet.
*   **Definition of Done:** 
    The entire OpenCV processing pipeline and AWS Bedrock API communications are executed securely on the AWS Graviton cloud instance and are accessible via REST API calls from a local machine.

### Day 4: Agentic User Interface (UI) and Panoramic Stitching
*   **Core Engineering Tasks:**
    1. A minimal web frontend must be constructed using React, Next.js, or plain HTML/JavaScript.
    2. A local processing loop must be implemented to feed sampled video frames into the OpenCV `Stitcher` module to continuously generate a 2D panorama.
    3. Agentic logic must be programmed to calculate the filled pixel area of the stitched panorama to estimate the percentage of visual coverage achieved by the mechanic.
    4. The Web Speech API must be integrated to provide real-time audio alerts to the user.
*   **Definition of Done:** 
    A continually expanding panoramic image is displayed by the UI as the camera is moved. A voice warning is spoken by the browser if the user attempts to complete the task with less than 90% visual coverage.

### Day 5: System Integration and the Lockout Loop
*   **Core Engineering Tasks:**
    1. The frontend UI must be connected to the AWS FastAPI backend.
    2. The complete state machine must be integrated: *Baseline Scan -> Working -> Post-Work Scan -> VLM Evaluation -> Lockout/Approval*.
    3. The UI must be refined to display a prominent red "Access Denied" screen if FOD is identified by the AWS Bedrock endpoint.
    4. The system must be stress-tested under varying lighting conditions and erratic camera movements.
*   **Definition of Done:** 
    The end-to-end loop is fully functional. A tool can be dropped into the mock-up, the system can be rescanned, and the UI lockout is successfully triggered without any new code being written or terminal commands being manually entered.

### Day 6: Demonstration Recording and Devpost Submission
*   **Core Engineering Tasks:**
    1. The official Devpost Technical Report must be drafted. The architecture, OpenCV implementations, and AWS Graviton metrics must be explicitly detailed.
    2. A continuous, unedited 3-minute demonstration video must be filmed.
    3. The system architecture diagram must be finalized and exported.
    4. Clean, well-documented source code must be pushed to a public GitHub repository, accompanied by a comprehensive `README.md` file.
*   **Definition of Done:** 
    The hackathon project is officially submitted to the OpenCV Devpost portal with the video, source code, and architecture diagram successfully attached prior to the 11:59 PM PDT deadline.

---

## 3. Technical Execution Directives

### 3.1. OpenCV Stitching Performance Constraints
The `cv2.Stitcher` class is computationally expensive. If an attempt is made to stitch 30 frames per second, the local device will freeze. 
*   **Mandatory Mitigation:** The video feed must be sampled. Only 1 frame every 0.5 seconds is to be sent to the stitcher. The frames must be downscaled to 480p before stitching is attempted. This maintains real-time UI responsiveness for the Agentic coverage map.

### 3.2. AWS Access and Graviton Compilation
Delays in cloud provisioning cannot be afforded during a 6-day build.
*   **Mandatory Mitigation:** Model access for Claude 3.5 Sonnet must be explicitly requested in the AWS Bedrock console on Day 1. If the official COOL Amazon Machine Image (AMI) is difficult to configure quickly, a standard Ubuntu ARM64 image must be launched, and the OpenCV compilation process must be documented to fulfill the judging rubric for cloud optimization.

### 3.3. Bypassing Vision-Language Model Latency
Sending an image to AWS Bedrock takes approximately 2 to 4 seconds. During a live demonstration, an idle screen creates the perception of a broken application.
*   **Mandatory Mitigation:** Optimistic UI loading states must be built. The moment the "After" scan finishes, the UI must immediately transition to an "Analyzing Engine State" screen featuring a spinning radar animation. An audio file stating, "Scanning for Foreign Object Debris," must be played. The API latency is perfectly masked by this sequence.

### 3.4. Hardware Setup Protocol
The physical camera input must be standardized to prevent tracking errors.
*   **Mandatory Mitigation:** The smartphone or webcam must be securely mounted to a standard flashlight using heavy tape or zip-ties. During Day 1 and Day 2 testing, a taped boundary must be established on a desk to act as the "Engine Bay." The testing environment must be rigidly controlled until Day 5. Chaotic lighting and erratic movement must only be introduced once the core mathematical pipeline is frozen and verified.
