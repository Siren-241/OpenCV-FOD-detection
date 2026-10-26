# Project Documentation: AI-Powered Foreign Object Debris (FOD) Detection System

## 1. Project Overview and Problem Statement
Foreign Object Debris (FOD) is defined as any loose item, such as a wrench, bolt, or rag, that is left inside an aircraft engine after maintenance. Catastrophic engine failure is caused by FOD if the engine is started. Millions of dollars are lost, and lives are endangered by these events. 

Currently, tools are tracked by the aviation industry using foam shadow boards and manual counts. However, mistakes are made by humans. A missing specialized tool is sometimes replaced by a common bolt to make a shadow board look full. A small socket is often left inside the engine while the wrench handle is returned. These anomalies are frequently missed by tired supervisors. A more reliable system is required.

## 2. Project Vision
The jet engine nacelle (housing) is treated by this project as a physical workspace that must be mathematically verified. Human memory and manual counting are entirely removed from the verification process. 

The vision is to ensure that the physical state of the engine before the maintenance is perfectly compared to the physical state of the engine after the maintenance. If a foreign object is introduced, it is caught by the system immediately. The mechanic is physically prevented from signing off the work order until the debris is removed. The safety of the aircraft is guaranteed by computer vision algorithms and cloud-based Artificial Intelligence (AI), rather than by human observation.

## 3. The Proposed Solution (Architecture)
A strict, minimal architecture is utilized to solve this problem. The solution is divided into hardware input, geometric processing, and semantic reasoning.

### 3.1. Hardware Input
Smartphones are not waved inside the engine. Instead, a Wi-Fi camera is attached to a standard mechanic's inspection flashlight. The engine crevices are illuminated and filmed by the mechanic using this wand. The video feed is transmitted to a local tablet or laptop.

### 3.2. Agentic Guidance Loop
Total visual coverage is ensured by an AI agent. As the flashlight is moved by the mechanic, a 2D panoramic map of the engine is stitched together by OpenCV 5. The map is displayed on the tablet. Unseen areas are shown as blank spaces. If a blind spot is detected by the AI agent, voice instructions are issued to the mechanic (e.g., "A gap is detected on the left. The wand must be moved lower"). The human is actively directed by the AI.

### 3.3. Geometric Layer (OpenCV 5 and AWS Graviton)
Custom machine learning models are not trained. Instead, mathematical subtraction is used.
*   **Keyframe Pose Matching:** Video frames from the "Before" scan are saved. When the "After" scan is performed, exact matching camera angles are found by the system.
*   **Edge-Space Diffing:** The matched images are not compared using raw colors. Shadows and lighting changes are ignored. The images are converted into structural edge outlines by OpenCV 5.
*   **Subtraction:** The "Before" edges are mathematically subtracted from the "After" edges by the algorithm. The background engine components are canceled out to zero. Only new, changed objects are left in the image.
*   **Cloud Processing:** These heavy matrix calculations are processed by the Cloud-Optimized OpenCV Library (COOL) on AWS Graviton servers.

### 3.4. Semantic Layer (AWS Bedrock)
The isolated new object is cropped from the image. The cropped image is sent to AWS Bedrock. The image is analyzed by a foundational Vision-Language Model (VLM), such as Claude 3.5 Sonnet. A prompt is given to the VLM to classify the item. The item is classified as either a "Valid Part" (a legitimate repair) or "FOD" (a dangerous tool). If FOD is detected, the digital sign-off is blocked by the system, and a red alert is displayed.

## 4. Expected Deliverables (6-Day Build Phase)
Because this project must be completed in exactly 6 days, the deliverables are strictly defined. The following items will be created:

*   **Physical Mock-Up:** A simulated engine bay is built using a cardboard box, tangled wires, and scrap metal. 
*   **Capture Script:** A Python application is written to capture the video feed from a webcam or smartphone camera. 
*   **OpenCV Processing Pipeline:** The Edge-Space Diffing and Keyframe Pose Matching logic is programmed using OpenCV.
*   **AWS Integration:** The image processing workload is deployed to an AWS Graviton instance. An API connection to AWS Bedrock is established for the semantic VLM classification.
*   **Agentic User Interface (UI):** A simple web page is built. The 2D panoramic coverage map is shown to the user. The AI voice alerts and the final "Approved / Denied" lockout screens are displayed by this UI.
*   **Video Demonstration:** A 3-minute continuous video is recorded. The mock-up is scanned, a tool is dropped inside, the engine is rescanned, and the system lockout is successfully demonstrated.

## 5. MVP Limitations
To ensure the project is finished within the 6-day timeframe, certain technical compromises are accepted. The Minimum Viable Product (MVP) is restricted by the following limitations:

*   **No 3D Rendering:** A full 3D voxel map is not built. Complex 3D rendering is too slow for the required timeframe. It is replaced by a 2D panoramic stitched image to show the mechanic's visual coverage.
*   **Lighting Constraints:** Absolute darkness cannot be handled by the current edge-detection script. A moderate amount of ambient light or a strong flashlight must be used during the demonstration.
*   **Cloud Latency:** The VLM classification is not performed on the local edge device. The cropped image is sent over the internet to AWS Bedrock. A delay of 2 to 4 seconds is expected and accepted.
*   **Single Camera Input:** Multiple camera streams are not supported. Only one video wand is processed by the system at a time.
*   **No Custom Training:** A custom neural network is not trained by the team. The system is completely reliant on the pre-existing knowledge of the AWS Bedrock foundational models.

## 6. Conclusion
A critical industrial safety problem is solved by this system. The architecture is kept minimal. The heavy lifting is done by OpenCV 5 geometry and AWS Bedrock reasoning. High technical capability is proven by this approach, and the project scope is kept realistic for a 6-day hackathon deadline.

