# ComfyUI Teskor's Utils

Utility nodes for **ComfyUI** focused on **video workflows**, **OpenPose stability**, and **chunk-based video generation**.

Main use cases:

- WanVideo
- AnimateDiff
- ControlNet OpenPose
- Long videos
- Chunked generation

Goal:

Improve **stability** and **visual consistency** in generated videos.


---

# Main Nodes

## OpenPose Smoother

### Purpose

Stabilizes OpenPose pose data across frames.

Raw OpenPose detection is noisy and causes:

- Pose jitter
- Hand flicker
- Missing joints
- Detection noise
- Random extra people

### What It Does

- Smooths keypoints over time
- Fills small gaps
- Filters unstable detections
- Optional extra-person removal
- Outputs smoothed POSEDATA

### Typical Pipeline

Load Video  
→ OpenPose Detection  
→ OpenPose Smoother  
→ ControlNet OpenPose  
→ Video Generation


---

## Color Match Sequential Bias

### Purpose

Removes **color drift between video chunks**.

Designed for chunk-based generation:

- WanVideo
- AnimateDiff
- Long videos

### Problem

Chunked generation causes small differences in:

- Brightness
- Color balance
- Contrast

This makes chunk boundaries visible.

### What It Does

For each chunk:

1. Measures previous chunk colors
2. Measures current chunk colors
3. Calculates difference
4. Applies correction

Result:

- Consistent colors
- Invisible chunk borders
- Continuous video

### Important Setting

chunk_size must match generation chunk size.

Example:

If chunks are 81 frames:

chunk_size = 81


### Typical Pipeline

WanVideo Animate Embeds  
→ Combine Frames  
→ Color Match Sequential Bias  
→ Save Video


---

# Other Nodes

Additional utilities:

- Batch video loading
- Pose data save/load
- Preview image without metadata
- Preview video without metadata
- Batch file renaming


---

# Recommended For

- WanVideo workflows
- AnimateDiff workflows
- OpenPose animation
- ControlNet OpenPose
- Long video generation
- Chunked workflows
