# Fall Detection Using Machine Learning and Pose Estimation
 
## Overview
 
This project is a real-time fall detection system that uses MediaPipe pose estimation and a PyTorch neural network to identify when a person has fallen. When a fall is detected, the system can automatically send an email notification to a caregiver or emergency contact.
 
The project was originally developed as a STEM Fair project, where it won:
 
- Best of Fair (Overall First Place)
- 1st Place in my age category
- Computer Science Award
- Qualification to compete at the 2026 Canada-Wide Science Fair
 
Rather than analyzing every pixel in a video frame, the system uses human pose landmarks extracted by MediaPipe. This significantly reduces the amount of data being processed while still providing enough information for accurate fall detection.
 

### Example screenshot of program running live
<img width="408" height="538" alt="Screenshot 2026-04-07 180946" src="https://github.com/user-attachments/assets/cef00c0c-7212-4ce6-bde4-de3455845938" />

--- 
## How It Works
 
1. A webcam video stream is captured using OpenCV.
2. MediaPipe detects human body landmarks.
3. Custom features are extracted, including:
   - Normalized body landmark coordinates
   - Body aspect ratio
   - Torso angle relative to the ground
4. Features collected over multiple frames are passed to a trained neural network.
5. If the model repeatedly predicts a fall, an email alert is sent.
 
---
 
## Repository Structure
 
### Main Application
 
#### `Fall_detector.py`
The real-time fall detection program. It uses a webcam feed, MediaPipe pose estimation, and the trained neural network to detect falls as they occur.
 
#### `common_functions.py`
Contains the helper functions used throughout the project, including:
- Pose landmark processing
- Feature extraction
- Landmark normalization
- Torso angle and body aspect ratio calculations
- Email notification functionality
 
#### `fall_model.pth`
Stores the trained neural network parameters used by the live fall detection system.
 
---
 
### Dataset Creation and Model Development
 
#### `Data_scraper.py`
Processes video files and converts them into machine learning training data by:
- Detecting pose landmarks
- Extracting custom features
- Labeling fall and non-fall videos
- Building and saving the dataset
 
#### `Fall_Detection_Models.ipynb`
Contains the model development process used to create the final detector. Multiple neural network architectures, optimizers, loss functions, hidden layer configurations, and hyperparameters were tested and compared before selecting the final model.
 
#### `fall_dataset.npz`
The generated training dataset used during model development.
 
---
 
### Example Data
 
#### `video_folder/`
Contains example videos used during development and testing.
 
The complete dataset is not included in this repository due to file size limitations, but sample videos are provided to demonstrate the expected input format.
Note that the video should have at least 3 seconds of footage where a person is fully in frame to make sure the data scraper program can collect enough data for a model input.
 
---
 
## Installation
 
Clone the repository:
 
```bash
git clone https://github.com/Vincent-Mannarino/Fall-Detection.git
cd Fall-Detection
```
 
Install the required dependencies:
 
```bash
pip install -r requirements.txt
```
 
---
 
## Running the Project
 
Before running the detector, configure the email notification settings.
 
Open:
 
```python
common_functions.py
```
 
Locate the `send_fall_email()` function at the bottom of the file and enter:
 
```python
sender_email = "your_email"
receiver_email = "recipient_email"
app_password = "gmail_app_password"
```
 
After configuring the email settings, run:
 
```bash
python Fall_detector.py
```
 
The webcam feed will open and the program will begin monitoring for falls in real time.
 
Press **Q** to close the application.
 
---
 
## Machine Learning Approach
 
Instead of training directly on image pixels, the model was trained using features derived from MediaPipe pose landmarks.
 
Features used include:
 
- Normalized body landmark coordinates centered around the hips
- Body aspect ratio
- Torso angle relative to the ground
- Motion information collected across multiple consecutive frames
 
Several machine learning models were tested using combinations of:
 
- Linear and non-linear neural networks
- One and two hidden layer architectures
- 16 and 32 hidden-unit configurations
- SGD and Adam optimizers
- Binary Cross Entropy and Mean Absolute Error loss functions
 
The best-performing model was selected and deployed into the live detection system.
 
### Final Model Performance
 
- **95.24% testing accuracy**
- Real-time CPU operation
- Email notification support
- Low computational requirements due to pose-landmark based feature extraction
 
---
 
## Project Motivation
 
Many existing fall-alert systems rely on wearable devices that must be worn continuously and manually activated after a fall. These systems may fail if the device is not being worn or if the user is unable to activate it after becoming injured.
 
This project explores whether computer vision and machine learning can provide an automated alternative by detecting falls directly from a camera feed without requiring wearable sensors.
 
The goal was to create a practical, low-cost system capable of operating in real time on everyday hardware.
 
---
 
## Future Improvements
 
Potential future improvements include:
 
- Expanding the dataset with more diverse fall and non-fall scenarios
- Reducing overfitting through additional training data
- Mobile or cloud-based emergency notifications
- Additional motion-based features for improved accuracy
- Voice-based emergency detection as a secondary trigger
- Testing advanced sequence models such as LSTMs or Transformers
 
---
 
## STEM Fair Research Project
 
This project was developed as an independent STEM Fair research project investigating whether pose-estimation-based machine learning models could accurately detect falls while remaining computationally efficient.
 
The project successfully demonstrated that a machine learning model trained on pose-derived features such as body landmark positions, torso orientation, and body aspect ratio can accurately classify falls while operating in real time using only a CPU.
 
---
 
## Technologies Used
 
- Python
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
 
---
 
## Acknowledgements
 
This project was created as part of an independent STEM Fair research project focused on applying machine learning and computer vision to a real-world safety problem. It combines pose estimation, feature engineering, and neural network classification to create an efficient fall detection system capable of operating in real time.
