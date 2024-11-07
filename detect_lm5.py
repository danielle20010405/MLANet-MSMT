import os
from mtcnn import MTCNN
import cv2

def detect_faces_and_save_keypoints(image_folder, output_folder):
    # Create an MTCNN face detector
    detector = MTCNN()

    # Get a list of all .jpg files in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    for image_file in image_files:
        # Load the image
        image = cv2.imread(os.path.join(image_folder, image_file))

        # Detect faces in the image
        detected_faces = detector.detect_faces(image)
        print('Detected Faces: ', detected_faces)

        # If a face is detected
        if detected_faces:
            # Get the keypoints for the first face detected
            keypoints = detected_faces[0]['keypoints']

            # Prepare the output string
            landmarks = ''
            for keypoint in keypoints.values():
                landmarks += str(keypoint[0]) + ' ' + str(keypoint[1]) + '\n'

            # Prepare the output file name
            output_file = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.txt')

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Write the keypoints to the output file
            with open(output_file, 'w') as f:
                f.write(landmarks)

# Use the function
detect_faces_and_save_keypoints('./datasets/img_align_celeba_valid', './datasets/img_align_celeba_valid/detections')