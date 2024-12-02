# import numpy as np
# import cv2 as cv
# from pathlib import Path

# def get_image():
#     Class = '1'
#     Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True)
#     cap=cv.VideoCapture(0)
#     if not cap.isOpened():
#         print("Unable to open cam")
#         exit()
#         i=0
#         while True:

#             ret, frame = cap.read()

#             if not ret:
#                 print("Unale to receive frame (stream end?)")
#                 break

#             i = i+1
#             if i % 5 == 0:
#                 cv.imwrite('DATASET/'+Class+'/'+str(i)+'.png',frame)

#                 cv.imshow('frame', frame)
#                 if cv.waitKey(1) == ord('q') or i>500:
#                     break
#         cap.release()
#         cv.destroyAllWindows()
# if __name__=='main':
#     get_image()

import numpy as np
import cv2 as cv
from pathlib import Path
import os

def get_images(class_label, output_dir='DATASET', max_images=500, skip_frames=5):
    """
    Capture and save images for a specific class to build a dataset.

    Args:
        class_label (str): The label of the class (e.g., "1", "2").
        output_dir (str): Directory to save the dataset.
        max_images (int): Maximum number of images to capture.
        skip_frames (int): Number of frames to skip between captures.
    """
    # Create directory for the class
    class_path = os.path.join(output_dir, class_label)
    Path(class_path).mkdir(parents=True, exist_ok=True)

    # Open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print(f"Collecting images for class '{class_label}'. Press 'q' to quit.")
    print(f"Images will be saved to: {class_path}")

    i = 0  # Frame counter
    saved_images = 0  # Saved images counter

    while saved_images < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame (stream end?).")
            break

        i += 1
        if i % skip_frames == 0:
            # Save the frame
            img_path = os.path.join(class_path, f'{saved_images+1}.png')
            cv.imwrite(img_path, frame)
            saved_images += 1

            # Display feedback to the user
            cv.putText(frame, f"Saved: {saved_images}/{max_images}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Capturing Dataset', frame)

        # Exit on 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Exiting capture...")
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"Dataset collection complete. Total images saved: {saved_images}/{max_images}")


if __name__ == '__main__':
    # Prompt the user for the class label
    class_label = input("Enter the class label for the dataset (e.g., '1', '2', 'thumbs_up'): ")
    get_images(class_label)
