import argparse
import cv2
from face_detector import FaceDetector
from face_recognition import FaceRecognizer
from face_recognition import FaceClustering


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# The training module of the face recognition system. In summary, training comprises the following workflow:
#   1) Capturing new video frame.
#   2) Run face detection / tracking.
#   3) Extract face embedding and update face identification (mode "indent") or clustering (mode "cluster").
#   4) Fit face identification (mode "indent") or clustering (mode "cluster") and save trained models.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The training mode ("ident" to train face identification, "cluster" for face clustering)
    parser.add_argument('--mode', type=str, default="ident")
    # The video capture input. In case of "None" the default video capture (webcam) is used. Use a filename(s) to read
    # video data from image file (see VideoCapture documentation)
    #parser.add_argument('--video', type=str, default="/eso/dfs/Home/thko7498/FAU/CV Projekt/exercise/solution/training_data/Nancy_Sinatra/%04d.jpg")

    #parser.add_argument('--video', type=str,default="datasets/training_data/Nancy_Sinatra/Nancy_Sinatra_Video.mp4")
    #parser.add_argument('--label', type=str, default="Nancy_Sinatra")

    #parser.add_argument('--video', type=str, default="datasets/training_data/Alan_Ball/Alan_Ball_Video.mp4")
    #parser.add_argument('--label', type=str, default="Alan_Ball")

    #parser.add_argument('--video', type=str, default="datasets/training_data/Manuel_Pellegrini/Manuel_Pellegrini_Video.mp4")
    #parser.add_argument('--label', type=str, default="Manuel_Pellegrini")

    #parser.add_argument('--video', type=str,default="datasets/training_data/Marina_Silva/Marina_Silva_Video.mp4")
    #parser.add_argument('--label', type=str, default="Marina_Silva")

    #parser.add_argument('--video', type=str, default="datasets/training_data/Peter_Gilmour/Peter_Gilmour_Video.mp4")
    #parser.add_argument('--label', type=str, default="Peter_Gilmour")


    args = parser.parse_args()

    # additional to check accuracy by archit
    if args.video.split('/')[-2] == 'Al_Pacino' or args.video.split('/')[-2] == 'Maria_Shriver':
        expected_label = 'Unknown'
    else:
        expected_label = args.video.split('/')[-2]
    print("Expected Label: ", expected_label)

    # Setup OpenCV video capture.
    if args.video is None:
        camera = cv2.VideoCapture(-1)
        wait_for_frame = 1
    else:
        camera = cv2.VideoCapture(args.video)
        wait_for_frame = 100
    camera.set(3, 640)
    camera.set(4, 480)

    # Image display
    cv2.namedWindow("Camera")
    cv2.moveWindow("Camera", 0, 0)

    # Prepare face detection, identification, and clustering.
    detector = FaceDetector()
    recognizer = FaceRecognizer(expected_label=expected_label)
    clustering = FaceClustering()
    #recognizer = None
    #clustering = None

    # The video capturing loop.
    state = ""
    num_samples = 0
    while True:

        char = cv2.waitKey(wait_for_frame) & 0xFF
        if char == 27:
            # Stop capturing using ESC.
            break

        # Capture new video frame.
        _, frame = camera.read()
        if frame is None:
            print("End of stream")
            break
        # Resize the frame.
        height, width, channels = frame.shape
        if width < 640:
            s = 640.0 / width
            frame = cv2.resize(frame, (int(s*width), int(s*height)))
        # Flip frame if it is live video.
        if args.video is None:
            frame = cv2.flip(frame, 1)

        #just for testing
        if num_samples == 13:
            print('Here')
        # Track (or initially detect if required) a face in the current frame.
        face = detector.track_face(frame,num_samples)

        if face is not None:
            # We detected a face in the current frame.
            num_samples = num_samples + 1

            if args.mode == "ident":
                # Update face identification.
                recognizer.update(face["aligned"], args.label)
                state = "{} ({} samples)".format(args.label, num_samples)
            if args.mode == "cluster":
                # Update face clustering.
                clustering.update(face["aligned"])
                state = "{} samples".format(num_samples)

                # Display annotations for face tracking and training.
            face_rect = face["rect"]
            cv2.rectangle(frame,
                          (face_rect[0], face_rect[1]),
                          (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1), (0, 255, 0), 2)
            cv2.putText(frame,
                        state,
                        (face_rect[0] + face_rect[2] + 10, face_rect[1] + face_rect[3] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Camera", frame)

    # Save trained models for face identification and clustering.
    if args.mode == "ident":
        print("Save trained face recognition model")
        recognizer.save()

    if args.mode == "cluster":
        clustering.fit()
        clustering.save()
        print("Save trained face clustering")