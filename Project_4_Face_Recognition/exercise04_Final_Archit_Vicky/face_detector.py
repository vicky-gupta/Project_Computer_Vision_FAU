import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=15, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None
        self.templater = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

    # ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.offset = tm_window_size

        #additional to check things
        self.matched = 0
        self.not_matched = 0

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image, frame_number=-1):
        # 1. Detect face with detect_face() MTCNN if havent already and store reference and template
        # 2. match with respect to template and offset
        # 2.1 if threshold with offset is low then do detect_face again and update the reference

        # 1 detect face when no reference
        #print('track_face '+str(frame_number)+'\n')
        if not self.reference:
            self.reference = self.detect_face(image)
            if self.reference is None:
                return None
            print('Changing template')
            self.templater = self.crop_face(self.reference['image'], self.reference['rect'])
            print('Matched ' + str(self.matched) + ' and not matched ' + str(self.not_matched) + '\n')
            return self.reference
        else:
            # match with template (i.e previous one) and offset
            #Hint: Use matchTemplate with TM_CCOEFF_NORMED similarity and minMaxLoc from OpenCV
            # image[self.reference['rect'] = original image ka face ka rect ka coordinates
            # image[self.reference['aligned'] = original image ka detected face only which we will use as template
            # offset image = new image uska original image ka rect +- offset


            top_offset = max(self.reference['rect'][1] - self.offset, 0)
            left_offset = max(self.reference['rect'][0] - self.offset , 0)
            bottom_offset = min(self.reference['rect'][1] + self.reference['rect'][3] - 1 + self.offset, image.shape[0] - 1)
            right_offset = min(self.reference['rect'][0] + self.reference['rect'][2] - 1 + self.offset, image.shape[1] - 1)

            offsetted_image = image[top_offset:bottom_offset, left_offset:right_offset, :]
            #offsetted_image = cv2.resize(offsetted_image, dsize=(self.aligned_image_size, self.aligned_image_size)) # is it needed ?

            # reference https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
            res = cv2.matchTemplate(offsetted_image, self.templater, cv2.TM_CCOEFF_NORMED)

            w, h, _ = image.shape[::-1]

            #cv2.imwrite('res.png', image)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            print('max_val: '+str(max_val))#+'\n for frame '+str(frame_number)+'\n')

            if max_val < self.tm_threshold:
                # less than threshold so do detect face again
                self.not_matched += 1
                self.reference = self.detect_face(image)
                if self.reference is None:
                    return None
                print('Changing template')
                self.templater = self.crop_face(self.reference['image'], self.reference['rect'])
                print('Matched ' + str(self.matched) + ' and not matched ' + str(self.not_matched) + '\n')
                return self.reference
            else:
                # doubt
                self.matched += 1
                top_left = (left_offset,top_offset) + max_loc

                face_rect = [top_left[0], top_left[1], self.reference['rect'][2], self.reference['rect'][3]]

                aligned = self.align_face(image, face_rect)
                self.reference = {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}
                #self.template = self.crop_face(self.reference['image'], self.reference['rect'])
                print('Matched ' + str(self.matched) + ' and not matched ' + str(self.not_matched) + '\n')
                return self.reference

        return None

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

