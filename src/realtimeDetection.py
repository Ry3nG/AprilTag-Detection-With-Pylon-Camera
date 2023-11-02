import cv2
import numpy as np
import pupil_apriltags as apriltag
from pypylon import pylon
import logging
import time

# Constants & Configurations
INTRINSIC_MATRIX = np.array(
    [
        [1.90148512e03, 0.00000000e00, 9.76377693e02],
        [0.00000000e00, 1.91018989e03, 5.51258034e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
DISTORTION_COEFFICIENTS = np.array(
    [0.04571354, -0.07891728, -0.0134629, 0.00627635, -0.47984874]
)

TAG_FAMILY = "tag36h11"
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Colors for X, Y, Z axes
OBJECT_POINTS = np.float32(
    [
        [-0.06, -0.06, 0],  # -6 cm to 6 cm (12 cm total side length)
        [0.06, -0.06, 0],
        [0.06, 0.06, 0],
        [-0.06, 0.06, 0],
    ]
)


def draw_3d_axis(image, pose, intrinsic_matrix):
    # Define the axis points in 3D space (X, Y, Z)
    axis_points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(
        -1, 3
    )

    # Get the rotation vector and ensure it's reshaped
    rotation_vec = pose[0].reshape(3, 1)

    # Project the 3D axis points back to 2D
    axis_points *= 0.1  # Scale axis length to 10 cm
    rotation_vec = rotation_vec.astype(np.float32)
    translation_vec = pose[1].astype(np.float32)
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    img_points, _ = cv2.projectPoints(
        axis_points,
        rotation_vec,
        translation_vec,
        intrinsic_matrix,
        DISTORTION_COEFFICIENTS,
    )

    for i, c in enumerate(AXIS_COLORS):
        start_point = tuple(img_points[3].ravel().astype(int))
        end_point = tuple(img_points[i].ravel().astype(int))
        cv2.arrowedLine(image, start_point, end_point, c, 2)


def detect_apriltag_in_frame(frame, intrinsic_matrix, detector):
    if not hasattr(detect_apriltag_in_frame, "prev_time"):
        detect_apriltag_in_frame.prev_time = time.time()
    # Display frame rate
    current_time = time.time()
    delta_time = current_time - detect_apriltag_in_frame.prev_time
    fps = 1 / (delta_time + 1e-9)  # add a small value to prevent division by zero
    detect_apriltag_in_frame.prev_time = current_time
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    # Convert frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize the AprilTag detector
    # detector = apriltag.Detector(families='tag36h11')

    # Detect AprilTag in the frame
    result = detector.detect(gray_frame)

    # Overlay detections on the image
    for detection in result:
        print(
            f"Detected tag {detection.tag_family} id {detection.tag_id} with hamming {detection.hamming}"
        )

        # Draw bounding rectangle and tag ID on the image
        rect_points = detection.corners.astype(int)
        cv2.polylines(
            frame, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2
        )
        centroid = rect_points.mean(axis=0).astype(int)
        cv2.putText(
            frame,
            str(detection.tag_id),
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Compute the pose of the tag
        _, rotation_vec, translation_vec = cv2.solvePnP(
            OBJECT_POINTS,
            detection.corners.astype(np.float32),
            intrinsic_matrix,
            DISTORTION_COEFFICIENTS,
        )

        # Compute the Euler angles from the rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Calculate Euler angles from the rotation matrix
        sy = np.sqrt(rotation_mat[0, 0] * rotation_mat[0, 0] + rotation_mat[1, 0] * rotation_mat[1, 0])
        is_singular = sy < 1e-6

        if not is_singular:
            x_angle = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            y_angle = np.arctan2(-rotation_mat[2, 0], sy)
            z_angle = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            x_angle = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            y_angle = np.arctan2(-rotation_mat[2, 0], sy)   
            z_angle = 0

        # Assigning roll, pitch, and yaw based on the specific convention
        roll = x_angle
        pitch = y_angle
        yaw = z_angle


        # Display distance and angles on the stream
        cv2.putText(
            frame,
            f"Roll: {np.degrees(roll):.2f} deg",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Pitch: {np.degrees(pitch):.2f} deg",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Yaw: {np.degrees(yaw):.2f} deg",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        tag_width = np.linalg.norm(rect_points[0] - rect_points[1])
        tag_height = np.linalg.norm(rect_points[1] - rect_points[2])
        cv2.putText(
            frame,
            f"Tag W: {tag_width:.2f}px H: {tag_height:.2f}px",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        draw_3d_axis(frame, (rotation_vec, translation_vec), intrinsic_matrix)

    # Resize for display
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    return frame


class PylonCamera:
    def __init__(self):
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            logging.info("Pylon camera initialized")
        except Exception as e:
            raise RuntimeError("Cannot connect to Pylon Camera: " + str(e))

    def start_stream(self):
        # Initialize the AprilTag detector once
        detector = apriltag.Detector(families="tag36h11")

        # Grabbing continuously with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()

        # Converting to OpenCV BGR format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Call the AprilTag detection function with the initialized detector
                detect_apriltag_in_frame(
                    img, INTRINSIC_MATRIX, detector
                )  # Modify your function to accept and use the passed detector

                # Display the result using OpenCV
                cv2.imshow("AprilTag Detection Stream", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit the loop if 'q' is pressed
                    break

            grabResult.Release()

    cv2.destroyAllWindows()

    def release(self):
        logging.info("Pylon released")
        self.camera.Close()


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    camera = PylonCamera()
    try:
        camera.start_stream()
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        camera.release()
