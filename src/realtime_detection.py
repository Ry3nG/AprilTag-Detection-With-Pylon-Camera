import logging
import time

import cv2
import numpy as np
import pupil_apriltags as apriltag
from pypylon import pylon

SCALE_FACTOR = 0.5

# Constants & Configurations
INTRINSIC_MATRIX = np.array(
    [
        [1.90148512e03, 0.00000000e00, 9.76377693e02],
        [0.00000000e00, 1.91018989e03, 5.51258034e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

INTRINSIC_MATRIX = INTRINSIC_MATRIX * SCALE_FACTOR

DISTORTION_COEFFICIENTS = np.array(
    [0.04571354, -0.07891728, -0.0134629, 0.00627635, -0.47984874]
)

TAG_FAMILY = "tag36h11"
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Colors for X, Y, Z axes
TAG_SIZE = 0.16  # Assuming the tag size is also 12 cm, replace with actual tag size if different

OBJECT_POINTS = np.array([
    [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],  # Bottom-left corner
    [TAG_SIZE / 2, -TAG_SIZE / 2, 0],   # Bottom-right corner
    [TAG_SIZE / 2, TAG_SIZE / 2, 0],    # Top-right corner
    [-TAG_SIZE / 2, TAG_SIZE / 2, 0]    # Top-left corner
], dtype=np.float32)

# Define the cube's corner points in 3D (assuming the tag is at the origin and the cube is 12cm each side)


CUBE_SIZE = TAG_SIZE
cube_points = np.float32(
    [
        [0, 0, 0],
        [CUBE_SIZE, 0, 0],
        [CUBE_SIZE, CUBE_SIZE, 0],
        [0, CUBE_SIZE, 0],
        [0, 0, -CUBE_SIZE],
        [CUBE_SIZE, 0, -CUBE_SIZE],
        [CUBE_SIZE, CUBE_SIZE, -CUBE_SIZE],
        [0, CUBE_SIZE, -CUBE_SIZE],
    ]
) - np.array(
    [TAG_SIZE / 2, TAG_SIZE / 2, -CUBE_SIZE]
)  # Offset to make the cube centered on the tag


def draw_cube(image, corners, imgpts):
    imgpts = imgpts.reshape(-1, 2)  # Reshape to ensure it is Nx2
    imgpts = imgpts.astype(int)

    # Define new colors in RGB format
    ground_floor_color = (0, 128, 255)
    pillars_color = (255, 0, 128)
    top_layer_color = (128, 0, 255)

    # Draw ground floor in purple
    for i in range(4):
        cv2.line(
            image, tuple(imgpts[i]), tuple(imgpts[(i + 1) % 4]), ground_floor_color, 3
        )

    # Draw pillars in orange color
    for i in range(4):
        cv2.line(image, tuple(imgpts[i]), tuple(imgpts[i + 4]), pillars_color, 3)

    # Draw top layer in yellow color
    cv2.drawContours(image, [imgpts[4:8]], -1, top_layer_color, 3)

    return image


def draw_3d_axis(image, pose, intrinsic_matrix):
    # Define the axis points in 3D space (X, Y, Z)
    axis_points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(
        -1, 3
    )

    # Get the rotation vector and ensure it's reshaped
    rotation_vec = pose[0].reshape(3, 1)

    # Project the 3D axis points back to 2D
    axis_points *= 0.05  # Scale axis length to 5 cm
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

def draw_plane(image, origin, pose, intrinsic_matrix, plane_size):
    # Define the four corners of the plane in the tag's coordinate system
    plane_points = np.array([
        [0, 0, 0],
        [plane_size[0], 0, 0],
        [plane_size[0], plane_size[1], 0],
        [0, plane_size[1], 0]
    ], dtype=np.float32)-origin
    
    # Project the plane corners to the image
    image_points, _ = cv2.projectPoints(
        plane_points,
        pose[0],  # rotation vector
        pose[1],  # translation vector
        intrinsic_matrix,
        DISTORTION_COEFFICIENTS
    )
    
    image_points = image_points.reshape(-1, 2)  # Reshape to ensure it is Nx2
    image_points = image_points.astype(int)  # Convert to integer for drawing functions
    
    # Draw the plane on the image using a filled convex polygon
    cv2.fillConvexPoly(image, np.array(image_points), (100, 200, 100), lineType=cv2.LINE_AA)
    
    return image


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

    # Detect AprilTag in the frame
    result = detector.detect(gray_frame)

    # Overlay detections on the image
    for detection in result:
        logging.info(
            f"Detected tag {detection.tag_family} id {detection.tag_id} with hamming {detection.hamming}"
        )

        # Draw bounding rectangle and tag ID on the image
        rect_points = detection.corners.astype(int)
        centroid = rect_points.mean(axis=0).astype(int)
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
        if _:
            draw_3d_axis(frame, (rotation_vec, translation_vec), intrinsic_matrix)
            # Inside your loop after pose estimation
            imgpts, _ = cv2.projectPoints(
                cube_points,
                rotation_vec,
                translation_vec,
                intrinsic_matrix,
                DISTORTION_COEFFICIENTS,
            )
            imgpts = imgpts.reshape(-1, 2)  # Reshape to ensure it is Nx2
            assert imgpts.shape == (8, 2), "imgpts should be a Nx2 array"
            frame = draw_cube(frame, rect_points, imgpts)

            # Calculate the distance to the tag
            distance = np.linalg.norm(translation_vec)
            cv2.putText(
                frame,
                f"Distance: {distance:.2f} m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            # Inside your loop after pose estimation
            plane_size = (0.19, 0.25) # size of an iPad
            frame = draw_plane(frame, origin=(OBJECT_POINTS[0]), pose=(rotation_vec, translation_vec), intrinsic_matrix=INTRINSIC_MATRIX, plane_size=plane_size)


    # Resize for display
    frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
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
        detector = apriltag.Detector(families=TAG_FAMILY)

        # Grabbing continuously with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()

        # Converting to OpenCV BGR format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )

            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                img = image.GetArray()

                # reduce the size of the image
                img = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

                # Call the AprilTag detection function with the initialized detector
                detect_apriltag_in_frame(
                    img, INTRINSIC_MATRIX, detector
                )  # Modify your function to accept and use the passed detector

                # Display the result using OpenCV
                cv2.imshow("AprilTag Detection Stream", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit the loop if 'q' is pressed
                    break

            grab_result.Release()

    cv2.destroyAllWindows()

    def release(self):
        logging.info("Pylon released")
        self.camera.Close()


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )

    camera = PylonCamera()
    try:
        camera.start_stream()
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        camera.release()
