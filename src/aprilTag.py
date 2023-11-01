import cv2
import numpy as np
import pupil_apriltags as apriltag


object_points = np.float32([    [-0.5, -0.5, 0],
    [0.5, -0.5, 0],
    [0.5, 0.5, 0],
    [-0.5, 0.5, 0]
])


def draw_3d_axis(image, pose, intrinsic_matrix):
    # Define the axis points in 3D space (X, Y, Z)
    axis_points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    
    # Get the rotation vector and ensure it's reshaped
    rotation_vec = pose[0].reshape(3, 1)

    print("Intrinsic matrix shape:", intrinsic_matrix.shape)
    print("Intrinsic matrix content:", intrinsic_matrix)

    # Project the 3D axis points back to 2D
    rotation_vec = rotation_vec.astype(np.float32)
    translation_vec = pose[1].astype(np.float32)
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    distortion_coeffs = np.zeros((4,1), dtype=np.float32)

    img_points, _ = cv2.projectPoints(axis_points, rotation_vec, translation_vec, intrinsic_matrix, distortion_coeffs)

    # Draw the 3D axis on the image
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Colors for X, Y, Z axes
    for i, c in enumerate(color):
        start_point = tuple(img_points[3].ravel().astype(int))
        end_point = tuple(img_points[i].ravel().astype(int))
        cv2.arrowedLine(image, start_point, end_point, c, 2)



def detect_apriltag(image_path,intrinsic_matrix):
    # Read the image in color mode
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        print(f"Failed to read image from {image_path}")
        return

    # Convert image to grayscale for detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the AprilTag detector
    detector = apriltag.Detector(families='tag36h11')

    # Detect AprilTag in the image
    result = detector.detect(gray_image)

    print(f"Number of detections: {len(result)}")

    if not result:
        print("No AprilTags detected!")
        return

    # Overlay detections on the image
    for detection in result:
        print(f"Detected tag {detection.tag_family} id {detection.tag_id} with hamming {detection.hamming}")

        # Draw bounding rectangle and tag ID on the image
        rect_points = detection.corners.astype(int)
        cv2.polylines(image, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2)
        centroid = rect_points.mean(axis=0).astype(int)
        cv2.putText(image, str(detection.tag_id), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Compute the pose of the tag
        _, rotation_vec, translation_vec = cv2.solvePnP(object_points, detection.corners.astype(np.float32), intrinsic_matrix, None)

        print("Rotation vector:", rotation_vec)
        print("Translation vector:", translation_vec)

        draw_3d_axis(image, (rotation_vec, translation_vec), intrinsic_matrix)

    # Display the image with detections
    cv2.imshow("AprilTag Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "tag36_11_00000.png"
    # You need to replace these with your actual camera parameters
    intrinsic_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    detect_apriltag(image_path, intrinsic_matrix)