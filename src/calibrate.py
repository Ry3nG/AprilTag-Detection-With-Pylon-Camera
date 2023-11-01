import cv2
import numpy as np
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for 8x11 checkerboard (7x10 inner corners)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10, 0:7].T.reshape(-1,2)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Look for .bmp files
images = glob.glob('C:\\Users\\Gong Zerui\\Desktop\\AprilTag\\calibration\\*.bmp')

for fname in images:
    print(f"Processing {fname}...")  # Print the current file being processed
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners for 7x10 inner corners
    ret, corners = cv2.findChessboardCorners(gray, (10,7), None)
    
    # If found, add object points, image points and show image with corners
    if ret:
        print(f"Detected corners in {fname}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Failed to detect corners in {fname}")

cv2.destroyAllWindows()

print("Starting camera calibration...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera calibration completed!")

print("Camera Matrix (Intrinsic Parameters): \n", mtx)
print("Distortion Coefficients: \n", dist)
