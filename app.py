from scipy.spatial import distance as dist
import cv2
import numpy as np


def register_object(centroid):
    global objects, disappeared, next_object_id
    objects[next_object_id] = centroid
    disappeared[next_object_id] = 0
    next_object_id += 1

def deregister_object(object_id):
    global objects, disappeared
    del objects[object_id]
    del disappeared[object_id]

def detect_and_track(video_path, subtractor_type='MOG2'):
    global objects, disappeared, next_object_id

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Create background subtractor
    if subtractor_type == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    elif subtractor_type == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    else:
        raise ValueError("Invalid subtractor type. Choose 'MOG2' or 'KNN'.")

    # Object tracking parameters
    max_disappeared = 30  # Maximum number of frames an object can disappear before we remove it
    min_distance = 50     # Minimum distance between centroids to consider it a new object

    # Dictionary to store tracked objects
    objects = {}
    disappeared = {}
    next_object_id = 0

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create random colors for drawing tracks
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # Apply background subtraction
        fgMask = backSub.apply(frame)

        # Thresholding to binary image and remove shadows
        _, thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)

        # Noise removal using morphological operations
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the image
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store centroids of current objects
        centroids = []

        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                centroids.append((int(x + w/2), int(y + h/2)))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # If we have no objects, register all centroids as new objects
        if len(objects) == 0:
            for i in range(len(centroids)):
                register_object(centroids[i])
        else:
            # Try to match existing objects with new centroids
            object_ids = list(objects.keys())
            object_centroids = list(objects.values())

            # Ensure object_centroids and centroids are not empty and are 2D arrays
            if len(object_centroids) > 0 and len(centroids) > 0:
                object_centroids_array = np.array(object_centroids)
                centroids_array = np.array(centroids)
                
                # Ensure arrays are 2D
                if object_centroids_array.ndim == 1:
                    object_centroids_array = object_centroids_array.reshape(-1, 2)
                if centroids_array.ndim == 1:
                    centroids_array = centroids_array.reshape(-1, 2)

                # Compute distances between each pair of object centroids and new centroids
                D = dist.cdist(object_centroids_array, centroids_array)

                # Find the smallest value in each row and sort the row indexes based on their minimum values
                rows = D.min(axis=1).argsort()

                # Find the smallest value in each column and sort using previously computed row index
                cols = D.argmin(axis=1)[rows]

                # Keep track of which rows and columns we have examined
                used_rows = set()
                used_cols = set()

                # Loop over the combination of the (row, column) index tuples
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    # If the distance is greater than the maximum distance, don't associate the two
                    if D[row, col] > min_distance:
                        continue

                    object_id = object_ids[row]
                    objects[object_id] = centroids[col]
                    disappeared[object_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

                # Compute both the row and column index we have NOT yet examined
                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)

                # If the number of object centroids is equal or greater than the number of input centroids
                # we need to check and see if some of these objects have potentially disappeared
                if D.shape[0] >= D.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        disappeared[object_id] += 1

                        if disappeared[object_id] > max_disappeared:
                            deregister_object(object_id)
                else:
                    for col in unused_cols:
                        register_object(centroids[col])
            else:
                # If either object_centroids or centroids is empty, register all centroids as new objects
                for centroid in centroids:
                    register_object(centroid)

        # Draw object IDs on the frame
        for (object_id, centroid) in objects.items():
            text = f"{object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Draw the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
        img = cv2.add(frame, mask)

        # Resize fgMask for overlay
        overlay_size = (frame.shape[1] // 4, frame.shape[0] // 4)
        fgMask_small = cv2.resize(fgMask, overlay_size)

        # Convert fgMask to 3-channel image
        fgMask_color = cv2.cvtColor(fgMask_small, cv2.COLOR_GRAY2BGR)

        # Create overlay in top right corner
        img[10:10+overlay_size[1], -10-overlay_size[0]:-10] = fgMask_color

        out.write(img)

        # Display the resulting frame
        cv2.imshow('Frame with Object Tracking and Optical Flow', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Release the video capture object and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_path = 'video.mp4'
detect_and_track(video_path, subtractor_type='MOG2')