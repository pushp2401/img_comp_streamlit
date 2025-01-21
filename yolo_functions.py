import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
import os
import numpy as np
import hashlib
import random
import matplotlib.pyplot as plt
import cv2


from ultralytics import YOLO 

# 1. Load a YOLOv8 segmentation model (pre-trained weights)
model = YOLO("best.pt") 

def get_label_color_id(label_id):
    """
    Generate a consistent BGR color for a numeric label_id by hashing the ID.
    This ensures that each numeric ID always maps to the same color.
    """
    label_str = str(int(label_id))
    # Use the MD5 hash of the label string as a seed
    seed_value = int(hashlib.md5(label_str.encode('utf-8')).hexdigest(), 16)
    random.seed(seed_value)
    # Return color in BGR format
    return (
        random.randint(50, 255),  # B
        random.randint(50, 255),  # G
        random.randint(50, 255)   # R
    )

def segment_large_image_with_tiles(
    model,
    large_image_path,
    tile_size=1080,
    overlap=60,  # Overlap in pixels
    alpha=0.4,
    display=True
):
    """
    1. Reads a large image from `large_image_path`.
    2. Tiles it into sub-images of size `tile_size` x `tile_size`,
       stepping by (tile_size - overlap) to have overlap regions.
    3. Runs `model.predict()` on each tile and accumulates all polygons (in global coords).
    4. For each class, merges overlapping polygons by:
       - filling them on a single-channel mask
       - finding final contours of the connected regions
    5. Draws merged polygons onto an overlay and alpha-blends with the original image.
    6. Returns the final annotated image (in RGB) and a dictionary of merged contours.
    """

    # Read the large image
    image_bgr = cv2.imread(large_image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {large_image_path}")

    # Convert to RGB (for plotting consistency)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = image_rgb.shape

    # Dictionary to store raw polygon coords for each class
    # (before merging)
    class_mask_dict = {}

    # Step size with overlap
    step = tile_size - overlap if overlap < tile_size else tile_size

    # ------------------------
    # 1) Perform Tiled Inference
    # ------------------------
    for top in range(0, H, step):
        for left in range(0, W, step):
            bottom = min(top + tile_size, H)
            right = min(left + tile_size, W)

            tile_rgb = image_rgb[top:bottom, left:right]

            # Run YOLOv8 model prediction
            results = model.predict(tile_rgb)
            if len(results) == 0:
                continue

            # Typically, results[0] holds the main predictions
            pred = results[0]

            # Check if we have valid masks
            if (pred.masks is None) or (pred.masks.xy is None):
                continue

            tile_masks_xy = pred.masks.xy  # list of polygon coords
            tile_labels = pred.boxes.cls   # list of class IDs

            # Convert to numpy int if needed
            if hasattr(tile_labels, 'cpu'):
                tile_labels = tile_labels.cpu().numpy()
            tile_labels = tile_labels.astype(int).tolist()

            # Accumulate polygon coords in global space
            for label_id, polygon in zip(tile_labels, tile_masks_xy):
                # Convert polygon float coords to int points in shape (N,1,2)
                polygon_pts = polygon.reshape((-1, 1, 2)).astype(np.int32)

                # Offset the polygon to the large image coords
                polygon_pts[:, 0, 0] += left  # x-offset
                polygon_pts[:, 0, 1] += top   # y-offset

                if label_id not in class_mask_dict:
                    class_mask_dict[label_id] = []
                class_mask_dict[label_id].append(polygon_pts)

    # -----------------------------------------
    # 2) Merge Overlapping Polygons For Each Class
    #    by rasterizing them in a mask and then
    #    finding final contours
    # -----------------------------------------
    merged_class_mask_dict = {}
    for label_id, polygons_cv in class_mask_dict.items():
        # Create a blank mask (single channel) for the entire image
        mask = np.zeros((H, W), dtype=np.uint8)

        # Fill all polygons for this label on the mask
        for pts in polygons_cv:
            cv2.fillPoly(mask, [pts], 255)

        # Now findContours to get merged regions
        # Use RETR_EXTERNAL so we just get outer boundaries of each connected region
        contours, _ = cv2.findContours(
            mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        # Store final merged contours
        merged_class_mask_dict[label_id] = contours

    # -----------------------
    # 3) Draw Merged Polygons
    # -----------------------
    overlay = image_rgb.copy()
    for label_id, contours in merged_class_mask_dict.items():
        color_bgr = get_label_color_id(label_id)
        for cnt in contours:
            # Fill each contour on the overlay
            cv2.fillPoly(overlay, [cnt], color_bgr)

    # 4) Alpha blend
    output = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    # 5) Optional Display
    if display:
        plt.figure(figsize=(12, 12))
        plt.imshow(output)
        plt.axis('off')
        plt.title("Segmentation on Large Image (Overlapped Tiles + Merged Polygons)")
        plt.show()

    return output, merged_class_mask_dict

def usable_data(img_results, image_1):
    """
    Extract bounding boxes, centers, and polygon areas from the segmentation
    results for a single image. Returns a dictionary keyed by label,
    with each value a list of object data: { 'bbox', 'center', 'area' }.
    """
    width, height = image_1.width, image_1.height
    image_data = {}
    for key in img_results.keys():
        image_data[key] = []
        for polygon in img_results[key]:
            polygon = np.array(polygon)

            # Handle varying polygon shapes
            # If shape is (N, 1, 2) e.g. from cv2 findContours
            if polygon.ndim == 3 and polygon.shape[1] == 1 and polygon.shape[2] == 2:
                polygon = polygon.reshape(-1, 2)
            elif polygon.ndim == 2 and polygon.shape[1] == 1:
                polygon = np.squeeze(polygon, axis=1)

            # Now we expect polygon to be (N, 2):
            xs = polygon[:, 0]
            ys = polygon[:, 1]

            # Bounding box
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            bbox = (xmin, ymin, xmax, ymax)

            # Center
            centerX = (xmin + xmax) / 2.0
            centerY = (ymin + ymax) / 2.0
            x = width/2
            y = height/2
            # Direction
            dx = x - centerX
            dy = centerY - y  # Invert y-axis for proper orientation
            if dx > 0 and dy > 0:
                direction = "NE"
            elif dx > 0 and dy < 0:
                direction = "SE"
            elif dx < 0 and dy > 0:
                direction = "NW"
            elif dx < 0 and dy < 0:
                direction = "SW"
            elif dx == 0 and dy > 0:
                direction = "N"
            elif dx == 0 and dy < 0:
                direction = "S"
            elif dy == 0 and dx > 0:
                direction = "E"
            elif dy == 0 and dx < 0:
                direction = "W"
            else:
                direction = "Center"


            # Polygon area (Shoelace formula)
            # area = 0.5 * | x0*y1 + x1*y2 + ... + x_{n-1}*y0 - (y0*x1 + y1*x2 + ... + y_{n-1}*x0 ) |
            area = 0.5 * np.abs(
                np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))
            )

            image_data[key].append({
                'bbox': bbox,
                'center': (centerX, centerY),
                'area': area,
                "direction": direction
            })
    return image_data

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_differences_on_image1(
    image1_path,
    mask_dict1,  # e.g., label_name -> list of contours for image1
    image2_path,
    mask_dict2,  # e.g., label_name -> list of contours for image2
    display=True
):
    """
    Compare two images (and their object masks). Plot all differences on Image 1 only:
      - Red: Objects that are missing on Image 1 (present in Image 2 but not Image 1).
      - Green: Objects that are missing on Image 2 (present in Image 1 but not Image 2).

    :param image1_path: Path to the first image
    :param mask_dict1:  dict[label_name] = [contour1, contour2, ...] for the first image
    :param image2_path: Path to the second image
    :param mask_dict2:  dict[label_name] = [contour1, contour2, ...] for the second image
    :param display:     If True, shows the final overlay with matplotlib.
    :return: A tuple:
             - overlay1 (numpy array in RGB) with all differences highlighted
             - list_of_differences: Names of labels with differences
             - difference_masks: A dict with keys "missing_on_img1" and "missing_on_img2",
               where each key maps to a list of contours (original format) for the respective differences.
    """

    # Read both images
    img1_bgr = cv2.imread(image1_path)
    img2_bgr = cv2.imread(image2_path)
    if img1_bgr is None or img2_bgr is None:
        raise ValueError("Could not read one of the input images.")

    # Convert to RGB
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    # Check matching dimensions
    H1, W1, _ = img1_rgb.shape
    H2, W2, _ = img2_rgb.shape
    if (H1 != H2) or (W1 != W2):
        raise ValueError("Images must be the same size to compare masks reliably.")

    # Prepare an overlay on top of Image 1
    overlay1 = img1_rgb.copy()

    # Take the union of all labels in both dictionaries
    all_labels = set(mask_dict1.keys()).union(set(mask_dict2.keys()))

    # Colors:
    RED = (255, 0, 0)    # (R, G, B)
    GREEN = (0, 255, 0)  # (R, G, B)

    # Track differences
    list_of_differences = []
    difference_masks = {
        "missing_on_img1": {},  # dict[label_name] = list of contours
        "missing_on_img2": {},  # dict[label_name] = list of contours
    }

    for label_id in all_labels:
        # Create binary masks for this label in each image
        mask1 = np.zeros((H1, W1), dtype=np.uint8)
        mask2 = np.zeros((H1, W1), dtype=np.uint8)

        # Fill polygons for label_id in Image 1
        if label_id in mask_dict1:
            for cnt in mask_dict1[label_id]:
                cv2.fillPoly(mask1, [cnt], 255)

        # Fill polygons for label_id in Image 2
        if label_id in mask_dict2:
            for cnt in mask_dict2[label_id]:
                cv2.fillPoly(mask2, [cnt], 255)

        # Missing on Image 1 (present in Image 2 but not in Image 1)
        # => mask2 AND (NOT mask1)
        missing_on_img1 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))

        # Missing on Image 2 (present in Image 1 but not in Image 2)
        # => mask1 AND (NOT mask2)
        missing_on_img2 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))

        # Extract contours of differences
        contours_missing_on_img1, _ = cv2.findContours(
            missing_on_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_missing_on_img2, _ = cv2.findContours(
            missing_on_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Store contours in difference masks
        if contours_missing_on_img1:
            difference_masks["missing_on_img1"][label_id] = contours_missing_on_img1
        if contours_missing_on_img2:
            difference_masks["missing_on_img2"][label_id] = contours_missing_on_img2

        # If there are differences, track the label name
        if contours_missing_on_img1 or contours_missing_on_img2:
            list_of_differences.append(label_id)

        # Color them on the overlay of Image 1:
        for cnt in contours_missing_on_img1:
            cv2.drawContours(overlay1, [cnt], -1, RED, -1)  # highlight in red
        for cnt in contours_missing_on_img2:
            cv2.drawContours(overlay1, [cnt], -1, GREEN, -1)  # highlight in green

    # Display if required
    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(overlay1)
        plt.title("Differences on Image 1\n(Red: Missing on Image 1, Green: Missing on Image 2)")
        plt.axis("off")
        plt.show()

    return overlay1, list_of_differences, difference_masks


system_prompt = """You are given two construction blueprint images along with their segmentation data.

Do not present any numeric bounding box or area values in your final answer.
Instead, produce a concise, high-level descriptive summary of the differences, using relative location references or known blueprint areas (e.g., “balcony,” “bathroom,” “central hallway,” etc.).
Treat two objects as identical (and thus ignore them) if:
They have the same label/class, and
Their center coordinates are very close.
If possible, provide an OCR-based overview of changed text or lines in those areas. For example, mention if the balcony area contains new textual annotations or if certain labels have been removed/added.
Output the result in brief, correct Markdown summarizing only the differences between the images (e.g., newly added structures, missing items, changed labeling or text).
Remember: No numeric bounding box or area data should be included in the final response. Use location references (“in the top-right corner,” “in the balcony,” etc.) and class names to describe changes.
"""

system_prompt_2 = """You are analyzing two construction blueprint images (Image 1 and Image 2). Each image has a set of detected objects, including “areas” like Balconies, Rooms, Hallways, etc., and smaller objects like Doors, Walls, or Stairs.

Key Points:

An object is considered to belong to an area if the object's center lies within or very close to that area’s bounding box.
Two objects in different images are considered the same object if:
They share the same label/class, and
Their centers are very close in coordinates. In such a case, ignore them (do not list them) because they have not changed significantly.
Focus only on describing the differences between Image 1 and Image 2, such as:
New objects or areas that appear in Image 2 but not in Image 1 (and vice versa).
Changes in labeling or text (e.g., from an OCR perspective).
Changes in object location or area assignment.
Do NOT output numeric bounding boxes, polygon areas, or center coordinates in your final explanation. Instead, provide a relative or area-based description (e.g., “The door is now located in the balcony,” “There are two new doors in the living room,” “A new label is added near the main hallway,” etc.).
Produce a concise and correct Markdown summary that highlights only significant differences.

"""

system_prompt_3 = """You are analyzing two construction blueprint images (Image 1 and Image 2). For each image, you have:

A set of objects (walls, doors, stairs, etc.) along with information on their labels and centers.
A set of “areas” (e.g., “Balcony,” “Living Room,” “Hallway,” “Bathroom,” etc.) with bounding boxes to identify where each area is located.
Task Requirements:
Identify differences between Image 1 and Image 2:
Newly added objects in Image 2 that were not in Image 1.
Missing objects in Image 2 that were in Image 1.
Objects that have changed location or have changed labels.
Text or label changes, if available.
For missing or newly added objects, describe their location in terms of relative position or known areas (not raw coordinates):
For example, say “the missing doors were originally near the top-left corner, adjacent to the main hallway,” or “new walls have been added in the southeast corner, near the living room.”
Avoid including numeric bounding boxes, polygon areas, or centers in the final explanation.
If two objects (one in Image 1 and one in Image 2) have the same label and nearly identical centers, consider them the same object and do not report them as a difference.
Whenever possible, use known area labels to describe positions (e.g., “within the dining area,” “just north of the bathroom,” “adjacent to the balcony,” etc.).
Return a concise and correct Markdown summary with these differences, focusing on where changes occur.
"""

system_prompt_4 = """You are given two sets of data from two blueprint images (Image 1 and Image 2). Along with each image’s extracted objects, you have:
A set of objects (walls, doors, stairs, etc.) along with information on their labels and centers.
A set of “areas” (e.g., “Balcony,” “Living Room,” “Hallway,” “Bathroom,” etc.) with bounding boxes to identify where each area is located.

A “nearest reference area” for each object, including a small textual description of distance and direction (e.g., “Door #2 is near the Balcony to the east”).
Identifications of which objects match across the two images (same label and close centers).
Your Task
Ignore any objects that match between the two images (same label, nearly identical location).
Summarize the differences: newly added or missing objects, label changes, and any changes in object location.
Use the relative position data (distance/direction text) to describe where each new or missing object is/was in terms of known areas (e.g., “the missing wall in the northern side of the corridor,” “the new door near the balcony,” etc.).
Do not output raw numeric distances, bounding boxes, or polygon areas in your final summary. Instead, give a natural-language location description (e.g., “near the east side of the main hallway,” “slightly south of the balcony,” etc.).
Provide your answer in a concise Markdown format, focusing only on significant differences."""

user_prompt = f"""I have two construction blueprint images, Image 1 and Image 2, and here are their segmentation results (with bounding boxes, centers, and areas). Please compare them and provide a short Markdown summary of the differences, ignoring any objects that match in both images:

Image 1:
image: {image_1}

json
Copy
{image_1_data}
Image 2:
image: {image_2}
json
Copy
{image_2_data}

Please:

Compare the two images in terms of architectural/structural changes.
Ignore objects that appear in both images (same label & near-identical centers).
Refer to changes in relative location or in known blueprint areas (e.g. “balcony,” “living room,” “main hallway”), not numeric bounding boxes or polygon areas.
Include mentions of new text or lines if any appear based on an OCR-like analysis.
Output only the differences in a concise Markdown summary."""

user_prompt_2 = f"""I have two construction blueprint images, Image 1 and Image 2, and here are their segmentation results (with bounding boxes, centers, and areas). Please compare them and provide a short Markdown summary of the differences, ignoring any objects that match in both images:

    Image 1:
    image: {image_1}
    
    json
    Copy
    {image_1_data}
    Image 2:
    image: {image_2}
    json
    Copy
    {image_2_data}
    
    Please:
    
    Ignore objects that appear in both images with matching labels and nearly identical centers.
    Use the bounding boxes of recognized “areas” (like “Balcony,” “Living Room,” “Bathroom,” etc.) to determine which area new or changed objects belong to. For instance, if a door’s center is inside or very close to the balcony’s bounding box, treat that door as being “in the balcony.”
    Do not display any raw bounding box coordinates, center points, or numeric area values in your final response.
    Summarize only the differences (e.g., newly added objects, missing objects, changed textual labels) in a brief Markdown format.
    Mention if there are text/label changes (e.g., from an OCR perspective) in any particular area or region"""
    
user_prompt_3 = f"""I have two construction blueprint images, Image 1 and Image 2, and here are their segmentation results (with bounding boxes, centers, and areas). Please compare them and provide a short Markdown summary of the differences, ignoring any objects that match in both images:
    
    Image 1:
    image: {image_1}
    
    json
    Copy
    {image_1_data}
    Image 2:
    image: {image_2}
    json
    Copy
    {image_2_data}
    
    Please:
    Compare the two images only in terms of differences—ignore any objects that match (same label and near-identical center).
    For objects missing in Image 2 (but present in Image 1), or newly added in Image 2, indicate their relative position using known areas or approximate directions. For instance, mention if the missing doors were “towards the north side, near the elevator,” or if new walls appeared “in the southeastern corner, near the balcony.”
    Summarize any changes in labels or text, again without giving raw bounding box or polygon coordinate data.
    Provide your final output in a short, clear Markdown summary that describes where objects have changed.
    Mention if there are text/label changes (e.g., from an OCR perspective) in any particular area or region
"""
