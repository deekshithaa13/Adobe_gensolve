import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cv2
from scipy import optimize
from scipy.spatial import distance

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs, title, filename):
    colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    for i, path in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in path:
            points = [(int(x), int(y)) for x, y in XY]
            group.add(dwg.polyline(points=points, fill='none', stroke=c, stroke_width=2))
    dwg.add(group)
    dwg.save()

def identify_shape(XY):
    contour = XY.astype(np.float32)
    
    if len(XY) == 2:
        return "line", XY

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter ** 2)

    if len(approx) == 3:
        return "triangle", XY
    elif len(approx) == 4:
        return "quadrilateral", XY
    elif len(approx) == 5:
        return "pentagon", XY
    elif 0.85 < circularity < 1.15:
        return "circle", XY
    elif 0.5 < circularity < 0.85:
        return "ellipse", XY
    elif len(approx) > 5:
        return "polygon", XY
    
    return "curve", XY

def regularize_curves(paths_XYs):
    regularized_paths = []
    for path in paths_XYs:
        regularized_path = []
        for XY in path:
            shape_type, points = identify_shape(XY)
            regularized_curve = regularize_shape(shape_type, points)
            regularized_path.append(regularized_curve)
        regularized_paths.append(regularized_path)
    return regularized_paths

def regularize_shape(shape_type, points):
    if shape_type == "line":
        return points
    elif shape_type in ["triangle", "quadrilateral", "pentagon", "polygon"]:
        perimeter = cv2.arcLength(points.astype(np.float32), True)
        approx = cv2.approxPolyDP(points.astype(np.float32), 0.02 * perimeter, True)
        return approx.reshape(-1, 2)
    elif shape_type == "circle":
        (x, y), radius = cv2.minEnclosingCircle(points.astype(np.float32))
        theta = np.linspace(0, 2*np.pi, 100)
        improved_circle = np.column_stack([x + radius * np.cos(theta), y + radius * np.sin(theta)])
        return improved_circle
    elif shape_type == "ellipse":
        if len(points) < 5:
            return points
        ellipse = cv2.fitEllipse(points.astype(np.float32))
        center, axes, angle = ellipse
        major_axis, minor_axis = axes
        theta = np.linspace(0, 2*np.pi, 100)
        x = center[0] + major_axis/2 * np.cos(theta) * np.cos(np.radians(angle)) - minor_axis/2 * np.sin(theta) * np.sin(np.radians(angle))
        y = center[1] + major_axis/2 * np.cos(theta) * np.sin(np.radians(angle)) + minor_axis/2 * np.sin(theta) * np.cos(np.radians(angle))
        return np.column_stack((x, y))
    else:  # curve
        return points  # Return the original points for curves

def detect_symmetry(paths_XYs):
    symmetric_paths = []
    for path in paths_XYs:
        symmetric_path = []
        for XY in path:
            symmetry_lines = find_symmetry_lines(XY)
            if symmetry_lines:
                symmetric_curve = apply_symmetry(XY, symmetry_lines)
                symmetric_path.append(symmetric_curve)
            else:
                symmetric_path.append(XY)
        symmetric_paths.append(symmetric_path)
    return symmetric_paths

def find_symmetry_lines(XY):
    symmetry_lines = []
    contour = XY.astype(np.float32)
    
    moments = cv2.moments(contour)
    center_x = moments['m10'] / moments['m00']
    center_y = moments['m01'] / moments['m00']
    
    left = XY[XY[:, 0] <= center_x]
    right = XY[XY[:, 0] >= center_x]
    left_distances = center_x - left[:, 0]
    right_distances = right[:, 0] - center_x
    
    if len(left) == len(right) and np.allclose(left_distances, right_distances, atol=1):
        symmetry_lines.append(("vertical", center_x))
    
    top = XY[XY[:, 1] <= center_y]
    bottom = XY[XY[:, 1] >= center_y]
    top_distances = center_y - top[:, 1]
    bottom_distances = bottom[:, 1] - center_y
    
    if len(top) == len(bottom) and np.allclose(top_distances, bottom_distances, atol=1):
        symmetry_lines.append(("horizontal", center_y))
    
    return symmetry_lines

def apply_symmetry(XY, symmetry_lines):
    symmetric_XY = XY.copy()
    for sym_type, sym_value in symmetry_lines:
        if sym_type == "vertical":
            symmetric_XY[:, 0] = 2 * sym_value - symmetric_XY[:, 0]
        elif sym_type == "horizontal":
            symmetric_XY[:, 1] = 2 * sym_value - symmetric_XY[:, 1]
    return symmetric_XY

def complete_curves(paths_XYs):
    completed_paths = []
    for path in paths_XYs:
        completed_path = []
        for XY in path:
            if is_incomplete(XY):
                completed_curve = complete_curve(XY)
                completed_path.append(completed_curve)
            else:
                completed_path.append(XY)
        completed_paths.append(completed_path)
    return completed_paths

def is_incomplete(XY):
    return not np.allclose(XY[0], XY[-1], atol=1)

def complete_curve(XY):
    shape_type, _ = identify_shape(XY)
    if shape_type in ["triangle", "quadrilateral", "pentagon", "polygon", "circle", "ellipse"]:
        return np.vstack((XY, XY[0]))
    else:  # curve
        return XY  # Return the original points for curves

def beautify_curves(paths_XYs):
    beautified_paths = []
    for path in paths_XYs:
        beautified_path = []
        for XY in path:
            shape_type, _ = identify_shape(XY)
            beautified_curve = regularize_shape(shape_type, XY)
            if shape_type in ["triangle", "quadrilateral", "pentagon", "polygon", "circle", "ellipse"]:
                if not np.allclose(beautified_curve[0], beautified_curve[-1]):
                    beautified_curve = np.vstack((beautified_curve, beautified_curve[0]))
            beautified_path.append(beautified_curve)
        beautified_paths.append(beautified_path)
    return beautified_paths

def save_to_csv(paths_XYs, filename):
    with open(filename, 'w') as f:
        for i, path in enumerate(paths_XYs):
            for j, XY in enumerate(path):
                for x, y in XY:
                    f.write(f"{i},{j},{x},{y}\n")

def main():
    input_file = "examples/isolated.csv"
    paths_XYs = read_csv(input_file)

    regularized_paths = regularize_curves(paths_XYs)
    symmetric_paths = detect_symmetry(regularized_paths)
    completed_paths = complete_curves(symmetric_paths)
    beautified_paths = beautify_curves(completed_paths)

    # Save beautified curves as CSV
    save_to_csv(beautified_paths, "beautified_curves.csv")

    # Save beautified curves as SVG
    polylines2svg(beautified_paths, "output.svg")

    # Save beautified curves as a single image
    plot(beautified_paths, "Beautified Curves", "output.png")

    print("Processing complete. Outputs saved as beautified_curves.csv, output.svg, and output.png")

if __name__ == "__main__":
    main()