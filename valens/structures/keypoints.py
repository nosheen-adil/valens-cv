import json
import trt_pose.coco
import  cv2

def get_topology(data_path):
    with open(data_path, 'r') as f:
        human_pose = json.load(f)
    
    return trt_pose.coco.coco_category_to_topology(human_pose)

def trt_to_json(object_counts, objects, normalized_peaks):
    count = int(object_counts[0])
    results = {"objects" : []}
    for i in range(count):
        result = {}
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = float(peak[1])
                y = float(peak[0])
                result[str(j)] = {"x" : x, "y" : y}
        results["objects"].append(result)
    return results

def scale(results, width, height):
    for keypoints in results["objects"]:
        for k, v in keypoints.items():
            keypoints[k]["x"] = round(v["x"] * width)
            keypoints[k]["y"] = round(v["y"] * height)

def draw_on_image(topology, frame, results, color=(0, 255, 0)):
    K = topology.shape[0]
    for keypoints in results["objects"]:
        for keypoint in keypoints.values():
            cv2.circle(frame, (keypoint["x"], keypoint["y"]), 3, color, 2)

        for k in range(K):
            c_a = str(int(topology[k][2]))
            c_b = str(int(topology[k][3]))
            if c_a in keypoints and c_b in keypoints:
                cv2.line(frame, (keypoints[c_a]["x"], keypoints[c_a]["y"]), (keypoints[c_b]["x"], keypoints[c_b]["y"]), color, 2)
