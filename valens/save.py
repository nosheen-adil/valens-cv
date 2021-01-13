import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt

MODEL_WEIGHTS = 'data/resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'data/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

print("Saving model")

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
print("Loaded human pose topology in JSON")

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
print("Loaded trt model")

model.load_state_dict(torch.load(MODEL_WEIGHTS))
print("Loaded model weights")

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
print("Created example data to optimize")

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
print("Optimized model using torch2trt")

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
print("Saved optimized model")
