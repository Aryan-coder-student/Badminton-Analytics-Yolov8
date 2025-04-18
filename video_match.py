import os 
import cv2
import torch
import numpy as np 
from more_itertools import chunked
from sklearn.cluster import KMeans
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel

model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model_path = os.path.join("runs/detect/yolov8n_custom/weights/best.pt")
yolo_model = YOLO(model_path)

#---------------------------------------------------------------------------------X-----------------------------------------------------------------

image = cv2.imread(os.path.join("data/badminton.v1i.yolov8/train/images/videoplayback_mp4-0882_jpg.rf.347258175c6d73e74f9e6e0a72c53ab8.jpg"))
results = yolo_model(image)[0]
detections = sv.Detections.from_ultralytics(results)
REFEREE_CLASS_ID = 1
filtered_detections = detections[detections.class_id != REFEREE_CLASS_ID]
box_annotation = sv.BoxAnnotator()
annotated_image = box_annotation.annotate(scene=image.copy(), detections=filtered_detections)
FOLDER_PATH_ANNOTATED = "Annotated_Image"
os.makedirs(FOLDER_PATH_ANNOTATED, exist_ok=True)
PATH_IMG = os.path.join(f"{FOLDER_PATH_ANNOTATED}/image_1", ".jpg")
cv2.imwrite(PATH_IMG, annotated_image)

with sv.ImageSink(target_dir_path=PATH_IMG) as sink:
    crop = []
    for xyxy in filtered_detections.xyxy:
        cropped_image = sv.crop_image(image=image, xyxy=xyxy)
        crop += [cropped_image]


#---------------------------------------------------------------------------------X--------------------------------------------------------------------
embeddings_list = []
crops = [sv.cv2_to_pillow(img) for img in crop]
BATCH_SIZE = 1
batch_img = chunked(crops, BATCH_SIZE)

with torch.no_grad():
    for img in batch_img:
        inputs = processor(images=img, return_tensors="pt")
        embeddings = model.get_image_features(**inputs)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        embeddings_list.append(embeddings)
embeddings_list = torch.cat(embeddings_list)

if len(embeddings_list) >= 2:
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(embeddings_list)
    
    # for i, (crop, cluster_id) in enumerate(zip(crops, clusters)):
    #     plt.subplot(1, len(crops), i+1)
    #     plt.imshow(crop)
    #     plt.title(f"Cluster {cluster_id}")
    #     plt.axis('off')
else:
    print("Need at least 2 players to cluster teams.")
#---------------------------------------------------------------------------------X------------------------------------------------------------------------

player_location_cluster = dict()
team_color = {
    0: (225, 0, 0),  # Red
    1: (0, 225, 0),  # Green
}
print("Called..............")
for i, (xyxy , cluster_id) in enumerate(zip(filtered_detections.xyxy, clusters)):
    player_location_cluster[i] = {
        "xyxy": xyxy,
        "cluster_id": cluster_id
    }

box_annotator_player = sv.BoxAnnotator()
copy_of_image = image.copy()

for plyr_id, track in player_location_cluster.items():
    ply_detect = sv.Detections(
        xyxy=np.array([track["xyxy"]]),
        class_id=np.array([track["cluster_id"]]),
    )
    box_annotator = sv.BoxAnnotator(color=sv.Color(*team_color[track["cluster_id"]]))
    annotated_image = box_annotator.annotate(
        scene=copy_of_image, detections=ply_detect
    )
    label_annotator = sv.LabelAnnotator()
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=ply_detect,
        labels=[f"Player {plyr_id} - Team {track['cluster_id']}"],
    )

output_path = os.path.join(FOLDER_PATH_ANNOTATED, "final_annotated_image.jpg")
cv2.imwrite(output_path, annotated_image)
print(f"Annotated image saved at {output_path}")














































        # sink.save_image(image=cropped_image)
# sv.plot_images_grid(
#     images=crop,
#     grid_size=(1, len(crop)),
# )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------X-------------------------------------------------------------------- 
# print(final_embeddings.shape , "Final Embeddings") 
# map = umap.UMAP(n_neighbors=min(15, final_embeddings.shape[0] - 1), n_components=min(2, final_embeddings.shape[0] - 1),min_dist=0.3)
# cluster_model = KMeans(n_clusters=2,random_state=42)
# projections = map.fit_transform(final_embeddings)

# cluster = cluster_model.fit_predict(projections)
# clustered_images = {i: [] for i in range(cluster_model.n_clusters)}
# for idx, label in enumerate(cluster):
#     clustered_images[label].append(crop[idx])
 
# for cluster_id, images in clustered_images.items():
#     print(f"Cluster {cluster_id}:")
#     sv.plot_images_grid(
#         images=images,
#         grid_size=(1, len(images)),
#     )
