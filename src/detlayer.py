import time, warnings
import os
import copy
import re
import json
from typing import List, Tuple, Union, Optional, BinaryIO,Dict
from PIL import Image, ImageColor, ImageDraw, ImageFont
import numpy as np
import traceback
from pathlib import Path
import math
import torch

from collections import defaultdict
import torchvision
from enum import Enum
import uuid
import torch.nn as nn
from torchvision.io import ImageReadMode
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.ops import box_iou
from torch.optim import SGD
from torchvision import transforms
from torch.cuda import amp
from torch.utils.data import DataLoader
from scipy.cluster.vq import kmeans
import random
import torch.nn.functional as F
import validate
from torchvision.models import mobilenet_v2, inception_v3, resnet50, densenet121
from torchvision.models.inception import InceptionOutputs

from tqdm import tqdm
import cv2 as cv


class DetectionModelTrainer:

    def __init__(self) -> None:
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__cuda = (self.__device != "cpu")
        self.__model_type = ""
        self.__model = None
        self.__optimizer = None
        self.__data_dir = ""
        self.__classes: List[str] = None
        self.__num_classes = None
        self.__anchors = None
        self.__dataset_name = None
        self.__mini_batch_size: int = None
        self.__scaler = amp.GradScaler(enabled=self.__cuda)
        self.__lr_lambda = None
        self.__custom_train_dataset = None
        self.__custom_val_dataset = None
        self.__train_loader = None
        self.__val_loader = None

        self.__model_path: str = None
        self.__epochs: int = None
        self.__output_models_dir: str = None
        self.__output_json_dir: str = None

    def __set_training_param(self, epochs: int, accumulate: int) -> None:
        
        self.__lr_lambda = lambda x: (1 - x / (epochs - 1)) * (1.0 - 0.01) + 0.01
        self.__anchors = generate_anchors(
            self.__custom_train_dataset,
            n=9 if self.__model_type == "yolov3" else 6
        )
        self.__anchors = [round(i) for i in self.__anchors.reshape(-1).tolist()]
        if self.__model_type == "yolov3":
            self.__model = YoloV3(
                num_classes=self.__num_classes,
                anchors=self.__anchors,
                device=self.__device
            )
        elif self.__model_type == "tiny-yolov3":
            self.__model = YoloV3Tiny(
                num_classes=self.__num_classes,
                anchors=self.__anchors,
                device=self.__device
            )
        if self.__model_path:
            self.__load_model()

        w_d = (5e-4) * (self.__mini_batch_size * accumulate / 64)  
        g0, g1, g2 = [], [], []  
        for m in self.__model.modules():
            if hasattr(m, 'bias') and isinstance(m.bias, torch.nn.Parameter):  
                g2.append(m.bias)
            if isinstance(m, torch.nn.BatchNorm2d):  
                g0.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, torch.nn.Parameter):  
                g1.append(m.weight)

        self.__optimizer = SGD(
            g0,
            lr=1e-2,
            momentum=0.6,

            nesterov=True
        )
        self.__optimizer.add_param_group({'params': g1, 'weight_decay': w_d})  
        self.__optimizer.add_param_group({'params': g2})  
        self.__lr_scheduler = lr_scheduler.LambdaLR(
            self.__optimizer,
            lr_lambda=self.__lr_lambda
        )
        del g0, g1, g2
        self.__model.to(self.__device)

    def __load_model(self) -> None:
        try:
            state_dict = torch.load(self.__model_path, map_location=self.__device)

            new_state_dict = {k: v for k, v in state_dict.items() if
                              k in self.__model.state_dict().keys() and v.shape == self.__model.state_dict()[k].shape}
            self.__model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print("pretrained weight loading failed. Defaulting to using random weight.")

        print("=" * 20)
        print("Pretrained YOLOv3 model loaded to initialize weights")
        print("=" * 20)

    def __load_data(self) -> None:
        self.__num_classes = len(self.__classes)
        self.__dataset_name = os.path.basename(os.path.dirname(self.__data_dir + os.path.sep))
        self.__custom_train_dataset = LoadImagesAndLabels(self.__data_dir, train=True)
        self.__custom_val_dataset = LoadImagesAndLabels(self.__data_dir, train=False)
        self.__train_loader = DataLoader(
            self.__custom_train_dataset, batch_size=self.__mini_batch_size,
            shuffle=True,
            collate_fn=self.__custom_train_dataset.collate_fn
        )
        self.__val_loader = DataLoader(
            self.__custom_val_dataset, batch_size=self.__mini_batch_size // 2,
            shuffle=True, collate_fn=self.__custom_val_dataset.collate_fn
        )

    def setModelTypeAsYOLOv3(self) -> None:
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self) -> None:
  
        self.__model_type = "tiny-yolov3"

    def setDataDirectory(self, data_directory: str):

        if os.path.isdir(data_directory):
            self.__data_dir = data_directory
        else:
            raise ValueError(
                "The parameter passed should point to a valid directory"
            )

    def setTrainConfig(self, object_names_array: List[str], batch_size: int = 4, num_experiments=100,
                       train_from_pretrained_model: str = None):
        self.__model_path = train_from_pretrained_model
        if self.__model_path:
            extension_check(self.__model_path)
        self.__classes = object_names_array
        self.__mini_batch_size = batch_size
        self.__epochs = num_experiments
        self.__output_models_dir = os.path.join(self.__data_dir, "models")
        self.__output_json_dir = os.path.join(self.__data_dir, "json")

    def trainModel(self) -> None:
        self.__load_data()
        os.makedirs(self.__output_models_dir, exist_ok=True)
        os.makedirs(self.__output_json_dir, exist_ok=True)

        mp, mr, map50, map50_95, best_fitness = 0, 0, 0, 0, 0.0
        nbs = 64
        nb = len(self.__train_loader)
        nw = max(3 * nb, 1000)
        last_opt_step = -1
        prev_save_name, recent_save_name = "", ""

        accumulate = max(round(nbs / self.__mini_batch_size), 1)

        self.__set_training_param(self.__epochs, accumulate)

        with open(os.path.join(self.__output_json_dir,
                               f"{self.__dataset_name}_{self.__model_type}_detection_config.json"),
                  "w") as configWriter:
            json.dump(
                {
                    "labels": self.__classes,
                    "anchors": self.__anchors
                },
                configWriter
            )

        since = time.time()

        self.__lr_scheduler.last_epoch = -1

        for epoch in range(1, self.__epochs + 1):
            self.__optimizer.zero_grad()
            mloss = torch.zeros(3, device=self.__device)
            print(f"Epoch {epoch}/{self.__epochs}", "-" * 10, sep="\n")

            for phase in ["train", "validation"]:
                if phase == "train":
                    self.__model.train()
                    print("Train: ")
                    for batch_i, (data, anns) in tqdm(enumerate(self.__train_loader)):
                        batches_done = batch_i + nb * epoch

                        data = data.to(self.__device)
                        anns = anns.to(self.__device)


                        if batches_done <= nw:
                            xi = [0, nw]
                            accumulate = max(1, np.interp(batches_done, xi, [1, nbs / self.__mini_batch_size]).round())
                            for j, x in enumerate(self.__optimizer.param_groups):

                                x['lr'] = np.interp(batches_done, xi,
                                                    [0.1 if j == 2 else 0.0, 0.01 * self.__lr_lambda(epoch)])
                                if 'momentum' in x:
                                    x['momentum'] = np.interp(batches_done, xi, [0.8, 0.9])

                        with amp.autocast(enabled=self.__cuda):
                            _ = self.__model(data)
                            loss_layers = self.__model.get_loss_layers()
                            loss, loss_components = compute_loss(loss_layers, anns.detach(), self.__device)

                        self.__scaler.scale(loss).backward()
                        mloss = (mloss * batch_i + loss_components) / (batch_i + 1)

                        
                        if batches_done - last_opt_step >= accumulate:
                            self.__scaler.step(self.__optimizer)  
                            self.__scaler.update()
                            self.__optimizer.zero_grad()
                            last_opt_step = batches_done

                    print(
                        f"    box loss-> {float(mloss[0]):.5f}, object loss-> {float(mloss[1]):.5f}, class loss-> {float(mloss[2]):.5f}")

                    self.__lr_scheduler.step()

                else:
                    self.__model.eval()
                    print("Validation:")

                    mp, mr, map50, map50_95 = validate.run(
                        self.__model, self.__val_loader,
                        self.__num_classes, device=self.__device
                    )

                    print(
                        f"    recall: {mr:0.6f} precision: {mp:0.6f} mAP@0.5: {map50:0.6f}, mAP@0.5-0.95: {map50_95:0.6f}" "\n")

                    if map50 > best_fitness:
                        best_fitness = map50
                        recent_save_name = self.__model_type + f"_{self.__dataset_name}_mAP-{best_fitness:0.5f}_epoch-{epoch}.pt"
                        if prev_save_name:
                            os.remove(os.path.join(self.__output_models_dir, prev_save_name))
                        torch.save(
                            self.__model.state_dict(),
                            os.path.join(self.__output_models_dir, recent_save_name)
                        )
                        prev_save_name = recent_save_name

            if epoch == self.__epochs:
                torch.save(
                    self.__model.state_dict(),
                    os.path.join(self.__output_models_dir, self.__model_type + f"_{self.__dataset_name}_last.pt")
                )

        elapsed_time = time.time() - since
        print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
        torch.cuda.empty_cache()


class CustomObjectDetection:

    def __init__(self) -> None:
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__anchors: List[int] = None
        self.__classes: List[str] = None
        self.__model = None
        self.__model_loaded: bool = False
        self.__model_path: str = None
        self.__json_path: str = None
        self.__model_type: str = None
        self.__nms_score = 0.4
        self.__objectness_score = 0.4

    def setModelTypeAsYOLOv3(self) -> None:
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self) -> None:
        self.__model_type = "tiny-yolov3"

    def setModelPath(self, model_path: str):
        if os.path.isfile(model_path):
            extension_check(model_path)
            self.__model_path = model_path
            self.__model_loaded = False
        else:
            raise ValueError(
                "invalid path, path not pointing to the weightfile."
            ) from None
        self.__model_path = model_path

    def setJsonPath(self, configuration_json: str):
        self.__json_path = configuration_json

    def __load_classes_and_anchors(self) -> List[str]:

        with open(self.__json_path) as f:
            json_config = json.load(f)
            self.__anchors = json_config["anchors"]
            self.__classes = json_config["labels"]

    def __load_image_yolo(self, input_image: Union[str, np.ndarray, Image.Image]) -> Tuple[
        List[str], List[np.ndarray], torch.Tensor, torch.Tensor]:

        allowed_exts = ["jpg", "jpeg", "png"]
        fnames = []
        original_dims = []
        inputs = []
        original_imgs = []
        if type(input_image) == str:
            if os.path.isfile(input_image):
                if input_image.rsplit('.')[-1].lower() in allowed_exts:
                    img = cv.imread(input_image)
            else:
                raise ValueError(f"image path '{input_image}' is not found or a valid file")
        elif type(input_image) == np.ndarray:
            img = input_image
        elif "PIL" in str(type(input_image)):
            img = np.asarray(input_image)
        else:
            raise ValueError(f"Invalid image input format")

        img_h, img_w, _ = img.shape

        original_imgs.append(np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB)).astype(np.uint8))
        original_dims.append((img_w, img_h))
        if type(input_image) == str:
            fnames.append(os.path.basename(input_image))
        else:
            fnames.append("")
        inputs.append(prepare_image(img, (416, 416)))

        if original_dims:
            return (
                fnames,
                original_imgs,
                torch.FloatTensor(original_dims).repeat(1, 2).to(self.__device),
                torch.cat(inputs, 0).to(self.__device)
            )
        raise RuntimeError(
            f"Error loading image."
            "\nEnsure the file is a valid image,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    def useCPU(self):
        self.__device = "cpu"
        if self.__model_loaded:
            self.__model_loaded = False
            self.loadModel()

    def loadModel(self) -> None:

        self.__load_classes_and_anchors()

        if self.__model_type == "yolov3":
            self.__model = YoloV3(
                anchors=self.__anchors,
                num_classes=len(self.__classes),
                device=self.__device
            )
        elif self.__model_type == "tiny-yolov3":
            self.__model = YoloV3Tiny(
                anchors=self.__anchors,
                num_classes=len(self.__classes),
                device=self.__device
            )
        else:
            raise ValueError(
                f"Invalid model type. Call setModelTypeAsYOLOv3() or setModelTypeAsTinyYOLOv3() to set a model type before loading the model")

        self.__model.to(self.__device)

        state_dict = torch.load(self.__model_path, map_location=self.__device)
        try:
            self.__model.load_state_dict(state_dict)
            self.__model_loaded = True
            self.__model.to(self.__device).eval()
        except Exception as e:
            raise RuntimeError(f"Invalid weights!!! {e}")

    def detectObjectsFromImage(self,
                               input_image: Union[str, np.ndarray, Image.Image],
                               output_image_path: str = None,
                               output_type: str = "file",
                               extract_detected_objects: bool = False, minimum_percentage_probability: int = 40,
                               display_percentage_probability: bool = True, display_object_name: bool = True,
                               display_box: bool = True,
                               custom_objects: List = None,
                               nms_treshold: float = 0.4,
                               objectness_treshold: float = 0.4,
                               ) -> Union[
        List[List[Tuple[str, float, Dict[str, int]]]], np.ndarray, List[np.ndarray], List[str]]:

        self.__nms_score = nms_treshold
        self.__objectness_score = objectness_treshold

        self.__model.eval()
        if not self.__model_loaded:
            if self.__model_path:
                warnings.warn(
                    "Model path has changed but pretrained weights in the"
                    " new path is yet to be loaded.",
                    ResourceWarning
                )
            else:
                raise RuntimeError(
                    "Model path isn't set, pretrained weights aren't used."
                )

        predictions = defaultdict(lambda: [])

        if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
            fnames, original_imgs, input_dims, imgs = self.__load_image_yolo(input_image)

            with torch.no_grad():
                output = self.__model(imgs)

            output = get_predictions(
                pred=output.to(self.__device), num_classes=len(self.__classes),
                nms_confidence_level=self.__nms_score, objectness_confidence=self.__objectness_score,
                device=self.__device
            )

            if output is None:
                if output_type == "array":
                    if extract_detected_objects:
                        return original_imgs[0], [], []
                    else:
                        return original_imgs[0], []
                else:
                    if extract_detected_objects:
                        return original_imgs[0], []
                    else:
                        return []


            input_dims = torch.index_select(input_dims, 0, output[:, 0].long())
            scaling_factor = torch.min(416 / input_dims, 1)[0].view(-1, 1)
            output[:, [1, 3]] -= (416 - (scaling_factor * input_dims[:, 0].view(-1, 1))) / 2
            output[:, [2, 4]] -= (416 - (scaling_factor * input_dims[:, 1].view(-1, 1))) / 2
            output[:, 1:5] /= scaling_factor


            for idx in range(output.shape[0]):
                output[idx, [1, 3]] = torch.clamp(output[idx, [1, 3]], 0.0, input_dims[idx, 0])
                output[idx, [2, 4]] = torch.clamp(output[idx, [2, 4]], 0.0, input_dims[idx, 1])

            for pred in output:
                pred_label = self.__classes[int(pred[-1])]
                if custom_objects:
                    if pred_label.replace(" ", "_") in custom_objects.keys():
                        if not custom_objects[pred_label.replace(" ", "_")]:
                            continue
                    else:
                        continue
                predictions[int(pred[0])].append((
                    pred_label,
                    float(pred[-2]),
                    {k: v for k, v in zip(["x1", "y1", "x2", "y2"], map(int, pred[1:5]))},
                ))


        original_input_image = None
        output_image_array = None
        extracted_objects = []

        if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
            original_input_image = cv.cvtColor(original_imgs[0], cv.COLOR_RGB2BGR)
            if isinstance(output, torch.Tensor):
                for pred in output:
                    percentage_conf = round(float(pred[-2]) * 100, 2)
                    if percentage_conf < minimum_percentage_probability:
                        continue

                    displayed_label = ""
                    if display_object_name:
                        displayed_label = f"{self.__classes[int(pred[-1].item())]} : "
                    if display_percentage_probability:
                        displayed_label += f" {percentage_conf}%"

                    original_imgs[int(pred[0].item())] = draw_bbox_and_label(pred[1:5].int() if display_box else None,
                                                                             displayed_label,
                                                                             original_imgs[int(pred[0].item())]
                                                                             )
                output_image_array = cv.cvtColor(original_imgs[0], cv.COLOR_RGB2BGR)


        predictions_batch = list(predictions.values())
        predictions_list = predictions_batch[0] if len(predictions_batch) > 0 else []
        min_probability = minimum_percentage_probability / 100

        if output_type == "file":
            if output_image_path:
                cv.imwrite(output_image_path, output_image_array)

                if extract_detected_objects:
                    extraction_dir = ".".join(output_image_path.split(".")[:-1]) + "-extracted"
                    os.mkdir(extraction_dir)
                    count = 0
                    for obj_prediction in predictions_list:
                        if obj_prediction[1] >= min_probability:
                            count += 1
                            extracted_path = os.path.join(
                                extraction_dir,
                                ".".join(os.path.basename(output_image_path).split(".")[:-1]) + f"-{count}.jpg"
                            )
                            obj_bbox = obj_prediction[2]
                            cv.imwrite(extracted_path, original_input_image[obj_bbox["y1"]: obj_bbox["y2"],
                                                        obj_bbox["x1"]: obj_bbox["x2"]])

                            extracted_objects.append(extracted_path)

        elif output_type == "array":
            if extract_detected_objects:
                for obj_prediction in predictions_list:
                    if obj_prediction[1] >= min_probability:
                        obj_bbox = obj_prediction[2]

                        extracted_objects.append(
                            original_input_image[obj_bbox["y1"]: obj_bbox["y2"], obj_bbox["x1"]: obj_bbox["x2"]])
        else:
            raise ValueError(f"Invalid output_type '{output_type}'. Supported values are 'file' and 'array' ")

        predictions_list = [
            {
                "name": prediction[0], "percentage_probability": round(prediction[1] * 100, 2),
                "box_points": [prediction[2]["x1"], prediction[2]["y1"], prediction[2]["x2"], prediction[2]["y2"]]
            } for prediction in predictions_list if prediction[1] >= min_probability
        ]

        if output_type == "array":
            if extract_detected_objects:
                return output_image_array, predictions_list, extracted_objects
            else:
                return output_image_array, predictions_list
        else:
            if extract_detected_objects:
                return predictions_list, extracted_objects
            else:
                return predictions_list


class CustomVideoObjectDetection:

    def __init__(self):
        self.__detector = CustomObjectDetection()

    def setModelTypeAsYOLOv3(self):
        self.__detector.setModelTypeAsYOLOv3()

    def setModelTypeAsTinyYOLOv3(self):
        self.__detector.setModelTypeAsTinyYOLOv3()

    def setModelPath(self, model_path: str):
        extension_check(model_path)
        self.__detector.setModelPath(model_path)

    def setJsonPath(self, configuration_json: str):
        self.__detector.setJsonPath(configuration_json)

    def loadModel(self):
        self.__detector.loadModel()

    def useCPU(self):
        self.__detector.useCPU()

    def detectObjectsFromVideo(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=40, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, display_box=True,
                               save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, detection_timeout=None):


        if (input_file_path == "" and camera_input == None):
            raise ValueError(
                "You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")
        elif (save_detected_video == True and output_file_path == ""):
            raise ValueError(
                "You must set 'output_video_filepath' to a valid video file name, in which the detected video will be saved. If you don't intend to save the detected video, set 'save_detected_video=False'")

        else:

            output_frames_dict = {}
            output_frames_count_dict = {}

            input_video = cv.VideoCapture(input_file_path)
            if (camera_input != None):
                input_video = camera_input

            output_video_filepath = output_file_path + '.mp4'

            frame_width = int(input_video.get(3))
            frame_height = int(input_video.get(4))
            output_video = cv.VideoWriter(output_video_filepath, cv.VideoWriter_fourcc(*"MP4V"),
                                           frames_per_second,
                                           (frame_width, frame_height))

            counting = 0

            detection_timeout_count = 0
            video_frames_count = 0

            while (input_video.isOpened()):
                ret, frame = input_video.read()

                if (ret == True):

                    video_frames_count += 1
                    if (detection_timeout != None):
                        if ((video_frames_count % frames_per_second) == 0):
                            detection_timeout_count += 1

                        if (detection_timeout_count >= detection_timeout):
                            break

                    output_objects_array = []

                    counting += 1

                    if (log_progress == True):
                        print("Processing Frame : ", str(counting))

                    detected_copy = frame.copy()

                    check_frame_interval = counting % frame_detection_interval

                    if (counting == 1 or check_frame_interval == 0):
                        try:
                            detected_copy, output_objects_array = self.__detector.detectObjectsFromImage(
                                input_image=frame, output_type="array",
                                minimum_percentage_probability=minimum_percentage_probability,
                                display_percentage_probability=display_percentage_probability,
                                display_object_name=display_object_name,
                                display_box=display_box)

                        except Exception as e:
                            warnings.warn()

                    if (save_detected_video == True):
                        output_video.write(detected_copy)

                    if detected_copy is not None and output_objects_array is not None:

                        output_frames_dict[counting] = output_objects_array

                        output_objects_count = {}
                        for eachItem in output_objects_array:
                            eachItemName = eachItem["name"]
                            try:
                                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                            except:
                                output_objects_count[eachItemName] = 1

                        output_frames_count_dict[counting] = output_objects_count

                        if (counting == 1 or check_frame_interval == 0):
                            if (per_frame_function != None):
                                if (return_detected_frame == True):
                                    per_frame_function(counting, output_objects_array, output_objects_count,
                                                       detected_copy)
                                elif (return_detected_frame == False):
                                    per_frame_function(counting, output_objects_array, output_objects_count)

                        if (per_second_function != None):
                            if (counting != 1 and (counting % frames_per_second) == 0):

                                this_second_output_object_array = []
                                this_second_counting_array = []
                                this_second_counting = {}

                                for aa in range(counting):
                                    if (aa >= (counting - frames_per_second)):
                                        this_second_output_object_array.append(output_frames_dict[aa + 1])
                                        this_second_counting_array.append(output_frames_count_dict[aa + 1])

                                for eachCountingDict in this_second_counting_array:
                                    for eachItem in eachCountingDict:
                                        try:
                                            this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                             eachCountingDict[eachItem]
                                        except:
                                            this_second_counting[eachItem] = eachCountingDict[eachItem]

                                for eachCountingItem in this_second_counting:
                                    this_second_counting[eachCountingItem] = int(
                                        this_second_counting[eachCountingItem] / frames_per_second)

                                if (return_detected_frame == True):
                                    per_second_function(int(counting / frames_per_second),
                                                        this_second_output_object_array, this_second_counting_array,
                                                        this_second_counting, detected_copy)

                                elif (return_detected_frame == False):
                                    per_second_function(int(counting / frames_per_second),
                                                        this_second_output_object_array, this_second_counting_array,
                                                        this_second_counting)

                        if (per_minute_function != None):

                            if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                                this_minute_output_object_array = []
                                this_minute_counting_array = []
                                this_minute_counting = {}

                                for aa in range(counting):
                                    if (aa >= (counting - (frames_per_second * 60))):
                                        this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                        this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                                for eachCountingDict in this_minute_counting_array:
                                    for eachItem in eachCountingDict:
                                        try:
                                            this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                             eachCountingDict[eachItem]
                                        except:
                                            this_minute_counting[eachItem] = eachCountingDict[eachItem]

                                for eachCountingItem in this_minute_counting:
                                    this_minute_counting[eachCountingItem] = int(
                                        this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                                if (return_detected_frame == True):
                                    per_minute_function(int(counting / (frames_per_second * 60)),
                                                        this_minute_output_object_array, this_minute_counting_array,
                                                        this_minute_counting, detected_copy)

                                elif (return_detected_frame == False):
                                    per_minute_function(int(counting / (frames_per_second * 60)),
                                                        this_minute_output_object_array, this_minute_counting_array,
                                                        this_minute_counting)
                else:
                    break

            if (video_complete_function != None):

                this_video_output_object_array = []
                this_video_counting_array = []
                this_video_counting = {}

                for aa in range(counting):
                    this_video_output_object_array.append(output_frames_dict[aa + 1])
                    this_video_counting_array.append(output_frames_count_dict[aa + 1])

                for eachCountingDict in this_video_counting_array:
                    for eachItem in eachCountingDict:
                        try:
                            this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                            eachCountingDict[eachItem]
                        except:
                            this_video_counting[eachItem] = eachCountingDict[eachItem]

                for eachCountingItem in this_video_counting:
                    this_video_counting[eachCountingItem] = int(this_video_counting[eachCountingItem] / counting)

                video_complete_function(this_video_output_object_array, this_video_counting_array,
                                        this_video_counting)

            input_video.release()
            output_video.release()

            if (save_detected_video == True):
                return output_video_filepath


def xywh2xyxy(box_coord: torch.Tensor):
    n = box_coord.clone()
    n[:, 0] = (box_coord[:, 0] - (box_coord[:, 2] / 2))
    n[:, 1] = (box_coord[:, 1] - (box_coord[:, 3] / 2))
    n[:, 2] = (box_coord[:, 0] + (box_coord[:, 2] / 2))
    n[:, 3] = (box_coord[:, 1] + (box_coord[:, 3] / 2))

    return n


def process_batch(detections, labels, iouv):
    detections[:, [1, 3]] = torch.clamp(detections[:, [1, 3]], 0.0, 416)
    detections[:, [2, 4]] = torch.clamp(detections[:, [2, 4]], 0.0, 416)

    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, 1:5])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 7]))  
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(model, val_dataloader, num_class, net_dim=416, nms_thresh=0.6, objectness_thresh=0.001, device="cpu"):
    model.eval()
    nc = int(num_class)  
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  
    niou = iouv.numel()

    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []

    for batch_i, (im, targets) in tqdm(enumerate(val_dataloader)):
        im = im.to(device)
        targets = targets.to(device)
        nb = im.shape[0]  

        
        out = model(im)  

        
        targets[:, 2:] *= torch.Tensor([net_dim, net_dim, net_dim, net_dim]).to(device)  
        out = get_predictions(
            pred=out.to(device), num_classes=nc,
            objectness_confidence=objectness_thresh,
            nms_confidence_level=nms_thresh, device=device
        )

        
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:]
            pred = out[out[:, 0] == si, :] if isinstance(out, torch.Tensor) else torch.zeros((0, 0), device=device)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool, device="cpu"), torch.Tensor(device="cpu"),
                                  torch.Tensor(device="cpu"), tcls))
                continue

            
            if nc == 1:
                pred[:, 7] = 0

            if pred.shape[0] > 300:
                pred = pred[:300, :]

            predn = pred.clone()

            
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]).to(device)  
                labelsn = torch.cat((labels[:, 0:1], tbox), 1).to(device)  
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 7].cpu(), tcls))  


    stats = [np.concatenate(x, 0) for x in zip(*stats)]  
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    return mp, mr, map50, map
def ap_per_class(tp, conf, pred_cls, target_cls):


    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]


    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]


    px = np.linspace(0, 1, 1000)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue
        else:

            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)


            recall = tpc / (n_l + 1e-16)  
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

            precision = tpc / (tpc + fpc)  
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)


            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])


    f1 = 2 * p * r / (p + r + 1e-16)
    i = f1.mean(0).argmax()

    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))


    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))


    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  

    return ap, mpre, mrec

class LoadImagesAndLabels(Dataset):

    def __init__(self, path: str, net_dim=(416, 416), train=True):
        if not os.path.isdir(path):
            raise NotADirectoryError("path is not a valid directory!!!")

        super().__init__()

        if train:
            path = os.path.join(path, "train")
        else:
            path = os.path.join(path, "validation")

        self.__net_width, self.__net_height = net_dim
        self.__images_paths = []
        self.shapes = []
        self.labels = []
        for img in os.listdir(os.path.join(path, "images")):
            p = os.path.join(path, "images", img)
            image = cv.imread(p)
            if isinstance(image, np.ndarray):
                l_p = self.__img_path2label_path(p)
                self.__images_paths.append(p)
                self.shapes.append((image.shape[1], image.shape[0]))
                self.labels.append(self.__load_raw_label(l_p))

        self.__nsamples = len(self.__images_paths)
        self.shapes = np.array(self.shapes)

    def __len__(self) -> int:
        return self.__nsamples

    def __img_path2label_path(self, path: str) -> str:
        im, lb = os.sep + "images" + os.sep, os.sep + "annotations" + os.sep
        return lb.join(path.rsplit(im, 1)).rsplit(".", 1)[0] + ".txt"

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.__nsamples:
            raise IndexError("Index out of range.")
        image_path = self.__images_paths[idx]
        label = self.labels[idx].copy()
        image, label = self.__load_data(image_path, label)
        return image, label

    def __xywhn2xyxy(self, nlabel: torch.Tensor, width: int, height: int) -> torch.Tensor:
        label = nlabel.clone()
        label[:, 1] = (nlabel[:, 1] - (nlabel[:, 3] / 2)) * width
        label[:, 2] = (nlabel[:, 2] - (nlabel[:, 4] / 2)) * height
        label[:, 3] = (nlabel[:, 1] + (nlabel[:, 3] / 2)) * width
        label[:, 4] = (nlabel[:, 2] + (nlabel[:, 4] / 2)) * height

        return label

    def __load_data(self, img_path: str, label: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv.imread(img_path)
        img_h, img_w = img.shape[:2]
        img = prepare_image(img[:, :, :3], [self.__net_width, self.__net_height])
        lab = self.__process_label(label, img_w, img_h)
        return img.squeeze(), lab

    def __load_raw_label(self, label_path: str):
        if os.path.isfile(label_path):
            with warnings.catch_warnings():
                l = np.loadtxt(label_path).reshape(-1, 5)
                assert (l >= 0).all(), "bounding box values should be positive and in range 0 - 1"
                assert (l[:, 1:] <= 1).all(), "bounding box values should be in the range 0 - 1"
        else:
            l = np.zeros((0, 5), dtype=np.float32)
        return l

    def __process_label(self, label: np.ndarray, image_width: int, image_height: int) -> torch.Tensor:

        scaling_factor = min(
            self.__net_width / image_width,
            self.__net_width / image_height
        )

        bs = torch.zeros((len(label), 6))
        if label.size > 0:
            nlabels = torch.from_numpy(label)
            labels = self.__xywhn2xyxy(nlabels, image_width, image_height)

            labels[:, [1, 3]] = ((labels[:, [1, 3]] * scaling_factor) + \
                                 (self.__net_width - (image_width * scaling_factor)) / 2)
            labels[:, [2, 4]] = ((labels[:, [2, 4]] * scaling_factor) + \
                                 (self.__net_width - (image_height * scaling_factor)) / 2)


            label_copy = labels.clone()
            labels[:, 1] = (label_copy[:, 3] + label_copy[:, 1]) / 2
            labels[:, 2] = (label_copy[:, 4] + label_copy[:, 2]) / 2
            labels[:, 3] = (label_copy[:, 3] - label_copy[:, 1])
            labels[:, 4] = (label_copy[:, 4] - label_copy[:, 2])


            labels[:, 1:5] /= self.__net_width
            bs[:, 1:] = labels[:, :]
        return bs

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = [data for data in batch if data is not None]
        imgs, bboxes = list(zip(*batch))

        imgs = torch.stack(imgs)

        for i, boxes in enumerate(bboxes):
            boxes[:, 0] = i
        bboxes = torch.cat(bboxes, 0)

        return imgs, bboxes
def generate_anchors(dataset, n=9, img_size=416, thr=4.0, gen=1000, verbose=True):
    thr = 1 / thr

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        return x, x.max(1)[0]  

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]
        if verbose:
            x, best = metric(k, wh0)
            bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  
            s = f'thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
                f'n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
                f'past_thr={x[x > thr].mean():.3f}-mean: '
            print(s)
        return k
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  

    
    i = (wh0 < 3.0).any(1).sum()
    if i and verbose:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  

    s = wh.std(0)  
    k, dist = kmeans(wh / s, n, iter=30)  
    assert len(k) == n, f'ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  
    wh0 = torch.tensor(wh0, dtype=torch.float32)  
    k = print_results(k, verbose=False)

    
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  
    if verbose:
        print("Generating anchor boxes for training images...")
    for _ in range(gen):
        v = np.ones(sh)
        while (v == 1).all():  
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()

    return print_results(k)

class ImageReadMode(Enum):

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


class ObjectDetection:


    def __init__(self) -> None:
        self.__device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__nms_score: float = 0.4
        self.__objectness_score: float = 0.5
        self.__anchors: List[int] = None
        self.__anchors_yolov3: List[int] = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373,
                                            326]
        self.__anchors_tiny_yolov3: List[int] = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]

        self.__classes = self.__load_classes(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco_classes.txt"))
        self.__model_type = ""
        self.__model = None
        self.__model_loaded = False
        self.__model_path = ""

    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image_yolo(self, input_image: Union[str, np.ndarray, Image.Image]) -> Tuple[
        List[str], List[np.ndarray], torch.Tensor, torch.Tensor]:
        allowed_exts = ["jpg", "jpeg", "png"]
        fnames = []
        original_dims = []
        inputs = []
        original_imgs = []
        if type(input_image) == str:
            if os.path.isfile(input_image):
                if input_image.rsplit('.')[-1].lower() in allowed_exts:
                    img = cv.imread(input_image)
            else:
                raise ValueError(f"image path '{input_image}' is not found or a valid file")
        elif type(input_image) == np.ndarray:
            img = input_image
        elif "PIL" in str(type(input_image)):
            img = np.asarray(input_image)
        else:
            raise ValueError(f"Invalid image input format")

        img_h, img_w, _ = img.shape

        original_imgs.append(np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB)).astype(np.uint8))
        original_dims.append((img_w, img_h))
        if type(input_image) == str:
            fnames.append(os.path.basename(input_image))
        else:
            fnames.append("")
        inputs.append(prepare_image(img, (416, 416)))

        if original_dims:
            return (
                fnames,
                original_imgs,
                torch.FloatTensor(original_dims).repeat(1, 2).to(self.__device),
                torch.cat(inputs, 0).to(self.__device)
            )
        raise RuntimeError(
            f"Error loading image."
            "\nEnsure the file is a valid image,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    def __save_temp_img(self, input_image: Union[np.ndarray, Image.Image]) -> str:

        temp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{str(uuid.uuid4())}.jpg"
        )
        if type(input_image) == np.ndarray:
            cv.imwrite(temp_path, input_image)
        elif "PIL" in str(type(input_image)):
            input_image.save(temp_path)
        else:
            raise ValueError(
                f"Invalid image input. Supported formats are OpenCV/Numpy array, PIL image or image file path"
            )

        return temp_path

    def __load_image_retinanet(self, input_image: str) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:

        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        scaled_images = []
        fnames = []

        delete_file = False
        if type(input_image) is not str:
            input_image = self.__save_temp_img(input_image=input_image)
            delete_file = True

        if os.path.isfile(input_image):
            if input_image.rsplit('.')[-1].lower() in allowed_file_extensions:
                img = read_image(input_image, ImageReadMode.RGB)
                images.append(img)
                scaled_images.append(img.div(255.0).to(self.__device))
                fnames.append(os.path.basename(input_image))
        else:
            raise ValueError(f"Input image with path {input_image} not a valid file")

        if delete_file:
            os.remove(input_image)

        if images:
            return (fnames, images, scaled_images)
        raise RuntimeError(
            f"Error loading image from input."
            "\nEnsure the folder contains images,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    def setModelTypeAsYOLOv3(self):

        self.__anchors = self.__anchors_yolov3
        self.__model_type = "yolov3"

    def setModelTypeAsTinyYOLOv3(self):

        self.__anchors = self.__anchors_tiny_yolov3
        self.__model_type = "tiny-yolov3"

    def setModelTypeAsRetinaNet(self):

        self.__anchors = self.__anchors_tiny_yolov3
        self.__model_type = "retinanet"

    def setModelPath(self, path: str) -> None:

        if os.path.isfile(path):
            extension_check(path)
            self.__model_path = path
            self.__model_loaded = False
        else:
            raise ValueError(
                "invalid path, path not pointing to a valid file."
            ) from None

    def useCPU(self):

        self.__device = "cpu"
        if self.__model_loaded:
            self.__model_loaded = False
            self.loadModel()

    def loadModel(self) -> None:

        if not self.__model_loaded:
            if self.__model_type == "yolov3":
                self.__model = YoloV3(
                    anchors=self.__anchors,
                    num_classes=len(self.__classes), \
                    device=self.__device
                )
            elif self.__model_type == "tiny-yolov3":
                self.__model = YoloV3Tiny(
                    anchors=self.__anchors,
                    num_classes=len(self.__classes),
                    device=self.__device
                )
            elif self.__model_type == "retinanet":

                self.__classes = self.__load_classes(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco91_classes.txt"))

                self.__model = torchvision.models.detection.retinanet_resnet50_fpn(
                    pretrained=False, num_classes=91,
                    pretrained_backbone=False
                )
            else:
                raise ValueError(
                    f"Invalid model type. Call setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3() or setModelTypeAsRetinaNet to set a model type before loading the model")

            state_dict = torch.load(self.__model_path, map_location=self.__device)
            try:
                self.__model.load_state_dict(state_dict)
                self.__model_loaded = True
                self.__model.to(self.__device).eval()
            except:
                raise RuntimeError("Invalid weights!!!") from None

    def CustomObjects(self, **kwargs):


        if not self.__model_loaded:
            self.loadModel()
        all_objects_str = (obj_label.replace(" ", "_") for obj_label in self.__classes)
        all_objects_dict = {}
        for object_str in all_objects_str:
            all_objects_dict[object_str] = False

        for karg in kwargs:
            if karg in all_objects_dict:
                all_objects_dict[karg] = kwargs[karg]
            else:
                raise ValueError(f" object '{karg}' doesn't exist in the supported object classes")

        return all_objects_dict

    def detectObjectsFromImage(self,
                               input_image: Union[str, np.ndarray, Image.Image],
                               output_image_path: str = None,
                               output_type: str = "file",
                               extract_detected_objects: bool = False, minimum_percentage_probability: int = 50,
                               display_percentage_probability: bool = True, display_object_name: bool = True,
                               display_box: bool = True,
                               custom_objects: List = None
                               ) -> Union[
        List[List[Tuple[str, float, Dict[str, int]]]], np.ndarray, List[np.ndarray], List[str]]:

        self.__model.eval()
        if not self.__model_loaded:
            if self.__model_path:
                warnings.warn(
                    "Model path has changed but pretrained weights in the"
                    " new path is yet to be loaded.",
                    ResourceWarning
                )
            else:
                raise RuntimeError(
                    "Model path isn't set, pretrained weights aren't used."
                )
        predictions = defaultdict(lambda: [])

        if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
            fnames, original_imgs, input_dims, imgs = self.__load_image_yolo(input_image)

            with torch.no_grad():
                output = self.__model(imgs)

            output = get_predictions(
                pred=output.to(self.__device), num_classes=len(self.__classes),
                nms_confidence_level=self.__nms_score, objectness_confidence=self.__objectness_score,
                device=self.__device
            )

            if output is None:
                if output_type == "array":
                    if extract_detected_objects:
                        return original_imgs[0], [], []
                    else:
                        return original_imgs[0], []
                else:
                    if extract_detected_objects:
                        return original_imgs[0], []
                    else:
                        return []

            
            input_dims = torch.index_select(input_dims, 0, output[:, 0].long())
            scaling_factor = torch.min(416 / input_dims, 1)[0].view(-1, 1)
            output[:, [1, 3]] -= (416 - (scaling_factor * input_dims[:, 0].view(-1, 1))) / 2
            output[:, [2, 4]] -= (416 - (scaling_factor * input_dims[:, 1].view(-1, 1))) / 2
            output[:, 1:5] /= scaling_factor

            
            for idx in range(output.shape[0]):
                output[idx, [1, 3]] = torch.clamp(output[idx, [1, 3]], 0.0, input_dims[idx, 0])
                output[idx, [2, 4]] = torch.clamp(output[idx, [2, 4]], 0.0, input_dims[idx, 1])

            for pred in output:
                pred_label = self.__classes[int(pred[-1])]
                if custom_objects:
                    if pred_label.replace(" ", "_") in custom_objects.keys():
                        if not custom_objects[pred_label.replace(" ", "_")]:
                            continue
                    else:
                        continue
                predictions[int(pred[0])].append((
                    pred_label,
                    float(pred[-2]),
                    {k: v for k, v in zip(["x1", "y1", "x2", "y2"], map(int, pred[1:5]))},
                ))
        elif self.__model_type == "retinanet":
            fnames, original_imgs, scaled_images = self.__load_image_retinanet(input_image)
            with torch.no_grad():
                output = self.__model(scaled_images)

            if output is None:
                if output_type == "array":
                    if extract_detected_objects:
                        return original_imgs[0], [], []
                    else:
                        return original_imgs[0], []
                else:
                    if extract_detected_objects:
                        return original_imgs[0], []
                    else:
                        return []

            for idx, pred in enumerate(output):
                for id in range(pred["labels"].shape[0]):
                    if pred["scores"][id] >= self.__objectness_score:
                        pred_label = self.__classes[pred["labels"][id]]

                        if custom_objects:
                            if pred_label.replace(" ", "_") in custom_objects.keys():
                                if not custom_objects[pred_label.replace(" ", "_")]:
                                    continue
                            else:
                                continue

                        predictions[idx].append(
                            (
                                pred_label,
                                pred["scores"][id].item(),
                                {k: v for k, v in zip(["x1", "y1", "x2", "y2"], map(int, pred["boxes"][id]))}
                            )
                        )

        
        original_input_image = None
        output_image_array = None
        extracted_objects = []

        if self.__model_type == "yolov3" or self.__model_type == "tiny-yolov3":
            original_input_image = cv.cvtColor(original_imgs[0], cv.COLOR_RGB2BGR)
            if isinstance(output, torch.Tensor):
                for pred in output:
                    percentage_conf = round(float(pred[-2]) * 100, 2)
                    if percentage_conf < minimum_percentage_probability:
                        continue

                    displayed_label = ""
                    if display_object_name:
                        displayed_label = f"{self.__classes[int(pred[-1].item())]} : "
                    if display_percentage_probability:
                        displayed_label += f" {percentage_conf}%"

                    original_imgs[int(pred[0].item())] = draw_bbox_and_label(pred[1:5].int() if display_box else None,
                                                                             displayed_label,
                                                                             original_imgs[int(pred[0].item())]
                                                                             )

                output_image_array = cv.cvtColor(original_imgs[0], cv.COLOR_RGB2BGR)

        elif self.__model_type == "retinanet":
            original_input_image = tensor_to_ndarray(original_imgs[0].div(255.0))
            original_input_image = cv.cvtColor(original_input_image, cv.COLOR_RGB2BGR)
            for idx, pred in predictions.items():

                max_dim = max(list(original_imgs[idx].size()))

                for label, score, bbox in pred:
                    percentage_conf = round(score * 100, 2)
                    if percentage_conf < minimum_percentage_probability:
                        continue

                    displayed_label = ""
                    if display_object_name:
                        displayed_label = f"{label} :"
                    if display_percentage_probability:
                        displayed_label += f" {percentage_conf}%"

                    original_imgs[idx] = draw_bounding_boxes_and_labels(
                        image=original_imgs[idx],
                        boxes=torch.Tensor([[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]]),
                        draw_boxes=display_box,
                        labels=[displayed_label],
                        label_color=(0, 0, 255),
                        box_color=(0, 255, 0),
                        width=1,
                        fill=False,
                        font_size=int(max_dim / 30)
                    )

            output_image_array = tensor_to_ndarray(original_imgs[0].div(255.0))
            output_image_array = cv.cvtColor(output_image_array, cv.COLOR_RGB2BGR)

        
        predictions_batch = list(predictions.values())
        predictions_list = predictions_batch[0] if len(predictions_batch) > 0 else []
        min_probability = minimum_percentage_probability / 100

        if output_type == "file":
            if output_image_path:
                cv.imwrite(output_image_path, output_image_array)

                if extract_detected_objects:
                    extraction_dir = ".".join(output_image_path.split(".")[:-1]) + "-extracted"
                    os.mkdir(extraction_dir)
                    count = 0
                    for obj_prediction in predictions_list:
                        if obj_prediction[1] >= min_probability:
                            count += 1
                            extracted_path = os.path.join(
                                extraction_dir,
                                ".".join(os.path.basename(output_image_path).split(".")[:-1]) + f"-{count}.jpg"
                            )
                            obj_bbox = obj_prediction[2]
                            cv.imwrite(extracted_path, original_input_image[obj_bbox["y1"]: obj_bbox["y2"],
                                                        obj_bbox["x1"]: obj_bbox["x2"]])

                            extracted_objects.append(extracted_path)

        elif output_type == "array":
            if extract_detected_objects:
                for obj_prediction in predictions_list:
                    if obj_prediction[1] >= min_probability:
                        obj_bbox = obj_prediction[2]

                        extracted_objects.append(
                            original_input_image[obj_bbox["y1"]: obj_bbox["y2"], obj_bbox["x1"]: obj_bbox["x2"]])
        else:
            raise ValueError(f"Invalid output_type '{output_type}'. Supported values are 'file' and 'array' ")

        predictions_list = [
            {
                "name": prediction[0], "percentage_probability": round(prediction[1] * 100, 2),
                "box_points": [prediction[2]["x1"], prediction[2]["y1"], prediction[2]["x2"], prediction[2]["y2"]]
            } for prediction in predictions_list if prediction[1] >= min_probability
        ]

        if output_type == "array":
            if extract_detected_objects:
                return output_image_array, predictions_list, extracted_objects
            else:
                return output_image_array, predictions_list
        else:
            if extract_detected_objects:
                return predictions_list, extracted_objects
            else:
                return predictions_list


class VideoObjectDetection:

    def __init__(self):
        self.__detector = ObjectDetection()

    def setModelTypeAsYOLOv3(self):
        self.__detector.setModelTypeAsYOLOv3()

    def setModelTypeAsTinyYOLOv3(self):
        self.__detector.setModelTypeAsTinyYOLOv3()

    def setModelTypeAsRetinaNet(self):
        self.__detector.setModelTypeAsRetinaNet()

    def setModelPath(self, model_path: str):
        extension_check(model_path)
        self.__detector.setModelPath(model_path)

    def loadModel(self):
        self.__detector.loadModel()

    def useCPU(self):
        self.__detector.useCPU()

    def CustomObjects(self, **kwargs):
        return self.__detector.CustomObjects(**kwargs)

    def detectObjectsFromVideo(self, input_file_path="", camera_input=None, output_file_path="", frames_per_second=20,
                               frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False,
                               display_percentage_probability=True, display_object_name=True, display_box=True,
                               save_detected_video=True,
                               per_frame_function=None, per_second_function=None, per_minute_function=None,
                               video_complete_function=None, return_detected_frame=False, detection_timeout=None,
                               custom_objects=None):

        if (input_file_path == "" and camera_input == None):
            raise ValueError(
                "You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")
        elif (save_detected_video == True and output_file_path == ""):
            raise ValueError(
                "You must set 'output_video_filepath' to a valid video file name, in which the detected video will be saved. If you don't intend to save the detected video, set 'save_detected_video=False'")

        else:
            try:

                output_frames_dict = {}
                output_frames_count_dict = {}

                input_video = cv.VideoCapture(input_file_path)
                if (camera_input != None):
                    input_video = camera_input

                output_video_filepath = output_file_path + '.mp4'

                frame_width = int(input_video.get(3))
                frame_height = int(input_video.get(4))
                output_video = cv.VideoWriter(output_video_filepath, cv.VideoWriter_fourcc(*"MP4V"),
                                               frames_per_second,
                                               (frame_width, frame_height))

                counting = 0

                detection_timeout_count = 0
                video_frames_count = 0

                while (input_video.isOpened()):
                    ret, frame = input_video.read()

                    if (ret == True):

                        video_frames_count += 1
                        if (detection_timeout != None):
                            if ((video_frames_count % frames_per_second) == 0):
                                detection_timeout_count += 1

                            if (detection_timeout_count >= detection_timeout):
                                break

                        output_objects_array = []

                        counting += 1

                        if (log_progress == True):
                            print("Processing Frame : ", str(counting))

                        detected_copy = frame.copy()

                        check_frame_interval = counting % frame_detection_interval

                        if (counting == 1 or check_frame_interval == 0):
                            try:
                                detected_copy, output_objects_array = self.__detector.detectObjectsFromImage(
                                    input_image=frame, output_type="array",
                                    minimum_percentage_probability=minimum_percentage_probability,
                                    display_percentage_probability=display_percentage_probability,
                                    display_object_name=display_object_name,
                                    display_box=display_box,
                                    custom_objects=custom_objects)
                            except:
                                None

                        output_frames_dict[counting] = output_objects_array

                        output_objects_count = {}
                        for eachItem in output_objects_array:
                            eachItemName = eachItem["name"]
                            try:
                                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                            except:
                                output_objects_count[eachItemName] = 1

                        output_frames_count_dict[counting] = output_objects_count

                        if (save_detected_video == True):
                            output_video.write(detected_copy)

                        if (counting == 1 or check_frame_interval == 0):
                            if (per_frame_function != None):
                                if (return_detected_frame == True):
                                    per_frame_function(counting, output_objects_array, output_objects_count,
                                                       detected_copy)
                                elif (return_detected_frame == False):
                                    per_frame_function(counting, output_objects_array, output_objects_count)

                        if (per_second_function != None):
                            if (counting != 1 and (counting % frames_per_second) == 0):

                                this_second_output_object_array = []
                                this_second_counting_array = []
                                this_second_counting = {}

                                for aa in range(counting):
                                    if (aa >= (counting - frames_per_second)):
                                        this_second_output_object_array.append(output_frames_dict[aa + 1])
                                        this_second_counting_array.append(output_frames_count_dict[aa + 1])

                                for eachCountingDict in this_second_counting_array:
                                    for eachItem in eachCountingDict:
                                        try:
                                            this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                             eachCountingDict[eachItem]
                                        except:
                                            this_second_counting[eachItem] = eachCountingDict[eachItem]

                                for eachCountingItem in this_second_counting:
                                    this_second_counting[eachCountingItem] = int(
                                        this_second_counting[eachCountingItem] / frames_per_second)

                                if (return_detected_frame == True):
                                    per_second_function(int(counting / frames_per_second),
                                                        this_second_output_object_array, this_second_counting_array,
                                                        this_second_counting, detected_copy)

                                elif (return_detected_frame == False):
                                    per_second_function(int(counting / frames_per_second),
                                                        this_second_output_object_array, this_second_counting_array,
                                                        this_second_counting)

                        if (per_minute_function != None):

                            if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                                this_minute_output_object_array = []
                                this_minute_counting_array = []
                                this_minute_counting = {}

                                for aa in range(counting):
                                    if (aa >= (counting - (frames_per_second * 60))):
                                        this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                        this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                                for eachCountingDict in this_minute_counting_array:
                                    for eachItem in eachCountingDict:
                                        try:
                                            this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                             eachCountingDict[eachItem]
                                        except:
                                            this_minute_counting[eachItem] = eachCountingDict[eachItem]

                                for eachCountingItem in this_minute_counting:
                                    this_minute_counting[eachCountingItem] = int(
                                        this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                                if (return_detected_frame == True):
                                    per_minute_function(int(counting / (frames_per_second * 60)),
                                                        this_minute_output_object_array, this_minute_counting_array,
                                                        this_minute_counting, detected_copy)

                                elif (return_detected_frame == False):
                                    per_minute_function(int(counting / (frames_per_second * 60)),
                                                        this_minute_output_object_array, this_minute_counting_array,
                                                        this_minute_counting)


                    else:
                        break

                if (video_complete_function != None):

                    this_video_output_object_array = []
                    this_video_counting_array = []
                    this_video_counting = {}

                    for aa in range(counting):
                        this_video_output_object_array.append(output_frames_dict[aa + 1])
                        this_video_counting_array.append(output_frames_count_dict[aa + 1])

                    for eachCountingDict in this_video_counting_array:
                        for eachItem in eachCountingDict:
                            try:
                                this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                                eachCountingDict[eachItem]
                            except:
                                this_video_counting[eachItem] = eachCountingDict[eachItem]

                    for eachCountingItem in this_video_counting:
                        this_video_counting[eachCountingItem] = int(this_video_counting[eachCountingItem] / counting)

                    video_complete_function(this_video_output_object_array, this_video_counting_array,
                                            this_video_counting)

                input_video.release()
                output_video.release()

                if (save_detected_video == True):
                    return output_video_filepath

            except:
                raise ValueError(
                    "An error occured. It may be that your input video is invalid. Ensure you specified a proper string value for 'output_file_path' is 'save_detected_video' is not False. "
                    "Also ensure your per_frame, per_second, per_minute or video_complete_analysis function is properly configured to receive the right parameters. ")
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    
    box2 = box2.T

    
    if x1y1x2y2:  
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  
        if CIoU or DIoU:  
            c2 = cw ** 2 + ch ** 2 + eps  
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  
            if DIoU:
                return iou - rho2 / c2  
            elif CIoU:  
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  
        else:  
            c_area = cw * ch + eps  
            return iou - (c_area - union) / c_area  
    else:
        return iou  


def compute_loss(loss_layers, targets, device="cpu"):
    nc = loss_layers[0].num_classes
    nl = len(loss_layers)
    
    predictions = [layer.pred for layer in loss_layers]

    
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    
    tcls, tbox, indices, anchors = build_targets(predictions, targets, loss_layers, device)  

    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    balance = [4.0, 1.0, 0.4]

    
    for layer_index, layer_predictions in enumerate(predictions):
        
        b, anchor, grid_j, grid_i = indices[layer_index]
        
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  
        
        
        
        num_targets = b.shape[0]
        
        if num_targets:
            
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            
            
            pxy = ps[:, :2].sigmoid() * 2 - 0.5
            
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[layer_index]
            
            pbox = torch.cat((pxy, pwh), 1)
            
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            
            lbox += (1.0 - iou).mean()  

            
            
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  

            
            
            if nc > 1:
                
                t = torch.full_like(ps[:, 5:], 0.0, device=device)  
                t[range(num_targets), tcls[layer_index]] = 1
                
                lcls += BCEcls(ps[:, 5:], t)  

        
        
        obji = BCEobj(layer_predictions[..., 4], tobj) 
        lobj += obji * balance[layer_index]

    lbox *= 0.05
    lobj *= (1.0 * ((416 / 640) ** 2)) 
    lcls *= (0.5 * (nc / 80))  

    
    loss = (lbox + lobj + lcls) * tobj.shape[0]

    return loss, (torch.cat((lbox, lobj, lcls))).detach()


def build_targets(p, targets, loss_layers, device="cpu"):
    
    na, nt = len(loss_layers[0].anchors), targets.shape[0]  
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=device)  
    
    ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)
    
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    g = 0.5
    off = torch.tensor([
                        [0, 0], [1, 0], [0, 1],
                        [-1, 0], [0, -1]
                        ], device=device).float() * g 

    for i, yolo_layer in enumerate(loss_layers):
        
        anchors = yolo_layer.anchors / yolo_layer.stride
        
        
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  
        
        t = targets * gain
        
        if nt:
            
            r = t[:, :, 4:6] / anchors[:, None]
            
            j = torch.max(r, 1.0 / r).max(2)[0] < 4.0  
            
            
            
            t = t[j]

            
            gxy = t[:, 2:4] 
            gxi = gain[[2,3]] - gxy
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        
        b, c = t[:, :2].long().T
        
        
        gxy = t[:, 2:4] 
        gwh = t[:, 4:6]  
        
        gij = (gxy - offsets).long()
        
        gi, gj = gij.T  

        
        a = t[:, 6].long()
        
        
        indices.append((b, a, gj.clamp_(0, int(gain[3] - 1)), gi.clamp_(0, int(gain[2] - 1))))
        
        tbox.append(torch.cat((gxy - gij, gwh), 1))  
        
        anch.append(anchors[a])
        
        tcls.append(c)

    return tcls, tbox, indices, anch

def noop(x):
    return x


class DetectionLayer(nn.Module):

    def __init__(
            self,
            anchors: Union[List[int], Tuple[int, ...]],
            anchor_masks: Tuple[int, int, int],
            layer: int,
            num_classes: int = 80,
            device: str = "cpu"
    ):
        super().__init__()
        self.height = 416
        self.width = 416
        self.num_classes = num_classes
        self.ignore_thresh = 0.7
        self.truth_thresh = 1
        self.rescore = 1
        self.device = device
        self.anchors = self.__get_anchors(anchors, anchor_masks)
        self.layer = layer
        self.layer_width = None
        self.layer_height = None
        self.layer_output = None
        self.pred = None
        self.stride = None
        self.grid = None
        self.anchor_grid = None

    def __get_anchors(
            self, anchors: Union[List[int], Tuple[int, ...]],
            anchor_masks: Tuple[int, int, int]
    ) -> torch.Tensor:
        a = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        return torch.tensor([a[i] for i in anchor_masks]).to(self.device)

    def forward(self, x: torch.Tensor):
        self.layer_height, self.layer_width = x.shape[2], x.shape[3]
        self.stride = self.height // self.layer_height
        if self.training:
            batch_size = x.shape[0]
            grid_size = x.shape[2]
            bbox_attrs = 5 + self.num_classes
            num_anchors = len(self.anchors)

            
            self.layer_output = x.detach()
            self.pred = x.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4,
                                                                                                  2).contiguous()

            self.layer_output = self.layer_output.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
            self.layer_output = self.layer_output.transpose(1, 2).contiguous()
            self.layer_output = self.layer_output.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

        else:
            
            
            self.layer_output = transform_prediction(
                x.data, self.width, self.anchors, self.num_classes,
                self.device
            )
        return self.layer_output


class ConvLayer(nn.Module):

    def __init__(self, in_f: int, out_f: int, kernel_size: int = 3,
                 stride: int = 1, use_batch_norm: bool = True,
                 activation: str = "leaky"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_f, out_f, stride=stride, kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False if use_batch_norm else True
        )
        self.batch_norm = nn.BatchNorm2d(out_f) if use_batch_norm else noop
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True) if activation == "leaky" else noop

    def forward(self, x: torch.Tensor):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YoloV3(nn.Module):

    def __init__(
            self,
            anchors: Union[List[int], Tuple[int, ...]],
            num_classes: int = 80,
            device: str = "cpu"):
        super().__init__()

        
        self.conv1 = ConvLayer(3, 32)
        self.conv2 = ConvLayer(32, 64, stride=2)
        self.conv3 = ConvLayer(64, 32, 1, 1)
        self.conv4 = ConvLayer(32, 64)
        
        self.conv5 = ConvLayer(64, 128, stride=2)
        self.conv6 = ConvLayer(128, 64, 1, 1)
        self.conv7 = ConvLayer(64, 128, stride=1)
        
        self.conv8 = ConvLayer(128, 64, 1, 1)
        self.conv9 = ConvLayer(64, 128, stride=1)
        
        self.conv10 = ConvLayer(128, 256, stride=2)
        self.conv11 = ConvLayer(256, 128, 1, 1)
        self.conv12 = ConvLayer(128, 256)
        
        self.conv13 = ConvLayer(256, 128, 1, 1)
        self.conv14 = ConvLayer(128, 256)
        
        self.conv15 = ConvLayer(256, 128, 1, 1)
        self.conv16 = ConvLayer(128, 256)
        
        self.conv17 = ConvLayer(256, 128, 1, 1)
        self.conv18 = ConvLayer(128, 256)
        
        self.conv19 = ConvLayer(256, 128, 1, 1)
        self.conv20 = ConvLayer(128, 256)
        
        self.conv21 = ConvLayer(256, 128, 1, 1)
        self.conv22 = ConvLayer(128, 256)
        
        self.conv23 = ConvLayer(256, 128, 1, 1)
        self.conv24 = ConvLayer(128, 256)
        
        self.conv25 = ConvLayer(256, 128, 1, 1)
        self.conv26 = ConvLayer(128, 256)
        
        self.conv27 = ConvLayer(256, 512, stride=2)
        self.conv28 = ConvLayer(512, 256, 1, 1)
        self.conv29 = ConvLayer(256, 512)
        
        self.conv30 = ConvLayer(512, 256, 1, 1)
        self.conv31 = ConvLayer(256, 512)
        
        self.conv32 = ConvLayer(512, 256, 1, 1)
        self.conv33 = ConvLayer(256, 512)
        
        self.conv34 = ConvLayer(512, 256, 1, 1)
        self.conv35 = ConvLayer(256, 512)
        
        self.conv36 = ConvLayer(512, 256, 1, 1)
        self.conv37 = ConvLayer(256, 512)
        
        self.conv38 = ConvLayer(512, 256, 1, 1)
        self.conv39 = ConvLayer(256, 512)
        
        self.conv40 = ConvLayer(512, 256, 1, 1)
        self.conv41 = ConvLayer(256, 512)
        
        self.conv42 = ConvLayer(512, 256, 1, 1)
        self.conv43 = ConvLayer(256, 512)
        
        self.conv44 = ConvLayer(512, 1024, stride=2)
        self.conv45 = ConvLayer(1024, 512, 1, 1)
        self.conv46 = ConvLayer(512, 1024)
        
        self.conv47 = ConvLayer(1024, 512, 1, 1)
        self.conv48 = ConvLayer(512, 1024)
        
        self.conv49 = ConvLayer(1024, 512, 1, 1)
        self.conv50 = ConvLayer(512, 1024)
        
        self.conv51 = ConvLayer(1024, 512, 1, 1)
        self.conv52 = ConvLayer(512, 1024)
        
        self.conv53 = ConvLayer(1024, 512, 1, 1)
        self.conv54 = ConvLayer(512, 1024)
        self.conv55 = ConvLayer(1024, 512, 1, 1)
        self.conv56 = ConvLayer(512, 1024)
        self.conv57 = ConvLayer(1024, 512, 1, 1)
        self.conv58 = ConvLayer(512, 1024)
        self.conv59 = ConvLayer(
            1024, (3 * (5 + num_classes)), 1, 1, use_batch_norm=False,
            activation="linear"
        )

        
        self.yolo1 = DetectionLayer(
            num_classes=num_classes, anchors=anchors,
            anchor_masks=(6, 7, 8), device=device, layer=1
        )

        
        self.conv60 = ConvLayer(512, 256, 1, 1)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="nearest"
            
        )
        
        self.conv61 = ConvLayer(768, 256, 1, 1)
        self.conv62 = ConvLayer(256, 512)
        self.conv63 = ConvLayer(512, 256, 1, 1)
        self.conv64 = ConvLayer(256, 512)
        self.conv65 = ConvLayer(512, 256, 1, 1)
        self.conv66 = ConvLayer(256, 512)
        self.conv67 = ConvLayer(
            512, (3 * (5 + num_classes)), 1, 1, use_batch_norm=False,
            activation="linear"
        )

        
        self.yolo2 = DetectionLayer(
            num_classes=num_classes, anchors=anchors,
            anchor_masks=(3, 4, 5), device=device, layer=2
        )

        
        self.conv68 = ConvLayer(256, 128, 1, 1)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="nearest"
            
        )
        

        self.conv69 = ConvLayer(384, 128, 1, 1)
        self.conv70 = ConvLayer(128, 256)
        self.conv71 = ConvLayer(256, 128, 1, 1)
        self.conv72 = ConvLayer(128, 256)
        self.conv73 = ConvLayer(256, 128, 1, 1)
        self.conv74 = ConvLayer(128, 256)
        self.conv75 = ConvLayer(
            256, (3 * (5 + num_classes)), 1, 1, use_batch_norm=False,
            activation="linear"
        )

        
        self.yolo3 = DetectionLayer(
            num_classes=num_classes, anchors=anchors,
            anchor_masks=(0, 1, 2), device=device, layer=3
        )

    def get_loss_layers(self) -> List[torch.Tensor]:
        return [self.yolo1, self.yolo2, self.yolo3]

    def __route_layer(self, y1: torch.Tensor, y2: Optional[torch.Tensor] = None):
        if isinstance(y2, torch.Tensor):
            return torch.cat([y1, y2], 1)
        return y1

    def __shortcut_layer(self,
                         y1: torch.Tensor, y2: torch.Tensor,
                         activation: str = "linear"
                         ) -> torch.Tensor:
        actv = noop if activation == "linear" else nn.LeakyReLU(0.1)
        return actv(y1 + y2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        
        y = self.conv5(self.__shortcut_layer(self.conv4(self.conv3(y)), y))
        y2 = self.conv7(self.conv6(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv9(self.conv8(y))
        
        y2 = self.conv10(self.__shortcut_layer(y2, y))
        y = self.conv12(self.conv11(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv14(self.conv13(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv16(self.conv15(self.__shortcut_layer(y2, y)))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv18(self.conv17(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv20(self.conv19(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv22(self.conv21(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv24(self.conv23(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv26(self.conv25(y2))
        
        r1 = self.__shortcut_layer(y, y2)  
        y = self.conv27(r1)
        y2 = self.conv29(self.conv28(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv31(self.conv30(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv33(self.conv32(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv35(self.conv34(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv37(self.conv36(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv39(self.conv38(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv41(self.conv40(y))
        
        y = self.__shortcut_layer(y2, y)
        y2 = self.conv43(self.conv42(y))
        
        r2 = self.__shortcut_layer(y2, y)  
        y2 = self.conv44(r2)
        y = self.conv46(self.conv45(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv48(self.conv47(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv50(self.conv49(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv52(self.conv51(y2))
        
        y2 = self.__shortcut_layer(y, y2)
        y = self.conv54(self.conv53(y2))
        r3 = self.conv57(self.conv56(self.conv55(y)))  
        y = self.conv59(self.conv58(r3))

        
        out = self.yolo1(y)
        y = self.conv60(self.__route_layer(r3))
        y = self.conv62(self.conv61(self.__route_layer(self.upsample1(y), r2)))
        r4 = self.conv65(self.conv64(self.conv63(y)))  
        y = self.conv67(self.conv66(r4))

        
        out = torch.cat([out, self.yolo2(y)], dim=1)
        y = self.conv68(self.__route_layer(r4))
        y = self.conv70(self.conv69(self.__route_layer(self.upsample2(y), r1)))
        y = self.conv75(self.conv74(self.conv73(self.conv72(self.conv71(y)))))

        
        out = torch.cat([out, self.yolo3(y)], dim=1)

        return out

def draw_bbox_and_label(x: torch.Tensor, label: str, img: np.ndarray) -> np.ndarray:

    x1, y1, x2, y2 = tuple(map(int, x))
    if x is not None:
        img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = (x1 + t_size[0] + 3, y1 + t_size[1] + 4)
    img = cv.putText(img, label, (x1, y1 + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    return img


def letterbox_image(
        image: np.ndarray,
        inp_dim: Tuple[int, int]) -> np.ndarray:

    img_w, img_h = image.shape[1], image.shape[0]  
    net_w, net_h = inp_dim  


    scale_factor = min(net_w / img_w, net_h / img_h)
    new_w = int(round(img_w * scale_factor))
    new_h = int(round(img_h * scale_factor))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    canvas = np.full((net_w, net_h, 3), 128)
    canvas[(net_h - new_h) // 2: (net_h - new_h) // 2 + new_h, (net_w - new_w) // 2: (net_w - new_w) // 2 + new_w,
    :] = resized_image
    return canvas


def prepare_image(
        image: np.ndarray,
        inp_dim: Tuple[int, int]) -> torch.Tensor:

    img = letterbox_image(image, inp_dim)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def bbox_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, device="cpu"):

    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape, device=device)) * \
                 torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_y2.shape, device=device))

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def transform_prediction(
        pred: torch.Tensor,
        inp_dim: int,
        anchors: Union[List[int], Tuple[int, ...], torch.Tensor],
        num_classes: int,
        device: str = "cpu"
) -> torch.Tensor:
    batch_size = pred.shape[0]
    grid_size = pred.shape[2]
    stride = inp_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    
    pred = pred.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    pred = pred.transpose(1, 2).contiguous()
    pred = pred.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    
    
    
    
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    
    pred[:, :, 0] = torch.sigmoid(pred[:, :, 0])
    pred[:, :, 1] = torch.sigmoid(pred[:, :, 1])
    pred[:, :, 4] = torch.sigmoid(pred[:, :, 4])

    
    grid = torch.arange(grid_size, dtype=torch.float)
    grid = np.arange(grid_size)
    x_o, y_o = np.meshgrid(grid, grid)
    

    x_offset = torch.FloatTensor(x_o).view(-1, 1).to(device)
    y_offset = torch.FloatTensor(y_o).view(-1, 1).to(device)
    
    

    x_y_offset = torch.cat([x_offset, y_offset], dim=1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    pred[:, :, :2] += x_y_offset

    
    anchors = torch.FloatTensor(anchors).to(device)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    pred[:, :, 2:4] = torch.exp(pred[:, :, 2:4]) * anchors

    
    pred[:, :, 5:5 + num_classes] = torch.sigmoid(pred[:, :, 5:5 + num_classes])

    
    pred[:, :, :4] *= stride

    return pred


def get_predictions(
        pred: torch.Tensor,
        num_classes: int,
        objectness_confidence: float = 0.5,
        nms_confidence_level: float = 0.4,
        device: str = "cpu") -> Union[torch.Tensor, int]:

    conf_mask = (pred[:, :, 4] > objectness_confidence).float().unsqueeze(2)
    pred = pred * conf_mask

    bbox_corner = pred.new(pred.shape)
    bbox_corner[:, :, 0] = (pred[:, :, 0] - (pred[:, :, 2] / 2))  
    bbox_corner[:, :, 1] = (pred[:, :, 1] - (pred[:, :, 3] / 2))  
    bbox_corner[:, :, 2] = (pred[:, :, 0] + (pred[:, :, 2] / 2))  
    bbox_corner[:, :, 3] = (pred[:, :, 1] + (pred[:, :, 3] / 2))  
    pred[:, :, :4] = bbox_corner[:, :, :4]

    output = None
    for idx in range(pred.shape[0]):
        img_pred = pred[idx]

        max_conf, max_idx = torch.max(img_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1).to(device)
        max_idx = max_idx.float().unsqueeze(1).to(device)
        img_pred = torch.cat([img_pred[:, :5], max_conf, max_idx], 1)

        non_zero_idx = torch.nonzero(img_pred[:, 4]).to(device)
        img_pred = img_pred[non_zero_idx.squeeze(), :].view(-1, 7).to(device)
        if not img_pred.shape[0]:
            continue

        img_classes = torch.unique(img_pred[:, -1]).to(device)

        for cls in img_classes:
            class_mask = img_pred * (img_pred[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(class_mask[:, -2]).squeeze()
            img_pred_class = img_pred[class_mask_idx].view(-1, 7)

            
            conf_sort_idx = torch.sort(img_pred_class[:, 4], descending=True)[1]
            img_pred_class = img_pred_class[conf_sort_idx]

            for d_idx in range(img_pred_class.shape[0]):
                try:
                    ious = bbox_iou(img_pred_class[d_idx].unsqueeze(0), img_pred_class[d_idx + 1:], device=device)
                except (IndexError, ValueError):
                    break

                
                iou_mask = (ious < nms_confidence_level).float().unsqueeze(1)
                img_pred_class[d_idx + 1:] *= iou_mask
                non_zero_idx = torch.nonzero(img_pred_class[:, 4]).squeeze()
                img_pred_class = img_pred_class[non_zero_idx].view(-1, 7)

            batch_idx = img_pred_class.new(img_pred_class.shape[0], 1).fill_(idx)
            if isinstance(output, torch.Tensor):
                out = torch.cat([batch_idx, img_pred_class], 1)
                output = torch.cat([output, out])
            else:
                output = torch.cat([batch_idx, img_pred_class], 1)
    return output

class YoloV3Tiny(nn.Module):

    def __init__(
            self,
            anchors: Union[List[int], Tuple[int, ...]],
            num_classes: int = 80,
            device: str = "cpu"
    ):
        super().__init__()

        
        self.conv1 = ConvLayer(3, 16)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvLayer(16, 32)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvLayer(32, 64)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvLayer(64, 128)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvLayer(128, 256)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.conv6 = ConvLayer(256, 512)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool6 = nn.MaxPool2d(2, 1)
        self.conv7 = ConvLayer(512, 1024)
        self.conv8 = ConvLayer(1024, 256, 1, 1)
        self.conv9 = ConvLayer(256, 512)
        self.conv10 = ConvLayer(
            512, (3 * (5 + num_classes)), 1, 1,
            use_batch_norm=False,
            activation="linear"
        )
        self.yolo1 = DetectionLayer(
            num_classes=num_classes, anchors=anchors,
            anchor_masks=(3, 4, 5), device=device, layer=1
        )
        
        self.conv11 = ConvLayer(256, 128, 1, 1)
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="nearest"
            
        )
        
        self.conv12 = ConvLayer(384, 256)
        self.conv13 = ConvLayer(
            256, (3 * (5 + num_classes)), 1, 1,
            use_batch_norm=False,
            activation="linear"
        )
        self.yolo2 = DetectionLayer(
            num_classes=num_classes, anchors=anchors,
            anchor_masks=(0, 1, 2), device=device, layer=2
        )

    def get_loss_layers(self) -> List[torch.Tensor]:
        return [self.yolo1, self.yolo2]

    def __route_layer(self, y1: torch.Tensor, y2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(y2, torch.Tensor):
            return torch.cat([y1, y2], 1)
        return y1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.maxpool2(self.conv2(self.maxpool1(self.conv1(x))))
        y = self.maxpool4(self.conv4(self.maxpool3(self.conv3(y))))
        r1 = self.conv5(y)  
        y = self.zeropad(self.conv6(self.maxpool5(r1)))
        y = self.conv7(self.maxpool6(y))
        r2 = self.conv8(y)  
        y = self.conv10(self.conv9(r2))

        
        out = self.yolo1(y)
        y = self.conv11(self.__route_layer(r2))
        y = self.__route_layer(self.upsample1(y), r1)
        y = self.conv13(self.conv12(y))

        
        out = torch.cat([out, self.yolo2(y)], 1)

        return out
warnings.filterwarnings("once", category=ResourceWarning)


class ResNet50Pretrained:

    def __init__(self, label_path: str) -> None:
        self.__model = torchvision.models.resnet50(pretrained=False)
        self.__classes = self.__load_classes(label_path)
        self.__has_loaded_weights = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model_path = ""

    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image(self, image_path: str) -> Tuple[List[str], torch.Tensor]:
        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        fnames = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            images.append(preprocess(img))
            fnames.append(os.path.basename(image_path))

        elif os.path.isdir(image_path):
            for file in os.listdir(image_path):
                if os.path.isfile(os.path.join(image_path, file)) and \
                        file.rsplit('.')[-1].lower() in allowed_file_extensions:
                    img = Image.open(os.path.join(image_path, file)).convert("RGB")
                    images.append(preprocess(img))
                    fnames.append(file)
        if images:
            return fnames, torch.stack(images)
        raise RuntimeError(
            f"Error loading images from {os.path.abspath(image_path)}."
            "\nEnsure the folder contains images,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    
    model_path = property(
        fget=lambda self: self.__model_path,
        fset=lambda self, path: self.set_model_path(path),
        doc="Path containing the pretrained weight."
    )

    def set_model_path(self, path: str) -> None:

        if os.path.isfile(path):
            self.__model_path = path
            self.__has_loaded_weights = False
        else:
            raise ValueError(
                "parameter path should be a path to the pretrianed weight file."
            )

    def load_model(self) -> None:

        if not self.__has_loaded_weights:
            try:
                self.__model.load_state_dict(
                    torch.load(self.__model_path, map_location=self.__device)
                )
                self.__has_loaded_weights = True
                self.__model.eval()
            except Exception:
                print("Weight loading failed.\nEnsure the model path is"
                      " set and the weight file is in the specified model path.")

    def classify(self, image_path: str, top_n: int = 5, verbose: bool = True) -> List[List[Tuple[str, str]]]:

        if not self.__has_loaded_weights:
            if self.__model_path:
                warnings.warn(
                    "Model path has changed but pretrained weights in the"
                    " new path are yet to be loaded.",
                    ResourceWarning
                )
            else:
                warnings.warn(
                    "Model path isn't set, pretrained weights aren't used.",
                    ResourceWarning
                )

        fnames, images = self.__load_image(image_path)
        images = images.to(self.__device)

        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        predictions = [
            [
                (self.__classes[top5_catid[i][j]], f"{top5_prob[i][j].item() * 100:.5f}%")
                for j in range(top5_prob.shape[1])
            ]
            for i in range(top5_prob.shape[0])
        ]

        if verbose:
            for idx, pred in enumerate(predictions):
                print("-" * 50, f"Top 5 predictions for {fnames[idx]}", "-" * 50, sep="\n")
                for label, score in pred:
                    print(f"\t{label}:{score: >10}")
                print("-" * 50, "\n")
        return predictions


class MobileNetV2Pretrained:

    def __init__(self, label_path: str) -> None:
        self.__model = torchvision.models.mobilenet_v2(pretrained=False)
        self.__classes = self.__load_classes(label_path)
        self.__has_loaded_weights = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model_path = ""

    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image(self, image_path: str) -> Tuple[List[str], torch.Tensor]:

        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        fnames = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            images.append(preprocess(img))
            fnames.append(os.path.basename(image_path))

        elif os.path.isdir(image_path):
            for file in os.listdir(image_path):
                if os.path.isfile(os.path.join(image_path, file)) and \
                        file.rsplit('.')[-1].lower() in allowed_file_extensions:
                    img = Image.open(os.path.join(image_path, file)).convert("RGB")
                    images.append(preprocess(img))
                    fnames.append(file)
        if images:
            return fnames, torch.stack(images)
        raise RuntimeError(
            f"Error loading images from {os.path.abspath(image_path)}."
            "\nEnsure the folder contains images,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    
    model_path = property(
        fget=lambda self: self.__model_path,
        fset=lambda self, path: self.set_model_path(path),
        doc="Path containing the pretrained weight."
    )

    def set_model_path(self, path: str) -> None:

        if os.path.isfile(path):
            self.__model_path = path
            self.__has_loaded_weight = False
        else:
            raise ValueError(
                "parameter path should be a valid path to the pretrianed weight file."
            )

    def load_model(self) -> None:

        if not self.__has_loaded_weights:
            try:
                self.__model.load_state_dict(
                    torch.load(self.__model_path, map_location=self.__device)
                )
                self.__has_loaded_weights = True
                self.__model.eval()
            except Exception:
                print("Weight loading failed.\nEnsure the model path is"
                      " set and the weight file is in the specified model path.")

    def classify(self, image_path: str, top_n: int = 5, verbose: bool = True) -> List[List[Tuple[str, str]]]:

        if not self.__has_loaded_weights:
            if self.__model_path:
                warnings.warn(
                    "Model path has changed but pretrained weights in the"
                    " new path are yet to be loaded.",
                    ResourceWarning
                )
            else:
                warnings.warn(
                    "Model path isn't set, pretrained weights aren't used.",
                    ResourceWarning
                )

        fnames, images = self.__load_image(image_path)
        images = images.to(self.__device)

        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        predictions = [
            [
                (self.__classes[top5_catid[i][j]], f"{top5_prob[i][j].item() * 100:.5f}%")
                for j in range(top5_prob.shape[1])
            ]
            for i in range(top5_prob.shape[0])
        ]

        if verbose:
            for idx, pred in enumerate(predictions):
                print("-" * 50, f"Top 5 predictions for {fnames[idx]}", "-" * 50, sep="\n")
                for label, score in pred:
                    print(f"\t{label}:{score: >10}")
                print("-" * 50, "\n")
        return predictions


class InceptionV3Pretrained:

    def __init__(self, label_path: str) -> None:
        self.__model = torchvision.models.inception_v3(pretrained=False)
        self.__classes = self.__load_classes(label_path)
        self.__has_loaded_weights = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model_path = ""

    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image(self, image_path: str) -> Tuple[List[str], torch.Tensor]:
        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        fnames = []
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            images.append(preprocess(img))
            fnames.append(os.path.basename(image_path))

        elif os.path.isdir(image_path):
            for file in os.listdir(image_path):
                if os.path.isfile(os.path.join(image_path, file)) and \
                        file.rsplit('.')[-1].lower() in allowed_file_extensions:
                    img = Image.open(os.path.join(image_path, file)).convert("RGB")
                    images.append(preprocess(img))
                    fnames.append(file)
        if images:
            return fnames, torch.stack(images)
        raise RuntimeError(
            f"Error loading images from {os.path.abspath(image_path)}."
            "\nEnsure the folder contains images,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    
    model_path = property(
        fget=lambda self: self.__model_path,
        fset=lambda self, path: self.set_model_path(path),
        doc="Path containing the pretrained weight."
    )

    def set_model_path(self, path: str) -> None:

        if os.path.isfile(path):
            self.__model_path = path
            self.__has_loaded_weights = False
        else:
            raise ValueError(
                "parameter path should be a path to the pretrianed weight file."
            )

    def load_model(self) -> None:

        if not self.__has_loaded_weights:
            try:
                self.__model.load_state_dict(
                    torch.load(self.__model_path, map_location=self.__device)
                )
                self.__has_loaded_weights = True
                self.__model.eval()
            except Exception:
                print("Weight loading failed.\nEnsure the model path is"
                      " set and the weight file is in the specified model path.")

    def classify(self, image_path: str, top_n: int = 5, verbose: bool = True) -> List[List[Tuple[str, str]]]:
        if not self.__has_loaded_weights:
            if self.__model_path:
                warnings.warn(
                    "Model path has changed but pretrained weights in the"
                    " new path are yet to be loaded.",
                    ResourceWarning
                )
            else:
                warnings.warn(
                    "Model path isn't set, pretrained weights aren't used.",
                    ResourceWarning
                )

        fnames, images = self.__load_image(image_path)
        images = images.to(self.__device)
        print(images.shape)

        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        with open(os.path.join(str(Path(__file__).resolve().parent.parent), "imagenet_classes.txt")) as f:
            categories = [c.strip() for c in f.readlines()]
        predictions = [
            [
                (categories[top5_catid[i][j]], f"{top5_prob[i][j].item() * 100:.5f}%")
                for j in range(top5_prob.shape[1])
            ]
            for i in range(top5_prob.shape[0])
        ]

        if verbose:
            for idx, pred in enumerate(predictions):
                print("-" * 50, f"Top 5 predictions for {fnames[idx]}", "-" * 50, sep="\n")
                for label, score in pred:
                    print(f"\t{label}:{score: >10}")
                print("-" * 50, "\n")
        return predictions


def read_file(path: str) -> torch.Tensor:

    data = torch.ops.image.read_file(path)
    return data


def decode_image(input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:

    output = torch.ops.image.decode_image(input, mode.value)
    return output


def read_image(path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:

    data = read_file(path)
    return decode_image(data, mode)


def _generate_color_palette(num_objects: int):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_objects)]


@torch.no_grad()
def make_grid(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: float = 0.0,
        **kwargs,
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if "range" in kwargs.keys():
        warnings.warn(
            "The parameter 'range' is deprecated since 0.12 and will be removed in 0.14. "
            "Please use 'value_range' instead."
        )
        value_range = kwargs["range"]

    
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  
        if tensor.size(0) == 1:  
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    assert isinstance(tensor, torch.Tensor)
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(  
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def draw_bounding_boxes_and_labels(
        image: torch.Tensor,
        boxes: torch.Tensor,
        draw_boxes: bool,
        labels: Optional[List[str]] = None,
        label_color: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        box_color: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        fill: Optional[bool] = False,
        width: int = 1,
        font: Optional[str] = None,
        font_size: int = 10,
) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")

    num_boxes = boxes.shape[0]

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for bbox, label in zip(img_boxes, labels):
        if draw_boxes:
            if fill:
                fill_color = label_color + (100,)
                draw.rectangle(bbox, width=width, outline=label_color, fill=fill_color)
            else:
                draw.rectangle(bbox, width=width, outline=box_color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=label_color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


@torch.no_grad()
def tensor_to_ndarray(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
) -> None:

    grid = make_grid(tensor, **kwargs)
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    return ndarr


class DenseNet121Pretrained:

    def __init__(self, label_path: str) -> None:
        self.__model = torchvision.models.densenet121(pretrained=False)
        self.__classes = self.__load_classes(label_path)
        self.__has_loaded_weights = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model_path = ""

    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def __load_image(self, image_path: str) -> Tuple[List[str], torch.Tensor]:
        allowed_file_extensions = ["jpg", "jpeg", "png"]
        images = []
        fnames = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if os.path.isfile(image_path):
            img = Image.open(image_path).convert("RGB")
            images.append(preprocess(img))
            fnames.append(os.path.basename(image_path))

        elif os.path.isdir(image_path):
            for file in os.listdir(image_path):
                if os.path.isfile(os.path.join(image_path, file)) and \
                        file.rsplit('.')[-1].lower() in allowed_file_extensions:
                    img = Image.open(os.path.join(image_path, file)).convert("RGB")
                    images.append(preprocess(img))
                    fnames.append(file)
        if images:
            return fnames, torch.stack(images)
        raise RuntimeError(
            f"Error loading images from {os.path.abspath(image_path)}."
            "\nEnsure the folder contains images,"
            " allowed file extensions are .jpg, .jpeg, .png"
        )

    
    model_path = property(
        fget=lambda self: self.__model_path,
        fset=lambda self, path: self.set_model_path(path),
        doc="Path containing the pretrained weight."
    )

    def set_model_path(self, path: str) -> None:
        if os.path.isfile(path):
            self.__model_path = path
            self.__has_loaded_weights = False
        else:
            raise ValueError(
                "parameter path should be a path to the pretrianed weight file."
            )

    def load_model(self) -> None:
        if not self.__has_loaded_weights:
            try:
                import re
                state_dict = torch.load(self.__model_path, map_location=self.__device)
                
                
                pattern = re.compile(
                    r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                    "(?:weight|bias|running_mean|running_var))$"
                )
                for key in list(state_dict.keys()):
                    res = pattern.match(key)
                    if res:
                        new_key = res.group(1) + res.group(2)
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                self.__model.load_state_dict(state_dict)
                self.__has_loaded_weights = True
                self.__model.eval()
            except Exception:
                print("Weight loading failed.\nEnsure the model path is"
                      " set and the weight file is in the specified model path.")

    def classify(self, image_path: str, top_n: int = 5, verbose: bool = True) -> List[List[Tuple[str, str]]]:
        if not self.__has_loaded_weights:
            warnings.warn("Pretrained weights aren't loaded", ResourceWarning)

        fnames, images = self.__load_image(image_path)
        images = images.to(self.__device)

        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        predictions = [
            [
                (self.__classes[top5_catid[i][j]], f"{top5_prob[i][j].item() * 100:.5f}%")
                for j in range(top5_prob.shape[1])
            ]
            for i in range(top5_prob.shape[0])
        ]

        if verbose:
            for idx, pred in enumerate(predictions):
                print("-" * 50, f"Top 5 predictions for {fnames[idx]}", "-" * 50, sep="\n")
                for label, score in pred:
                    print(f"\t{label}:{score: >10}")
                print("-" * 50, "\n")
        return predictions
classification_models = {
    "resnet50": {
        "model": resnet50(pretrained=False)
    },
    "densenet121": {
        "model": densenet121(pretrained=False)
    },
    "inceptionv3": {
        "model": inception_v3(pretrained=False)
    },
    "mobilenetv2": {
        "model": mobilenet_v2(pretrained=False)
    }
}


class ImageClassification:
    def __init__(self) -> None:
        self.__model_type: str = None
        self.__model: Union[resnet50, densenet121, mobilenet_v2, inception_v3] = None
        self.__model_path: str = None
        self.__classes_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_classes.txt")
        self.__model_loaded: bool = False
        self.__device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.__classes: List[str] = []

    def setModelPath(self, path: str):
        if os.path.isfile(path):
            extension_check(path)
            self.__model_path = path
        else:
            raise ValueError(
                f"The path '{path}' isn't a valid file. Ensure you specify the path to a valid trained model file."
            )

    def __load_classes(self) -> List[str]:
        with open(self.__classes_path) as f:
            self.__classes = [c.strip() for c in f.readlines()]

    def __load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        images = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if type(image_input) == str:
            if os.path.isfile(image_input):
                img = Image.open(image_input).convert("RGB")
                images.append(preprocess(img))
            else:
                raise ValueError(f"image path '{image_input}' is not found or a valid file")
        elif type(image_input) == np.ndarray:
            img = Image.fromarray(image_input).convert("RGB")
            images.append(preprocess(img))
        elif "PIL" in str(type(image_input)):
            img = image_input.convert("RGB")
            images.append(preprocess(img))
        else:
            raise ValueError(f"Invalid image input format")

        return torch.stack(images)

    def setModelTypeAsResNet50(self):
        if self.__model_type == None:
            self.__model_type = "resnet50"

    def setModelTypeAsDenseNet121(self):
        if self.__model_type == None:
            self.__model_type = "densenet121"

    def setModelTypeAsInceptionV3(self):
        if self.__model_type == None:
            self.__model_type = "inceptionv3"

    def setModelTypeAsMobileNetV2(self):
        if self.__model_type == None:
            self.__model_type = "mobilenetv2"

    def useCPU(self):
        self.__device = "cpu"
        if self.__model_loaded:
            self.__model_loaded = False
            self.loadModel()

    def loadModel(self):
        if not self.__model_loaded:
            try:
                if self.__model_path == None:
                    raise ValueError(
                        "Model path not specified. Call '.setModelPath()' and parse the path to the model file before loading the model."
                    )

                if self.__model_type in classification_models.keys():
                    self.__model = classification_models[self.__model_type]["model"]
                else:
                    raise ValueError(
                        f"Model type '{self.__model_type}' not supported."
                    )
                state_dict = torch.load(self.__model_path)
                if self.__model_type == "densenet121":
                    
                    
                    pattern = re.compile(
                        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                        "(?:weight|bias|running_mean|running_var))$"
                    )
                    for key in list(state_dict.keys()):
                        res = pattern.match(key)
                        if res:
                            new_key = res.group(1) + res.group(2)
                            state_dict[new_key] = state_dict[key]
                            del state_dict[key]

                self.__model.load_state_dict(
                    state_dict
                )
                self.__model.to(self.__device)
                self.__model_loaded = True
                self.__model.eval()
                self.__load_classes()
            except Exception:
                print(traceback.print_exc())
                print("Weight loading failed.\nEnsure the model path is"
                      " set and the weight file is in the specified model path.")

    def classifyImage(self, image_input: Union[str, np.ndarray, Image.Image], result_count: int = 5) -> Tuple[
        List[str], List[float]]:
        if not self.__model_loaded:
            raise RuntimeError(
                "Model not yet loaded. You need to call '.loadModel()' before performing image classification"
            )

        images = self.__load_image(image_input)
        images = images.to(self.__device)

        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        topN_prob, topN_catid = torch.topk(probabilities, result_count)

        predictions = [
            [
                (self.__classes[topN_catid[i][j]], topN_prob[i][j].item() * 100)
                for j in range(topN_prob.shape[1])
            ]
            for i in range(topN_prob.shape[0])
        ]

        labels_pred = []
        probabilities_pred = []

        for idx, pred in enumerate(predictions):
            for label, score in pred:
                labels_pred.append(label)
                probabilities_pred.append(round(score, 4))

        return labels_pred, probabilities_pred

def resnet50_train_params():
    model = resnet50(pretrained=False)
    return {
        "model": model,
        "optimizer": SGD,
        "weight_decay": 1e-4,
        "lr": 0.1,
        "lr_decay_rate": None,
        "lr_step_size": None
    }


def inception_v3_train_params():
    model = inception_v3(pretrained=False, init_weights=False)

    return {
        "model": model,
        "optimizer": SGD,
        "weight_decay": 0,
        "lr": 0.045,
        "lr_decay_rate": 0.94,
        "lr_step_size": 2
    }


def mobilenet_v2_train_params():
    model = mobilenet_v2(pretrained=False)

    return {
        "model": model,
        "optimizer": SGD,
        "weight_decay": 4e-5,
        "lr": 0.045,
        "lr_decay_rate": 0.98,
        "lr_step_size": 1
    }


def densenet121_train_params():
    model = densenet121(pretrained=False)

    return {
        "model": model,
        "optimizer": SGD,
        "weight_decay": 1e-4,
        "lr": 0.1,
        "lr_decay_rate": None,
        "lr_step_size": None,
    }
data_transforms1 = {
            "train":transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]
                                    )
                    ]),
            "test": transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]
                                    )
                    ])
        }

data_transforms2 = {
            "train":transforms.Compose([
                        transforms.RandomResizedCrop(299),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]
                                    )
                    ]),
            "test": transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]
                                    )
                    ])
        }

def extension_check(file_path: str):
    if file_path.endswith(".h5"):
        raise RuntimeError("Fara modele .h5")
    elif file_path.endswith(".pt") == False and file_path.endswith(".pth") == False:
        raise ValueError(f"Model nevalid {os.path.basename(file_path)}. Adaugati model '.pt' sau '.pth' ")

try:
    import torch
    import torchvision
except:
    try:
        import tensorflow
        import keras

        raise RuntimeError("Lipsa Torch")
    except:
        raise RuntimeError("Torch neinstalat")

class ClassificationModelTrainer():
    def __init__(self) -> None:
        self.__model_type = ""
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__data_dir = ""
        self.__data_loaders = None
        self.__class_names = None
        self.__dataset_sizes = None
        self.__dataset_name = ""
        self.__model = None
        self.__optimizer = None
        self.__lr_scheduler = None
        self.__loss_fn = nn.CrossEntropyLoss()
        self.__transfer_learning_mode = "fine_tune_all"
        self.__model_path = ""
        self.__training_params = None

    def __set_training_param(self) -> None:
        if not self.__model_type:
            raise RuntimeError("The model type is not set!!!")
        self.__model = self.__training_params["model"]
        optimizer = self.__training_params["optimizer"]
        lr_decay_rate = self.__training_params["lr_decay_rate"]
        lr_step_size = self.__training_params["lr_step_size"]
        lr = self.__training_params["lr"]
        weight_decay = self.__training_params["weight_decay"]

        if self.__model_path:
            self.__set_transfer_learning_mode()
            print("==> Transfer learning enabled")

        
        
        
        if self.__model_type == "mobilenet_v2":
            in_features = self.__model.classifier[1].in_features
            self.__model.classifier[1] = nn.Linear(in_features, len(self.__class_names))
        elif self.__model_type == "densenet121":
            in_features = self.__model.classifier.in_features
            self.__model.classifier = nn.Linear(in_features, len(self.__class_names))
        else:
            in_features = self.__model.fc.in_features
            self.__model.fc = nn.Linear(in_features, len(self.__class_names))

        self.__model.to(self.__device)
        self.__optimizer = optimizer(
            self.__model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        if lr_decay_rate and lr_step_size:
            self.__lr_scheduler = lr_scheduler.StepLR(
                self.__optimizer,
                gamma=lr_decay_rate,
                step_size=lr_step_size
            )

    def __set_transfer_learning_mode(self) -> None:

        state_dict = torch.load(self.__model_path)
        if self.__model_type == "densenet121":
            
            
            pattern = re.compile(
                r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                "(?:weight|bias|running_mean|running_var))$"
            )
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        self.__model.load_state_dict(state_dict)
        self.__model.to(self.__device)

        if self.__transfer_learning_mode == "freeze_all":
            for param in self.__model.parameters():
                param.requires_grad = False

    def __load_data(self, batch_size: int = 8) -> None:

        if not self.__data_dir:
            raise RuntimeError("The dataset directory not yet set.")
        image_dataset = {
            x: datasets.ImageFolder(
                os.path.join(self.__data_dir, x),
                data_transforms2[x] if self.__model_type == "inception_v3" else data_transforms1[x]
            )
            for x in ["train", "test"]
        }
        self.__data_loaders = {
            x: torch.utils.data.DataLoader(
                image_dataset[x], batch_size=batch_size,
                shuffle=True
            )
            for x in ["train", "test"]
        }
        self.__dataset_sizes = {x: len(image_dataset[x]) for x in ["train", "test"]}
        self.__class_names = image_dataset["train"].classes
        self.__dataset_name = os.path.basename(self.__data_dir.rstrip(os.path.sep))

    def setDataDirectory(self, data_directory: str = "") -> None:
        if os.path.isdir(data_directory):
            self.__data_dir = data_directory
            return
        raise ValueError("expected a path to a directory")

    def setModelTypeAsMobileNetV2(self) -> None:
        self.__model_type = "mobilenet_v2"
        self.__training_params = mobilenet_v2_train_params()

    def setModelTypeAsResNet50(self) -> None:
        self.__model_type = "resnet50"
        self.__training_params = resnet50_train_params()

    def setModelTypeAsInceptionV3(self) -> None:
        self.__model_type = "inception_v3"
        self.__training_params = inception_v3_train_params()

    def setModelTypeAsDenseNet121(self) -> None:
        self.__model_type = "densenet121"
        self.__training_params = densenet121_train_params()

    def freezeAllLayers(self) -> None:
        self.__transfer_learning_mode = "freeze_all"

    def fineTuneAllLayers(self) -> None:
        self.__transfer_learning_mode = "fine_tune_all"

    def trainModel(
            self,
            num_experiments: int = 100,
            batch_size: int = 8,
            model_directory: str = None,
            transfer_from_model: str = None,
            verbose: bool = True
    ) -> None:


        self.__load_data(batch_size)

        if transfer_from_model:
            extension_check(transfer_from_model)
            self.__model_path = transfer_from_model

        self.__set_training_param()

        if not model_directory:
            model_directory = os.path.join(self.__data_dir, "models")

        if not os.path.exists(model_directory):
            os.mkdir(model_directory)

        with open(os.path.join(model_directory, f"{self.__dataset_name}_model_classes.json"), "w") as f:
            classes_dict = {}
            class_list = sorted(self.__class_names)
            for i in range(len(class_list)):
                classes_dict[str(i)] = class_list[i]
            json.dump(classes_dict, f)

        since = time.time()

        best_model_weights = copy.deepcopy(self.__model.state_dict())
        best_acc = 0.0
        prev_save_name, recent_save_name = "", ""

        print("=" * 50)
        print("Training with GPU") if self.__device == "cuda" else print(
            "Training with CPU. This might cause slower train.")
        print("=" * 50)

        for epoch in range(num_experiments):
            if verbose:
                print(f"Epoch {epoch + 1}/{num_experiments}", "-" * 10, sep="\n")

            
            for phase in ["train", "test"]:
                if phase == "train":
                    self.__model.train()
                else:
                    self.__model.eval()

                running_loss = 0.0
                running_corrects = 0

                
                for imgs, labels in tqdm(self.__data_loaders[phase]):
                    imgs = imgs.to(self.__device)
                    labels = labels.to(self.__device)

                    self.__optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        output = self.__model(imgs)
                        if self.__model_type == "inception_v3" and type(output) == InceptionOutputs:
                            output = output[0]
                        _, preds = torch.max(output, 1)
                        loss = self.__loss_fn(output, labels)

                        if phase == "train":
                            loss.backward()
                            self.__optimizer.step()
                    running_loss += loss.item() * imgs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                
                if phase == "train" and isinstance(self.__lr_scheduler, torch.optim.lr_scheduler.StepLR):
                    self.__lr_scheduler.step()

                epoch_loss = running_loss / self.__dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.__dataset_sizes[phase]

                if verbose:
                    print(f"{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    recent_save_name = self.__model_type + f"-{self.__dataset_name}-test_acc_{best_acc:.5f}_epoch-{epoch}.pt"
                    if prev_save_name:
                        os.remove(os.path.join(model_directory, prev_save_name))
                    best_model_weights = copy.deepcopy(self.__model.state_dict())
                    torch.save(
                        best_model_weights, os.path.join(model_directory, recent_save_name)
                    )
                    prev_save_name = recent_save_name

        time_elapsed = time.time() - since
        print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best test accuracy: {best_acc:.4f}")


class CustomImageClassification:

    def __init__(self) -> None:
        self.__model = None
        self.__model_type = ""
        self.__model_loaded = False
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__json_path = None
        self.__class_names = None
        self.__model_loaded = False

    def __load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        images = []
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if type(image_input) == str:
            if os.path.isfile(image_input):
                img = Image.open(image_input).convert("RGB")
                images.append(preprocess(img))
            else:
                raise ValueError(f"image path '{image_input}' is not found or a valid file")
        elif type(image_input) == np.ndarray:
            img = Image.fromarray(image_input).convert("RGB")
            images.append(preprocess(img))
        elif "PIL" in str(type(image_input)):
            img = image_input.convert("RGB")
            images.append(preprocess(img))
        else:
            raise ValueError(f"Invalid image input format")

        return torch.stack(images)

    def __load_classes(self):
        if self.__json_path:
            with open(self.__json_path, 'r') as f:
                self.__class_names = list(json.load(f).values())
        else:
            raise ValueError("Invalid json path. Set a valid json mapping path by calling the 'setJsonPath()' function")

    def setModelPath(self, path: str) -> None:

        if os.path.isfile(path):
            extension_check(path)
            self.__model_path = path
            self.__model_loaded = False
        else:
            raise ValueError(
                f"The path '{path}' isn't a valid file. Ensure you specify the path to a valid trained model file."
            )

    def setJsonPath(self, path: str) -> None:

        if os.path.isfile(path):
            self.__json_path = path
        else:
            raise ValueError(
                "parameter path should be a valid path to the json mapping file."
            )

    def setModelTypeAsMobileNetV2(self) -> None:
        self.__model_type = "mobilenet_v2"

    def setModelTypeAsResNet50(self) -> None:
        self.__model_type = "resnet50"

    def setModelTypeAsInceptionV3(self) -> None:
        self.__model_type = "inception_v3"

    def setModelTypeAsDenseNet121(self) -> None:
        self.__model_type = "densenet121"

    def useCPU(self):
        self.__device = "cpu"
        if self.__model_loaded:
            self.__model_loaded = False
            self.loadModel()

    def loadModel(self) -> None:

        if not self.__model_loaded:
            self.__load_classes()
            try:
                
                
                

                if self.__model_type == "resnet50":
                    self.__model = resnet50(pretrained=False)
                    in_features = self.__model.fc.in_features
                    self.__model.fc = nn.Linear(in_features, len(self.__class_names))
                elif self.__model_type == "mobilenet_v2":
                    self.__model = mobilenet_v2(pretrained=False)
                    in_features = self.__model.classifier[1].in_features
                    self.__model.classifier[1] = nn.Linear(in_features, len(self.__class_names))
                elif self.__model_type == "inception_v3":
                    self.__model = inception_v3(pretrained=False)
                    in_features = self.__model.fc.in_features
                    self.__model.fc = nn.Linear(in_features, len(self.__class_names))
                elif self.__model_type == "densenet121":
                    self.__model = densenet121(pretrained=False)
                    in_features = self.__model.classifier.in_features
                    self.__model.classifier = nn.Linear(in_features, len(self.__class_names))
                else:
                    raise RuntimeError("Unknown model type.\nEnsure the model type is properly set.")

                state_dict = torch.load(self.__model_path, map_location=self.__device)

                if self.__model_type == "densenet121":
                    
                    
                    pattern = re.compile(
                        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\."
                        "(?:weight|bias|running_mean|running_var))$"
                    )
                    for key in list(state_dict.keys()):
                        res = pattern.match(key)
                        if res:
                            new_key = res.group(1) + res.group(2)
                            state_dict[new_key] = state_dict[key]
                            del state_dict[key]

                self.__model.load_state_dict(state_dict)
                self.__model.to(self.__device).eval()
                self.__model_loaded = True

            except Exception as e:
                raise Exception("Weight loading failed.\nEnsure the model path is"
                                " set and the weight file is in the specified model path.")

    def classifyImage(self, image_input: Union[str, np.ndarray, Image.Image], result_count: int) -> Tuple[
        List[str], List[float]]:
        if not self.__model_loaded:
            raise RuntimeError(
                "Model not yet loaded. You need to call '.loadModel()' before performing image classification"
            )

        images = self.__load_image(image_input)
        images = images.to(self.__device)

        with torch.no_grad():
            output = self.__model(images)
        probabilities = torch.softmax(output, dim=1)
        topN_prob, topN_catid = torch.topk(probabilities, result_count)

        predictions = [
            [
                (self.__class_names[topN_catid[i][j]], topN_prob[i][j].item() * 100)
                for j in range(topN_prob.shape[1])
            ]
            for i in range(topN_prob.shape[0])
        ]

        labels_pred = []
        probabilities_pred = []

        for idx, pred in enumerate(predictions):
            for label, score in pred:
                labels_pred.append(label)
                probabilities_pred.append(round(score, 4))

        return labels_pred, probabilities_pred