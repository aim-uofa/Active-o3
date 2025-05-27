import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import numpy as np
import re,json
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
model_path = "YOUR_MODEL_PATH"  # Replace with your model path

VIS_OUTPUT_DIR = model_path + "/visualization_vstar"

VISUALIZE = True

if not os.path.exists(VIS_OUTPUT_DIR):
    os.makedirs(VIS_OUTPUT_DIR)
if not os.path.exists(os.path.join(VIS_OUTPUT_DIR, 'json')):
    os.makedirs(os.path.join(VIS_OUTPUT_DIR, 'json'))
if not os.path.exists(os.path.join(VIS_OUTPUT_DIR, 'cropped')):
    os.makedirs(os.path.join(VIS_OUTPUT_DIR, 'cropped'))
if not os.path.exists(os.path.join(VIS_OUTPUT_DIR, 'visualizations')):
    os.makedirs(os.path.join(VIS_OUTPUT_DIR, 'visualizations'))
    
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)



def fix_bbox_dict_list(data):
    """
    fix the bbox_2d format in the list of dictionaries
    """
    fixed_data = []

    for item in data:
        new_item = {}
        for key, value in item.items():
            if key == "bbox_2d" and isinstance(value, list):
                # 正确格式，直接保留
                new_item[key] = value
            else:
                # 匹配错误格式 'bbox_2d [x1, y1, x2, y2]'
                match = re.match(r'^bbox_2d\s*\[(.*?)\]$', key)
                if match:
                    try:
                        bbox_values = [int(v.strip()) for v in match.group(1).split(',')]
                        new_item['bbox_2d'] = bbox_values
                        # 原 value 是 label
                        new_item['label'] = value
                    except ValueError:
                        print(f"Warning: 无法解析 bbox 值: {key}")
                else:
                    # 普通字段直接保留
                    new_item[key] = value
        fixed_data.append(new_item)

    return fixed_data


  
    
  
from qwen_omni_utils import process_mm_info, process_vision_info
def soft_load_json(json_str: str) -> list:
  json_pattern = r'{[^}]+}'  
  json_match = re.findall(json_pattern, json_str)
  return_list = []
  for json_str in json_match:
      try:
          data = json.loads(json_str)
          return_list.append(data)
      except json.JSONDecodeError:
          print(f"Error decoding JSON: {json_str}")
  return return_list
def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    # print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    # img.show()



def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    #img.show()
    return
  points = data['points']
  description = data['phrase']

  font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  
  #img.show()
  
def convert_text_to_json(text):
    parsed_text = parse_json(text)
    try:
      json_output = ast.literal_eval(parsed_text)
    except Exception as e:
      end_idx = parsed_text.rfind('"}') + len('"}')
      truncated_text = parsed_text[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)
    return json_output

def plot_points_json(im, json_output, input_width, input_height):
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
      'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
      'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
      'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors
    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    if isinstance(json_output, str):
      json_output = convert_text_to_json(json_output)
    for i, sample in enumerate(json_output):
        point = sample['point_2d']
        description = sample['label']
        color = colors[i % len(colors)]
        abs_x1 = int(point[0])/input_width * width
        abs_y1 = int(point[1])/input_height * height
        radius = 2
        draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
        draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)


    #img.show()




# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
        if line == "<answer>":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("</answer>")[0]  # Remove everything after the closing "```"
            break 
    if "```json" in json_output:
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
    return json_output


def parse_answer(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "<answer>":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("</answer>")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024, do_sample=False):
  image = Image.open(img_url)
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1300, do_sample=do_sample, temperature=0.1, top_p=0.95, top_k=50, num_return_sequences=1)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  # print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text[0], input_height, input_width


def do_grounding_list(image_path_list, object, system_prompt="You are a helpful assistant", max_new_tokens=1024, do_sample=False, question=None):
  # try:
  #   image = Image.open(image_path)
  # except Exception as e:
  #   image = image_path
  messages_list = [ ]
  for image_path in image_path_list:
    messages = [
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": [
          # {
          #   "type": "text",
          #   "text": prompt
          # }
          {
            "image": image_path,
             "min_pixels": 900*28*28
          },
          {
            "type": "text",
            "text": f'Please find all {object} in the image and return the bounding box coordinates in JSON format. ' if question is None else question
          }

        ]
      }
    ]
    messages_list.append(messages)
  # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
  # import pdb
  # pdb.set_trace()
  image_input = process_vision_info(messages_list)[0]
  # print (process_vision_info(messages_list))
  inputs = processor(text=text_list, images=image_input, padding=True, return_tensors="pt",padding_side="left").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1300, do_sample=do_sample, temperature=0.1, top_p=0.95, top_k=50, num_return_sequences=1)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  results = []
  for i, output_text in enumerate(output_texts):
        input_height = inputs['image_grid_thw'][i][1] * 14
        input_width = inputs['image_grid_thw'][i][2] * 14
        # print(f"output[{i}]:\n", output_text)
        results.append({
            "image": image_path_list[i],
            "output": output_text,
            "input_height": input_height,
            "input_width": input_width
        })
  return results

def do_grounding(image_path, object, system_prompt="You are a helpful assistant", max_new_tokens=1024, do_sample=False):
  try:
    image = Image.open(image_path)
  except Exception as e:
    image = image_path
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        # {
        #   "type": "text",
        #   "text": prompt
        # }
        {
          "image": image_path, 
          "min_pixels": 900*28*28
        },
        {
          "type": "text",
          "text": f'Please find the {object} in the image and return the bounding box coordinates in JSON format. '
        }

      ]
    }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1300, do_sample=do_sample, temperature=0.1, top_p=0.95, top_k=50, num_return_sequences=1)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  # print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text[0], input_height, input_width

def visualize_boxes_on_original_image(image_path, boxes, object_name):
  """Visualize detected boxes on the original image without any resizing"""
  try:
    # Try to find the original image path if we're using a cached image
    original_image_path = image_path
    if '.cache/' in image_path:
      # If we're using a cached resized image, find the original path
      orig_filename = os.path.basename(image_path)
      original_image_path = orig_filename
    
    print(f"Visualizing with original image: {original_image_path}")
    
    # Load the original image directly
    try:
      vis_image = Image.open(original_image_path)
      print(f"Original image size for visualization: {vis_image.size}")
    except Exception as e:
      print(f"Error opening original file {original_image_path}: {e}, trying {image_path}")
      vis_image = Image.open(image_path)
      print(f"Fallback image size: {vis_image.size}")
    
    draw = ImageDraw.Draw(vis_image)
    colors = [
      'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown',
      'cyan', 'magenta', 'lime', 'violet', 'gold', 
    ] + additional_colors
    
    # Try to load font, use default if not available
    try:
      font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
      font = ImageFont.load_default()
    
    # Draw each box directly using coordinates
    for i, box in enumerate(boxes):
      x1, y1, x2, y2 = box
      color = colors[i % len(colors)]
      
      # Draw rectangle
      draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=4)
      
      # Add label text with background
      text = f"{object_name} {i+1}"
      draw.rectangle(((x1, y1-30), (x1 + 120, y1)), fill=color)
      draw.text((x1 + 8, y1-25), text, fill='white', font=font)
    
    # Save visualization
    vis_dir = os.path.join(VIS_OUTPUT_DIR, 'visualizations')
    if not os.path.exists(vis_dir):
      os.makedirs(vis_dir)
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    vis_path = os.path.join(vis_dir, f"{base_name}_{object_name}_original.jpg")
    vis_image.save(vis_path)
    print(f"Original image visualization saved to {vis_path}")
    return vis_path
  except Exception as e:
    print(f"Error visualizing image: {e}")
    return None

def inference_and_plot(image_path, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024, save_path=None, max_size=None, do_sample=False, object=None, category_id=None, image_id=None, question=None):
  """Process an image, detect objects and return results in COCO format or return answer in JSON format."""
  json_path = os.path.join(VIS_OUTPUT_DIR, 'json', os.path.basename(image_path).replace(".jpg", "").split('/')[-1] + f"-{object}.json")
  
  # Store the original image path before any processing
  original_image_path = image_path
  
    
  image = Image.open(image_path)
  real_ori_img_width, real_ori_img_height = image.size
  
  min_size = 1024
  
  # Create cache directory if it doesn't exist
  cache_dir = '.cache'
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    
  cache_image_path = f'{cache_dir}/{os.path.basename(image_path)}'
  
  # Store resized image path for model input
  resized_image_path = image_path
  
  if image.size[0] < min_size or image.size[1] < min_size:
    # resize image to a larger size
    scale = min_size / min(image.size)
    new_size = (int(image.size[0]*scale), int(image.size[1]*scale))
    image = image.resize(new_size, Image.LANCZOS)
    image.save(cache_image_path)
    # print("resize image to", image.size)
    resized_image_path = cache_image_path  # Use the cached path for further processing
  
  if max_size is not None:
    if image.size[0] > max_size or image.size[1] > max_size:
      # resize image to a smaller size
      scale = max_size / max(image.size)
      new_size = (int(image.size[0]*scale), int(image.size[1]*scale))
      image = image.resize(new_size, Image.LANCZOS)
      image.save(cache_image_path)
      # print("resize image to", image.size)
      resized_image_path = cache_image_path  # Use the cached path for further processing
      
  # From now on, use resized_image_path for model input
  response, input_height, input_width = inference(resized_image_path, prompt, do_sample=do_sample)
  # print("input_height", input_height)
  # print("input_width", input_width)
  # print("response", response)
  print(response)
  json_result = soft_load_json(response)
  
  if not json_result:
    print("No regions detected in the image")
    return response
    
  bbox_list = [item['bbox_2d'] for item in json_result]
  bbox_np_to_crop = np.array(bbox_list).astype(np.float32)
  
  # convert to relative coordinate
  bbox_np_to_crop[:, 0] = bbox_np_to_crop[:, 0] / float(input_width)
  bbox_np_to_crop[:, 1] = bbox_np_to_crop[:, 1] / float(input_height)
  bbox_np_to_crop[:, 2] = bbox_np_to_crop[:, 2] / float(input_width)
  bbox_np_to_crop[:, 3] = bbox_np_to_crop[:, 3] / float(input_height)
  
  # Get original image dimensions
  ori_img_width, ori_img_height = image.size
  print(ori_img_width, ori_img_height)

  bbox_np_to_ori = np.zeros_like(bbox_np_to_crop)
  bbox_np_to_ori[:, 0] = bbox_np_to_crop[:, 0] * float(ori_img_width)
  bbox_np_to_ori[:, 1] = bbox_np_to_crop[:, 1] * float(ori_img_height)
  bbox_np_to_ori[:, 2] = bbox_np_to_crop[:, 2] * float(ori_img_width)
  bbox_np_to_ori[:, 3] = bbox_np_to_crop[:, 3] * float(ori_img_height)

  # write the json file for middle box results
  output_json = {
    "image_id": image_id,
    "question": question,
    "file_name": os.path.basename(image_path),
    "image_width": ori_img_width,
    "image_height": ori_img_height,
    "pred_json": json_result,
    "pred_bbox": bbox_np_to_ori.tolist(),
    "response": response,
  }
  with open(json_path.replace('.json', f'_mid.json'), "w") as f:
    json.dump(output_json, f, indent=2)

  
  def crop_image_by_bbox(image, bbox_list):
    cropped_images = []
    print("Original image size:", ori_img_width, ori_img_height)
    for i, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = bbox
        # Convert to absolute coordinates
        abs_x1 = int(x1 * ori_img_width)
        abs_y1 = int(y1 * ori_img_height)
        abs_x2 = int(x2 * ori_img_width)
        abs_y2 = int(y2 * ori_img_height)
        
        # Debug print
        # print(f"Box {i}: Relative coords: {x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}")
        # print(f"Box {i}: Absolute coords: {abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}")
        
        # Ensure box coordinates are valid (x1 < x2, y1 < y2)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        # Ensure coordinates are within image bounds
        abs_x1 = max(0, min(abs_x1, ori_img_width - 1))
        abs_y1 = max(0, min(abs_y1, ori_img_height - 1))
        abs_x2 = max(0, min(abs_x2, ori_img_width))
        abs_y2 = max(0, min(abs_y2, ori_img_height))
        
        # Check if box dimensions are reasonable
        box_width = abs_x2 - abs_x1
        box_height = abs_y2 - abs_y1
        if box_width <= 0 or box_height <= 0:
            # print(f"Warning: Invalid box dimensions: {box_width}x{box_height}, skipping")
            continue
        
        # print(f"Final crop box {i}: {abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}, Size: {box_width}x{box_height}")
        
        # Crop the image
        try:
            cropped_image = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
            crop_path = os.path.join(VIS_OUTPUT_DIR, 'cropped', f"{os.path.basename(image_path).replace('.jpg', '')}_crop_{i}.jpg")
            if VISUALIZE:
              cropped_image.save(crop_path)
            cropped_images.append(crop_path)
        except Exception as e:
            print(f"Error cropping image: {e}")
    
    return cropped_images

  crop_images = crop_image_by_bbox(image, bbox_np_to_crop)
  
  if not crop_images:
    print("No valid crop regions found")
    return []
    
  # Run object detection on each crop
  results = do_grounding_list(crop_images, object,question=question)
  input_h_list = [float(result['input_height']) for result in results]
  input_w_list = [float(result['input_width']) for result in results]
  extract_crop_bbox_list = []
  txt_path = json_path.replace('.json', f'_mid.txt')
  print("txt_path", txt_path)
  with open(txt_path, "w") as f:
    for i, result in enumerate(results):
      output_text = result['output']
      print(f"output[{i}]:\n", output_text)
      f.write(f"output[{i}]:{output_text}\n")
      json_result = soft_load_json(output_text)
      json_result = fix_bbox_dict_list(json_result)

      bbox_list = [item['bbox_2d'] for item in json_result]
      
      bbox_np = np.array(bbox_list).astype(np.float32)
      # convert to relative coordinate
      if bbox_np.shape[0] != 0:
        bbox_np[:, 0] = bbox_np[:, 0] / float(input_w_list[i])
        bbox_np[:, 1] = bbox_np[:, 1] / float(input_h_list[i])
        bbox_np[:, 2] = bbox_np[:, 2] / float(input_w_list[i])
        bbox_np[:, 3] = bbox_np[:, 3] / float(input_h_list[i])
      extract_crop_bbox_list.append(bbox_np)
  
  # Transform coordinates back to original image space
  final_bbox_list_relative_to_ori_img = []
  # Iterate through the bounding boxes found in each crop and the corresponding crop box definition
  assert len(extract_crop_bbox_list) == len(bbox_np_to_crop)
  for ground_bboxes_in_crop_img, crop_box in zip(extract_crop_bbox_list, bbox_np_to_crop):
      # crop_box: [X1, Y1, X2, Y2] relative to original image
      # ground_bboxes_in_crop_img: np.array([[x1_c, y1_c, x2_c, y2_c], ...]) relative to the cropped image
      X1, Y1, X2, Y2 = crop_box
      W = X2 - X1 # Width of the crop box in original relative coordinates
      H = Y2 - Y1 # Height of the crop box in original relative coordinates

      # Avoid division by zero if crop box has zero width or height
      if W <= 0 or H <= 0:
          print(f"Warning: Skipping crop box with zero dimension: {crop_box}")
          continue

      if ground_bboxes_in_crop_img.shape[0] > 0:
          # Extract columns (shape: (K,))
          x1_c = ground_bboxes_in_crop_img[:, 0]
          y1_c = ground_bboxes_in_crop_img[:, 1]
          x2_c = ground_bboxes_in_crop_img[:, 2]
          y2_c = ground_bboxes_in_crop_img[:, 3]

          # Apply transformation vectorized (results are shape: (K,))
          x1_orig = X1 + x1_c * W
          y1_orig = Y1 + y1_c * H
          x2_orig = X1 + x2_c * W
          y2_orig = Y1 + y2_c * H

          # Stack them back into (K, 4) format
          transformed_bboxes_np = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)

          # Add the transformed boxes from this crop to the final list
          # Convert back to list of lists if needed for consistency, or keep as list of arrays
          final_bbox_list_relative_to_ori_img.extend(transformed_bboxes_np.tolist())
  
  # Convert results to absolute coordinates
  abs_boxes = []
  
  # Calculate scale factors between resized and original image
  scale_x = real_ori_img_width / ori_img_width if ori_img_width > 0 else 1
  scale_y = real_ori_img_height / ori_img_height if ori_img_height > 0 else 1
  
  for bbox in final_bbox_list_relative_to_ori_img:
      x1, y1, x2, y2 = bbox
      # Convert to absolute coordinates in the original image space
      abs_x1 = int(x1 * ori_img_width * scale_x)
      abs_y1 = int(y1 * ori_img_height * scale_y)
      abs_x2 = int(x2 * ori_img_width * scale_x)
      abs_y2 = int(y2 * ori_img_height * scale_y)
      abs_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
  
  # Save raw bounding box results
  if not os.path.exists(os.path.dirname(json_path)):
    os.makedirs(os.path.dirname(json_path))
    
  with open(json_path, "w") as f:
    json.dump(abs_boxes, f)
  
  
  # Convert to COCO format and return
  return convert_to_coco_format(abs_boxes, image_id, category_id)

def convert_to_coco_format(boxes, image_id, category_id):
    """Convert [x1, y1, x2, y2] format boxes to COCO format [x, y, width, height]"""
    coco_results = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        coco_results.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, width, height],
            "score": 1.0  # Default score for detected objects
        })
    return coco_results



QUESTION_TEMPLATE = (
    "Find up to three different regions in the image that are most likely to help answer the question: '{question}'. "
    "Even if the answer is not directly visible, infer which regions might contain relevant visual cues. "
    "Each region should include meaningful context that contributes to answering the question—such as objects, text, or scene elements. "
    "The selected regions should be as distinct as possible, with minimal or no overlap. "
    "Return the coordinates in JSON format as: "
    "{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"region relevant to the question\"}. "
    "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer>. "
    "i.e., <think> reasoning process here </think> "
    "<answer> [JSON list of regions] </answer>"
)



root = 'example_data/'
i = 0
start = 0
end = 20

for file in os.listdir(root):
  
  if i < start:
    i += 1
    continue
  print(file)
  if file.endswith('.jpg'):
    image_path = os.path.join(root, file)
    json_path = os.path.join(root, file.replace('.jpg', '.json'))
    print(image_path)
    print(json_path)
    with open(json_path, "r") as f:
      ann = json.load(f)
    question = ann['question']
    prompt = question
    formatted_prompt = QUESTION_TEMPLATE.replace("{question}", prompt)
    results = inference_and_plot(
        image_path,
        prompt,
        object='animal',
        question=prompt,
    )
    output_txt_path = VIS_OUTPUT_DIR + '/json/' + file.replace('.jpg', '.txt')
    with open(output_txt_path, "w") as f:
      f.write(f"{results}\n")
    results = inference_and_plot(
        image_path,
        formatted_prompt,
        object='animal',
        question=prompt,
    )
  
  i += 1
  if i > end:
    break




