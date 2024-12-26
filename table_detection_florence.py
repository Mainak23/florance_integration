from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import supervision as sv

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(task_prompt, image, max_new_tokens=128):
    """Run object detection on the given image."""
    prompt = task_prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=max_new_tokens,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def plot_and_save_bbox(image, data, output_dir, filename):
    """Plot bounding boxes on the image and save the result."""
    fig, ax = plt.subplots()
    ax.imshow(image)
    labels = []
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        labels.append(f"{label}: ({x1}, {y1}, {x2}, {y2})")
    ax.axis('off')

    # Save the annotated image
    annotated_image_path = os.path.join(output_dir, f"{filename}_annotated.png")
    plt.savefig(annotated_image_path, bbox_inches='tight')
    plt.close(fig)

    # Save the labels to a text file
    labels_path = os.path.join(output_dir, f"{filename}_labels.txt")
    with open(labels_path, 'w') as f:
        f.write("\n".join(labels))
    print(f"Saved annotated image to {annotated_image_path}")
    print(f"Saved labels to {labels_path}")

def main():
    test_folder = os.path.join(os.getcwd(),'output_images/test_folder')
    output_dir = os.path.join(os.getcwd(),'output_images/anoted_folder')
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the test folder
    for file_name in os.listdir(test_folder):
        file_path = os.path.join(test_folder, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {file_name}...")
            image = Image.open(file_path).convert("RGB")
            parsed_answer = run_example("<OCR_WITH_REGION>", image=image)
            print(parsed_answer)
            bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, parsed_answer, resolution_wh=image.size)
            annotated = bounding_box_annotator.annotate(image, detections=detections)
            annotated = label_annotator.annotate(annotated, detections=detections)
            sv.plot_image(annotated)
    print("Processing complete!")