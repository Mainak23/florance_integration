import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load model and processor
#model_id = "ucsahin/Florence-2-large-TableDetection"
model_id = "microsoft/Florence-2-large"
#model_id = "microsoft/Florence-2-large"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda")
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
    print(generated_text)
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

def calculate_area(box):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return width * height


def main():
    test_folder = os.path.join(os.getcwd(),'output_images/test_folder')
    output_dir = os.path.join(os.getcwd(),'output_images/anoted_folder')
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the test folder
    for file_name in os.listdir(test_folder):
        file_path = os.path.join(test_folder, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg','.pdf')):
            print(f"Processing {file_name}...")
            image = Image.open(file_path).convert("RGB")
            parsed_answer = run_example("<OD>", image=image)
            bboxes=parsed_answer['<OD>']['bboxes']
            areas = [calculate_area(box) for box in bboxes]
            max_index = areas.index(max(areas))
            max_bbox = bboxes[max_index]
            task_prompt = "<OCR_WITH_REGION>"
            answer = run_example(task_prompt=task_prompt, image=image)
            print(answer)
            plot_and_save_bbox(image, parsed_answer["<OD>"], output_dir, os.path.splitext(file_name)[0])
    print("Processing complete!")

if __name__ == "__main__":
    main()

