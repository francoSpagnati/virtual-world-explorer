import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

model_name = "google/owlvit-base-patch32"
processor = OwlViTProcessor.from_pretrained(model_name)
model = OwlViTForObjectDetection.from_pretrained(model_name)

# Create 4 fake images
images = [Image.new('RGB', (200, 200), color='white') for _ in range(4)]
texts = [["a chair", "a table", "a lamp"]] * 4

inputs = processor(text=texts, images=images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([img.size[::-1] for img in images])
results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)

print(f"Results length: {len(results)}")
for i, res in enumerate(results):
    print(f"Image {i}: {len(res['scores'])} boxes")
