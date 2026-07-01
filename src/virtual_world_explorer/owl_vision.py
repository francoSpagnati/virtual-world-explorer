import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OwlVisionDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32"):

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[OWL-ViT] Caricamento modello {model_name} su {self.device}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        print("[OWL-ViT] Modello caricato con successo.")

    def detect_target_multiview(self, images: list[Image.Image], target_names: list[str], threshold: float = 0.03) -> tuple[int, int, bool]:

        text_queries = [f"an object that is a {name}" for name in target_names]
        texts = [text_queries] * len(images)
        
        inputs = self.processor(text=texts, images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([img.size[::-1] for img in images])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        
        best_score = -1.0
        best_box = None
        best_image_idx = -1
        
        for img_idx, res in enumerate(results):
            boxes = res["boxes"]
            scores = res["scores"]
            labels = res["labels"]
            
            for i, score in enumerate(scores):
                if labels[i] == 0 and score.item() > best_score:
                    best_score = score.item()
                    best_box = boxes[i].tolist()
                    best_image_idx = img_idx
                    
        if best_image_idx == -1:
            print("[OWL-ViT] Nessun target primario trovato nelle 4 direzioni")
            return 0, 0, False
            
        print(f"[OWL-ViT] Trovato '{target_names[0]}' con confidenza {best_score:.3f} nella telecamera {best_image_idx}")
        
        img_w, img_h = images[best_image_idx].size
        center_x, center_y = img_w / 2.0, img_h / 2.0
        
        box_center_x = (best_box[0] + best_box[2]) / 2.0
        margin_x = img_w * 0.05
        
        local_dx = 0
        if box_center_x < center_x - margin_x:
            local_dx = -1
        elif box_center_x > center_x + margin_x:
            local_dx = 1
            
        local_dy = 1
        
        if best_image_idx == 0:
            global_dx = local_dx
            global_dy = -local_dy
        elif best_image_idx == 1:
            global_dx = -local_dx
            global_dy = local_dy
        elif best_image_idx == 2:
            global_dx = -local_dy
            global_dy = -local_dx
        else:
            global_dx = local_dy
            global_dy = local_dx
            
        return global_dx, global_dy, True

if __name__ == "__main__":

    detector = OwlVisionDetector()
    
    test_images = [Image.new('RGB', (720, 720), color = 'white') for _ in range(4)]
    dx, dy, vis = detector.detect_target_multiview(test_images, ["chair", "table"])
    print(f"Risultato test finto: dx={dx}, dy={dy}, visible={vis}")
