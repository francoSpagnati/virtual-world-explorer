import torch
import math
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OwlVisionDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        """
        Inizializza il modello OWL-ViT per la detection visuale zero-shot.
        """
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[OWL-ViT] Caricamento modello {model_name} su {self.device}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        print("[OWL-ViT] Modello caricato con successo.")

    def detect_target_multiview(self, images: list[Image.Image], target_names: list[str], threshold: float = 0.2) -> tuple[float, float, bool]:
        """
        Analizza un batch di 8 immagini a 360° gradi e cerca il target primario.
        Restituisce vettori continui (global_dx, global_dy, visible).
        """
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
            print(f"[OWL-ViT] Nessun target primario trovato nelle {len(images)} direzioni")
            return 0.0, 0.0, False
            
        print(f"[OWL-ViT] Trovato '{target_names[0]}' con confidenza {best_score:.3f} nella telecamera {best_image_idx}")
        
        img_w, _ = images[best_image_idx].size
        center_x = img_w / 2.0
        box_center_x = (best_box[0] + best_box[2]) / 2.0
        
        # Calcolo dell'orientamento globale continuo basato sull'indice della telecamera
        # 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SO, 6=O, 7=NO
        # Angoli di orientamento mondo: N=270° (-90°), E=0°, S=90°, O=180°
        base_angle_deg = (best_image_idx * 45.0) - 90.0
        
        # Offset orizzontale all'interno del frame (FOV di 90 gradi totale, quindi max offset = ±45 gradi)
        offset_fraction = (box_center_x - center_x) / center_x
        local_offset_deg = offset_fraction * 45.0
        
        global_angle_deg = base_angle_deg + local_offset_deg
        global_angle_rad = math.radians(global_angle_deg)
        
        # Conversione in coordinate cartesiane normalizzate
        global_dx = math.cos(global_angle_rad)
        global_dy = math.sin(global_angle_rad)
            
        return global_dx, global_dy, True

if __name__ == "__main__":
    detector = OwlVisionDetector()
    test_images = [Image.new('RGB', (720, 720), color = 'white') for _ in range(8)]
    dx, dy, vis = detector.detect_target_multiview(test_images, ["chair", "table"])
    print(f"Risultato test 8-cam: dx={dx:.3f}, dy={dy:.3f}, visible={vis}")