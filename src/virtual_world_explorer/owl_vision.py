import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OwlVisionDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        """
        Inizializza il modello OWL-ViT per la detection visuale.
        """
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[OWL-ViT] Caricamento modello {model_name} su {self.device}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        print("[OWL-ViT] Modello caricato con successo.")

    def detect_target(self, image: Image.Image, target_name: str, threshold: float = 0.02) -> tuple[int, int, bool]:
        """
        Analizza un'immagine e cerca il target specificato.
        Restituisce (dx, dy, visible) normalizzati rispetto alla posizione dell'agente.
        """
        # Formattiamo il testo nel prompt richiesto da OWL-ViT (es. "a photo of a chair")
        text_queries = [f"an object that is a {target_name}"]
        
        # Processiamo l'input
        inputs = self.processor(text=[text_queries], images=image, return_tensors="pt").to(self.device)
        
        # Inferenza
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Ridimensioniamo le bounding box sulla dimensione reale dell'immagine
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        boxes = results["boxes"]
        scores = results["scores"]
        
        if len(scores) == 0:
            # Nessun oggetto trovato oltre la soglia
            print("[OWL-ViT] Nessun oggetto trovato")
            return 0, 0, False
            
        # Prendiamo la bounding box con il punteggio di confidenza più alto
        best_idx = torch.argmax(scores).item()
        best_box = boxes[best_idx].tolist() # [xmin, ymin, xmax, ymax]
        best_score = scores[best_idx].item()
        
        # Calcoliamo il centro della bounding box
        box_center_x = (best_box[0] + best_box[2]) / 2.0
        box_center_y = (best_box[1] + best_box[3]) / 2.0
        
        print(f"[OWL-ViT] Trovato '{target_name}' con confidenza {best_score:.3f} al centro ({box_center_x:.1f}, {box_center_y:.1f})")
        
        img_w, img_h = image.size
        center_x, center_y = img_w / 2.0, img_h / 2.0
        
        # Definiamo dei margini stretti (es. 5%) attorno al centro per la "zona neutra"
        margin_x = img_w * 0.05
        margin_y = img_h * 0.05
        
        dx = 0
        dy = 0
        
        # X asse: -1 (sinistra), 1 (destra)
        if box_center_x < center_x - margin_x:
            dx = -1
        elif box_center_x > center_x + margin_x:
            dx = 1
            
        # Y asse: L'immagine OpenGL ha Y=0 in alto.
        # Se la sedia è "più in alto" sullo schermo, significa che è "più lontana" lungo l'asse Y della griglia.
        # Una Y maggiore sulla griglia significa "più in basso".
        if box_center_y < center_y - margin_y:
            dy = 1 # Target più lontano -> Y griglia maggiore
        elif box_center_y > center_y + margin_y:
            dy = -1  # Target più vicino -> Y griglia minore
            
        return dx, dy, True

if __name__ == "__main__":
    # Piccolo script di test 
    detector = OwlVisionDetector()
    
    # finta immagine per testare la pipeline
    test_img = Image.new('RGB', (720, 720), color = 'white')
    dx, dy, vis = detector.detect_target(test_img, "chair")
    print(f"Risultato test finto: dx={dx}, dy={dy}, visible={vis}")
