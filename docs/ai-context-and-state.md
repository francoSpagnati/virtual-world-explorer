# AI Context & Project State — Virtual World Explorer

Questo file funge da "mappa" e "contesto" per qualsiasi Intelligenza Artificiale che deve comprendere, modificare o espandere questo progetto. Leggilo attentamente prima di proporre modifiche.

## 1. Dominio e Contesto
Siamo in un progetto Python che simula un ambiente 3D ("Virtual World Explorer") in cui un agente controllato da algoritmi di Reinforcement Learning (Q-learning tabellare) deve imparare a navigare una griglia evitando ostacoli (es. tavolo, lampada) per raggiungere un bersaglio (sedia).
L'aspetto visivo è interamente delegato a un renderer **OpenGL 3D (immediate mode)**.
L'obiettivo a lungo termine del progetto è integrare modelli di Vision-Language (come **OWL-ViT**) per dare all'agente input semantici estratti dai frame renderizzati (es. per capire dov'è la sedia "guardando" lo schermo).

## 2. Il Vincolo Principale (GFX-Only)
Esiste un limite tecnologico ferreo imposto all'architettura:
> Nel seguito, qualsiasi grafica di basso livello come OpenGL o WebGL viene chiamata **GFX**.
> Qualsiasi framework di alto livello, come Unity, Unreal Engine o librerie grafiche JavaScript, sono **vietati**.

Il rendering *deve* avvenire manipolando direttamente lo stato OpenGL tramite `PyOpenGL` e `GLFW`. Attualmente viene usata la "immediate mode" (`glBegin`/`glEnd`). Nonostante i modelli 3D vengano parsati tramite la libreria ausiliaria `trimesh` (esplicitamente autorizzata per snellire il codice matematico), il loop di rendering effettivo su schermo è e deve restare esclusivamente GFX.

## 3. Stato Attuale del Progetto (Configurazione)
Il progetto è in uno stato avanzato e pienamente funzionante, configurato come segue:
- **Agente RL:** `agent.py` utilizza Q-Learning con inizializzazione ottimistica (`Q=1.0`) e decrescita rigida per forzare l'esplorazione nei primi episodi.
- **Spazio di Stato Ottimizzato (7-D):** L'agente percepisce il mondo tramite una tupla relativa: `(dx_target, dy_target, visible, danger_up, danger_down, danger_left, danger_right)`. Le coordinate assolute sono state rimosse.
- **Sensore Logico (Detector):** Durante il training, la `vision_radius` è impostata a 10 (visione infinita) per permettere all'agente di imparare il pathing ottimale da ogni cella in 5000 episodi senza perdersi in wandering casuale.
- **Motore 3D a Doppia Telecamera:** `render.py` adotta un'ingegnosa architettura dual-camera. Ad ogni step renderizza la scena due volte:
  1. *Per l'Utente:* Una telecamera fissa prospettica (`glFrustum`), visivamente stabile e piacevole.
  2. *Per OWL-ViT:* Una telecamera egocentrica ortografica (`glOrtho`) nascosta nel back-buffer, che segue l'agente. Questo elimina la distorsione prospettica, assicurando che le coordinate 2D dell'immagine corrispondano perfettamente ai vettori direzionali 3D.
- **Visione Artificiale (OWL-ViT):** Il modulo `owl_vision.py` utilizza il modello foundation OWL-ViT per dedurre la direzione del target (`dx, dy`) puramente guardando il rendering ortografico. Inferisce in un thread in background. La soglia di confidenza è bassa (0.005) per gestire modelli renderizzati a basse risoluzioni senza falsi negativi.
- **Esecuzione:** `main.py` orchestra prima il training dell'agente e successivamente lancia una Demo 3D visiva, integrando OWL-ViT tramite una coda thread-safe.

## 4. Cosa Manca / Prossimi Passi
Per completare la visione finale e raggiungere l'obiettivo del progetto, mancano le seguenti espansioni:
1. **Generalizzazione Spaziale:**
   La griglia attuale è fissa a 7x7. Grazie alla rimozione delle coordinate assolute, il passaggio ad arene più grandi dovrebbe essere trasparente per l'agente, ma richiederà calibrazione sul modulo di visual-detection che dovrà scalare le bounding box 2D ottenute ai vettori direzionali 3D.
