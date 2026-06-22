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
- **Agente RL (Tuning):** `agent.py` utilizza Q-Learning. La Q-table è ora inizializzata a `0.0` (anziché ottimisticamente a 1.0) per evitare che il discount factor renda l'azione vincente "peggiore" delle azioni inesplorate. Il limite di esplorazione è stato ridotto a 30 mosse e le penalità per i passi a vuoto sono state decuplicate (`-0.1`) per forzare percorsi diretti e perfetti.
- **Spazio di Stato Ottimizzato (7-D):** L'agente percepisce il mondo tramite una tupla relativa: `(dx_target, dy_target, visible, danger_up, danger_down, danger_left, danger_right)`. Le coordinate assolute sono state rimosse.
- **Sensore Logico & Momentum:** `vision_radius` è impostata a 3 (limitata). Quando l'agente non vede la sedia (visible=0), usa una logica di "Momentum" (`last_action` in `main.py`) per muoversi in linee rette veloci come un Roomba, esplorando la mappa senza perdersi in wandering casuali. Appena `visible=1`, il momentum si disattiva e l'agente obbedisce ciecamente alla Q-table per colpire il target.
- **Motore 3D a Doppia Telecamera:** `render.py` adotta un'ingegnosa architettura dual-camera. Ad ogni step renderizza la scena due volte:
  1. *Per l'Utente:* Una telecamera fissa prospettica (`glFrustum`), visivamente stabile e piacevole.
  2. *Per OWL-ViT:* Una telecamera egocentrica ortografica (`glOrtho`) limitata a un raggio visivo di 3 blocchi (uguale al sensore logico). Questo elimina la distorsione prospettica.
- **Visione Artificiale (OWL-ViT Sincrona):** Il modulo `owl_vision.py` deduce la direzione. A differenza di prima, l'inferenza è ora strettamente **sincrona** rispetto allo step del motore: il loop principale di `main.py` "congela" l'agente finché OWL-ViT non restituisce il risultato, azzerando la latenza ed evitando che l'agente superi i target mentre aspetta il processamento asincrono. La finestra OpenGL è mantenuta responsiva tramite `poll()`.
- **Esecuzione:** `main.py` orchestra prima il training dell'agente e successivamente lancia una Demo 3D visiva. Al lancio, la simulazione attende il caricamento dei pesi OWL-ViT prima di permettere il primo passo.

## 4. Cosa Manca / Prossimi Passi
Per completare la visione finale e raggiungere l'obiettivo del progetto, mancano le seguenti espansioni:
1. **Generalizzazione Spaziale:**
   La griglia attuale è fissa a 7x7. Grazie alla rimozione delle coordinate assolute, il passaggio ad arene più grandi dovrebbe essere trasparente per l'agente, ma richiederà calibrazione sul modulo di visual-detection che dovrà scalare le bounding box 2D ottenute ai vettori direzionali 3D.
