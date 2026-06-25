# AI Context & Project State — Virtual World Explorer

Questo file funge da "mappa" e "contesto" per qualsiasi Intelligenza Artificiale che deve comprendere, modificare o espandere questo progetto. Leggilo attentamente prima di proporre modifiche.

## 1. Dominio e Contesto
Siamo in un progetto Python che simula un ambiente 3D ("Virtual World Explorer") in cui un agente controllato da algoritmi di Deep Reinforcement Learning (Deep Q-Network, DQN) in PyTorch deve imparare a navigare una griglia evitando ostacoli (es. tavolo, lampada) per raggiungere un bersaglio (sedia).
L'aspetto visivo è interamente delegato a un renderer **OpenGL 3D (immediate mode)**.
L'obiettivo a lungo termine del progetto è integrare modelli di Vision-Language (come **OWL-ViT**) per dare all'agente input semantici estratti dai frame renderizzati (es. per capire dov'è la sedia "guardando" lo schermo).

## 2. Il Vincolo Principale (GFX-Only)
Esiste un limite tecnologico ferreo imposto all'architettura:
> Nel seguito, qualsiasi grafica di basso livello come OpenGL o WebGL viene chiamata **GFX**.
> Qualsiasi framework di alto livello, come Unity, Unreal Engine o librerie grafiche JavaScript, sono **vietati**.

Il rendering *deve* avvenire manipolando direttamente lo stato OpenGL tramite `PyOpenGL` e `GLFW`. Attualmente viene usata la "immediate mode" (`glBegin`/`glEnd`). Nonostante i modelli 3D vengano parsati tramite la libreria ausiliaria `trimesh` (esplicitamente autorizzata per snellire il codice matematico), il loop di rendering effettivo su schermo è e deve restare esclusivamente GFX.

## 3. Stato Attuale del Progetto (Configurazione)
Il progetto è in uno stato avanzato e pienamente funzionante, configurato come segue:
- **Agente RL (Tuning):** `agent.py` utilizza Deep Q-Learning (DQN) con PyTorch. L'addestramento è stato esteso a 30000 episodi per permettere all'agente di padroneggiare mappe molto congestionate (ora con 6 ostacoli invece di 2). I Q-values sono predetti da una rete neurale, eliminando la necessità di una Q-table fissa e garantendo maggiore generalizzazione continua. Il limite di mosse impedisce percorsi infiniti e le penalità forzano percorsi ottimali.
- **Spazio di Stato Ottimizzato (7-D):** L'agente percepisce il mondo tramite una tupla relativa continua: `(dx_target, dy_target, visible, danger_up, danger_down, danger_left, danger_right)`. Le coordinate assolute sono state rimosse. I sensori di prossimità (`danger_*`) si sono dimostrati formidabili.
- **Sensore Logico & Momentum:** `vision_radius` è impostata a 3.5. Quando l'agente non vede la sedia (visible=0), usa una logica di "Momentum" (`last_action` in `main.py`) per muoversi in linee rette. Appena `visible=1`, il momentum si disattiva e l'agente obbedisce ciecamente alla policy di navigazione basata su rete neurale.
- **Motore 3D a Doppia Telecamera e Spazio Continuo:** `render.py` adotta un'ingegnosa architettura dual-camera operante in uno spazio continuo (`float`).
  1. *Per l'Utente:* Una telecamera fissa prospettica (`glFrustum`), visivamente stabile e piacevole. L'inquadratura è calcolata tramite traslazioni e rotazioni esatte per centrare perfettamente la griglia al centro della vista utente, indipendentemente dalla prospettiva.
  2. *Per OWL-ViT:* Una telecamera egocentrica ortografica/prospettica in grado di catturare fotogrammi senza l'uso di coordinate assolute di griglia fisse (nessun offset manuale, l'IA "vede" sempre dall'esatto centro continuo dell'agente).
- **Visione Artificiale (OWL-ViT a 360° Batched):** Il modulo `owl_vision.py` sfrutta l'accelerazione hardware della GPU eseguendo l'inferenza in *batch* contemporaneamente su 4 inquadrature (N, S, O, E). Questo permette di mappare istantaneamente il bersaglio sulle coordinate globali in qualsiasi direzione si trovi, equiparando l'agente al suo addestramento radar a 360°. I falsi positivi (tavoli e lampade visti come sedie) sono stati risolti interrogando il modello con label di esclusione parallele. La logica di estrazione dei frame è incapsulata in helper puliti all'interno di `main.py` (`_update_owl_vision_state`).
- **Esecuzione:** `main.py` orchestra prima il training dell'agente e successivamente lancia una Demo 3D visiva. La simulazione è stata aggiornata per interrompersi autonomamente dopo un numero massimo di episodi (es. 5), evitando loop infiniti. Al lancio, la simulazione attende il caricamento dei pesi OWL-ViT prima di permettere il primo passo.

## 4. Cosa Manca / Prossimi Passi
Per completare la visione finale e raggiungere l'obiettivo del progetto, mancano le seguenti espansioni:
1. **Generalizzazione Spaziale:**
   La griglia attuale è fissa a 7x7. Grazie alla rimozione delle coordinate assolute, il passaggio ad arene più grandi dovrebbe essere trasparente per l'agente, ma richiederà calibrazione sul modulo di visual-detection che dovrà scalare le bounding box 2D ottenute ai vettori direzionali 3D.
