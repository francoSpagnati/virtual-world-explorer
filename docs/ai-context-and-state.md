# AI Context & Project State — Virtual World Explorer

Questo file funge da "mappa" e "contesto" per qualsiasi Intelligenza Artificiale che deve comprendere, modificare o espandere questo progetto. Leggilo attentamente prima di proporre modifiche.

## 1. Dominio e Contesto
Siamo in un progetto Python che simula un ambiente 3D ("Virtual World Explorer") in cui un agente controllato da algoritmi di Deep Reinforcement Learning in PyTorch deve imparare a navigare una griglia evitando ostacoli (es. tavolo, lampada) per raggiungere un bersaglio (sedia).
L'aspetto visivo è interamente delegato a un renderer **OpenGL 3D (immediate mode)**.
L'obiettivo a lungo termine del progetto è integrare modelli di Vision-Language (come **OWL-ViT**) per dare all'agente input semantici estratti dai frame renderizzati (es. per capire dov'è la sedia "guardando" lo schermo).

## 2. Il Vincolo Principale (GFX-Only)
Esiste un limite tecnologico ferreo imposto all'architettura:
> Nel seguito, qualsiasi grafica di basso livello come OpenGL o WebGL viene chiamata **GFX**.
> Qualsiasi framework di alto livello, come Unity, Unreal Engine o librerie grafiche JavaScript, sono **vietati**.

Il rendering *deve* avvenire manipolando direttamente lo stato OpenGL tramite `PyOpenGL` e `GLFW`. Attualmente viene usata la "immediate mode" (`glBegin`/`glEnd`). Nonostante i modelli 3D vengano parsati tramite la libreria ausiliaria `trimesh` (esplicitamente autorizzata per snellire il codice matematico), il loop di rendering effettivo su schermo è e deve restare esclusivamente GFX.

## 3. Stato Attuale del Progetto (Configurazione)
Il progetto è in uno stato avanzato e pienamente funzionante, configurato come segue:
- **Agente RL (Tuning):** `agent.py` utilizza un approccio continuo con `ContinuousPolicyNetwork` in PyTorch (output v, w). L'addestramento è stato esteso a 30000 episodi per permettere all'agente di padroneggiare mappe molto congestionate (ora con 6 ostacoli invece di 2). I controlli sono predetti da una rete neurale con esplorazione basata su rumore gaussiano, eliminando la necessità di una Q-table fissa e garantendo maggiore generalizzazione continua nell'intero spazio d'azione. Il limite di mosse impedisce percorsi infiniti e le penalità forzano percorsi ottimali.
- **Spazio di Stato Ottimizzato (11-D):** L'agente percepisce il mondo tramite una tupla relativa continua: `(dx_target, dy_target, visible, danger_0, danger_45, danger_90, danger_135, danger_180, danger_225, danger_270, danger_315)`. Le coordinate assolute sono state rimosse. I sensori di prossimità (`danger_*`) formano un radar a 8 canali per una navigazione omnidirezionale formidabile.
- **Sensore Logico & Momentum:** Quando l'agente non vede la sedia (visible=0), usa una logica di "Momentum" in `main.py` che fonde l'azione corrente con quella precedente per forzare l'esplorazione in linea retta ed evitare minimi locali. Inoltre, se il radar rileva una collisione imminente, sterza bruscamente sul posto.
- **Motore 3D a Doppia Telecamera e Spazio Continuo:** `render.py` adotta un'ingegnosa architettura dual-camera operante in uno spazio continuo (`float`).
  1. *Per l'Utente:* Una telecamera fissa prospettica (`glFrustum`). L'agente è ora renderizzato utilizzando un **modello 3D di un Robot** texturizzato (`Robot.obj`) che si sostituisce al cubo originale. L'HUD di sistema formatta accuratamente le posizioni floating point per rimanere leggibile.
  2. *Per OWL-ViT:* Una telecamera egocentrica ortografica/prospettica in grado di catturare fotogrammi dalle esatte coordinate continue dell'agente.
- **Visione Artificiale (OWL-ViT a 360° Batched):** Il modulo `owl_vision.py` sfrutta l'inferenza in *batch* sulle 8 inquadrature ruotate di 45 gradi. La threshold di confidenza è impostata a `0.10` per azzerare i falsi positivi. I vettori continui generati dal modello `(dx, dy)` vengono calcolati geometricamente basandosi sull'offset orizzontale e **rigorosamente normalizzati** in `main.py` (lunghezza 1.0) prima di essere passati alla rete neurale.
- **Esecuzione & Testing:** `main.py` orchestra training e Demo 3D visiva. La simulazione grafica si interrompe autonomamente dopo un numero massimo di episodi (es. 6), o dopo aver raggiunto un limite massimo di passi per episodio pari all'area della mappa (`size * size`, ovvero 49 passi). È presente anche una `test_suite.py` progettata per aggirare la generazione randomica dell'arena e caricare posizionamenti prestabiliti per misurare la performance dell'euristica.

## 4. Cosa Manca / Prossimi Passi
Per completare la visione finale e raggiungere l'obiettivo del progetto, mancano le seguenti espansioni:
1. **Generalizzazione Spaziale:**
   La griglia attuale è fissa a 7x7. Grazie alla rimozione delle coordinate assolute, il passaggio ad arene più grandi dovrebbe essere trasparente per l'agente, ma richiederà calibrazione sul modulo di visual-detection che dovrà scalare le bounding box 2D ottenute ai vettori direzionali 3D.
