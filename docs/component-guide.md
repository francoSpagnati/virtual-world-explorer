# Component Guide — Virtual World Explorer 3D

Questo documento spiega **ogni singola funzione** del progetto "Virtual World Explorer",
come si incastrano tra loro, e le ultime evoluzioni tecniche introdotte (il passaggio al 3D e alla gestione delle texture).

---

## Che cosa fa questo progetto?

Immagina un **robottino** (l'agente) che si muove su una **scacchiera 7×7**.
Sulla scacchiera ci sono tre oggetti tridimensionali o testurizzati: una **sedia** (il bersaglio da raggiungere),
un **tavolo** e una **lampada** (distrattori / ostacoli).

Il robottino all'inizio non sa dove sia la sedia. Deve **imparare da solo**,
provando e sbagliando. Ogni volta che si avvicina alla sedia, riceve una **ricompensa positiva** (caramella).
Ogni volta che sbatte contro un ostacolo, riceve una **punizione** e resta fermo.
Alla fine, dopo tanti tentativi, impara a raggiungere la sedia **evitando** tavolo e lampada.

Il tutto è mostrato in una **finestra grafica 3D (OpenGL)** inclinata prospetticamente, dove gli oggetti sono **modelli 3D** (OBJ) caricati da file — sedia con texture fotorealistica, tavolo e lampada con colori da materiali MTL.

---

## Come sono organizzati i file

```
virtual-world-explorer/
├── src/virtual_world_explorer/
│   ├── __init__.py          — Marcatore package
│   ├── main.py              — Entry point, training loop, demo loop
│   ├── render.py            — Renderer 3D OpenGL (prospettico, modelli 3D, cubo agente, HUD, luci)
│   ├── model3d.py           — Caricamento OBJ via trimesh + rendering OpenGL immediate mode
│   ├── env.py               — Ambiente GridWorld (griglia 7×7, oggetti scena, step/reset)
│   ├── agent.py             — Agente Q-learning tabellare
│   ├── detector.py          — Sensore semantico base (griglia)
│   └── owl_vision.py        — Integrazione visiva con OWL-ViT (Zero-Shot Object Detection)
├── assets/
│   ├── models/
│   │   ├── chair/
│   │   │   ├── Chair.obj            — Modello 3D sedia (60 facce, texturizzato)
│   │   │   ├── Chair.mtl            — Materiale sedia (riferisce Chair_BaseColor.png)
│   │   │   └── Chair_BaseColor.png  — Texture fotorealistica sedia
│   │   ├── lamp/
│   │   │   ├── Standing_lamp_01.obj — Modello 3D lampada (192 facce, 4 materiali)
│   │   │   └── Standing_lamp_01.mtl — Materiali lampada (colori Kd, nessuna texture)
│   │   └── table/
│   │       ├── model.obj            — Modello 3D tavolo (394 facce, 26 materiali)
│   │       └── materials.mtl        — Materiali tavolo (colori Kd)
│   ├── chair.png            — (Obsoleto) Sostituito da Chair.obj
│   ├── table.png            — (Obsoleto) Sostituito da model.obj
│   ├── lamp.png             — (Obsoleto) Sostituito da Standing_lamp_01.obj
│   ├── Chair.zip            — Archivio OBJ sedia
│   ├── Lamp.zip             — Archivio OBJ lampada
│   └── Table.zip            — Archivio OBJ tavolo
└── docs/
    └── component-guide.md   — Questo file
```

---

## I moduli uno per uno

### `main.py` — Il collante di tutto

| Funzione | Cosa fa |
|---|---|
| `train_agent()` | Crea `GridWorldEnv` + `QLearningAgent`, esegue 5000 episodi di addestramento (senza OWL per velocità). Restituisce ambiente e agente. |
| `run_demo()` | Apre finestra GLFW/OpenGL, imposta epsilon=0 (greedy), loop di visualizzazione. |
| `owl_worker()` | Thread in background che calcola l'inferenza di OWL-ViT. Ora il `main` aspetta *sincronamente* il suo responso per azzerare la latenza visiva, permettendo al contempo a OpenGL di non bloccarsi. |
| `_preview_position()` | Simula dove finirebbe l'agente con una data azione (per controllare collisioni prima di eseguire). |
| `_choose_action_without_loop()` | Azione greedy dinamica con *Momentum* in esplorazione cieca (tende ad andare dritto per scansionare velocemente) e penalità per mosse cicliche. |

### `render.py` — Il cuore del 3D

**Passaggio 2D → 3D sta tutto qui.** Gli altri moduli (env, agent, detector) sono rimasti identici.

| Componente | 2D (prima) | 3D (ora) |
|---|---|---|
| Proiezione | `glOrtho` (ortografica) | Doppia: `glFrustum` (utente), `glOrtho` (AI) |
| Profondità | Nessuna | `GL_DEPTH_TEST`, `GL_DEPTH_BUFFER_BIT` |
| Agente | `glRectf` (quadrato 2D) | `_draw_cube()` (6 `GL_QUADS` + normals → cubo 3D con luce) |
| Oggetti | `glRectf` colorati | Modelli 3D OBJ (da 60 a 394 facce) con texture o materiali MTL |
| Camera | Dall'alto (top-down) | Doppia: Fissa inclinata (-55° su X) per utente, Egocentrica per AI |
| Illuminazione | Nessuna | `GL_LIGHT0` + `GL_NORMALIZE` + `glMaterialfv` |
| Formati | Nessuno | OBJ + MTL + PNG texture |
| Trasparenza | Nessuna | `GL_BLEND` + `GL_SRC_ALPHA` |
| Titolo finestra | "Virtual World Explorer" | "Virtual World Explorer 3D" |

**Metodi principali:**

| Metodo | Cosa fa |
|---|---|
| `initialize()` | Crea finestra GLFW, abilita `GL_DEPTH_TEST`, `GL_BLEND`, `GL_LIGHTING`, `GL_LIGHT0`, carica modelli OBJ + texture |
| `draw()` | Imposta proiezione prospettica, camera inclinata, chiama i sotto-metodi di disegno, swap buffer |
| `_draw_grid()` | Griglia 7×7 con `GL_LINES` |
| `_draw_cube()` | Cubo solido con 6 `GL_QUADS` + normals per-face (usato per l'agente) |
| `_draw_objects()` | Itera `SceneObject`, carica `Model3D.render()` con push/pop matrix |
| `_draw_agent()` | Imposta materiale bianco, disegna cubo agente |
| `_draw_hud_overlay()` | Passa a proiezione ortografica, disabilita luci, pannello info (label, coordinate, visibilità) |
| `_draw_text()` / `_draw_glyph()` | Font bitmap con glifo 7×5 pixel in `glRectf` |
| `capture_frame()` | Renderizza in back-buffer la visuale *egocentrica ortografica* dell'AI e restituisce array NumPy RGB |

### `model3d.py` — Rendering 3D con trimesh

Delega il caricamento dei modelli 3D e dei materiali alla libreria `trimesh`, mantenendo però il rendering in OpenGL raw (immediate mode). Rispetta il vincolo GFX-only.

**Classi:**

| Classe / Metodo | Cosa fa |
|---|---|
| `Material` | Dataclass: nome, colore diffuse (Kd), ambient (Ka), path texture, ID texture OpenGL |
| `FaceGroup` | Gruppo di facce con un materiale associato |
| `Model3D(path)` | Costruttore: carica mesh/scene con trimesh, estrae vertici, facce e materiali. |
| `load_textures_gl()` | Carica texture PNG in VRAM per ogni materiale che ne ha una |
| `render(target_size)` | Disegna modello in immediate mode: centra, scala, applica materiali/texture, normali per illuminazione |

La libreria `trimesh` gestisce automaticamente il caricamento di texture, colori base, normali e facce. Il modulo `model3d.py` provvede poi a convertire i dati estratti dal sistema Y-up al sistema Z-up utilizzato da OpenGL nella scena (`(x, y, z) → (x, z, y)`), e ad applicare il corretto ridimensionamento proporzionale (bounding box scalato a `target_size`, default 0.55).

### `env.py` — L'ambiente

| Classe / Funzione | Cosa fa |
|---|---|
| `SceneObject` | Dataclass: `label` (str), `x`, `y`, `color` |
| `GridWorldEnv` | Ambiente RL |
| `reset()` | Posiziona agente + sedia/tavolo/lampada in punti casuali non sovrapposti. Restituisce observation (9 tuple) |
| `step(action)` | Applica movimento (SU/GIÙ/SX/DX). Se distrattore → reward -1, fermo. Se sedia → reward +1, done=True. Altrimenti passo pena + bonus avvicinamento |
| `_observation()` | Costruisce tupla 9 elementi da posizione agente + `SemanticDetector.detect()` |
| `_sample_positions(n)` | Sceglie `n` posizioni casuali non sovrapposte sulla griglia |
| `_manhattan_distance()` | Distanza L1 (euristica) |

**Observation** (7 valori):
```
(dx_sedia, dy_sedia, visible_flag,
 pericolo_su, pericolo_giù, pericolo_sinistra, pericolo_destra)
```
Le coordinate assolute dell'agente sono state omesse per garantire l'invarianza traslazionale e abbattere radicalmente lo spazio degli stati.

### `agent.py` — L'apprendista

| Metodo | Cosa fa |
|---|---|
| `choose_action(state)` | ε-greedy: sceglie azione casuale con prob ε, altrimenti migliore dalla Q-table |
| `learn(state, action, reward, next_state, done)` | `Q[s][a] += α * (reward + γ * max(Q[s']) - Q[s][a])` |
| `decay_exploration(min=0.05, factor=0.997)` | Annealing di ε — esplora tanto all'inizio, poi sfrutta |

**Iperparametri:** α=0.1, γ=0.9, ε iniziale=1.0. Inizializzazione **pessimistica** dei Q-values a `0.0` (anziché 1.0) per impedire che le azioni inesplorate dominino sulle azioni utili che hanno un discount factor applicato. Limite massimo di 30 passi per episodio.

### `detector.py` — Gli occhi dell'agente

| Classe / Funzione | Cosa fa |
|---|---|
| `Detection` | Dataclass: `label`, `dx`, `dy`, `visible` |
| `SemanticDetector` | Sensore semantico |
| `detect(objects, agent_position, target_label)` | Se target entro `vision_radius` (default 10 per coprire tutta la griglia), restituisce direzione normalizzata (dx ∈ {-1,0,1}, dy ∈ {-1,0,1}) e `visible=True`. Altrimenti `visible=False`, direzione zero |

**Cosa vede l'agente:** non l'immagine 3D, ma una direzione astratta. Il 3D è solo per l'occhio umano.

### `owl_vision.py` — La vista basata su Deep Learning

| Classe / Metodo | Cosa fa |
|---|---|
| `OwlVisionDetector` | Carica il modello pre-addestrato `google/owlvit-base-patch32` da HuggingFace. |
| `detect_target(image, target_name)` | Prende un'immagine ortografica, esegue l'inferenza zero-shot di OWL-ViT, calcola la bounding box, e restituisce `dx`, `dy`, `visible`. Usa confidenza a 0.02. |

**Nota Architetturale Sulla Latenza:** Inizialmente OWL-ViT girava in modo puramente asincrono causando uno sfasamento cognitivo (l'agente elaborava immagini di 3 passi fa e ignorava gli ostacoli vicini). Ora l'integrazione è **sincrona rispetto al movimento**: l'agente attende fisicamente ("congela") che il modello visivo processi il frame del suo esatto step attuale, prima di decidere l'azione, eliminando errori di posizionamento ed eseguendo un *detect* chirurgico appena il bersaglio entra nel campo visivo ravvicinato di 3 blocchi.

---

## Come gira il tutto

```
main.py
  │
  ├── train_agent()
  │     ├── GridWorldEnv.reset()
  │     ├── loop 5000 episodi:
  │     │     ├── QLearningAgent.choose_action()
  │     │     ├── GridWorldEnv.step()
  │     │     └── QLearningAgent.learn()
  │     └── restituisce (env, agent)
  │
  └── run_demo(env, agent)
        ├── OpenGLRenderer.initialize()
        ├── loop finché finestra aperta:
        │     ├── _choose_action_without_loop()
        │     ├── agent epsilon = 0 (greedy)
        │     ├── GridWorldEnv.step()
        │     └── OpenGLRenderer.draw()
        │           ├── glFrustum (prospettiva 3D)
        │           ├── _draw_grid()
        │           ├── _draw_objects()  ← Model3D.render() OBJ texturizzati
        │           │     └── Model3D: parse OBJ/MTL, Y-up→Z-up, scala, materiali Kd, texture
        │           ├── _draw_agent()    ← cubo 3D con normali
        │           └── _draw_hud_overlay()  ← luci disabilitate
        └── GLFW window close
```

---

## Schema architetturale

```
┌─────────────────────────────────────────────────┐
│                    main.py                       │
│  (orchestra training + demo, azioni anti-loop)   │
└────────┬────────────┬───────────────┬────────────┘
         │            │               │
         ▼            ▼               ▼
   ┌──────────┐ ┌──────────┐ ┌──────────────────────┐
   │ agent.py │ │ env.py   │ │   render.py          │
   │ Q-table  │ │ 7×7 grid │ │   OpenGL 3D          │
   │ ε-greedy │ │ reward   │ │   prospettico + luci │
   │ α/γ/ε    │ │ step()   │ │   Model3D.render()   │
   └──────────┘ └────┬─────┘ │   cubo agente + HUD  │
                     │       └────────┬─────────────┘
                     ▼                │
              ┌────────────┐          ▼
              │ detector.py│   ┌──────────────┐
              │ sensore    │   │  model3d.py  │
              │ semantico  │   │ parse OBJ/MTL│
              │ raggio=3   │   │ render GFX   │
              └────────────┘   └──────────────┘
```

Il **detector** fornisce alla observation la direzione verso la sedia (se in vista). L'**env** usa detector per costruire la observation, l'**agent** impara la Q-table, il **renderer** mostra tutto in 3D. La logica di gioco (env, agent, detector) è indipendente dal rendering — si può sostituire il 3D con un altro frontend visivo senza toccare l'RL.
