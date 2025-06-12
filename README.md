# 🔎 Edge Detection Attack Lab

---

## 1 . Project overview

This is a self-contained Python playground that **simulates, attacks, and evaluates** classical edge-detection pipelines (Sobel, Canny, Laplacian of Gaussian, Roberts Cross).
It lets you

- preprocess and normalise a set of demo images;
- run a variety of _targeted_ (algorithm-specific) and _progressive_ (generic) adversarial attacks;
- compute effectiveness metrics (edge-density drop, contour-area reduction, fragmentation); and
- save side-by-side artefacts (**clean / attacked / edge-maps**) plus JSON scorecards for later analysis.

Everything is pure NumPy + OpenCV—no deep-learning toolkit required.

---

## 2 . Key components

| Module                      | Purpose                                                                                                                                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `targeted_attacks.py`       | Three attacks that focus on _strong-edge_ pixels (blur, gradient reversal, contour disruption).                                                                                                                   |
| `progressive_attacks.py`    | Pixel-noise escalation **and** generic edge/contour attacks executed over multiple “gentle → aggressive” rounds.                                                                                                  |
| `canny_targeted_attacks.py` | Five attacks that exploit _specific_ Canny stages (hysteresis noise, gradient smoothing, non-max-suppression confusion, connectivity gaps, multi-scale perturbations) and sweep three Canny hyper-parameter sets. |
| `simulation_engine.py`      | Object-oriented façade (`EdgeAttackSimulationEngine`) with > 40 tunable parameters – useful for interactive notebooks or web front-ends.                                                                          |

Each script deposits PNG artefacts plus a `*_results.json` manifest in its own output folder (`targeted_attacks/`, `progressive_attacks/`, …).

---

## 3 . Installation

```bash
# 1. Clone and cd
git clone https://github.com/<you>/edge-attack-lab.git
cd edge-attack-lab

# 2. Create environment (≈ Python 3.9+)
python -m venv .venv
source .venv/bin/activate                # Linux / macOS
# .venv\Scripts\activate.bat             # Windows

# 3. Install runtime deps
pip install -r requirements.txt
# requirements.txt
# ├─ numpy
# └─ opencv-python
```

> **Tip:** add `opencv-contrib-python` if you later need advanced CV functions.

---

## 4 . Running the demos

```bash
# 1. Generic Sobel / contour / gradient demo
python targeted_attacks.py

# 2. Six-stage progressive escalation
python progressive_attacks.py

# 3. Canny-specific research sweep
python canny_targeted_attacks.py

# 4. Custom batch with OO engine
python simulation_engine.py
```

Each run ends with a concise success tally:

```
Results saved to 'progressive_attacks/'
Successful attacks: 11/18
```

and drops artefacts such as:

```
progressive_attacks/
│  street_scene-sobel-gentle_pixels-clean.png
│  street_scene-sobel-gentle_pixels-pert.png
│  street_scene-sobel-gentle_pixels-edges-clean.png
│  street_scene-sobel-gentle_pixels-edges-pert.png
└─ metadata.json
```

---

## 5 . Understanding the metrics

| Metric                   | Meaning                                 | Success threshold |
| ------------------------ | --------------------------------------- | ----------------- |
| `edge_density_reduction` | Fractional drop in total edge pixels.   | > 15 %            |
| `contour_area_reduction` | Drop in largest connected edge region.  | > 20 %            |
| `fragmentation_increase` | (Canny-only) rise in contour count.     | > 50 %            |
| `attack_success`         | **True** if _any_ threshold is crossed. | –                 |

---

## 6 . Extending the lab

1. **Add a new attack**
   Implement a function that takes `gray_image: np.ndarray` and returns a perturbed image of identical shape.
   Register it inside the relevant `attack_functions` dictionary.

2. **Plug in your own images**
   Drop them into `source_images/` and extend the `IMAGE_SOURCES` mapping.

3. **Integrate with notebooks / web UI**
   Import `EdgeAttackSimulationEngine`, tweak `SimulationConfig`, and call `execute_simulation_on_image(...)`.

---

## 7 . Roadmap

- 🔬 Add patch-wise **learned** perturbations via gradient-free optimisation.
- 🖥 Streamline a **Streamlit** or **Next.js** front-end for interactive exploration.
- 📊 Auto-generate HTML reports comparing detectors, attacks, and thresholds.

---

## 8 . Contributing & licence

Pull requests and issue reports are welcome!
All code is released under the MIT License—see [`LICENSE`](LICENSE) for details.

```

```
