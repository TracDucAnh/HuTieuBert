# HuTieuBERT Attention Visualization

Visualize and compare attention heatmaps across layers of **HuTieuBERT** with and without bias.

---

## How to Run

### 1. Clone the repository
```bash
git https://github.com/TracDucAnh/HuTieuBert.git
cd HuTieuBert
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment
- Window
    ```bash
    venv\Scripts\activate
    ```
- MacOS, Linux
    ```bash
    source venv/bin/activate
    ```
### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the script

```bash
python observe_changes.py
```

## To modify visualization config
In observe_changes.py
```python
layers_to_visualize = [5]   # <-- config layer bias list
heads = list(range(12))        # apply bias to all heads
```

## To modify how strong bias matrix effect to the overall attention scores
In observe_changes.py
```python
alpha_config = 0.5 # <-- strengthen attention scores inside a compound word 
beta_config = -0.25 # <-- weelen attention scores within compounds
gamma_config = 0.0 # <-- bias single token
delta_config = 0.0 # <-- bias self token
```