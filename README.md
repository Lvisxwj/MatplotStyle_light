To QXX, my love: 

To the world you may be one person, but to one person you are the world. I wish things couldve been different. I love you.

# MatplotStyle_light
Designed for 2025 CUMCU, first project ever. It's trash but, what more can I say :)


A unified visualization style framework for **matplotlib** / **seaborn**,  
powered by **YAML configuration** + **Python class wrapper**.  
Designed for **2025 CUMCU** — keep your plots **consistent** and **professional**.

---

## ✨ Features
- **Centralized style control**: all fonts, colors, linewidths, transparency, legends, grids, etc. are defined in `style_config.yaml`.
- **Matplotlib-like API**: write plots as usual (`lineplot`, `scatterplot`, `barplot`, etc.) but styles are automatically applied.
- **Extended support**: includes not only basic plots, but also heatmap, boxplot, violinplot, pie, polar plot, radar chart, and colorbar.
- **Flexible toggles**: switches like `grid`, `legend`, `baseline` are controlled in `main` function calls, not hardcoded.
- **Notebook-friendly**: smart `show()` detects Jupyter Notebook vs Python script to avoid duplicate rendering.

---

## 📂 Project Structure
```text
.
├── Viz_Style.py        # Core class XWJ_Style
├── style_config.yaml   # Global YAML style configuration

```

---

## ⚙️ Installation
Clone this repository and install dependencies:
```bash
https://github.com/Lvisxwj/MatplotStyle_light.git
cd XWJ_Style
```

Dependencies:
- matplotlib
- seaborn
- pyyaml
- numpy

---

## 🛠️ Usage Example

### 1. Basic Line Plot
```python
from Viz_Style import XWJ_Style
import numpy as np

style = XWJ_Style()

x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)

style._new_figure()
style.lineplot(x, y1, label="sin(x)")
style.lineplot(x, y2, label="cos(x)")
style._apply_labels(xlabel="X Axis", ylabel="Value", title="Sine vs Cosine")
style._apply_legend(add_legend=True, title="Functions")
style.show()
```

---

### 2. Multiple Subplots + Suptitle
```python
fig, axes = style.subplots(2, 2)
for i, ax in enumerate(axes.flat, start=1):
    x = np.linspace(0, 5, 50)
    y = np.sin(x + i)
    ax.plot(x, y, label=f"sin(x+{i})", color=style.get_next_color())
    ax.legend()
    ax.set_title(f"Subplot {i}")

style.suptitle("Multiple Subplots Example")
style.tight_layout()
style.show()
```

---

### 3. Heatmap + Colorbar
```python
import numpy as np
data = np.random.rand(6, 6)

style._new_figure()
ax = style.heatmap(data, cbar=False)
style.colorbar()   # apply YAML config to colorbar
style.set_title("Heatmap with Custom Colorbar")
style.show()
```

---

### 4. Polar Plot & Radar Chart
```python
theta = np.linspace(0, 2*np.pi, 100)
r = np.abs(np.sin(2*theta))
style.polarplot(theta, r, label="Polar Function")
style._apply_legend(add_legend=True, title="Polar Legend")
style.show()

categories = ["Speed", "Power", "Accuracy", "Endurance", "Agility"]
values = [7, 8, 6, 9, 5]
style.radarchart(categories, values, label="Player A")
style._apply_legend(add_legend=True, title="Radar Legend")
style.show()
```

---

## 📝 Configuration (style_config.yaml)

Example snippet:
```yaml
font:
  family: "centurygothic_bold"
  size: 12
  title_size: 14
  title_weight: bold

figure:
  figsize: [12, 8]

colors:
  DeepPurple: "#592E83"
  GutsyFuchsia: "#D72657"
  Blue: "#2E86AB"
  Red: "#C73E1D"

legend:
  loc: "upper right"
  frameon: true
  fancybox: true
  shadow: true
  title_fontsize: 11
  fontsize: 12

grid:
  alpha: 0.3
  linewidth: 0.5
  linestyle: "-"
```

Edit `style_config.yaml` to instantly change the theme of all your plots.

---

## 📊 Supported Plot Types
- Line, Scatter, Bar, Histogram
- Fill Between
- Heatmap
- Boxplot, Violinplot
- Pie chart
- Polar plot
- Radar chart
- Colorbar

---

## 📖 License
MIT License.

---

## 🙌 Contributing
Found a bug or want to add new plot types?  
Fork this repo, create a branch, and submit a pull request!
I am but a noob, so bugs are everywhere :) 
