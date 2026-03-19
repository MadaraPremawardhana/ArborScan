# 🌳 ArborScan

**Tree detection from satellite & aerial imagery using JAX-powered computer vision.**

ArborScan analyses satellite or aerial images to extract individual tree positions, crown radii, and confidence scores — outputting a geo-referenced 2D point cloud in WGS-84 coordinates. The full detection pipeline runs client-side in the browser, with no server or backend required.

![ArborScan screenshot](assets/screenshot.png)

---

## Features

- **JAX-inspired detection pipeline** — NDVI vegetation indexing, Gaussian blur, connected-component labelling, blob radius estimation
- **Geo-referencing** — each detected tree mapped to real-world latitude/longitude using WGS-84
- **Interactive 2D map** — pan, zoom, hover tooltips, crown radius filter
- **Google Maps URL parser** — paste a Maps link and coordinates are auto-extracted
- **Point cloud export** — CSV, GeoJSON (QGIS/Mapbox-ready), and JSON
- **Zero dependencies** — single `.html` file, runs entirely in the browser

---

## Quick Start

### Option 1 — Open directly

```bash
git clone https://github.com/YOUR_USERNAME/arborscan.git
cd arborscan
open src/tree-detection.html
```

No build step. No server. Just open the file in any modern browser.

### Option 2 — Serve locally (recommended for file uploads)

```bash
# Python
python3 -m http.server 8080

# Node
npx serve .
```

Then navigate to `http://localhost:8080/src/tree-detection.html`.

---

## Usage

### Upload an image

1. Drag and drop a satellite or aerial image (PNG, JPEG, TIFF) onto the upload zone
2. Set the **centre latitude/longitude** of your image and the **coverage area in metres**
3. Click **Run JAX Detection**
4. Switch to the **Map View** tab to see detected trees

### Use a Google Maps link

1. Go to [Google Maps](https://maps.google.com), navigate to your area of interest, switch to **Satellite** view
2. Copy the URL (it will contain `@lat,lon,zoom`)
3. Paste into the **Google Maps URL** field — coordinates are parsed automatically
4. Click **Generate Demo Detection** to run detection at those coordinates on synthetic imagery

> **Note:** Browser security restrictions (CORS) prevent live Maps tile fetching. For real analysis, use the image upload path with exported satellite tiles.

---

## Detection Pipeline

The pipeline mirrors what a Python JAX implementation would execute:

```
Raw image pixels
       │
       ▼
Normalise [0, 1]
       │
       ▼
NDVI vegetation index: (G - R) / (G + R + ε)
       │
       ▼
Gaussian blur  σ = 2.5
       │
       ▼
Threshold mask  > 0.18
       │
       ▼
Connected-component labelling
       │
       ▼
Blob analysis → centroid (cx, cy) + radius r
       │
       ▼
Geo-reference → WGS-84 (lat, lon)
       │
       ▼
2D point cloud  [{ lat, lon, radius_m, confidence }]
```

### Python / JAX equivalent

If you want to run this pipeline server-side with real JAX:

```python
import jax
import jax.numpy as jnp
from jax import jit
from PIL import Image
import numpy as np

@jit
def compute_ndvi(img):
    """img: float32 array shape (H, W, 3), values in [0, 1]"""
    r, g = img[..., 0], img[..., 1]
    return (g - r) / (g + r + 1e-6)

@jit
def gaussian_blur(arr, sigma=2.5):
    """Simple separable Gaussian blur via convolution."""
    radius = int(jnp.ceil(sigma * 2))
    k = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    kernel = jnp.exp(-k**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    # Apply separably along H then W
    arr = jnp.convolve(arr.reshape(-1), kernel, mode='same').reshape(arr.shape)
    return arr

def detect_trees(image_path, centre_lat, centre_lon, span_metres):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32) / 255.0
    H, W = img.shape[:2]

    ndvi     = compute_ndvi(jnp.array(img))
    blurred  = gaussian_blur(ndvi)
    mask     = (blurred > 0.18).astype(jnp.int32)

    # Connected components (use scipy for now — JAX-native CC in progress)
    from scipy import ndimage
    labeled, n_blobs = ndimage.label(np.array(mask))

    trees = []
    deg_lat = span_metres / 111320
    deg_lon = span_metres / (111320 * np.cos(np.radians(centre_lat)))

    for label in range(1, n_blobs + 1):
        blob = np.argwhere(labeled == label)
        if len(blob) < 8:
            continue
        cy, cx = blob.mean(axis=0)
        radius_px = np.sqrt(len(blob) / np.pi)
        radius_m  = radius_px * span_metres / max(W, H)
        if not (1.5 <= radius_m <= 40):
            continue
        trees.append({
            'lat':        centre_lat + (0.5 - cy / H) * deg_lat,
            'lon':        centre_lon + (cx / W - 0.5) * deg_lon,
            'radius_m':   round(radius_m, 1),
            'confidence': min(0.99, 0.5 + len(blob) / 600),
        })

    return trees
```

Install requirements:

```bash
pip install jax jaxlib pillow scipy numpy
```

---

## Output Format

Detected trees are exported as a GeoJSON FeatureCollection or flat CSV/JSON:

### GeoJSON

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 1,
        "radius_m": 7.4,
        "confidence": 0.87
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-0.1271543, 51.5068291]
      }
    }
  ]
}
```

### CSV

```
id,latitude,longitude,radius_m,confidence
1,51.5068291,-0.1271543,7.4,0.870
2,51.5072104,-0.1265832,5.1,0.762
```

GeoJSON output is compatible with **QGIS**, **Mapbox GL JS**, **Leaflet**, **ArcGIS**, and any GIS tool that accepts standard GeoJSON.

---

## File Structure

```
arborscan/
├── src/
│   └── tree-detection.html   # Complete single-file web app
├── examples/
│   └── sample_output.geojson # Example point cloud output
├── assets/
│   └── screenshot.png        # UI screenshot
├── .gitignore
├── LICENSE
└── README.md
```

---

## Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome 90+ | ✅ Full |
| Firefox 88+ | ✅ Full |
| Safari 14+ | ✅ Full |
| Edge 90+ | ✅ Full |

Requires Canvas 2D API and Clipboard API (for JSON copy). No WebGL needed.

---

## Roadmap

- [ ] Python JAX backend with FastAPI endpoint
- [ ] WebAssembly JAX runtime (Pyodide integration)
- [ ] Multi-scale detection for varying zoom levels
- [ ] Species classification layer (broadleaf vs conifer)
- [ ] Tile-based processing for large-area imagery
- [ ] Temporal change detection (compare two dates)

---

## Contributing

Contributions welcome. Please open an issue first to discuss larger changes.

```bash
git checkout -b feature/your-feature
git commit -m "feat: describe your change"
git push origin feature/your-feature
# then open a Pull Request
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Detection algorithm inspired by NDVI-based vegetation segmentation methods used in remote sensing research. Tree crown delineation approach based on connected-component analysis of vegetation index masks.
