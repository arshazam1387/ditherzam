# Masking in ditherzam

This guide covers the editable raster masks in the **Layers** panel. A raster
mask controls which parts of one layer are visible without deleting image
pixels.

## The basic rule

- **White reveals** the layer.
- **Black hides** the layer.
- **Gray partially reveals** the layer.

A raster mask belongs to one layer and follows that layer when it is moved or
resized. Mask editing is non-destructive: the source image remains unchanged.

## Quick start: hide a background

1. Load an image and select its layer in the **Layers** panel.
2. Open the **Mask** tab. Clicking **Add Mask** in the layer row also opens it.
3. Choose **Create → Reveal All**. This starts with the whole layer visible.
4. Choose **Paint Mask**, set the brush mode to **Hide**, and paint over the
   background on the canvas.
5. Use **Mask Only** to check the mask or **Red Overlay** to see hidden areas on
   top of the artwork.
6. Choose **Pointer** when you are finished painting.

For the opposite workflow, choose **Create → Hide All**, switch the brush to
**Reveal**, and paint the subject back in.

## Smart Mask and raster masks are different

ditherzam has two masking systems:

| Mask | What it controls | Where it lives |
|---|---|---|
| **Smart Mask** | Where the current dithered Look and its outside treatment appear | Inside the Look controls |
| **Raster mask** | The final visibility of the whole layer after the Look is rendered | In the Layers panel |

Use Smart Mask when you want an effect to apply to the subject or background.
Use a raster mask when you want to cut out, reveal, repair, or blend the layer
itself. You can also turn a Smart Mask result into an editable raster mask.

## Creating a raster mask

Select a layer, open **Mask**, then choose **Create**:

| Command | Result |
|---|---|
| **Reveal All** | A white mask; the whole layer starts visible |
| **Hide All** | A black mask; the whole layer starts hidden |
| **From Transparency** | Copies the source image's alpha into the mask |
| **From Smart Mask** | Copies the current completed Smart Mask result |
| **From Shadows / Midtones / Highlights** | Creates a soft mask from source-image brightness |
| **Linear / Radial Gradient** | Creates a directional or radial fade |
| **Pattern** | Creates a Bayer, line, or seeded-noise mask |
| **Freeze Smart** | Opens threshold, grow/shrink, feather, and invert controls before creating an exact raster mask |

If the layer already has an edited mask, ditherzam asks before replacing it.
Nothing is overwritten until you choose **Replace**.

### Combining a new mask with the current mask

The **Combine** control changes how generated, imported, or Smart Mask
candidates interact with an existing mask:

- **Replace** uses the new candidate by itself.
- **Add** reveals the union of the current and new coverage.
- **Subtract** removes the new coverage from the current mask.
- **Intersect** keeps only their overlapping coverage.

## Painting a mask

1. Select the layer and make sure its raster mask is **Enabled**.
2. Choose **Paint Mask**. You can also activate the mask target in the layer
   row.
3. Choose **Reveal** or **Hide**.
4. Adjust the brush and paint on the canvas.
5. Choose **Pointer** to return to normal canvas interaction.

**Paint Mask** stays unavailable until the selected layer has an enabled raster
mask. If it is disabled, enable the mask or create/import one first.

Brush controls:

| Control | Effect |
|---|---|
| **Tip** | Round, Square, Diamond, or deterministic Texture |
| **Size** | Brush diameter in source-image pixels |
| **Hardness** | Edge sharpness |
| **Strength** | Amount revealed or hidden by each pass |
| **Spacing** | Distance between brush stamps as a percentage of brush size |

Useful brush keys:

- `[` and `]` decrease or increase brush size.
- `X` swaps Reveal and Hide.
- Hold `Space` to pan temporarily.
- `Esc` cancels the current, unfinished stroke.
- **Edit → Undo** reverses a completed stroke; **Redo** restores it.

Each completed stroke is one undo step. A cancelled stroke does not modify the
mask.

## Making selections

Selections are temporary boundaries for mask work. They do not become part of
the document history, and they are not saved as masks until you choose **From
Selection**.

1. Choose **Rectangle**, **Ellipse**, **Polygon**, or **Freehand**.
2. Choose an operation:
   - **Replace** starts a new selection.
   - **Add** adds to the current selection.
   - **Subtract** removes from the current selection.
3. Choose **Select on Canvas** and draw. Shape tools return to the normal
   pointer after the selection is completed.
4. Optionally use **Grow**, **Shrink**, **Feather**, **Invert**, or **All**.
5. Choose **From Selection** to create a raster mask, or keep the selection
   active to limit brush, fill, invert, and generated-mask edits.
6. Choose **Clear** when the selection should no longer restrict edits.

For a polygon, click each corner and double-click or press `Enter` to close it.
Press `Esc` to leave a selection tool without completing it.

### Color Range

Use Color Range to select similar source colors:

1. Choose **Color Range**, then click a color on the selected image layer.
2. Adjust **Tolerance** to include a wider or narrower color range.
3. Adjust **Softness** to control the falloff at the edge of that range.
4. Choose **Confirm** to keep the temporary selection or **Cancel** to restore
   the previous selection.
5. Choose **From Selection** if you want the result to become a raster mask.

## Inspecting and adjusting a mask

The inspection control changes only the canvas display. It never changes the
mask, thumbnails, or exported pixels.

| View | Use it for |
|---|---|
| **Normal** | Judge the composited artwork |
| **Red Overlay** | See hidden areas over the artwork |
| **Mask Only** | Inspect the exact black, white, and gray mask coverage |

Other mask controls:

- **Enabled** bypasses or restores the raster mask without deleting it.
- **Density** fades the mask's hiding effect. At 0%, the mask is bypassed; at
  100%, its stored coverage is used exactly.
- The Mask tab's **Edit → Invert** swaps revealed and hidden areas.
- **Fill White / Fill Black** reveals or hides the current selection, or the
  entire mask when no selection is active.
- **Dither the Mask** thresholds a soft mask through Bayer, Lines, or Noise and
  can mix the result back with the original softness.
- **Reset / Replace** proposes a fresh fully revealed mask.
- **Delete** removes the raster mask from the selected layer.

## Importing, exporting, and current limits

- **Import** accepts a grayscale mask PNG. It must match the selected layer's
  source dimensions.
- **Export** writes the stored mask as an exact grayscale PNG.
- Main PNG/JPEG export composites the complete layer stack and its enabled
  raster masks.
- Editable project-file persistence is not available yet. Export important
  masks as PNG files if you need to reuse them in another session.
- Manual raster masking is currently a still-image workflow. Animation, video,
  SVG, and batch masking are outside this workflow.
- Smart Mask runs locally. If its local model is unavailable, manual masks,
  selections, luminance masks, gradients, patterns, and mask PNG import still
  work offline.

## Troubleshooting

**The brush does nothing**

Check that the correct layer is selected, a raster mask exists, the mask is
enabled, and **Paint Mask** is active. A temporary selection may also be
restricting the editable area.

**The whole layer disappeared**

You probably created **Hide All**. Switch the brush to **Reveal**, paint the
area you want, or use the Mask tab's **Edit → Fill White**.

**A generated mask did not replace my current work**

Look for the inline **Replace / Cancel** confirmation. Existing edited masks
are never silently overwritten.

**The canvas is red or black and white**

Change the inspection view back to **Normal**. Red Overlay and Mask Only are
display aids, not changes to the exported image.

**Smart Mask is unavailable**

The local model may not be installed or inference may still be running. Use a
manual or generated raster mask, or wait for the current Smart Mask result.
