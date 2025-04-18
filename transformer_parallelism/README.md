# Transformer Parallelism Visualization

This folder contains a compiled React-based visualization for tensor parallelism in transformer models. The visualization is designed to be integrated with a Jekyll site while being completely isolated from the site's styling.

## Integration with Jekyll

### Option 1: Direct Copy (Recommended)

1. Copy the entire `transformer_parallelism` directory to your Jekyll site's root directory
2. The visualization will be available at: `https://github.com/damek/STAT-4830/transformer_parallelism/`

### Important: Style Isolation

This visualization is configured to be completely isolated from your Jekyll site's theme and styles:

- The `layout: null` in index.html prevents Jekyll from applying any layouts
- A comprehensive CSS reset in the HTML file prevents styles from leaking in
- The local `_config.yml` further isolates this directory from site-wide settings
- The `.nojekyll` file provides an additional hint to Jekyll to treat this folder specially

These measures ensure the visualization will display correctly with its own styling regardless of your Jekyll site's theme.

## Linking to the Visualization

Add a link in your Jekyll site to the visualization:

```html
<a href="{{ site.baseurl }}/transformer_parallelism/">Transformer Parallelism Visualization</a>
```

## Features

The visualization demonstrates:

- Tensor Parallelism in transformer models
- Column and row parallel layers
- Forward and backward passes
- Memory and communication requirements

## Troubleshooting

If you encounter any issues:

1. Make sure all asset paths are correctly resolved (paths starting with `./assets/`)
2. Check browser console for any JavaScript errors
3. Ensure KaTeX is properly loaded for mathematical formula rendering