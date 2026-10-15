# Gesture Meme Overlay

A webcam app that detects hand gestures and facial expressions in real time and overlays a matching meme image on screen.

## How it works

Using MediaPipe's hand and face landmark models, the app tracks hand positions and facial blendshapes (smile, jaw open, eyebrow raise, etc.) from your webcam feed. Each frame is checked against a set of gesture rules, from simplest/most specific to most general, and the first match triggers its meme:

- Prayer hands
- Hands on head
- Hands on cheeks
- Shushing (finger to lips)
- Peace sign
- Rock sign
- Biting finger/nail
- Hand near ear
- Open palm
- Tongue out
- Laughing
- Squinting/pucker
- Wide eyes
- Looking up
- Frowning
- Worried (raised eyebrows)

The matching meme image is pulled from the `memes` folder and overlaid on the side of the video feed.

## Requirements

```
opencv-python
numpy
mediapipe
```

## Setup

1. Install dependencies: `pip install opencv-python numpy mediapipe`
2. Add meme images to a `memes` folder next to the script, named after each gesture (e.g. `peace.png`, `shh.jpg`). Missing memes fall back to a placeholder image.
3. Run `python gestureMeme.py`. The hand and face landmark models are downloaded automatically on first run.

## Controls

- `Q` or `Esc`: quit
- `D`: toggle debug overlay (shows live blendshape values and detected finger states)
