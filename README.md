# BallHunter Web App

BallHunter is a mobile-first browser app for locating lost play balls such as tennis, rubber, and plastic balls.

## BallHunter Vision V2

- Create a BallPrint from 1–6 reference photos of the exact ball
- Combine colour distribution, brightness/saturation variation, and a lightweight texture signature
- Quick colour search when no saved reference exists
- Rear-camera live search with candidate-blob extraction
- Score candidates using colour, partial shape, compactness, and local contrast
- Track candidates across frames with occlusion persistence
- Guide the user to change angle using parallax-reveal instructions
- Maintain a small on-screen evidence heatmap
- Save BallPrint profiles locally in the browser
- Installable/offline-capable PWA shell

## How to test

1. Open the GitHub Pages site on a phone.
2. Add a ball and use 2–4 close reference views when possible.
3. Choose **Find my saved ball** and select the BallPrint.
4. Allow rear-camera access.
5. Sweep the ground slowly.
6. When a candidate appears, keep it in view and follow the sideways/lower-angle guidance.
7. Confirm **I found the ball** when recovered.

## Important limitation

A normal RGB phone camera cannot literally see through fully opaque objects. Vision V2 improves detection of visible and partially visible fragments, but fully hidden-ball recovery requires inference from information such as trajectory, disappearance point, depth, or additional sensors.
