# BallHunter Web App

BallHunter is a mobile-first browser app for locating lost play balls such as tennis, rubber, and plastic balls.

## Current MVP

- Save a reference photo of the exact ball before playing
- Create a local BallPrint colour fingerprint from the reference photo
- Quick colour search when no saved reference exists
- Rear-camera live search
- Multi-frame probability scoring across a camera grid
- Candidate highlighting and directional guidance
- Local-only ball profiles using browser storage

## How to test

1. Open the GitHub Pages site on a phone.
2. Add a ball and photograph it closely, with the ball centered.
3. Choose **Find saved ball** and select the saved ball.
4. Allow rear-camera access.
5. Move the camera slowly across the search area.
6. BallHunter highlights the highest-probability region.

## Important limitation

A normal RGB phone camera cannot literally see through fully opaque objects. This MVP can detect matching visible fragments and repeatedly score likely regions across frames. Future versions can add object segmentation, learned visual embeddings, trajectory prediction, depth sensing, and sensor-assisted search.
