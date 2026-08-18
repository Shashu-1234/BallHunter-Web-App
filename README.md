# BallHunter Vision V4

BallHunter is a mobile-first lost-ball search application for tennis, rubber, plastic and other play balls.

## Vision V4

Vision V4 is deliberately evidence-gated: a matching colour is only a candidate proposal, not a ball detection.

The live search pipeline combines:

- Target-colour candidate proposals
- Shape Gate with aspect ratio, compactness, circularity, boundary radial consistency and corner occupancy
- Explicit penalties for elongated, rectangular and oversized colour patches
- Multi-frame tracking that stabilizes hypotheses without boosting a rejected shape
- Short-term occlusion persistence
- Parallax Reveal guidance for uncertain/partially hidden candidates
- An 8 x 8 Search Probability Map based on ball probability rather than raw colour
- Saved-ball neural re-identification using BallPrint reference images when the browser model is available
- Optional AI mask refinement for strong saved-ball candidates
- Local search history and confirmed-result feedback

## Decision language

Quick Colour Search can report:

- Colour match — not ball-like
- Object under review
- Possible ball
- Likely ball

Saved BallPrint Search can additionally report:

- Ball-like object — identity pending
- Ball-like, but not your saved ball
- Possible saved-ball match
- Likely your ball

## BallPrint

Create a BallPrint with 1–6 close reference views. Multiple angles improve the saved-ball reference. Profiles are stored locally in the browser.

## Test protocol

1. Choose a colour such as orange.
2. Point the camera at orange non-ball objects such as cloth, boxes, books or packets. Shape Gate should suppress or reject them rather than calling them a likely ball.
3. Point the camera at a real ball of the selected colour. Keep it visible for multiple frames and confirm that the Ball Shape score and Ball Probability rise.
4. Repeat with the ball partially hidden by grass/leaves and follow the Parallax Reveal guidance.
5. In Saved BallPrint mode, wait for exact-ball identity to become ready and compare the saved ball with other balls/objects.

## Important limitation

A normal RGB phone camera cannot see through a completely opaque object. V4 can preserve a previously supported hypothesis through brief occlusion, but fully hidden-ball recovery ultimately needs additional evidence such as trajectory, disappearance history, depth or external sensors.

## Future model

Vision V4 is not yet a custom BallHunter-trained object detector. A specialized model requires a labelled dataset of real balls, severe occlusion cases and hard negatives such as wrappers, flowers, bottle caps, stones and similarly coloured objects.
