# Teleimager Upstream

This directory is vendored directly into `url_g1_deploy` so the G1 onboard
deployment has one clone and one tested revision.

- Upstream: https://github.com/unitreerobotics/teleimager
- Imported commit: `57cf2a40572227273fa001cd17833b755331ec97`
- Import date: 2026-07-16
- Upstream license: Apache-2.0 (`LICENSE` in this directory)

Local changes add a backward-compatible timestamped ZMQ recording stream. The
legacy raw-JPEG ZMQ stream and WebRTC stream remain available.

When refreshing from upstream, compare from the imported commit above, retain
the recording protocol tests, and update this file to the new tested commit.
