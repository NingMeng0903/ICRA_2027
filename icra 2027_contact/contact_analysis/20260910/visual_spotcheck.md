# Visual spot check

This spot check covers only the event montages for `yameng_1`, `yameng_2`, and
`zhongkai`. I selected the largest detected dark-band event and one short or
boundary event for each person. The red rectangle in each montage is the
detector's candidate region. The times and frames below are from
`events.csv`; the peak is the frame with the largest reported dark-width
fraction.

| person | montage | detector interval | peak | visual judgement |
|---|---|---|---|---|
| yameng_1 | `yameng_1/RH_Per_S_DtP/event_01.jpg` | frame 0–111, 0.000–3.699 s | frame 61, 2.04 s, width 0.531 | Strongest observed case. At the peak, the left shallow boxed strip becomes a broad, nearly black vertical region while the adjacent muscle/fascial pattern remains visible. This is consistent with a lateral near-field echo loss or oblique probe coupling. The deeper black fan below the bright curved bone line is separate and is compatible with bone shadow. The interval is left-censored because the recording starts with the candidate already present, so its physical onset is unknown. |
| yameng_1 | `yameng_1/RH_Per_L_PtD/event_01.jpg` | frame 14–55, 0.464–1.832 s | frame 43, 1.43 s, width 0.188 | Short transient with a visible onset, broadening around frames 43–55, and disappearance by frame 64 in the montage. The box is on the left edge and shallow, while the center and deeper structures remain visible. It is a plausible coupling/angle transient, but its narrow edge location makes it weaker evidence than the long event. |
| yameng_2 | `yameng_2/LH_Per_S_DtP/event_01.jpg` | frame 73–158, 2.436–5.268 s | frame 106, 3.54 s, width 0.406 | Clear interior event: no candidate box at frame 64, a left near-field dark strip appears at frame 73, becomes broad and nearly black at frame 106, narrows by frame 158, and is absent at frame 164. The surrounding anatomy stays structured. This supports a localized near-field/angle loss signal, with the usual anisotropy and edge-of-sector alternatives. |
| yameng_2 | `yameng_2/RH_Per_C_PtD/event_02.jpg` | frame 146–152, 4.868–5.068 s | frame 146, 4.87 s, width 0.125 | Very short, right-edge event. The montage shows a narrow dark vertical band at the right boundary that contracts and disappears by frame 160. Because the box is narrow and clipped by the image edge, this is visually marginal and could be edge/sector geometry or a threshold fluctuation. It should be used as a boundary warning, not as proof of detachment. |
| zhongkai | `zhongkai/RH_Per_S_PtD/event_01.jpg` | frame 95–158, 3.168–5.248 s | frame 118, 3.93 s, width 0.312 | Strong localized event. The left boxed shallow strip widens from frame 95 to a broad black band at frame 118, then narrows by frame 158 and is absent at frame 167. Internal muscle and bone structures remain visible outside the box, so this is not a global frame blackout. The deep dark area under the bone is a separate shadow pattern. |
| zhongkai | `zhongkai/LH_Per_S_DtP/event_01.jpg` | frame 0–32, 0.000–1.068 s | frame 0, 0.00 s, width 0.156 | Short boundary event. The candidate is already present in the first frame and is gone by frame 41 (1.36 s), so the start is left-censored and cannot be dated. The boxed strip is shallow and left-sided; most of the deep dark region is beneath a bright bone echo. This is weak evidence of transient coupling loss and is best retained as a boundary case. |

## What the images support

The five longer or clearer cases show the same visual pattern: a dark region
starts at the shallow left or right edge, often expands laterally, and then
contracts while the rest of the image still contains recognizable layered
echoes. That pattern is more compatible with localized coupling, beam-angle
change, anisotropy, or sector-edge loss than with a complete loss of contact.
The selected short events are especially vulnerable to edge clipping and
threshold effects.

The bright curved bone echo and its deep acoustic shadow recur in these
forearm views. A deep shadow alone must not be counted as probe detachment;
the candidate decision needs the shallow boxed region and neighboring tissue
context. These JPEG montages cannot distinguish a true probe lift-off from a
small tilt, pressure redistribution, anisotropy, or a change of tissue under
the beam. They therefore validate the *appearance* of a suspicious dark band,
not the mechanical cause or a physical separation distance.

No medical interpretation is made here. The next causal analysis should use
the event intervals together with force, compensated torque, and measured TCP
angular velocity, retaining the left/right edge cases as low-confidence
labels rather than treating them as confirmed detachment.
