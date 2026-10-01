const fs = require("fs");
const d = require("/tmp/claude-0/-home-user-INDIGO/1e8b94bc-4e7a-5be2-8a31-c6e1f5fef990/scratchpad/node_modules/docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, AlignmentType, BorderStyle, PageBreak,
  LevelFormat, convertInchesToTwip,
} = d;

const OUT = "analyses/scaling/results/INDIGO_scaling_meeting.docx";
const R = "analyses/scaling/results/";
const INK = "1a1a1a", MUTED = "5a5a5a", RULE = "c9c9c9";
const HDR_BG = "eef2f7", ALT_BG = "f7f8fa";

const T = (t, o = {}) => new TextRun({ text: t, size: o.size ?? 21, color: o.color ?? INK, italics: o.italics });
const B = (t, o = {}) => new TextRun({ text: t, bold: true, size: o.size ?? 21, color: o.color ?? INK });
const CODE = t => new TextRun({ text: t, font: "Consolas", size: 19, color: INK });
const P = (runs, o = {}) => new Paragraph({
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: 276 },
  children: [].concat(runs).map(t => typeof t === "string" ? T(t) : t),
});
const H1 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1 });
const H2 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2 });
const BULLET = runs => new Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 272 },
  children: [].concat(runs).map(t => typeof t === "string" ? T(t) : t),
});
const NUM = runs => new Paragraph({
  numbering: { reference: "num", level: 0 }, spacing: { after: 80, line: 272 },
  children: [].concat(runs).map(t => typeof t === "string" ? T(t) : t),
});

function img(file, widthIn) {
  const buf = fs.readFileSync(R + file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20), W = widthIn * 96;
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 90, after: 50 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(W), height: Math.round(W * h / w) } })],
  });
}
const CAP = t => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 170 },
  children: [new TextRun({ text: t, size: 17, color: MUTED, italics: true })],
});

function table(head, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, o = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.bg ? { type: ShadingType.CLEAR, fill: o.bg, color: "auto" } : undefined,
    margins: { top: 55, bottom: 55, left: 90, right: 90 },
    children: [new Paragraph({ alignment: o.align, spacing: { after: 0 },
      children: [new TextRun({ text: String(txt), size: 18, bold: o.bold, color: INK })] })],
  });
  return new Table({
    columnWidths: widths, width: { size: total, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "e3e3e3" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({ tableHeader: true, children: head.map((h, i) =>
        cell(h, widths[i], { bold: true, bg: HDR_BG, align: i ? AlignmentType.RIGHT : AlignmentType.LEFT })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) =>
        cell(c, widths[i], { bg: ri % 2 ? ALT_BG : undefined, align: i ? AlignmentType.RIGHT : AlignmentType.LEFT })) })),
    ],
  });
}

const body = [];
body.push(new Paragraph({ spacing: { after: 40 },
  children: [new TextRun({ text: "INDIGO compute-optimal scaling", bold: true, size: 40, color: INK })] }));
body.push(new Paragraph({ spacing: { after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: RULE, space: 8 } },
  children: [new TextRun({ text: "48-run IsoFLOP sweep, 40M corpus, Porian-corrected", size: 24, color: MUTED })] }));

// 1 headline
body.push(H1("1.  Headline"));
body.push(NUM([B("N* ∝ C"), new TextRun({ text: "0.94", superScript: true, size: 21, color: INK }),
  T(" on ΔE₀₀, C"), new TextRun({ text: "0.83", superScript: true, size: 21, color: INK }),
  T(" on CE, 95% CI [+0.81, +1.00]. Chinchilla is 0.50. Examples per parameter falls as C"), new TextRun({ text: "−0.78", superScript: true, size: 21, color: INK }), T("; Chinchilla's is flat.")]));
body.push(NUM([B("Both metrics bottom at C = 2.75e15"), T(" and worsen after. Best pooled ΔE there: 10.04.")]));
body.push(NUM([B("CE is not the well-behaved control."), T(" It breaks at the same budget, so this is not a ΔE artefact.")]));
body.push(NUM([B("Not data repetition."), T(" 47 of 48 runs stayed under one epoch.")]));
body.push(NUM([B("Low chroma is the only clean law"), T(" (α = 0.65, β = 0.40, α + β ≈ 1). Every bucket now fits five of six rungs.")]));
body.push(P([T("Working claim: the ceiling is problem-side. "), B("Section 8 is the one confound I cannot rule out, and what I would run to close it. That is the section I want your read on.")],
  { before: 70, after: 150 }));

// 2 method
body.push(H1("2.  Method"));
body.push(BULLET([T("Chinchilla Approach 2. Six budgets × eight runs. Parabola in log N per budget gives N*, then a power law in C. Final checkpoint, aspect ratio held in a 28 to 72 band, D counted in examples.")]));
body.push(BULLET([B("Compute axis: C = 3 · F(N) · D"), T(", with the forward cost per example F computed analytically in "), CODE("src/scaling/flops.py"), T(". The per-token heuristic does not apply here: the pointer head runs per (position, slot), 352 times per example, and the slot encoder runs over 32 pool slots rather than the 11-position decoder sequence. Over this ladder C/(N·D) falls from 306 to 218, so a per-token count would be off by 36× to 51× and the error itself drifts 1.4× across the grid. This is Porian correction #1, far larger here than in the paper.")]));
body.push(BULLET([T("Porian #2: warmup as a fraction of the budget. Porian #3: LR re-tuned per scale, which is section 8. The rung and power-law estimators are ported from the authors' released code rather than reimplemented from the paper; section 4 and "), CODE("analyses/scaling/METHOD_DIFFS.md"), T(" give the audit.")]));

// 3 sweep
body.push(H1("3.  The sweep"));
body.push(img("isoflop_fit.png", 6.5));
body.push(CAP("Figure 1.  Left: one colour per FLOP budget, dots are the eight model sizes, star is the fitted N*. Right: N*(C) per chroma bucket."));
body.push(BULLET([T("The parabolas flatten as C rises. At C = 8.3e15 the star sits at the edge of the window; at 2.5e16 there is no interior minimum and the fit is refused.")]));
body.push(BULLET([T("The quantity being fitted, curvature in log N, is going to zero. ΔE stops caring how big the model is.")]));
body.push(P([B("Table 1.  Per-budget summary.")], { before: 110, after: 60 }));
body.push(table(["C (FLOPs)", "N sampled", "N* (ΔE)", "D* (Mex)", "best ΔE₀₀", "best CE", "max epochs"],
  [["1.00e14", "0.08 - 1.16M", "0.11M", "3.14", "12.39", "6.144", "0.11"],
   ["3.02e14", "0.18 - 2.09M", "0.34M", "3.22", "10.84", "6.075", "0.14"],
   ["9.10e14", "0.46 - 3.42M", "1.08M", "3.24", "10.46", "6.042", "0.18"],
   ["2.75e15", "0.57 - 5.84M", "2.56M", "4.51", "10.04", "6.024", "0.46"],
   ["8.29e15", "0.97 - 9.82M", "6.51M", "5.58", "10.75", "6.030", "0.84"],
   ["2.50e16", "1.78 - 17.84M", "on edge", "–", "11.40", "6.035", "1.45"]],
  [1500, 1760, 1180, 1100, 1300, 1160, 1360]));
body.push(CAP("N* is the median of a 1000-draw seed-noise bootstrap; best values are seed-averaged. N* moves 61× across the span, D* only 1.8×. The top rung's argmin sits on the boundary and is rejected."));

// 4 CE
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("4.  The estimator, taken from their released code"));
body.push(P([T("Porian et al. publish the analysis code behind the paper, so the estimator here is a port of it rather than a reading of the method section. Three of their choices differ from ours, and the difference is not cosmetic.")]));
body.push(BULLET([B("Akima spline, not a parabola. "), T("They interpolate the rung in log-log space and take N* as the argmin, rejecting it when it lands on the boundary. A parabola cannot decline: fed a flat series it still returns a confident interior minimum.")]));
body.push(BULLET([B("A seed-noise bootstrap whose median is the observation. "), T("A thousand redraws per rung; sigma is their log-space spread, inflated when draws fall off the edge; a rung is dropped only when fewer than half survive. Ours refused the whole fit when the Monte Carlo was unstable, which discarded usable rungs.")]));
body.push(BULLET([B("A 1/\u03C3\u00B2-weighted step-2 fit. "), T("Ours was unweighted, so the least identified rungs pulled as hard as the best.")]));
body.push(img("porian_pooled.png", 6.5));
body.push(CAP("Figure 2.  The same 48 runs under their estimator. A: rungs with the interpolated curve and its argmin; the rejected top rung is marked with a cross. B: N*(C) with bootstrap error bars and confidence region. C: the multiplier law, which they report and we had not."));
body.push(P([B("The number barely moves. Its credibility moves a lot.")], { after: 60 }));
body.push(table(["Pooled \u0394E\u2080\u2080", "parabola (ours)", "Akima (theirs)"],
  [["usable rungs", "4 of 6", "5 of 6"],
   ["\u03B1", "+0.937", "+0.923"],
   ["95% CI", "0.53 to 1.42", "0.81 to 1.00"],
   ["noise-propagated fit", "all 4000 draws refused", "survives"]],
  [3400, 3000, 2960]));
body.push(CAP("The rescued rung is the bottom one, whose parabola vertex fell below the smallest model sampled. The top rung stays rejected, now for a reason that holds up."));
body.push(P([B("Two things their method adds that we had no equivalent of. "), T("The multiplier law, D*/N* against compute, is the Chinchilla headline quantity and ours goes as C"), new TextRun({ text: "\u22120.78", superScript: true, size: 21, color: INK }), T(" where theirs is flat. And a saturating fit, value(C) = power law plus an irreducible floor, which would be the direct measurement of the \u0394E floor section 7 argues for. It comes back "), B("unidentified"), T(": the model only decreases, our curve turns upward at 2.75e15, and four descending budgets cannot pin a three-parameter asymptote. Adding rungs below 1e14, the cheapest runs in the study, would fix that.")]));

body.push(H1("5.  Cross-entropy, same procedure"));
body.push(img("ce_vs_de_scaling.png", 6.5));
body.push(CAP("Figure 3.  The identical IsoFLOP procedure run on val_loss. Panel A carries a Chinchilla α = 0.5 slope for reference."));
body.push(BULLET([T("Both exponents near 0.9. Both metrics turn at 2.75e15.")]));
body.push(BULLET([T("This kills anything that blames the ΔE evaluator, CIEDE2000, or the greedy decode: token-level CE turns in lockstep.")]));

// 5 epochs
body.push(H1("6.  Epoch-ceiling theory, refuted"));
body.push(img("epoch_ceiling_refuted.png", 6.5));
body.push(CAP("Figure 4.  Left: epochs over the 40M corpus, all 48 runs. Right: every model size trained at two or more budgets, N held exactly fixed inside each panel."));
body.push(BULLET([T("Degradation happens "), B("inside the first pass"), T(", so it is not memorisation from repeats.")]));
body.push(BULLET([T("In four of the nine panels, more "), B("unique"), T(" data makes ΔE worse. That is the opposite of a repetition effect.")]));

// 6 ceiling
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("7.  Where the ceiling comes from"));
body.push(P([B("Table 2.  Fitted exponents by chroma bucket.")], { after: 60 }));
body.push(table(["Bucket", "α (N*)", "95% CI", "β (D*)", "α + β", "rungs fit", "D*/N* exp."],
  [["low", "+0.646", "0.336 to 0.818", "+0.405", "1.05", "5 of 6", "−0.241"],
   ["mid", "+0.936", "0.817 to 1.014", "+0.129", "1.07", "5 of 6", "−0.807"],
   ["high", "+0.822", "0.573 to 0.955", "+0.236", "1.06", "5 of 6", "−0.586"],
   ["pooled", "+0.923", "0.814 to 0.999", "+0.140", "1.06", "5 of 6", "−0.783"],
   ["cross-entropy", "+0.813", "0.730 to 0.878", "+0.241", "1.05", "4 of 6", "−0.572"]],
  [1200, 1180, 1900, 1180, 1000, 1200, 1700]));
body.push(CAP("Intervals come from refitting the 1/σ²-weighted law on each bootstrap draw. The last column is the multiplier law, examples per parameter against compute, which Chinchilla puts near zero."));
body.push(P([B("Table 3.  Best ΔE₀₀ by bucket, across the full 250× span.")], { before: 130, after: 60 }));
body.push(table(["Bucket", "1.0e14", "3.0e14", "9.1e14", "2.8e15", "8.3e15", "2.5e16", "best / worst"],
  [["low", "6.86", "6.14", "5.62", "5.05", "5.52", "6.02", "1.36×"],
   ["mid", "15.27", "12.43", "12.16", "12.04", "12.89", "12.39", "1.27×"],
   ["high", "21.51", "18.83", "18.12", "18.79", "20.21", "21.55", "1.19×"]],
  [1180, 1130, 1130, 1130, 1130, 1130, 1130, 1400].slice(0, 8)));
body.push(CAP("High chroma never gets below 18 ΔE, which is not a usable design, and moves 1.19× across 250× in compute."));
body.push(H2("The M factor"));
body.push(BULLET([T("pool_size is drawn per example from [4, 32], stacks are at most 10 layers, thicknesses live on a 2 nm grid. The reachable colour set is "), B("finite and fixed by the pool"), T(", not by the model.")]));
body.push(BULLET([T("A floor flattens the parabola: once the model finds the near-pool-optimal stack, extra N buys nothing and N* stops being identified. That is Figure 1.")]));
body.push(BULLET([T("A floor should be chroma-dependent, and is. Low chroma is the only bucket with a clean, noise-robust law; high chroma is pinned at 18 to 21.")]));
body.push(BULLET([T("So the pooled curve is a "), B("mixture"), T(" of a still-improving component and a pinned one, and mixtures do not have clean power laws.")]));
body.push(H2("The alternative"));
body.push(P([T("Not the pool but the degeneracy: the map from (target Lab, pool) to stack is massively one-to-many, and a point-estimate decoder on a token-level objective cannot rank that set well enough for its argmax to be good. "), B("The two differ on one testable thing: whether enlarging the pool moves the floor.")]));
body.push(P([B("Test: "), T("for ~2000 val examples compute the best ΔE achievable given that example's own pool, using "), CODE("optical_sim.py"), T(" and "), CODE("thickness_optimizer.py"), T(", then plot model ΔE against it by pool_size and chroma. M factor says the gap has already closed at large C; degeneracy says the gap is everything and never closes. A few CPU hours.")]));

// 7 LR
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("8.  Learning rate: what we fit, and what I propose fitting instead"));
body.push(H2("8.1  What the sweep does now"));
body.push(P([T("Porian correction #3 is to re-tune LR at every scale rather than inherit one global value. We apply it through a fitted law rather than a sweep per config: "), CODE("fit_lr_law()"), T(" in "), CODE("src/scaling/configs.py"), T(" fits "), B("log lr = a + b·log N"), T(" to three measured optima, giving "), B("lr(N) = 1.573 · N"), new TextRun({ text: "−0.567", superScript: true, size: 21, color: INK }), T(". Across the ladder's 220× range in N that runs from 2.58e-3 to 1.21e-4. It is a real improvement on the alternative: the production value of 6e-5 left every probe model untrained.")]));
body.push(H2("8.2  What the law rests on"));
body.push(img("lr_law.png", 6.5));
body.push(CAP("Figure 5.  A: the three measured optima, with every LR each grid tried shown hollow. B: the 48 sweep runs in (D, N); a horizontal line joins runs that share an N and therefore share an LR."));
body.push(BULLET([T("Two of the three points come from a "), B("3-point"), T(" grid, 1e-4 to 3e-3, whose only candidates were {1e-4, 5.48e-4, 3e-3}. "), B("Both sizes returned the middle one"), T(", so that measurement carries no size dependence at all.")]));
body.push(BULLET([T("The d512 point comes from a different 4-point grid, 1e-4 to 5e-4, and returned 1.00e-4, which is that grid's lower bound. "), B("The optimum was never bracketed"), T("; the true one may sit below it.")]));
body.push(BULLET([T("So the slope rests entirely on one unbracketed boundary value. Leave-one-out shows the cost: dropping d256 gives b = −0.54, dropping d128 gives b = −0.94, which is "), B("8.4×"), T(" in the LR handed to the smallest model in the sweep.")]));
body.push(H2("8.3  The larger gap: there is no D term"));
body.push(P([T("All three measurements were taken at "), B("one dataset size, D = 614,400 examples"), T(". "), CODE("lr_for()"), T(" takes only N, so LR is constant along lines of constant N (Figure 5B). Runs at a fixed N span up to "), B("9× in D"), T(" (N = 1.47M trains on 2.5M, 7.4M and 22.4M examples, all at lr = 4.98e-4), and the largest D in the sweep is "), B("95×"), T(" the tuned one.")]));
body.push(P([T("The repo already asserts the opposite premise. "), CODE("scripts/fit_lr_scaling.py"), T(" opens with: \"optimal LR depends primarily on "), B("the total number of optimization steps"), T("\", which is D / batch_size, and fits against "), CODE("train_examples"), T(". "), CODE("configs.py"), T(" fits against parameter count. Two different laws; we deployed the N one.")]));
body.push(P([B("Why this is live rather than pedantic. "), T("Within a rung C is fixed, so N and D are anti-correlated and a missing D term is largely absorbed by the N term. "), B("Across rungs it is not"), T(": the same N appears at three different D. If the optimum falls with D, as the cosine-schedule argument says, then at fixed N the large-budget runs train at too high an LR and the error grows with the budget. That would produce exactly what we see: degradation switching on at high C, on both metrics, unrelated to repetition.")]));
body.push(P([T("Probably not the whole story, since section 7's clean low-chroma law is hard to get out of it. But it is the only remaining explanation I cannot rule out, and it is cheap to rule out.")]));
body.push(H2("8.4  Proposed experiment"));
body.push(P([T("Replace the 1-D law with "), B("lr_opt(N, D) = A · N"), new TextRun({ text: "b", superScript: true, size: 21, color: INK }), B(" · D"), new TextRun({ text: "c", superScript: true, size: 21, color: INK }), T(", fitted where both axes move independently.")]));
body.push(BULLET([B("Minimal, 3 sweeps, ~37 GPU-hours. "), T("Hold N at d128/se2 (0.77M, already measured) and run "), CODE("lr_tuning.py --epochs 1"), T(" at D ∈ {0.6M, 4M, 24M}, bracketing the sweep's real D range. Measures c directly. If |c| is inside noise of zero, the current law is fine and we say so.")]));
body.push(BULLET([B("Full, 3×3 grid, 9 sweeps, ~110 GPU-hours. "), T("Only if c is clearly nonzero. Gives b and c jointly and replaces "), CODE("MEASURED_LR"), T(".")]));
body.push(BULLET([B("Either way, widen the grid and require bracketing. "), CODE("--n-lrs 6 --lr-min 1e-5 --lr-max 5e-3"), T(", and reject any result landing on an endpoint.")]));
body.push(P([B("The question: "), T("is ~1.5 GPU-days worth spending to close this before we commit to \"the ceiling is problem-side\"? My read is yes for the 3-sweep version, since it is the last mechanism that could produce budget-dependent degradation, and a measured c ≈ 0 lets us state the conclusion cleanly. The 9-sweep version only if the cheap one comes back nonzero.")], { before: 80 }));

// 8 caveats
body.push(H1("9.  Caveats"));
body.push(BULLET([T("Five of six rungs are usable and the pooled interval is [+0.81, +1.00]. Under the parabola only four were usable and the interval was [+0.53, +1.42], so treat any earlier quote of that spread as superseded.")]));
body.push(BULLET([T("The pooled law rests on five points. The top rung is genuinely unusable: its interpolated argmin sits on the boundary of the sampled range.")]));
body.push(BULLET([T("Largest model here is 17.8M. The turn could be an artefact of the size window; the counter is that the same window gave a clean low-chroma law.")]));
body.push(BULLET([T("Quote 10.04, not 9.17, at 2.75e15. The min over eight runs is biased low; 9.17 is one seed of a three-seed cluster averaging 10.04.")]));
body.push(BULLET([T("LR has no D term (section 8). Unablated.")]));

// 9 takeaway
body.push(H1("10.  Takeaway"));
body.push(NUM([T("48 runs, Porian-corrected, aspect-controlled, 40M corpus, analytic FLOP axis.")]));
body.push(NUM([T("INDIGO's frontier is not Chinchilla's. Extra compute goes into parameters, not data.")]));
body.push(NUM([T("Past C ≈ 2.75e15 more compute makes the model worse, on both metrics.")]));
body.push(NUM([T("Not repetition, not the metric. One confound left: no D term in the LR law.")]));
body.push(NUM([T("Low chroma is the only clean, noise-robust law. The method works where the problem is solvable.")]));
body.push(NUM([T("If that holds, the payoff is inference-time search and RL against the simulator, not a bigger pretrain. Two cheap experiments decide it: the oracle-floor eval (section 7) and the D-term LR sweep (section 8).")]));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 28, bold: true, color: INK },
        paragraph: { spacing: { before: 300, after: 130 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 5 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 23, bold: true, color: "33475b" },
        paragraph: { spacing: { before: 230, after: 95 } } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.2) } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.25) } } } }] },
  ] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 },
    margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: body }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote " + OUT); });
