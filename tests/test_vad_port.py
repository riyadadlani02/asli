"""The browser upload lane runs its own copy of the VAD. This proves the two agree.

The page invites people to point the harness at their own recording, so the number the
browser prints has to be the number `asli real` would print. Two implementations in two
languages is the cost of a static page with no backend; this is the check that keeps
them honest.

    python tests/test_vad_port.py        # needs node and the corpus, skips without them

Skips rather than fails when node or the corpus is absent: it is not part of the
no-keys, no-network suite in test_asli.py.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asli.fit import read_audio  # noqa: E402
from asli.real import _bounds, spec_from_audio  # noqa: E402

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "corpus" / "GV_Dev_5h" / "Audio"
N_FILES = 14

# Extract the ported functions from the built page rather than a copy kept here, so the
# thing under test is the thing that ships.
JS_FROM, JS_TO = "function pctl(", "async function buildUpload("

DRIVER = """
const fs=require('fs');
%(src)s
let bad=0;
for(const c of JSON.parse(fs.readFileSync(process.argv[2],'utf8'))){
  const b=fs.readFileSync(c.pcm);
  const js=specFromAudio(new Int16Array(b.buffer,b.byteOffset,b.byteLength/2),c.rate);
  const py=c.py, probs=[];
  if(!!py!==!!js) probs.push("one side null");
  else if(py){
    for(const k of ["startMs","trueEnd","nPauses","spread"])
      if(py[k]!==js[k]) probs.push(`${k} py=${py[k]} js=${js[k]}`);
    if(JSON.stringify(py.pauses)!==JSON.stringify(js.pauses)) probs.push("pause list");
  }
  // documented divergence: the harness skips a file with no qualifying pause; the
  // browser continues and warns. Assert it tracks exactly that condition.
  if(c.spec_is_none !== (!js||js.pauses.length===0)) probs.push("skip-gate");
  if(probs.length){ bad++; console.log(`MISMATCH ${c.id}: ${probs.join(" | ")}`); }
}
console.log(bad ? `FAIL ${bad}` : "PASS");
"""


def js_source() -> str:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    a, b = html.index(JS_FROM), html.index(JS_TO)
    return "const FRAME_MS=20, MIN_PAUSE_MS=500, TOL_MS=100;\n" + html[a:b]


def main() -> int:
    node = shutil.which("node")
    files = sorted(CORPUS.glob("*.mp3"))[:N_FILES] if CORPUS.is_dir() else []
    if not node or not files:
        print(f"  skip  test_vad_port ({'node not installed' if not node else 'no corpus'})")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for f in files:
            pcm, rate = read_audio(f)
            raw = Path(tmp) / f"{f.stem}.pcm"
            raw.write_bytes(pcm.astype("<i2").tobytes())
            b = _bounds(pcm, rate)
            py = None
            if b is not None:
                start, end, runs = b
                spread = max((abs(alt[1] - end) for alt in
                              (_bounds(pcm, rate, s) for s in (0.5, 2.0)) if alt),
                             default=0)
                py = {"startMs": start, "trueEnd": end, "nPauses": len(runs), "spread": spread,
                      "pauses": [list(r) for r in runs if r[1] - r[0] >= 500]}
            cases.append({"id": f.stem, "rate": int(rate), "pcm": str(raw), "py": py,
                          "spec_is_none": spec_from_audio(pcm, rate, f.stem) is None})

        spec = Path(tmp) / "cases.json"
        spec.write_text(json.dumps(cases))
        drv = Path(tmp) / "drv.js"
        drv.write_text(DRIVER % {"src": js_source()})
        out = subprocess.run([node, str(drv), str(spec)],
                             capture_output=True, text=True)

    if out.returncode or "PASS" not in out.stdout:
        print(out.stdout or out.stderr)
        print(f"\nFAILED — the browser VAD has drifted from asli/real.py")
        return 1
    print(f"  ok  test_vad_port  ({len(cases)} real recordings, every field identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
