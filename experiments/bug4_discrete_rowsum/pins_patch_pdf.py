"""Bug 4 §2.6 step 3: refresh ONLY the 'pdf' field of each case record from
the generator. The other differing fields are documented intended changes of
the discretize refactor (EXPECTED_CHANGES / ROUNDTRIP_EXPECTED_CHANGES): the
records deliberately keep their old values and the pin tests assert that, so
a whole-file regeneration would be wrong. They are identical on the master
build (pins_diff_master.log) and are not touched."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "pytest"))
import _discretize_pins as pins
old = json.load(open(pins.RECORDS))
new = json.loads(json.dumps(pins.build_records(), sort_keys=True))
n = 0
for name, rec in old["cases"].items():
    if "pdf" in rec and rec["pdf"] != new["cases"][name]["pdf"]:
        rec["pdf"] = new["cases"][name]["pdf"]
        n += 1
with open(pins.RECORDS, "w") as f:
    json.dump(old, f, indent=1, sort_keys=True)
print("pdf records refreshed:", n)
