#!/bin/bash
# Review 2, plan v3: gate_compare.py on doctored result directories.
GC=/Users/kmt/phasic/.git/claude-scratch/gate_compare.py
W=$(dirname "$0")/r6_work; rm -rf "$W"; mkdir -p "$W"
NEW=tests/pytest/test_new.py
xml() { # dir file "name:outcome ..."
  n=$(echo "$2" | tr '/' '_'); cls=$(echo "${2%.py}" | tr '/' '.')
  { echo '<testsuites><testsuite>'
    for t in $3; do nm=${t%%:*}; o=${t##*:}
      case $o in passed) echo "<testcase classname=\"$cls\" name=\"$nm\" time=\"1\"/>";;
        failed) echo "<testcase classname=\"$cls\" name=\"$nm\" time=\"1\"><failure/></testcase>";;
        skipped) echo "<testcase classname=\"$cls\" name=\"$nm\" time=\"1\"><skipped/></testcase>";;
        error) echo "<testcase classname=\"$cls\" name=\"$nm\" time=\"1\"><error/></testcase>";; esac
    done; echo '</testsuite></testsuites>'; } > "$1/$n.xml"; }
mk() { # dir ; then lines "file rc tests" via add
  mkdir -p "$1"; : > "$1/status.tsv"; touch "$1/DONE"; }
add() { printf '%s\t%s\t%s\n' "$2" "$3" "summary" >> "$1/status.tsv"; [ -n "$4" ] && xml "$1" "$2" "$4"; }
base() { mk "$1"; add "$1" tests/pytest/test_a.py 0 "t1:passed t2:passed"; add "$1" tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"; }
run() { echo "=== $1 (expect $2)"; shift; shift; python3 $GC "$@" | grep -E "^(FAIL|GATE|unusable|note)" ; }

base $W/b
base $W/c1; add $W/c1 $NEW 0 "n1:passed n2:passed"
run "clean, new file passes" PASS $W/b $W/c1 --new-file $NEW
base $W/c2; add $W/c2 $NEW 1 "n1:passed n2:failed"
run "new test fails" FAIL $W/b $W/c2 --new-file $NEW
base $W/c3; add $W/c3 $NEW 0 "n1:passed n2:skipped"
run "new test skipped" FAIL $W/b $W/c3 --new-file $NEW
run "new file not declared" FAIL $W/b $W/c1
run "declared new file absent" FAIL $W/b $W/b --new-file $NEW
base $W/c4; add $W/c4 $NEW 5 ""; xml $W/c4 $NEW ""
run "new file collects zero tests (exit 5)" "FAIL wanted" $W/b $W/c4 --new-file $NEW
base $W/c5; add $W/c5 $NEW 2 ""
run "new file collection error (exit 2, no xml)" FAIL $W/b $W/c5 --new-file $NEW
mk $W/b6; add $W/b6 tests/pytest/test_a.py 137 ""; add $W/b6 tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"
run "baseline file killed, candidate fine" FAIL $W/b6 $W/c1 --new-file $NEW
mk $W/c7; add $W/c7 tests/pytest/test_a.py 137 ""; add $W/c7 tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"; add $W/c7 $NEW 0 "n1:passed"
run "unusable both sides, no ack" FAIL $W/b6 $W/c7 --new-file $NEW
run "unusable both sides, acked" PASS $W/b6 $W/c7 --new-file $NEW --ack-unusable tests/pytest/test_a.py
mk $W/c8; add $W/c8 tests/pytest/test_a.py 2 ""; add $W/c8 tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"; add $W/c8 $NEW 0 "n1:passed"
run "unusable both sides, different exit codes, acked" FAIL $W/b6 $W/c8 --new-file $NEW --ack-unusable tests/pytest/test_a.py
base $W/c9; add $W/c9 $NEW 0 "n1:passed"; rm $W/c9/DONE
run "candidate without DONE" FAIL $W/b $W/c9 --new-file $NEW
mk $W/c10; add $W/c10 tests/pytest/test_a.py 1 "t1:passed t2:failed"; add $W/c10 tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"; add $W/c10 $NEW 0 "n1:passed"
run "regression" FAIL $W/b $W/c10 --new-file $NEW
mk $W/c11; add $W/c11 tests/pytest/test_a.py 0 "t1:passed"; add $W/c11 tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"; add $W/c11 $NEW 0 "n1:passed"
run "test disappeared" FAIL $W/b $W/c11 --new-file $NEW
mk $W/c12; add $W/c12 tests/pytest/test_a.py 0 "t1:passed t2:passed"; add $W/c12 $NEW 0 "n1:passed"
run "file missing in candidate" FAIL $W/b $W/c12 --new-file $NEW
mk $W/c13; add $W/c13 tests/pytest/test_a.py 0 "t1:passed t2:passed t9:failed"; add $W/c13 tests/pytest/test_b.py 1 "u1:passed u2:failed u3:skipped"; add $W/c13 $NEW 0 "n1:passed"
run "new failing test id in an EXISTING file" FAIL $W/b $W/c13 --new-file $NEW
# gate_rerun.sh line replacement logic (no pytest run): emulate its status edit
cp -r $W/c7 $W/c14; f=tests/pytest/test_a.py
grep -v "^$f	" $W/c14/status.tsv > $W/c14/status.tmp; mv $W/c14/status.tmp $W/c14/status.tsv; add $W/c14 $f 0 "t1:passed t2:passed"
echo "=== rerun-style status edit keeps other lines: $(wc -l < $W/c14/status.tsv) lines, DONE kept: $(ls $W/c14/DONE >/dev/null 2>&1 && echo yes)"
rm -rf "$W"
