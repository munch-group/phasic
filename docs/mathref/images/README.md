for F in *.pdf; do sips -s format png $F --out "$(basename $F .pdf).png" ; done
ls

for F in *.pdf; do pdf2svg $F "$(basename $F .pdf).svg" ; done
