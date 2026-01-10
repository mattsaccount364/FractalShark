# Close Adobe Acrobat (Reader or Acrobat DC)
Get-Process -Name "AcroRd32","Acrobat" -ErrorAction SilentlyContinue | Stop-Process -Force

pdflatex notes.tex
bibtex notes
pdflatex notes.tex
pdflatex notes.tex

# Open the PDF
Start-Process .\notes.pdf
