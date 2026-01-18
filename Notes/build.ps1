# Close Adobe Acrobat (Reader or Acrobat DC)
Get-Process -Name "AcroRd32","Acrobat" -ErrorAction SilentlyContinue | Stop-Process -Force

pdflatex FractalShark.tex
bibtex FractalShark
pdflatex FractalShark.tex
pdflatex FractalShark.tex

# Open the PDF
Start-Process .\FractalShark.pdf
