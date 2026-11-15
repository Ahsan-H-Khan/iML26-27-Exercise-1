"""Package rendered slides into the required five-page PDF (pip install reportlab)."""

from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

out = Path("presentation/presentation.pdf")
canvas = Canvas(str(out), pagesize=(960, 540))
canvas.setTitle("Cognitive-Structural Complexity - iML Exercise 1")
for number in range(1, 6):
    canvas.drawImage(str(out.parent / "previews" / f"slide-{number}.png"), 0, 0, width=960, height=540)
    canvas.showPage()
canvas.save()
print(out)
