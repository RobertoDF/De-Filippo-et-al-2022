from docxtpl import DocxTemplate
from Figures.legends_supplementary import legends_supplementary
from Utils.Settings import manuscript_folder

doc = DocxTemplate(f"{manuscript_folder}/Manuscript_supplementary_template.docx")
context = {"legends": legends_supplementary}
doc.render(context)
doc.save(f"{manuscript_folder}/De Filippo et al., 2022_supplementary.docx")

