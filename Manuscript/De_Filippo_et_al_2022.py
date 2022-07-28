from docxtpl import DocxTemplate
from Manuscript.abstract import abstract
from Manuscript.results import results
from Manuscript.discussion import discussion
from Manuscript.introduction import introduction
from Manuscript.methods import methods
from Figures.legends import legends
from Utils.Settings import manuscript_folder
# to work with citations use {Abi-Saab, 1999 #888}. A Endnote travelling library is provided in the manuscript folder.

doc = DocxTemplate(f"{manuscript_folder}/Manuscript_template.docx")

title = "Differential ripple propagation along the hippocampal longitudinal axis"
authors = "Roberto De Filippo¹ and Dietmar Schmitz¹"
affiliations = "¹ Charité Universitätsmedizin Berlin, corporate member of Freie Universität Berlin, Humboldt-Universität" \
               " zu Berlin,and Berlin Institute of Health; Neuroscience Research Center, 10117 Berlin, Germany"
correspondence_to = "roberto.de-filippo@charite.de and dietmar.schmitz@charite.de"
keywords = "Hippocampal ripples, Ripples propagation, Anisotropy"

acknowledgements = "This work was supported by the Bundesministerium for Bildung und Forschung (SFB1315-327654276) grant.  " \
                "We thank J.T. Tukker, N. Maier for feedback on an early version of the manuscript and the members of the Schmitz lab for " \
                   "scientific discussion. We thank Willy Schiegel and Tiziano Zito for technical expertise in cluster computing. We thank Federico Claudi " \
                   "for support with brainrender. " \
                "The authors declare that they have no competing interests. "

contributions = "Conceptualization, data curation, formal analysis, investigation, visualization:  RDF. Writing - original draft: RDF. " \
                "Writing - review & editing: RDF, DS. " \
                "Funding acquisition: DS."

data_availability = "All the code used to process the dataset is available at https://github.com/RobertoDF/De-Filippo-et-al-2022, pre-computed data structures "\
                    "can be downloaded at 10.6084/m9.figshare.20209913. "\
                    "All figures and text can be reproduced using code present in this repository, each number present in the text is directly "\
                    "linked to a python data structure. The original dataset is provided by the Allen Institute and available at "\
                    "https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html."

context = {'title': title, "authors": authors, "affiliations": affiliations, "correspondence_to": correspondence_to,
           "keywords": keywords, "abstract": abstract, "introduction": introduction,
           "discussion": discussion, "methods": methods, "results": results, "legends": legends, "acknowledgements": acknowledgements,
           "contributions": contributions, "data_availability": data_availability}

doc.render(context, autoescape=True)

doc.save(f"{manuscript_folder}/De Filippo et al., 2022.docx")

