CONVERSION INSTRUCTIONS
=======================

Since direct conversion tools (like pandoc or latex) are not available in this environment, this directory has been prepared to facilitate easy conversion to DOCX.

Files included:
1. Paper_for_Word.tex: A single flattened LaTeX file containing the entire paper content (all \input files merged).
2. references.bib: The bibliography file.
3. figures/: Directory containing all required images.

How to Convert to DOCX:

Option 1: Online Converters (Recommended)
-----------------------------------------
1. Zip the entire content of this `docx_conversion` directory.
2. Upload the zip file to an online LaTeX to Word converter such as:
   - Overleaf (Open project -> Menu -> Download -> Word/Docx, if available)
   - CloudConvert (https://cloudconvert.com/tex-to-docx)
   - GrindEQ (http://www.grindeq.com/)

Option 2: Using Pandoc (If installed locally)
---------------------------------------------
If you have Pandoc installed on your local machine, run the following command in this directory:

   pandoc Paper_for_Word.tex -o Paper_Weather_Nowcasting.docx --bibliography=references.bib --citeproc

Option 3: Manual Import
-----------------------
You can open `Paper_for_Word.tex` in a text editor and copy-paste the content into Word, though formatting will be lost. The flattened structure makes this easier than copying from multiple files.
