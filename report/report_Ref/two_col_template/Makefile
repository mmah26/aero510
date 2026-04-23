all : main
	rm -f *.aux *.log *.dvi *.blg *.toc *.bbl *~ *.out

main : main.tex
	pdflatex main 
	pdflatex main
	bibtex main
	pdflatex main 
	pdflatex main 
