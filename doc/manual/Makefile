all: manual.pdf


manual.pdf: mumax2-api.tex *.tex
	pdflatex --halt-on-error manual.tex
	pdflatex --halt-on-error manual.tex
	pdflatex --halt-on-error manual.tex

mumax2-api.tex:
	ln -sf ../../src/python/mumax2-api.tex .


.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ind *.ilg *.log *.toc *.out manual.pdf *.idx
