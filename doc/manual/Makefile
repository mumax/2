all: manual.pdf


.PHONY: manual.pdf
manual.pdf: mumax2-api.tex flags.tex *.tex
	pdflatex --halt-on-error manual.tex
	makeindex manual
	pdflatex --halt-on-error manual.tex

.PHONY: mumax2-api.tex
mumax2-api.tex:
	ln -sf ../../src/python/mumax2-api.tex .
	./texgen.sh

.PHONY: flags.tex
flags.tex:
	mumax2 -h 2> flags.tex

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ilg *.log *.toc *.out manual.pdf *.idx *.ind modules
