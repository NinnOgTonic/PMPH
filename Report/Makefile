TARGET=PMPH-project.pdf
TEXFILES=*.tex
HELPFILES=

all: $(TARGET)

$(TARGET): $(TEXFILES) Makefile

clean:
	rm -rf *.blg *.mapleout *.bbl *.aux *.log *.out *.toc *.eps *.data *.o *.hi *~ $(TARGET)

%.pdf: %.tex
	pdflatex $*.tex
	pdflatex $*.tex
