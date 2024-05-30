all: build/LHCb_CPV.pdf

# hier Python-Skripte:
build/plot.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/LHCb_CPV.pdf deklarieren:
build/LHCb_CPV.pdf: build/plot.pdf

build/LHCb_CPV.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	LHCb_CPV.tex
	cp build/LHCb_CPV.pdf LHCb_CPV.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm LHCb_CPV.pdf

FORCE:

.PHONY: all clean
