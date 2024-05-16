all: build/SiliconStrip.pdf

# hier Python-Skripte:
build/leakage.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/SiliconStrip.pdf deklarieren:
build/SiliconStrip.pdf: build/leakage.pdf

build/SiliconStrip.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	SiliconStrip.tex
	cp build/SiliconStrip.pdf SiliconStrip.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm SiliconStrip.pdf

FORCE:

.PHONY: all clean
