all: build/SciFi.pdf

# hier Python-Skripte:

build/lights_on1.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

content/pics/r_exit_cut.pdf: simulation.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python simulation.py

content/pics/rmin_sim_core.pdf: simulation_plots.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python simulation_plots.py


# hier weitere Abhängigkeiten für build/SciFi.pdf deklarieren:
build/SciFi.pdf: build/lights_on1.pdf content/pics/r_exit_cut.pdf content/pics/rmin_sim_core.pdf

build/SciFi.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	SciFi.tex
	cp build/SciFi.pdf SciFi.pdf
#	open "https://www.youtube.com/watch?v=xvFZjo5PgG0"

build:
	mkdir -p build

clean:
	rm -rf build
	rm SciFi.pdf
FORCE:

.PHONY: all clean
