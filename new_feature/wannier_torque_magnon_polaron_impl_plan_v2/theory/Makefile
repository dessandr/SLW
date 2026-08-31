PDF=wannier_torque_formalism.pdf

all: $(PDF)

$(PDF): main.tex sections/*.tex references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=wannier_torque_formalism main.tex

clean:
	latexmk -C -jobname=wannier_torque_formalism main.tex
