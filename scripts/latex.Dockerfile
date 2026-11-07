FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-xetex texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-science latexmk fonts-dejavu-core \
    fonts-texgyre poppler-utils ca-certificates \
    && apt-get clean
WORKDIR /work/report/latex
ENTRYPOINT ["latexmk", "-xelatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", "main.tex"]
