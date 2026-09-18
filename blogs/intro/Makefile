PANDOC ?= pandoc
PDFTOPPM ?= pdftoppm
SOURCE := blogpost.md
CSS := style.css
HTML := blogpost.html
FIGURES := $(wildcard figures/*)

.PHONY: all html clean check-pandoc

all: html

html: check-pandoc $(HTML)

check-pandoc:
	@command -v $(PANDOC) >/dev/null 2>&1 || { \
		echo "Pandoc is required to build the blog post."; \
		exit 1; \
	}

$(HTML): $(SOURCE) references.bib $(CSS) $(FIGURES)
	$(PANDOC) $(SOURCE) \
		--from=markdown+tex_math_dollars \
		--to=html5 \
		--standalone \
		--toc \
		--toc-depth=2 \
		--css=$(CSS) \
		--math-method=mathjax \
		--citeproc \
		--output=$(HTML)

figures/quailb_throughput.png: figures/quailb_throughput.pdf
	$(PDFTOPPM) -f 1 -singlefile -r 300 -png $< figures/quailb_throughput

clean:
	rm -f $(HTML)
