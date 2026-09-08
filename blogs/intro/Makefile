PANDOC ?= pandoc
SOURCE := blogpost.md
CSS := style.css
OUTPUT_DIR := build
HTML := $(OUTPUT_DIR)/blogpost.html
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
	@mkdir -p $(OUTPUT_DIR)/figures
	@cp -R figures/. $(OUTPUT_DIR)/figures/
	@cp $(CSS) $(OUTPUT_DIR)/$(CSS)
	$(PANDOC) $(SOURCE) \
		--from=markdown+tex_math_dollars \
		--to=html5 \
		--standalone \
		--css=$(CSS) \
		--math-method=mathjax \
		--citeproc \
		--output=$(HTML)

clean:
	rm -rf $(OUTPUT_DIR)
