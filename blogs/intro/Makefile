PANDOC ?= pandoc
SOURCE := blogpost.md
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

$(HTML): $(SOURCE) references.bib $(FIGURES)
	@mkdir -p $(OUTPUT_DIR)/figures
	@cp -R figures/. $(OUTPUT_DIR)/figures/
	$(PANDOC) $(SOURCE) \
		--from=markdown+tex_math_dollars \
		--to=html5 \
		--standalone \
		--mathjax \
		--citeproc \
		--output=$(HTML)

clean:
	rm -rf $(OUTPUT_DIR)
