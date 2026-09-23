.PHONY: all main sync-figures clean tables supp verify manifest bundle submission

all: main supp

main: sync-figures
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

sync-figures:
	mkdir -p figures
	cp ../outputs/reports/20260923_review_figures/method_pipeline.pdf figures/method_pipeline.pdf
	cp ../outputs/reports/20260923_review_figures/selective_correspondence_overview.pdf figures/selective_correspondence_overview.pdf
	cp ../outputs/reports/20260923_review_figures/failure_taxonomy.pdf figures/failure_taxonomy.pdf

tables:
	conda run -n worldcache python ../scripts/generate_supplementary_tables.py --root .. --output-dir .

supp: tables
	pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex

verify: submission bundle
	conda run -n worldcache python ../scripts/verify_submission_consistency.py --root ..

manifest:
	conda run -n worldcache python ../scripts/generate_submission_manifest.py --root ..

bundle: manifest
	conda run -n worldcache python ../scripts/build_submission_bundle.py --root ..

submission: all
	conda run -n worldcache python ../scripts/build_anonymous_submission.py --root ..

clean:
	rm -f main.aux main.log main.out main.pdf supplementary.aux supplementary.log supplementary.out supplementary.pdf generated_*.tex
