# .latexmkrc

# Output directory
$out_dir = 'out';
$aux_dir = 'out';

# Create out directory automatically
unless (-d $out_dir) {
    mkdir $out_dir or die "Cannot create directory '$out_dir': $!";
}

# For Japanese LaTeX: platex + dvipdfmx
$latex = 'platex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=%O %S';
$bibtex = 'pbibtex %O %B';
$dvipdf = 'dvipdfmx -o %D %S';

# Build PDF via DVI
$pdf_mode = 3;

# Clean targets
$clean_ext = 'synctex.gz bbl blg aux log dvi fls fdb_latexmk toc out';