#!/usr/bin/env perl

# pdfLaTeX
$pdflatex = 'pdflatex -synctex=1 -halt-on-error -file-line-error %O %S';
$pdf_mode = 1;
$max_repeat = 15;

# BibTeX
$bibtex = 'bibtex %O %S';
$out_dir = 'out';

# preview
$pvc_view_file_via_temporary = 0;
if ($^O eq 'linux') {
    $pdf_previewer = "xdg-open %S";
} elsif ($^O eq 'darwin') {
    $pdf_previewer = "open %S";
} else {
    $pdf_previewer = "start %S";
}

# clean up
$clean_full_ext = "%R.synctex.gz";