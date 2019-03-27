# Convert the markdown files to html files
MD_FILES="$(ls *.md)"
for filename_with_ext in $MD_FILES; do
    if [ "$filename_with_ext" = "index.md" ]; then
        split=${filename_with_ext%.md}
        filename_wo_ext=${split##*/}
        pandoc "$filename_with_ext" -f markdown -t html -s -o html/${filename_wo_ext}.html --css ../css/markdown.css
    else
        split=${filename_with_ext%.md}
        filename_wo_ext=${split##*/}
        pandoc index.md "$filename_with_ext" -f markdown -t html -s -o html/${filename_wo_ext}.html --css ../css/markdown.css
    fi  
done

# Convert the jupyter notebook files to html files
NB_FILES="$(ls *.ipynb)"
for filename_with_ext in $NB_FILES; do
    jupyter nbconvert --template full --output-dir='./html' "$filename_with_ext"  
done