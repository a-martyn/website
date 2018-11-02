# Convert the markdown files in markdown/standalone to html files
MD_FILES="$(ls markdown)"
for filename_with_ext in $MD_FILES; do
    if [ "$filename_with_ext" = "index.md" ]; then
        split=${filename_with_ext%.md}
        filename_wo_ext=${split##*/}
        pandoc markdown/"$filename_with_ext" -f markdown -t html -s -o html/${filename_wo_ext}.html --css ../css/markdown.css
    else
        split=${filename_with_ext%.md}
        filename_wo_ext=${split##*/}
        pandoc markdown/index.md markdown/"$filename_with_ext" -f markdown -t html -s -o html/${filename_wo_ext}.html --css ../css/markdown.css
    fi  
done