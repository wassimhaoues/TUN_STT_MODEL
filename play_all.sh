#!/bin/bash

while read -r file; do
    echo "Playing: $file"
    ffplay -nodisp -autoexit "$file"
    sleep 1
done < file_list.txt
