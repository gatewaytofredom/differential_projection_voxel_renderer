#!/bin/bash

# Script to collate all Rust source files into a single text file
OUTPUT_FILE="rust_project_collated.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Find all .rs files (excluding target directory and other build artifacts) and process them
find . -name "*.rs" -type f \
    -not -path "./target/*" \
    -not -path "./.cargo/*" \
    -not -path "*/node_modules/*" | sort | while read -r file; do
    echo "========================================" >> "$OUTPUT_FILE"
    echo "File: $file" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Collation complete! Output saved to $OUTPUT_FILE"
echo "Total files processed: $(find . -name "*.rs" -type f -not -path "./target/*" -not -path "./.cargo/*" -not -path "*/node_modules/*" | wc -l)"
