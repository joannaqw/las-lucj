#!/bin/bash

file="$1"

awk '/Iteration [0-9]+ took/ {
    sum += $(NF-1)
    count += 1
}
END {
    if (count > 0) {
        printf "File: %s\n", FILENAME
        printf "Number of iterations: %d\n", count
        printf "Total time: %.4f s\n", sum
        printf "Average time per iteration: %.4f s\n\n", sum/count
    }
}' "$file"
