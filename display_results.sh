#!/bin/bash

# Simple script to visualize RGB values from rgb_output.txt

# Check if the file exists
if [ ! -f "rgb_output.txt" ]; then
    echo "Error: rgb_output.txt not found"
    exit 1
fi

# Skip the header line
tail -n +2 rgb_output.txt | while read line; do
    # Extract RGB values
    r=$(echo $line | awk '{print $1}')
    g=$(echo $line | awk '{print $2}')
    b=$(echo $line | awk '{print $3}')
    
    # Create ANSI color code for background
    # Convert RGB (0-255) to terminal color (0-5)
    r_term=$(( $r * 6 / 256 ))
    g_term=$(( $g * 6 / 256 ))
    b_term=$(( $b * 6 / 256 ))
    
    # Calculate the 216 color cube index (16 + 36*r + 6*g + b)
    color=$(( 16 + 36*$r_term + 6*$g_term + $b_term ))
    
    # Print a colored block
    echo -e "\e[48;5;${color}m          \e[0m RGB: $r, $g, $b"
    
    # Sleep briefly to simulate animation
    sleep 0.1
done

echo "Visualization complete!" 