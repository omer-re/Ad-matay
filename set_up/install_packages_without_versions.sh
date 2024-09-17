#!/bin/bash

# HOW TO USE:
# chmod +x install_packages_with_versions.sh
# ./install_packages_without_versions.sh

# Update the package list to ensure the latest package info is available
sudo apt-get update

# Loop through each package in the packages.txt file
while read -r package; do
    # Check if the package is already installed
    if dpkg -s "$package" >/dev/null 2>&1; then
        echo "$package is already installed."
    else
        echo "Installing $package"
        sudo apt-get install -y "$package"
    fi
done < packages.txt

echo "All specified packages have been installed!"
