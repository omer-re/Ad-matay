#!/bin/bash

# HOW TO USE:
# chmod +x install_packages_with_versions.sh
# ./install_packages_with_versions.sh

# Update the package list to ensure the latest package info is available
sudo apt-get update

# Loop through each line in the packages_with_versions.txt file
while read -r package; do
    # Check if the package is already installed and matches the desired version
    package_name=$(echo "$package" | cut -d '=' -f 1)
    package_version=$(echo "$package" | cut -d '=' -f 2)

    # Check if the exact version is already installed
    installed_version=$(dpkg-query -W --showformat='${Version}\n' "$package_name" 2>/dev/null)

    if [ "$installed_version" = "$package_version" ]; then
        echo "$package_name is already installed with version $package_version."
    else
        echo "Installing $package_name version $package_version"
        sudo apt-get install -y "$package"
    fi
done < packages_with_versions.txt

echo "All specified packages have been installed!"
