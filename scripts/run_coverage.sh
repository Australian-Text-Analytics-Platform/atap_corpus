#!/bin/zsh

if [[ $(dirname $0) != "./scripts" ]]; then
  echo "-- Please run this script from project root."
  exit 1
fi


echo "PYTHON=$(which python3) version=$(python3 --version | awk '{print $2}')"

echo "++ Running coverage..."
coverage run -m unittest discover -s tests/
coverage report
rm .coverage  # clean up generated artifact.