#!/bin/zsh

if [[ $(dirname $0) != "./scripts" ]]; then
  echo "-- Please run this script from project root."
  exit 1
fi


echo "PYTHON=$(which python3) version=$(python3 --version | awk '{print $2}')"

echo "++ Running all unit tests..."
python3 -m unittest discover -s tests/
