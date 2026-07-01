#!/bin/bash
set -e

if [ ! -d "tennis_atp-master" ]; then
    echo "Downloading ATP match data from Jeff Sackmann's repo..."
    git clone https://github.com/JeffSackmann/tennis_atp tennis_atp-master
    echo "Done."
else
    echo "tennis_atp-master/ already exists, skipping download."
fi

echo "Installing dependencies..."
pip install -r requirements.txt
echo "Setup complete. Run: python model_trainer.py && python main.py"
