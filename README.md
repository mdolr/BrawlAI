# Brawl-AI
A weekend project to try reinforcement learning with an AI that learns to play [Brawlhalla](https://www.brawlhalla.com/) (a PvP fighting game), the bot is set to train on a custom map with a black background, white platforms (see custom assets). The bot can train in practice mode thanks to a custom environment that plays directly using a screen capture as inputs and the keyboard as output. 

## Requirements
* Python v3.8+
* Tensorflow
* Windows & Brawlhalla
* Adjust the screen capture settings in `custom_env/image_capture.py` to your setup

## Run
`python main.py`
