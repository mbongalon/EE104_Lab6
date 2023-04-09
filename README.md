The python programs require the following libraries to be imported:
-keras
-numpy
-os
-ssl
-tensorflow
-matplotlib
-pathlib
-certifi
-random
-time
-openai
-flask
-dotenv

Instructions for using the cnn.py program:
1. Set the desired number of convolutional filters, weight decay, batch size, and epochs.
2. Run the code to score the accuracy and save the model to a .h5 file.

Instructions for using the test_image.py program:
1. Make sure the program is in the same location as the saved .h5 model and cnn.py.
2. Load in the .h5 model.
3. Change the URL variables to images that are within the airplane, automobile, bird, cat, deer,
   dog, frog, horse, ship, and truck classes.
4. Run the code to compare the .h5 model predictions to the actual results.

Instructions for using the garden.py program:
1. Run the game from the command prompt.
2. Use the arrow keys to move the cow and the SPACEBAR to water the flowers.
3. The game ends if the cow collides with a fangflower.
4. The game ends if a flower remains wilted for more than 10 seconds.

Instructions for using the app.py program:
1. Update the .env file with your OpenAI API key.
2. Create and activate the virtual environment.
3. Install the necessary modules from requirements.txt.
4. Type "flask run" in the command prompt and go to http://localhost:5000.
5. Enter a person, animal, or object to generate four names.

Instructions for using the chatgpt_query.py program:
1. Update the .env file with your OpenAI API key.
2. Set the desired engine, max_tokens, n, stop, and temperature.
3. Run the code and enter a prompt to query chatgpt.
