# Wellness shshacks2023

You sad? Well download Wellness onto your phone and we'll travel through the journey of mental health together.

## Frontend
We mainly used Flutter and Dart made by Google to code the mobile apps, along side Swift for iOS and Java for android.
Android Studio, Xcode, Visual Studio Code were the main IDEs used.
We would like to thank the creators of camerawesome for the package used to record videos, the creators for awesome_notifications for the package used for notifications.

## Backend
The backend was created using python and javascript. Node.js was used to download videos from the firebase storage while python acted as the web server that converted the video to audio using Google's speech to text API, our own sentimental model, and spacy.


## AI
We self-trained a sentimental modal that takes text and outputs a value between 0-1, 0 being negative and 1 being positive. We would like to give our thanks to [Bentrevett](https://github.com/bentrevett/pytorch-sentiment-analysis) for providing a base template where we updated the code and added our own dataset. Here's our model: [Google Drive File](https://drive.google.com/file/d/1dZK4sldxCtd6yKOhDxo5REok0K7CsUxa/view?usp=sharing)
