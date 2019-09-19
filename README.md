# Intel-Hackathan

Objective: To develop a solution which can provide live support to customer representatives.

Application: There are times when we call customer care for a problem. Not all representatives are very well trained, they might not be able to provide us a perfect solution. Over the call, they might search for the problems customer is concerned about but it’s a time consuming process. In order to help them come up with a good solution in minimal real time, this application come into existence. This application will fetch the relevant information from web/company provided help document about which the conversation is going over the call and displays it in a web page.

Implementation: We will extract the text spoken by the customer in the live call and take its samples in real time. These sample are passed to a ML model to generate the possible solutions which will be displayed in a webpage. The process will start when the call starts and vice-versa.

Tools and Technologies: Speech Processing, Multithreading, Python, Pytorch, Deep Learning, NLP, Flask, etc.

# Libraries required:
1. Pandas
2. sklearn
3. speech_recognition
4. nltk
5. flask


# How to run:
--make sure you have install all the above mentioned libraries and run below commands in the terminal from the file directory.
1. python preprocessing.py
2. python flask_ui.py
