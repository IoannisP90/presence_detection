This project contains an API that returns whether a user is expected to be present
at home on a specific date and time, including the probability of that.

The probability is based on the average quantile values given the inside temperature and electricity reading
for that user over that specific hour and part of day for the months of October and November.

1. How to create the model probabilities:
For that you need to execute the train.py script:
$ python -m train

This will load the csvs under the data folder, do the preprocessing, determine the user presence probability per hour and save the resulting dataFrame as a csv under "./model/hourly_user_probabilities.csv"

2. How to run the service:
$ uvicorn main:app

This will create the API on localhost:
e.g. http://127.0.0.1:8000/
You can send requests as follows: http:@localhost/{user_id}/{utc_datatime}/{hour}
API documentation can be found here: http:@localhost/docs (e.g. http://127.0.0.1:8000/docs)

3. Update probability table:
Open a terminal in the project folder and run:
$ python -m train

Restart the API:
$ uvicorn main:app
