## Analyzing message data for disaster response

The objective of this project is to analyze data from Figure Eight to build a model for an API that classifies disaster messages base on some related categories, eg. food, water, shelter, clothing, etc.

## Libraries used

To run the associated python scripts and flask API please ensure your python environment contains the following libraries:

- plotly
- flask
- scikit-learn
- nltk
- pandas==1.2.3
- SQLAlchemy

## How to run the scripts

The following is the directory structure of the project:

```sh
disaster-response
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── categories.csv
│   ├── database.db
│   ├── messages.csv
│   └── process_data.py
└── models
    ├── classifier.pkl
    └── train_classifier.py
```

### 1. Create the data base

The data is inside the data directory, it is in raw format inside two csv files, messages and categories. The first step is to create the SQL Lite data base.
In order to create the data base run the below command inside the data directory
```sh
python process_data.py messages.csv categories.csv database.db
```
As you can see the the parameters are the messages file, the categories file and finally the database name you want.

### 2. Create the model

Once you have create the database the second step is to train a model. 






## Summary of the results of the analysis

It may seem obvious that at lower pricess the occupancy will be higer and indeed that was the case, but that also means that in general Boston as a City have very similar places all around the city so there is not a clear differentiator beside the price to help to choose a place.
In the below image it can be see how the occupancy is highly correlated to the average price. Any sudden change in the price drives the occupancy up or down.

![Distribution of Monthly pricesses](./img/month-prices.png)

Dividing the price in four different price groups, and plotting each group in the map, show that there is not a clear cluster of places, and low and high prices are mixed all around the city.

Price Distribution | Price group mapping
------------------ | -------------------
![Price Distribution](./img/dist-prices.png) | ![Price group mapping](./img/boston.png)

After working on the amenities to include them as features to be used in the price predictor using a multiple linear regression model, the following features were used


features|coefficients|p_values
--------|------------|--------
accommodates|5.00|0.00
bathrooms|7.95|0.00
Cable TV|16.67|0.00
Hair Dryer|0.96|0.00
Essentials|1.28|0.00
Dryer|4.26|0.00
Washer|10.56|0.00
Family/Kid Friendly|2.78|0.00
Heating|-3.52|0.00
Kitchen|-17.95|0.00
Air Conditioning|17.42|0.00
Internet|0.68|0.00
TV|15.24|0.00
bed_type_Real Bed|3.96|0.00
bed_type_Pull-out Sofa|-10.84|0.00
bed_type_Futon|-31.49|0.00
room_type_Shared room|-83.98|0.00
room_type_Private room|-75.83|0.00
guests_included|2.74|0.00
beds|-2.89|0.00
bedrooms|21.81|0.00
24-Hour Check-in|-5.80|0.00

The R2 for the model is 0.561; the model explains only about 56% of the price variability in Boston AirBnB, which suggests there are several other factors affecting price that are not accounted for in the model.

## License

The content of this repository is licensed under a [Creative Commons Attribution License](http://creativecommons.org/licenses/by/3.0/us/)



[1]: CRISP-DM. (2021, July 31). Data Science Process Alliance. https://www.datascience-pm.com/crisp-dm-2/






