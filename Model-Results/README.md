[baseline_solar_30k]: https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_baseline_solar_30k.png
[baseline_solar_5k]:  https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_baseline_solar_5k.png

[prophet_solar_30k]:   https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_prophet_solar_30k.png
[prophet_solar_5k]:    https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_prophet_solar_5k.png
[prophet_household_1]: https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_prophet_1.png
[prophet_household_2]: https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_prophet_5.png

[nn_solar_30k]:   https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_125_200.png
[nn_solar_5k]:    https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_234_400.png
[nn_household_1]: https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_553_100.png
[nn_household_2]: https://github.com/Jurgis1997/Solar-Prediction/blob/main/Model-Results/Plots/model_686_100.png


## Baseline results
To compare models created during this study, a baseline model was created using only historical monthly data along with the distance to the solar noon feature defined above. Historical data used to create this model was acquired from [here](https://www.saulesgraza.lt/saules-elektrines-generacija) and contained generation data for every month between the year of 2014 to 2019. 

Following images illustrate comparison between prediction results acquired using the baseline model and actual testing values:

### Comparison plot of the "Solar 30K" baseline model
![Comparison plot of the "Solar 30K" baseline model][baseline_solar_30k]

### Comparison plot of the "Solar 5K" baseline model
![Comparison plot of the "Solar 5K" baseline model][baseline_solar_5k]


## Facebook Prophet
Following images illustrate comparison between prediction results acquired using the Facebook Prophet models and actual testing values:

### Comparison plot of the "Solar 30K" Facebook Prophet model
![Comparison plot of the "Solar 30K" Facebook Prophet model][prophet_solar_30k]

### Comparison plot of the "Solar 5K" Facebook Prophet model
![Comparison plot of the "Solar 5K" Facebook Prophet model][prophet_solar_5k]

### Comparison plot of the "Household 1" Facebook Prophet model
![Comparison plot of the "Household 1" Facebook Prophet model][prophet_household_1]

### Comparison plot of the "Household 2" Facebook Prophet model
![Comparison plot of the "Household 2" Facebook Prophet model][prophet_household_2]

## Neural-Networks
Following images illustrate comparison between prediction results acquired using the Neural-Networks models and actual testing values:

### Comparison plot of the "Solar 30K" Neural-Networks model
![Comparison plot of the "Solar 30K" Neural-Networks model][nn_solar_30k]

### Comparison plot of the "Solar 5K" Neural-Networks model
![Comparison plot of the "Solar 5K" Neural-Networks model][nn_solar_5k]

### Comparison plot of the "Household 1" Neural-Networks model
![Comparison plot of the "Household 1" Neural-Networks model][nn_household_1]

### Comparison plot of the "Household 2" Neural-Networks model
![Comparison plot of the "Household 2" Neural-Networks model][nn_household_2]

