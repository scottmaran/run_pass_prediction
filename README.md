

## Prior Work
Richard Anderson with Open Source Football estimates the probability of a QB Dropback with an XGBoost model using the features:
- Down (limited to 1,2,3)
- Yards for first down
- Yard line
- Score Differential
- Quarter
- Time remaining in half
- Number of timeouts for the offense and defense
With 100K training examples from 2016-2019, he gets 69.1% accuracy.

### Sources 
1) [Open Source Football](https://opensourcefootball.com/posts/2020-09-07-estimating-runpass-tendencies-with-tidymodels-and-nflfastr/)