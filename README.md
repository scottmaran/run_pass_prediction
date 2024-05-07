
## Exploratory Data Analysis
Notable tendencies:
- Pass 61.2% of the time
- Pass 74.1% of the time during 2Minute drills

![pass rate by field position](images/pass_rate_by_field_pos_line_chart.png)

![pass rate by Down](images/pass_rate_by_down_bar_chart.png)

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

## Results

We consider the naive baseline as the simple heuristic of always predicting a pass - one will be correct around 60-62% of the time.

| Model |  Validation Accuracy | Test Accuracy | 
| :---------------- | :------: | ----: |
| Naive Baseline: | 62.64%% | 60.95%%   | 
| Base Model: | 69.73% | 69.52%    | 
| Continuous Features Model: | 69.73% | 69.63%   | 
| Teams Included Model: | 69.88% | 69.04%   | 
| All Features Model: | 70.53% | 70.07   | 
| Fully Tuned All Features Model: | 71.33% | 71.81   | 

### Feature Importance 

Here is the feature importance for the basic model. Clock feels like it should be more important?

![Feature Importance of All_Feature_Model](models/base_model/feature_importance.png)

And here it is for the fully tuned, extra additional features model. A lot of the variables at the top stay the same, which makes sense as intuitively we should get a lot of mileage from knowing the down, distance, quarter, score, and timeouts remaining.

![Feature Importance of All_Feature_Model](models/all_features_model_fully_tuned/feature_importance.png)

How much of the previous features are just proxies for "was the previous play a pass".



### Tree Structure

We can also get an idea about some of the relationships learned by the model from examining the tree structure. Since this is a boosted model consisting of hundreds of trees, we cannot examine all of them, nor will any one tree contribute a majority to the final prediction.

We saw that previous net gain/loss in yards was the second most important feature, and in the first tree it is the root node. The splitting decision is whether or not the previous play gained five yards.

![Root node](images/tree_analysis/root_node.png)

The likely reason this decision was made for the root node is that other important features like down and distance have more complicated interactions between them. For instance, in the exploratory data analysis we saw pass rate by distance significantly differs depending on if its before third down or not. If we solely look at distance without extra features like down, then previous yards gain is a more discriminatory decision: we can draw a clear cut point in the previous gain graph (left) as opposed to the distance graph (right).

![Comparing passing rates by previous yards gained and distance](images/tree_analysis/pass_rates_by_splits.png)

If we keep following the tree left, our next split is "Distance < 2" followed by one more split that ends in two leaves with [prediction scores](xgboost.readthedocs.io/en/latest/tutorials/model.html) of -0.083 and -0.055 (while they're not exactly log odds, you can think of them as proportional to probabilities, so the lower the score the lower the probability of a pass).

![Root node](images/tree_analysis/leftmost_short_branch.png)

At a high-level this is telling us that when a team gained less than five yards on their previous play and is less than two yards from a first down, the model is more likely to predict a run than a pass. This aligns with our intuitions that a team is more likely to run when so close to a first down. This is confirmed empirically as teams ran the ball 72.3% of the time when less than two yards from a first down. 

We should note that this tendency is not affected by the previous yards gained by a team. Teams ran the ball 71.7% of the time when less than two yards and gaining less than five yards in their previous plays - effectively no difference. Instead this is more indicative of the general trend in this tree to use 'Distance < 2' as a powerful split early in branches, regardless of the preceding node. The reason for this is evident in the exploratory data analysis, where we inspected pass rates based on down and distance. While pass rates significantly diverged as a function of distance conditioned on downs, all pass rates were low for distances less than two, regardless of down.

![pass rate by field position](images/tree_analysis/pass_rate_by_distances_to_five.png)

Speaking of this relationship...this is exactly what this side of the tree captures. If distance is greater or equal to two, the next split is if its before third down or not.

![pass rate by field position](images/tree_analysis/left_bigger_branch.png)

We can see that the leaves on the no side of the 'DOWN: {1,2}' split have positive scores - indicating a higher probability of a pass. The leaf on the yes side of the 'DOWN: {1,2}' split have negative scores, i.e. more likely to run, which is what we saw in the above graph.


### Further Thoughts 

#### Dummy vs One-Hot

In regression important.

What about tree models? Definitely don't have the regression unidentifiable problem, so safer to include all levels.

#### RPO
Only 2.8% of plays had the OPTION feature designated as true. And out of all of those instances, every play was a run. Thus I just left them in and treated them as runs.

#### game_prev_PENALTY < 0.0571
Addressing this node in the tree structure section would have interrupted the more pertinent discussion around the features with the highest importance, but I did want to bring this up. (EXPLAIN WHY LESS SIGNIFICANT)

If there is a penalty called, it is three times as likely for the play to have been a pass than a run in our dataset. Thus I think the model may be trying to pick up that, if the team has been committing a lot of penalties in the game, then they may be throwing the ball more often than average. Thus if the offensive team's penalty rate during the game is above the threshold, the score is more positive than if it were below the threshold.

But again, this is on the margin (REWORD).



### Sources 
1) [Open Source Football](https://opensourcefootball.com/posts/2020-09-07-estimating-runpass-tendencies-with-tidymodels-and-nflfastr/)