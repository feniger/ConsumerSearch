# ConsumerSearch

## RESTART DATA: Remove all <stamp>.txt files.

## ------------------------------
## --- Uniform Distribution -----
## ------------------------------

### Optional: RESTART DATA

### In: def main()
#### 	Set:	testcase_distributions = [int_uniform_dist(1, 101)]
#### 	Set:	mechanisms = [u_random_mechanism()]
### In: def compare_mechanisms()
#### 	Set: 	dist_identifier = "Integer uniform distribution, low=1,  high=101"

#### 	Equilibrium price:
#### 	Uncomment ONLY: comp_eq_price(dist_identifier=dist_identifier)

####	Seller Revenue	
#### 	Uncomment ONLY: comp_seller_revenue(dist_identifier=dist_identifier)

#### 	Utility
#### 	Uncomment ONLY: comp_utility(dist_identifier=dist_identifier)

#### 	Social Welfare
#### 	Uncomment ONLY: comp_social_welfare(dist_identifier=dist_identifier)

### Run:
#### If restarted data, produce <stamp>.txt files
#### python $SRC/RunBRD.py -all			

### Plot Search cost to equilibrium price:
#### python $SRC/RunBRD.py -compare		 

