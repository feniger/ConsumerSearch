import glob, os
import sys
import csv
import argparse
import time
import datetime
import dateutil
import matplotlib.pyplot as plt
import numpy as np
from pylab import show
import operator
import math
import copy
from scipy.special import btdtr
import re

#---------------------------------------------------#
#----------------Globals----------------------------#
#---------------------------------------------------#

#Global: numeric precision 
precision=0.000001

#Global: number of best responses until halt 
stopper = 1000

#Global: number steps in search cost / threshold/ alpha/ etc 
easy_find_step_num = 42
step_num = easy_find_step_num

#Global
all_mechanism_identifiers = ["LF", "Exponential", "Threshold", "Uniform Random"]

#Golbal legend parameters
legend_size=22

#Global: output file
outfile = "./out_file.txt"

#Global: sellers ids are {1, 2}

#---------------------------------------------------#
#----------Global helper functions------------------#
#---------------------------------------------------#


def frange(start, stop, step):
	x = start
	while x < stop:
		yield x
		x += step


#---------------------------------------------------#
#----------Distributions----------------------------#
#---------------------------------------------------#

#constructs [val, prob] pairs
#Does not check validity!
class Distribution:
	def __init__(self, dist=[]):
		self.dist = dist 
		self.cdf = {}
		self.Pr ={}
		#If actual distribution
		if self.dist != []:
			for [val, prob] in self.dist:
				self.Pr[val] = prob



	def WhoAmI(self):
		return "Distribution= " + str(self.dist)

	def Get(self):
		return self.dist

	def IsEmpty(self):
		return (self.dist == [])

	#Validate distribution
	def IsValid(self):
		tot_prob = 0
		for [val, prob] in self.dist:
			tot_prob += prob
		#precision is a global variable here
		if abs(1-tot_prob) < precision:
			return True
		else:
			print "INVALID distribution!! total prob = " + str(tot_prob) + ", tot prob <= 1:" + str(tot_prob <= 1) + ", precision:" + str(1-tot_prob < precision)
			return False

	#dist = [val, prob] pairs, return support
	def Supp(self):
		if hasattr(self, 'support'):
			return self.support
		else:
			self.support = [x for [x, y] in self.dist]
			return self.support

	#return max in support
	def Supp_max(self):	
		return max(self.Supp())
	
	#return max in support
	def Supp_min(self):	
		return min(self.Supp())

	#returns Pr[v < x] for v <- dist.  That way 1-dist(x) = Pr[val >= x]
	def CDF(self, x):
		if x in self.cdf:
			return self.cdf[x]
		retVal = 0
		for [val, prob] in self.dist:
			if val < x:
				retVal += prob
		self.cdf[x] = retVal
		return retVal

	#Returns expectation
	def Expectation(self):
		if self.dist == []:
			raise NotImplementedError( "Should have implemented this" )

		if hasattr(self, 'expectation'):
			return self.expectation
		else:
			retVal = 0
			for [val, prob] in self.dist:
				retVal += val * prob
			self.expectation = retVal
			return self.expectation


class beta_dist(Distribution):
	def __init__(self, a, b, precision):
		Distribution.__init__(self)
		self.a = a
		self.b = b
		#Define precision (i.e., support of) beta distribution (integral values from 1 to precision)
		self.precision = precision
		#Create distribution
		self.dist = []
		prev_cdf = 0
		cur_cdf = 0
		for i in range(1, self.precision, 1):
			cur_cdf = btdtr(self.a, self.b, float(i) / float(self.precision))
			self.dist.append([i, cur_cdf - prev_cdf])
			prev_cdf = cur_cdf

		
		self.dist.append([self.precision, 1 - cur_cdf])

		#Check validity
		if not self.IsValid():
			print (self.WhoAmI())
			print ("I Created an invalid distribution!")

	
	def WhoAmI(self):
		return "Beta distribution, parameter=" + str(self.a) + ',' + str(self.b) + ',' + str(self.precision)




#Integer uniform distribution
#vector of [value, probability] pairs.
#equal weights for integers low...high-1
class int_uniform_dist(Distribution):
	def __init__(self, low, high):
		Distribution.__init__(self)
		self.low = low
		self.high = high
		#Create distribution
		self.dist = []
		num = high - low
		for i in xrange(low, high, 1):
			self.dist.append([i, float(1)/num])

		#Check validity	
		if not self.IsValid():
			print self.WhoAmI()
			print "I Created an invalid distribution!"

	def WhoAmI(self):
		return "Integer uniform distribution, low=" + str(self.low) + ",  high=" + str(self.high)


#Geometric distribution
#vector of [value, probability] pairs.
class geometric_dist(Distribution):
	def __init__(self, p, precision):
		Distribution.__init__(self)
		#Geometric distribution parameter
		self.p = float(p)
		#Define precision (i.e., support of) geometric distribution
		self.precision = precision
		#Create distribution
		self.dist = []
		curProb = self.p
		tot_prob = 0
		for i in xrange(1, self.precision, 1):
			self.dist.append([i, curProb])
			tot_prob += curProb
			curProb = curProb * (1-float(self.p))
			
		self.dist.append([self.precision, 1 - tot_prob])

		#Check validity
		if not self.IsValid():
			print self.WhoAmI()
			print "I Created an invalid distribution!"

	
	def WhoAmI(self):
		return "Geometric distribution, parameter=" + str(self.p) + ", precision=" + str(self.precision)


#equal revenue distribution
#vector of [value, probability] pairs.
class equal_revenue_dist(Distribution):
	def __init__(self, precision):
		Distribution.__init__(self)
		#Define precision (i.e., support of) geometric distribution
		self.precision = precision
		#Create distribution
		self.dist = []
		tot_prob = 0
		curProb = 0
		for i in xrange(1, self.precision, 1):
			# Pr[ER = i] = Pr[ER >= i] - Pr[ER >= i+1]
			curProb = float(1)/i - float(1)/(i+1)
			self.dist.append([i, curProb])
			tot_prob += curProb
			
		self.dist.append([self.precision, 1 - tot_prob])

		#Check validity
		if not self.IsValid():
			print self.WhoAmI()
			print "I Created an invalid distribution!"

	
	def WhoAmI(self):
		return "Equal revenue distribution, precision=" + str(self.precision)



#almost equal revenue distribution (1 - 1/(v+1))
#vector of [value, probability] pairs.
class almost_equal_revenue_dist(Distribution):
	def __init__(self, precision):
		Distribution.__init__(self)
		#Define precision (i.e., support of) geometric distribution
		self.precision = precision
		#Create distribution
		self.dist = []
		tot_prob = 0
		curProb = 0
		for i in xrange(0, self.precision, 1):
			# Pr[v = i] = Pr[v >= i] - Pr[v >= i+1]
			curProb = float(1)/(i+1) - float(1)/(i+2)
			self.dist.append([i, curProb])
			tot_prob += curProb
			
		self.dist.append([self.precision, 1 - tot_prob])

		#Check validity
		if not self.IsValid():
			print self.WhoAmI()
			print "I Created an invalid distribution!"

	
	def WhoAmI(self):
		return "Almost Equal revenue distribution, precision=" + str(self.precision)

#Self made distribution (input: val, prob, val prob, ....)
class self_made_dist(Distribution):
	#TBD: I think I can use python's 'reshape' here 
	def __init__(self, *args):
		Distribution.__init__(self)
		self.dist = []
		cur = []
		for a in args:
			cur.append(a)
			if len(cur) == 2:
				retDist.append(cur)
				cur = []

		#Check validity
		if not self.IsValid():
			print self.WhoAmI()
			print "I Created an invalid distribution!"
	
	def WhoAmI(self):
		return "Self made distribution, dist=" + str(self.dist)



#---------------------------------------------------#
#-------------Mechanisms----------------------------#
#---------------------------------------------------#

class Mechanism:
	def __init__(self, dist, search_cost):
		self.dist = dist 
		self.search_cost = search_cost

	def WhoAmI(self):
		return "Mechanism"


	#return a distribution ([id, prob] pairs) of who is in the Buy-Box
	def BuyBoxWinner(self, p1_price, p2_price):
		raise NotImplementedError( "Should have implemented this" )



#lowest price gets buybox w.p. 1, breaking ties uniformly at random
#Independent of distribution and search cost
class Low_first_mechanism(Mechanism):
	def __init__(self):
		Mechanism.__init__(self, None, None)
		self.shift = 13

	def WhoAmI(self):
		return "< LF > Mechanism"

	#return a distribution ([id, prob] pairs) of who is in the Buy-Box
	def BuyBoxWinner(self, p1_price, p2_price):
		if p1_price < p2_price:
			#Seller 1 in BB w.p. 1
			return Distribution([[1, 1]])
		elif p2_price < p1_price:
			#Seller 2 in BB w.p. 1
			return Distribution([[2, 1]])
		else:
			#uniformly at random 
			return Distribution([[1, 0.5], [2, 0.5]])


#BB winner is selected uniformly at random
#Independent of distribution and search cost
class u_random_mechanism(Mechanism):
	def __init__(self):
		Mechanism.__init__(self, None, None)
		self.shift = 10

	def WhoAmI(self):
		return "< Uniform Random > Mechanism"

	#return a distribution ([id, prob] pairs) of who is in the Buy-Box
	def BuyBoxWinner(self, p1_price, p2_price):
		return Distribution([[1, 0.5], [2, 0.5]])


#BB winner is selected according to predefined weights
#Independent of distribution and search cost
class weighted_random_mechanism(Mechanism):
	def __init__(self, p1_weight, p2_weight):
		Mechanism.__init__(self, None, None)
		self.p1_weight = p1_weight
		self.p2_weight = p2_weight

	def WhoAmI(self):
		return "< Weighted Random > Mechanism, p1_weight=" + str(self.p1_weight) + ",  p2_weight=" + str(self.p2_weight)

	#return a distribution ([id, prob] pairs) of who is in the Buy-Box
	#Recall: IDs are {1, 2}
	def BuyBoxWinner(self, p1_price, p2_price):
		return Distribution([[1, self.p1_weight], [2, self.p2_weight]])


#BB winner is selected according to predefined weights
#Independent of distribution and search cost
class threshold_mechanism(Mechanism):
	def __init__(self, threshold):
		Mechanism.__init__(self, None, None)
		self.threshold = threshold
		self.shift = 6

	def WhoAmI(self):
		return "< Threshold > Mechanism, threshold=" + str(self.threshold)

	#return a distribution ([id, prob] pairs) of who is in the Buy-Box
	#Recall: IDs are {1, 2}
	def BuyBoxWinner(self, p1_price, p2_price):
		below = []
		if p1_price <= self.threshold:
			below.append(1)
		if p2_price <= self.threshold:
			below.append(2)

		#Below threshold get equal probability
		retDist = []
		if len(below) == 0:
			return Distribution()
		for p_id in below:
			retDist.append([p_id, float(1)/len(below)])

		return Distribution(retDist)


def taylor_exp(num):
	return  1+ num + ( num * num )/2 + ( num * num * num )/6 


#BB winner is selected according to predefined weights
#Independent of distribution and search cost
class exponential_mechanism(Mechanism):
	def __init__(self, alpha):
		Mechanism.__init__(self, None, None)
		self.alpha = alpha
		self.shift = 2

	def WhoAmI(self):
		return "< Exponential > Mechanism, alpha=" + str(self.alpha)

	#return a 'Distribution' of who is in the Buy-Box
	#Recall: IDs are {1, 2}
	def BuyBoxWinner(self, p1_price, p2_price):
		prob1 = float(1) / ( 1 + math.exp(self.alpha * ( p2_price - p1_price )  ) )
		# prob1 = math.exp(self.alpha*p1_price)/(math.exp(self.alpha*p1_price)+math.exp(self.alpha*p2_price))
		# print p1_price
		# print p2_price
		# print "alpha:" + str(self.alpha)
		# print taylor_exp(self.alpha*p1_price)
		# print taylor_exp(self.alpha*p2_price)

		# prob1 = taylor_exp(self.alpha*p1_price)/(taylor_exp(self.alpha*p1_price)+taylor_exp(self.alpha*p2_price))
		return Distribution([[1, prob1], [2, 1 - prob1]])

#---------------------------------------------------#
#----------Profile----------------------------------#
#---------------------------------------------------#

class Profile():
	def __init__(self, dist=None, search_cost=None, mechanism=None, p1_price=None, p2_price=None, productionCosts=None): ### NEW PC
		self.dist = dist
		self.search_cost = search_cost
		self.mechanism = None
		self.p1_price = p1_price
		self.p2_price = p2_price
		self.r_price = None
		self.productionCosts = productionCosts ### NEW PC

	def IsEmpty(self):
		return self.dist == None

	def WhoAmI(self):
		retStr = ""
		retStr += "search_cost=" + str(self.search_cost) + "\n" 
		
		if self.dist != None:
			retStr += "dist=" + str(self.dist.WhoAmI()) + "\n" + "r_price=" + str(self.Res_price()) + "\n" 
                
		if self.mechanism != None:
			retStr += "mechanism=" + str(self.mechanism.WhoAmI()) + "\n"

		retStr += "p1_price=" + str(self.p1_price) + "\n" + "p2_price=" + str(self.p2_price) + "\n"
		
		return retStr

	#Careful when updating search cost!!
	def SetSearchCost(self, sc):
		self.search_cost = sc
		self.r_price = None


	#precision: accuracy of reservation price
	def Res_price(self, rerun=False):
		if self.r_price == None or rerun == True:
			if self.search_cost == 0:
				self.r_price = self.dist.Supp_max() + 1 
				return self.r_price
			#Check search cost validity
			if self.dist.Expectation() < self.search_cost:
				print "Search cost too high"
				print "E[dist] = " + str(self.dist.Expectation())
				print "search cost = " + str(self.search_cost) 
				raise NotImplementedError("Search cost too high")
                        
			#Check precision
			if abs(self.dist.Expectation() - self.search_cost) < precision:
				print "bad search cost! precision-wise"
				return None

			high = self.dist.Supp_max()
			low = self.dist.Supp_min()
			found = False 
			#Find reservation price up to precision
			while not found:
				cur = (float(high) + low)/2

                                if cur - low < precision:
                                        print "Search cost too high"
                                        print "E[dist] = " + str(self.dist.Expectation())
                                        print "search cost = " + str(self.search_cost) 
                                        raise NotImplementedError("Search cost too high")
				
				E = expected_risk(self.dist, cur)
				# print "E/c/h/l: " + str(E) + "/" + str(cur) + "/" + str(high) + "/" + str(low)
				margin = E - self.search_cost
				
				if margin > precision:
					low = cur
					#print "low = " + str(low)
				elif margin < -precision:
					high = cur
					#print "high = " + str(high)
				else:
					found = True
					# print "Reservation price: " + str(cur) # + ",  h/l:" + str(high) + "/" + str(low)
			self.r_price = cur

		return self.r_price



#computes E[(v-res_price)^+] for v <- dist 
def expected_risk(dist, res_price):
	retVal = 0
	for [val, prob] in dist.Get():
		if val - res_price > 0:
			retVal += (val - res_price) * prob
	return retVal


#dist = [val, prob] pairs, return max in val column (0)
#precision: accuracy of reservation price
def res_price(dist, search_cost):
	if search_cost == 0:
		return dist.Supp_max() + 1
	#Check search cost validity
	if dist.Expectation() < search_cost:
		print "Search cost too high"
		print "E[dist] = " + str(dist.Expectation())
		print "search cost = " + str(search_cost) 
		return None

	#Check precision
	if abs(dist.Expectation() - search_cost) < precision:
		print "bad search cost! precision-wise"
		return None

	high = dist.Supp_max()
	low = dist.Supp_min()
	found = False 
	#Reservation price precision
	while not found:
		cur = (float(high) + low)/2


                if cur - low < precision:
                        print "Search cost too high"
                        print "E[dist] = " + str(self.dist.Expectation())
                        print "search cost = " + str(self.search_cost) 
                        raise NotImplementedError("Search cost too high")

		E = expected_risk(dist, cur)
		# print "E/c/h/l: " + str(E) + "/" + str(cur) + "/" + str(high) + "/" + str(low)
		margin = E - search_cost
		if margin > precision:
			low = cur
		elif margin < -precision:
			high = cur
		else:
			found = True
#			print "Reservation price: " + str(cur) # + ",  h/l:" + str(high) + "/" + str(low)
	return cur	



#----------------------------------------------------------#
#-----Demand1 and Demand2 assume a Buy-Box slot -----------#
#----------------------------------------------------------#

def Demand1_ver_2(dist, res_price, my_price, competing_price):
	demand = 0
	#if the other seller never gets searched
	if competing_price > res_price:
		#Pr[v1 >= my_price]
		return (1-dist.CDF(my_price))
	
	# value for me is so high that the other bidder doesn't get searched
	demand += 1 - dist.CDF(res_price - competing_price + my_price)
	
	# the other seller gets searched but the buyer returns to me
	for [val, prob] in dist.Get():
		if val < my_price:
			continue
		elif val >= res_price - competing_price + my_price:
			break
		else:
			demand += prob * dist.CDF(val - my_price + competing_price)
	return demand

def Demand2_ver_2(dist, res_price, my_price, competing_price):
	demand = 0
	# if I never get searched
	if my_price > res_price:
		return 0

	# if I get searched and the buyer chooses me
	for [val, prob] in dist.Get():
		if val >= res_price - my_price + competing_price:
			break
		elif val < competing_price:
			demand += prob * (1 - dist.CDF(my_price))
		else:
			demand += prob * (1 - dist.CDF(val - competing_price + my_price))
	
	return demand


#Returns demand function of Buy-Box slot
#For v1, v2 <- dist
#D_1(p|q) = \int_{0}^\infty \Pr[v1 - p > \max\{0,  \min \{r, v2\} -q \} ] dF(v2) 
def Demand1(dist, res_price, my_price, competing_price):
	demand = 0
	#If other never gets searched
	if competing_price > res_price:
		#Pr[v1 >= my_price]
		return (1-dist.CDF(my_price))

	#Pr[v2 < competing_price] * Pr[v1 >= my_price]
	#Seller 1 always selected (regardless of search)
	demand += dist.CDF(competing_price) * (1-dist.CDF(my_price))

	# print "demand1 =  " + str(demand)

	# integrate v2 from competing_price to res_price (Pr[v2] * Pr[v1 - my_price >= v2 - competing_price])
	# I.e. Tie (v1 - my_price = v2 - competing_price) is broken in favor of first slot
	# Seller 1 selected if beats seller 2
	for [val, prob] in dist.Get():
		if val >= competing_price and val < res_price:
			demand += (1 - dist.CDF(val - competing_price + my_price)) * prob
			# print "demand1 =  " + str(demand)

	#Pr[v2 >= res_price] * Pr[v1 - my_price >= res_price - competing_price ]
	#seller 2 is not searched (because if searched then selected):
	demand += (1 - dist.CDF(res_price)) * (1 - dist.CDF(res_price - competing_price + my_price))
	# print "demand1 =  " + str(demand)
	return demand


#Returns demand function of non Buy-Box slot
#For v_p, v_q <- dist
#D_1(p|q) = \int_{0}^\infty \Pr[v_p - p > \max\{0,  \min \{r, v_q\} -q \} ] dF(v_q) 
def Demand2(dist, res_price, my_price, competing_price):
	#If my_price is too high to be searched
	if my_price > res_price:
		return 0

	demand = 0
	#Pr[v1 < competing_price] * Pr[v2 >= my_price]
	#Seller 2 always selected (because always searched and seller 1 has negative utility)
	demand += dist.CDF(competing_price) * (1-dist.CDF(my_price))
	#print "demand2 =  " + str(demand)

	#integrate v1 from competing_price to competing_price + res_price - my_price
	for [val, prob] in dist.Get():
		# print str(val) + ":" + str(prob)
		# print str(val) + ", " + str(competing_price) + ", " + str(res_price) + ", " + str(my_price)
		if val >= competing_price and val <=res_price - my_price + competing_price:
			# Pr[v = my_price + val - competing_price]
			pr_eq =  dist.Pr.get(my_price + val - competing_price, 0)
			# Pr[my_value > my_price + min(res_price, cur) - competing_price] 
			# I.e. Tie (my_value - my_price = val - competing_price) is broken in favor of first (the other) slot
			demand += (1 - dist.CDF(my_price + val - competing_price) - pr_eq) * prob
			#print "demand2 =  " + str(demand)
	return demand


#--------------------------------------------------------------------------#
#-----Demand1NBB and Demand2NBB assume there is no Buy-Box slot -----------#
#--------------------------------------------------------------------------#

#Returns demand of first slot (both slots have search cost)
def Demand1NBB(dist, res_price, my_price, competing_price):
	#If my_price is too high to be searched
	if my_price > res_price:
		return 0
	#I'm searched 
	return Demand1(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price)

#Returns demand of second slot (both slots have search cost)
def Demand2NBB(dist, res_price, my_price, competing_price):
	#If competing_price is too high to be searched and my price isn't
	if competing_price > res_price and my_price <= res_price:
		#Pr[v2 >= my_price]
		return (1 - dist.CDF(my_price))

	#First is searched 
	return Demand2(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price)


#Returns revenue of Buy-Box slot
def Revenue1(dist, res_price, my_price, competing_price, productionCost): ### NEW PC
	d = Demand1(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price)
	# print "demand1:" + str(d)
	return ((my_price - productionCost) * d) # Demand1(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price)


#Returns revenue of non Buy-Box slot
def Revenue2(dist, res_price, my_price, competing_price, productionCost): ### NEW PC
	return ((my_price - productionCost) * Demand2(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price))

#Returns revenue of first slot (no Buy-box)
def Revenue1NBB(dist, res_price, my_price, competing_price, productionCost): ### NEW PC
	return ((my_price - productionCost) * Demand1NBB(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price))


#Returns revenue of second slot (no Buy-box)
def Revenue2NBB(dist, res_price, my_price, competing_price, productionCost): ### NEW PC
	return ((my_price - productionCost) * Demand2NBB(dist=dist, res_price=res_price, my_price=my_price, competing_price=competing_price))

#Expected revenue over distribution BuyBoxDist  
def ExpectedRev(profile, my_id, my_price):
	#Set competing price and outcome
	if my_id == 1:
		competing_price = profile.p2_price
		BuyBoxDist = profile.mechanism.BuyBoxWinner(p1_price=my_price, p2_price=profile.p2_price)
		productionCost = profile.productionCosts[0] ### NEW PC
	else:
		competing_price = profile.p1_price
		BuyBoxDist = profile.mechanism.BuyBoxWinner(p1_price=profile.p1_price, p2_price=my_price)
		productionCost = profile.productionCosts[1] ### NEW PC

	
	# print "BBdist:"  + BuyBoxDist.WhoAmI()
	# print "my price = " + str(my_price)
	# print "competing_price=" + str(competing_price)
	#Compute reservation price
	r_price = profile.Res_price()
	# print "res price: " + str(r_price)
#---------No Buy-Box slot:
	if BuyBoxDist.IsEmpty():
		if my_price < competing_price:
			return Revenue1NBB(dist=profile.dist, res_price=r_price, my_price=my_price, competing_price=competing_price, productionCost=productionCost) ### NEW PC
		elif competing_price < my_price:
			return Revenue2NBB(dist=profile.dist, res_price=r_price, my_price=my_price, competing_price=competing_price, productionCost=productionCost) ### NEW PC
		#buyer selects uniformly at random
		else:
			return (0.5 * Revenue1NBB(dist=profile.dist, res_price=r_price, my_price=my_price, competing_price=competing_price, productionCost=productionCost) ### NEW PC
				   +
				   0.5 * Revenue2NBB(dist=profile.dist, res_price=r_price, my_price=my_price, competing_price=competing_price, productionCost=productionCost)) ### NEW PC




#---------With Buy-Box slot:
	#Validate BB distribution (function yells invalid)
	if not BuyBoxDist.IsValid():
		return None

	#Compute expected revenue
	retRev = 0
	for [winner, prob] in BuyBoxDist.Get():
		if winner == my_id:
			# print "111prob:" + str(prob)
			retRev += prob * Revenue1(dist=profile.dist, res_price=r_price, my_price=my_price, competing_price=competing_price, productionCost=productionCost) ### NEW PC
			# print "retRev 1: " + str(retRev)
		#assuming sellers ids are {1, 2}
		elif winner == 3 - my_id:
			retRev += prob * Revenue2(dist=profile.dist, res_price=r_price, my_price=my_price, competing_price=competing_price, productionCost=productionCost) ### NEW PC
			# print "retRev 2: " + str(retRev)
		else:
			print "Wrong format of BuyBoxDist!!"
	return retRev

#Buyer utility when Buy-box seller price is BB_price (etc.) 
def BUwithOrder(dist, search_cost, BB_price, NBB_price):
	#Compute reservation price
	r_price = res_price(dist=dist, search_cost=search_cost)

	tot_util = 0
	#For each match value with Buy-Box seller
	for [v1, p1] in dist.Get():
		# print v1
		#Match first immediately 
		if v1- BB_price > max(0, r_price - NBB_price):
			tot_util += p1 * (v1 - BB_price)

		#Search second seller	
		elif r_price - NBB_price >= max(0, v1 - BB_price):
			#Pay search cost
			tot_util -= p1 * search_cost

			#Match highest (if any)
			for [v2, p2] in dist.Get():
				tot_util += p1 * p2 * max(v1 - BB_price, v2 - NBB_price, 0)
		
		# #No match case
		# else:
		# 	print "No match no search: BB utility =" + str(v1 - BB_price) + " NBB expected utility=" + str(r_price - NBB_price)

	# print "tot util: " + str(tot_util)
	return tot_util

#Buyer utility when there is no Buy-box 
def BU_NBB_withOrder(dist, search_cost, first_price, second_price):
	#Compute reservation price
	r_price = res_price(dist=dist, search_cost=search_cost)

	tot_util = 0
	#For each match value with first seller
	for [v1, p1] in dist.Get():
		#Pay search cost for first seller
		tot_util -= p1 * search_cost

		# print v1
		#Match first immediately 
		if v1- first_price > max(0, r_price - second_price):
			tot_util += p1 * (v1 - first_price)

		#Search second seller	
		elif r_price - second_price >= max(0, v1 - first_price):
			#Pay search cost for second seller
			tot_util -= p1 * search_cost

			#Match highest 
			for [v2, p2] in dist.Get():
				tot_util += p1 * p2 * max(v1 - first_price, v2 - second_price, 0)
	# print "tot util: " + tot_util
	return tot_util


#Gets a state and returns expected buyer utility (value - price)
# Assuming: match value distribution is the same for both sellers
# Assuming: if value - price = reservation price, then DO search.
#TBD
def buyerUtility(profile):
	#Get sellers (possibly randomized) order
	buyBoxDist = profile.mechanism.BuyBoxWinner(p1_price=profile.p1_price, p2_price=profile.p2_price)

	#If no buy-box
	if buyBoxDist.IsEmpty():
		return BU_NBB_withOrder(dist=profile.dist, search_cost=profile.search_cost, 
								first_price= min(profile.p1_price, profile.p2_price),
								second_price=max(profile.p1_price, profile.p2_price))

	#Validate buybox distribution
	if not buyBoxDist.IsValid():
		print buyBoxDist.WhoAmI()
		raise NotImplementedError( "Buy box didn't return a distribution" )

	#Compute utility
	tot_util = 0
	for [BB_id, prob] in buyBoxDist.Get():
		# print str(BB_id) +  " : " + str(prob) 
		if BB_id == 1:
			tot_util += prob * BUwithOrder(dist=profile.dist, search_cost=profile.search_cost, BB_price=profile.p1_price, NBB_price=profile.p2_price)
		if BB_id == 2:
			tot_util += prob * BUwithOrder(dist=profile.dist, search_cost=profile.search_cost, BB_price=profile.p2_price, NBB_price=profile.p1_price)

	return tot_util
		



	#Suppose pX_id is in Buy-Box and pY_id is second (paying search cost)
	BB_price = pX_price
	NBB_price = pY_price


#Best response versus a competing_price, given a mechanism. 
def BR(profile, my_id, expressive=False):
	# print "my id=" + str(my_id)
	# print profile.WhoAmI()
	maxRev = -1
	bestPrice = None

	if expressive:
		print "my id " + str(my_id)
		print profile.WhoAmI()
		
        ### NEW PC : choose correct production cost
	if my_id == 1:
                productionCost = profile.productionCosts[0]
        else:
                productionCost = profile.productionCosts[1]
	
	#For each value in the support of dist 
	for val in profile.dist.Supp():
		# print 
		# print
		curRev = ExpectedRev(profile=profile, my_id=my_id, my_price=val)
		# print str(val) + ": " + str(curRev)

		if curRev > maxRev and val >= productionCost: ### NEW PC
			maxRev = curRev
			bestPrice = val

	if expressive:
		print "Rev from best price " + str(bestPrice) + " is " + str(maxRev) 

	return bestPrice


#ids need to be 1 and 2
# Returns: 	if reached equilibrium: [id1, eq_price1, id2, eq_price2]
#			otherwise: None
# cycle_special: Whenever a cycle is found, computer it's average utility and return
def BRD(profile, cycle_special=False, expressive=False):
	# print mechanism.WhoAmI()
	#TBD - in the first iteration if br pX is br then stop not at equilibrium.

	#Identify a repeating state
	cycle_identifier = []

	#Initial state
	cur_profile = copy.deepcopy(profile)

	#Start with Seller 1
	turn = 1
	
	#Initial best responses
	br1 = BR(profile=cur_profile, my_id=1, expressive=expressive)
	br2 = cur_profile.p2_price

	#Equilibrium flag
	is_eq = False
	

	#number of BRDs until stop at "no equilibrium"
	s = stopper

	#while not in equlibrium
	while s >0 and (not is_eq):
		#assuming sellers ids are {1, 2}
		if turn == 1:
			br1 = BR(profile=cur_profile, my_id=1, expressive=expressive)
		if turn == 2: 
			br2 = BR(profile=cur_profile, my_id=2, expressive=expressive)

		#swap sellers (assuming sellers ids are {1, 2})
		turn = 3-turn
		#update stopper
		s -= 1
		
		if expressive:
		#Print state
			print "state:"
			print cur_profile.WhoAmI()

		#If equilibrium stop and return
		if br1 == cur_profile.p1_price and br2 == cur_profile.p2_price:
			print "Equilibrium!"
			print "P1: " + str(br1) + ",   P2: " + str(br2)
			#Return equilibrium
			return ["profile", cur_profile] #[pX_id, brX, pY_id, brY]

		#Update profile
		cur_profile.p1_price = br1
		cur_profile.p2_price = br2

		#Check for cycle
		if [cur_profile.p1_price, cur_profile.p2_price] in cycle_identifier:
			if expressive:
				print "Cycle!" # + " returned to:"
				print cur_profile.WhoAmI()

			cycle_identifier.append([cur_profile.p1_price, cur_profile.p2_price])
			#In case we want to return the utility of a cycle, do it here for now:
			if cycle_special:
				return ["cycle",  cycle_study(profile=cur_profile, cycle_identifier=cycle_identifier)]

			return ["cycle", Profile()] 
		else:
			cycle_identifier.append([cur_profile.p1_price, cur_profile.p2_price])

	if s == 0:
		print "ERROR!!!: Stopped due to stopper!"
		return ["profile", Profile()] 

	print "ERROORR!!!! how did I get here?!?????"


# Assumption: 'cycle identifier' has the end of the cycle in the last element
# Returns: cycle average buyer utility 
# 			cycle length
#			string of cycle |pr1 pr2|pr1 pr2|....|pr1 pr2
def cycle_study(profile, cycle_identifier):
	sum_bu = 0
	sum_p1_rev = 0
	sum_p2_rev = 0
	sum_pr1 = 0
	sum_pr2 = 0
	last_state = cycle_identifier[-1]
	for idx, state in enumerate(cycle_identifier):
		if state == last_state:
			start_idx = idx
			break

	# print "utility counting"
	cycle_str = ""
	tot = 0
	for [pr1, pr2] in cycle_identifier[idx+1:]:
		# print str(pr1) + " , " + str(pr2)
		tot += 1
		cycle_str += "|" +str(pr1) + " " + str(pr2)
		profile.p1_price = pr1
		profile.p2_price = pr2
		# bu = buyerUtility(profile)
		[bu, p1_rev, p2_rev] = studyState(profile)
		# print bu
		sum_bu += bu
		sum_p1_rev += p1_rev
		sum_p2_rev += p2_rev
		sum_pr1 += pr1
		sum_pr2 += pr2
	
	ave_bu = float(sum_bu) / tot
	ave_p1_rev = float(sum_p1_rev) / tot
	ave_p2_rev = float(sum_p2_rev) / tot
	ave_pr1 = float(sum_pr1) / tot
	ave_pr2 = float(sum_pr2) / tot

	return [ave_bu, ave_p1_rev, ave_p2_rev, ave_pr1, ave_pr2, tot, cycle_str]

def printToFile(plotArray, meta_data=""):
	filename =str(time.time()) + ".txt" 
	with open(filename , "w") as f:
		f.write(meta_data)
		f.write("# Data: \r\n")
		for line in plotArray:
			for item in line:
				f.write(str(item) + ", ")
			f.write("\r\n")

	print "printed to: " + filename

#Reads lines from f until metadata is done
def parse_metadata(filehandle):
	eq_or_cyc = ""
	distribution = ""
	mechanism = ""
	opt_par_file = None
	mech_pars = {}
	headers = []
	for line in filehandle:
		if line.startswith("#"):
			#headers
			m = re.search('.*\[(.+?)\].*', line)
			if m:
				# print "found headers"
				found = m.group(1)
				headers = [h.strip('\' ') for h in found.split(",")]
				# print headers
				# for h in headers:
				# 	print h

			#Mechanism
			m = re.search('# Mechanism:.*<(.+?)>.*', line)
			if m:
				# print "found mechanism"
				mechanism = m.group(1).strip()
				if mechanism == "Exponential":
					m = re.search('alpha=(.*)' , line)
					mech_pars["alpha"] = m.group(1).strip()
					m = re.search('-(.*)', mech_pars["alpha"])
					if m:
						mech_pars["alpha"] = m.group(1).strip()


			#Distribution
			m = re.search('# Distribution:(.+)', line)
			if m:
				distribution = m.group(1).strip()

			#Solution concept
			m = re.search('# Solution concept: (.+)', line)
			if m:
				# print "found Solution concept"
				eq_or_cyc = m.group(1).strip()

			m = re.search('# optimal (.*) per search cost', line)
			if m:
				opt_par_file = m.group(1).strip()
				# print "opt par:" + opt_par_file

			#Meta data ended
			m = re.search('# Data:', line)
			if m:
				return [eq_or_cyc, opt_par_file, distribution, [mechanism, mech_pars], headers]



	
	raise NotImplementedError("Shouldn't be here!!")


def plotFile(filename):
	title = ""
	with open(filename , "r") as f:
		#Reads lines from f until metadata is done
		[eq_or_cyc, opt_par_file, distribution, [mechanism, mech_pars], headers] = parse_metadata(filehandle=f)
		title =  str(eq_or_cyc)  + ", Dist:"  + str(distribution) + ", Mechanism: " + str(mechanism)
		print "title: " + title
		np_arr = np.loadtxt(f, delimiter = ',', usecols = range(len(headers)))

	plotThis1(title=title, plotArray=np_arr, params_labels=headers)

		
#plotArray should be a numpy array. Line format: [x_axis, val1, val2, ... valk] 
def plotThis1(title, plotArray, params_labels):		
	#pX plot
	fig1 = plt.figure()
	fig1.canvas.set_window_title(title)
	for (idx, lab) in enumerate(params_labels[1:]):
		plt.plot(plotArray[:,0], plotArray[:,idx+1], label=lab, marker='o', linestyle="None")



	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel(params_labels[0])
	plt.show()


#PlotArray entry: [search_cost, pX_id, pX_price, pY_id, pY_price, buyer_utility, pX_rev, pY_rev]
def plotThis(dist, mech, plotArray, X_axis_label, Y_axis_label):
	# for entry in plotArray:
	# 	print entry
	X_axis = []
	pX_id = plotArray[0][1]
	pY_id = plotArray[0][3]
	pX_prices = []
	pY_prices = []
	buyer_utilities = []
	pX_revs = []
	pY_revs = []
	for [x_axis_val, x_id, pX_price, y_id, pY_price, bu, pX_rev, pY_rev] in plotArray:
		#Verify validity of input
		if x_id != pX_id or y_id != pY_id:
			print "ID PROBLEMS!!!!"
			return None

		X_axis.append(x_axis_val)
		pX_prices.append(pX_price)
		pY_prices.append(pY_price)
		buyer_utilities.append(bu)
		pX_revs.append(pX_rev)
		pY_revs.append(pY_rev)

	#pX plot
	fig1 = plt.figure()
	fig1.canvas.set_window_title("Mechanism: " + mech.WhoAmI() + ". Distribution: " + dist.WhoAmI())
	plt.plot(X_axis, pX_prices, "rx", label= "p" + str(pX_id) + " price")
	plt.plot(X_axis, pY_prices, 'b+', label="p" + str(pY_id) + " price")
	plt.plot(X_axis, buyer_utilities, 'g^', label="utility")
	plt.plot(X_axis, pX_revs, 'b|', label="p" + str(pX_id) + " revenue")
	plt.plot(X_axis, pY_revs, 'k_', label="p" + str(pY_id) + " revenue")


	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.ylabel(Y_axis_label)
	plt.xlabel(X_axis_label)
	plt.show()
	# fig2 = plt.figure()
	# fig2.canvas.set_window_title("Search cost to prices of second seller")
	# plt.plot(search_costs, p2_prices, 'o')
	# show()


#Returns [buyer_utility, pX_id_revenue, pX_id_revenue]
def studyState(profile):
	# print "studying state"
	#Compute buyer utility in equilibrium
	bu = buyerUtility(profile=profile)
	##Compute sellers revenue in equilibrium
	p1_rev = ExpectedRev(profile=profile, my_id=1, my_price=profile.p1_price)
	p2_rev = ExpectedRev(profile=profile, my_id=2, my_price=profile.p2_price)
	return [bu, p1_rev, p2_rev]


#P is the profile
def opt_threshold(P):
	#[threshold, equilibrium price] pairs
	th_pr = {}
	#[threshold, buyer utility] pairs
	th_bu = {}
	real_step_num = step_num
	while int(P.dist.Expectation()) / real_step_num == 0:
		real_step_num -= 1

	if real_step_num == 0:
		print "expectation:" + str(P.dist.Expectation())
		raise NotImplementedError("SOMETHING WRONG HEREEEEEEE expectation to small?")


	#Compute [threshold, price] pairs for symmetric equilibria
	for threshold in xrange(1, int(P.dist.Expectation())-1, int(P.dist.Expectation()) / real_step_num):
		# print "threshold: " + str(threshold)

		#Define mechanism 
		P.mechanism = threshold_mechanism(threshold)
		#Run best response dynamics
		[rv, end_profile] = BRD(P)
		#If BRD returned a symmetric equilibrium, save it
		if (not end_profile.IsEmpty()) and end_profile.p1_price == end_profile.p2_price:
			# print "A: " + str(threshold) + ", " + str(pX_price)
			bu = buyerUtility(end_profile)
			th_bu[threshold] = bu
			th_pr[threshold] = end_profile.p1_price

	#find threshold that maximizes buyer utility
	if th_bu:

		max_th_bu = max(th_bu.iteritems(), key=operator.itemgetter(1))
		min_th = max_th_bu[0]
		eq_pr = th_pr[min_th] 

		#redefine mechanism
		P.mechanism = threshold_mechanism(min_th)
		#Set eq prices
		P.p1_price = P.p2_price = eq_pr

		if P.IsEmpty():
			raise NotImplementedError("P is empty")

		[bu, p1_rev, p2_rev] = studyState(P)
		if p1_rev != p2_rev:
			print "ERROR!!! something is wrong here, equilibrium should be symmetric"
			return [None, None, None, None]
		#Print data
		# print "sc= " + str(sc) + ", th=" + str(min_th) + ", eq_price=" + str(eq_pr) + ", utility=" +str(bu) + ", rev=" + str(p_rev)

		#All data
		return [eq_pr, bu, p1_rev, min_th]
		print "----------------"
	else:
		return [None, None, None, None]


def opt_alpha(P):
	alpha_to_price = {}
	alpha_to_bu = {}
	low_alpha = -0.5
	high_alpha = 0
	for cur in frange(low_alpha, high_alpha, float(high_alpha - low_alpha) / step_num):
		#set mechanism 
		P.mechanism = exponential_mechanism(cur)
		#Run best response dynamics
		[rv, end_profile] = BRD(P)

		if (not end_profile.IsEmpty()) and end_profile.p1_price == end_profile.p2_price:
			alpha_to_price[cur] = end_profile.p1_price
			bu = buyerUtility(end_profile)
			alpha_to_bu[cur] = bu
			print "Equil. : alpha = " + str(cur) + ": price = " + str(end_profile.p1_price) + " : sc=" + str(end_profile.search_cost) + "buyer utility=" + str(bu)

	#If there is an equilibrium
	if alpha_to_price:
		#Find buyer utility maximizing 
		max_alpha_bu = max(alpha_to_bu.iteritems(), key=operator.itemgetter(1))
		# min_al_pr = min(alpha_to_price.iteritems(), key=operator.itemgetter(1))
		min_alpha = max_alpha_bu[0]
		eq_pr = alpha_to_price[min_alpha] 

		# print "All alpha to price: " + str(alpha_to_price)
		# print "min pair: " + str(min_al_pr)

		# min_alpha = min_al_pr[0]
		# min_price = min_al_pr[1]

		#redefine mechanism
		P.mechanism = exponential_mechanism(min_alpha)
		#Set equilibrium prices
		P.p1_price = P.p2_price = eq_pr
		[bu, p1_rev, p2_rev] = studyState(P)
		if p1_rev != p2_rev:
			print "ERROR!!! something is wrong here, equilibrium should be symmetric"
			print str(p1_rev) + "!=" + str(p2_rev)
			return [None, None, None, None]
		#Print data
		# print "sc= " + str(sc) + ", th=" + str(min_th) + ", eq_price=" + str(eq_pr) + ", utility=" +str(bu) + ", rev=" + str(p_rev)
		return [eq_pr, bu, p1_rev, min_alpha] #, math.exp(math.exp(-min_alpha))

	else:
		return [None, None, None, None]

def do_plotSCtoLowestPriceUsingThreshold(distributions, productionCosts, for_cycles=False):
	P = Profile(search_cost=0, p1_price=0, p2_price=0)
        P.productionCosts = productionCosts ### NEW PC

	#Define distribution: equal weights for integers 
	# P.dist = int_uniform_dist(1, 301)
	# P.dist = geometric_dist(0.01, 200)
	# P.dist = equal_revenue_dist(1000)
	# P.dist = almost_equal_revenue_dist(1000)
	# P.dist = beta_dist(2, 5, 100)

	for distr in distributions:
		P.dist = distr
	
		meta_data = ""
		if for_cycles:
			#meta data
			meta_data = "# Solution concept: Cycles \r\n"
		else:
			meta_data = "# Solution concept: Equilibria \r\n"


		#Write distribution to meta data
		meta_data += "# Distribution:" + P.dist.WhoAmI() + "\r\n"

		#Define mechanism 
		P.mechanism = threshold_mechanism(0)

		#Write distribution to meta data
		meta_data += "# Mechanism:" + P.mechanism.WhoAmI() + "\r\n"
		meta_data += "# optimal threshold per search cost \r\n"

		print "expectation = " + str(P.dist.Expectation())
		# print "dist:" 
		# for [val, prob] in P.dist.Get():
		# 	print "1- Pr[v < " + str(val) +"] =" + str( 1 - P.dist.CDF(val))

		# Returns for each search cost, the threshold that guarantees a symmetric equilibrium, 
		# and minimizes the equilibrium price  
		retArr = []
		#For each search cost
		real_step_num = float(P.dist.Expectation()) / (step_num + P.mechanism.shift)
		for sc in frange(0, int(P.dist.Expectation()) - 1,  real_step_num):
			print "----------------"
			#Set search cost (carefully!)
			P.SetSearchCost(sc)
			#Print start state
			print "Start profile: \n" + P.WhoAmI()
			#Compute optimal threshold, the induced equilibrium price, the utility and revenue in that stateS
			[eq_pr, bu, p1_rev, min_th] = opt_threshold(P)
			if eq_pr != None:
				retArr.append([sc, eq_pr, bu, p1_rev, min_th])
			else: 
				print "Search cost " + str(sc) + " has no equilibrium"

		np_arr = np.array(retArr)	
		params_labels=["Search cost", "Equilibrium price", "Utility", "Seller revenue", "Optimal threshold"]

		#Write meta data
		meta_data += "# " + str(params_labels) + "\r\n"

		printToFile(plotArray=np_arr, meta_data=meta_data)
		# plotThis1(title="Search cost to min threshold", plotArray=np_arr,params_labels=params_labels)
		# plotThis(dist=dist, mech=mech, plotArray=retArr,X_axis_label="Search costs", Y_axis_label="Sellers prices")

def do_plotSCtoMinAlphaInExponentialMech(distributions, productionCosts, for_cycles=False):

	#Define distribution: equal weights for integers 
	# P.dist = int_uniform_dist(1, 101)
	# P.dist = beta_dist(0.5, 0.5, 100)
	# P.dist = beta_dist(5, 1, 100)
	# P.dist = beta_dist(1, 3, 100)
	# P.dist = beta_dist(2, 2, 100)
	# P.dist = beta_dist(2, 5, 100)
	# P.dist = geometric_dist(0.01, 300)
	
	for distr in distributions:
		#Empty profile
		P = Profile(search_cost=0, p1_price=0, p2_price=0)
		P.dist = distr

		P.productionCosts = productionCosts ### NEW PC

		meta_data = ""
		if for_cycles:
			#meta data
			meta_data = "# Solution concept: Cycles \r\n"
		else:
			meta_data = "# Solution concept: Equilibria \r\n"
	
		#Write distribution to meta data
		meta_data += "# Distribution:" + P.dist.WhoAmI() + "\r\n"

		#set mechanism 
		P.mechanism = exponential_mechanism(0)

		#Write distribution to meta data
		meta_data += "# Mechanism:" + P.mechanism.WhoAmI() + "\r\n"
		meta_data += "# optimal alpha per search cost \r\n"

		#Returns for each search cost, the threshold that guarantees a symmetric equilibrium, and minimizes the equilibrium price  
		retArr = []
		#For each search cost
		real_step_num = float(P.dist.Expectation()) / (step_num + P.mechanism.shift)
		for sc in frange(0, (int(P.dist.Expectation()) - 1),  real_step_num):
			#Set search cost (carefully!)
			P.SetSearchCost(sc)

			[min_price, bu, p1_rev, min_alpha] = opt_alpha(P)
			if min_price != None:
				#All data
				retArr.append([sc, min_price, bu, p1_rev, min_alpha]) #, math.exp(math.exp(-min_alpha))
			else:
				print "Search cost " + str(sc) + " has no equilibrium"

		np_arr = np.array(retArr)
		params_labels=["Search cost", "Equilibrium price", "Utility", "Seller revenue", "optimal alpha"]

		#Write meta data
		meta_data += "# " + str(params_labels) + "\r\n"

		printToFile(plotArray=np_arr, meta_data=meta_data)
		# plotThis1(title="Search cost to min alpha. dist=" + P.dist.WhoAmI(), plotArray=np_arr,params_labels=params_labels)
		# plotThis(dist=dist, mech=mech, plotArray=retArr,X_axis_label="Search costs", Y_axis_label="Sellers prices")

def do_plotSearchCostVsEqPrice(distributions, mechanisms, productionCosts, for_cycles=False): ### NEW PC
	print "In do_plotSearchCostVsEqPrice"
	for mechan in mechanisms:
		print "In Mechanism "
		print mechan.WhoAmI()
		for distr in distributions:
			#Empty profile
			P = Profile(search_cost=0, p1_price=30, p2_price=30)

                        P.productionCosts = productionCosts ### NEW PC
                        
			P.dist = distr

			supp_size = len(P.dist.Supp())

			# starting_points = [ [P.dist.Supp()[supp_size / 10], P.dist.Supp()[supp_size / 10]], 
			# 					[P.dist.Supp()[supp_size / 3], P.dist.Supp()[supp_size / 3]],
			# 					[P.dist.Supp()[supp_size / 10], P.dist.Supp()[supp_size / 3]], 
			# 				]
			# for [s1, s2] in starting_points:
			# 	P.p1_price = s1
			# 	P.p2_price = s2
			if for_cycles:
				#meta data
				meta_data = "# Solution concept: Cycles \r\n"
			else:
				meta_data = "# Solution concept: Equilibria \r\n"

			# meta_data = "# Starting point: " + str(s1) + ", " + str(s2) + "\r\n"

			#Write distribution to meta data
			meta_data += "# Distribution:" + P.dist.WhoAmI() + "\r\n"

			#Define mechanism 
			# P.mechanism = Low_first_mechanism()
			# P.mechanism = u_random_mechanism()
			# P.mechanism = weighted_random_mechanism(0.5, 0.5)
			# P.mechanism = threshold_mechanism(8)
			# P.mechanism = exponential_mechanism(-2)

			P.mechanism = mechan
			#Write distribution to meta data
			meta_data += "# Mechanism:" + P.mechanism.WhoAmI() + "\r\n"

			#Print start state
			print "Start profile: \n" + P.WhoAmI()

			#Returns for each search cost, a pair of equilibrium prices if exist, and the buyer utility at equilibrium
			retArr = []
			shift = 0

			#For each search cost
			real_step_num = float(P.dist.Expectation()) / (step_num + mechan.shift)
			for sc in frange(0, int(P.dist.Expectation()) - 1, real_step_num):
				#Set search cost (carefully!)
				P.SetSearchCost(sc)
				print "----------------"
				print "Start state:\n" + P.WhoAmI()


				if for_cycles:
					[rv, outcome] = BRD(P, cycle_special=for_cycles)
					if rv == "cycle":
						[ave_bu, ave_p1_rev, ave_p2_rev, ave_pr1, ave_pr2, tot, cycle_str] = outcome
						# [cycle_ave_bu, cyc_len, cycle_str] = outcome
						print "cycle length:" + str(tot) 

						# #All data
						retArr.append([sc, ave_bu, ave_p1_rev, ave_p2_rev, ave_pr1, ave_pr2, tot]) # , cycle_str
					else: 
						continue
				else:
					#Run best response dynamics
					[rv, end_profile] = BRD(P)

					#If BRD returned a symmetric equilibrium
					if not end_profile.IsEmpty() and end_profile.p1_price == end_profile.p2_price: #[pX_id, pX_price, pY_id, pY_price]
						[bu, p1_rev, p2_rev] = studyState(end_profile)
						if p1_rev != p2_rev:
							print end_profile.WhoAmI()
							print "revenues" + str(p1_rev) + ", " + str(p2_rev)
							raise NotImplementedError("Prices should be equal")
						#All data
						retArr.append([sc, end_profile.p1_price, bu, p1_rev])
					print "----------------"

			np_arr = np.array(retArr)
			params_labels = []
			if for_cycles:
				params_labels=["Search cost", "Buyer average Utility", 
							"Seller 1 average Revenue", "Seller 2 average Revenue", 
							"Seller 1 average Price", "Seller 2 average Price", 
							"Cycle length"] # , "Cycle description"
			else:
				params_labels=["Search cost", "Equilibrium price", "Utility", "Seller revenue"]
			
			#Write meta data
			meta_data += "# " + str(params_labels) + "\r\n"

			printToFile(plotArray=np_arr, meta_data=meta_data)
			# plotThis1(title="Search cost to stuff. " + P.mechanism.WhoAmI(), plotArray=np_arr,params_labels=params_labels)

			# plotThis(dist=dist, mech=mech, plotArray=retArr, X_axis_label="Search costs", Y_axis_label="Sellers prices")

class file_ID:
	def __init__(self, dist_identifier, mechanism_identifier, cycle_or_equilibrium, headers_to_plot, opt_par_file=None):
		self.dist_identifier = dist_identifier 
		self.mechanism_identifier = mechanism_identifier
		self.cycle_or_equilibrium = cycle_or_equilibrium
		self.opt_par_file = opt_par_file
		self.headers_to_plot = headers_to_plot

	def WhoAmI(self):
		return ("File id. \n " 
			+ self.dist_identifier
			+ "\n" + self.mechanism_identifier
			+ "\n" + self.cycle_or_equilibrium
			+ "\n" + self.opt_par_file
			+ "\n" + self.headers_to_plot)

	def IsSatisfied(self, dist_identifier, mechanism_identifier, cycle_or_equilibrium, opt_par_file):
		return ( (self.dist_identifier in dist_identifier) and 
					(self.mechanism_identifier in mechanism_identifier) and 
					(self.cycle_or_equilibrium in cycle_or_equilibrium) and 
					(self.opt_par_file == opt_par_file) 
				)


def extract_data_new(list_of_file_IDs):
	all_data = []
	#Assuming os is in the right directory
	for filename in sorted(glob.glob("*.txt")):
		with open(filename , "r") as f:
			#Parse metadata
			[eq_or_cyc, opt_par_file, distribution, [mechanism, mech_pars], headers] = parse_metadata(filehandle=f)
			#If file contains requested data
			for file_ID in list_of_file_IDs:
				if file_ID.IsSatisfied(dist_identifier=distribution, 
					mechanism_identifier=mechanism, 
					cycle_or_equilibrium=eq_or_cyc, 
					opt_par_file=opt_par_file):
					cur_data = []
					print
					print "filename: " + filename
					print "mechanism:" + mechanism
					print "mech pars:" + str(mech_pars)
	 				print "headers: " +str(headers)
					print 
					headers_to_plot_indices = [headers.index(header) for header in file_ID.headers_to_plot]
					# print headers_to_plot_indices
					
					#Start reading data from file
					np_arr = np.loadtxt(f, delimiter = ',', usecols = range(len(headers)))
					if np_arr.size > 0:
						# print np_arr
						# print headers_to_plot_indices
						good_cols = np_arr[:,headers_to_plot_indices]
						all_data.append([distribution, [mechanism, mech_pars], file_ID.headers_to_plot, good_cols])
	return all_data


def extract_data(dist_identifier, cycle_or_equilibrium, is_opt_par_file, headers_to_plot, mechanism_identifier=all_mechanism_identifiers):
	all_data = []
	#Assuming os is in the right directory
	for filename in glob.glob("*.txt"):
		with open(filename , "r") as f:
			#Parse metadata
			[eq_or_cyc, opt_par_file, distribution, [mechanism, mech_pars], headers] = parse_metadata(filehandle=f)
			#If file contains requested data
			# print mechanism
			if ( (dist_identifier in distribution) and 
				(cycle_or_equilibrium in eq_or_cyc) and 
				(mechanism in mechanism_identifier) and 
				( (opt_par_file != None) == is_opt_par_file)
				):
				cur_data = []
				print
				print "filename: " + filename
				print "mechanism:" + mechanism
				print "mech pars:" + str(mech_pars)
 				print "headers: " +str(headers)
				print 
				headers_to_plot_indices = [headers.index(header) for header in headers_to_plot]
				# print headers_to_plot_indices
				
				#Start reading data from file
				np_arr = np.loadtxt(f, delimiter = ',', usecols = range(len(headers)))
				if np_arr.size > 0:
					# print np_arr
					# print headers_to_plot_indices
					good_cols = np_arr[:,headers_to_plot_indices]
					all_data.append([distribution, [mechanism, mech_pars], headers_to_plot, good_cols])

	return all_data

#Global: markers to be used in graphs 
mech_color = {"LF" : "red", "Exponential" : "green" , "Threshold" : "blue" , "Uniform Random" : "purple", "A" : "orange"}# , "B" : "brown"  
mech_marker ={"LF" : "o", "Exponential" : "v", "Threshold" : '^', "Uniform Random" : '*', "A" : "s"} # , , "B" : "x"

num_to_mech = {0 : "Uniform Random", 1 : "Exponential", 2 : "Threshold", 3 : "A"   }

#Plot_cap = percentage (from 0 to 1) of search costs you want to draw
def plot_new(plthandler, all_data, plot_cap, exclude_mechs=[], expressive_label=False):
	#plot cycles data
	for idx1, [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in enumerate(all_data):
		if not (mechanism in exclude_mechs):
			set_label = mechanism 
			# print mechanism
			set_color = mech_color[mechanism]
			set_marker = mech_marker[mechanism]
			#Expressive label just iterates over colors instead
			if expressive_label:
				set_color = mech_color[num_to_mech[idx1]]
				set_marker = mech_marker[num_to_mech[idx1]]
				for key, value in mech_pars.iteritems():
					set_label += " " + key + "=" + value

			#First header is always search cost
			for (idx, lab) in enumerate(headers_to_plot[1:]):
				search_costs = good_cols[:,0]
				other_column = good_cols[:,idx+1] 
				cap = int(len(search_costs) * plot_cap) 
				plthandler.plot(search_costs[:cap], other_column[:cap], 
					label=set_label, 
					color=set_color,
					marker=set_marker, 
					linestyle="None")



def comp_social_welfare(dist_identifier):

	fig1 = plt.figure()
	fig1.canvas.set_window_title("social welfare" + " , " + dist_identifier)


	cycle_or_equilibrium = "Cycles"
	# Search cost should always appear
	headers_to_plot = ["Search cost", "Buyer average Utility", "Seller 1 average Revenue", "Seller 2 average Revenue"]
	# headers_to_plot = ["Search cost", "Buyer average Utility"] #
	all_cyc_data = extract_data(dist_identifier=dist_identifier, 
		cycle_or_equilibrium=cycle_or_equilibrium, 
		is_opt_par_file=True,
		headers_to_plot=headers_to_plot)

	all_cyc_sw = []

	#Create average social welfare
	for [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in all_cyc_data:
		print "mech: " + mechanism
		# print good_cols
		#sw_cols = list of [sc, sw]
		#sw = bau + p1_rev + p2_rev
		np_sw = np.array([ [sc, bau + p1_rev + p2_rev]  for [sc, bau, p1_rev, p2_rev] in good_cols])
		# print np_sw

		all_cyc_sw.append([distribution, [mechanism, mech_pars], ["Search cost", "Social Welfare"], np_sw])

	plot_new(plthandler=plt, all_data=all_cyc_sw, plot_cap=1)


	cycle_or_equilibrium = "Equilibria"
	#Search cost should always appear
	#Mechanisms with optimal parameter (Threshold, alpha)
	headers_to_plot = ["Search cost", "Utility", "Seller revenue"] 
	all_eq_data = extract_data(dist_identifier=dist_identifier, 
				cycle_or_equilibrium=cycle_or_equilibrium, 
				is_opt_par_file=False,
				headers_to_plot=headers_to_plot)

	# print all_eq_data

	all_eq_sw = []

	#Create equilibrium social welfare
	for [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in all_eq_data:
		# print good_cols
		#sw_cols = list of [sc, sw]
		#sw = bau + p1_rev + p2_rev
		np_sw = np.array([ [sc, bu + 2 * p1_rev]  for [sc, bu, p1_rev] in good_cols])
		all_eq_sw.append([distribution, [mechanism, mech_pars], ["Search cost", "Social Welfare"], np_sw])


	list_of_file_IDs = [file_ID(dist_identifier=dist_identifier, 
						mechanism_identifier="Uniform Random", 
						cycle_or_equilibrium = "Equilibria", 
						headers_to_plot=headers_to_plot, opt_par_file=None)					
	]


	plot_new(plthandler=plt, all_data=all_eq_sw, exclude_mechs=["LF"], plot_cap=1)

	all_uniform_data = extract_data_new(list_of_file_IDs)

	all_eq_sw = []

	#Create equilibrium social welfare
	for [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in all_uniform_data:
		# print good_cols
		#sw_cols = list of [sc, sw]
		#sw = bau + p1_rev + p2_rev
		np_sw = np.array([ [sc, bu + 2 * p1_rev]  for [sc, bu, p1_rev] in good_cols])
		all_eq_sw.append([distribution, [mechanism, mech_pars], ["Search cost", "Social Welfare"], np_sw])


	plot_new(plthandler=plt, all_data=all_eq_sw, plot_cap=1)



	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()


def comp_cycle_len(plot_cap=1):
	fig1 = plt.figure()
	fig1.canvas.set_window_title("Cycle length")

	cycle_or_equilibrium = "Cycles"

	dist_identifiers = ["Integer uniform distribution, low=1,  high=101", 
					"Beta distribution, parameter=0.5,0.5,100", 
					"Geometric distribution, parameter=0.01, precision=400"
					]
	headers_to_plot = ["Search cost", "Cycle length"]

	for dist_identifier in dist_identifiers:
		all_cyc_data = extract_data(dist_identifier=dist_identifier, 
			cycle_or_equilibrium=cycle_or_equilibrium, 
			is_opt_par_file=False,
			headers_to_plot=headers_to_plot)

		for [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in all_cyc_data:
			#First header is always search cost
			for (idx, lab) in enumerate(headers_to_plot[1:]):
				search_costs = good_cols[:,0]
				other_column = good_cols[:,idx+1] 
				cap = int(len(search_costs) * plot_cap) 
				plt.plot(search_costs[:cap], other_column[:cap], label=distribution, marker=markers[idx], linestyle="None")


	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()


def comp_opt_alpha(plot_cap=1):
	fig1 = plt.figure()
	fig1.canvas.set_window_title("Optimal alpha in the exponential mechanism")

	cycle_or_equilibrium = "Equilibria"
	dist_identifiers = ["Integer uniform distribution, low=1,  high=101"
	# , 
	# 				"Beta distribution, parameter=0.5,0.5,100", 
	# 				"Geometric distribution, parameter=0.01, precision=400"
					]

	headers_to_plot = ["Search cost", "optimal alpha"] 
	for dist_identifier in dist_identifiers:
		all_eq_data = extract_data(dist_identifier=dist_identifier, 
									cycle_or_equilibrium=cycle_or_equilibrium, 
									is_opt_par_file=True,
									headers_to_plot=headers_to_plot, 
									mechanism_identifier="Exponential"
									)
		for idx1, [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in enumerate(all_eq_data):
			set_label = distribution 
			# print mechanism
			#First header is always search cost
			for (idx, lab) in enumerate(headers_to_plot[1:]):
				set_marker = mech_marker.values()[idx1]
				set_color = mech_color.values()[idx1]
				search_costs = good_cols[:,0]
				other_column = good_cols[:,idx+1] 
				cap = int(len(search_costs) * plot_cap) 
				plt.plot(search_costs[:cap], other_column[:cap], color=set_color, 
					label=set_label, 
					marker=set_marker, 
					linestyle="None")

	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()



def comp_utility(dist_identifier, plot_cap=1):
	cycle_or_equilibrium = "Cycles"
	headers_to_plot = ["Search cost", "Buyer average Utility"] #
	all_cyc_data = extract_data(dist_identifier=dist_identifier, 
		cycle_or_equilibrium=cycle_or_equilibrium, 
		is_opt_par_file=False,
		headers_to_plot=headers_to_plot)

	cycle_or_equilibrium = "Equilibria"
	headers_to_plot = ["Search cost", "Utility"]
	all_eq_data = extract_data(dist_identifier=dist_identifier, 
			cycle_or_equilibrium=cycle_or_equilibrium, 
			is_opt_par_file=False,
			headers_to_plot=headers_to_plot)

	fig1 = plt.figure()
	fig1.canvas.set_window_title(headers_to_plot[1] + " , " + dist_identifier)

	plot_new(plthandler=plt, all_data=all_cyc_data, plot_cap=plot_cap)
	plot_new(plthandler=plt, all_data=all_eq_data, exclude_mechs=["LF"], plot_cap=plot_cap)

	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()


def comp_eq_price(dist_identifier, plot_cap=1):
	cycle_or_equilibrium = "Cycles"
	headers_to_plot = ["Search cost", "Seller 1 average Price", "Seller 2 average Price"] #
	all_cyc_data = extract_data(dist_identifier=dist_identifier, 
					cycle_or_equilibrium=cycle_or_equilibrium, 
					is_opt_par_file=True,
					headers_to_plot=headers_to_plot)

	all_cyc_pr = []
	#Create average seller price
	for [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in all_cyc_data:
		np_ave_pr = np.array([ [sc, float(p1_ave_pr + p2_ave_pr) / 2]  for [sc, p1_ave_pr, p2_ave_pr] in good_cols])
		all_cyc_pr.append([distribution, [mechanism, mech_pars], ["Search cost", "Sellers' average price"], np_ave_pr])



	cycle_or_equilibrium = "Equilibria"
	headers_to_plot = ["Search cost", "Equilibrium price"]
	all_eq_data = extract_data(dist_identifier=dist_identifier, 
		cycle_or_equilibrium=cycle_or_equilibrium, 
		is_opt_par_file=True,
		headers_to_plot=headers_to_plot)



	list_of_file_IDs = [file_ID(dist_identifier=dist_identifier, 
						mechanism_identifier="Uniform Random", 
						opt_par_file=None,
						cycle_or_equilibrium = "Equilibria", 
						headers_to_plot=headers_to_plot)
	# ,
	# 					file_ID(dist_identifier=dist_identifier, 
	# 					mechanism_identifier="Exponential", 
	# 					cycle_or_equilibrium = "Equilibria", 
	# 					headers_to_plot=headers_to_plot, opt_par_file=None), 
						# file_ID(dist_identifier=dist_identifier, 
						# mechanism_identifier="Threshold", 
						# cycle_or_equilibrium = "Equilibria", 
						# headers_to_plot=headers_to_plot, opt_par_file="threshold"), 
	]

	all_data = extract_data_new(list_of_file_IDs)



	fig1 = plt.figure()
	plt.margins(0.02)
	fig1.canvas.set_window_title(headers_to_plot[1] + " , " + dist_identifier)

	plot_new(plthandler=plt, all_data=all_cyc_pr, plot_cap=plot_cap)
	plot_new(plthandler=plt, all_data=all_eq_data, exclude_mechs=["LF"], plot_cap=plot_cap)
	plot_new(plthandler=plt, all_data=all_data, exclude_mechs=["LF"], plot_cap=plot_cap)

	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()



def comp_seller_revenue(dist_identifier, plot_cap=1):
	cycle_or_equilibrium = "Cycles"
	headers_to_plot = ["Search cost", "Seller 1 average Revenue", "Seller 2 average Revenue"] #
	all_cyc_data = extract_data(dist_identifier=dist_identifier, 
		cycle_or_equilibrium=cycle_or_equilibrium, 
		is_opt_par_file=True,
		headers_to_plot=headers_to_plot)

	all_cyc_rev = []

	#Create average seller revenue
	for [distribution, [mechanism, mech_pars], headers_to_plot, good_cols] in all_cyc_data:
		np_ave_rev = np.array([ [sc, float(p1_ave_rev + p2_ave_rev) / 2]  for [sc, p1_ave_rev, p2_ave_rev] in good_cols])
		all_cyc_rev.append([distribution, [mechanism, mech_pars], ["Search cost", "Sellers' average revenue"], np_ave_rev])



	cycle_or_equilibrium = "Equilibria"
	headers_to_plot = ["Search cost", "Seller revenue"]
	all_eq_data = extract_data(dist_identifier=dist_identifier, 
		cycle_or_equilibrium=cycle_or_equilibrium, 
		is_opt_par_file=True,
		headers_to_plot=headers_to_plot)

	fig1 = plt.figure()
	fig1.canvas.set_window_title(headers_to_plot[1] + " , " + dist_identifier)

	plot_new(plthandler=plt, all_data=all_cyc_rev, plot_cap=plot_cap)
	plot_new(plthandler=plt, all_data=all_eq_data, exclude_mechs=["LF"], plot_cap=plot_cap)

	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()


def comp_exp_spec(dist_identifier, plot_cap=1):
	cycle_or_equilibrium = "Equilibria"
	headers_to_plot = ["Search cost", "Equilibrium price"]
	mech_identifiers = ["Exponential"]

	list_of_file_IDs = [file_ID(dist_identifier=dist_identifier, 
						mechanism_identifier="Uniform Random", 
						opt_par_file=None,
						cycle_or_equilibrium = "Equilibria", 
						headers_to_plot=headers_to_plot),
						file_ID(dist_identifier=dist_identifier, 
						mechanism_identifier="Exponential", 
						cycle_or_equilibrium = "Equilibria", 
						headers_to_plot=headers_to_plot, opt_par_file=None), 
						# file_ID(dist_identifier=dist_identifier, 
						# mechanism_identifier="Threshold", 
						# cycle_or_equilibrium = "Equilibria", 
						# headers_to_plot=headers_to_plot, opt_par_file="threshold"), 
	]

	all_data = extract_data_new(list_of_file_IDs)
	
	fig1 = plt.figure()
	fig1.canvas.set_window_title("Specific exponential mechanisms, and Uniform Random" + " , " + dist_identifier)

	plot_new(plthandler=plt, all_data=all_data, plot_cap=plot_cap, expressive_label=True)
	# plot_new(plthandler=plt, all_data=all_thresh_data, plot_cap=plot_cap, expressive_label=True)

	plt.legend(loc=legend_loc, prop={'size': legend_size})
	plt.xlabel("Search cost")
	plt.show()





legend_loc=7 # center right
# legend_loc=1 # top right
# legend_loc=2 # top left
# legend_loc=4 # bottom right
# legend_loc=3 # bottom left
# legend_loc=9 # top center

def compare_mechanisms():

	# cycle_or_equilibrium = "Cycles"
	dist_identifier = "Integer uniform distribution, low=1,  high=101"
	# dist_identifier = "Beta distribution, parameter=0.5,0.5,100"	
	# dist_identifier = "Beta distribution, parameter=5,1,100"
	# dist_identifier = "Beta distribution, parameter=1,3,100"		
	# dist_identifier = "Beta distribution, parameter=2,2,210"
	# dist_identifier = "Beta distribution, parameter=2,5,130"
	# dist_identifier = "Equal revenue distribution, precision=200"
	# dist_identifier = "Geometric distribution, parameter=0.01, precision=500"
	# dist_identifier = "Almost Equal revenue distribution, precision=300"

	#comp_social_welfare(dist_identifier=dist_identifier)
	# return

	#comp_eq_price(dist_identifier=dist_identifier)

	#comp_seller_revenue(dist_identifier=dist_identifier)

	comp_utility(dist_identifier=dist_identifier)
	# return

	# comp_cycle_len()
	# return

	# comp_opt_alpha()
	# return

	# comp_exp_spec(dist_identifier=dist_identifier)
	return 



def main():
	#Input (use one of the following flags on the command line)
	parser = argparse.ArgumentParser(description='Consumer search simulation')
	parser.add_argument('-r','--reservePrice', action="store_true", help='run res_price', required=False)
	parser.add_argument('-inFile','--inFile', dest="file_to_plot", help='plot this file', required=False)
	parser.add_argument('-PF','--plotFile', action="store_true", help='plot file under flag "inFile" ', required=False)
	parser.add_argument('-t','--test', action="store_true", help='test all functions', required=False)
	parser.add_argument('-p1','--plotSearchCostVsEqPrice', action="store_true", help='plot SC to prices', required=False)
	parser.add_argument('-p2','--plotThresholdVsEqPrice', action="store_true", help='plot Threshold to prices', required=False)
	parser.add_argument('-thresh','--plotSCtoLowestPriceUsingThreshold', action="store_true", help='plot SC to (eq prices, threshold) for Lowest (Symmetric equilibrium) Price Using a Threshold mechanism', required=False)
	parser.add_argument('-exp','--plotSCtoMinAlphaInExponentialMech', action="store_true", help='plot SC to (alpha, price) for Lowest (Symmetric equilibrium) Price Using a alha-exponential mechanism', required=False)
	parser.add_argument('-LFU_eq','--plotSCtoLFandUniformEquilibria', action="store_true", help='plot SC to (alpha, price) for Lowest (Symmetric equilibrium) Price Using a alha-exponential mechanism', required=False)
	parser.add_argument('-LFU_cyc','--plotSCtoLFandUniformCycles', action="store_true", help='plot SC to (alpha, price) for Lowest (Symmetric equilibrium) Price Using a alha-exponential mechanism', required=False)
	parser.add_argument('-exp_spec','--plotSCtoEqPriceSpecExp', action="store_true", help='plot SC to price for Symmetric equilibrium Price for specific alphas', required=False)


	# parser.add_argument('-p7','--plotSCtoMechanisms', action="store_true", help='plot SC to opt eq price in mechanisms', required=False)
	parser.add_argument('-all','--doAllRuns', action="store_true", help='Exahustively create data', required=False)
	parser.add_argument('-compare','--compareMechanisms', action="store_true", help='plot a comparison', required=False)

	args = vars(parser.parse_args())

	testcase_distributions = [
					# beta_dist(0.5, 0.5, 100) 
					# beta_dist(5, 1, 100) 
					# beta_dist(1, 3, 100) 
					# beta_dist(2, 2, 200) 
					# beta_dist(2, 5, 200) 
					# geometric_dist(0.01, 400)
					# ,geometric_dist(0.02, 300)
					int_uniform_dist(1, 101)
					# int_uniform_dist(1, 1001)
					# int_uniform_dist(400, 600)
					# equal_revenue_dist(200)
					# ,almost_equal_revenue_dist(100) 
					# almost_equal_revenue_dist(300)
					]

	#The mechanisms generally at interest for sc versus price
	mechanisms = [Low_first_mechanism(), u_random_mechanism()]
	#mechanisms = [u_random_mechanism()]

	productionCosts = [0,0] ### NEW PC
#-----
	if args['plotFile']:
		print "file to plot:" + args['file_to_plot']
		plotFile(filename=args['file_to_plot'])

	if args['reservePrice']:
		print "Reservation price: " + str(res_price(dist=dist, search_cost=search_cost))

# 
	if args['test']:
		#Empty profile
		P = Profile()

		#meta data is what will be printed to an output file
		meta_data = "" 

		#Define search cost
		P.SetSearchCost(5)

		### NEW PC
		P.productionCosts = [10,5]

		#Write search cost to meta data
		meta_data += "# search cost =  \r\n"

		#Define setting (uncomment the distribution you want to use)
		P.dist = int_uniform_dist(1, 101)
		#P.dist = int_uniform_dist(1, 11)
		#P.dist = geometric_dist(0.01, 400)
		#P.dist = beta_dist(2, 5, 100)
		#P.dist = beta_dist(5, 1, 100)
		#P.dist = almost_equal_revenue_dist(1000)
		#P.dist =  equal_revenue_dist(200)
                
		# print str(dist.Get())
		
		#Write distribution to meta data
		meta_data += "# Distribution:" + P.dist.WhoAmI() + "\r\n"

		#Define starting state
		P.p1_price=0
		P.p2_price=0

		#Define mechanism (uncomment the mechanism you want to use)
		#P.mechanism = Low_first_mechanism()
		#P.mechanism = weighted_random_mechanism(0.7, 0.3)
		#P.mechanism = weighted_random_mechanism(0.5, 0.5)
		#P.mechanism = u_random_mechanism()
		#P.mechanism = threshold_mechanism(40)
		P.mechanism = exponential_mechanism(-0.4)

		#Write distribution to meta data
		meta_data += "# Mechanism:" + P.mechanism.WhoAmI() + "\r\n"

		print P.mechanism.WhoAmI()
                
		#Print start state
		print "Start profile: \n" + P.WhoAmI()
                
		print "Buyer utility at start profile: " + str(buyerUtility(P))
                
		# #Compute reservation price
		# r_price = res_price(dist=dist, search_cost=search_cost)
		# print "Reservation price: " + str(r_price)

		# print "1 best response: " + str(BR(profile=P, my_id=1))
		# print "2 best response: " + str(BR(profile=P, my_id=2))
		[rv, end_profile] = BRD(P, expressive=True)
		if not end_profile.IsEmpty():
			[bu, p1_rev, p2_rev] = studyState(end_profile)
			print "Equil. buyer utility: " + str(bu)
			print "Equi. p1 rev:" + str(p1_rev)
			print "Equi. p2 rev:" + str(p2_rev)
		else: 
			print "No equilibrium!"




	if args['plotSearchCostVsEqPrice']:
		#Define distribution: equal weights for integers 
		# P.dist = int_uniform_dist(1, 101)
		# P.dist = geometric_dist(0.01, 1000)
		# P.dist = equal_revenue_dist(200)
		# P.dist = almost_equal_revenue_dist(1000)
		# P.dist = beta_dist(2, 5, 100)

		# exp_spec_distributions = [int_uniform_dist(1, 101), geometric_dist(0.01, 400), beta_dist(5, 5, 100)]
		# exp_mechanisms = [exponential_mechanism(-0.25), exponential_mechanism(-0.5), exponential_mechanism(-0.75)]

		exp_spec_distributions = [beta_dist(2, 2, 210)] #geometric_dist(0.01, 500) # int_uniform_dist(1, 101) # beta_dist(2, 5, 130) # equal_revenue_dist(500)
		exp_mechanisms = [
						u_random_mechanism(),
						exponential_mechanism(-0.1), 
						exponential_mechanism(-0.2),
						# exponential_mechanism(-0.25),
						exponential_mechanism(-0.3)
						]
                ### NEW PC
		do_plotSearchCostVsEqPrice(distributions=exp_spec_distributions, mechanisms=exp_mechanisms, productionCosts=productionCosts, for_cycles=False)

		# do_plotSearchCostVsEqPrice(distributions=testcase_distributions, mechanisms=mechanisms, for_cycles=True)
	
	if args['plotThresholdVsEqPrice']:
		#Empty profile
		P = Profile(search_cost=0, p1_price=0, p2_price=0)

		P.productionCosts = productionCosts ### NEW PC

		meta_data = ""
		#Define distribution: equal weights for integers 
		# P.dist = int_uniform_dist(1, 101)
		# P.dist = geometric_dist(0.02, 1000)
		P.dist = equal_revenue_dist(200)

		#Write distribution to meta data
		meta_data += "# Distribution:" + P.dist.WhoAmI() + "\r\n"

		#Define mechanism 
		P.mechanism = threshold_mechanism(0)

		#Write distribution to meta data
		meta_data += "# Mechanism:" + P.mechanism.WhoAmI() + "\r\n"
		meta_data += "# iterating over many thresholds per search cost \r\n"

		

		#Returns for each threshold, a pair of equilibrium prices if exist 
		retArr = []
		#For each search cost
		real_step_num = float(P.dist.Expectation()) / step_num
		for sc in frange(0, int(P.dist.Expectation()) - 1, real_step_num): #
			P.p1_price=10
			P.p2_price=10
			len_retArr = len(retArr)

			#Define search cost
			P.SetSearchCost(sc) 
			#For each threshold
			for threshold in frange(1, int(P.dist.Expectation())-1, P.dist.Expectation() / step_num):
				print "----------------"
				#Define mechanism 
				P.mechanism = threshold_mechanism(threshold)

				#Print start state
				print "Start profile: \n" + P.WhoAmI()

				#Run best response dynamics
				[rv, end_profile] = BRD(P)

				#If BRD returned an equilibrium
				if not end_profile.IsEmpty(): #[pX_id, pX_price, pY_id, pY_price]
					[bu, p1_rev, p2_rev] = studyState(end_profile)
					#All data
					retArr.append([sc, threshold, end_profile.p1_price, end_profile.p2_price, bu, p1_rev, p2_rev])
				print "----------------"

			if len(retArr) == len_retArr:
				print "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n no equilibrium at all!"



		np_arr = np.array(retArr)	
		if np_arr.size == 0:
			print "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n no equilibrium at all!"
			return
		params_labels=["Search cost", "Thresholds", "Eq price 1", "Eq price 2", "Utility", "rev 1", "rev 2"]

		#Write meta data
		meta_data += "# " + str(params_labels) + "\r\n"

		# printToFile(plotArray=np_arr, meta_data=meta_data)
		plotThis1(title="Threshold to stuff. " + P.mechanism.WhoAmI(), plotArray=np_arr,params_labels=params_labels)


	if args['plotSCtoLowestPriceUsingThreshold']: ### NEW PC
		do_plotSCtoLowestPriceUsingThreshold(distributions=testcase_distributions, productionCosts=productionCosts)


	if args['plotSCtoMinAlphaInExponentialMech']: ### NEW PC
		do_plotSCtoMinAlphaInExponentialMech(distributions=testcase_distributions, productionCosts=productionCosts)

	if args['plotSCtoLFandUniformEquilibria']:
		print "\n\n\n\n do_plotSearchCostVsEqPrice FALSE\n\n\n\n"

                ### NEW PC
		do_plotSearchCostVsEqPrice(distributions=testcase_distributions, mechanisms=mechanisms, productionCosts=productionCosts, for_cycles=False)

	if args['plotSCtoLFandUniformCycles']:
		# # # LF and uniform random (cycles)
		print "\n\n\n\n do_plotSearchCostVsEqPrice TRUE \n\n\n\n"

		### NEW PC
		do_plotSearchCostVsEqPrice(distributions=testcase_distributions, mechanisms=mechanisms, productionCosts=productionCosts, for_cycles=True)


	if args['doAllRuns']: ### NEW PC
		# # LF and uniform random (equilibria)
		print "\n\n\n\n do_plotSearchCostVsEqPrice FALSE\n\n\n\n"
		do_plotSearchCostVsEqPrice(distributions=testcase_distributions, mechanisms=mechanisms, productionCosts=productionCosts, for_cycles=False)
		# # # LF and uniform random (cycles)
		print "\n\n\n\n do_plotSearchCostVsEqPrice TRUE \n\n\n\n"
		do_plotSearchCostVsEqPrice(distributions=testcase_distributions, mechanisms=mechanisms, productionCosts=productionCosts, for_cycles=True)
		#  # threshold
		print "\n\n\n\n do_plotSCtoLowestPriceUsingThreshold \n\n\n\n"
		do_plotSCtoLowestPriceUsingThreshold(distributions=testcase_distributions, productionCosts=productionCosts)
		# exponential
		print "\n\n\n\n do_plotSCtoMinAlphaInExponentialMech \n\n\n\n"
		do_plotSCtoMinAlphaInExponentialMech(distributions=testcase_distributions, productionCosts=productionCosts)








	if args['compareMechanisms']:
		compare_mechanisms()

	# if args['plotSCtoMechanisms']:
	# 	#Empty profile
	# 	P = Profile(search_cost=0, p1_price=0, p2_price=0)

	# 	meta_data = ""
	# 	#Define distribution: equal weights for integers 
	# 	P.dist = int_uniform_dist(1, 161)
	# 	# P.dist = beta_dist(0.5, 0.5, 100)
	# 	# P.dist = geometric_dist(0.02, 100)
	
	# 	#Write distribution to meta data
	# 	meta_data += "# " + P.dist.WhoAmI() + "\r\n"

	# 	#Write to meta data
	# 	meta_data += "#  Comparing mechanisms \r\n"
		
	# 	#Returns for each search cost, the threshold that guarantees a symmetric equilibrium, and minimizes the equilibrium price  
	# 	retArr = []

	# 	#For each search cost
	# 	for sc in xrange(50, (int(P.dist.Expectation()) - 1), int(P.dist.Expectation()) - 1 / step_num):
	# 		#Set search cost (carefully!)
	# 		P.SetSearchCost(sc)

	# 		print "\n running For threshold \n"

	# 		#Compute optimal threshold, the induced equilibrium price, the utility and revenue in that stateS
	# 		[eq_pr_thresh, bu_thresh, p1_rev_thresh, min_th] = opt_threshold(P)

	# 		print "\n running mechanism:" + str(P.mechanism.WhoAmI()) + "\n"

	# 		[min_price_exp, bu_exp, p1_rev_exp, min_alpha] = opt_alpha(P)


	# 		#Define mechanism 
	# 		P.mechanism = u_random_mechanism()


	# 		print "\n running mechanism:" + str(P.mechanism.WhoAmI()) + "\n"

	# 		#Run best response dynamics
	# 		[rv, end_profile] = BRD(P, expressive=True)

	# 		if rv == "profile":
	# 			[bu_rand, p1_rev_rand, p2_rev_rand] = studyState(end_profile)
	# 			if p1_rev_rand != p2_rev_rand:
	# 				print "ERROR!!! something is wrong here, equilibrium should be symmetric"
	# 				# return None

	# 		#Define mechanism 
	# 		P.mechanism = Low_first_mechanism()

	# 		print "\n running mechanism:" + str(P.mechanism.WhoAmI()) + "\n"

	# 		#Run best response dynamics
	# 		[rv, outcome] = BRD(P, cycle_special=True)
			
	# 		if rv == "cycle":
	# 			[cycle_ave_bu, cyc_len, cycle_str] = outcome
	# 			print "cycle length:" + str(cyc_len) 

	# 			# #All data
	# 			# retArr.append([sc, eq_pr_thresh, bu_thresh, p1_rev_thresh, min_th, min_price_exp, bu_exp, p1_rev_exp, min_alpha, bu_rand, p1_rev_rand]) #, math.exp(math.exp(-min_alpha))
	# 			retArr.append([sc, bu_thresh, bu_exp, bu_rand, cycle_ave_bu]) #, math.exp(math.exp(-min_alpha))
	# 		else:
	# 			print "BRD returned an equilibrium!"


	# 	np_arr = np.array(retArr)	
	# 	# params_labels=["Search cost", "Equilibrium price (threshold)", "Utility (threshold)", "Seller revenue (threshold)", "optimal threshold", 
	# 	# 			  "Equilibrium price (exp)", "Utility (exp)", "Seller revenue (exp)", "optimal alpha", 
	# 	# 			  "Utility (random)", "Seller revenue (random)"] 

	# 	params_labels=["Search cost", "Utility (threshold)", "Utility (exp)", "Utility (random)", "Utility (LPF cycle)"] 

	# 	#Write meta data
	# 	meta_data += "# " + str(params_labels) + "\r\n"

	# 	printToFile(plotArray=np_arr, meta_data=meta_data)
	# 	plotThis1(title="Search cost to min alpha. dist=" + P.dist.WhoAmI(), plotArray=np_arr,params_labels=params_labels)
	# 	# plotThis(dist=dist, mech=mech, plotArray=retArr,X_axis_label="Search costs", Y_axis_label="Sellers prices")


if __name__ == '__main__':
	main()

