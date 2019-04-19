import sys
import pickle
from collections import Counter
import math
from tqdm import tqdm
import numpy as np
from nltk.corpus import wordnet as wn

# calculate the entropy of given empirical count
def calculate_entropy(type_counter):
	total_count = sum(type_counter.values())
	result = 0.0

	for _type in type_counter:
		prob = type_counter[_type] * 1.0 / total_count
		result += (-prob * math.log(prob, 2))

	return result

# calculate the conditional entropy of given empirical count
def calculate_conditional_entropy(conditional_type_counters, conditioned_type_counter):
	conditioned_type_total_count = sum(conditioned_type_counter.values())
	result = 0.0

	for _type in conditioned_type_counter:
		conditional_type_counter = conditional_type_counters[_type] \
									if _type in conditional_type_counters \
									else Counter()

		prob_type = conditioned_type_counter[_type] * 1.0 / conditioned_type_total_count
		result += prob_type * calculate_entropy(conditional_type_counter)

	return result

# calculate the conditional entropy of classifiers over nouns normalized by number of nouns
def calculate_normalized_entropy(classifier_on_noun_counters):
	num_noun_total = len(classifier_on_noun_counters)
	result = 0.0
	for noun, classifier_counter in classifier_on_noun_counters.items():
		result += calculate_entropy(classifier_counter)

	result /= num_noun_total

	return result


# calculate the mutual information between classifiers and adjectives
def calculate_I_classifier_adj(adj_counter, classifier_counter, classifier_on_adj_counters):
	H_classifier = calculate_entropy(classifier_counter)
	H_classifier_on_adj = calculate_conditional_entropy(classifier_on_adj_counters, adj_counter)
	I_classifier_adj = H_classifier - H_classifier_on_adj	
	return H_classifier, H_classifier_on_adj, I_classifier_adj

# calculate the mutual information between classifiers and nouns
def calculate_I_classifier_noun(noun_counter, classifier_counter, classifier_on_noun_counters):
	H_classifier = calculate_entropy(classifier_counter)
	H_classifier_on_noun = calculate_normalized_entropy(classifier_on_noun_counters)
	I_classifier_noun = H_classifier - H_classifier_on_noun
	return H_classifier, H_classifier_on_noun, I_classifier_noun

# calculate the mutual information between classifiers and noun senses
def calculate_I_classifier_noun_senses(wn_sense_counter, classifier_counter, classifier_on_wn_sense_counters):
	H_classifier = calculate_entropy(classifier_counter)
	H_classifier_on_noun_senses = calculate_conditional_entropy(classifier_on_wn_sense_counters, wn_sense_counter)
	I_classifier_noun_senses = H_classifier - H_classifier_on_noun_senses	
	return H_classifier, H_classifier_on_noun_senses, I_classifier_noun_senses

# build counters for input data for easy computing
def build_counters(noun_adj_classifier_counter):
	# (adj, count)
	adj_counter = Counter()
	# (classifier, count)
	classifier_counter = Counter()
	# (noun, count)
	noun_counter = Counter()
	# (noun, (classifier, count))
	classifier_on_noun_counters = {}
	# (adj, (classifier, count))
	classifier_on_adj_counters = {}

	for noun, adj, classifier in noun_adj_classifier_counter:
		count = noun_adj_classifier_counter[(noun, adj, classifier)]
		adj_counter[adj] += count
		classifier_counter[classifier] += count
		if adj not in classifier_on_adj_counters:
			classifier_on_adj_counters[adj] = Counter()
		classifier_on_adj_counters[adj][classifier] += count
		if noun not in classifier_on_noun_counters:
			classifier_on_noun_counters[noun] = Counter()
		classifier_on_noun_counters[noun][classifier] += count


	return adj_counter, classifier_counter, noun_counter, \
			classifier_on_noun_counters, classifier_on_adj_counters


# build a counter for Wordnet Synset Senses for nouns
def build_wordnet_sense_counters(noun_adj_classifier_counter):
	wn_sense_counter = Counter()
	classifier_on_wn_sense_counters = {}
	for noun, adj, classifier in noun_adj_classifier_counter:
		synsets = wn.synsets(noun, lang='cmn')
		count_senses(synsets, classifier, wn_sense_counter, classifier_on_wn_sense_counters)

	return wn_sense_counter, classifier_on_wn_sense_counters


def count_senses(synsets, classifier, wn_sense_counter, classifier_on_wn_sense_counters):
	if len(synsets) == 0:
		return
	for synset in synsets:
		synset_name = synset.name()
		wn_sense_counter[synset_name] += 1
		if synset_name not in classifier_on_wn_sense_counters:
			classifier_on_wn_sense_counters[synset_name] = Counter()
		classifier_on_wn_sense_counters[synset_name][classifier] += 1

# resample the entire dataset with replacement
def resample_data(counter):
	prob_list = []
	counter_key_list = []
	sum_count = sum(counter.values())
	for counter_key in counter:
		prob_list.append(counter[counter_key] * 1.0 / sum_count)
		counter_key_list.append(counter_key)
	prob_list = np.array(prob_list)
	sample_size = len(prob_list)

	list_idx_selected = np.random.choice(
		a = sample_size, size = sum_count, replace = True, p = prob_list 
	)

	resampled_counter = Counter()
	for idx in list_idx_selected:
		resampled_counter[counter_key_list[idx]] += 1

	return resampled_counter

# compute bootstrap confidence interval
def bootstrap_ci(data_list, measured_data, confidence_level=0.95):
	quantile_low = 0.5 * (1.0 - confidence_level) * 100.0
	quantile_high = quantile_low + confidence_level * 100.0

	data_list.sort()
	data_list = np.array(data_list)

	bias = np.mean(data_list) - measured_data

	interval_low = np.percentile(a = data_list, q = quantile_low) - bias
	interval_high = np.percentile(a = data_list, q = quantile_high) - bias

	return (interval_low, interval_high)


if __name__ == '__main__':
	noun_classifier_counter = pickle.load(open("noun_classifier.pkl", "rb"))
	noun_adj_classifier_counter = pickle.load(open("noun_adj_classifier.pkl", "rb"))

	adj_counter, classifier_counter, noun_counter, \
	classifier_on_noun_counters, classifier_on_adj_counters = \
		build_counters(noun_adj_classifier_counter)


	if sys.argv[1] == '-ICA': # mutual_information between classifiers and adjectives
		H_classifier, H_classifier_on_adj, I_classifier_adj = \
			calculate_I_classifier_adj(adj_counter, classifier_counter, classifier_on_adj_counters)
		print("H(C)=%.4f, H(C|A)=%.4f, I(C;A)=%.4f" % \
			(H_classifier, H_classifier_on_adj, I_classifier_adj))
		# bootstrap
		bootstrap_data_list = []
		print("bootstraping...")
		for i in tqdm(range(100)):
			resampled_counter = resample_data(noun_adj_classifier_counter)
			adj_counter, classifier_counter, noun_counter, \
			classifier_on_noun_counters, classifier_on_adj_counters = \
				build_counters(resampled_counter)
			H_classifier_bs, H_classifier_on_adj_bs, I_classifier_adj_bs = calculate_I_classifier_adj(adj_counter, classifier_counter, classifier_on_adj_counters)
			bootstrap_data_list.append(I_classifier_adj_bs)

		ci_low, ci_high = bootstrap_ci(bootstrap_data_list, I_classifier_adj)
		print("95% confidence interval of I(C;A)")
		print((ci_low, ci_high, ci_high - ci_low))

	elif sys.argv[1] == '-ICN': # mutual_information between classifiers and nouns
		H_classifier, H_classifier_on_noun, I_classifier_noun = \
			calculate_I_classifier_noun(noun_counter, classifier_counter, classifier_on_noun_counters)
		print("H(C)=%.4f, H(C|N)=%.4f, I(C;N)=%.4f" % \
			(H_classifier, H_classifier_on_noun, I_classifier_noun))
		# bootstrap
		bootstrap_data_list = []
		print("bootstraping...")
		for i in tqdm(range(100)):
			resampled_counter = resample_data(noun_adj_classifier_counter)
			adj_counter, classifier_counter, noun_counter, \
			classifier_on_noun_counters, classifier_on_adj_counters = \
				build_counters(resampled_counter)
			H_classifier_bs, H_classifier_on_noun_bs, I_classifier_noun_bs = calculate_I_classifier_noun(noun_counter, classifier_counter, classifier_on_noun_counters)
			bootstrap_data_list.append(I_classifier_noun_bs)

		ci_low, ci_high = bootstrap_ci(bootstrap_data_list, I_classifier_noun)
		print("95% confidence interval of I(C;N)")
		print((ci_low, ci_high, ci_high - ci_low))


	elif sys.argv[1] == '-ICAS':  # mutual_information between classifiers and adjectives over multiple senses
		noun_adj_classifier_counters = pickle.load(open("adj_supersenses.pkl", "rb"))
		for adj_sense in noun_adj_classifier_counters:
			noun_adj_classifier_counter = noun_adj_classifier_counters[adj_sense]
			adj_counter, classifier_counter, noun_counter, \
			classifier_on_noun_counters, classifier_on_adj_counters = \
				build_counters(noun_adj_classifier_counter)
			print(adj_sense)
			H_classifier, H_classifier_on_adj, I_classifier_adj = calculate_I_classifier_adj(adj_counter, classifier_counter, classifier_on_adj_counters)
			print("H(C)=%.4f, H(C|A)=%.4f, I(C;A)=%.4f" % \
				(H_classifier, H_classifier_on_adj, I_classifier_adj))
			# bootstrap
			bootstrap_data_list = []
			print("bootstraping...")
			for i in tqdm(range(100)):
				resampled_counter = resample_data(noun_adj_classifier_counter)
				adj_counter, classifier_counter, noun_counter, \
				classifier_on_noun_counters, classifier_on_adj_counters = \
					build_counters(resampled_counter)
				H_classifier_bs, H_classifier_on_adj_bs, I_classifier_adj_bs = calculate_I_classifier_adj(adj_counter, classifier_counter, classifier_on_adj_counters)
				bootstrap_data_list.append(I_classifier_adj_bs)

			ci_low, ci_high = bootstrap_ci(bootstrap_data_list, I_classifier_adj)
			print("95% confidence interval of I(C;A)")
			print((ci_low, ci_high, ci_high - ci_low))			

	elif sys.argv[1] == '-ICNS': # mutual_information between classifiers and nouns over multiple senses
		noun_adj_classifier_counters = pickle.load(open("noun_supersenses.pkl", "rb"))
		for noun_sense in noun_adj_classifier_counters:
			noun_adj_classifier_counter = noun_adj_classifier_counters[noun_sense]
			adj_counter, classifier_counter, noun_counter, \
			classifier_on_noun_counters, classifier_on_adj_counters = \
				build_counters(noun_adj_classifier_counter)
			print(noun_sense)
			H_classifier, H_classifier_on_noun, I_classifier_noun = calculate_I_classifier_noun(noun_counter, classifier_counter, classifier_on_noun_counters)
			print("H(C)=%.4f, H(C|N)=%.4f, I(C;N)=%.4f" % \
				(H_classifier, H_classifier_on_noun, I_classifier_noun))
			# bootstrap
			bootstrap_data_list = []
			print("bootstraping...")
			for i in tqdm(range(100)):
				resampled_counter = resample_data(noun_adj_classifier_counter)
				adj_counter, classifier_counter, noun_counter, \
				classifier_on_noun_counters, classifier_on_adj_counters = \
					build_counters(resampled_counter)
				H_classifier_bs, H_classifier_on_noun_bs, I_classifier_noun_bs = calculate_I_classifier_noun(noun_counter, classifier_counter, classifier_on_noun_counters)
				bootstrap_data_list.append(I_classifier_noun_bs)

			ci_low, ci_high = bootstrap_ci(bootstrap_data_list, I_classifier_noun)
			print("95% confidence interval of I(C;N)")
			print((ci_low, ci_high, ci_high - ci_low))
	

	elif sys.argv[1] == '-ICS': # mutual_information between classifiers and wordnet senses
		wn_sense_counter, classifier_on_wn_sense_counters = build_wordnet_sense_counters(noun_adj_classifier_counter)
		H_classifier, H_classifier_on_wn_sense, I_classifier_wn_sense = calculate_I_classifier_noun_senses(wn_sense_counter, classifier_counter, classifier_on_wn_sense_counters)
		print("H(C)=%.4f, H(C|S)=%.4f, I(C;S)=%.4f" % \
			(H_classifier, H_classifier_on_wn_sense, I_classifier_wn_sense))
		# bootstrap
		bootstrap_data_list = []
		print("bootstraping...")
		for i in tqdm(range(100)):
			resampled_counter = resample_data(noun_adj_classifier_counter)
			adj_counter, classifier_counter, noun_counter, \
			classifier_on_noun_counters, classifier_on_adj_counters = \
				build_counters(resampled_counter)
			H_classifier_bs, H_classifier_on_wn_sense_bs, I_classifier_wn_sense_bs = calculate_I_classifier_noun(noun_counter, classifier_counter, classifier_on_noun_counters)
			bootstrap_data_list.append(I_classifier_wn_sense_bs)

		ci_low, ci_high = bootstrap_ci(bootstrap_data_list, I_classifier_wn_sense)
		print("95% confidence interval of I(C;S)")
		print((ci_low, ci_high, ci_high - ci_low))	






