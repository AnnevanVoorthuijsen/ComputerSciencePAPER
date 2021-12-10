# ComputerSciencePAPER

What is this project about
This project is about detecting duplicates in a JSON type data set. The project uses minhashing, LSH and a Multi-Component similarity Method. 

Structure of the code
The code consists of four main functions:

1.	Preprocessing: this function prepares the data for LSH. This function has three sub functions.
a.	Bootstrap: this function divides the data in a train and a test set. 
b.	Binaryvectors: this function makes binary representation vectors of the products. 
c.	Createhash: hashes the binary vectors and creates the signature matrix. 
2.	LSH_complete: this function performs LSH. This function has 2 sub functions.
a.	Splitsignature: splits the signature matrix in bands.
b.	LSH: this function performs the actual LSH.
3.	MSM: this function finds the potential duplicates. This function has 3 sub functions.
a.	Duplicates_in_list: calculates the duplicates between two lists.
b.	Qgrams: calculates the similarity based on two strings.
c.	Dissimilarity_matrix_func: generates a dissimilarity matrix for clustering.
Apart from these 3 functions the MSM function clusters and finds the potential duplicates.
4.	Output: this function generates the output. This function has 2 sub functions
a.	Findtrueduplicates:  this function finds the true duplicates in the complete dataset. 
b.	F1_score: calculates the F1 score and the truepositives, falsepositives and falsenegatives. 

After these four function, there is a loop that performs all the functions to create output. The final output is the average of the loops. The output consists of F1-score, F1*-score, pair completeness, pair quality and fraction of comparisons. With this output, 4 plots are created.
F1-score vs fraction of comparisons
F1*-score vs fraction of comparisons
Pair completeness vs fraction of comparisons
Pair quality vs fraction of comparisons

How to use the code
A JSON file is needed to use this code. If the JSON file is imported in the Python project the code find the duplicates in the dictionary of the JSON file. 
