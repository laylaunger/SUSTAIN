# Authors: Nathaniel Blanco, Robert Ralston, Layla Unger, Olivera Savic and Mengcun Gao

##################################
########## DESCRIPTION ###########
##################################

# This script implements the SUSTAIN model described in Love, Medin & Gureckis 2004
# See the paper for a full description of the model.

# When using this model, it is assumed that you have a real or fake dataset consisting
# of a sequence of trials. In each trial, a stimulus is presented to a learner. In supervised
# learning, the learner is prompted to make some choice about a missing dimension of the stimulus
# (e.g., its label in classification, or the value on a hidden dimension in inference learning).
# In unsupervised learning, no responses relevant to category membership are made. 

# This script implements the process of using SUSTAIN to model category learning from the
# same sequence of stimuli as in your dataset, and identifying parameter values that capture
# the estimate the responses of the real or simulated learner.

# Below is a description of aspects of the dataset, model parameters, and the model
# You do not need to actually run this code.

### CATEGORY LEARNING DATASET ###

# Aspects of your dataset that you should know before fitting:

# (1) Number of dimensions
n_dim = 3 # M in the equations in Love & Gureckis 2004

# (2) Number of feature values on each dimension
n_values = c(3, 3, 2) # k in the paper

# (3) For each trial, the dimensions that are queried and that
#     are present in the stimulus 
present_dimensions <- c(1,2)
queried_dimensions <- 3

########## PARAMETERS ##########

# Parameters of sustain to be estimated from data:

r = 1 # attentional tuning parameter; r >= 0
beta = 1 # lateral inhibition between clusters; beta >= 0
d = 1  #  decision determinism parameter; d >= 0
eta = 0.5 # learning rate; 0 <= eta <= 1 or maybe just eta >= 0
tau = 0.5 # unsupervised cluster creation parameter; 0 <= tau <= 1

####### MODEL COMPONENTS ######## 

# Internal components of sustain: 

# Attention tuning for each dimension; length = number of dimensions
lambdas = c() 

# Number of clusters; can change over the course of learning
n_clusters =  0 

# Activations of each cluster upon being presented with a stimulus on a trial;
# changes across trials. Hact in paper.
cluster_activations = c()

# Output of each cluster following activation and lateral inhibition between
# clusters. Hout in paper. 
cluster_outputs = c()

# Weights between clusters and response units
# Implemented here as a matrix in which clusters are rows, and response
# units (across all dimensions) are columns. Thus, n_clusters x n_values 
weights = matrix() 

##################################
### FUNCTIONS: DATA PROCESSING ###
##################################

# The recode_features function takes the set of features that comprise 
# a stimulus (including the label) and converts it into one-hot encoding. 
# In the output, each feature is a vector containing 0s and a single 1. 
# For each dimension, the length of the vectors for features equals the number 
# of possible feature values for that dimension (e.g., if the dimension "shape" 
# has 2 values, the vectors will be of length 2). The position of the 1 in 
# the vector discriminates between the different feature values of a dimension 
# (e.g., "triangle" = 0 1, and "square" = 1 0).
# Arguments:
# features: Vector of original feature values for each dimension of a stimulus.
#           Make sure that the original feature values for each dimension are 
#           encoded starting at 1 (arbitrary feature codes will not work), because
#           they are used to index the position of the 1 in the one-hot vector.
# feature_variants: Vector containing the number possible features for each dimension 
# Output:
# A list of the new one-hot vectors for each feature of a stimulus.

recode_features <- function(features, feature_variants) {
  output_list <- list()  # Create an empty list
  
  # For each dimension feature...
  for (i in 1:length(features)) {
    
    # make vector of 0's with length feature_variants[i] (num of features in the dimension)
    dim_vector <- rep(0, feature_variants[i])
    
    # set the value of the vector at the position of the feature to 1. 
    # Use the value of the feature in features as an index of the position 
    # where the 1 should go. Note: This works b/c we assume all feature values
    # are numbered consecutively starting at 1. 
    dim_vector[ features[i] ] <- 1
    
    # add dim_vector to the output in the correct position for the dimension
    output_list[[i]] <- dim_vector
  }
  
  return(output_list)
}


# Example: uncomment to run
# recode_features(c( 1, 2, 4), c(3, 3, 4))

####################################
######## FUNCTIONS: SUSTAIN ########
####################################

### PART I: STIMULUS -> RESPONSE ###


# Calculate the distance between the stimulus and a cluster 
# on a dimension.
# Both the feature value of the stimulus on a dimension and the
# position of the cluster on a dimension are vectors (for 
# features of stimuli, these are one-hot encoded vectors, e.g.,
# (0, 1, 0))
# Equation 4 in SUSTAIN paper that calculates Mu_ij
dim_distance <- function(stim_dim, cluster_dim) {
  difference <- abs(stim_dim - cluster_dim)
  distance <- sum(difference)/2
  return(distance)
}

# Calculate the activation of a cluster based on: (1) The distances
# for each dimension between the position of the cluster and the
# feature value of the stimulus, (2) The lambda (i.e., tuning/attention weight)
# for each dimension, and (3) The freely estimated attentional parameter r, which
# modulates the effects of lambdas (large r accentuates lambda differences; small
# r tends to equalize them)
# Equation 5 in SUSTAIN paper that calculates Hact
cluster_activation <- function(distances, lambdas, r) {
  numerator <- sum( (lambdas^r) * exp(-lambdas*distances) )
  denominator <- sum( (lambdas^r) )
  activation <- numerator/denominator
  return(activation)
}

# Find the index of the cluster with the highest activation
find_winning_cluster <- function(activations) {
  winner_index <- which(activations == max(activations))
  return(winner_index[1]) # take the first one in case there are ties
}

# Calculate the outputs of the clusters following lateral inhibition
# (competition) between clusters. Activations of the non-winning clusters 
# contribute to inhibiting the winning cluster based on the value of
# beta, one of the freely estimated parameters. The ultimate output of 
# non-winning clusters is 0. 
# Equation 6 in SUSTAIN paper that calculates Hout
cluster_output <- function(activations, beta) {
  # Get index of winning cluster
  winner <- find_winning_cluster(activations)
  
  # Initialize the output of each cluster to 0
  output <- rep(0, length(activations))
  
  # Use Equation 6 to calculate the output of the winning cluster based on inhibition
  # between all clusters
  winning_output <- ((activations[winner]^beta)/sum(activations^beta)) * activations[winner]
  
  # Replace output for just the winning cluster with the winning_output
  output[winner] <- winning_output
  return(output)
}

# Use the cluster outputs and weights with each response unit to calculate the 
# activation of each response unit. 
# Equation 7 in SUSTAIN paper to calculate Cout
# Note: Equation 7 calculates activation for only queried response unit. However, we
# calculate activation for all response units, and index only the queried unit later.
response_unit_output <- function(cluster_outputs, weights) {
  response_unit_outputs <- cluster_outputs %*% weights # matrix multiplication
  return(response_unit_outputs)
}

# Calculate the probabilities of each response on the queried dimension based on
# (1) The response unit activation for only responses on the queried dimension, 
# and (2) The freely estimated parameter d, which controls response determinism.
# Equation 8 in SUSTAIN paper that calculates Pr(k)
response_probabilities <- function( queried_unit_outputs, d ) {
  numerators <- exp(d * queried_unit_outputs)
  denominator <- sum(numerators)
  probabilities <- numerators/denominator
  return(probabilities)
}

# Combine all component functions in PART I into "forward_pass" function

# Arguments:
# The current sustain model (free parameters r, beta, d, eta, tau*, cluster positions, lambdas, weights),
# the stimulus on a trial, the dimension(s) queried on a trial, the dimensions present in 
# the stimulus, and an argument that specifies whether learning is unsupervised.

# What function does:
# Identifies indeces of response units corresponding to queried dimension(s)
# Checks if there are 0 clusters. If so, returns empty output for unsupervised, and output that 
# contains only equal probabilities across queried response units for supervised
# If there are clusters, the stimulus activates the clusters based on distances, clusters inhibit 
# each other, the winning cluster activates response units via weights, and response unit activations lead to 
# response probabilities.

# Output:
# Probabilities of each response on queried dimension. 
# For updating sustain (see PART II), output also includes cluster activations, distances, index of 
# winning cluster, and response unit activations.

# Note: Stimulus and cluster positions will be in this general format
# stimulus <- list( c(0,1,0),
                  # c(0,0,1),
                  # c(0, 1) )

forward_pass <- function(sustain, stimulus, queried_dimensions, present_dimensions, unsupervised=FALSE) {
  
  # To start, it is useful to identify indeces of response units that correspond to the queried dimension(s)
  # This is because, in both the objects that will contain response unit weights and probabilities, 
  # values corresponding to each response unit are concatenated across dimensions. 
  # For example, the columns of the weights matrix are response units across dimensions. So, it is important 
  # to know which response units correspond to features on the queried dimensions.

  # Initialize queried_indeces, which will be a list, in which each element contains the indeces corresponding
  # to a queried dimension 
  queried_indeces <- rep(list(NA), length(queried_dimensions))
  for(i in 1:length(queried_dimensions)) {
    # To find the first index corresponding to a queried dimension:
    # If queried_dimension == 1, then the first index == 1. 
    # If queried dimension > 1: count the number of features on dimensions leading up to 
    # the queried dimension (i.e., sum(lengths(stimulus)[1:(queried_dimension-1)]) ), and then start with the next column
    first_queried_index <- ifelse(queried_dimensions[i] == 1, 1, 
                                  sum(lengths(stimulus)[1:(queried_dimensions[i]-1)]) + 1)
    
    # To find the last index corresponding to a queried dimension:
    # Count the number of features on dimensions leading up to and including the queried dimension features
    last_queried_index <- sum(lengths(stimulus)[1:(queried_dimensions[i])])
    
    # All indeces corresponding to a queried dimension
    dimension_indeces <- first_queried_index:last_queried_index
    
    # Add to vector of queried indeces
    queried_indeces[[i]] <- dimension_indeces
  }
  
  # First, check whether there 0 clusters (which will be the case on the first trial).
  # If there are 0 clusters and learning is unsupervised, there is no output from forward_pass. 
  # If there are 0 clusters and learning is supervised, we need to just set the probabilities 
  # of each response for a given queried dimension to be equal (because there are no clusters to activate).
  if(length(sustain$clusters) == 0) {
    
    if (unsupervised==TRUE) {
      
      return_list <- list( Hout = NA, distances = NA,
                           Hact = NA, winner_index = NA, Cout = NA )
      return(return_list)
    }
    
    # Set the probabilities of each response for a given queried dimension to be equal
    probabilities <- rep(NA, sum(lengths(stimulus)))
    for (i in 1:length(queried_dimensions)) {
      dimension_probabilities <- rep(1/lengths(stimulus[queried_dimensions[i]]), lengths(stimulus[queried_dimensions[i]])) 
      probabilities[queried_indeces[[i]]] <- dimension_probabilities
    }
    
    return_list <- list( probabilities = probabilities, Hout = NA, 
                         distances = NA, winner_index = NA, 
                         Cout = NA )
    return(return_list)
    
  }
  
  # Create vector of the subset of lambdas for only the dimensions present in stimulus
  present_lambdas <- sustain$lambdas[present_dimensions]
  
  # Initialize a vector of activations for each cluster
  activations <- rep(NA, length(sustain$clusters))
  
  # Initialize a matrix of distances between each cluster and the stimulus on each dimension.
  # Each row is a cluster, each column is a dimension
  # Note: Distances for all dimensions are calculated first, and then only distances for present
  # dimensions are used to calculate cluster activations.
  cluster_distances <- matrix(,nrow=length(sustain$clusters), ncol=length(stimulus))
  
  # Fill in the cluster_distances matrix, and use it to fill in the activations vector.
  # First, go through each cluster present in SUSTAIN (represented as a list of cluster dimensions)
  for (cluster_index in 1:length(sustain$clusters)) {
    
    # Initialize a distances vector that will be filled in with distances for each dimension
    # (present dimensions indexed later)
    distances <- rep(NA, length(stimulus))
    
    # For each dimension... 
    for (dimension_index in 1:length(stimulus)) {
      # fill in the distance between the stimulus and the current cluster
      distances[dimension_index] <- dim_distance(stimulus[[dimension_index]], 
                                                 sustain$clusters[[cluster_index]][[dimension_index]] )
    }
    
    # use the distances vector to fill in the distances between the stimulus and the current cluster
    cluster_distances[cluster_index,] <- distances
    
    # Fill in the activation of the current cluster using distances and lambdas for only present dimensions
    # (Hact in SUSTAIN paper)
    activations[cluster_index] <- cluster_activation(distances[present_dimensions], present_lambdas, sustain$r)
    
  }
  
  # Identify the index of the winning cluster (i.e., the one with the highest activation)
  winner_index <- find_winning_cluster(activations)
  
  # Calculate the outputs of the clusters after lateral inhibition has taken place (Hout in SUSTAIN paper)
  outputs <- cluster_output(activations, sustain$beta)
  
  # Calculate the activation of all response units based on the cluster outputs
  response_outputs <- response_unit_output(outputs, sustain$weights)
  
  # For unsupervised learning, return cluster distances, activations, winner index, outputs, 
  # and all response unit outputs 
  if (unsupervised==TRUE) {
    
    return_list <- list( Hout = outputs[winner_index], distances = cluster_distances[winner_index,],
                         Hact = activations[winner_index], winner_index=winner_index, Cout = response_outputs )
    return(return_list)
  }
  
  # For supervised learning, use parameter d and response_outputs for response units on only the queried dimension(s),
  # to calculate response probabilities. 
  # Note: response_outputs is a matrix where each row is a cluster, and each column is a response unit. 
  # The columns of the matrix are response units *across all dimensions*. Response units columns are ordered by dimension 
  # and feature (e.g., the first column is the response unit corresponding to the first feature on a dimension). 
  # So, we can use queried_indeces defined above to identify the columns of response_outputs 
  # that correspond to the queried dimension(s).
  
  # Initialize a vector of probabilities across all response units
  probabilities <- rep(NA, sum(lengths(stimulus)))
  
  # Fill in the probabilities for response units on only the queried dimension(s)
  for (i in 1:length(queried_dimensions)) {
    queried_outputs <- response_outputs[,queried_indeces[[i]]]  # Identify response_outputs only for queried dimension
    queried_probabilities <- response_probabilities(queried_outputs, sustain$d) # Calculate probability of queried response units
    
    probabilities[queried_indeces[[i]]] <- queried_probabilities
  }
  
  # For supervised learning, return winning cluster distances and activations, response
  # probabilities, the index of the winning cluster, and activation of response units.
  return_list <- list( probabilities = probabilities, Hout = outputs[winner_index], 
                       distances = cluster_distances[winner_index,], winner_index=winner_index, 
                       Cout = response_outputs )
  return(return_list)
}


### PART II: UPDATE AFTER FEEDBACK ###


# When the response is correct, adjust the position of the winning
# cluster to be closer to the position of the stimulus (modulated
# by eta, the freely estimated learning rate parameter) 
# Equation 12 in SUSTAIN paper
adjust_cluster <- function(cluster, stimulus, eta) {
  
  # Calculate the difference between the feature values of the stimulus 
  # and the position of the cluster for each dimension 
  differences <- mapply('-', stimulus, cluster, SIMPLIFY = FALSE)
  # Multiply each difference by eta
  deltas <- lapply(differences, '*', eta)
  #Set the new position of the winning cluster to be closer to the stimulus
  new_cluster <- mapply('+', cluster, deltas, SIMPLIFY = FALSE)
  
  # e.g. cluster = c(0,1,0, 0,0.25,0.75) 
  return(new_cluster)
}

# Adjust the lambdas for each dimension based on the distances between the
# stimulus and the winning cluster on each dimension. 
# Note: When the response was incorrect, the "winning cluster" is a new cluster
# positioned on the stimulus. Thus, all distances will be 0, and all lambdas will
# be updated equally. 
# Equation 13 in SUSTAIN paper

adjust_lambdas <- function(lambdas, distances, eta) {
  new_lambdas <- lambdas + (eta * exp(-lambdas*distances) * (1-lambdas*distances))
  return(new_lambdas)
}

# Adjust the weights between the winning cluster (newly created cluster in case
# of incorrect response) and the response units for the queried dimension based
# on: 
# (1) The difference between the response unit output and the correct response
# conveyed by feedback
# (2) The learning rate, and 
# (3) The output of the winning cluster (so that only weights for the winning cluster
#     are updated) 

adjust_weights <- function(weights, feedback, response_unit_output, cluster_output, eta) {
  new_weights <- weights + (eta * (feedback - response_unit_output) * cluster_output)
  return(new_weights)
}

# Adjust feedback according to humble teacher. 
# Note: Feedback will be the one-hot vector for the queried dimension of
# the stimulus (or, if multiple features are queried, the concatenation of these vectors)
# Equation 9 in SUSTAIN paper

humble_teacher <- function(output, feedback) {
  adjusted_feedback <- rep(NA, length(feedback)) #Initialize a vector of NAs
  for(i in 1:length(feedback)){
    if (feedback[i] == 0) {
      adjusted_feedback[i] <- min(output[i], 0)
    }
    else {
      adjusted_feedback[i] <- max(output[i], 1)
    }
  }
  return(adjusted_feedback)
}

# Combine all component functions in PART II into "update_sustain" function

# Arguments:
# The current sustain model (free parameters r, beta, d, eta, tau*, cluster positions, lambdas, weights),
# the stimulus on a trial, the dimension(s) queried on a trial, the dimensions present in 
# the stimulus, the list output by the forward_pass function ("trial_output"), and an argument that 
# specifies whether learning is unsupervised.

# What function does:
# Identifies the indeces of response units that correspond to the queried dimension(s)
# Identifies whether a new cluster must be made (e.g., if there are no clusters), and if so, makes 
# a new cluster positioned on the stimulus with weights initialized to 0, then re-runs forward_pass
# When no new cluster is formed, adjusts existing clusters/lambdas/weights

# Output:
# The updated sustain model

update_sustain <- function(sustain, stimulus, queried_dimensions, 
                           present_dimensions, trial_output, unsupervised=FALSE) {
  
  # To start, use the same approach as in forward_pass to make a list, in which each elemnt
  # contains the indeces that correspond to the response units on a queried dimension
  queried_indeces <- rep(list(NA), length(queried_dimensions))
  for(i in 1:length(queried_dimensions)) {
    first_queried_index <- ifelse(queried_dimensions[i] == 1, 1, 
                                  sum(lengths(stimulus)[1:(queried_dimensions[i]-1)]) + 1)
    last_queried_index <- sum(lengths(stimulus)[1:(queried_dimensions[i])])
    dimension_indeces <- first_queried_index:last_queried_index
    queried_indeces[[i]] <- dimension_indeces
  }
  
  # First, check whether conditions for forming a new cluster are met.
  # Initialize form_new_cluster to FALSE
  form_new_cluster=FALSE
  
  # Condition 1: When there are no clusters
  if (length(sustain$clusters)==0) {
    form_new_cluster <- TRUE
  }
  # Condition 2: When learning is unsupervised, and the activation of the winning cluster
  # is < tau (Hact in the output of forward_pass when unsupervised == TRUE is for the winning
  # cluster only)
  else {
    # for unsupervised if Hact < tau make a new cluster
    if (unsupervised==TRUE) {
      if (trial_output$Hact < sustain$tau) {
        form_new_cluster <- TRUE
      }
    }
    # Condition 3: When learning is supervised, and the response was incorrect. 
    # Compares two vectors: the response unit probabilities for each feature on the 
    # queried dimension(s), and the feedback in which the correct feature on the queried dimension(s) == 1,
    # and other features == 0. To compare, checks whether the index of the highest response
    # probability is the same as the index of the correct feature.
    
    # NOTE: Response is determined to be correct or incorrect based on whether the highest probability 
    # response is the same as the correct response. Since these are probabilities, it is conceivable 
    # that an "actual" response would be different. However, feedback & updating is based on probabilities
    # of responses, not "actual" responses.
    else {
      
      accuracy <- rep(NA, length(queried_dimensions))
      # Check accuracy of response for each queried dimension
      for(i in 1:length(accuracy)) {
        # First identify the probabilities for each response unit on the queried dimension, and the
        # correct response unit valies for the queried dimension
        queried_probabilities <- trial_output$probabilities[queried_indeces[[i]]]
        queried_dimension <- stimulus[[queried_dimensions[i]]]
        # Then identify the index of the response unit with the maximum probability, and of the 
        # response that is correct
        max_probability_index <- which(queried_probabilities == max(queried_probabilities))[1]
        correct_index <- which(queried_dimension == 1)
        
        #If indeces are the same, accuracy = 1; else, it is 0
        accuracy[i] <- ifelse(max_probability_index == correct_index, 1, 0)
        
        #print(queried_probabilities)
        #print(max_probability_index)
        #print(correct_index)
        #print(accuracy)
      }
      

      
      
      # Form a new cluster if any of the responses were incorrect
      if(any(accuracy == 0)) {
        form_new_cluster <- TRUE
      }
    }
  }
  
  # If any condition for forming a new cluster is met, form a new cluster at the position of
  # the stimulus and add it to the sustain model. Also add weights between new cluster and
  # response units. Run forward_pass again with this new cluster included to get activations,
  # weights, etc for this new cluster.
  if (form_new_cluster) {
    # Add stimulus as a new cluster
    sustain$clusters <- c( sustain$clusters, list(stimulus))
    
    # Add new row the the weights matrix (0 prior to running forward_pass again)
    sustain$weights <- as.matrix(rbind(sustain$weights, rep(0, sum(lengths(stimulus))))
    )
    
    # re-run the forward pass now that new cluster is added to get the correct values for the update
    trial_output <- forward_pass(sustain, stimulus, queried_dimensions, present_dimensions, unsupervised=unsupervised)
    
  }
  
  # When no new cluster was formed, adjust existing clusters/lambdas/weights
  # Adjust the position of the winning cluster to be closer to the stimulus (equation 12)
  sustain$clusters[[trial_output$winner_index]] <- adjust_cluster(sustain$clusters[[trial_output$winner_index]], 
                                                                  stimulus, sustain$eta)
  
  # Adjust the lambdas based on the dimension distances between the stimulus and 
  # the winning cluster (equation 13)
  sustain$lambdas <- adjust_lambdas(sustain$lambdas, trial_output$distances, sustain$eta)
  
  # Adjust the weights between the winning cluster and the response units for the 
  # queried dimension. For unsupervised learning, all dimensions are "queried" (equation 14)
  
  if (unsupervised) { 
    queried_columns <- 1:sum(lengths(stimulus))
  }
  
  queried_columns <- unlist(queried_indeces)
  
  # Use humble teacher to get the modified feedback (t_zk) from the paper (equation 9)
  # The raw feedback is the stimulus vector(s) for the queried dimension(s). Unlist is used
  # so that if there are multiple queried dimensions, vectors across queried dimensions are
  # concatenated.
  feedback <- humble_teacher(trial_output$Cout[queried_columns], unlist(stimulus[queried_dimensions]))
  
  # Update the weights between the winning cluster and queried response units based on feedback.
  # Note: Weights is a matrix in which clusters are rows, and response units are columns.
  sustain$weights[trial_output$winner_index, queried_columns] <- 
                                adjust_weights(sustain$weights[trial_output$winner_index, queried_columns], 
                                               feedback, trial_output$Cout[queried_columns], 
                                               trial_output$Hout, sustain$eta)
  
  return(sustain)
  
}

### PART III: LIKELIHOOD FUNCTION ###

# This function combines forward_pass (PART I) and update_sustain (PART II) into
# a likelihood function. 
# This function progresses through a sequence of trials. 
# For each trial, forward_pass produces probabilities associated with each possible 
# response on the queried dimension(s). 
# We then index the probability for the response actually made (which we assume is coded as 
# an integer - e.g., if the category label is queried, responding with one label might be coded
# as 1, and the other as 2). 
# Next, we get the negative log likelihood of this probability. This value is added to a running 
# sum of negative log likelihoods across trials. 
# Finally, we update sustain, and then move to the next trial. 

get_full_likelihood <- function(stimuli, responses, queried_dimensions, 
                                 present_dimensions, sustain, return_sustain = FALSE) {
  
  # Initialize likelihood to 0
  likelihood <- 0
  
  # For each trial...
  for (i in 1:length(stimuli)) {
    
    # Get the initial output of sustain for the trial
    trial_output <- forward_pass(sustain, stimuli[[i]], queried_dimensions[[i]], present_dimensions[[i]], unsupervised=FALSE)
    
    # Get the actual response for the trial
    trial_responses <- responses[[i]]
    
    # Make a list called queried_indeces. Each element of the list corresponds to a queried dimension, 
    # and is a vector of the indeces of the probabilities that correspond to a queried dimension. We
    # will use these indeces to get the likelihood of the response(s) on the queried 
    # dimension(s)
    queried_indeces <- rep(list(NA), length(queried_dimensions[[i]]))
    
    # For each queried dimension on the trial, get the indeces of probabilities corresponding to that
    # dimension 
    for(j in 1:length(queried_dimensions[[i]])) {
      first_queried_index <- ifelse(queried_dimensions[[i]][j] == 1, 1, 
                                    sum(lengths(stimuli[[i]])[1:(queried_dimensions[[i]][j]-1)]) + 1)
      last_queried_index <- sum(lengths(stimuli[[i]])[1:(queried_dimensions[[i]][j])])
      dimension_indeces <- first_queried_index:last_queried_index
      queried_indeces[[j]] <- dimension_indeces
    }
    
    # Get the likelihood for the trial
    # First, initialize a vector that will contain the likelihood(s) of the response(s) on the queried
    # dimension(s)
    trial_likelihoods <- rep(NA, length(queried_dimensions[[i]]))
    
    # For each queried dimension, get the response probabilities. From these probabilities, identify
    # the one that corresponds to the actual response, and is thus the likelihood
    for(j in 1:length(trial_likelihoods)) {
      queried_probabilities <- trial_output$probabilities[queried_indeces[[j]]]
      response_likelihood <- queried_probabilities[trial_responses[j]]
      trial_likelihoods[j] <-  -(log(response_likelihood))
    }
    
    # Sum the negative log likelihoods for the response(s) on a trial
    full_trial_likelihood <- sum(trial_likelihoods)
    
    # Update sustain
    sustain <- update_sustain(sustain, stimuli[[i]], queried_dimensions[[i]], present_dimensions[[i]], trial_output)
    
    # Add likelihood for trial to the total likelihood
    likelihood <- likelihood + full_trial_likelihood
    #print(trial_output$probabilities)
  }
  
  # Option to return the sustain model at the end of the trials
  if(return_sustain) {
    return(list(sustain, likelihood))
  }
  
  return(likelihood)
}

####################################
############# EXAMPLE ##############
########### SUPERVISED #############
# ONE TRIAL, FAKE DATA, FAKE MODEL #
####################################

# This section contains an example to test forward_pass and update_sustain
# on one fake trial at a time, starting with a fake sustain model.

# A fake SUSTAIN model without clusters
sustain_empty <- list( r = 1.0, beta = 1.0, d = 10.0, eta= 0.5,
                       # Each cluster is a list. The list contains vectors that specify
                       # the position of the cluster on each dimension. 
                       # These positions are updated with learning.
                       clusters = list(),
                       # Lambas that specify attention tuning for each dimension
                       lambdas = c(1.0, 1.0, 1.0),
                       
                       # Weights that specify the weight between each cluster and each
                       # response unit. Rows are clusters, response units are columns
                       # In this example, 2 clusters, 8 response units (summed number
                       # of features across dimensions - i.e., 3 + 3 + 2)
                       weights = NULL
)

# A fake SUSTAIN model with clusters
sustain_clusters <- list( r = 1.0, beta = 1.0, d = 10.0, eta= 0.5,
                          # Each cluster is a list. The list contains vectors that specify
                          # the position of the cluster on each dimension. 
                          # These positions are updated with learning.
                          clusters = list(
                            # cluster 1
                            list(c(0.1, 0.8, .1),
                                 c(0.1, 0,  0.9),
                                 c(1,   0)),
                            # cluster 2
                            list(c(1,0,0),
                                 c(0,1,0),
                                 c(0, 1))
                          ),
                          # Lambas that specify attention tuning for each dimension
                          lambdas = c(1.5, 1.0, 1.0),
                          
                          # Weights that specify the weight between each cluster and each
                          # response unit. Rows are clusters, response units are columns
                          # In this example, 2 clusters, 8 response units (summed number
                          # of features across dimensions - i.e., 3 + 3 + 2)
                          weights = matrix( c(0, 0, 0, 0.4, 0.25, 0.35, 0.8, 0.2, 
                                              0, 0, 0, 0,   0,    0,    0.2, 0.8),
                                            nrow = 2, ncol = 8, byrow=TRUE)
)


# Fake stimuli
stimulus1 <- list(c(1, 0, 0),
                 c(0, 1, 0),
                 c(1, 0))

stimulus2 <- list(c(0, 1, 0),
                 c(0, 1, 0),
                 c(1, 0))


# Fake values for queried and present dimensions for a trial
# You can vary these to try out different dimensions, and numbers of queried vs 
# present dimensions
queried_dimensions1 = 3
present_dimensions1 = c(1,2)

queried_dimensions2 = c(1,3)
present_dimensions2 = 2

# Test forward_pass and update_sustain functions
# You can vary the stimulus, queried_dimensions, and present_dimensions
trial_output1 <- forward_pass(sustain_empty, stimulus1, queried_dimensions1, present_dimensions1)

trial_output1


updated_sustain1 <- update_sustain(sustain_empty, stimulus1, queried_dimensions1, 
                           present_dimensions1, trial_output1, unsupervised=FALSE)

updated_sustain1


trial_output2 <- forward_pass(updated_sustain1, stimulus2, queried_dimensions2, present_dimensions2)

trial_output2


updated_sustain2 <- update_sustain(updated_sustain1, stimulus2, queried_dimensions2, 
                                  present_dimensions2, trial_output2, unsupervised=FALSE)

updated_sustain2

######################################
############## EXAMPLE ###############
############ SUPERVISED ##############
# MULTI TRIAL, FAKE DATA, FAKE MODEL #
######################################

# This section contains an example that tests the likelihood function,
# which uses forward_pass and update_sustain iteratively on a sequence
# of trials

# A fake SUSTAIN model without clusters
sustain_empty <- list( r = 1.0, beta = 1.0, d = 10.0, eta= 0.5,
                       clusters = list(),
                       lambdas = c(1.0, 1.0, 1.0),
                       weights = NULL
)


# A list of fake stimuli
stimulus1 <- list(c(1, 0, 0),
                  c(0, 1, 0),
                  c(1, 0))

stimulus2 <- list(c(1, 0, 0),
                  c(0, 1, 0),
                  c(1, 0))

stimulus3 <- list(c(0, 1, 0),
                  c(0, 0, 1),
                  c(0, 1))

stimulus4 <- list(c(1, 0, 0),
                  c(1, 0, 0),
                  c(1, 0))

stimuli <- list(stimulus1, stimulus2, stimulus3, stimulus4)

# Fake values for queried and present dimensions for a trial
queried_dimensions = list(3, 3, 3, 3)
present_dimensions = list(c(1,2), c(1,2), c(1,2), c(1,2))


# Fake responses
responses <- list(c(1), c(1), c(2), c(1))

test_full_likelihood <- get_full_likelihood(stimuli, responses, queried_dimensions, present_dimensions, sustain_empty)

test_full_likelihood

######################################
############## EXAMPLE ###############
########### UNSUPERVISED #############
# MULTI TRIAL, FAKE DATA, FAKE MODEL #
######################################

# This section contains an example to test forward_pass and update_sustain
# on one fake trial at a time, starting with a fake sustain model.

# A fake SUSTAIN model without clusters
sustain_empty <- list( r = 1.0, beta = 1.0, d = 10.0, eta= 0.5, tau=0.5,
                       # Each cluster is a list. The list contains vectors that specify
                       # the position of the cluster on each dimension. 
                       # These positions are updated with learning.
                       clusters = list(),
                       # Lambas that specify attention tuning for each dimension
                       lambdas = c(1.0, 1.0, 1.0),
                       
                       # Weights that specify the weight between each cluster and each
                       # response unit. Rows are clusters, response units are columns
                       # In this example, 2 clusters, 8 response units (summed number
                       # of features across dimensions - i.e., 3 + 3 + 2)
                       weights = NULL
)

# Fake stimuli
stimulus1 <- list(c(1, 0, 0),
                  c(1, 0, 0),
                  c(1, 0))

stimulus2 <- list(c(0, 1, 0),
                  c(0, 1, 0),
                  c(1, 0))

stimulus3 <- list(c(0, 0, 1),
                  c(0, 0, 1),
                  c(0, 1))

stimulus4 <- list(c(0, 1, 0),
                  c(0, 1, 0),
                  c(0, 1))

stimulus5 <- list(c(1, 0, 0),
                  c(1, 0, 0),
                  c(0, 1))

stimuli <- list(stimulus1, stimulus2, stimulus3, stimulus4, stimulus5)


# Fake values for queried and present dimensions - 
# for unsupervised, all dimensions are treated as both present
# and queried
queried_dimensions = c(1,2,3)
present_dimensions = c(1,2,3)

# Test forward_pass and update_sustain functions
trial_output <- rep(list(NA), length(stimuli))
updated_sustain <- rep(list(NA), length(stimuli))

trial_output[[1]] <- forward_pass(sustain_empty, stimuli[[1]], queried_dimensions, 
                              present_dimensions, unsupervised = TRUE)
updated_sustain[[1]] <- update_sustain(sustain_empty, stimuli[[1]], queried_dimensions, 
                                   present_dimensions, trial_output[[1]], unsupervised=TRUE)

for(i in 2:length(trial_output)) {
  trial_output[[i]] <- forward_pass(updated_sustain[[i-1]], stimuli[[i]], queried_dimensions, 
                                    present_dimensions, unsupervised=TRUE)
  updated_sustain[[i]] <- update_sustain(updated_sustain[[i-1]], stimuli[[i]], queried_dimensions, 
                                     present_dimensions, trial_output[[i]], unsupervised=TRUE)
}

# Look at final sustain model
updated_sustain[[length(stimuli)]]

####################################
############# EXAMPLE ##############
############ REAL DATA #############
####################################

# This section shows an example of re-formating a dataset so that you can fit it with sustain.
# The exact steps for re-formatting data will vary depending on how your original dataset is formatted,
# but you should end up with objects containing a sequence of trials that specify the stimulus, 
# present dimension(s), queried dimension(s), and response(s) for each trial. 

# set working directory
setwd("LOCATION OF EXAMPLE DATA")

# Read in the data
example_data <- read.table('sustain example data.txt', header=F, colClasses='character')
names(example_data) <- c('subj', 'trial', 'resp', 'label', 'D_dim', 'I_dim', 'rt', 'features', 'phase' )

# Make a new column called label_features. For each trial, this column contains a string that specifies 
# the feature values for each dimension of the stimulus on that trial, starting with the category label. 
# Ensure that each feature value is coded as an integer, and that the integers for features on a given 
# dimension are consecutive. 
example_data$label_features <- paste(example_data$label, example_data$features, sep='')

# Reformat the stimulus string for each trial, so that each trial is an element of a list, in which
# the string for the trial is split into a character vector
split_features <- strsplit(example_data$label_features, '')

# Ensure that the feature values for each dimension start at 1. In this dataset, they actually started at 0,
# so we need to add 1 to each value. To do this, we need to convert the character values to numeric and
# add 1.
numeric_features <- lapply(split_features, function(x){as.numeric(x) + 1})

# Make a vector that specifies the total number of feature values for each dimension, in the same order
# as the dimensions appear in the stimuli.
n_feature_variants <- c(2, 2, 2, 2, 2, 2, 3, 3)

# Use the recode_features function defined above to convert the stimuli to one-hot encoding that sustain uses.
# You will end up with a list. Each element of the list is a stimulus on a given trial. Each stimulus is itself
# a list, where each element of the stimulus list is a dimension. In each dimension, the feature value on the 
# dimension is specified as a one-hot vector. 
coded_stimuli <- lapply(numeric_features, recode_features, n_feature_variants )


# Make lists for the present dimension(s) and queried dimension(s) for each trial. 
# In this dataset, the first dimension of each stimulus is the category label, which was queried. The remaining
# 7 dimensions (dimensions 2 through 8) were all present.  
present_dimensions_test <- rep(list(c(2:8)), length(coded_stimuli))
queried_dimensions_test <- rep(list(1), length(coded_stimuli))

# Make a list for the response(s) for each trial
# In this dataset, responses were category labels. Remember to make sure these are consecutive integers
# starting at 1.
responses_test <- as.list(as.numeric(example_data$resp) + 1)


# A fake SUSTAIN model with some parameter values, no clusters, and the correct number of lambdas
sustain_empty <- list( r = 1.0, beta = 1.0, d = 2, eta = 0.5, tau=0.5,
                       clusters = list(),
                       lambdas = rep(1.0, 8),
                       weights = NULL
)

# Try out SUSTAIN functions for a single trial
trial_output1 <- forward_pass(sustain_empty, coded_stimuli[[1]], queried_dimensions_test[[1]], present_dimensions_test[[1]])

updated_sustain1 <- update_sustain(sustain_empty, coded_stimuli[[1]], queried_dimensions_test[[1]], present_dimensions_test[[1]],
                                   trial_output1, unsupervised = FALSE)

# Get full likelihood for the chosen parameter values
test_full_likelihood <- get_full_likelihood(coded_stimuli, responses_test, queried_dimensions_test, present_dimensions_test, sustain_empty)


####################################
####### ESTIMATE PARAMETERS ########
####################################

# Function to use with optim (see below) that calculates the full likelihood given
# a set of candidate values for parameters and data (stimuli, responses, queried and 
# present dimensions). Parameter values are constrained to be positive and <= 30.
nll_optim <- function(params, stimuli_list, response_list, queried_list, present_list, print_values = F){
  
  # Constrain parameter values to be positive and <= 30
  for(i in 1:length(params)) {
    params[i] <- abs(params[i])
    if(params[i] > 30) {
      params[i] <- 30/1+exp(-params[i])
    }
  }
  
  # Option to print parameter values during optimization
  if(print_values) {
    print(paste("r",params[1]))
    print(paste("beta",params[2]))
    print(paste("d", params[3]))
    print(paste("eta", params[4]))
  }
  
  # Generate the initial sustain model at the start of learning with a 
  # given set of parameter values
  sustain_start <- list( r = params[1], beta = params[2], 
                         d = params[3], eta = params[4],
                         clusters = list(),
                         lambdas = rep(1.0, length(coded_stimuli[[1]])),
                         weights = NULL
  )
  
  # Get the full likelihood for the set of parameter values
  test_full_likelihood <- get_full_likelihood(stimuli_list, response_list, 
                                              queried_list, present_list, 
                                              sustain_start)
  return(test_full_likelihood)
}


# Example 1: Start with an initial set of parameter values, and use optim function
# to estimate best fitting parameter values

# Initial parameter values
inits <- c(1, 1, 2, .5)

# Optimization function - find the parameters that minimize the negative
# log likelihood function
best_model <- optim(inits, nll_optim, stimuli_list=coded_stimuli,
                    response_list=responses_test, 
                    queried_list=queried_dimensions_test,
                    present_list=present_dimensions_test,
                    control=list(maxit=5000))



# Example 2: Start with many randomly selected initial sets of parameter values,
# and use optim function on each to estimate best fitting parameter values
no_initial_values <- 30

initial_parameter_values <- rep(list(NA), no_initial_values)
model_fits <- rep(list(NA), no_initial_values)
nll_values <- rep(NA, no_initial_values)

for (i in 1:no_initial_values){
  print(paste("Iteration ", as.character(i), " / ", as.character(no_initial_values)))
  
  inits <- c(5 * rgamma(1, 1, 1), # r
             5 * rgamma(1, 1, 1), # beta
             5 * rgamma(1, 1, 1), # d
             5 * rgamma(1, 1, 1)) # eta
  
  initial_parameter_values[[i]] <- inits
  
  # Optimization function - find the parameters that minimize the negative
  # log likelihood function
  best_model <- optim(inits, nll_optim, stimuli_list=coded_stimuli,
                      response_list=responses_test, 
                      queried_list=queried_dimensions_test,
                      present_list=present_dimensions_test,
                      control=list(maxit=5000))
  
  model_fits[[i]] <- best_model
  nll_values[i] <- best_model$value
}


# Extract parameters from fits and apply constraints
extract_parameter_values <- function(input_fit) {
  params <- input_fit$par
  
  for(i in 1:length(params)) {
    params[i] <- abs(params[i])
    if(params[i] > 30) {
      params[i] <- 30/1+exp(-params[i])
    }
  }
  
  params <- round(params, 2)
  
  params <- setNames(data.frame(t(params)), c("r", "beta", "d", "eta"))
  params$nll <- input_fit$value
  return(params)
}

parameter_fits <- ldply(model_fits, extract_parameter_values)

# Visualize the distribution of best fitting parameter values
for(i in 1:(ncol(parameter_fits)-1)) {
  hist(parameter_fits[,i], xlab = names(parameter_fits)[i], main = names(parameter_fits)[i])
}

# Get the parameter values corresponding to the best negative log likelihood
parameter_fits[parameter_fits$nll == min(parameter_fits$nll),]